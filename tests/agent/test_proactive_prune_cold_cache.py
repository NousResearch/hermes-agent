"""Cold-cache proactive prune: when the provider prompt cache has already expired, a rewrite costs no cache
break, so the prune commits at a lower reclaim gate without touching the warm regrowth runway, and what it
frees counts toward the next warm gate. Replay of 80 real sessions: see PR #136488 (ledger rows D6-D9)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.context_compressor_prune import _COLD_CACHE_MIN_RECLAIM_TOKENS
from agent.turn_context_compaction import CompactionOutcome, run_turn_start_compaction
from agent.turn_proactive_prune import run_proactive_prune, turn_start_cold_prune


def _shipped(**kw):
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        return ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=4, **kw)


def _warm_on():
    """The warm prune opted in at the replay's settings (it is off by default)."""
    return _shipped(proactive_prune_tokens=64_000, proactive_prune_min_result_chars=1_000,
                    proactive_prune_min_reclaim_tokens=64_000)


def _history(n_pairs, big, chars=20_000, start=0):
    msgs = [{"role": "system", "content": "sys"}] if start == 0 else []
    for i in range(start, start + n_pairs):
        cid = f"call_{i}"
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'}}]})
        msgs.append({"role": "tool", "tool_call_id": cid,
                     "content": chr(65 + i % 26) * chars if i in big else "ok"})
    return msgs


# ~25K tokens reclaimable: under the 64K warm gate, over the 16K cold gate.
def _small_batch():
    return _history(10, big=set(range(5)))


def test_shipped_defaults_prune_only_a_cold_cache():
    """Owner scope: warm pruning stays off by default; an already-expired cache prunes with no setting."""
    c = _shipped()
    assert (c.proactive_prune_tokens, c.proactive_prune_min_result_chars, c.proactive_prune_min_reclaim_tokens) == (
        0, 8_000, 4_096)
    msgs = _small_batch()
    warm, n_warm = c.prune_tool_results_only(msgs, current_tokens=500_000)
    assert n_warm == 0 and warm is msgs
    cold, n_cold = c.prune_tool_results_only(msgs, current_tokens=500_000, cold_cache=True)
    assert n_cold >= 5 and cold is not msgs
    _small, n_small = c.prune_tool_results_only(_history(10, big={0}), current_tokens=500_000, cold_cache=True)
    assert n_small == 0  # ~5K reclaimable: under the 16K cold gate (the 4096 warm default does not lower it)


def test_cold_cache_commits_below_the_warm_gate():
    c = _warm_on()
    msgs = _small_batch()
    warm, n_warm = c.prune_tool_results_only(msgs, current_tokens=100_000)
    assert n_warm == 0 and warm is msgs  # warm cache: 25K is below 64K, no cache break
    cold, n_cold = c.prune_tool_results_only(msgs, current_tokens=100_000, cold_cache=True)
    assert n_cold >= 5 and cold is not msgs


def test_cold_cache_still_needs_the_cold_gate():
    c = _shipped()
    msgs = _history(10, big={0})  # ~5K tokens reclaimable
    out, n = c.prune_tool_results_only(msgs, current_tokens=100_000, cold_cache=True)
    assert n == 0 and out is msgs


def test_cold_prune_leaves_runway_and_credits_the_next_warm_gate():
    """The two fixes that took the worst session from +19% to no session worse: a cold prune neither moves
    the regrowth runway nor resets the reclaim it already did; the next warm gate is 64K minus that credit."""
    c = _warm_on()
    c._proactive_prune_rearm_tokens = 0
    first, n1 = c.prune_tool_results_only(_small_batch(), current_tokens=100_000, cold_cache=True)
    assert n1 and c._proactive_prune_rearm_tokens == 0
    credit = c._proactive_prune_cold_credit
    assert credit >= _COLD_CACHE_MIN_RECLAIM_TOKENS
    # A warm batch smaller than 64K but bigger than 64K - credit now commits.
    grown = first + _history(16, big=set(range(10, 18)), start=10)  # ~40K new reclaimable
    second, n2 = c.prune_tool_results_only(grown, current_tokens=100_000)
    assert n2 and second is not grown
    assert c._proactive_prune_cold_credit == 0 and c._proactive_prune_rearm_tokens > 0


def test_warm_runway_still_blocks_after_a_warm_prune_but_cold_ignores_it():
    c = _warm_on()
    big = _history(20, big=set(range(16)))
    first, n1 = c.prune_tool_results_only(big, current_tokens=100_000)
    assert n1
    grown = first + _history(10, big=set(range(20, 25)), start=20)
    blocked, n2 = c.prune_tool_results_only(grown, current_tokens=100_000)
    assert n2 == 0 and blocked is grown  # warm: needs a full regrowth
    cold, n3 = c.prune_tool_results_only(grown, current_tokens=100_000, cold_cache=True)
    assert n3 and cold is not grown


def test_cold_gate_never_exceeds_a_lower_configured_gate():
    c = _warm_on()
    c.proactive_prune_min_reclaim_tokens = 4_096
    out, n = c.prune_tool_results_only(_history(10, big={0}, chars=20_000), current_tokens=100_000, cold_cache=True)
    assert n and out is not None


def test_reset_clears_the_cold_credit():
    c = _shipped()
    c._proactive_prune_cold_credit = 30_000
    c._reset_proactive_prune_rearm()
    assert c._proactive_prune_cold_credit == 0


# --- the shared cache clock decides "cold" ------------------------------------------------------------------

_ROUTE = ("anthropic", "https://api.anthropic.com", "claude-test")


def _clock_agent(agent, *, estimated_until, guaranteed_until=None, confidence="proven"):
    """Give *agent* a live cache-clock event on its own route (no durable generation: a fresh compressor)."""
    from agent.prompt_cache_clock import CacheEvent, _route_key
    agent.provider, agent.base_url, agent.model = _ROUTE
    agent._session_db = None
    agent.context_compressor.compression_count = 0
    route = _route_key(*_ROUTE)
    agent._prompt_cache_clock = CacheEvent(
        route=route, destination=route, route_class="anthropic_native", requested_ttl="5m", sent_at=0.0,
        read_tokens=100, write_tokens=0, tier="5m", confidence=confidence, reason="test",
        guaranteed_until=guaranteed_until, estimated_until=estimated_until, session_id="s1")
    agent._prompt_cache_compression_mark = 0
    agent._prompt_cache_clock_restored = True
    return agent


@pytest.mark.parametrize("window, expected", [
    ("cold", True), ("maybe_warm", False), ("warm", False), ("unknown", False), ("never_touched", False),
])
def test_only_a_window_the_clock_calls_cold_prunes(window, expected):
    """Invariant: the prune's cold reading is the shared clock's, never a TTL of its own. A stale
    last-response stamp (the old private clock's input) no longer makes a maybe-warm cache cold, and an
    unknown window (no cache touched, or a rewrite since the last response) fails closed."""
    import time
    now = time.time()
    prune, calls = _recording_prune(lambda m, cold: (m, 0))
    agent = _site_agent(prune, cold=True)
    # The old private clock's inputs: caching on, 5m TTL, last response long ago (it called every case cold).
    agent._use_prompt_caching, agent._cache_disabled, agent._cache_ttl = True, False, "5m"
    agent._last_api_response_at = 0.0
    ends = {"cold": (now - 1, None, "proven"), "maybe_warm": (now + 3_600, now - 1, "assumed"),
            "warm": (now + 300, now + 300, "proven"), "unknown": (None, None, "unknown")}
    if window == "never_touched":
        agent._prompt_cache_clock, agent._prompt_cache_clock_restored = None, True
    else:
        estimated, guaranteed, confidence = ends[window]
        _clock_agent(agent, estimated_until=estimated, guaranteed_until=guaranteed, confidence=confidence)
    run_proactive_prune(agent, _history(3, big=set()), 100_000, "t")
    assert calls == [expected]


# --- call sites -------------------------------------------------------------------------------------------

def _site_agent(prune, *, cold, session_db=True, checkpoint=False, session_id="s1"):
    compressor = MagicMock()
    compressor.prune_tool_results_only = prune
    compressor._session_db = MagicMock() if session_db else None
    compressor._session_id = session_id
    agent = MagicMock()
    agent.context_compressor = compressor
    agent.compression_enabled = True
    agent.compression_checkpoint_required = checkpoint
    agent.session_id = "s1"
    return _clock_agent(agent, estimated_until=0.0 if cold else 9e18)


def _recording_prune(result_for):
    calls = []

    def prune(messages, current_tokens=None, *, cold_cache=False):
        calls.append(cold_cache)
        return result_for(messages, cold_cache)
    return prune, calls


def test_post_tool_site_passes_cold_cache():
    prune, calls = _recording_prune(lambda m, cold: (m, 0))
    msgs = _history(3, big=set())
    run_proactive_prune(_site_agent(prune, cold=True), msgs, 100_000, "t")
    run_proactive_prune(_site_agent(prune, cold=False), msgs, 100_000, "t")
    assert calls == [True, False]


def test_legacy_engine_signature_gets_no_cold_kwarg():
    seen = []

    def legacy(messages, current_tokens=None):
        seen.append(current_tokens)
        return messages, 0
    msgs = _history(3, big=set())
    assert run_proactive_prune(_site_agent(legacy, cold=True), msgs, 123, "t") is msgs
    assert seen == [123]


@pytest.mark.parametrize("gate", ["no_store", "checkpoint", "no_session_id"])
def test_post_tool_site_keeps_its_gates(gate):
    prune, calls = _recording_prune(lambda m, cold: (m[:1], 1))
    agent = _site_agent(prune, cold=True, session_db=gate != "no_store", checkpoint=gate == "checkpoint",
                        session_id="" if gate == "no_session_id" else "s1")
    msgs = _history(3, big=set())
    assert run_proactive_prune(agent, msgs, 100_000, "t") is msgs and calls == []


def _outcome(msgs, idx):
    return CompactionOutcome(messages=msgs, active_system_prompt="sys", conversation_history=None,
                             current_turn_user_idx=idx)


def _with_turn(prefix):
    return prefix + [{"role": "user", "content": "next question"}], len(prefix)


def test_turn_start_prunes_only_the_prefix_on_a_cold_cache():
    prefix = _history(3, big={0, 1})
    msgs, idx = _with_turn(prefix)

    def prune(messages, current_tokens=None, *, cold_cache=False):
        assert cold_cache is True and messages == prefix  # never sees the unpersisted user row
        return [dict(m, content="x") if m.get("role") == "tool" else m for m in messages], 2
    out = _outcome(msgs, idx)
    with patch("agent.turn_context._preflight_request_tokens", return_value=100_000), \
         patch("agent.compaction_events.publish_prune"):
        turn_start_cold_prune(_site_agent(prune, cold=True), out, "t")
    assert out.messages is not msgs and len(out.messages) == len(msgs)
    assert out.messages[idx] is msgs[idx] and out.messages[idx]["role"] == "user"
    assert all(m["content"] == "x" for m in out.messages if m.get("role") == "tool")


@pytest.mark.parametrize("case", ["warm", "compressed", "blocked", "disabled", "bad_idx", "row_saved", "row_marked"])
def test_turn_start_stands_down(case):
    """``row_saved``/``row_marked``: TUI/Desktop wrote this turn's user row at submit; archiving only the prefix
    would clone it under a new id while the live dict keeps the old ``_row_id`` (later api_content stamps miss)."""
    from agent.context_compressor import _DB_PERSISTED_MARKER
    prune, calls = _recording_prune(lambda m, cold: (m[:1], 1))
    agent = _site_agent(prune, cold=case != "warm")
    agent.compression_enabled = case != "disabled"
    msgs, idx = _with_turn(_history(3, big={0}))
    if case == "row_saved":
        msgs[idx]["_row_id"] = 42
    if case == "row_marked":
        msgs[idx][_DB_PERSISTED_MARKER] = True
    out = _outcome(msgs, -1 if case == "bad_idx" else idx)
    out.compressed, out.blocked = case == "compressed", case == "blocked"
    with patch("agent.turn_context._preflight_request_tokens", return_value=100_000):
        turn_start_cold_prune(agent, out, "t")
    assert out.messages is msgs and calls == []


def test_turn_start_compaction_runs_the_cold_prune():
    """Wiring: the turn-start entry point (build_turn_context's only compaction call) reaches the prune."""
    agent = MagicMock()
    agent.compression_enabled = False  # skips idle + preflight; the cold prune still owns its own gate
    msgs, idx = _with_turn(_history(3, big={0}))
    with patch("agent.turn_proactive_prune.turn_start_cold_prune") as cold:
        out = run_turn_start_compaction(
            agent, messages=msgs, system_message=None, active_system_prompt="sys", conversation_history=None,
            current_turn_user_idx=idx, user_message="next question", effective_task_id="t")
    cold.assert_called_once_with(agent, out, "t")


def _bound(db, sid="s1"):
    c = _shipped()
    c.bind_session_state(session_db=db, session_id=sid)
    return c


def test_engine_refuses_a_store_without_a_session_row():
    """Engine invariant (not only the call site): a store with no bound row cannot archive the originals."""
    c = _bound(MagicMock(), sid="")
    msgs = _small_batch()
    out, n = c.prune_tool_results_only(msgs, current_tokens=100_000, cold_cache=True)
    assert n == 0 and out is msgs


def test_unflushed_tool_results_keep_their_original_on_disk(tmp_path, monkeypatch):
    """A result the store has not seen yet can still be shortened only because its original went to disk
    first: the stub names a file holding the exact body (kvnloo's #136488 finding: before, the next flush
    stored only the stub and the original was gone)."""
    from agent.context_compressor import _DB_PERSISTED_MARKER
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli", model="test")
    db.append_message("s1", role="user", content="Previous turn")
    saved = db.get_messages_as_conversation("s1", include_row_ids=True)
    for message in saved:
        message[_DB_PERSISTED_MARKER] = True
    pending = _history(20, big=set(range(18)), chars=24_000, start=1)
    messages = [{"role": "system", "content": "sys"}, *saved, {"role": "user", "content": "run checks"}, *pending]
    result, count = _bound(db).prune_tool_results_only(messages, current_tokens=300_000, cold_cache=True)

    assert count
    for old, new in zip(messages, result):
        if old["content"] != new["content"]:
            path = new["content"].rsplit("full result: ", 1)[1].replace("~", str(Path.home()), 1)
            assert Path(path).read_text(encoding="utf-8") == old["content"]


def test_cold_credit_survives_a_rebuilt_compressor(tmp_path):
    """api_server builds an agent per request and resumes rebuild the compressor: the credit is persisted with
    the runway in the same commit, and restored on bind."""
    from hermes_state import SessionDB
    from agent.context_compressor import _DB_PERSISTED_MARKER
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", source="cli", model="m")
    for m in _small_batch()[1:]:
        db.append_message("s1", role=m["role"], content=m["content"], tool_calls=m.get("tool_calls"),
                          tool_call_id=m.get("tool_call_id"))
    live = [{"role": "system", "content": "sys"}] + db.get_messages_as_conversation("s1", include_row_ids=True)
    for m in live[1:]:
        m[_DB_PERSISTED_MARKER] = True
    first = _bound(db)
    _out, n = first.prune_tool_results_only(live, current_tokens=100_000, cold_cache=True)
    assert n and first._proactive_prune_cold_credit >= _COLD_CACHE_MIN_RECLAIM_TOKENS
    rebuilt = _bound(db)
    assert rebuilt._proactive_prune_cold_credit == first._proactive_prune_cold_credit
    assert rebuilt._proactive_prune_rearm_tokens == first._proactive_prune_rearm_tokens
    rebuilt._clear_durable_proactive_prune_rearm()
    assert _bound(db)._proactive_prune_cold_credit == 0


def test_attempt_restore_keeps_the_cold_credit():
    from agent.conversation_compression import _restore_prune_rearm_tokens
    c = _shipped()
    _restore_prune_rearm_tokens(c, {"_proactive_prune_rearm_tokens": 90_000, "_proactive_prune_cold_credit": 20_000})
    assert (c._proactive_prune_rearm_tokens, c._proactive_prune_cold_credit) == (90_000, 20_000)


def test_legacy_session_resumed_while_warm_is_not_pruned(tmp_path, monkeypatch):
    """Invariant: a pre-clock session's window is rebuilt from TODAY's TTL, not the one it was cached at,
    so it never licenses a cold prune. Cached for 1h, resumed 10 min later under a 5m setting: still warm."""
    import time
    from hermes_state import SessionDB
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "hi")
        db.append_message("s1", "assistant", "hello", timestamp=time.time() - 600)
        db.update_token_counts("s1", cache_write_tokens=4000, api_call_count=1)  # no stored clock
        prune, calls = _recording_prune(lambda m, cold: (m, 0))
        agent = _site_agent(prune, cold=True)
        agent.api_mode, agent._use_prompt_caching, agent._cache_ttl = "anthropic_messages", True, "5m"
        agent._session_db, agent._prompt_cache_clock, agent._prompt_cache_clock_restored = db, None, False
        run_proactive_prune(agent, _history(3, big=set()), 100_000, "t")
        assert calls == [False]
    finally:
        db.close()
