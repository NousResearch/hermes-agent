"""Cold-cache proactive prune: when the provider prompt cache has already expired, a rewrite costs no cache
break, so the prune commits at a lower reclaim gate without touching the warm regrowth runway, and what it
frees counts toward the next warm gate. Replay of 80 real sessions: see PR #136488 (ledger rows D6-D9)."""

from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import _COLD_CACHE_MIN_RECLAIM_TOKENS, ContextCompressor
from agent.prompt_caching import provider_cache_expired
from agent.turn_context_compaction import CompactionOutcome, run_turn_start_compaction
from agent.turn_proactive_prune import run_proactive_prune, turn_start_cold_prune


def _shipped():
    with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
        return ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=4)


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


def test_cold_cache_commits_below_the_warm_gate():
    c = _shipped()
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
    c = _shipped()
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
    c = _shipped()
    big = _history(20, big=set(range(16)))
    first, n1 = c.prune_tool_results_only(big, current_tokens=100_000)
    assert n1
    grown = first + _history(10, big=set(range(20, 25)), start=20)
    blocked, n2 = c.prune_tool_results_only(grown, current_tokens=100_000)
    assert n2 == 0 and blocked is grown  # warm: needs a full regrowth
    cold, n3 = c.prune_tool_results_only(grown, current_tokens=100_000, cold_cache=True)
    assert n3 and cold is not grown


def test_cold_gate_never_exceeds_a_lower_configured_gate():
    c = _shipped()
    c.proactive_prune_min_reclaim_tokens = 4_096
    out, n = c.prune_tool_results_only(_history(10, big={0}, chars=20_000), current_tokens=100_000, cold_cache=True)
    assert n and out is not None


def test_reset_clears_the_cold_credit():
    c = _shipped()
    c._proactive_prune_cold_credit = 30_000
    c._reset_proactive_prune_rearm()
    assert c._proactive_prune_cold_credit == 0


# --- provider_cache_expired -------------------------------------------------------------------------------

def _agent(**kw):
    base = dict(_use_prompt_caching=True, _cache_disabled=False, _cache_ttl="5m", model="test-model", provider="test-provider",
                _last_api_response_at=1_000.0)
    base.update(kw)
    return MagicMock(spec=list(base), **base)


@pytest.mark.parametrize("ttl, gap, expected", [
    ("5m", 300, False), ("5m", 320, False), ("5m", 331, True), ("1h", 331, False), ("1h", 3620, False),
    ("1h", 3631, True), (None, 331, True),
])
def test_expiry_follows_the_effective_ttl(ttl, gap, expected):
    assert provider_cache_expired(_agent(_cache_ttl=ttl), now=1_000.0 + gap) is expected


def test_caching_off_or_no_response_is_never_cold():
    assert provider_cache_expired(_agent(_use_prompt_caching=False), now=99_999.0) is False
    assert provider_cache_expired(_agent(_cache_disabled=True), now=99_999.0) is False
    assert provider_cache_expired(_agent(_last_api_response_at=None), messages=[], now=99_999.0) is False


def test_resumed_session_reads_the_last_assistant_timestamp():
    agent = _agent(_last_api_response_at=None)
    msgs = [{"role": "user", "content": "q", "timestamp": 10.0}, {"role": "assistant", "content": "a", "timestamp": 500.0},
            {"role": "user", "content": "q2", "timestamp": 9_000.0}]
    assert provider_cache_expired(agent, msgs, now=500.0 + 400) is True
    assert provider_cache_expired(agent, msgs, now=500.0 + 100) is False


def test_qwen_route_clamps_1h_to_5m():
    assert provider_cache_expired(_agent(_cache_ttl="1h", model="qwen3-max", provider="alibaba"), now=1_000.0 + 400) is True


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
    agent._use_prompt_caching, agent._cache_disabled, agent._cache_ttl = True, False, "5m"
    agent.model, agent.provider = "test-model", "test-provider"
    agent._last_api_response_at = 0.0 if cold else 9e18
    return agent


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
