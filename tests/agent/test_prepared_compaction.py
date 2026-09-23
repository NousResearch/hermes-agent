"""Opt-in prepared compaction (``compression.prepare_ahead``, agent/prepared_compaction.py).

Behaviour contracts, with the summariser stubbed at its model boundary:

* flag off (the default) leaves the whole compression path as it was: no state, no pass, inline summary;
* a pass never changes the live transcript or compressor, and runs under the caller's profile scope;
* append-only growth keeps a candidate, which splices at its own boundary with newer messages kept as tail;
* a rewritten prefix, a moved head or another session's transcript never splices;
* a completed candidate splices while an extension pass runs; a pass that has not finished is never
  waited on, and a pass overtaken by compaction or a session boundary cannot publish;
* manual, focus, overflow-recovery and memory-context compactions summarise fresh;
* the no-LLM prune and blank-echo drop between pass and splice do not invalidate the candidate.
"""
from __future__ import annotations

import copy
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import prepared_compaction as pc
from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor
from agent.prepared_compaction import PreparedCompaction


class _StubCompressor(ContextCompressor):
    """Records every summariser call; ``gate`` holds background passes only (shared with the worker copy)."""

    def _generate_summary(self, turns_to_summarize, focus_topic=None, memory_context="", bypass_cooldown=False):
        from hermes_constants import get_hermes_home

        worker = threading.current_thread().name == "compaction-prepare"
        if worker and self.gate is not None:
            assert self.gate.wait(5.0), "pass never released"
        self.calls.append({
            "worker": worker,
            "n_turns": len(turns_to_summarize),
            "first": turns_to_summarize[0].get("content", "")[:10],
            "previous": self._previous_summary,
            "focus": focus_topic,
            "home": str(get_hermes_home()),
        })
        body = f"Summary v{len(self.calls)} of {len(turns_to_summarize)} turns."
        self._previous_summary = body
        return self._with_summary_prefix(body)


def _compressor(*, enabled: bool = True, session_id: str = "S1") -> _StubCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=40960):
        cc = _StubCompressor(model="test/model", protect_first_n=3, protect_last_n=5, quiet_mode=True)
        cc.context_length = 40960
    cc.threshold_tokens = 20000
    cc.tail_token_budget = 2500
    cc.calls, cc.gate = [], None
    cc._session_id = session_id
    if enabled:
        cc.prepared_compaction = PreparedCompaction()
    return cc


def _messages(n: int, size: int = 1200, tag: str = "m") -> list:
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"{tag}{i} " + "z" * size})
    return msgs


def _grow(msgs: list, n: int, size: int = 1200, tag: str = "new") -> list:
    return msgs + _messages(n, size=size, tag=tag)[1:]


def _agent(cc, tokens: int = 19500):
    cc.last_prompt_tokens = tokens
    return SimpleNamespace(context_compressor=cc, compression_enabled=True, api_mode="chat_completions")


def _prepare(cc, msgs, tokens: int = 19500):
    thread = pc.maybe_prepare(_agent(cc, tokens), msgs, origin="test")
    assert thread is not None, "expected a pass to start"
    thread.join(5.0)
    entry = cc.prepared_compaction._entry
    assert entry is not None
    return entry


def _summary_row(compressed: list) -> dict:
    rows = [m for m in compressed if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert len(rows) == 1
    return rows[0]


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# --------------------------------------------------------------------------- default off


def test_flag_off_leaves_compression_inline_and_stateless():
    cc = _compressor(enabled=False)
    msgs = _messages(30)
    assert pc.maybe_prepare(_agent(cc), msgs, origin="test") is None
    compressed = cc.compress(msgs, current_tokens=cc.threshold_tokens + 1)
    assert [c["worker"] for c in cc.calls] == [False]
    assert cc.prepared_compaction is None
    assert "Summary v1" in _summary_row(compressed)["content"]


def test_config_flag_reaches_the_compressor_through_the_loader(tmp_path, monkeypatch):
    from agent.agent_init import _parse_compression_config
    from hermes_cli.config import DEFAULT_CONFIG, load_config_readonly

    agent = SimpleNamespace(model="m", provider="openrouter", api_mode="chat_completions", quiet_mode=True)
    assert DEFAULT_CONFIG["compression"]["prepare_ahead"] is False
    assert _parse_compression_config(agent, load_config_readonly()).prepare_ahead is False

    home = tmp_path / "profile"
    home.mkdir()
    (home / "config.yaml").write_text("compression:\n  prepare_ahead: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _parse_compression_config(agent, load_config_readonly()).prepare_ahead is True


# --------------------------------------------------------------------------- the pass


def test_pass_leaves_live_transcript_and_compressor_untouched():
    cc = _compressor()
    msgs = _messages(30)
    before = copy.deepcopy(msgs)
    entry = _prepare(cc, msgs)
    assert msgs == before
    assert cc._previous_summary is None and cc.compression_count == 0
    assert [c["worker"] for c in cc.calls] == [True]
    working, _, start, end = cc._plan_compaction_window(msgs)
    assert (entry.compress_start, entry.compress_end) == (start, end)
    assert entry.body == f"Summary v1 of {end - start} turns."


@pytest.mark.parametrize("setup", [
    lambda a, cc: setattr(cc, "last_prompt_tokens", 20000 - int(40960 * pc.PREPARE_BAND_RATIO) - 1),
    lambda a, cc: setattr(cc, "awaiting_real_usage_after_compression", True),
    lambda a, cc: setattr(a, "compression_enabled", False),
    lambda a, cc: setattr(a, "_persist_disabled", True),  # background-review fork
    lambda a, cc: setattr(a, "api_mode", "codex_app_server"),
    lambda a, cc: setattr(a, "codex_responses_native_compaction", True),
    lambda a, cc: setattr(cc, "_micro_compact_enabled", True),
    lambda a, cc: setattr(cc, "_summary_failure_cooldown_until", time.monotonic() + 60),
], ids=["below-band", "awaiting-usage", "disabled", "fork", "app-server", "native", "micro", "cooldown"])
def test_gates_start_no_pass(setup):
    cc = _compressor()
    agent = _agent(cc)
    setup(agent, cc)
    assert pc.maybe_prepare(agent, _messages(30), origin="test") is None
    assert cc.calls == []


def test_pass_runs_under_the_callers_profile_scope(tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    homes = [tmp_path / "a", tmp_path / "b", tmp_path / "a"]
    compressors = []
    for home in homes:
        cc = _compressor()
        token = set_hermes_home_override(home)
        try:
            _prepare(cc, _messages(30))
        finally:
            reset_hermes_home_override(token)
        compressors.append(cc)
    assert [cc.calls[0]["home"] for cc in compressors] == [str(h) for h in homes]


# --------------------------------------------------------------------------- the splice


def test_append_only_growth_splices_at_the_candidate_boundary_keeping_newer_messages():
    cc = _compressor()
    msgs = _messages(30)
    entry = _prepare(cc, msgs)
    later = _grow(msgs, 8)
    assert cc._plan_compaction_window(later)[3] > entry.compress_end  # a fresh plan would cut later

    compressed = cc.compress(later, current_tokens=cc.threshold_tokens + 1)

    assert [c["worker"] for c in cc.calls] == [True], "no summariser call on the turn"
    assert entry.summary in _summary_row(compressed)["content"]
    kept = later[entry.compress_end:]
    for original, row in zip(kept, compressed[-len(kept):]):
        assert original["content"] in row["content"]
    assert cc._previous_summary == entry.body
    assert cc.prepared_compaction._entry is None


@pytest.mark.parametrize("change", ["middle rewritten", "head moved", "other session"])
def test_changed_prefix_never_splices(change):
    cc = _compressor()
    msgs = _messages(30)
    entry = _prepare(cc, msgs)
    live = copy.deepcopy(msgs)
    if change == "middle rewritten":
        live[entry.compress_start + 1]["content"] = "rewritten"
    elif change == "head moved":
        cc.compression_count = 1  # head protection decays after a compaction
    else:
        cc._session_id = "S2"
    compressed = cc.compress(live, current_tokens=cc.threshold_tokens + 1)
    assert [c["worker"] for c in cc.calls] == [True, False], "inline summary expected"
    assert "Summary v2" in _summary_row(compressed)["content"]


@pytest.mark.parametrize("kwargs", [
    {"force": True}, {"focus_topic": "auth bug"}, {"bypass_cooldown": True}, {"memory_context": "provider note"},
], ids=["manual", "focus", "overflow-recovery", "memory-context"])
def test_fresh_summary_paths_ignore_and_drop_the_candidate(kwargs):
    cc = _compressor()
    msgs = _messages(30)
    _prepare(cc, msgs)
    cc.compress(msgs, current_tokens=cc.threshold_tokens + 1, **kwargs)
    assert [c["worker"] for c in cc.calls] == [True, False]
    assert cc.prepared_compaction._entry is None


def test_prune_and_blank_echo_drop_do_not_invalidate_the_candidate():
    cc = _compressor()
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(12):  # tool rounds whose bodies and arguments the prune rewrites
        msgs += [
            {"role": "user", "content": f"u{i} " + "q" * 400},
            {"role": "assistant", "content": f"a{i}", "tool_calls": [{
                "id": f"c{i}", "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "%s"}' % ("p" * 900)},
            }]},
            {"role": "tool", "tool_call_id": f"c{i}", "name": "read_file", "content": f"r{i} " + "r" * 3000},
            {"role": "assistant", "content": f"done {i} " + "d" * 400},
        ]
    msgs += [{"role": "user", "content": "latest ask"}, {"role": "user", "content": " "},
             {"role": "assistant", "content": "working on it"}]
    working, pruned, _, _ = cc._plan_compaction_window(msgs)
    assert pruned > 0 and len(working) == len(msgs) - 1, "prune ran and the trailing blank echo was dropped"
    entry = _prepare(cc, msgs)

    live = _grow(msgs, 4)  # a newer user turn: the old echo is no longer trailing, so it stays
    live_working, live_pruned, start, end = cc._plan_compaction_window(live)
    assert len(live_working) == len(live) and live_pruned >= pruned
    assert start == entry.compress_start and end >= entry.compress_end

    cc.compress(live, current_tokens=cc.threshold_tokens + 1)
    assert [c["worker"] for c in cc.calls] == [True], "the candidate must still splice"


def test_fingerprint_survives_a_session_db_round_trip(tmp_path):
    from hermes_state import SessionDB

    msgs = _messages(20)
    msgs[3]["content"] += "  \n"  # trailing whitespace the store strips
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("S1", source="cli")
    for m in msgs[1:]:
        db.append_message("S1", role=m["role"], content=m["content"])
    replayed = [msgs[0]] + db.get_messages_as_conversation("S1")
    assert pc.fingerprint(replayed, "S1", 4, 15) == pc.fingerprint(msgs, "S1", 4, 15)
    replayed[5]["content"] = "edited"
    assert pc.fingerprint(replayed, "S1", 4, 15) != pc.fingerprint(msgs, "S1", 4, 15)


# --------------------------------------------------------------------------- running passes


def test_completed_candidate_splices_while_an_extension_pass_runs():
    cc = _compressor()
    msgs = _messages(30)
    first = _prepare(cc, msgs)
    grown = _grow(msgs, 14, size=2600)  # clears MIN_EXTENSION_DELTA_TOKENS
    cc.gate = threading.Event()
    running = pc.maybe_prepare(_agent(cc), grown, origin="test")
    assert running is not None

    started = time.monotonic()
    compressed = cc.compress(grown, current_tokens=cc.threshold_tokens + 1)
    assert time.monotonic() - started < 2.0, "must not wait for the running pass"
    assert first.summary in _summary_row(compressed)["content"]

    cc.gate.set()
    running.join(5.0)
    assert len(cc.calls) == 2 and cc.calls[1]["previous"] == first.body, "the pass was an extension"
    assert cc.prepared_compaction._entry is None, "an overtaken pass cannot publish"


def test_unfinished_pass_is_not_waited_on_and_cannot_publish_later():
    cc = _compressor()
    cc.gate = threading.Event()
    msgs = _messages(30)
    running = pc.maybe_prepare(_agent(cc), msgs, origin="test")
    assert running is not None

    started = time.monotonic()
    cc.compress(msgs, current_tokens=cc.threshold_tokens + 1)
    assert time.monotonic() - started < 2.0
    assert _wait_until(lambda: len(cc.calls) == 1) and cc.calls[0]["worker"] is False, "inline summary"

    cc.gate.set()
    running.join(5.0)
    assert cc.prepared_compaction._entry is None


@pytest.mark.parametrize("boundary", ["reset", "end"])
def test_session_boundary_discards_candidate_and_running_pass(boundary):
    cc = _compressor()
    msgs = _messages(30)
    _prepare(cc, msgs)
    cc.gate = threading.Event()
    running = pc.maybe_prepare(_agent(cc), _grow(msgs, 14, size=2600), origin="test")
    assert running is not None
    if boundary == "reset":
        cc.on_session_reset()
    else:
        cc.on_session_end("S1", msgs)
    cc.gate.set()
    running.join(5.0)
    assert cc.prepared_compaction._entry is None and cc.prepared_compaction._pending is None


# --------------------------------------------------------------------------- extension, failures


def test_small_delta_keeps_the_candidate_without_copying_or_taking_a_slot(monkeypatch):
    cc = _compressor()
    msgs = _messages(30)
    first = _prepare(cc, msgs)
    copies = []
    monkeypatch.setattr(pc.copy, "deepcopy", lambda value, *a: copies.append(value) or value)
    held = [pc._slots.acquire(blocking=False)]  # leave exactly one slot free
    try:
        assert pc.maybe_prepare(_agent(cc), _grow(msgs, 4), origin="test") is None
        assert pc._slots.acquire(blocking=False), "the skip held a slot"
        pc._slots.release()
    finally:
        for ok in held:
            if ok:
                pc._slots.release()
    assert copies == [] and cc.prepared_compaction._entry is first and len(cc.calls) == 1


def test_extension_pass_summarises_only_the_delta_on_top_of_the_previous_body():
    cc = _compressor()
    msgs = _messages(30)
    first = _prepare(cc, msgs)
    grown = _grow(msgs, 14, size=2600)
    second = _prepare(cc, grown)
    call = cc.calls[1]
    assert call["previous"] == first.body
    assert call["n_turns"] == second.compress_end - first.compress_end
    assert call["first"] == grown[first.compress_end]["content"][:10]
    cc.compress(grown, current_tokens=cc.threshold_tokens + 1)
    assert len(cc.calls) == 2 and cc._previous_summary == second.body


def test_failed_pass_backs_off_without_touching_live_failure_state(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    import agent.context_compressor as module

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("S1", source="cli")
    cc = _compressor()
    cc.bind_session_state(db, "S1")
    msgs = _messages(30)
    first = _prepare(cc, msgs)

    monkeypatch.setattr(_StubCompressor, "_generate_summary", module.ContextCompressor._generate_summary)
    call = MagicMock(side_effect=RuntimeError("No LLM provider configured"))
    monkeypatch.setattr(module, "call_llm", call)
    grown = _grow(msgs, 14, size=2600)
    thread = pc.maybe_prepare(_agent(cc), grown, origin="test")
    thread.join(5.0)

    assert call.call_count == 1
    assert cc.prepared_compaction._entry is first, "a failed extension keeps the previous candidate"
    assert cc._summary_failure_cooldown_until == 0.0 and cc._last_summary_error is None
    assert db.get_compression_failure_cooldown("S1") is None
    assert pc.maybe_prepare(_agent(cc), grown, origin="test") is None, "backs off"
    assert call.call_count == 1


# --------------------------------------------------------------------------- the real turn loop


def _loop_response(*, prompt_tokens: int, tool_call_id: str | None = None, content=None):
    tool_calls = [SimpleNamespace(
        id=tool_call_id, type="function", function=SimpleNamespace(name="web_search", arguments='{"query": "q"}'),
    )] if tool_call_id else None
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning_content=None, reasoning=None)
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if tool_calls else "stop")], model="test/model",
    )
    resp.usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=50, total_tokens=prompt_tokens + 50)
    return resp


@pytest.mark.parametrize("prepare_ahead", [True, False])
def test_turn_that_crosses_the_trigger_splices_the_summary_prepared_during_its_tools(prepare_ahead):
    """Real run_conversation with the provider and tool boundaries mocked; config via the real loader.

    Response 1 lands in the band: with the flag, the pass runs while that batch's tool executes.
    Response 2 crosses the trigger, so the loop compacts after the second batch: a splice with the
    flag (and no second pass), an inline summary without it.
    """
    import os
    from pathlib import Path

    from run_agent import AIAgent

    if prepare_ahead:
        Path(os.environ["HERMES_HOME"], "config.yaml").write_text(
            "compression:\n  prepare_ahead: true\n", encoding="utf-8",
        )
    tool_defs = [{"type": "function", "function": {
        "name": "web_search", "description": "search", "parameters": {"type": "object", "properties": {}},
    }}]
    with (
        patch("model_tools.get_tool_definitions", return_value=tool_defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
                        quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.save_trajectories = False
    cc = agent.context_compressor
    assert (cc.prepared_compaction is not None) is prepare_ahead
    cc.context_length, cc.threshold_tokens, cc.tail_token_budget = 40960, 20000, 2500
    band_floor = cc.threshold_tokens - int(cc.context_length * pc.PREPARE_BAND_RATIO)

    summaries, tool_started = [], threading.Event()

    def summarise(turns, focus_topic=None, **_):
        on_worker = threading.current_thread().name == "compaction-prepare"
        if on_worker:  # the pass is meant to overlap the tool
            assert tool_started.wait(5.0), "no tool ran while the pass was pending"
        summaries.append(on_worker)
        return cc._with_summary_prefix(f"Summary of {len(turns)} turns.")

    cc._generate_summary = summarise
    pass_during_tool = []

    def run_tool(*_a, **_k):
        state = cc.prepared_compaction
        pass_during_tool.append(state is not None and state._pending is not None)
        tool_started.set()
        if state is not None:
            assert _wait_until(lambda: state._pending is None)
        return '{"ok": true}'

    agent.client.chat.completions.create.side_effect = [
        _loop_response(prompt_tokens=band_floor + 1000, tool_call_id="c1"),
        _loop_response(prompt_tokens=cc.threshold_tokens + 1000, tool_call_id="c2"),
        _loop_response(prompt_tokens=6000, content="done"),
    ]
    with (
        patch("model_tools.handle_function_call", side_effect=run_tool),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("continue", conversation_history=_messages(30)[1:])

    assert result["completed"] is True and result["final_response"] == "done"
    assert len([m for m in result["messages"] if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]) == 1
    assert pass_during_tool == [prepare_ahead, False]
    assert summaries == [prepare_ahead], "one summariser call: on the worker with the flag, inline without"
