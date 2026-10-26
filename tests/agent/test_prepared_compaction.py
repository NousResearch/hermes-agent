"""Opt-in prepared compaction (``compression.prepare_ahead``, agent/prepared_compaction.py).

Behaviour contracts, with the summariser stubbed at its model boundary:

* flag off (the default) leaves the whole compression path as it was: no state, no pass, inline summary;
* a pass never changes the live transcript or compressor, and runs under the caller's profile scope;
* one pass per compaction: the candidate splices at its own boundary and every message after it, however
  many, stays verbatim in the tail;
* in-place text rewrites (plugins, the no-LLM prune, blank-echo drop) keep the candidate; a changed message
  structure, a rewound transcript, a moved head or another session's transcript never splices;
* without a completed candidate, compaction waits for a running pass whose window fits (forwarding its
  progress to the host) until the host's deadline, and a pass overtaken by compaction or a session boundary
  cannot publish (a same-session rebind keeps the candidate);
* manual, focus, overflow-recovery and memory-context compactions summarise fresh, and memory-provider
  context stops further passes;
* a splice leaves the live compressor as the inline call would: main-model fallback state and telemetry.
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
from agent.model_metadata import estimate_messages_tokens_rough
from agent.prepared_compaction import PreparedCompaction
from tests.waiting import wait_until


class _StubCompressor(ContextCompressor):
    """Records every summariser call; ``gate`` holds background passes only (shared with the worker copy)."""

    def _generate_summary(self, turns_to_summarize, focus_topic=None, memory_context="", bypass_cooldown=False):
        from hermes_constants import get_hermes_home

        worker = threading.current_thread().name == "compaction-prepare"
        if worker and self.read_seconds:
            from agent.local_prompt_progress import current_prompt_reading_hook

            pending = self.prepared_compaction._pending
            wait_until(lambda: pending.on_reading is not None, what="compaction waiting on the pass")
            current_prompt_reading_hook()(self.read_seconds)  # a local server reading the prompt
            self.after_read()
        if worker and self.gate is not None:
            from agent.auxiliary_client import AuxiliaryExplicitCancellation, _aux_interrupt_cancel_requested

            wait_until(lambda: self.gate.is_set() or _aux_interrupt_cancel_requested(), what="pass released or cancelled")
            if not self.gate.is_set():
                raise AuxiliaryExplicitCancellation()  # how the aux client ends a cancelled summary stream
        if worker and self.stream_progress:
            from agent.auxiliary_client import _aux_progress

            _aux_progress.hook()  # one streamed chunk
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
    cc.calls, cc.gate, cc.stream_progress, cc.read_seconds, cc.after_read = [], None, False, 0.0, None
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
    thread = pc.maybe_prepare(_agent(cc, tokens), msgs, origin="tool-batch")
    assert thread is not None, "expected a pass to start"
    thread.join(5.0)
    entry = cc.prepared_compaction._entry
    assert entry is not None
    return entry


def _summary_row(compressed: list) -> dict:
    rows = [m for m in compressed if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
    assert len(rows) == 1
    return rows[0]


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
    _working, _, start, end = cc._plan_compaction_window(msgs)
    assert (entry.compress_start, entry.compress_end) == (start, end)
    assert entry.body == f"Summary v1 of {end - start} turns."


@pytest.mark.parametrize("setup", [
    lambda a, cc: setattr(cc, "last_prompt_tokens", int(20000 * (1 - pc.PREPARE_BAND_RATIO)) - 1),
    # A small absolute trigger on a large window: the band is sized from the trigger, not the window.
    lambda a, cc: (setattr(cc, "context_length", 200_000), setattr(cc, "threshold_tokens", 20000),
                   setattr(cc, "last_prompt_tokens", 1000)),
    lambda a, cc: setattr(cc, "awaiting_real_usage_after_compression", True),
    lambda a, cc: setattr(a, "compression_enabled", False),
    lambda a, cc: setattr(a, "_persist_disabled", True),  # background-review fork
    lambda a, cc: setattr(a, "api_mode", "codex_app_server"),
    lambda a, cc: setattr(a, "codex_responses_native_compaction", True),
    lambda a, cc: setattr(cc, "_micro_compact_enabled", True),
    lambda a, cc: setattr(cc, "_summary_failure_cooldown_until", time.monotonic() + 60),
    lambda a, cc: setattr(cc, "_fallback_compression_streak", 2),  # summary model benched
], ids=["below-band", "small-trigger-large-window", "awaiting-usage", "disabled", "fork", "app-server", "native",
        "micro", "cooldown", "benched"])
def test_gates_start_no_pass(setup):
    cc = _compressor()
    agent = _agent(cc)
    setup(agent, cc)
    assert cc.threshold_tokens == 20000
    assert pc.maybe_prepare(agent, _messages(30), origin="test") is None
    assert cc.calls == []


@pytest.mark.parametrize("agent_attrs", [{"platform": "cron"}, {"platform": "subagent"}, {"_delegate_depth": 1}],
                         ids=["cron", "subagent-platform", "delegated-child"])
def test_turn_end_pass_skipped_where_the_session_ends_with_the_turn(agent_attrs):
    cc = _compressor()
    agent = _agent(cc)
    for name, value in agent_attrs.items():
        setattr(agent, name, value)
    assert pc.maybe_prepare(agent, _messages(30), origin="turn-end") is None
    thread = pc.maybe_prepare(agent, _messages(30), origin="tool-batch")  # mid-turn passes still run
    assert thread is not None
    thread.join(5.0)


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


@pytest.mark.parametrize("pass_state", ["completed", "running"])
def test_growth_past_the_candidate_beyond_the_tail_budget_still_splices_keeping_it_all_in_the_tail(pass_state):
    """However much arrived after the pass started, it joins the tail: no second summary call."""
    cc = _compressor()
    msgs = _messages(30)
    if pass_state == "running":
        cc.gate = threading.Event()
        running = pc.maybe_prepare(_agent(cc), msgs, origin="test")
        assert running is not None
        pending = cc.prepared_compaction._pending
        threading.Thread(target=lambda: wait_until(lambda: pending.on_progress is not None) and cc.gate.set()).start()
    else:
        _prepare(cc, msgs)
    window = (cc.prepared_compaction._entry or cc.prepared_compaction._pending.window)
    later = _grow(msgs, 24)
    _, _, start, end = cc._plan_compaction_window(later)
    assert start == window.compress_start and end > window.compress_end
    assert estimate_messages_tokens_rough(later[window.compress_end:end]) > cc.tail_token_budget

    from agent.auxiliary_client import aux_progress_hook

    with aux_progress_hook(lambda: None):  # a host hook marks the moment the wait starts
        compressed = cc.compress(later, current_tokens=cc.threshold_tokens + 1)

    if pass_state == "running":
        running.join(5.0)
    assert [c["worker"] for c in cc.calls] == [True], "one summary, written by the pass"
    kept = later[window.compress_end:]
    for original, row in zip(kept, compressed[-len(kept):]):
        assert original["content"] in row["content"]
    assert cc.prepared_compaction._entry is None


def test_in_place_text_rewrite_of_summarised_messages_keeps_the_candidate():
    """Plugins rewrite message text in place (redaction, pre-compaction notes): still the same messages."""
    cc = _compressor()
    msgs = _messages(30)
    for i, msg in enumerate(msgs):
        msg["message_uid"] = f"uid-{i}"
    entry = _prepare(cc, msgs)
    live = copy.deepcopy(msgs)
    live[entry.compress_start + 1]["content"] = "rewritten by a plugin"
    live[entry.compress_start + 2]["content"] = [{"type": "input_text", "text": "structured rewrite"}]
    compressed = cc.compress(live, current_tokens=cc.threshold_tokens + 1)
    assert [c["worker"] for c in cc.calls] == [True]
    assert entry.summary in _summary_row(compressed)["content"]


@pytest.mark.parametrize("change", ["message dropped", "rewound", "head moved", "other session"])
def test_changed_prefix_never_splices(change):
    cc = _compressor()
    msgs = _messages(30)
    for i, msg in enumerate(msgs):
        msg["message_uid"] = f"uid-{i}"
    entry = _prepare(cc, msgs)
    live = copy.deepcopy(msgs)
    if change == "message dropped":
        del live[entry.compress_start + 1:entry.compress_start + 3]  # roles still alternate; positions shift
        live[entry.compress_start + 1]["tool_calls"] = [{"id": "c1", "function": {"name": "x"}}]
    elif change == "rewound":
        # A rewind into the window that regrew the same shape: the rows are new messages.
        live[entry.compress_start + 1]["message_uid"] = "uid-after-rewind"
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
    # Provider context would be needed by every later compaction too: no more passes paid for and dropped.
    stopped = pc.maybe_prepare(_agent(cc), msgs, origin="test") is None
    assert stopped is ("memory_context" in kwargs)


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


def test_identity_survives_a_session_db_round_trip_and_catches_a_rewind(tmp_path):
    from hermes_state import SessionDB

    msgs = _messages(20)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("S1", source="cli")
    for m in msgs[1:]:
        db.append_message("S1", role=m["role"], content=m["content"])
    replayed = [msgs[0]] + db.get_messages_as_conversation("S1")
    window = pc._Window("S1", 4, 15, pc.prefix_identities(replayed, 15))
    assert pc._mismatch(window, msgs, "S1", 4, 15) is None, "the live dicts before their first persist"
    assert pc._mismatch(window, replayed, "S1", 4, 15) is None
    rewound = copy.deepcopy(replayed)
    rewound[6]["message_uid"] = "a-new-row"
    assert "different message" in pc._mismatch(window, rewound, "S1", 4, 15)


# --------------------------------------------------------------------------- running passes


def test_compaction_waits_for_a_running_pass_and_forwards_its_progress():
    """No candidate yet: the compaction waits for the running pass instead of summarising again, and each
    streamed chunk of the pass touches the compacting thread's progress hook (the host's idle watchdog)."""
    from agent.auxiliary_client import aux_progress_hook

    cc = _compressor()
    cc.gate, cc.stream_progress = threading.Event(), True
    msgs = _messages(30)
    running = pc.maybe_prepare(_agent(cc), msgs, origin="test")
    assert running is not None
    host_progress = []
    pending = cc.prepared_compaction._pending
    # Release the pass only once the compaction is waiting on it.
    threading.Thread(target=lambda: wait_until(lambda: pending.on_progress is not None) and cc.gate.set()).start()
    with aux_progress_hook(lambda: host_progress.append(threading.current_thread().name)):
        compressed = cc.compress(msgs, current_tokens=cc.threshold_tokens + 1)
    running.join(5.0)

    assert [c["worker"] for c in cc.calls] == [True], "the pass's summary was spliced, no inline call"
    assert "Summary v1" in _summary_row(compressed)["content"]
    assert host_progress == ["compaction-prepare"], "pass progress reached the waiting host"
    assert cc.prepared_compaction._pending is None and cc.prepared_compaction._entry is None


def test_wait_ends_at_the_host_deadline_then_summarises_inline_and_the_pass_cannot_publish():
    from agent.auxiliary_client import aux_stream_deadline

    cc = _compressor()
    cc.gate = threading.Event()
    msgs = _messages(30)
    running = pc.maybe_prepare(_agent(cc), msgs, origin="test")
    assert running is not None

    with aux_stream_deadline(time.monotonic() + 0.2):
        cc.compress(msgs, current_tokens=cc.threshold_tokens + 1)
    assert [c["worker"] for c in cc.calls] == [False], "inline summary after the deadline"

    cc.gate.set()
    running.join(5.0)
    assert cc.prepared_compaction._entry is None


def test_prompt_reading_in_the_pass_extends_the_deadline_the_wait_honours():
    """A local server still reading the pass's prompt pushes the host deadline back exactly as it would
    inline, and the wait follows the live deadline instead of the one it started with."""
    from agent.auxiliary_client import aux_stream_deadline
    from agent.local_prompt_progress import prompt_reading_hook

    cc = _compressor()
    deadline = [time.monotonic() + 1.0]
    initial = deadline[0]
    # The pass finishes only after the deadline the wait started with has passed.
    cc.read_seconds, cc.after_read = 5.0, lambda: wait_until(lambda: time.monotonic() > initial + 0.3)
    msgs = _messages(30)
    running = pc.maybe_prepare(_agent(cc), msgs, origin="test")
    assert running is not None
    with aux_stream_deadline(lambda: deadline[0]), prompt_reading_hook(lambda s: deadline.__setitem__(0, deadline[0] + s)):
        compressed = cc.compress(msgs, current_tokens=cc.threshold_tokens + 1)
    running.join(5.0)

    assert [c["worker"] for c in cc.calls] == [True], "the extended wait outlived the initial deadline"
    assert "Summary v1" in _summary_row(compressed)["content"]


def test_host_cancel_ends_the_wait_and_cancels_the_superseded_pass(monkeypatch):
    from agent.auxiliary_client import aux_interrupt_protection, aux_progress_hook

    escaped = []
    monkeypatch.setattr(threading, "excepthook", escaped.append)
    cc = _compressor()
    cc.gate = threading.Event()
    msgs = _messages(30)
    running = pc.maybe_prepare(_agent(cc), msgs, origin="test")
    assert running is not None
    pending, host_cancel = cc.prepared_compaction._pending, threading.Event()
    threading.Thread(target=lambda: wait_until(lambda: pending.on_progress is not None)
                     and host_cancel.set()).start()
    with aux_interrupt_protection(cancel_event=host_cancel), aux_progress_hook(lambda: None):
        cc.compress(msgs, current_tokens=cc.threshold_tokens + 1)
    assert [c["worker"] for c in cc.calls] == [False], "inline summary after the host cancelled the wait"
    assert pending.cancelled.is_set(), "the pass nobody can use is told to stop streaming"
    running.join(5.0)
    assert not running.is_alive() and escaped == [], "the cancelled stream ends the pass quietly"


def test_session_reset_during_the_wait_wakes_it_without_waiting_for_the_pass():
    from agent.auxiliary_client import aux_progress_hook

    cc = _compressor()
    cc.gate = threading.Event()
    msgs = _messages(30)
    running = pc.maybe_prepare(_agent(cc), msgs, origin="test")
    assert running is not None
    pending = cc.prepared_compaction._pending
    threading.Thread(target=lambda: wait_until(lambda: pending.on_progress is not None)
                     and cc.prepared_compaction.discard("session reset")).start()
    with aux_progress_hook(lambda: None):  # a host hook marks the moment the wait starts
        cc.compress(msgs, current_tokens=cc.threshold_tokens + 1)
    assert [c["worker"] for c in cc.calls] == [False]
    cc.gate.set()
    running.join(5.0)
    assert cc.prepared_compaction._entry is None


@pytest.mark.parametrize("boundary", ["reset", "end", "rebind"])
def test_session_boundary_discards_candidate_and_running_pass(boundary):
    def cross(cc, msgs):
        if boundary == "reset":
            cc.on_session_reset()
        elif boundary == "end":
            cc.on_session_end("S1", msgs)
        else:  # /resume or /branch onto another session
            cc.bind_session_state(None, "S2")

    msgs = _messages(30)
    completed = _compressor()
    _prepare(completed, msgs)
    cross(completed, msgs)
    assert completed.prepared_compaction._entry is None

    cc = _compressor()
    cc.gate = threading.Event()
    running = pc.maybe_prepare(_agent(cc), msgs, origin="test")
    assert running is not None
    cross(cc, msgs)
    cc.gate.set()
    running.join(5.0)
    assert cc.prepared_compaction._entry is None and cc.prepared_compaction._pending is None


def test_same_session_rebind_keeps_the_candidate():
    cc = _compressor()
    msgs = _messages(30)
    entry = _prepare(cc, msgs)
    cc.bind_session_state(None, "S1")  # e.g. on_session_start after construction already bound S1
    assert cc.prepared_compaction._entry is entry
    cc.compress(_grow(msgs, 4), current_tokens=cc.threshold_tokens + 1)
    assert [c["worker"] for c in cc.calls] == [True]


# --------------------------------------------------------------------------- one pass, failures, fallback


def test_a_held_candidate_starts_no_further_pass_however_far_the_prompt_grows(monkeypatch):
    cc = _compressor()
    msgs = _messages(30)
    first = _prepare(cc, msgs)
    plans, plan = [], cc._plan_compaction_window
    monkeypatch.setattr(cc, "_plan_compaction_window", lambda m: plans.append(1) or plan(m))
    assert pc.maybe_prepare(_agent(cc, cc.threshold_tokens - 1), _grow(msgs, 30, size=2600), origin="test") is None
    assert plans == [] and len(cc.calls) == 1 and cc.prepared_compaction._entry is first


def test_failed_pass_backs_off_without_touching_live_failure_state(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    import agent.context_compressor as module

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("S1", source="cli")
    cc = _compressor()
    cc.bind_session_state(db, "S1")
    msgs = _messages(30)

    monkeypatch.setattr(_StubCompressor, "_generate_summary", module.ContextCompressor._generate_summary)
    call = MagicMock(side_effect=RuntimeError("No LLM provider configured"))
    monkeypatch.setattr(module, "call_llm", call)
    thread = pc.maybe_prepare(_agent(cc), msgs, origin="test")
    thread.join(5.0)

    assert call.call_count == 1
    assert cc.prepared_compaction._entry is None
    assert cc._summary_failure_cooldown_until == 0.0 and cc._last_summary_error is None
    assert db.get_compression_failure_cooldown("S1") is None
    assert pc.maybe_prepare(_agent(cc), msgs, origin="test") is None, "backs off"
    assert call.call_count == 1


def test_splice_carries_the_passs_main_model_fallback_and_call_telemetry(monkeypatch):
    """The summary model failed inside the pass and the main model answered: the live compressor ends up
    as the inline call would leave it (sticky main-model route, the user warning's source, the call's
    telemetry), not as if the broken summary model had worked."""
    import agent.context_compressor as module

    cc = _compressor()
    cc.summary_model = "broken/summary-model"
    msgs = _messages(30)
    monkeypatch.setattr(_StubCompressor, "_generate_summary", module.ContextCompressor._generate_summary)

    calls = []

    def call_llm(**kwargs):
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            raise RuntimeError("model not found")  # the configured summary model is broken
        message = SimpleNamespace(content="Recovered summary.", reasoning_content=None, reasoning=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    monkeypatch.setattr(module, "call_llm", call_llm)
    _prepare(cc, msgs)
    cc.compress(msgs, current_tokens=cc.threshold_tokens + 1)

    assert calls == ["compaction-prepare"] * 2, "summary model failed, main model answered, both in the pass"
    assert cc.summary_model == "", "later compactions go straight to the main model, as inline"
    assert cc._last_aux_model_failure_model == "broken/summary-model"
    telemetry = cc._last_compression_telemetry
    assert telemetry["prepared_summary"] is True and telemetry["fallback_used"] is True
    assert telemetry["aux_model"] and telemetry["aux_call_duration_ms"] is not None
    assert telemetry["method"] == "aux_fallback_main"


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
    band_floor = int(cc.threshold_tokens * (1 - pc.PREPARE_BAND_RATIO))

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
            wait_until(lambda: state._pending is None, what="the pass to finish")
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
