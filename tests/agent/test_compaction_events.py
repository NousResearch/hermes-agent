"""``hermes.compaction`` v1 records: one content-free record per history rewrite, on every path.

The record is the contract a Relay consumer reads, so these tests pin the field set, the closed enums,
the ``compaction`` vs ``compaction.attempt`` name rule, and that publishing can never change what a
compaction returns.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import compaction_events
from agent.context_compressor import ContextCompressor
from agent.conversation_compression import CompressionCheckpointUnavailable, CompressionCommitFence, compress_context
from agent.transports.codex_app_server_session import TurnResult

SECRET = "TOPSECRET_TRANSCRIPT_TEXT"
V1_FIELDS = {
    "harness", "kind", "scope", "official", "method", "trigger", "trigger_class", "overflow_reason", "outcome",
    "failure_class", "attempt_id", "session_id", "tokens_before", "tokens_after", "tokens_reclaimed",
    "token_count_method", "messages_before", "messages_after", "items_dropped", "context_limit", "threshold_tokens",
    "protected_head_tokens", "middle_window_tokens", "protected_tail_tokens", "duration_ms",
    "summary_generation_ms", "aux_call_duration_ms", "queue_wait_ms", "commit_ms", "summarizer_provider",
    "summarizer_model", "tool_results_pruned", "reasoning_items_pruned", "summary_input_omitted_chars",
    "compression_count", "in_place", "session_rotated", "split_status", "cache_break", "has_focus_topic",
}


@pytest.fixture
def published(monkeypatch):
    """Every (session_id, mark name, payload) handed to the Relay emitter."""
    marks = []
    monkeypatch.setattr("agent.relay_compaction.compaction_target_available", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "agent.relay_compaction.emit_compaction_mark",
        lambda session_id, name, data, **_kwargs: marks.append((session_id, name, data)) or True,
    )
    return marks


class _TodoStore:
    def format_for_injection(self):
        return ""


class _Agent:
    def __init__(self, compressor):
        self.context_compressor = compressor
        self.session_id = "session-events-test"
        self.platform = "cli"
        self.model = "test/main-model"
        self.provider = "test-provider"
        self.tools = []
        self._compression_feasibility_checked = True
        self.compression_in_place = False
        self._memory_manager = None
        self._session_db = None
        self._todo_store = _TodoStore()
        self._cached_system_prompt = None

    def _emit_status(self, _message):
        pass

    def _emit_warning(self, _message):
        pass

    def _invalidate_system_prompt(self):
        self._cached_system_prompt = None

    def _build_system_prompt(self, system_message):
        return system_message

    def commit_memory_session(self, _messages):
        pass


def _compressor(**kwargs):
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/main-model", provider="test-provider", threshold_percent=0.50, quiet_mode=True,
            config_context_length=100_000, **kwargs,
        )
    compressor.tail_token_budget = 10
    return compressor


def _messages():
    filler = " ".join(["context"] * 200)
    msgs = [{"role": "system", "content": "system prompt"}]
    for idx in range(10):
        msgs.append({"role": "user", "content": f"user message {idx} {SECRET} {filler}"})
        msgs.append({"role": "assistant", "content": f"assistant reply {idx} {SECRET} {filler}"})
    return msgs


def test_committed_summary_publishes_one_compaction_mark_with_the_v1_shape(published):
    agent = _Agent(_compressor())

    with patch.object(agent.context_compressor, "_generate_summary", return_value="SANITIZED SUMMARY"):
        compress_context(
            agent, _messages(), "system prompt", approx_tokens=80_000, trigger="pre_api",
            focus_topic=f"focus on {SECRET}",
        )

    [(session_id, name, payload)] = published
    assert (session_id, name) == ("session-events-test", "compaction")
    assert set(payload) == V1_FIELDS
    assert payload["harness"] == "hermes"
    assert (payload["kind"], payload["scope"], payload["official"]) == ("summarize", "history", True)
    assert (payload["method"], payload["outcome"], payload["failure_class"]) == ("llm_summary", "committed", "none")
    assert (payload["trigger"], payload["trigger_class"]) == ("pre_api", "auto")
    assert payload["tokens_reclaimed"] == payload["tokens_before"] - payload["tokens_after"] > 0
    assert payload["token_count_method"] == "estimate_rough"
    assert payload["compression_count"] == 1 and payload["cache_break"] is True
    assert payload["has_focus_topic"] is True
    raw = json.dumps(payload)
    assert SECRET not in raw and "SANITIZED SUMMARY" not in raw and "focus on" not in raw


def test_deterministic_fallback_is_a_committed_compaction_with_its_method(published):
    agent = _Agent(_compressor())

    with patch.object(agent.context_compressor, "_generate_summary", return_value=None):
        compress_context(agent, _messages(), "system prompt", approx_tokens=80_000)

    [(_sid, name, payload)] = published
    assert name == "compaction"
    assert (payload["method"], payload["failure_class"]) == ("deterministic_fallback", "summary_generation_failed")
    assert payload["items_dropped"] > 0
    assert payload["trigger"] == "unknown"  # an unlabelled automatic caller


def test_failed_summary_is_an_attempt_mark_not_a_compaction(published):
    agent = _Agent(_compressor())
    agent.context_compressor.abort_on_summary_failure = True

    with patch.object(agent.context_compressor, "_generate_summary", return_value=None):
        compress_context(agent, _messages(), "system prompt", approx_tokens=80_000, trigger="post_tool")

    [(_sid, name, payload)] = published
    assert name == "compaction.attempt"
    assert (payload["outcome"], payload["failure_class"]) == ("failed", "summary_generation_aborted")
    assert payload["cache_break"] is False and payload["in_place"] is None


def test_an_engine_that_returns_an_empty_transcript_publishes_one_failed_attempt_mark(published):
    agent = _Agent(_compressor())

    with patch.object(agent.context_compressor, "compress", return_value=[]):
        returned, _ = compress_context(agent, _messages(), "system prompt", approx_tokens=80_000, trigger="pre_api")

    assert returned == _messages()
    [(session_id, name, payload)] = published
    assert (session_id, name) == ("session-events-test", "compaction.attempt")
    assert (payload["outcome"], payload["failure_class"], payload["method"]) == ("failed", "empty_transcript", "none")


def test_a_raising_emitter_never_changes_the_compaction_result(monkeypatch, caplog):
    def explode(*_args, **_kwargs):
        raise RuntimeError("relay down")

    monkeypatch.setattr("agent.relay_compaction.emit_compaction_mark", explode)
    monkeypatch.setattr(compaction_events, "_warned", False)
    results = []
    for _ in range(2):
        agent = _Agent(_compressor())
        with patch.object(agent.context_compressor, "_generate_summary", return_value="SANITIZED SUMMARY"):
            with caplog.at_level(logging.DEBUG, logger="agent.compaction_events"):
                results.append(compress_context(agent, _messages(), "system prompt", approx_tokens=80_000))

    assert all(len(compressed) < len(_messages()) for compressed, _ in results)
    failures = [r for r in caplog.records if r.getMessage() == "compaction event publish failed"]
    assert [r.levelno for r in failures] == [logging.WARNING, logging.DEBUG]
    assert failures[0].exc_info is not None


def test_prune_without_relay_target_skips_transcript_estimates(monkeypatch):
    agent = _Agent(_compressor())
    messages = _messages()
    monkeypatch.setattr("agent.relay_compaction.compaction_target_available", lambda *_args, **_kwargs: False)
    with patch("agent.compaction_events.estimate_messages_tokens_rough") as estimate:
        compaction_events.publish_prune(agent, messages, messages[:-1], 1)
    estimate.assert_not_called()


def test_overflow_attempt_records_its_classified_overflow_reason(published):
    agent = _Agent(_compressor())

    with patch.object(agent.context_compressor, "_generate_summary", return_value="SANITIZED SUMMARY"):
        compress_context(
            agent, _messages(), "system prompt", approx_tokens=80_000, bypass_cooldown=True, trigger="overflow",
            overflow_reason="context_overflow",
        )

    [(_sid, name, payload)] = published
    assert name == "compaction"
    assert (payload["trigger"], payload["trigger_class"], payload["overflow_reason"]) == (
        "overflow", "overflow", "context_overflow",
    )


def test_gate_blocked_attempt_publishes_exactly_one_blocked_attempt_mark(published):
    compressor = _compressor()
    agent = _Agent(compressor)

    with patch.object(type(compressor), "_compression_block_reason", return_value="cooldown:42"), \
            patch.object(type(compressor), "_automatic_compression_blocked", return_value=True):
        compress_context(agent, _messages(), "system prompt", approx_tokens=80_000, trigger="idle")

    [(_sid, name, payload)] = published
    assert name == "compaction.attempt"
    assert (payload["outcome"], payload["failure_class"], payload["trigger"]) == ("blocked", "blocked:cooldown", "idle")


def _codex_agent(result, auto_mode):
    agent = SimpleNamespace(
        api_mode="codex_app_server", codex_app_server_auto_compaction=auto_mode, session_id="codex-session",
        platform="cli", model="gpt-test", provider="openai-codex", _cached_system_prompt="cached prompt",
        _codex_session=SimpleNamespace(compact_thread=lambda: result, close=lambda: None),
        _compression_activity_heartbeat_interval=0.1,
        context_compressor=SimpleNamespace(
            compression_count=0, last_compression_rough_tokens=0, last_prompt_tokens=1, last_completion_tokens=0,
            awaiting_real_usage_after_compression=False,
        ),
    )
    agent._touch_activity = lambda *_args, **_kwargs: None
    agent._emit_status = agent._emit_warning = lambda _message: None
    return agent


@pytest.mark.parametrize(
    ("auto_mode", "force", "result", "name", "outcome"),
    [
        ("native", False, TurnResult(thread_id="t1"), "compaction.attempt", "skipped"),
        ("hermes", True, TurnResult(thread_id="t1", turn_id="c1"), "compaction", "committed"),
        ("hermes", True, TurnResult(thread_id="t1", error="boom"), "compaction.attempt", "failed"),
    ],
)
def test_codex_route_publishes_one_provider_record_per_exit(published, auto_mode, force, result, name, outcome):
    compress_context(_codex_agent(result, auto_mode), [{"role": "user", "content": SECRET}], "system", force=force)

    [(session_id, mark, payload)] = published
    assert (session_id, mark, payload["outcome"]) == ("codex-session", name, outcome)
    assert (payload["kind"], payload["scope"], payload["official"]) == ("provider_native", "provider", True)
    assert "boom" not in json.dumps(payload)


def _raising_codex_agent():
    def _compact_thread():
        raise RuntimeError(f"boom {SECRET}")

    agent = _codex_agent(None, "hermes")
    agent._codex_session = SimpleNamespace(compact_thread=_compact_thread, close=lambda: None)
    return agent


def _checkpoint_required_codex_agent():
    agent = _codex_agent(TurnResult(thread_id="t1"), "hermes")
    agent.compression_checkpoint_required = True
    return agent


@pytest.mark.parametrize(
    ("make_agent", "error"),
    [
        pytest.param(_raising_codex_agent, RuntimeError, id="compact-thread-raises"),
        pytest.param(_checkpoint_required_codex_agent, CompressionCheckpointUnavailable, id="checkpoint-required"),
    ],
)
def test_codex_route_that_raises_publishes_one_failed_attempt_mark(published, make_agent, error):
    with pytest.raises(error):
        compress_context(make_agent(), [{"role": "user", "content": SECRET}], "system", force=True)

    [(session_id, mark, payload)] = published
    assert (session_id, mark) == ("codex-session", "compaction.attempt")
    assert (payload["kind"], payload["outcome"], payload["failure_class"]) == ("provider_native", "failed", "exception")
    assert SECRET not in json.dumps(payload) and "boom" not in json.dumps(payload)


def test_codex_native_compaction_observed_by_hermes_is_an_unofficial_provider_compaction(published):
    from agent.codex_runtime import _record_codex_app_server_compaction

    agent = _codex_agent(None, "native")
    agent.event_callback = None
    _record_codex_app_server_compaction(agent, TurnResult(thread_id="t1", turn_id="n1", compacted=True))

    [(_sid, name, payload)] = published
    assert name == "compaction"
    assert (payload["kind"], payload["official"], payload["trigger"]) == ("provider_native", False, "provider")
    assert payload["trigger_class"] == "provider" and payload["compression_count"] == 1


@pytest.mark.parametrize(
    ("outcome", "name", "expected"),
    [
        ("absorbed", "compaction", ("committed", "llm_summary", "none")),
        ("defrag", "compaction", ("committed", "llm_summary", "none")),
        ("stale_generation", "compaction.attempt", ("skipped", "none", "stale_generation")),
        ("summarize_failed", "compaction.attempt", ("failed", "none", "summarize_failed")),
    ],
)
def test_micro_compaction_pass_publishes_one_record(published, outcome, name, expected):
    compressor = _compressor()
    compressor._session_id = "micro-session"
    compressor._emit_micro_compaction_telemetry(
        outcome=outcome, messages_before=12, messages_after=10, tokens_before=900, tokens_after=600, duration_ms=5,
    )

    [(session_id, mark, payload)] = published
    assert (session_id, mark) == ("micro-session", name)
    assert (payload["outcome"], payload["method"], payload["failure_class"]) == expected
    assert (payload["kind"], payload["official"], payload["trigger"]) == ("micro_summarize", False, "between_turns")
    assert payload["tokens_reclaimed"] == 300


def _prune_after_tool_results(persist_disabled):
    """Drive the post-tool caller with the attempt budget spent, so it takes the proactive prune branch."""
    from agent.turn_preflight import compress_after_tool_results

    compressor = _compressor(
        proactive_prune_tokens=48_000, proactive_prune_min_result_chars=8_000, proactive_prune_min_reclaim_tokens=4096,
        protect_first_n=2, protect_last_n=4,
    )
    compressor.last_prompt_tokens = 120_000
    compressor._session_db = object()  # a bound store; the prune is skipped without one
    agent = SimpleNamespace(
        context_compressor=compressor, compression_enabled=True, session_id="prune-session", tools=[],
        _persist_disabled=persist_disabled, _usage_anchor=None, _compression_feasibility_checked=True,
        _warn_context_overflow_blocked=lambda *_args: None,
    )
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(8):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
        ]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": (chr(65 + i) + SECRET) * 600 if i < 3 else "ok"})
    verdict = compress_after_tool_results(
        agent, messages=msgs, system_message="sys", user_message="ask", active_system_prompt="sys",
        conversation_history=None, compression_attempts=3, max_compression_attempts=3, effective_task_id="task",
        final_response=None, turn_exit_reason=None, current_turn_user_idx=0,
    )
    return msgs, verdict.messages


def test_committed_proactive_prune_publishes_one_record(published):
    msgs, pruned = _prune_after_tool_results(persist_disabled=False)

    assert pruned is not msgs
    [(session_id, name, payload)] = published
    assert (session_id, name) == ("prune-session", "compaction")
    assert (payload["kind"], payload["method"], payload["trigger"]) == (
        "prune_tool_results", "deterministic_prune", "proactive_prune",
    )
    assert payload["tool_results_pruned"] > 0 and payload["tokens_reclaimed"] > 0
    assert payload["token_count_method"] == "estimate_rough"
    assert payload["messages_before"] == payload["messages_after"] == len(msgs)
    assert SECRET not in json.dumps(payload)


def test_a_persistence_detached_fork_publishes_nothing(published):
    """A background-review fork shares the live session id but rewrites only its own transcript."""
    agent = _Agent(_compressor())
    agent._persist_disabled = True

    with patch.object(agent.context_compressor, "_generate_summary", return_value="SANITIZED SUMMARY"):
        compressed, _ = compress_context(agent, _messages(), "system prompt", approx_tokens=80_000, trigger="pre_api")
    msgs, pruned = _prune_after_tool_results(persist_disabled=True)

    assert len(compressed) < len(_messages()) and pruned is not msgs
    assert published == []


def test_an_attempt_aborted_after_a_commit_marks_only_its_own_attempt(published):
    agent = _Agent(_compressor())

    with patch.object(agent.context_compressor, "_generate_summary", return_value="SANITIZED SUMMARY"):
        compress_context(agent, _messages(), "system prompt", approx_tokens=80_000, trigger="pre_api")
        compress_context(
            agent, _messages(), "system prompt", approx_tokens=80_000, trigger="post_tool",
            snapshot_is_current=lambda: False,
        )

    [(_, first, committed), (_, second, aborted)] = published
    assert (first, second) == ("compaction", "compaction.attempt")
    assert aborted["attempt_id"] != committed["attempt_id"]
    assert (aborted["trigger"], aborted["failure_class"], aborted["method"]) == ("post_tool", "snapshot_stale", "none")
    assert aborted["tokens_reclaimed"] is None and aborted["cache_break"] is False


def test_rotation_publishes_after_the_split_with_the_attempts_own_session(published):
    """A rotating commit re-points the agent at a child session and its memory flush resets the compressor's
    telemetry; the record is still built from this attempt's telemetry and names the session it started in."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
            session_db=SessionDB(db_path=Path(tmpdir) / "state.db"), session_id="rotating-session",
            skip_context_files=True, skip_memory=True,
        )
        agent.compression_in_place = False
        agent.context_compressor.tail_token_budget = 10
        with patch.object(agent.context_compressor, "_generate_summary", return_value="SANITIZED SUMMARY"):
            agent._compress_context(_messages(), "sys", approx_tokens=80_000, trigger="pre_api")
        agent.close()

    assert agent.session_id != "rotating-session"
    [(session_id, name, payload)] = published
    assert (session_id, name) == ("rotating-session", "compaction")
    assert (payload["split_status"], payload["session_rotated"], payload["in_place"]) == (
        "rotated_committed", True, False,
    )
    assert payload["tokens_after"] is not None and payload["method"] == "llm_summary"


def test_a_split_that_fails_after_rewriting_history_is_still_a_compaction(published):
    """The in-place commit stored the compacted transcript, then a later bookkeeping write failed: the model
    sees the compacted history next, so Relay must reset freshness, and the record says what failed."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        db = SessionDB(db_path=Path(tmpdir) / "state.db")
        db.create_session("in-place-session", "cli", model="test/model")
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
            session_db=db, session_id="in-place-session", skip_context_files=True, skip_memory=True,
        )
        agent.compression_in_place = True
        agent.context_compressor.tail_token_budget = 10
        with patch.object(agent.context_compressor, "_generate_summary", return_value="SANITIZED SUMMARY"), \
                patch.object(SessionDB, "update_system_prompt", side_effect=OSError("disk full")):
            compressed, _ = agent._compress_context(_messages(), "sys", approx_tokens=80_000, trigger="pre_api")
        agent.close()

    assert len(compressed) < len(_messages())
    [(_sid, name, payload)] = published
    assert name == "compaction"
    assert (payload["outcome"], payload["failure_class"], payload["split_status"]) == (
        "committed", "session_split_failed", "failed_not_indexed",
    )
    assert (payload["in_place"], payload["session_rotated"], payload["cache_break"]) == (None, None, True)


def test_an_in_place_engine_whose_split_fails_after_the_archive_is_still_a_compaction(published):
    """An engine may compact its input list in place, so the returned list IS the input. After the atomic
    archive stored it, a later bookkeeping failure must still mark a compaction (freshness reset)."""
    from hermes_state import SessionDB
    from run_agent import AIAgent

    def _in_place(msgs, **_kwargs):
        compressor = agent.context_compressor
        compressor._active_compression_telemetry = compressor._last_compression_telemetry = (
            compressor._begin_compression_telemetry(current_tokens=80_000)
        )
        compressor._active_compression_telemetry.update(messages_before=len(msgs), messages_after=999)
        msgs[:] = [msgs[0], {"role": "assistant", "content": "SANITIZED SUMMARY"}, msgs[-1]]
        return msgs

    with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        db = SessionDB(db_path=Path(tmpdir) / "state.db")
        db.create_session("in-place-engine", "cli", model="test/model")
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
            session_db=db, session_id="in-place-engine", skip_context_files=True, skip_memory=True,
        )
        agent.compression_in_place = True
        messages = _messages()
        with patch.object(type(agent.context_compressor), "compress", side_effect=_in_place), \
                patch.object(SessionDB, "update_system_prompt", side_effect=OSError("disk full")):
            # A caller-owned fence takes the direct path, where the engine mutates the caller's list itself.
            compressed, _ = agent._compress_context(
                messages, "sys", approx_tokens=80_000, trigger="pre_api", commit_fence=CompressionCommitFence(),
            )
        agent.close()

    assert compressed is messages and len(compressed) < len(_messages())
    [(_sid, name, payload)] = published
    assert (name, payload["outcome"], payload["split_status"]) == ("compaction", "committed", "failed_not_indexed")
    # The candidate's counts are replaced by the transcript that survived.
    assert payload["messages_after"] == len(compressed)
