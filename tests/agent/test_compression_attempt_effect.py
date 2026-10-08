"""Attempt telemetry names why a compression ran, what it changed, and how the summary was made.

Covers the fields added to the content-free attempt record (trigger label, effect counts, method) and the
records for exits that never reach the local compressor (automatic gate, Codex route).
"""

import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import compress_context
from agent.transports.codex_app_server_session import TurnResult
from hermes_cli.observability import shared_metrics_events
from hermes_cli.observability.shared_metrics_fields import compression_fields

SECRET = "TOPSECRET_TRANSCRIPT_TEXT"


class _TodoStore:
    def format_for_injection(self):
        return ""


class _Agent:
    def __init__(self, compressor):
        self.context_compressor = compressor
        self.session_id = "session-effect-test"
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


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/main-model", provider="test-provider", threshold_percent=0.50, quiet_mode=True,
            config_context_length=100_000,
        )
    compressor.tail_token_budget = 10
    return compressor


def _messages():
    msgs = [{"role": "system", "content": "system prompt"}]
    filler = " ".join(["context"] * 200)
    for idx in range(10):
        msgs.append({"role": "user", "content": f"user message {idx} {SECRET} {filler}"})
        msgs.append({"role": "assistant", "content": f"assistant reply {idx} {SECRET} {filler}"})
    return msgs


def _attempt_records(caplog):
    prefix = "context compression attempt telemetry: "
    return [
        json.loads(record.getMessage().split(prefix, 1)[1])
        for record in caplog.records
        if prefix in record.getMessage()
    ]


@pytest.fixture
def shared_metric_rows(monkeypatch):
    rows = []
    monkeypatch.setattr(shared_metrics_events, "record_compression", lambda **kw: rows.append(compression_fields(**kw)))
    return rows


def test_committed_summary_records_trigger_effect_and_method_without_content(caplog, shared_metric_rows):
    agent = _Agent(_compressor())

    with patch.object(agent.context_compressor, "_generate_summary", return_value="SANITIZED SUMMARY"):
        with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
            compressed, _ = compress_context(agent, _messages(), "system prompt", approx_tokens=80_000, trigger="pre_api")

    [record] = _attempt_records(caplog)
    assert record["trigger_source"] == "pre_api"
    assert (record["route"], record["commit_status"], record["method"]) == ("hermes", "committed", "llm_summary")
    assert record["items_dropped"] == 0
    assert record["messages_before"] == len(_messages())
    assert record["messages_after"] < record["messages_before"]
    assert record["tokens_after"] < record["tokens_before"]
    assert record["tokens_reclaimed"] == record["tokens_before"] - record["tokens_after"]
    assert record["token_count_method"] == "estimate_rough"
    assert record["tool_results_pruned"] == 0 and record["reasoning_items_pruned"] == 0
    assert SECRET not in json.dumps(record) and "SANITIZED SUMMARY" not in json.dumps(record)
    # The precise label stays in the log; the shared metric keeps its coarse bucket.
    assert shared_metric_rows == [
        {"trigger": "auto", "outcome": "success", "context_fill_bucket": "75_to_90", "failure_class": "none"}
    ]


def test_deterministic_fallback_commit_records_its_method_and_dropped_items(caplog):
    agent = _Agent(_compressor())

    with patch.object(agent.context_compressor, "_generate_summary", return_value=None):
        with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
            compressed, _ = compress_context(agent, _messages(), "system prompt", approx_tokens=80_000)

    [record] = _attempt_records(caplog)
    assert record["commit_status"] == "committed"
    assert record["method"] == "deterministic_fallback"
    assert record["items_dropped"] > 0
    assert record["failure_class"] == "summary_generation_failed"
    assert SECRET not in json.dumps(record)


def test_gate_blocked_attempt_logs_one_blocked_record_and_no_shared_metric(caplog, shared_metric_rows):
    compressor = _compressor()
    agent = _Agent(compressor)
    messages = _messages()

    with patch.object(type(compressor), "_compression_block_reason", return_value="cooldown:42"), \
            patch.object(type(compressor), "_automatic_compression_blocked", return_value=True), \
            patch.object(compressor, "_generate_summary") as summary:
        with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
            returned, _ = compress_context(agent, messages, "system prompt", approx_tokens=80_000, trigger="post_tool")

    assert returned is messages
    summary.assert_not_called()
    [record] = _attempt_records(caplog)
    assert (record["commit_status"], record["failure_class"]) == ("blocked", "blocked:cooldown")
    assert (record["trigger_source"], record["current_estimated_tokens"]) == ("post_tool", 80_000)
    assert record["attempt_id"] == agent._compression_attempt_id
    assert shared_metric_rows == []


class _CodexSession:
    def __init__(self, result):
        self.result = result

    def compact_thread(self):
        return self.result

    def close(self):
        pass


def _codex_agent(result, auto_mode):
    agent = SimpleNamespace(
        api_mode="codex_app_server", codex_app_server_auto_compaction=auto_mode, session_id="codex-session",
        platform="cli", model="gpt-test", provider="openai-codex", _cached_system_prompt="cached prompt",
        _codex_session=_CodexSession(result), _compression_activity_heartbeat_interval=0.1,
        context_compressor=SimpleNamespace(
            compression_count=0, last_compression_rough_tokens=0, last_prompt_tokens=1, last_completion_tokens=0,
            awaiting_real_usage_after_compression=False,
        ),
    )
    agent._touch_activity = lambda *_args, **_kwargs: None
    agent._emit_status = agent._emit_warning = lambda _message: None
    return agent


@pytest.mark.parametrize(
    ("auto_mode", "force", "result", "expected"),
    [
        ("native", False, TurnResult(thread_id="t1"), ("skipped", "codex_auto_native", "none")),
        ("hermes", True, TurnResult(thread_id="t1", turn_id="c1"), ("committed", None, "provider")),
        ("hermes", True, TurnResult(thread_id="t1", error="boom"), ("failed", "codex_compaction_failed", "none")),
    ],
)
def test_codex_route_logs_exactly_one_record_per_exit(caplog, shared_metric_rows, auto_mode, force, result, expected):
    agent = _codex_agent(result, auto_mode)

    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        compress_context(agent, [{"role": "user", "content": SECRET}], "system", approx_tokens=90_000, force=force)

    [record] = _attempt_records(caplog)
    assert (record["commit_status"], record["failure_class"], record["method"]) == expected
    assert record["route"] == "codex_app_server"
    assert record["trigger_source"] == ("manual" if force else "auto")
    assert SECRET not in json.dumps(record) and "boom" not in json.dumps(record)
    assert shared_metric_rows == []


@pytest.mark.parametrize(
    "trigger", ["idle", "turn_start_threshold", "engine_preflight", "pre_api", "post_tool", "gateway_hygiene"]
)
def test_precise_automatic_triggers_keep_the_coarse_shared_metric_bucket(trigger):
    fields = compression_fields(trigger=trigger, outcome="success", tokens_before=1, context_length=10)
    assert fields["trigger"] == "auto"


def test_blocked_record_never_carries_the_cooldown_seconds(caplog):
    from agent.conversation_compression_telemetry import _emit_blocked_attempt_telemetry

    compressor = SimpleNamespace(_compression_block_reason=lambda: "structural_backoff:917")
    agent = SimpleNamespace(context_compressor=compressor, session_id="s", _compression_attempt_id="a")

    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        _emit_blocked_attempt_telemetry(agent, time.monotonic(), None)

    [record] = _attempt_records(caplog)
    assert record["failure_class"] == "blocked:structural_backoff"
    assert "917" not in json.dumps(record)


def _abort_on_stale_snapshot(agent):
    compress_context(
        agent, _messages(), "system prompt", approx_tokens=80_000, trigger="post_tool",
        snapshot_is_current=lambda: False,
    )


def _abort_at_the_commit_fence(agent):
    # The fence refuses the commit after the summary ran: the attempt restores the compressor's pre-attempt
    # state, which puts the PREVIOUS attempt's telemetry dict back before the record is written.
    from agent.conversation_compression import CompressionCommitFence

    fence = CompressionCommitFence()
    with patch.object(fence, "begin_commit", return_value=False):
        compress_context(
            agent, _messages(), "system prompt", approx_tokens=80_000, trigger="post_tool", commit_fence=fence,
        )


def _refuse_on_a_saturated_pool(agent):
    from agent.conversation_compression import run_compress_context_with_progress_timeout

    with patch("agent.conversation_compression._try_admit_compression_job", return_value=False):
        run_compress_context_with_progress_timeout(
            worker=lambda _fence: pytest.fail("a refused job must not run"), messages=_messages(),
            system_prompt_fallback="system prompt", idle_timeout_seconds=5.0, total_ceiling_seconds=5.0,
            telemetry_agent=agent,
        )


@pytest.mark.parametrize(
    ("abort", "trigger", "failure_class"),
    [
        (_abort_on_stale_snapshot, "post_tool", "snapshot_stale"),
        (_abort_at_the_commit_fence, "post_tool", "commit_fence_cancelled"),
        # Refused before any attempt begins: a fresh id, and the host does not know the trigger.
        (_refuse_on_a_saturated_pool, "unknown", "pool_saturated"),
    ],
)
def test_aborted_record_after_a_commit_describes_its_own_attempt(caplog, abort, trigger, failure_class):
    agent = _Agent(_compressor())

    with patch.object(agent.context_compressor, "_generate_summary", return_value="SUMMARY"):
        with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
            compress_context(agent, _messages(), "system prompt", approx_tokens=80_000, trigger="pre_api")
            abort(agent)

    committed, aborted = _attempt_records(caplog)
    assert committed["commit_status"] == "committed" and committed["tokens_reclaimed"] > 0
    assert aborted["attempt_id"] != committed["attempt_id"]
    assert (aborted["commit_status"], aborted["failure_class"]) == ("aborted", failure_class)
    assert (aborted["trigger_source"], aborted["session_id"]) == (trigger, agent.session_id)
    # No rewrite happened in this attempt, so it claims none.
    assert aborted["method"] == "none" and not aborted["fallback_used"]
    assert [aborted.get(key) for key in ("tokens_before", "tokens_after", "tokens_reclaimed")] == [None] * 3


def _adopt_child(agent, _db, _parent_sid):
    agent.session_id = "adopted-child"
    return [{"role": "user", "content": "child transcript"}]


def _unreadable_ownership(_db, _sid):
    raise OSError("database is locked")


@pytest.mark.parametrize(
    ("ownership_patch", "expected"),
    [
        # The lease found the parent already rotated and adopted the live child.
        ({"_session_was_rotated_by_compression": lambda _db, _sid: True, "_adopt_live_compression_child": _adopt_child},
         ("skipped", "session_ownership_lost")),
        # The ownership lookup itself failed: ownership is unknown, not lost.
        ({"_session_was_rotated_by_compression": _unreadable_ownership}, ("aborted", "session_ownership_unreadable")),
    ],
)
def test_ownership_exit_records_the_session_the_attempt_ran_in(caplog, tmp_path, ownership_patch, expected):
    from hermes_state import SessionDB

    agent = _Agent(_compressor())
    agent._session_db = SessionDB(db_path=tmp_path / "state.db")
    agent._session_db.create_session(agent.session_id, "cli", model="test/main-model")

    with patch.multiple("agent.conversation_compression", **ownership_patch), \
            patch.object(agent.context_compressor, "_generate_summary") as summary:
        with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
            compress_context(agent, _messages(), "system prompt", approx_tokens=80_000, trigger="post_tool")

    summary.assert_not_called()
    [record] = _attempt_records(caplog)
    assert (record["commit_status"], record["failure_class"]) == expected
    assert (record["session_id"], record["trigger_source"]) == ("session-effect-test", "post_tool")
