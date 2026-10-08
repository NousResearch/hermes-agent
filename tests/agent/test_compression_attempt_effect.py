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
