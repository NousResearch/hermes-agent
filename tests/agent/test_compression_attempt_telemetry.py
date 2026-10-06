import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.conversation_compression import _emit_aborted_attempt_telemetry, compress_context
from agent.context_compressor import ContextCompressor


class _TodoStore:
    def format_for_injection(self):
        return ""


class _Agent:
    def __init__(self, compressor):
        self.context_compressor = compressor
        self.session_id = "session-telemetry-test"
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


def _messages(secret_text="TOPSECRET_TRANSCRIPT_TEXT"):
    msgs = [{"role": "system", "content": "system prompt"}]
    for idx in range(10):
        msgs.append({"role": "user", "content": f"user message {idx} {secret_text}"})
        msgs.append({"role": "assistant", "content": f"assistant reply {idx} {secret_text}"})
    return msgs


def _extract_telemetry(caplog):
    records = [
        record.getMessage()
        for record in caplog.records
        if "context compression attempt telemetry:" in record.getMessage()
    ]
    assert len(records) == 1
    return json.loads(records[0].split("context compression attempt telemetry: ", 1)[1])


def test_compression_attempt_telemetry_is_metadata_only(caplog):
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/main-model",
            provider="test-provider",
            threshold_percent=0.50,
            quiet_mode=True,
            config_context_length=100_000,
        )
    compressor.tail_token_budget = 10
    agent = _Agent(compressor)

    with patch.object(compressor, "_generate_summary", return_value="SANITIZED SUMMARY"):
        with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
            compressed, system_prompt = compress_context(
                agent,
                _messages(),
                "system prompt",
                approx_tokens=75_000,
                force=True,
            )

    assert system_prompt == "system prompt"
    assert compressed is not None
    payload = _extract_telemetry(caplog)

    assert payload["event"] == "compression_attempt"
    assert payload["attempt_id"]
    assert payload["session_id"] == "session-telemetry-test"
    assert payload["trigger_source"] == "manual"
    assert payload["main_model"] == "test/main-model"
    assert payload["main_context_limit"] == 100_000
    assert payload["current_estimated_tokens"] == 75_000
    assert payload["effective_threshold"] == compressor.threshold_tokens
    assert payload["protected_head_tokens"] is not None
    assert payload["protected_tail_tokens"] is not None
    assert payload["middle_window_tokens"] is not None
    assert payload["chunking"] is False
    assert payload["chunk_count"] in {0, 1}
    assert payload["commit_status"] == "committed"
    assert payload["split_status"] == "not_applicable"
    assert payload["fallback_used"] is False
    assert isinstance(payload["total_duration_ms"], int)
    assert isinstance(payload["commit_ms"], int)
    assert payload["queue_wait_ms"] is None
    assert payload["prompt_build_ms"] is None
    assert payload["time_to_first_progress_ms"] is None
    assert payload["summary_generation_ms"] is None

    raw_log = json.dumps(payload)
    assert "TOPSECRET_TRANSCRIPT_TEXT" not in raw_log
    assert "SANITIZED SUMMARY" not in raw_log
    assert "user message" not in raw_log
    assert "assistant reply" not in raw_log


def test_aux_call_telemetry_records_durations_without_content(caplog):
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/main-model",
            provider="test-provider",
            threshold_percent=0.50,
            quiet_mode=True,
            config_context_length=100_000,
        )
    compressor.tail_token_budget = 10
    agent = _Agent(compressor)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="SANITIZED SUMMARY"))]
    )

    with patch("agent.context_compressor.call_llm", return_value=response):
        with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
            compress_context(
                agent,
                _messages(),
                "system prompt",
                approx_tokens=75_000,
            )

    payload = _extract_telemetry(caplog)
    assert payload["aux_prompt_tokens"] is not None
    # Current main intentionally omits max_tokens from the aux summary call
    # (the summary budget is prompt-level guidance only), so no output
    # reservation is recorded.
    assert payload["aux_output_reservation"] is None
    assert isinstance(payload["aux_call_duration_ms"], int)
    assert payload["aux_provider"]
    assert payload["aux_model"]

    raw_log = json.dumps(payload)
    assert "TOPSECRET_TRANSCRIPT_TEXT" not in raw_log
    assert "SANITIZED SUMMARY" not in raw_log


def test_aux_call_telemetry_records_content_free_phase_timings():
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/main-model",
            provider="test-provider",
            threshold_percent=0.50,
            quiet_mode=True,
            config_context_length=100_000,
        )
    compressor._begin_compression_telemetry(current_tokens=75_000)

    compressor._record_aux_compression_call(
        prompt_messages=[{"role": "user", "content": "TOPSECRET_TRANSCRIPT_TEXT"}],
        max_tokens=1400,
        duration_ms=22,
        aux_provider="ollama",
        aux_model="qwen3:8b",
        phase_timings={
            "queue_wait_ms": 3,
            "prompt_build_ms": 5,
            "time_to_first_progress_ms": 7,
            "summary_generation_ms": 19,
            "commit_ms": 11,
        },
    )

    payload = compressor._last_compression_telemetry
    assert payload is not None
    assert {key: payload[key] for key in (
        "queue_wait_ms",
        "prompt_build_ms",
        "time_to_first_progress_ms",
        "summary_generation_ms",
        "commit_ms",
    )} == {
        "queue_wait_ms": 3,
        "prompt_build_ms": 5,
        "time_to_first_progress_ms": 7,
        "summary_generation_ms": 19,
        "commit_ms": 11,
    }
    assert "TOPSECRET_TRANSCRIPT_TEXT" not in json.dumps(payload)


def test_automatic_compaction_counts_once_in_shared_metrics(monkeypatch):
    from hermes_cli.observability import shared_metrics_events
    from hermes_cli.observability.shared_metrics_fields import compression_fields

    calls = []
    monkeypatch.setattr(shared_metrics_events, "record_compression", lambda **kw: calls.append(kw))
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/main-model", provider="test-provider", threshold_percent=0.50, quiet_mode=True,
            config_context_length=100_000,
        )
    compressor.tail_token_budget = 10
    agent = _Agent(compressor)

    with patch.object(compressor, "_generate_summary", return_value="SANITIZED SUMMARY"):
        compress_context(agent, _messages(), "system prompt", approx_tokens=80_000)

    assert [compression_fields(**kw) for kw in calls] == [
        {"trigger": "auto", "outcome": "success", "context_fill_bucket": "75_to_90", "failure_class": "none"}
    ]


def test_structural_no_op_is_counted_skipped_with_the_compressors_class(caplog, monkeypatch):
    """A transcript that fits inside the protected tail makes no summary call: the attempt log names
    ``no_compressible_window`` (not the caller's generic ``no_progress``) and the shared metric counts
    it ``skipped``, not ``failed`` (#131412)."""
    from hermes_cli.observability import shared_metrics_events
    from hermes_cli.observability.shared_metrics_fields import compression_fields

    recorded = []
    monkeypatch.setattr(shared_metrics_events, "record_compression", lambda **kw: recorded.append(kw))
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/main-model", provider="test-provider", threshold_percent=0.50, quiet_mode=True,
            config_context_length=100_000,
        )
    agent = _Agent(compressor)
    messages = _messages()

    with patch.object(compressor, "_find_tail_cut_by_tokens", return_value=0):
        with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
            compressed, _ = compress_context(agent, messages, "system prompt", approx_tokens=75_000)

    assert compressed == messages
    payload = _extract_telemetry(caplog)
    assert (payload["commit_status"], payload["failure_class"]) == ("aborted", "no_compressible_window")
    assert [(f["outcome"], f["failure_class"]) for f in (compression_fields(**kw) for kw in recorded)] == [
        ("skipped", "no_compressible_window")
    ]


@pytest.mark.parametrize(
    ("recorded", "caller", "expected"),
    [
        # Nothing recorded: the caller's verdict is the only information there is.
        (None, "no_progress", "no_progress"),
        # Sibling generic verdict: a terminal summary failure keeps its documented class.
        ("summary_auth_failure", "summary_generation_aborted", "summary_auth_failure"),
        # A caller-side event the compressor cannot see still outranks a recorded class.
        ("no_compressible_window", "attempt_superseded", "attempt_superseded"),
        # A snapshot restore put the PREVIOUS attempt's telemetry back: its class is not this attempt's.
        (("no_compressible_window", "previous-attempt"), "no_progress", "no_progress"),
    ],
)
def test_generic_caller_verdicts_defer_to_the_recorded_failure_class(caplog, monkeypatch, recorded, caller, expected):
    from hermes_cli.observability import shared_metrics_events

    recorded, attempt_id = recorded if isinstance(recorded, tuple) else (recorded, "this-attempt")
    monkeypatch.setattr(shared_metrics_events, "finish_compression_attempt", lambda *_args, **_kwargs: None)
    compressor = SimpleNamespace(
        _last_compression_telemetry={"failure_class": recorded, "attempt_id": attempt_id}, context_length=100_000,
    )
    agent = SimpleNamespace(
        context_compressor=compressor, session_id="session-telemetry-test", _compression_attempt_id="this-attempt",
    )

    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        _emit_aborted_attempt_telemetry(agent, time.monotonic(), caller)

    assert _extract_telemetry(caplog)["failure_class"] == expected


def test_in_place_commit_preserves_seeded_telemetry_fields(caplog):
    """#118580: the memory-flush session-end inside the commit must not wipe attempt telemetry."""
    import contextlib
    import os
    import tempfile
    from pathlib import Path

    from agent.conversation_compression import CompressionCommitFence, run_compress_context_with_progress_timeout

    with tempfile.TemporaryDirectory() as tmp:
        from hermes_state import SessionDB
        from run_agent import AIAgent

        db = SessionDB(db_path=Path(tmp) / "t.db")
        sid = "20260920_234956_0c74e527"
        db.create_session(sid, "cli", model="test/model")
        for idx in range(8):
            db.append_message(
                session_id=sid, role="user" if idx % 2 == 0 else "assistant", content=f"msg {idx}",
            )
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            agent = AIAgent(
                api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                quiet_mode=True, session_db=db, session_id=sid,
                skip_context_files=True, skip_memory=True,
            )
        agent.compression_in_place = True
        agent._compression_feasibility_checked = True
        messages = [
            {"role": "user" if idx % 2 == 0 else "assistant",
             "content": f"turn message {idx} padding " * 20}
            for idx in range(30)
        ]

        def worker(fence):
            return compress_context(
                agent, messages, "system prompt",
                approx_tokens=75_000, force=True, commit_fence=fence,
            )

        with patch.object(agent.context_compressor, "_generate_summary", return_value="SUMMARY"):
            with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
                run_compress_context_with_progress_timeout(
                    worker=worker, messages=messages, system_prompt_fallback="system prompt",
                    idle_timeout_seconds=30, total_ceiling_seconds=120, fence=CompressionCommitFence(),
                    telemetry_agent=agent, stall_fallback=False,
                )
        with contextlib.suppress(Exception):
            db.close()
    payload = _extract_telemetry(caplog)
    assert payload["commit_status"] == "committed"
    assert payload["split_status"] == "in_place_committed"
    for key in (
        "trigger_source", "main_model", "middle_window_tokens",
        "protected_head_tokens", "protected_tail_tokens",
    ):
        assert key in payload, f"seeded field lost: {key}"
    assert payload["middle_window_tokens"] is not None


def test_new_attempt_clears_previous_attempt_telemetry():
    """A new attempt must not inherit the previous attempt's telemetry (#118580 follow-up).

    The commit-time hold keeps the trio for its own attempt's emit; the next attempt
    (and any emit before it re-seeds) must start from a clean slate.
    """
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/main-model", provider="test-provider", threshold_percent=0.50, quiet_mode=True,
            config_context_length=100_000,
        )
    compressor.tail_token_budget = 10
    compressor._last_compression_telemetry = {
        "event": "compression_attempt", "attempt_id": "prior-attempt", "session_id": "prior-session", "commit_ms": 73,
    }
    compressor._active_compression_telemetry = compressor._last_compression_telemetry
    compressor._compression_telemetry_seed = {"attempt_id": "prior-attempt"}
    agent = _Agent(compressor)

    from agent.conversation_compression import _begin_compression_attempt

    _begin_compression_attempt(agent, force=True, defer_notification=False)

    assert compressor._last_compression_telemetry is None
    assert compressor._active_compression_telemetry is None
    assert compressor._compression_telemetry_seed["attempt_id"] == agent._compression_attempt_id


def test_pool_saturation_emit_does_not_inherit_previous_attempt_telemetry(caplog):
    """The pool-refusal emit fires before any attempt begins: no previous attempt's numbers ride along."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/main-model", provider="test-provider", threshold_percent=0.50, quiet_mode=True,
            config_context_length=100_000,
        )
    compressor.tail_token_budget = 10
    compressor._last_compression_telemetry = {
        "event": "compression_attempt", "attempt_id": "prior-attempt", "session_id": "prior-session",
        "commit_ms": 73, "middle_window_tokens": 2407,
    }
    compressor._active_compression_telemetry = compressor._last_compression_telemetry
    agent = _Agent(compressor)

    from agent.conversation_compression import CompressionCommitFence, run_compress_context_with_progress_timeout

    with patch("agent.conversation_compression._try_admit_compression_job", return_value=False):
        with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
            run_compress_context_with_progress_timeout(
                worker=lambda fence: ([], "system prompt"), messages=[], system_prompt_fallback="system prompt",
                idle_timeout_seconds=30, total_ceiling_seconds=120, fence=CompressionCommitFence(),
                telemetry_agent=agent, stall_fallback=False,
            )

    payload = _extract_telemetry(caplog)
    assert payload["failure_class"] == "pool_saturated"
    assert payload["session_id"] == "session-telemetry-test"
    assert payload.get("commit_ms") is None
    assert payload.get("middle_window_tokens") is None
