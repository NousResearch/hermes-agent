"""Every compaction names the arm that fired it.

``trigger`` flows from each call site through ``compress_context`` into three places: the
``context compression started`` log line (``trigger=<label>``), the user-visible status (a
``" (reason)"`` suffix on ``COMPACTION_STATUS``) and attempt telemetry (the coarse
``manual``/``auto``/``overflow`` class). Arms driven end to end through ``run_conversation``,
``compress_now`` and gateway hygiene are pinned in their own suites; this file drives the real
``compress_context`` and the turn-start arms that have no other recorder.
"""

import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import (
    _TRIGGER_REASON_CLAUSES,
    _TRIGGER_TELEMETRY_CLASS,
    COMPACTION_STATUS,
    MANUAL_TRIGGER_REASON,
    compaction_reason_clause,
    compaction_telemetry_trigger,
    compress_context,
)
from agent.turn_context_compaction import (
    CompactionOutcome,
    _engine_preflight_maintenance,
    _idle_compaction,
    _run_preflight_passes,
)
from tests.agent.test_compression_attempt_telemetry import _Agent, _messages


class _Called(BaseException):
    """Raised by the recording ``_compress_context`` once the arm reached it (BaseException so
    arms that swallow ``Exception`` around compression cannot hide the call)."""


class _Recorder:
    def __init__(self, **attrs):
        self.calls = []
        self.statuses = []
        for key, value in attrs.items():
            setattr(self, key, value)

    def _compress_context(self, messages, system_message, **kwargs):
        self.calls.append(kwargs)
        raise _Called()

    def _emit_status(self, text):
        self.statuses.append(text)


# ---------------------------------------------------------------------------
# Vocabulary contract
# ---------------------------------------------------------------------------

def test_every_fine_label_has_a_clause_and_a_stable_telemetry_class():
    for label, coarse in _TRIGGER_TELEMETRY_CLASS.items():
        assert coarse in {"manual", "auto", "overflow"}
        assert compaction_telemetry_trigger(label, force=False) == coarse
        clause = compaction_reason_clause(label)
        assert clause == f" ({_TRIGGER_REASON_CLAUSES[label]})"
    assert "manual" in compaction_reason_clause(MANUAL_TRIGGER_REASON)


def test_missing_and_unknown_labels_degrade_readably():
    assert compaction_reason_clause(None) == ""
    assert compaction_reason_clause("  ") == ""
    assert compaction_reason_clause("brand_new_arm") == " (trigger: brand_new_arm)"
    # No label keeps the pre-attribution telemetry default; an unknown label passes through.
    assert compaction_telemetry_trigger(None, force=True) == "manual"
    assert compaction_telemetry_trigger(None, force=False) == "auto"
    assert compaction_telemetry_trigger("brand_new_arm", force=False) == "brand_new_arm"


# ---------------------------------------------------------------------------
# The real compress_context path
# ---------------------------------------------------------------------------

def _compress(caplog, *, force=False, **kwargs):
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="test/main-model", provider="test-provider", threshold_percent=0.50,
            quiet_mode=True, config_context_length=100_000,
        )
    compressor.tail_token_budget = 10
    agent = _Agent(compressor)
    statuses = []
    agent._emit_status = statuses.append
    with patch.object(compressor, "_generate_summary", return_value="SUMMARY"):
        with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
            compressed, _ = compress_context(
                agent, _messages(), "system prompt", approx_tokens=75_000, force=force, **kwargs
            )
    assert compressed is not None
    telemetry = [
        json.loads(r.getMessage().split("context compression attempt telemetry: ", 1)[1])
        for r in caplog.records if "context compression attempt telemetry:" in r.getMessage()
    ]
    assert len(telemetry) == 1
    started = [s for s in statuses if s.startswith(COMPACTION_STATUS)]
    assert len(started) == 1, statuses
    return started[0], caplog.text, telemetry[0]


@pytest.mark.parametrize(
    ("label", "force", "coarse"),
    [
        ("threshold", False, "auto"),
        ("overflow_413", False, "overflow"),
        ("tier_reduction", False, "overflow"),
        (MANUAL_TRIGGER_REASON, True, "manual"),
    ],
)
def test_compaction_names_its_trigger_in_log_status_and_telemetry(caplog, label, force, coarse):
    status, log_text, telemetry = _compress(caplog, force=force, trigger=label)

    assert f"trigger={label} " in log_text
    assert "UNATTRIBUTED" not in log_text
    # Suffix, never an infix: the gateway noise filter and progress matcher key on the leading wording.
    assert status == COMPACTION_STATUS + compaction_reason_clause(label)
    assert telemetry["trigger_source"] == coarse


def test_unattributed_caller_is_logged_loudly_and_renders_no_clause(caplog):
    status, log_text, telemetry = _compress(caplog)

    assert "trigger=UNATTRIBUTED" in log_text
    assert "every compaction must name its arm" in log_text
    assert status == COMPACTION_STATUS
    assert telemetry["trigger_source"] == "auto"


# ---------------------------------------------------------------------------
# Turn-start arms
# ---------------------------------------------------------------------------

def _outcome(messages):
    return CompactionOutcome(
        messages=messages, active_system_prompt="sys", conversation_history=None, current_turn_user_idx=0,
    )


def test_engine_preflight_maintenance_arm_passes_its_label():
    compressor = SimpleNamespace(should_compress_preflight=lambda _m: True, threshold_tokens=1_000)
    agent = _Recorder()
    with pytest.raises(_Called):
        _engine_preflight_maintenance(agent, _outcome([{"role": "user", "content": "x"}]), compressor, 500, "sys", "t")
    assert [c.get("trigger") for c in agent.calls] == ["engine_preflight_maintenance"]


def test_threshold_preflight_arm_passes_its_label():
    compressor = SimpleNamespace(threshold_tokens=1_000, context_length=8_000)
    agent = _Recorder(model="m", max_compression_attempts=3, _clear_context_overflow_warn=lambda: None)
    with pytest.raises(_Called):
        _run_preflight_passes(agent, _outcome([{"role": "user", "content": "x"}]), compressor, 2_000, "sys", "t")
    assert [c.get("trigger") for c in agent.calls] == ["threshold"]


def test_idle_resume_arm_passes_its_label(monkeypatch):
    from agent import turn_context as tc

    monkeypatch.setattr(tc, "_preflight_request_tokens", lambda *_a, **_k: 50_000)
    monkeypatch.setattr(tc, "_should_idle_compact", lambda **_k: True)
    compressor = SimpleNamespace(
        threshold_tokens=100_000, summary_target_ratio=0.2, awaiting_real_usage_after_compression=False,
    )
    agent = _Recorder(
        compression_enabled=True, compression_idle_compact_after_seconds=60,
        _last_activity_ts=time.time() - 3_600, context_compressor=compressor, session_id="s", model="m",
    )
    with pytest.raises(_Called):
        _idle_compaction(agent, _outcome([{"role": "user", "content": "x"}]), "sys", "hi", "t")
    assert [c.get("trigger") for c in agent.calls] == ["idle_resume"]


def test_output_cap_recovery_arm_passes_its_label(monkeypatch):
    from agent.turn_overflow import _clamp_output_cap, _Recovery
    from agent.turn_retry_state import TurnRetryState

    seen = []

    def _record(self, request_tokens, *, trigger="overflow", fail_on_timeout=False):
        seen.append(trigger)
        raise _Called()

    monkeypatch.setattr(_Recovery, "compress_scored_by_tokens", _record)
    agent = SimpleNamespace(tools=None, _buffer_vprint=lambda *_a, **_k: None)
    st = _Recovery(
        agent=agent, messages=[{"role": "user", "content": "x"}], api_messages=[{"role": "user", "content": "x"}],
        system_message="sys", active_system_prompt="sys", conversation_history=[], approx_tokens=10,
        compression_attempts=0, effective_task_id="t", api_call_count=1, max_compression_attempts=3,
    )
    with pytest.raises(_Called):
        _clamp_output_cap(st, TurnRetryState(), 1_000, 8_000)
    assert seen == ["output_cap"]
