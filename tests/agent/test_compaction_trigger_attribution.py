"""Every compaction says why it fired.

Call sites already pass their own ``trigger`` label to ``compress_context`` (idle, turn_start_threshold,
engine_preflight, pre_api, post_tool, gateway_hygiene, overflow + overflow_reason, manual). This file pins
the two human-facing renderings of that label: the ``trigger=`` field of the ``context compression
started`` log line and the ``" (reason)"`` suffix on the compacting status.
"""

import logging
from unittest.mock import patch

import pytest

from agent.context_compressor import ContextCompressor
from agent.conversation_compression import COMPACTION_STATUS, compress_context
from agent.conversation_compression_trigger import (
    MANUAL_TRIGGER_REASON,
    _OVERFLOW_REASON_CLAUSES,
    _TRIGGER_REASON_CLAUSES,
    compaction_reason_clause,
)
from tests.agent.test_compression_attempt_telemetry import _Agent, _messages


# ---------------------------------------------------------------------------
# Vocabulary contract: the clauses speak the labels telemetry already buckets
# ---------------------------------------------------------------------------

def test_every_clause_label_is_a_known_telemetry_trigger():
    from agent.compaction_events import _TRIGGERS, _OVERFLOW_REASONS
    from hermes_cli.observability.shared_metrics_fields import _COMPRESSION_TRIGGERS

    for label in _TRIGGER_REASON_CLAUSES:
        assert _COMPRESSION_TRIGGERS.get(label) in {"auto", "manual", "overflow"}, label
        assert label in _TRIGGERS, label
    assert set(_OVERFLOW_REASON_CLAUSES) == set(_OVERFLOW_REASONS)
    assert MANUAL_TRIGGER_REASON == "manual"


def test_missing_unknown_and_overflow_reason_labels_render_readably():
    assert compaction_reason_clause(None) == ""
    assert compaction_reason_clause("  ") == ""
    assert compaction_reason_clause("auto") == ""
    assert compaction_reason_clause("brand_new_arm") == " (trigger: brand_new_arm)"
    assert compaction_reason_clause("overflow") == f" ({_TRIGGER_REASON_CLAUSES['overflow']})"
    assert compaction_reason_clause("overflow", "payload_too_large") == " (the API rejected an oversize request — 413)"
    # A classifier reason without a clause falls back to the generic overflow wording.
    assert compaction_reason_clause("overflow", "rate_limit") == f" ({_TRIGGER_REASON_CLAUSES['overflow']})"
    # overflow_reason only qualifies the overflow arm.
    assert compaction_reason_clause("pre_api", "payload_too_large") == f" ({_TRIGGER_REASON_CLAUSES['pre_api']})"


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
    started = [s for s in statuses if s.startswith(COMPACTION_STATUS)]
    assert len(started) == 1, statuses
    return started[0], caplog.text


@pytest.mark.parametrize(
    ("label", "force", "overflow_reason"),
    [
        ("turn_start_threshold", False, None),
        ("pre_api", False, None),
        ("gateway_hygiene", False, None),
        ("overflow", False, "payload_too_large"),
        (MANUAL_TRIGGER_REASON, True, None),
    ],
)
def test_compaction_names_its_trigger_in_log_and_status(caplog, label, force, overflow_reason):
    status, log_text = _compress(caplog, force=force, trigger=label, overflow_reason=overflow_reason)

    assert f"context compression started: session=session-telemetry-test trigger={label} " in log_text
    # Suffix, never an infix: the gateway noise filter and progress matcher key on the leading wording.
    assert status == COMPACTION_STATUS + compaction_reason_clause(label, overflow_reason)
    assert status != COMPACTION_STATUS


@pytest.mark.parametrize(("force", "default"), [(False, "auto"), (True, "manual")])
def test_unlabelled_caller_logs_and_renders_the_force_default(caplog, force, default):
    status, log_text = _compress(caplog, force=force)

    assert f"trigger={default} " in log_text
    # The bare automatic default adds nothing an automatic status does not already say.
    assert status == COMPACTION_STATUS + ("" if default == "auto" else compaction_reason_clause(MANUAL_TRIGGER_REASON))

