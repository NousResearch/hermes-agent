"""Summary-boundary audit of exact identifiers (A1, OpenClaw identifier-policy pattern).

The summarizer is ORDERED to preserve identifiers verbatim and lean mode appends a
mechanical anchor index, but nothing checked the finished summary. The audit diffs
harvested region identifiers against the summary and appends a bounded verbatim
repair for genuinely dropped ones — LLM-free (the one-summary-call-per-attempt
budget, #96603, is unchanged), redaction-safe, and telemetry-visible.
"""

from __future__ import annotations

# Pin the entry-point bootstrap before pytest's per-test real-home guard installs:
# _generate_summary lazily imports agent.conversation_loop, whose import chain runs
# hermes_bootstrap's import-time interrupted-update check (a stat under the checkout's
# git dir) — refused if that first import happens inside a test on a dev box whose
# checkout lives under the real home (pattern: test_codex_happy_eyeballs.py).
import hermes_bootstrap
from unittest.mock import MagicMock, patch

from agent.compaction_summary_audit import (
    EXTRA_AUDIT_PATTERNS,
    REPAIR_HEADING,
    audit_and_repair_summary,
    find_missing_identifiers,
)
from agent.context_compressor import ContextCompressor, _ANCHOR_PATTERNS

_ALL_PATTERNS = [*_ANCHOR_PATTERNS, *EXTRA_AUDIT_PATTERNS]


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(model="test/model", quiet_mode=True)


def _response(content: str):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def test_missing_identifier_gets_bounded_verbatim_repair():
    region = "deploy agent/context_compressor_offload.py for PR #90001 uuid 3f2b7c9a-1d4e-4f60-8a2b-6c6d5e4f3a1b"
    summary = "## Historical Task Snapshot\nNone\n\nDeployed the offload module and its PR."
    repaired, telemetry = audit_and_repair_summary(summary, region, _ALL_PATTERNS)
    assert REPAIR_HEADING in repaired
    assert "agent/context_compressor_offload.py" in repaired.split(REPAIR_HEADING)[1]
    assert "3f2b7c9a-1d4e-4f60-8a2b-6c6d5e4f3a1b" in repaired.split(REPAIR_HEADING)[1]
    assert telemetry["audit_repaired"] >= 2


def test_present_identifiers_trigger_no_repair():
    region = "fix in agent/widget.py referenced by #90002"
    summary = f"Fixed agent/widget.py (#90002). {REPAIR_HEADING}\nunchanged"
    repaired, telemetry = audit_and_repair_summary(summary, region, _ALL_PATTERNS)
    assert repaired is summary
    assert telemetry["audit_identifiers_missing"] == 0
    assert telemetry["audit_repaired"] == 0


def test_redaction_boundary_is_never_repaired_back():
    # A key-shaped identifier is rewritten by the compaction redactor: its absence
    # from the summary is policy, not loss, so the audit must not append it.
    region = "use key sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH now"
    missing = find_missing_identifiers(region, "no key here at all", _ALL_PATTERNS)
    assert not any(value.startswith("sk-proj-") for value in missing)


def test_generate_summary_appends_repair_for_dropped_identifiers():
    c = _compressor()
    # Identifiers in TOOL output, not the last user ask: lean mode's anchor index
    # keeps files/PRs, but a raw uuid is in neither the anchor categories nor the
    # grounded task snapshot — exactly what the audit exists to catch.
    uuid = "3f2b7c9a-1d4e-4f60-8a2b-6c6d5e4f3a1b"
    turns = [
        {"role": "user", "content": "review the deploy"},
        {"role": "assistant", "content": "building", "tool_calls": [
            {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": f"release id {uuid} pushed ok"},
    ]
    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("## Historical Task Snapshot\nreview the deploy\n\nReviewed the release build."),
    ):
        summary = c._generate_summary(turns)
    assert summary is not None
    assert REPAIR_HEADING in summary
    assert uuid in summary.split(REPAIR_HEADING)[1]


def test_legacy_mode_still_gets_the_audit():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(model="test/model", quiet_mode=True, tail_mode="legacy")
    with patch(
        "agent.context_compressor.call_llm",
        return_value=_response("## Historical Task Snapshot\ncheck it\n\nChecked things."),
    ):
        summary = c._generate_summary([
            {"role": "user", "content": "check it"},
            {"role": "assistant", "content": "shipped agent/dropped_module.py; runner ci.internal.net:8443 unhealthy"},
        ])
    assert summary is not None
    tail = summary.split(REPAIR_HEADING)[1]
    assert "agent/dropped_module.py" in tail
    assert "ci.internal.net:8443" in tail
