"""Report-only golden transcript retention evaluation tests."""

from __future__ import annotations

from agent.context_retention_eval import (
    CriticalFact,
    ContextGoldenTranscript,
    evaluate_context_retention,
)


def test_golden_fact_recall_survives_compacted_text_and_resolvable_refs():
    golden = ContextGoldenTranscript(
        name="tool-heavy-debugging",
        raw_context=(
            "Verbose pytest output... " * 40
            + "User fixed bug after pytest showed ERR42 in /tmp/app/tests/test_api.py"
        ),
        compacted_context=(
            "## Critical facts\n"
            "- ERR42 is the failing error code.\n"
            "- See ref:file:test_api for the exact file path."
        ),
        required_facts=[
            CriticalFact(id="error-code", text="ERR42"),
            CriticalFact(id="test-file", text="/tmp/app/tests/test_api.py", source_ref="ref:file:test_api"),
        ],
        refs={"ref:file:test_api": "/tmp/app/tests/test_api.py"},
    )

    result = evaluate_context_retention(golden)

    assert result.critical_fact_recall == 1.0
    assert result.ref_recoverability == 1.0
    assert result.missing_fact_ids == []
    assert result.unresolved_ref_ids == []
    assert result.compressed_tokens < result.raw_tokens
    assert result.passed is True


def test_missing_required_fact_fails_closed_with_actionable_ids():
    golden = ContextGoldenTranscript(
        name="lost-in-middle-regression",
        raw_context="The deployment target is staging and rollback command is scripts/rollback.sh",
        compacted_context="The deployment target is staging.",
        required_facts=[
            CriticalFact(id="target", text="staging"),
            CriticalFact(id="rollback-command", text="scripts/rollback.sh"),
        ],
        refs={},
    )

    result = evaluate_context_retention(golden)

    assert result.critical_fact_recall == 0.5
    assert result.missing_fact_ids == ["rollback-command"]
    assert result.passed is False


def test_unresolvable_required_ref_fails_even_when_fact_text_survives():
    golden = ContextGoldenTranscript(
        name="ref-recoverability-regression",
        raw_context="Full terminal output stored under ref:tool:abc123",
        compacted_context="Full terminal output stored under ref:tool:abc123",
        required_facts=[
            CriticalFact(id="tool-ref", text="ref:tool:abc123", source_ref="ref:tool:abc123"),
        ],
        refs={},
    )

    result = evaluate_context_retention(golden)

    assert result.critical_fact_recall == 1.0
    assert result.ref_recoverability == 0.0
    assert result.unresolved_ref_ids == ["ref:tool:abc123"]
    assert result.passed is False
