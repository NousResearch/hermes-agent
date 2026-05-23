from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MEMORY_PLAN = ROOT / "docs" / "HERMES_MEMORY_PLAN.md"
EXECUTION_PLAN = ROOT / "docs" / "HERMES_EXECUTION_PLAN.md"
TESTING_PLAN = ROOT / "docs" / "HERMES_TESTING_PLAN.md"
FINAL_REPORT = ROOT / "docs" / "HERMES_FINAL_REPORT.md"


def test_memory_plan_requires_two_exact_current_run_approval_gates():
    text = MEMORY_PLAN.read_text(encoding="utf-8")

    assert "APPROVE HERMES PRIVATE MEMORY COMPACTION DRAFT" in text
    assert "APPLY HERMES PRIVATE MEMORY COMPACTION" in text
    assert "Approval expires at the end of the current Codex run" in text
    assert "Prior chat history" in text
    assert "does not count" in text


def test_memory_plan_records_applied_default_scope_and_blocks_extra_mutation():
    text = MEMORY_PLAN.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    required_boundaries = [
        "approved default-file draft applied",
        "did not read, rewrite, compact, delete, backfill, or reconcile additional private memory stores",
        "It does not allow rewriting live memory",
        "The second approval is required",
        "Do not paste raw private memory contents",
        "Additional private memory stores remain out of scope and untouched",
    ]
    for boundary in required_boundaries:
        assert boundary in normalized


def test_memory_plan_limits_default_scope_and_private_artifacts():
    text = MEMORY_PLAN.read_text(encoding="utf-8")

    for allowed in [
        "~/.hermes/memories/MEMORY.md",
        "~/.hermes/memories/USER.md",
    ]:
        assert allowed in text

    for excluded in [
        "provider facts",
        "logs",
        "caches",
        "backups and snapshots",
        "screenshots",
        "audio and video",
        "documents",
        "credentials, tokens, keys, auth files, and Keychain values",
    ]:
        assert excluded in text

    assert "owner-only" in text
    assert "`0700` directories, `0600` files" in text


def test_execution_and_final_report_reference_completed_apply_scope():
    execution = EXECUTION_PLAN.read_text(encoding="utf-8")
    final = FINAL_REPORT.read_text(encoding="utf-8")
    normalized_final = " ".join(final.split())

    assert "explicit private-memory compaction approval plan" in execution
    assert "reviewed draft to only" in execution
    assert "an explicit private-memory compaction approval plan was added" in final
    assert "the approved default-file draft was applied" in final
    assert "Additional private memory stores remain out of scope" in normalized_final


def test_testing_plan_has_read_only_validation_expectations():
    text = TESTING_PLAN.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "Private memory compaction approval-plan cleanup validation" in text
    assert "does not read, rewrite, compact, delete, backfill, or reconcile" in normalized
    assert "memory audit --json --redact" in text
    assert "memory status" in text
    assert "ai.hermes.gateway" in text
    assert "/Users/agent1/Operator/scripts/hermes-gateway.sh" in text
