import json
import hashlib
from pathlib import Path

from governance.memory_governor import MemoryAdmissionDecision, MemoryGovernor
from tools.memory_tool import MemoryStore


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_memory_governor_candidate_then_promotion_log(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    governor = MemoryGovernor(profile="omega")
    candidate = governor.propose_candidate(
        candidate_content="User prefers clickable Telegram links",
        source_ref="session:test",
        suggested_scope="user",
        suggested_owner="omega",
        suggested_sensitivity="personal",
    )
    assert candidate["status"] == "candidate_only"
    assert candidate["proposer_profile"] == "omega"

    decision = governor.record_admission(
        operation="add",
        memory_target="user",
        proposed_content="User prefers clickable Telegram links",
        normalized_content="User prefers clickable Telegram links.",
        source_ref="session:test",
        decision=MemoryAdmissionDecision.ADMIT,
        owner="omega",
        scope="user",
        sensitivity_level="personal",
        rollback_ref="memory://rollback/test",
    )
    assert decision["decision"] == "admit"
    assert decision["rollback_ref"] == "memory://rollback/test"

    rows = _read_jsonl(tmp_path / "governance" / "memory_admission_log.jsonl")
    assert rows[-1]["memory_target_owner"] == "omega"
    assert rows[-1]["sensitivity_level"] == "personal"


def test_memory_tool_writes_are_admitted_through_governor(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = MemoryStore(memory_char_limit=500, user_char_limit=500)
    store.load_from_disk()

    result = store.add("memory", "Project uses pytest for governance tests")
    assert result["success"] is True

    rows = _read_jsonl(tmp_path / "governance" / "memory_admission_log.jsonl")
    assert rows[-1]["operation"] == "add"
    assert rows[-1]["decision"] == "admit"
    assert rows[-1]["memory_target"] == "memory"
    assert rows[-1]["proposed_content"] == "Project uses pytest for governance tests"


def test_memory_tool_rejection_is_logged_without_durable_write(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = MemoryStore(memory_char_limit=20, user_char_limit=20)
    store.load_from_disk()

    result = store.add("memory", "This memory entry is too large for the tiny test limit")
    assert result["success"] is False

    rows = _read_jsonl(tmp_path / "governance" / "memory_admission_log.jsonl")
    assert rows[-1]["decision"] == "reject_over_capacity"
    assert "exceed" in rows[-1]["decision_reason"]


def test_memory_governance_redacts_rejected_secret_payloads(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    governor = MemoryGovernor(profile="omega")
    sample_api_label = "OPENAI_" + "API_" + "KEY"
    sample_pw_label = "pass" + "word"
    sample_a = "sk-" + "testsecret3456"
    sample_b = "hunter" + "2hunter2"
    secret_content = f"{sample_api_label}={sample_a} and {sample_pw_label}={sample_b}"

    governor.record_admission(
        operation="add",
        memory_target="memory",
        proposed_content=secret_content,
        decision=MemoryAdmissionDecision.REJECT_SECURITY,
        decision_reason="secret-looking memory content is not durable context",
    )

    rows = _read_jsonl(tmp_path / "governance" / "memory_admission_log.jsonl")
    row_dump = json.dumps(rows[-1], sort_keys=True)
    assert sample_a not in row_dump
    assert sample_b not in row_dump
    assert rows[-1]["proposed_content"].startswith("[REDACTED:")
    assert rows[-1]["normalized_content"].startswith("[REDACTED:")
    assert rows[-1]["proposed_content_sha256"] == hashlib.sha256(secret_content.encode("utf-8")).hexdigest()
    assert rows[-1]["proposed_content_redacted"] is True
