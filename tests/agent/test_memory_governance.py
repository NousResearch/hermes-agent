import json

from hermes_constants import get_hermes_home
from agent.memory_governance import (
    GovernanceLabel,
    classify_memory_candidate,
    enqueue_memory_governance_review,
    load_memory_governance_review_queue,
    memory_governance_queue_path,
)


def test_transient_project_note_is_not_user_memory():
    decision = classify_memory_candidate(
        "Current task progress: pending-interaction handoff tests passed; next step is docs.",
        source_type="session_progress",
    )

    assert decision.label == GovernanceLabel.SESSION_ONLY
    assert decision.requires_approval is False
    assert "not persistent user memory" in decision.reason


def test_boot_critical_user_preference_goes_to_hermes_memory():
    decision = classify_memory_candidate(
        "User prefers concise final answers and no cheerleading.",
        source_type="user_message",
    )

    assert decision.label == GovernanceLabel.HERMES_MEMORY
    assert decision.confidence >= 0.8
    assert decision.requires_approval is True


def test_profile_role_fact_goes_to_hermes_memory():
    decision = classify_memory_candidate(
        "Umbbi is the maintainer of the Hermes profile and Obsidian is canonical curated knowledge.",
        source_type="profile_fact",
    )

    assert decision.label == GovernanceLabel.HERMES_MEMORY
    assert "boot-critical" in decision.reason


def test_money_flow_radar_state_goes_to_project_state():
    decision = classify_memory_candidate(
        "Money Flow Radar state: last checked block 19388201, open alert count 4.",
        source_type="project_update",
    )

    assert decision.label == GovernanceLabel.PROJECT_STATE
    assert decision.requires_approval is False
    assert "not USER memory" in decision.reason


def test_cron_followup_reply_stays_session_only():
    decision = classify_memory_candidate(
        "Yes, run the radar again tomorrow morning.",
        source_type="cron_reply",
    )

    assert decision.label == GovernanceLabel.SESSION_ONLY
    assert "cron" in decision.reason.lower()


def test_reusable_procedure_goes_to_skill():
    decision = classify_memory_candidate(
        "Checklist: before changing memory storage, classify the candidate, queue review, then run targeted tests.",
        source_type="assistant_note",
    )

    assert decision.label == GovernanceLabel.SKILL
    assert decision.suggested_artifact_path == "skills/memory-governance/SKILL.md"


def test_canonical_project_decision_goes_to_obsidian_promote():
    decision = classify_memory_candidate(
        "ADR: Obsidian remains the canonical curated knowledge store; Honcho is runtime recall only. Evidence: docs/pending-interaction-handoff.md.",
        source_type="research_synthesis",
    )

    assert decision.label == GovernanceLabel.OBSIDIAN_PROMOTE
    assert decision.requires_approval is True
    assert decision.suggested_artifact_path == "Obsidian/Hermes/Memory Governance.md"


def test_secret_looking_text_is_rejected_and_redacted():
    decision = classify_memory_candidate(
        "OPENAI_API_KEY=sk-testsecretvalue1234567890",
        source_type="user_message",
    )

    assert decision.label == GovernanceLabel.REJECT
    assert decision.confidence == 1.0
    assert "sk-testsecret" not in decision.candidate_summary
    assert "[REDACTED]" in decision.candidate_summary


def test_destructive_compaction_proposal_requires_approval():
    decision = classify_memory_candidate(
        "Remove all old USER.md entries about prior projects during memory compaction.",
        source_type="memory_compaction",
    )

    assert decision.label == GovernanceLabel.REJECT
    assert decision.destructive is True
    assert decision.requires_approval is True


def test_review_queue_is_profile_local_and_stores_decision_only(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-profile"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    decision = classify_memory_candidate(
        "Money Flow Radar state: cursor updated to run-42.",
        source_type="project_update",
    )
    item = enqueue_memory_governance_review(decision)

    queue_path = memory_governance_queue_path()
    assert queue_path == get_hermes_home() / "memory_governance" / "review_queue.json"
    assert queue_path.exists()
    assert not (get_hermes_home() / "memories" / "USER.md").exists()
    assert not (get_hermes_home() / "memories" / "MEMORY.md").exists()

    raw = json.loads(queue_path.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["items"][0]["id"] == item["id"]
    assert raw["items"][0]["decision"]["label"] == "PROJECT_STATE"

    loaded = load_memory_governance_review_queue()
    assert loaded[0]["status"] == "pending_review"
    assert loaded[0]["decision"]["candidate_summary"] == decision.candidate_summary
