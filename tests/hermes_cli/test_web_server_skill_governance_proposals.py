"""Focused tests for Skill Governance proposal dashboard endpoints."""

import json
from pathlib import Path

import pytest

from tools.skill_governance_proposals import create_or_update_skill_governance_proposal


def _set_hermes_home(monkeypatch, tmp_path: Path) -> Path:
    hermes_home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


@pytest.fixture
def client(monkeypatch, tmp_path):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    _set_hermes_home(monkeypatch, tmp_path)
    from hermes_cli.web_server import app

    return TestClient(app)


def _create_dashboard_proposal():
    return create_or_update_skill_governance_proposal(
        {
            "proposal_id": "curator-20260506-dashboard",
            "source": "curator_dry_run",
            "source_run_id": "20260506-071737",
            "action": "consolidate",
            "title": "Consolidate dashboard/HUDUI skills",
            "pm_summary": "Create one dashboard umbrella skill.",
            "impact_summary": "Reduces duplicated dashboard recipes while preserving details as references.",
            "target_skills": ["hermes-dashboard-cron-operations", "hermes-hudui-frontend-workflow"],
            "risk_level": "medium",
            "evidence": ["Curator dry-run identified overlap."],
        }
    )


def test_get_governance_proposals_returns_empty_list_when_store_missing(client):
    response = client.get("/api/skills/governance/proposals")

    assert response.status_code == 200
    assert response.json() == []


def test_get_governance_proposals_filters_by_decision_status_and_limit(client):
    first = _create_dashboard_proposal()
    second = create_or_update_skill_governance_proposal(
        {
            "proposal_id": "curator-20260506-training",
            "source": "curator_dry_run",
            "source_run_id": "20260506-071737",
            "action": "consolidate",
            "title": "Consolidate LLM training skills",
            "pm_summary": "Create one training umbrella skill.",
            "target_skills": ["grpo-rl-training", "peft-fine-tuning"],
            "risk_level": "medium",
            "decision_status": "bad_test_target",
            "decision_by": "hermes-policy",
            "decision_note": "Not a good first validation target for this user.",
        }
    )

    response = client.get(
        "/api/skills/governance/proposals",
        params={"decision_status": "pending", "limit": 1},
    )

    assert response.status_code == 200
    data = response.json()
    assert [proposal["proposal_id"] for proposal in data] == [first["proposal_id"]]
    assert second["proposal_id"] not in {proposal["proposal_id"] for proposal in data}


def test_get_governance_proposal_detail_and_record_decision(client):
    proposal = _create_dashboard_proposal()

    detail = client.get(f"/api/skills/governance/proposals/{proposal['proposal_id']}")
    assert detail.status_code == 200
    assert detail.json()["title"] == "Consolidate dashboard/HUDUI skills"
    assert "approved" in detail.json()["allowed_decision_statuses"]

    decided = client.post(
        f"/api/skills/governance/proposals/{proposal['proposal_id']}/decision",
        json={"status": "approved", "note": "Good first Curator target.", "decided_by": "test-user"},
    )
    assert decided.status_code == 200
    data = decided.json()
    assert data["decision_status"] == "approved"
    assert data["decision_note"] == "Good first Curator target."
    assert data["decision_by"] == "test-user"

    refreshed = client.get(f"/api/skills/governance/proposals/{proposal['proposal_id']}").json()
    assert refreshed["decision_status"] == "approved"


def test_governance_proposal_detail_exposes_artifacts_but_list_omits_text(client):
    proposal = create_or_update_skill_governance_proposal(
        {
            "proposal_id": "curator-20260506-artifact",
            "source": "curator_dry_run",
            "source_run_id": "20260506-071737",
            "action": "consolidate",
            "title": "Show proposal artifact",
            "pm_summary": "Expose a dry-run excerpt without returning large text in the list endpoint.",
            "target_skills": ["hermes-dashboard-feature-fixture-layer"],
            "risk_level": "medium",
            "artifact_texts": {"curator_report_excerpt.md": "dry-run excerpt for PM review"},
        }
    )

    listed = client.get("/api/skills/governance/proposals").json()[0]
    detail = client.get(f"/api/skills/governance/proposals/{proposal['proposal_id']}")

    assert "artifact_texts" not in listed
    assert detail.status_code == 200
    assert detail.json()["artifact_texts"] == {"curator_report_excerpt.md": "dry-run excerpt for PM review"}
    assert detail.json()["diff_text"] is None


def test_bad_test_target_transition_returns_specific_error_and_allowed_statuses(client):
    proposal = create_or_update_skill_governance_proposal(
        {
            "proposal_id": "curator-20260506-bad-target",
            "source": "curator_dry_run",
            "source_run_id": "20260506-071737",
            "action": "consolidate",
            "title": "Bad target",
            "pm_summary": "Already marked as a poor validation target.",
            "target_skills": ["grpo-rl-training"],
            "risk_level": "medium",
            "decision_status": "bad_test_target",
            "decision_by": "hermes-policy",
        }
    )

    detail = client.get(f"/api/skills/governance/proposals/{proposal['proposal_id']}")
    assert "approved" not in detail.json()["allowed_decision_statuses"]

    decided = client.post(
        f"/api/skills/governance/proposals/{proposal['proposal_id']}/decision",
        json={"status": "approved"},
    )
    assert decided.status_code == 400
    assert "bad_test_target" in decided.json()["detail"]
    assert "approved" in decided.json()["detail"]


def test_governance_proposal_endpoints_return_404_and_400(client):
    assert client.get("/api/skills/governance/proposals/missing").status_code == 404

    proposal = _create_dashboard_proposal()
    bad = client.post(
        f"/api/skills/governance/proposals/{proposal['proposal_id']}/decision",
        json={"status": "applied"},
    )
    assert bad.status_code == 400


def test_governance_proposal_endpoints_reject_path_shaped_ids(client):
    _create_dashboard_proposal()

    for proposal_id in ("..%2Fx", "a%2Fb", "space%20id"):
        assert client.get(f"/api/skills/governance/proposals/{proposal_id}").status_code == 404
        assert client.post(
            f"/api/skills/governance/proposals/{proposal_id}/decision",
            json={"status": "approved"},
        ).status_code == 404
