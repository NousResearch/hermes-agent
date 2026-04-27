"""Focused tests for skill change ledger dashboard endpoints."""

from pathlib import Path

import pytest

from tools.skill_change_ledger import record_skill_change


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


def test_get_skill_changes_returns_empty_list_when_ledger_missing(client):
    response = client.get("/api/skills/changes")

    assert response.status_code == 200
    assert response.json() == []


def test_get_skill_changes_filters_by_skill_limit_and_unreviewed(client):
    first = record_skill_change(
        skill="github-code-review",
        category="github",
        action="patch",
        actor="hermes-agent",
        source="skill_manage",
        reason="Add GitHub review verification step.",
    )
    second = record_skill_change(
        skill="obsidian",
        category="note-taking",
        action="edit",
        actor="hermes-agent",
        source="skill_manage",
        reason="Clarify vault source note workflow.",
    )
    third = record_skill_change(
        skill="github-code-review",
        category="github",
        action="edit",
        actor="hermes-agent",
        source="skill_manage",
        reason="Add API-rate-limit pitfall.",
    )

    reviewed = client.post(
        f"/api/skills/changes/{second['event_id']}/review",
        json={"status": "reviewed", "note": "Looks fine."},
    )
    assert reviewed.status_code == 200

    response = client.get(
        "/api/skills/changes",
        params={"skill": "github-code-review", "limit": 1, "unreviewed": "true"},
    )

    assert response.status_code == 200
    data = response.json()
    assert [event["event_id"] for event in data] == [third["event_id"]]
    assert data[0]["skill"] == "github-code-review"
    assert data[0]["review_status"] == "unreviewed"
    assert first["event_id"] != third["event_id"]

def test_get_skill_changes_caps_excessive_limit(client):
    for idx in range(105):
        record_skill_change(
            skill=f"skill-{idx}",
            action="patch",
            actor="hermes-agent",
            source="unit-test",
            reason="Populate enough events to test endpoint limit clamping.",
        )

    response = client.get("/api/skills/changes", params={"limit": 500})

    assert response.status_code == 200
    assert len(response.json()) == 100


def test_get_skill_history_returns_events_for_one_skill(client):
    other = record_skill_change(
        skill="obsidian",
        action="patch",
        actor="hermes-agent",
        source="unit-test",
        reason="Other skill.",
    )
    target = record_skill_change(
        skill="github-pr-workflow",
        category="github",
        action="patch",
        actor="hermes-agent",
        source="unit-test",
        reason="Add PR checklist.",
    )

    response = client.get("/api/skills/github-pr-workflow/history")

    assert response.status_code == 200
    data = response.json()
    assert [event["event_id"] for event in data] == [target["event_id"]]
    assert other["event_id"] not in {event["event_id"] for event in data}


def test_get_skill_change_detail_includes_diff_text(client):
    event = record_skill_change(
        skill="github-code-review",
        category="github",
        action="patch",
        actor="hermes-agent",
        source="unit-test",
        reason="Add exact diff example.",
        before_text={"SKILL.md": "old\n"},
        after_text={"SKILL.md": "new\n"},
    )

    response = client.get(f"/api/skills/changes/{event['event_id']}")

    assert response.status_code == 200
    data = response.json()
    assert data["event_id"] == event["event_id"]
    assert "-old" in data["diff_text"]
    assert "+new" in data["diff_text"]


def test_get_skill_change_detail_returns_404_for_missing_event(client):
    response = client.get("/api/skills/changes/not-found")

    assert response.status_code == 404


def test_review_skill_change_updates_review_state(client):
    event = record_skill_change(
        skill="github-code-review",
        category="github",
        action="patch",
        actor="hermes-agent",
        source="unit-test",
        reason="Needs review.",
    )

    response = client.post(
        f"/api/skills/changes/{event['event_id']}/review",
        json={"status": "needs_followup", "note": "Check against upstream docs."},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["event_id"] == event["event_id"]
    assert data["review_status"] == "needs_followup"
    assert data["review_note"] == "Check against upstream docs."
    assert data["reviewed_at"] is not None

    detail = client.get(f"/api/skills/changes/{event['event_id']}").json()
    assert detail["review_status"] == "needs_followup"


def test_review_skill_change_rejects_invalid_status(client):
    event = record_skill_change(
        skill="github-code-review",
        action="patch",
        actor="hermes-agent",
        source="unit-test",
        reason="Needs review.",
    )

    response = client.post(
        f"/api/skills/changes/{event['event_id']}/review",
        json={"status": "accepted"},
    )

    assert response.status_code == 400
