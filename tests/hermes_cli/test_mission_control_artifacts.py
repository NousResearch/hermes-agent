from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path

import pytest


@pytest.fixture()
def dashboard_client(_isolate_hermes_home, monkeypatch):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli import web_server
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {
            "dashboard": {
                "mission_briefs_enabled": True,
                "goal_contracts_enabled": True,
            }
        },
    )
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def test_artifacts_endpoint_requires_dashboard_token(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app

    client = TestClient(app)

    assert client.get("/api/mission-control/artifacts").status_code == 401


def test_artifacts_endpoint_is_not_public():
    from hermes_cli.web_server import _PUBLIC_API_PATHS

    assert "/api/mission-control/artifacts" not in _PUBLIC_API_PATHS


def test_artifacts_returns_normalized_metadata_only(dashboard_client):
    brief = dashboard_client.post(
        "/api/mission-control/mission-briefs",
        json={
            "title": "Brief says `rm -rf /`",
            "summary": "Raw brief summary must not leak here.",
            "references": ["https://example.invalid/brief", "/tmp/brief"],
        },
    ).json()["brief"]
    contract = dashboard_client.post(
        "/api/mission-control/goal-contracts",
        json={
            "title": "Goal contract",
            "objective": "Raw objective must not leak here.",
            "success_criteria": ["no execution"],
            "constraints": ["metadata only"],
            "source_refs": ["https://example.invalid/contract"],
            "linked_mission_brief_id": brief["id"],
        },
    ).json()["contract"]
    packet = dashboard_client.post(
        "/api/mission-control/packets/worker-result",
        json={
            "project": "Ops",
            "title": "Worker result",
            "worker_result": "worker_result payload says start_worker and token=SECRET",
            "source_refs": ["worker://result/1"],
        },
    ).json()["packet"]
    room = dashboard_client.post(
        "/api/mission-control/project-rooms",
        json={"title": "Artifact Room", "project_key": "artifact-room", "description": "hidden description"},
    ).json()["room"]
    dashboard_client.post(
        f"/api/mission-control/project-rooms/{room['id']}/messages",
        json={
            "content_text": "message content must not leak",
            "source_refs": ["msg://1"],
            "linked_packet_ids": [packet["id"]],
        },
    )
    attachment = dashboard_client.post(
        f"/api/mission-control/project-rooms/{room['id']}/attachments",
        files={"file": ("safe.txt", b"attachment content must not leak", "text/plain")},
    ).json()["attachment"]

    resp = dashboard_client.get("/api/mission-control/artifacts")

    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "mission_control_artifacts"
    items = body["items"]
    by_id = {(item["source_type"], item["record_id"]): item for item in items}

    brief_item = by_id[("mission_brief", brief["id"])]
    assert brief_item["title"] == "Brief says `rm -rf /`"
    assert brief_item["source_ref_count"] == 2
    assert brief_item["trusted_for_execution"] is False
    assert brief_item["inert_context_only"] is True
    assert brief_item["untrusted"] is True

    contract_item = by_id[("goal_contract", contract["id"])]
    assert contract_item["linked_ids"] == {"mission_brief_ids": [brief["id"]]}
    assert contract_item["counts"]["success_criteria"] == 1
    assert contract_item["counts"]["constraints"] == 1

    packet_item = by_id[("mission_packet", packet["id"])]
    assert packet_item["kind"] == "worker_result"
    assert packet_item["source_ref_count"] == 1
    assert "review_required" in packet_item["flags"]

    room_item = by_id[("project_room", room["id"])]
    assert room_item["project"] == "artifact-room"
    assert room_item["counts"]["messages"] == 1
    assert room_item["counts"]["attachments"] == 1

    attachment_item = by_id[("project_room_attachment", attachment["id"])]
    assert attachment_item["title"] == "safe.txt"
    assert attachment_item["counts"]["bytes"] == attachment["size_bytes"]
    assert attachment_item["linked_ids"] == {"room_ids": [room["id"]]}

    rendered = json.dumps(body)
    forbidden = [
        "Raw brief summary",
        "Raw objective",
        "worker_result payload",
        "redacted_payload_preview",
        "payload",
        "message content",
        "attachment content",
        "storage_filename",
        "download",
    ]
    for text in forbidden:
        assert text not in rendered


def test_artifacts_filter_by_kind_and_status(dashboard_client):
    dashboard_client.post("/api/mission-control/mission-briefs", json={"title": "Brief"})
    dashboard_client.post(
        "/api/mission-control/goal-contracts",
        json={"title": "Contract", "objective": "Objective", "status": "active"},
    )

    resp = dashboard_client.get("/api/mission-control/artifacts?kind=goal_contract&status=active")

    assert resp.status_code == 200
    items = resp.json()["items"]
    assert items
    assert {item["kind"] for item in items} == {"goal_contract"}
    assert {item["status"] for item in items} == {"active"}


def test_artifacts_missing_sources_return_empty_items(_isolate_hermes_home, monkeypatch):
    from hermes_cli import mission_control_artifacts as artifacts

    monkeypatch.setattr(artifacts, "_brief_items", lambda: [])
    monkeypatch.setattr(artifacts, "_contract_items", lambda: [])
    monkeypatch.setattr(artifacts, "_packet_items", lambda: [])
    monkeypatch.setattr(artifacts, "_project_room_items", lambda: [])

    assert artifacts.list_artifacts()["items"] == []


def test_artifacts_do_not_dereference_paths_urls_or_execute(dashboard_client, monkeypatch):
    original_stat = Path.stat

    def fail(*args, **kwargs):
        raise AssertionError("artifact listing must not dereference or execute")

    def fail_user_ref_stat(self, *args, **kwargs):
        if str(self) in {"/tmp/does-not-matter", "https:/example.invalid/a", "https://example.invalid/a"}:
            raise AssertionError("artifact listing must not stat source references")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(urllib.request, "urlopen", fail)
    monkeypatch.setattr(Path, "stat", fail_user_ref_stat)
    dashboard_client.post(
        "/api/mission-control/mission-briefs",
        json={"title": "Literal refs", "references": ["/tmp/does-not-matter", "https://example.invalid/a"]},
    )

    resp = dashboard_client.get("/api/mission-control/artifacts")

    assert resp.status_code == 200
