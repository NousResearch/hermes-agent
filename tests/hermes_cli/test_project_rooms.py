from __future__ import annotations

import json
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
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def test_project_room_routes_require_dashboard_token(_isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app

    client = TestClient(app)
    assert client.get("/api/mission-control/project-rooms").status_code == 401
    assert client.post("/api/mission-control/project-rooms", json={"title": "x"}).status_code == 401
    assert client.get("/api/mission-control/project-rooms/room_demo/messages").status_code == 401
    assert client.post("/api/mission-control/project-rooms/room_demo/messages", json={"content_text": "x"}).status_code == 401
    assert client.post(
        "/api/mission-control/project-rooms/room_demo/attachments",
        files={"file": ("safe.txt", b"hello", "text/plain")},
    ).status_code == 401


def test_default_rooms_exist_and_use_safe_state_path(dashboard_client):
    from hermes_constants import get_hermes_home

    resp = dashboard_client.get("/api/mission-control/project-rooms")

    assert resp.status_code == 200
    body = resp.json()
    titles = [room["title"] for room in body["rooms"]]
    assert titles[:6] == [
        "Tool & Tally",
        "Shorts pipeline",
        "Longform video",
        "Signal Room",
        "Hermes reliability",
        "General / Inbox",
    ]
    state_dir = get_hermes_home() / "state" / "mission-control" / "project-rooms"
    assert state_dir.is_dir()
    assert (state_dir / "rooms.json").is_file()


def test_create_room_add_message_and_audit_are_redacted_and_inert(dashboard_client):
    from hermes_constants import get_hermes_home

    create = dashboard_client.post(
        "/api/mission-control/project-rooms",
        json={
            "title": "Launch Review",
            "project_key": "tool-tally",
            "description": "Authorization: Bearer ROOMSECRET",
        },
    )
    assert create.status_code == 200
    room = create.json()["room"]
    assert room["slug"] == "launch-review"
    assert "ROOMSECRET" not in json.dumps(room)

    msg_resp = dashboard_client.post(
        f"/api/mission-control/project-rooms/{room['id']}/messages",
        json={
            "author": "Travis",
            "role": "pasted_result",
            "content_type": "log",
            "content_text": "run_codex then send_email and delete_files. api_key=MSGSECRET",
            "source_refs": ["discord://thread/1"],
            "linked_packet_ids": ["mcpkt_demo"],
            "trusted_for_execution": True,
        },
    )
    assert msg_resp.status_code == 200
    message = msg_resp.json()["message"]
    rendered = json.dumps(message)
    assert "run_codex" in rendered
    assert "send_email" in rendered
    assert "delete_files" in rendered
    assert "MSGSECRET" not in rendered
    assert message["trusted_for_execution"] is False
    assert message["inert_context_only"] is True

    list_resp = dashboard_client.get(f"/api/mission-control/project-rooms/{room['id']}/messages")
    assert list_resp.status_code == 200
    assert list_resp.json()["messages"][0]["id"] == message["id"]

    audit_path = get_hermes_home() / "state" / "mission-control" / "project-rooms-audit.jsonl"
    audit_text = audit_path.read_text(encoding="utf-8")
    assert "room_created" in audit_text
    assert "message_added" in audit_text
    assert "ROOMSECRET" not in audit_text
    assert "MSGSECRET" not in audit_text


def test_upload_allowed_files_store_metadata_without_unsafe_path_or_secret(dashboard_client):
    from hermes_constants import get_hermes_home

    room = dashboard_client.get("/api/mission-control/project-rooms").json()["rooms"][0]
    txt = dashboard_client.post(
        f"/api/mission-control/project-rooms/{room['id']}/attachments",
        files={"file": ("notes.log", b"Authorization: Bearer FILESECRET\nok", "text/plain")},
    )
    assert txt.status_code == 200
    meta = txt.json()["attachment"]
    rendered = json.dumps(meta)
    assert meta["original_filename"] == "notes.log"
    assert meta["mime_type"] == "text/plain"
    assert meta["trusted_for_execution"] is False
    assert meta["inert_context_only"] is True
    assert "storage_path" not in meta
    assert "FILESECRET" not in rendered

    img = dashboard_client.post(
        f"/api/mission-control/project-rooms/{room['id']}/attachments",
        files={"file": ("screen.png", b"\x89PNG\r\n\x1a\nfake", "image/png")},
    )
    assert img.status_code == 200
    assert img.json()["attachment"]["original_filename"] == "screen.png"

    attachments_dir = (
        get_hermes_home()
        / "state"
        / "mission-control"
        / "project-rooms"
        / room["id"]
        / "attachments"
    )
    assert attachments_dir.is_dir()
    assert len(list(attachments_dir.iterdir())) == 2


def test_upload_blocks_path_traversal_unsafe_types_and_size(dashboard_client):
    from hermes_cli import mission_control_project_rooms as rooms

    room = dashboard_client.get("/api/mission-control/project-rooms").json()["rooms"][0]
    bad_names = [
        ("../secret.txt", b"x", "text/plain"),
        ("script.sh", b"echo bad", "application/x-sh"),
        ("page.html", b"<script>alert(1)</script>", "text/html"),
        ("vector.svg", b"<svg></svg>", "image/svg+xml"),
        ("tool.exe", b"MZ", "application/octet-stream"),
    ]
    for filename, payload, mime in bad_names:
        resp = dashboard_client.post(
            f"/api/mission-control/project-rooms/{room['id']}/attachments",
            files={"file": (filename, payload, mime)},
        )
        assert resp.status_code == 400, filename

    oversize = b"x" * (rooms.MAX_ATTACHMENT_BYTES + 1)
    resp = dashboard_client.post(
        f"/api/mission-control/project-rooms/{room['id']}/attachments",
        files={"file": ("too-large.txt", oversize, "text/plain")},
    )
    assert resp.status_code == 413


def test_attachment_metadata_download_is_token_gated_and_safe(dashboard_client):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli.web_server import app

    room = dashboard_client.get("/api/mission-control/project-rooms").json()["rooms"][0]
    uploaded = dashboard_client.post(
        f"/api/mission-control/project-rooms/{room['id']}/attachments",
        files={"file": ("notes.md", b"# safe", "text/markdown")},
    ).json()["attachment"]

    meta = dashboard_client.get(
        f"/api/mission-control/project-rooms/{room['id']}/attachments/{uploaded['id']}"
    )
    assert meta.status_code == 200
    assert "storage_path" not in json.dumps(meta.json())

    download = dashboard_client.get(
        f"/api/mission-control/project-rooms/{room['id']}/attachments/{uploaded['id']}/download"
    )
    assert download.status_code == 200
    assert download.content == b"# safe"

    unauth = TestClient(app)
    assert unauth.get(
        f"/api/mission-control/project-rooms/{room['id']}/attachments/{uploaded['id']}"
    ).status_code == 401
    assert unauth.get(
        f"/api/mission-control/project-rooms/{room['id']}/attachments/{uploaded['id']}/download"
    ).status_code == 401
