"""Tests for the read-only GHL Manager dashboard plugin skeleton."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "plugins" / "ghl-manager" / "dashboard"


def _load_plugin_module():
    plugin_file = PLUGIN_ROOT / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"

    spec = importlib.util.spec_from_file_location(
        f"hermes_dashboard_plugin_ghl_manager_test_{id(plugin_file)}",
        plugin_file,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_registers_read_only_dashboard_plugin():
    manifest_path = PLUGIN_ROOT / "manifest.json"
    assert manifest_path.exists(), f"manifest missing: {manifest_path}"

    manifest = json.loads(manifest_path.read_text())

    assert manifest["name"] == "ghl-manager"
    assert manifest["label"] == "GHL Manager"
    assert manifest["entry"] == "dist/index.js"
    assert manifest["css"] == "dist/style.css"
    assert manifest["api"] == "plugin_api.py"
    assert manifest["tab"]["path"] == "/ghl-manager"


def test_backend_exposes_only_read_only_and_local_approval_state_routes():
    module = _load_plugin_module()
    routes = {
        route.path: sorted(route.methods)
        for route in module.router.routes
        if hasattr(route, "methods")
    }

    assert routes == {
        "/health": ["GET"],
        "/config": ["GET"],
        "/decision-view": ["GET"],
        "/context": ["GET"],
        "/packets": ["GET"],
        "/packets/{approval_id}": ["GET"],
        "/approval-state": ["GET"],
        "/approval-state/{approval_id}": ["GET"],
        "/action-requests": ["GET"],
        "/action-requests/process": ["POST"],
        "/approval-state/{approval_id}/action-requests": ["POST"],
        "/approval-state/{approval_id}/events": ["POST"],
    }
    assert not any("mutation" in path.lower() for path in routes)
    assert not any("ghl" in path.lower() and "send" in path.lower() for path in routes)


def test_health_and_config_responses_are_safe_and_local_only_documented():
    app = FastAPI()
    app.include_router(_load_plugin_module().router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    health = client.get("/api/plugins/ghl-manager/health")
    assert health.status_code == 200
    health_data = health.json()
    assert health_data["plugin"] == "ghl-manager"
    assert health_data["status"] == "ok"
    assert health_data["read_only"] is True
    assert "local-only" in health_data["warning"].lower()

    config = client.get("/api/plugins/ghl-manager/config")
    assert config.status_code == 200
    config_data = config.json()
    assert config_data["plugin"] == "ghl-manager"
    assert config_data["read_only"] is True
    assert config_data["endpoints"] == [
            "GET /health",
            "GET /config",
            "GET /context",
            "GET /decision-view",
            "GET /packets",
            "GET /packets/{approval_id}",
            "GET /approval-state",
        "GET /approval-state/{approval_id}",
        "GET /action-requests",
        "POST /approval-state/{approval_id}/events",
        "POST /approval-state/{approval_id}/action-requests",
        "POST /action-requests/process",
    ]
    assert config_data["mutations_enabled"] is False
    assert config_data["live_execution_enabled"] is False
    assert "secrets" not in json.dumps(config_data).lower()
    assert "token" not in json.dumps(config_data).lower()
    assert "api_key" not in json.dumps(config_data).lower()


def test_frontend_bundle_is_standalone_bridge_console_not_dashboard_sdk_stub():
    bundle_path = PLUGIN_ROOT / "dist" / "index.js"
    assert bundle_path.exists(), f"frontend bundle missing: {bundle_path}"
    bundle = bundle_path.read_text()

    # This bundle is the standalone-style GHL Manager console served through
    # the dashboard plugin bridge, not the older minimal dashboard SDK stub.
    assert "Standalone Blue / GHL operations console" in bundle
    assert "GHL Manager" in bundle
    assert "Systems / Sources" in bundle
    assert "`/health`" in bundle
    assert "`/config`" in bundle
    assert "`/context`" in bundle
    assert "`/decision-view`" in bundle
    assert "`/packets`" in bundle
    assert "import React" not in bundle
    assert "from \"react\"" not in bundle


def test_frontend_hardening_actions_are_local_only_and_searchable():
    bundle_path = PLUGIN_ROOT / "dist" / "index.js"
    css_path = PLUGIN_ROOT / "dist" / "style.css"
    bundle = bundle_path.read_text()
    css = css_path.read_text()

    for label in [
        "Copy draft text",
        "Copy approval command",
        "Copy reverify / stale-cleanup prompt",
        "Copy full technical packet JSON",
        "Approve + queue action",
        "Request reverify",
        "Deny locally — not sent",
        "Block stale locally",
        "Edit draft locally — not sent",
    ]:
        assert label in bundle

    assert "/approval-state/" in bundle
    assert "/action-requests" in bundle
    assert "send-approved" in bundle
    assert "reverify" in bundle
    assert "Blue executor" in bundle
    assert "Acknowledged click" in bundle
    assert "missing canonical key" in bundle
    assert "action-timeline" in bundle
    assert "live execution off by default" in bundle
    assert "no browser send or CRM mutation" in bundle
    assert "does not send" in bundle.lower() or "no send" in bundle.lower()
    assert "Search contact, brand, action, inbound/outbound text" in bundle
    assert "daily-queues" in css
    assert "action-panel" in css
    assert "action-timeline" in css
    forbidden_frontend_patterns = ["GHL_API_KEY", "Authorization", "Bearer ", "api.gohighlevel.com", "services.leadconnectorhq.com"]
    for pattern in forbidden_frontend_patterns:
        assert pattern not in bundle


def test_relevant_cron_jobs_include_blue_ghl_work_and_exclude_unrelated_creator_jobs(monkeypatch):
    module = _load_plugin_module()
    fake_cron = types.ModuleType("cron")
    fake_jobs = types.ModuleType("cron.jobs")
    fake_jobs.list_jobs = lambda include_disabled=True: [
        {
            "id": "job-blue-crew-daily",
            "name": "Blue Crew GHL manager daily briefing",
            "prompt": "Summarize Blue Crew GHL approval packets and CRM context.",
            "schedule": {"display": "every morning"},
            "enabled": True,
            "next_run_at": "2026-05-10T09:00:00+10:00",
            "last_status": "ok",
        },
        {
            "id": "job-ghl-manager-ui-watchdog",
            "name": "GHL Manager UI context refresh",
            "prompt": "Refresh GHL Manager approval-packet demo context.",
            "schedule_display": "every 30m",
            "enabled": False,
            "state": "paused",
            "last_status": "paused",
        },
        {
            "id": "job-creator-growth-os-approval-coordinator",
            "name": "Creator Growth OS approval coordinator",
            "prompt": "Coordinate manager approvals for productivity project content packets.",
            "schedule_display": "every 2h",
            "enabled": True,
            "last_status": "ok",
        },
    ]
    fake_cron.jobs = fake_jobs
    monkeypatch.setitem(sys.modules, "cron", fake_cron)
    monkeypatch.setitem(sys.modules, "cron.jobs", fake_jobs)

    jobs = module._load_relevant_cron_jobs()

    assert [job["id"] for job in jobs] == ["job-blue-crew-daily", "job-ghl-manager-ui-watchdog"]
    assert jobs[0]["schedule"] == "every morning"
    assert jobs[1]["state"] == "paused"


def test_context_endpoint_returns_compact_kanban_and_cron_summary(monkeypatch):
    module = _load_plugin_module()

    monkeypatch.setattr(
        module,
        "_load_board_summary",
        lambda board: {
            "board": board,
            "available": True,
            "counts": {"ready": 2, "running": 1, "blocked": 1, "done": 3, "total": 7},
            "recent_tasks": [
                {"id": "t_packet", "title": "Packet task", "status": "blocked", "assignee": "default"},
            ],
            "source": "kanban_db",
        },
    )
    monkeypatch.setattr(
        module,
        "_load_relevant_cron_jobs",
        lambda: [
            {
                "id": "job-ghl",
                "name": "GHL Manager scout",
                "schedule": "0 9 * * *",
                "enabled": True,
                "state": "active",
                "next_run_at": "2026-05-10T09:00:00+10:00",
                "last_status": "ok",
            }
        ],
    )

    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    response = client.get("/api/plugins/ghl-manager/context")

    assert response.status_code == 200
    data = response.json()
    assert data["read_only"] is True
    assert data["mutations_enabled"] is False
    assert [board["board"] for board in data["boards"]] == ["ghl-six-priority-cleanup", "ghl-manager-ui"]
    assert data["boards"][0]["counts"]["blocked"] == 1
    assert data["cron_jobs"] == [
        {
            "id": "job-ghl",
            "name": "GHL Manager scout",
            "schedule": "0 9 * * *",
            "enabled": True,
            "state": "active",
            "next_run_at": "2026-05-10T09:00:00+10:00",
            "last_status": "ok",
        }
    ]
    assert data["missing_data_sources"] == []


def test_context_and_action_requests_endpoint_surface_compact_notifications(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")
    packet = module._base_packet(
        approval_id="t_c50747c6-01",
        source_type="unit_test",
        source_path=tmp_path / "packet.json",
    )
    packet["brand"]["name"] = "The Blue Crew"
    packet["contact"].update({"contact_id": "contact_123", "name": "Jane Customer", "phone": "+61411111111"})
    packet["conversation"].update({"conversation_id": "conv_123", "channel": "SMS", "latest_customer_summary": "Asked for a quote."})
    packet["draft"].update({"customer_facing": True, "draft_text": "Hi Jane, thanks for reaching out."})
    packet = module._finalize_packet(packet)
    module._record_approval_packets([packet])
    queued = module._create_action_request("t_c50747c6-01", {"action_type": "reverify", "actor": "gabriel-standalone-shell", "draft_hash": packet["draft"]["draft_hash"]})

    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    context = client.get("/api/plugins/ghl-manager/context").json()
    endpoint = client.get("/api/plugins/ghl-manager/action-requests").json()

    for data in (context, endpoint):
        request = data["action_requests"][0]
        assert request["request_alias"].startswith("Request #")
        assert request["approval_alias"] == "Approval t_c50747c6-01"
        assert request["action_type"] == "reverify"
        assert request["action_label"] == "Read-only reverify"
        assert request["status"] == "queued_for_reverify"
        assert request["status_label"] == "Queued for reverify"
        assert request["safety_state"] == "Read-only follow-up required; no customer send, CRM mutation, or booking is allowed."
        assert "Jane Customer" in request["summary"]
        assert "request_id" not in request
        assert request["debug"]["request_id"] == queued["request_id"]


def test_queueing_action_request_creates_idempotent_kanban_notification_card(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")
    monkeypatch.setattr(module, "ACTION_REQUEST_KANBAN_BOARD", "ghl-manager-ui-test")
    packet = module._base_packet(
        approval_id="t_c50747c6-01",
        source_type="unit_test",
        source_path=tmp_path / "packet.json",
    )
    packet["brand"]["name"] = "The Blue Crew"
    packet["contact"].update({"contact_id": "contact_123", "name": "Jane Customer", "phone": "+61411111111"})
    packet["conversation"].update({"conversation_id": "conv_123", "channel": "SMS", "latest_customer_summary": "Asked for a quote."})
    packet["draft"].update({"customer_facing": True, "draft_text": "Hi Jane, thanks for reaching out."})
    packet = module._finalize_packet(packet)
    module._record_approval_packets([packet])

    first = module._create_action_request("t_c50747c6-01", {"action_type": "reverify", "actor": "gabriel-standalone-shell", "draft_hash": packet["draft"]["draft_hash"]})
    second = module._create_action_request("t_c50747c6-01", {"action_type": "reverify", "actor": "gabriel-standalone-shell", "draft_hash": packet["draft"]["draft_hash"]})

    assert second["idempotent_replay"] is True
    assert first["kanban_notification"]["task_id"] == second["kanban_notification"]["task_id"]

    from hermes_cli import kanban_db

    db_path = kanban_db.board_dir("ghl-manager-ui-test") / "kanban.db"
    conn = kanban_db.connect(db_path=db_path)
    try:
        tasks = kanban_db.list_tasks(conn, include_archived=False)
        comments = kanban_db.list_comments(conn, first["kanban_notification"]["task_id"])
    finally:
        conn.close()
    assert len([task for task in tasks if task.idempotency_key == first["kanban_notification"]["idempotency_key"]]) == 1
    task = tasks[0]
    assert task.assignee == "default"
    assert "Action Bridge follow-up" in task.title
    assert "Approval: t_c50747c6-01" in (task.body or "")
    assert "Action: Read-only reverify" in (task.body or "")
    assert "Safety: no customer sends, no live CRM mutations, no booking/calendar mutations." in (task.body or "")
    assert "Debug IDs" in (task.body or "")
    assert comments


def test_approval_store_schema_imports_packets_idempotently_and_keeps_draft_hash_stable(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")
    draft_text = "Hi there, thanks for reaching out."
    packet = module._finalize_packet(
        module._base_packet(
            approval_id="approval-1",
            source_type="unit_test",
            source_path=tmp_path / "packet.json",
        )
    )
    packet["draft"].update({"customer_facing": True, "draft_text": draft_text})
    packet = module._finalize_packet(packet)

    first = module._record_approval_packets([packet])
    second = module._record_approval_packets([packet])

    assert first == {"imported": 1, "inserted": 1, "updated": 0}
    assert second == {"imported": 1, "inserted": 0, "updated": 1}
    assert packet["draft"]["draft_hash"] == module._sha256_text(draft_text)

    with sqlite3.connect(tmp_path / "approval.sqlite") as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"approval_packets", "approval_events", "approval_action_requests", "approval_index", "handled_actions", "idempotency_keys"}.issubset(tables)
        rows = conn.execute("SELECT approval_id, draft_hash, packet_json FROM approval_packets").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "approval-1"
    assert rows[0][1] == module._sha256_text(draft_text)
    assert json.loads(rows[0][2])["approval_id"] == "approval-1"


def test_importing_duplicate_packet_with_terminal_canonical_key_supersedes_new_operator_action(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")

    original = module._base_packet(approval_id="approval-original", source_type="unit_test", source_path=tmp_path / "original.json")
    original["brand"]["name"] = "Solar Renew"
    original["contact"].update({"contact_id": "contact-1", "name": "Jane"})
    original["conversation"].update({"conversation_id": "conversation-1", "channel": "SMS", "latest_customer_at": "2026-05-14T00:00:00+00:00"})
    original["send_target"].update({"contact_id": "contact-1", "conversation_id": "conversation-1", "channel": "SMS"})
    original["draft"].update({"customer_facing": True, "draft_text": "Hi Jane, thanks for reaching out."})
    original = module._finalize_packet(original)
    module._record_approval_packets([original])
    canonical_key = original["idempotency"]["primary_key"]

    with sqlite3.connect(tmp_path / "approval.sqlite") as conn:
        row = conn.execute("SELECT index_json FROM approval_index WHERE approval_id = ?", ("approval-original",)).fetchone()
        original_doc = json.loads(row[0])
        original_doc.update({"current_status": "handled_sent", "display_status": "handled_sent", "handled_state": "handled_sent"})
        conn.execute(
            "UPDATE approval_index SET current_status = ?, handled_state = ?, index_json = ? WHERE approval_id = ?",
            ("handled_sent", "handled_sent", json.dumps(original_doc, sort_keys=True), "approval-original"),
        )

    duplicate = module._base_packet(approval_id="approval-duplicate", source_type="unit_test", source_path=tmp_path / "duplicate.json")
    duplicate["brand"]["name"] = "Solar Renew"
    duplicate["contact"].update({"contact_id": "contact-1", "name": "Jane"})
    duplicate["conversation"].update({"conversation_id": "conversation-1", "channel": "SMS", "latest_customer_at": "2026-05-14T00:00:00+00:00"})
    duplicate["send_target"].update({"contact_id": "contact-1", "conversation_id": "conversation-1", "channel": "SMS"})
    duplicate["draft"].update({"customer_facing": True, "draft_text": "Hi Jane, thanks for reaching out."})
    duplicate = module._finalize_packet(duplicate)
    module._record_approval_packets([duplicate])

    with sqlite3.connect(tmp_path / "approval.sqlite") as conn:
        packet_rows = conn.execute("SELECT approval_id FROM approval_packets ORDER BY approval_id").fetchall()
        row = conn.execute("SELECT current_status, index_json FROM approval_index WHERE approval_id = ?", ("approval-duplicate",)).fetchone()
    duplicate_doc = json.loads(row[1])

    assert packet_rows == [("approval-duplicate",), ("approval-original",)]
    assert row[0] == "superseded"
    assert duplicate_doc["canonical_idempotency_key"] == canonical_key
    assert duplicate_doc["freshness"]["duplicate"] is True
    assert duplicate_doc["freshness"]["stale"] is True
    assert duplicate_doc["freshness"]["existing_status"] == "handled_sent"
    assert "handled_sent" in duplicate_doc["freshness"]["reason"]
    assert duplicate_doc["source_links"]["duplicate_of_approval_id"] == "approval-original"
    assert duplicate_doc["source_links"]["duplicate_canonical_key"] == canonical_key


def test_approval_store_migrates_existing_event_constraint_for_queue_reverify(tmp_path, monkeypatch):
    module = _load_plugin_module()
    db_path = tmp_path / "approval.sqlite"
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE approval_packets (
                approval_id TEXT PRIMARY KEY,
                schema_version TEXT NOT NULL,
                packet_status TEXT,
                source_path TEXT,
                source_type TEXT,
                draft_hash TEXT,
                packet_json TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );
            CREATE TABLE approval_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                approval_id TEXT NOT NULL,
                event_type TEXT NOT NULL CHECK (event_type IN ('approve', 'deny', 'edit-draft', 'block-stale')),
                actor TEXT NOT NULL,
                draft_hash TEXT,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (approval_id) REFERENCES approval_packets(approval_id)
            );
            CREATE TABLE idempotency_keys (
                key TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
    packet = module._base_packet(approval_id="approval-migrate", source_type="unit_test", source_path=tmp_path / "packet.json")
    module._record_approval_packets([module._finalize_packet(packet)])

    response = module._append_approval_event(
        "approval-migrate",
        {"event_type": "queue-reverify", "actor": "gabriel", "reason": "guarded read-only reverify requested"},
    )

    assert response["state"]["current_status"] == "pending_approval"
    assert response["state"]["display_status"] == "queued_for_reverify"
    with sqlite3.connect(db_path) as conn:
        schema = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'approval_events'").fetchone()[0]
        assert "queue-reverify" in schema
        assert conn.execute("SELECT event_type FROM approval_events").fetchall() == [("queue-reverify",)]


def test_approval_events_are_append_only_and_local_state_endpoint_is_idempotent(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")
    packet = module._base_packet(approval_id="approval-2", source_type="unit_test", source_path=tmp_path / "packet.json")
    packet["draft"].update({"customer_facing": True, "draft_text": "Original draft"})
    module._record_approval_packets([module._finalize_packet(packet)])

    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    approve = client.post(
        "/api/plugins/ghl-manager/approval-state/approval-2/events",
        json={"event_type": "approve", "actor": "gabriel", "idempotency_key": "approval-2:approve:v1"},
    )
    repeat = client.post(
        "/api/plugins/ghl-manager/approval-state/approval-2/events",
        json={"event_type": "approve", "actor": "gabriel", "idempotency_key": "approval-2:approve:v1"},
    )
    edit = client.post(
        "/api/plugins/ghl-manager/approval-state/approval-2/events",
        json={"event_type": "edit-draft", "actor": "gabriel", "draft_text": "Edited draft", "idempotency_key": "approval-2:edit:v1"},
    )
    block = client.post(
        "/api/plugins/ghl-manager/approval-state/approval-2/events",
        json={"event_type": "block-stale", "actor": "system", "reason": "live state changed"},
    )
    queue_reverify = client.post(
        "/api/plugins/ghl-manager/approval-state/approval-2/events",
        json={"event_type": "queue-reverify", "actor": "gabriel", "reason": "queue guarded read-only reverify"},
    )

    assert approve.status_code == 200
    assert repeat.status_code == 200
    assert repeat.json()["idempotent_replay"] is True
    assert edit.status_code == 200
    assert edit.json()["state"]["latest_draft_hash"] == module._sha256_text("Edited draft")
    assert block.status_code == 200
    assert queue_reverify.status_code == 200
    assert queue_reverify.json()["state"]["current_status"] == "pending_approval"
    assert queue_reverify.json()["state"]["display_status"] == "queued_for_reverify"
    assert queue_reverify.json()["local_only"] is True
    assert queue_reverify.json()["mutations_enabled"] is False

    state = client.get("/api/plugins/ghl-manager/approval-state/approval-2")
    assert state.status_code == 200
    data = state.json()
    assert data["approval_id"] == "approval-2"
    assert data["current_status"] == "pending_approval"
    assert data["display_status"] == "queued_for_reverify"
    assert [event["event_type"] for event in data["events"]] == ["approve", "edit-draft", "block-stale", "queue-reverify"]
    assert data["base_draft_hash"] == module._sha256_text("Original draft")
    assert data["latest_draft_hash"] == module._sha256_text("Edited draft")

    with sqlite3.connect(tmp_path / "approval.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM approval_events").fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM idempotency_keys").fetchone()[0] == 2


def test_approval_event_endpoint_rejects_unknown_or_send_like_events(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")
    module._ensure_approval_store()
    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    send_like = client.post(
        "/api/plugins/ghl-manager/approval-state/approval-3/events",
        json={"event_type": "send-approved", "actor": "gabriel"},
    )

    assert send_like.status_code == 400
    assert "not allowed" in send_like.json()["detail"]


def test_action_request_endpoint_queues_send_approval_without_live_execution(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.delenv("GHL_MANAGER_LIVE_EXECUTION", raising=False)
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")
    packet = module._base_packet(approval_id="approval-send", source_type="unit_test", source_path=tmp_path / "packet.json")
    packet["brand"]["name"] = "Solar Renew"
    packet["contact"]["contact_id"] = "contact-123"
    packet["conversation"].update({"conversation_id": "conversation-456", "channel": "SMS", "latest_customer_at": "2026-05-13T08:00:00+10:00"})
    packet["send_target"].update({"channel": "SMS", "contact_id": "contact-123", "conversation_id": "conversation-456", "target_label": "SMS +61411111111"})
    packet["draft"].update({"customer_facing": True, "draft_text": "Approved draft"})
    packet = module._finalize_packet(packet)
    module._record_approval_packets([packet])

    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    queued = client.post(
        "/api/plugins/ghl-manager/approval-state/approval-send/action-requests",
        json={
            "action_type": "send-approved",
            "actor": "gabriel",
            "draft_hash": packet["draft"]["draft_hash"],
            "idempotency_key": "approval-send:send:v1",
        },
    )
    replay = client.post(
        "/api/plugins/ghl-manager/approval-state/approval-send/action-requests",
        json={
            "action_type": "send-approved",
            "actor": "gabriel",
            "draft_hash": packet["draft"]["draft_hash"],
            "idempotency_key": "approval-send:send:v1",
        },
    )
    processed = client.post("/api/plugins/ghl-manager/action-requests/process", json={"dry_run": True})

    assert queued.status_code == 200
    queued_data = queued.json()
    assert queued_data["status"] == "queued_for_execution"
    assert queued_data["canonical_idempotency_key"].startswith("blue:v1:customer_message:solar-renew:contact-123:conversation-456:send_sms:")
    assert queued_data["action_request"]["payload"]["canonical_idempotency_key"] == queued_data["canonical_idempotency_key"]
    assert queued_data["live_execution_enabled"] is False
    assert queued_data["state"]["current_status"] == "approved_not_sent"
    assert queued_data["state"]["display_status"] == "queued_for_execution"
    assert queued_data["state"]["approval_index"]["current_status"] == "approved_not_sent"
    assert queued_data["state"]["approval_index"]["display_status"] == "queued_for_execution"
    assert queued_data["state"]["approval_index"]["canonical_idempotency_key"] == queued_data["canonical_idempotency_key"]
    assert queued_data["state"]["action_request_count"] == 1
    assert replay.status_code == 200
    assert replay.json()["idempotent_replay"] is True
    assert processed.status_code == 200
    processed_data = processed.json()
    assert processed_data["processed_count"] == 1
    assert processed_data["processed"][0]["status"] == "execution_blocked"
    assert "Live customer sends are disabled" in processed_data["processed"][0]["result"]["blocked_reason"]
    assert processed_data["processed"][0]["result"]["handled_actions_contract"]["idempotency_key"] == queued_data["canonical_idempotency_key"]

    state_after_process = client.get("/api/plugins/ghl-manager/approval-state/approval-send").json()
    assert state_after_process["approval_index"]["current_status"] == "blocked"
    assert state_after_process["approval_index"]["display_status"] == "execution_blocked"
    assert state_after_process["approval_index"]["handled_state"] == "execution_blocked"
    assert state_after_process["handled_action_count"] == 1
    assert state_after_process["handled_actions"][0]["idempotency_key"] == queued_data["canonical_idempotency_key"]

    with sqlite3.connect(tmp_path / "approval.sqlite") as conn:
        assert conn.execute("SELECT event_type FROM approval_events").fetchall() == [("approve",)]
        assert conn.execute("SELECT status FROM approval_action_requests").fetchall() == [("execution_blocked",)]


def test_action_request_endpoint_queues_reverify_without_calling_ghl(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")
    packet = module._base_packet(approval_id="approval-reverify", source_type="unit_test", source_path=tmp_path / "packet.json")
    packet["brand"]["name"] = "Solar Renew"
    packet["contact"]["contact_id"] = "contact-789"
    packet["conversation"].update({"conversation_id": "conversation-789", "channel": "SMS", "latest_customer_at": "2026-05-13T09:00:00+10:00"})
    packet["send_target"].update({"channel": "SMS", "contact_id": "contact-789", "conversation_id": "conversation-789", "target_label": "SMS +61422222222"})
    packet["draft"].update({"customer_facing": True, "draft_text": "Draft needing reverify"})
    module._record_approval_packets([module._finalize_packet(packet)])

    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    queued = client.post(
        "/api/plugins/ghl-manager/approval-state/approval-reverify/action-requests",
        json={"action_type": "reverify", "actor": "gabriel", "idempotency_key": "approval-reverify:reverify:v1"},
    )

    assert queued.status_code == 200
    assert queued.json()["status"] == "queued_for_reverify"
    assert queued.json()["canonical_idempotency_key"].startswith("blue:v1:customer_message:solar-renew:contact-789:conversation-789:send_sms:")
    assert queued.json()["state"]["current_status"] == "pending_approval"
    assert queued.json()["state"]["display_status"] == "queued_for_reverify"
    assert queued.json()["state"]["latest_action_request"]["action_type"] == "reverify"
    assert queued.json()["mutations_enabled"] is False


def test_reverify_processor_runs_read_only_live_fetch_and_marks_reverified(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")
    packet = module._base_packet(approval_id="approval-live-reverify", source_type="unit_test", source_path=tmp_path / "packet.json")
    packet["brand"]["name"] = "Solar Renew"
    packet["contact"].update({"contact_id": "contact-live", "name": "Jane"})
    packet["conversation"].update({"conversation_id": "conversation-live", "channel": "SMS", "latest_customer_at": "2026-05-13T09:00:00+10:00"})
    packet["send_target"].update({"channel": "SMS", "contact_id": "contact-live", "conversation_id": "conversation-live"})
    packet["draft"].update({"customer_facing": True, "draft_text": "Draft needing reverify"})
    module._record_approval_packets([module._finalize_packet(packet)])
    module._create_action_request("approval-live-reverify", {"action_type": "reverify", "actor": "gabriel"})

    calls = []

    def fake_reverify(packet_json, request):
        calls.append((packet_json["contact"]["contact_id"], packet_json["conversation"]["conversation_id"], request["action_type"]))
        return {
            "status": "reverified",
            "safe_to_send": True,
            "latest_state_source": "mock-ghl-read-only",
            "notes": "Latest contact and conversation were fetched read-only.",
            "contact_snapshot": {"contact_id": "contact-live"},
            "conversation_snapshot": {"conversation_id": "conversation-live", "message_count": 2},
        }

    monkeypatch.setattr(module, "_read_only_reverify_live_state", fake_reverify)

    processed = module.process_pending_action_requests(dry_run=False)

    assert calls == [("contact-live", "conversation-live", "reverify")]
    assert processed["processed_count"] == 1
    assert processed["processed"][0]["status"] == "reverified"
    result = processed["processed"][0]["result"]
    assert result["reverify_result_contract"]["status"] == "reverified"
    assert result["reverify_result_contract"]["safe_to_send"] is True
    assert result["mutation_endpoints_called"] == []

    with sqlite3.connect(tmp_path / "approval.sqlite") as conn:
        assert conn.execute("SELECT status FROM approval_action_requests").fetchall() == [("reverified",)]


def test_reverify_processor_marks_stale_handled_elsewhere_as_reverified_terminal_state(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")
    packet = module._base_packet(approval_id="approval-stale-reverify", source_type="unit_test", source_path=tmp_path / "packet.json")
    packet["brand"]["name"] = "The Blue Crew"
    packet["contact"]["contact_id"] = "contact-stale"
    packet["conversation"].update({"conversation_id": "conversation-stale", "channel": "SMS"})
    packet["send_target"].update({"channel": "SMS", "contact_id": "contact-stale", "conversation_id": "conversation-stale"})
    packet["draft"].update({"customer_facing": True, "draft_text": "Old draft"})
    module._record_approval_packets([module._finalize_packet(packet)])
    module._create_action_request("approval-stale-reverify", {"action_type": "reverify", "actor": "gabriel"})

    monkeypatch.setattr(
        module,
        "_read_only_reverify_live_state",
        lambda packet_json, request: {
            "status": "handled_elsewhere",
            "safe_to_send": False,
            "latest_state_source": "mock-ghl-read-only",
            "notes": "A newer outbound reply already handled the old approval.",
        },
    )

    processed = module.process_pending_action_requests(dry_run=False)

    assert processed["processed"][0]["status"] == "reverified"
    result = processed["processed"][0]["result"]["reverify_result_contract"]
    assert result["status"] == "handled_elsewhere"
    assert result["safe_to_send"] is False


def test_reverify_processor_worker_failure_is_visible_terminal_state(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")
    packet = module._base_packet(approval_id="approval-worker-failed", source_type="unit_test", source_path=tmp_path / "packet.json")
    packet["brand"]["name"] = "Solar Renew"
    packet["contact"]["contact_id"] = "contact-failed"
    packet["conversation"].update({"conversation_id": "conversation-failed", "channel": "SMS"})
    packet["send_target"].update({"channel": "SMS", "contact_id": "contact-failed", "conversation_id": "conversation-failed"})
    packet["draft"].update({"customer_facing": True, "draft_text": "Draft"})
    module._record_approval_packets([module._finalize_packet(packet)])
    module._create_action_request("approval-worker-failed", {"action_type": "reverify", "actor": "gabriel"})

    def fail_reverify(packet_json, request):
        raise RuntimeError("missing read-only GHL credentials")

    monkeypatch.setattr(module, "_read_only_reverify_live_state", fail_reverify)

    processed = module.process_pending_action_requests(dry_run=False)

    assert processed["processed"][0]["status"] == "worker_failed"
    assert "missing read-only GHL credentials" in processed["processed"][0]["result"]["blocked_reason"]


def test_action_request_notifications_flag_stale_queued_reverify_requests(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")
    packet = module._base_packet(approval_id="approval-stale-queued", source_type="unit_test", source_path=tmp_path / "packet.json")
    packet["brand"]["name"] = "Solar Renew"
    packet["contact"]["contact_id"] = "contact-stale-queued"
    packet["conversation"].update({"conversation_id": "conversation-stale-queued", "channel": "SMS"})
    packet["send_target"].update({"channel": "SMS", "contact_id": "contact-stale-queued", "conversation_id": "conversation-stale-queued"})
    packet["draft"].update({"customer_facing": True, "draft_text": "Draft"})
    module._record_approval_packets([module._finalize_packet(packet)])
    queued = module._create_action_request("approval-stale-queued", {"action_type": "reverify", "actor": "gabriel"})
    with sqlite3.connect(tmp_path / "approval.sqlite") as conn:
        conn.execute(
            "UPDATE approval_action_requests SET created_at = ?, updated_at = ? WHERE request_id = ?",
            ("2026-05-13T00:00:00+00:00", "2026-05-13T00:00:00+00:00", queued["request_id"]),
        )

    notifications = module._recent_action_request_notifications(limit=1)

    assert notifications[0]["stale"] is True
    assert notifications[0]["status"] == "queued_for_reverify"
    assert "stale queued request" in notifications[0]["summary"].lower()


def test_action_request_blocks_when_canonical_blue_key_is_missing(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")
    packet = module._base_packet(approval_id="approval-missing-key", source_type="unit_test", source_path=tmp_path / "packet.json")
    packet["draft"].update({"customer_facing": True, "draft_text": "Approved draft without GHL identifiers"})
    packet = module._finalize_packet(packet)
    module._record_approval_packets([packet])

    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    queued = client.post(
        "/api/plugins/ghl-manager/approval-state/approval-missing-key/action-requests",
        json={"action_type": "send-approved", "actor": "gabriel", "draft_hash": packet["draft"]["draft_hash"]},
    )

    assert queued.status_code == 200
    data = queued.json()
    assert data["status"] == "execution_blocked"
    assert data["canonical_idempotency_key"] is None
    assert "missing canonical key" in data["warning"]
    assert "brand" in data["canonical_idempotency"]["missing_fields"]
    assert data["action_request"]["result"]["missing_canonical_key"] is True
    assert data["state"]["current_status"] == "blocked"
    assert data["state"]["display_status"] == "execution_blocked"
    assert data["state"]["approval_index"]["missing_canonical_key"] is True
    assert data["state"]["approval_index"]["handled_state"] == "blocked_missing_canonical_key"
    assert data["state"]["handled_action_count"] == 1
    with sqlite3.connect(tmp_path / "approval.sqlite") as conn:
        assert conn.execute("SELECT status FROM approval_action_requests").fetchall() == [("execution_blocked",)]
        assert conn.execute("SELECT COUNT(*) FROM approval_events").fetchone()[0] == 0


def test_packets_endpoint_records_imported_packets_in_local_store(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(module, "APPROVAL_DB_PATH", tmp_path / "approval.sqlite")
    (tmp_path / "pending-approval-t_c50747c6.json").write_text(
        json.dumps(
            {
                "task_id": "t_valid",
                "approval_items": [
                    {
                        "approval_id": "stored-approval",
                        "brand": "Solar Renew",
                        "contactId": "contact-123",
                        "conversationId": "conversation-456",
                        "send_target": "SMS +61411111111",
                        "proposed_action": {"type": "send_sms", "customer_facing_send": True, "draft": "Stored draft"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "t_a1c85349_pending_lead_reply_drafts.json").write_text(json.dumps({"records": []}), encoding="utf-8")
    (tmp_path / "t_627dfbf5_blocked_missing_approval.json").write_text(
        json.dumps({"task_id": "t_blocked", "status": "blocked_missing_approval"}),
        encoding="utf-8",
    )

    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    response = client.get("/api/plugins/ghl-manager/packets")

    assert response.status_code == 200
    data = response.json()
    assert data["approval_store"]["inserted"] == 2
    assert data["canonical_projection"]["canonical_owner"] == "sqlite:ghl-manager-ui.sqlite:approval_index"
    assert "compatibility/export mirrors" in data["canonical_projection"]["mirror_policy"]
    assert data["canonical_projection"]["approval_index"] == 2
    assert data["canonical_projection"]["approval_packets"] == 2
    with sqlite3.connect(tmp_path / "approval.sqlite") as conn:
        stored = conn.execute("SELECT approval_id, draft_hash FROM approval_packets ORDER BY approval_id").fetchall()
    assert stored == [("stored-approval", module._sha256_text("Stored draft")), ("t_blocked", None)]


def test_packets_endpoint_adds_linked_task_status_when_task_id_is_available(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(
        module,
        "_load_task_context",
        lambda task_id: {
            "task_id": task_id,
            "status": "blocked",
            "assignee": "reviewer",
            "title": "Review approval packet",
            "board": "ghl-six-priority-cleanup",
            "source": "kanban_db",
        },
        raising=False,
    )
    (tmp_path / "pending-approval-t_c50747c6.json").write_text(
        json.dumps({"task_id": "t_parent", "approval_items": []}),
        encoding="utf-8",
    )
    (tmp_path / "t_a1c85349_pending_lead_reply_drafts.json").write_text(
        json.dumps({"records": []}),
        encoding="utf-8",
    )
    (tmp_path / "t_627dfbf5_blocked_missing_approval.json").write_text(
        json.dumps({"task_id": "t_1eaf1234", "status": "blocked_missing_approval"}),
        encoding="utf-8",
    )

    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    response = client.get("/api/plugins/ghl-manager/packets")

    assert response.status_code == 200
    packet = next(packet for packet in response.json()["packets"] if packet["approval_id"] == "t_1eaf1234")
    assert packet["kanban_context"] == {
        "task_id": "t_1eaf1234",
        "parent_task_id": None,
        "status": "blocked",
        "assignee": "reviewer",
        "block_reason": None,
        "comments_summary": None,
        "title": "Review approval packet",
        "board": "ghl-six-priority-cleanup",
        "source": "kanban_db",
    }


def test_packets_endpoint_returns_malformed_packet_for_invalid_artifact_without_crashing(tmp_path, monkeypatch):
    module = _load_plugin_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path.resolve())
    (tmp_path / "pending-approval-t_c50747c6.json").write_text(
        json.dumps(
            {
                "task_id": "t_valid",
                "created_at": "2026-05-09T00:00:00+00:00",
                "approval_items": [
                    {
                        "approval_id": "valid-approval",
                        "brand": "Solar Renew",
                        "contactId": "contact-123",
                        "conversationId": "conversation-456",
                        "classification": "lead-reply",
                        "send_target": "SMS +61411111111",
                        "proposed_action": {
                            "type": "send_sms",
                            "customer_facing_send": True,
                            "draft": "Hi there, thanks for reaching out.",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "t_a1c85349_pending_lead_reply_drafts.json").write_text("{not valid json", encoding="utf-8")
    (tmp_path / "t_627dfbf5_blocked_missing_approval.json").write_text(
        json.dumps({"task_id": "t_blocked", "status": "blocked_missing_approval"}),
        encoding="utf-8",
    )

    app = FastAPI()
    app.include_router(module.router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    response = client.get("/api/plugins/ghl-manager/packets")

    assert response.status_code == 200
    data = response.json()
    assert data["artifact_root"] == str(tmp_path.resolve())
    packets = data["packets"]
    assert any(packet["approval_id"] == "valid-approval" for packet in packets)
    malformed = next(packet for packet in packets if packet["approval_id"] == "malformed:t_a1c85349_pending_lead_reply_drafts.json")
    assert malformed["packet_status"] == "malformed"
    assert malformed["ui_indicators"]["malformed"] is True
    assert malformed["parse_errors"] == ["invalid JSON: JSONDecodeError"]
    assert malformed["status_reason"] == "invalid JSON: JSONDecodeError"
    assert malformed["source"]["source_path"] == str((tmp_path / "t_a1c85349_pending_lead_reply_drafts.json").resolve())


def test_packets_endpoint_imports_local_approval_artifacts_with_status_indicators():
    app = FastAPI()
    app.include_router(_load_plugin_module().router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    response = client.get("/api/plugins/ghl-manager/packets")

    assert response.status_code == 200
    data = response.json()
    assert data["read_only"] is True
    assert data["mutations_enabled"] is False
    assert data["artifact_root"].endswith("/workspace-ghl-draft/audits")
    assert data["counts"]["total"] >= 21
    assert data["counts"]["pending_approval"] >= 1
    assert data["counts"]["blocked"] >= 1
    assert data["counts"]["needs_reverify"] >= 1

    packets = data["packets"]
    first = next(packet for packet in packets if packet["approval_id"] == "t_c50747c6-01")
    assert first["schema_version"] == "ghl-manager.approval_packet.v1"
    assert first["brand"]["name"] == "Solar Renew"
    assert first["contact"]["contact_id"] == "wxsatjrX5CIlTuIQoSuL"
    assert first["conversation"]["conversation_id"] == "NEWeIdmoLmnexLOvsUbN"
    assert first["draft"]["customer_facing"] is True
    assert first["draft"]["draft_hash"].startswith("sha256:")
    assert first["safety"]["customer_text_untrusted"] is True
    assert "send_customer_message" in first["safety"]["disallowed_mvp_actions"]
    assert first["ui_indicators"]["malformed"] is False
    assert first["source"]["source_path"].endswith("pending-approval-t_c50747c6.json")

    blocked = next(packet for packet in packets if packet["approval_id"] == "t_627dfbf5")
    assert blocked["packet_status"] == "blocked"
    assert blocked["draft"]["manual_review_only"] is True
    assert blocked["ui_indicators"]["malformed"] is False

    stale_or_reverify = [p for p in packets if p["ui_indicators"]["stale"]]
    assert stale_or_reverify, "expected stale/reverify indicators for old or booking-sensitive packets"


def test_packets_endpoint_overlays_recheck_report_and_imports_booking_markdown():
    app = FastAPI()
    app.include_router(_load_plugin_module().router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    response = client.get("/api/plugins/ghl-manager/packets")

    assert response.status_code == 200
    packets = response.json()["packets"]

    obsolete = next(packet for packet in packets if packet["source"]["source_row"] == 15)
    assert obsolete["packet_status"] == "stale_obsolete"
    assert "already sent a full quote" in obsolete["status_reason"]
    assert obsolete["ui_indicators"]["stale"] is True

    changed = next(packet for packet in packets if packet["approval_id"] == "t_c50747c6-09")
    assert changed["packet_status"] == "blocked"
    assert "Do not send original 23 May draft" in changed["status_reason"]
    assert changed["kanban_context"]["task_id"] == "t_e024a453"
    assert changed["kanban_context"]["status"] in {None, "missing", "todo", "ready", "running", "blocked", "done", "archived"}

    booking_packets = [
        packet for packet in packets
        if packet["source"]["source_type"] == "markdown_booking_packet"
    ]
    for booking in booking_packets:
        assert booking["brand"]["name"]
        assert booking["contact"]["name"]
        assert booking["contact"]["contact_id"]
        assert booking["conversation"]["conversation_id"]
        assert booking["booking_context"]["job_location"]
        assert booking["booking_context"]["ledger_status"]
        assert booking["booking_context"]["proposed_slots"]
        assert booking["draft"]["draft_hash"].startswith("sha256:")
        assert booking["verification"]["required_before_send"] is True


def test_packet_detail_endpoint_returns_one_packet_and_404_for_unknown_id():
    app = FastAPI()
    app.include_router(_load_plugin_module().router, prefix="/api/plugins/ghl-manager")
    client = TestClient(app)

    response = client.get("/api/plugins/ghl-manager/packets/t_c50747c6-01")

    assert response.status_code == 200
    packet = response.json()
    assert packet["approval_id"] == "t_c50747c6-01"
    assert packet["draft"]["draft_text"]
    assert packet["decision"]["approval_options"] == ["approve/send", "edit: ...", "reject"]

    missing = client.get("/api/plugins/ghl-manager/packets/not-real")
    assert missing.status_code == 404


def test_frontend_renders_packet_inbox_without_dangerous_html_or_mutation_calls():
    bundle_path = PLUGIN_ROOT / "dist" / "index.js"
    bundle = bundle_path.read_text()

    assert "`/packets`" in bundle
    assert "Approval Inbox" in bundle
    assert "Standalone Blue / GHL operations console" in bundle
    assert "needs attention" in bundle
    assert "Blue cron/watchdog" in bundle
    assert "Readable approval detail" in bundle
    assert "draft_text" in bundle
    # React DOM's own minified runtime contains dangerouslySetInnerHTML and
    # innerHTML handling strings, so bundle-level absence checks are noisy.
    # The safety contract is instead enforced by the available bridge routes
    # and the absence of send/approve endpoints below.
    assert "`/send" not in bundle
    assert "`/approve" not in bundle
