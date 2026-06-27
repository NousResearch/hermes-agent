"""Tests for the Desktop Work Queue API."""

import pytest


def _client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover - dependency guard
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli import web_server

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client, web_server


def test_work_queue_manual_item_round_trip(monkeypatch, _isolate_hermes_home):
    client, _web_server = _client(monkeypatch, _isolate_hermes_home)

    resp = client.get("/api/work-queue")
    assert resp.status_code == 200
    assert resp.json() == {"items": []}

    resp = client.post(
        "/api/work-queue/items",
        json={"title": "Call client", "summary": "Needs follow-up", "priority": "high"},
    )
    assert resp.status_code == 200
    item = resp.json()
    assert item["id"].startswith("manual:")
    assert item["title"] == "Call client"
    assert item["priority"] == "high"
    assert item["derived"] is False

    resp = client.patch(f"/api/work-queue/items/{item['id']}", json={"status": "done"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "done"

    resp = client.post(
        f"/api/work-queue/items/{item['id']}/snooze",
        json={"snoozed_until": "2030-01-01T00:00:00Z"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "snoozed"
    assert resp.json()["snoozed_until"] == "2030-01-01T00:00:00Z"

    resp = client.get("/api/work-queue?status=snoozed")
    assert resp.status_code == 200
    assert [row["id"] for row in resp.json()["items"]] == [item["id"]]


def test_work_queue_derived_overrides_merge_and_archive(monkeypatch, _isolate_hermes_home):
    client, web_server = _client(monkeypatch, _isolate_hermes_home)

    def fake_derived():
        return [
            web_server._work_item(
                id="cron:failed:default:abc",
                source="cron",
                status="failed",
                priority="high",
                title="Cron failed: abc",
                metadata={"job_id": "abc"},
            )
        ]

    monkeypatch.setattr(web_server, "_derived_work_items", fake_derived)

    resp = client.get("/api/work-queue")
    assert resp.status_code == 200
    rows = [row for row in resp.json()["items"] if row["id"] == "cron:failed:default:abc"]
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"

    resp = client.patch("/api/work-queue/items/cron:failed:default:abc", json={"status": "done"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "done"
    assert resp.json()["title"] == "Cron failed: abc"

    resp = client.get("/api/work-queue")
    assert resp.status_code == 200
    rows = [row for row in resp.json()["items"] if row["id"] == "cron:failed:default:abc"]
    assert len(rows) == 1
    assert rows[0]["status"] == "done"
    assert rows[0]["metadata"]["job_id"] == "abc"

    resp = client.post("/api/work-queue/items/cron:failed:default:abc/archive")
    assert resp.status_code == 200
    assert resp.json()["status"] == "archived"

    resp = client.get("/api/work-queue")
    assert resp.status_code == 200
    rows = [row for row in resp.json()["items"] if row["id"] == "cron:failed:default:abc"]
    assert rows == []
