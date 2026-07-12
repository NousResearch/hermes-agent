"""Dashboard Kanban endpoints used by the desktop task panel."""

from starlette.testclient import TestClient


def _client():
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


def test_kanban_task_panel_lifecycle(_isolate_hermes_home):
    client = _client()

    listed = client.get("/api/kanban/tasks")
    assert listed.status_code == 200
    assert listed.json()["tasks"] == []

    created = client.post(
        "/api/kanban/tasks",
        json={"title": "Panel task", "body": "Visible in desktop", "status": "ready"},
    )
    assert created.status_code == 200
    task = created.json()["task"]
    assert task["title"] == "Panel task"
    assert task["status"] == "ready"

    listed = client.get("/api/kanban/tasks")
    assert listed.status_code == 200
    payload = listed.json()
    assert payload["counts"]["ready"] == 1
    assert [row["id"] for row in payload["tasks"]] == [task["id"]]

    blocked = client.post(
        f"/api/kanban/tasks/{task['id']}/block",
        json={"reason": "Waiting on user"},
    )
    assert blocked.status_code == 200
    assert blocked.json()["task"]["status"] == "blocked"

    unblocked = client.post(f"/api/kanban/tasks/{task['id']}/unblock")
    assert unblocked.status_code == 200
    assert unblocked.json()["task"]["status"] == "ready"

    completed = client.post(
        f"/api/kanban/tasks/{task['id']}/complete",
        json={"summary": "Done from task panel"},
    )
    assert completed.status_code == 200
    assert completed.json()["task"]["status"] == "done"


def test_kanban_task_panel_comments(_isolate_hermes_home):
    client = _client()

    created = client.post("/api/kanban/tasks", json={"title": "Needs context"})
    assert created.status_code == 200
    task_id = created.json()["task"]["id"]

    commented = client.post(
        f"/api/kanban/tasks/{task_id}/comment",
        json={"author": "desktop", "body": "Remember this"},
    )
    assert commented.status_code == 200

    detail = client.get(f"/api/kanban/tasks/{task_id}")
    assert detail.status_code == 200
    comments = detail.json()["comments"]
    assert len(comments) == 1
    assert comments[0]["author"] == "desktop"
    assert comments[0]["body"] == "Remember this"
