from __future__ import annotations

import importlib
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from hermes_state import SessionDB


def _seed_dashboard_data(hermes_home: Path) -> None:
    logs_dir = hermes_home / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    db = SessionDB(db_path=hermes_home / "state.db")
    db.create_session(session_id="s1", source="cli")
    db.create_task(
        task_id="t1",
        session_id="s1",
        status="completed",
        model_used="gpt-4o-mini",
        current_step="done",
        checkpoint_data={"phase": "final"},
        token_usage={"input": 10, "output": 20},
        error_info=None,
    )
    db.close()

    (logs_dir / "structured.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"event": "tool_result", "tool_name": "terminal", "duration_ms": 4200, "task_id": "t1"}),
                json.dumps({"event": "loop_detected", "tool_name": "terminal", "count": 2, "task_id": "t1"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _start_dashboard_server(dashboard_mod):
    server = HTTPServer(("127.0.0.1", 0), dashboard_mod.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _seed_project_dashboard_data(hermes_home: Path) -> None:
    logs_dir = hermes_home / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    db = SessionDB(db_path=hermes_home / "state.db")

    db.create_session(session_id="p1", source="cli")
    db.set_session_title("p1", "Alpha Project")
    db.create_task(
        task_id="p1-task-1",
        session_id="p1",
        status="running",
        model_used="gpt-4.1-mini",
        current_step="planning",
        checkpoint_data={"phase": "planning"},
        token_usage={"input": 40, "output": 12},
        error_info=None,
    )

    db.create_session(session_id="p2", source="telegram")
    db.set_session_title("p2", "Beta Project")
    db.create_task(
        task_id="p2-task-1",
        session_id="p2",
        status="completed",
        model_used="gpt-4o-mini",
        current_step="done",
        checkpoint_data={"phase": "final"},
        token_usage={"input": 60, "output": 20},
        error_info=None,
    )
    db.create_task(
        task_id="p2-task-2",
        session_id="p2",
        status="failed",
        model_used="gpt-4o-mini",
        current_step="retrying",
        checkpoint_data={"phase": "retry"},
        token_usage={"input": 24, "output": 8},
        error_info="tool timeout",
    )

    db.close()

    (logs_dir / "structured.jsonl").write_text("\n", encoding="utf-8")


def _seed_delete_dependency_data(hermes_home: Path) -> None:
    db = SessionDB(db_path=hermes_home / "state.db")
    db.create_session(session_id="parent", source="cli")
    db.create_session(session_id="child", source="cli", parent_session_id="parent")
    db.create_task(
        task_id="task-parent",
        session_id="parent",
        status="completed",
        model_used="gpt-4o-mini",
        current_step="done",
        checkpoint_data={"phase": "done"},
        token_usage={"input": 12, "output": 4},
        error_info=None,
    )
    db.close()

    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    (hermes_home / "logs" / "structured.jsonl").write_text("\n", encoding="utf-8")


def _seed_prune_dependency_data(hermes_home: Path) -> None:
    db = SessionDB(db_path=hermes_home / "state.db")

    db.create_session(session_id="keep", source="cli")
    db.end_session("keep", end_reason="done")
    db._conn.execute(
        "UPDATE sessions SET ended_at = ?, started_at = ? WHERE id = ?",
        (time.time(), time.time(), "keep"),
    )

    db.create_session(session_id="old", source="cli")
    db.end_session("old", end_reason="done")
    db._conn.execute(
        "UPDATE sessions SET ended_at = ?, started_at = ? WHERE id = ?",
        (time.time() - 100 * 86400, time.time() - 100 * 86400, "old"),
    )
    db.create_session(session_id="child", source="cli", parent_session_id="old")
    db.create_task(
        task_id="task-old",
        session_id="old",
        status="running",
        model_used="gpt-4o-mini",
        current_step="step-1",
        checkpoint_data={"step": 1},
        token_usage={"input": 5, "output": 2},
        error_info=None,
    )
    db._conn.commit()
    db.close()

    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    (hermes_home / "logs" / "structured.jsonl").write_text("\n", encoding="utf-8")


def _seed_chat_dashboard_data(hermes_home: Path) -> None:
    db = SessionDB(db_path=hermes_home / "state.db")
    now = time.time()

    for idx in range(55):
        sid = f"active-{idx}"
        db.create_session(session_id=sid, source="cli")
        db._conn.execute("UPDATE sessions SET started_at = ?, ended_at = NULL WHERE id = ?", (now - idx, sid))

    db.create_session(session_id="ended-new", source="cli")
    db.set_session_title("ended-new", "Newest closed thread")
    db.end_session("ended-new", end_reason="done")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, pinned_at = NULL WHERE id = ?",
        (now - 200, now - 190, "ended-new"),
    )
    db.append_message("ended-new", "user", "Need help polishing the chat dashboard")

    db.create_session(session_id="ended-old", source="cli")
    db.set_session_title("ended-old", "Pinned reference thread")
    db.end_session("ended-old", end_reason="done")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, pinned_at = ? WHERE id = ?",
        (now - 400, now - 390, now - 389, "ended-old"),
    )
    db.append_message("ended-old", "user", "This should stay pinned")

    db._conn.commit()
    db.close()

    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    (hermes_home / "logs" / "structured.jsonl").write_text("\n", encoding="utf-8")


def _http_get(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.read().decode("utf-8"), dict(resp.headers)


def _http_post_json(url: str, payload: dict, headers: dict | None = None):
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=req_headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.read().decode("utf-8"), dict(resp.headers)


def test_dashboard_serves_html_and_api_under_isolated_hermes_home(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    fake_home = tmp_path / "wrong-home"
    _seed_dashboard_data(hermes_home)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    import hermes_cli.dashboard as dashboard_mod
    import hermes_cli.ops as ops_mod

    importlib.reload(ops_mod)
    importlib.reload(dashboard_mod)

    server, thread = _start_dashboard_server(dashboard_mod)
    host, port = server.server_address
    base = f"http://{host}:{port}"

    try:
        status, html, _ = _http_get(base + "/")
        assert status == 200
        assert "id=\"error-banner\"" in html
        assert "error-banner-retry" in html
        assert "id=\"projects\"" in html
        assert "Projects" in html
        assert "id=\"chat-prune\"" in html
        assert "Prune old unpinned threads" in html
        assert "/api/projects?limit=8" in html
        assert "AbortController" in html
        assert "Promise.allSettled" in html
        assert "FETCH_TIMEOUT_MS = 2500" in html
        assert "refreshInFlight" in html

        status, body, _ = _http_get(base + "/api/summary")
        assert status == 200
        summary = json.loads(body)
        assert summary["total"] == 1
        assert summary["models_used"]["gpt-4o-mini"] == 1

        _status, _body, headers = _http_get(base + "/api/summary")
        assert headers.get("Access-Control-Allow-Origin") is None

        status, body, _ = _http_get(base + "/api/tasks?limit=5")
        tasks = json.loads(body)
        assert status == 200
        assert tasks[0]["task_id"] == "t1"

        status, body, _ = _http_get(base + "/api/events?limit=5")
        events = json.loads(body)
        assert status == 200
        assert events[0]["tool_name"] == "terminal"

        status, body, _ = _http_get(base + "/api/slow")
        slow = json.loads(body)
        assert status == 200
        assert slow[0]["duration_ms"] == 4200

        status, body, _ = _http_get(base + "/api/loops?limit=5")
        loops = json.loads(body)
        assert status == 200
        assert loops[0]["count"] == 2

        assert not (fake_home / ".hermes").exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_dashboard_renders_project_section_and_projects_api(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    fake_home = tmp_path / "wrong-home"
    _seed_project_dashboard_data(hermes_home)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    import hermes_cli.dashboard as dashboard_mod
    import hermes_cli.ops as ops_mod

    importlib.reload(ops_mod)
    importlib.reload(dashboard_mod)

    server, thread = _start_dashboard_server(dashboard_mod)
    host, port = server.server_address
    base = f"http://{host}:{port}"

    try:
        status, html, _ = _http_get(base + "/")
        assert status == 200
        assert "id=\"projects\"" in html
        assert "/api/projects?limit=8" in html

        status, body, _ = _http_get(base + "/api/projects?limit=8")
        assert status == 200
        projects = json.loads(body)
        assert len(projects) == 2
        assert projects[0]["display_name"] == "Beta Project"
        assert projects[0]["task_count"] == 2
        assert projects[0]["latest_task"]["status"] == "failed"
        assert projects[1]["display_name"] == "Alpha Project"
        assert projects[1]["latest_task"]["status"] == "running"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_dashboard_chat_list_metadata_and_rename(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    fake_home = tmp_path / "wrong-home"
    _seed_chat_dashboard_data(hermes_home)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    import hermes_cli.dashboard as dashboard_mod
    import hermes_cli.ops as ops_mod

    importlib.reload(ops_mod)
    importlib.reload(dashboard_mod)

    server, thread = _start_dashboard_server(dashboard_mod)
    host, port = server.server_address
    base = f"http://{host}:{port}"

    try:
        status, body, _ = _http_get(base + "/api/chats?limit=1")
        assert status == 200
        payload = json.loads(body)
        assert payload["meta"]["total_historical"] == 2
        assert payload["meta"]["visible_count"] == 1
        assert payload["meta"]["pinned_count"] == 1
        assert len(payload["items"]) == 1
        assert payload["items"][0]["session_id"] == "ended-old"

        status, body, _ = _http_post_json(base + "/api/chats/title", {"session_id": "ended-new", "title": "Refined chat title"})
        assert status == 200
        updated = json.loads(body)
        assert updated["session"]["title"] == "Refined chat title"

        try:
            _http_post_json(
                base + "/api/chats/title",
                {"session_id": "ended-new", "title": "evil"},
                headers={"Origin": "https://evil.example"},
            )
            raise AssertionError("expected cross-origin POST to be rejected")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403

        status, body, _ = _http_get(base + "/api/chats?limit=2")
        payload = json.loads(body)
        titles = {item["session_id"]: item["title"] for item in payload["items"]}
        assert titles["ended-new"] == "Refined chat title"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_dashboard_api_surfaces_endpoint_failures(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    fake_home = tmp_path / "wrong-home"
    hermes_home.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    import hermes_cli.dashboard as dashboard_mod
    import hermes_cli.ops as ops_mod

    importlib.reload(ops_mod)
    importlib.reload(dashboard_mod)

    monkeypatch.setattr(ops_mod, "recent_events", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    server, thread = _start_dashboard_server(dashboard_mod)
    host, port = server.server_address
    base = f"http://{host}:{port}"

    try:
        try:
            urllib.request.urlopen(base + "/api/events?limit=5", timeout=5)
            raise AssertionError("expected HTTPError")
        except urllib.error.HTTPError as exc:
            assert exc.code == 500
            body = exc.read().decode("utf-8")
            payload = json.loads(body)
            assert payload["error"] == "boom"
            assert "boom" in body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_dashboard_chat_delete_handles_tasks_and_children(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    fake_home = tmp_path / "wrong-home"
    _seed_delete_dependency_data(hermes_home)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", hermes_home / "state.db")

    import hermes_cli.dashboard as dashboard_mod
    import hermes_cli.ops as ops_mod

    importlib.reload(ops_mod)
    importlib.reload(dashboard_mod)

    server, thread = _start_dashboard_server(dashboard_mod)
    host, port = server.server_address
    base = f"http://{host}:{port}"

    try:
        status, body, _ = _http_post_json(base + "/api/chats/delete", {"session_id": "parent"})
        assert status == 200
        payload = json.loads(body)
        assert payload == {"ok": True}

        db = SessionDB(db_path=hermes_home / "state.db")
        try:
            assert db.get_session("parent") is None
            child = db.get_session("child")
            assert child is not None
            assert child["parent_session_id"] is None
            task_count = db._conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE session_id = ?", ("parent",)
            ).fetchone()[0]
            assert task_count == 0
        finally:
            db.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_dashboard_chat_prune_handles_tasks_and_children(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    fake_home = tmp_path / "wrong-home"
    _seed_prune_dependency_data(hermes_home)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", hermes_home / "state.db")

    import hermes_cli.dashboard as dashboard_mod
    import hermes_cli.ops as ops_mod

    importlib.reload(ops_mod)
    importlib.reload(dashboard_mod)

    server, thread = _start_dashboard_server(dashboard_mod)
    host, port = server.server_address
    base = f"http://{host}:{port}"

    try:
        status, body, _ = _http_post_json(base + "/api/chats/prune", {"keep": 1})
        assert status == 200
        payload = json.loads(body)
        assert payload == {"ok": True, "deleted": 1}

        db = SessionDB(db_path=hermes_home / "state.db")
        try:
            assert db.get_session("old") is None
            assert db.get_session("keep") is not None
            child = db.get_session("child")
            assert child is not None
            assert child["parent_session_id"] is None
            task_count = db._conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE session_id = ?", ("old",)
            ).fetchone()[0]
            assert task_count == 0
        finally:
            db.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
