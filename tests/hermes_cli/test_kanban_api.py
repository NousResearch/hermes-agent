from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_api import router


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kanban_db._INITIALIZED_PATHS.clear()
    app = FastAPI()
    app.include_router(router, prefix="/api/plugins/kanban/v1")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def transcripts_on() -> None:
    from hermes_constants import get_hermes_home

    (get_hermes_home() / "config.yaml").write_text(_EXPOSE_TRANSCRIPTS)


_EXPOSE_TRANSCRIPTS = "kanban:\n  api_expose_transcripts: true\n"


def _worker_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, expose: bool = True) -> Path:
    """Home of the profile that runs the workers; every profile name resolves to it."""
    from hermes_cli import profiles as profiles_mod

    home = tmp_path / "worker-home"
    home.mkdir()
    if expose:
        (home / "config.yaml").write_text(_EXPOSE_TRANSCRIPTS)
    monkeypatch.setattr(profiles_mod, "resolve_profile_env", lambda name: str(home))
    return home


def _create(client: TestClient, **overrides) -> dict:
    payload = {
        "title": "External operation",
        "body": "private execution instructions",
        "tenant": "ops",
        "priority": 5,
        "idempotency_key": "operation-001",
    }
    payload.update(overrides)
    response = client.post("/api/plugins/kanban/v1/tasks", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_health_capabilities_and_safe_board_dtos(client: TestClient) -> None:
    health = client.get("/api/plugins/kanban/v1/health")
    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "service": "hermes-kanban",
        "api_version": "1",
        "current_board": "default",
    }

    capabilities = client.get("/api/plugins/kanban/v1/capabilities")
    assert capabilities.status_code == 200
    body = capabilities.json()
    assert body["idempotent_task_creation"] is True
    assert body["profile_execution"] is False
    assert "ready" in body["task_statuses"]

    boards = client.get("/api/plugins/kanban/v1/boards").json()
    assert boards["current"] == "default"
    assert boards["boards"][0]["id"] == "default"
    assert "db_path" not in boards["boards"][0]
    assert "default_workdir" not in boards["boards"][0]

    detail = client.get("/api/plugins/kanban/v1/boards/Default")
    assert detail.status_code == 200
    assert detail.json()["board"]["id"] == "default"


def test_create_list_get_patch_are_idempotent_and_sanitized(client: TestClient) -> None:
    first = _create(client)
    task_id = first["task"]["id"]
    assert first["created"] is True
    assert first["task"]["links"] == {"parents": [], "children": []}

    repeated = client.post(
        "/api/plugins/kanban/v1/tasks",
        headers={"Idempotency-Key": "operation-001"},
        json={"title": "ignored on replay"},
    )
    assert repeated.status_code == 200
    assert repeated.json()["created"] is False
    assert repeated.json()["task"]["id"] == task_id

    listed = client.get("/api/plugins/kanban/v1/tasks", params={"tenant": "ops"})
    assert listed.status_code == 200
    assert listed.json()["count"] == 1

    detail = client.get(f"/api/plugins/kanban/v1/tasks/{task_id}")
    assert detail.status_code == 200
    task = detail.json()["task"]
    assert task["title"] == "External operation"
    forbidden = {
        "body", "result", "workspace_path", "branch_name", "claim_lock",
        "worker_pid", "session_id", "idempotency_key", "last_failure_error",
    }
    assert forbidden.isdisjoint(task)
    assert "private execution instructions" not in detail.text

    patched = client.patch(
        f"/api/plugins/kanban/v1/tasks/{task_id}",
        json={"title": "Renamed operation", "priority": 9, "assignee": "Worker-One"},
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["task"]["title"] == "Renamed operation"
    assert patched.json()["task"]["priority"] == 9
    assert patched.json()["task"]["assignee"] == "worker-one"


def test_links_actions_and_observability_are_sanitized(client: TestClient) -> None:
    parent_id = _create(client, title="Parent", idempotency_key="parent-001")["task"]["id"]
    child_id = _create(client, title="Child", idempotency_key="child-001")["task"]["id"]

    linked = client.post(f"/api/plugins/kanban/v1/tasks/{parent_id}/links/{child_id}")
    assert linked.status_code == 200, linked.text
    child = client.get(f"/api/plugins/kanban/v1/tasks/{child_id}").json()["task"]
    assert child["links"]["parents"] == [parent_id]
    assert child["status"] == "todo"

    comment = client.post(
        f"/api/plugins/kanban/v1/tasks/{parent_id}/comment",
        json={"body": "private operator note"},
    )
    assert comment.status_code == 201
    assert "private operator note" not in comment.text

    completed = client.post(
        f"/api/plugins/kanban/v1/tasks/{parent_id}/complete",
        json={"summary": "private completion handoff"},
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["task"]["status"] == "done"
    assert "private completion handoff" not in completed.text

    events = client.get(f"/api/plugins/kanban/v1/tasks/{parent_id}/events")
    assert events.status_code == 200
    assert events.json()["events"]
    assert all("payload" not in event for event in events.json()["events"])
    assert "private" not in events.text

    with kbc.connect(board="default") as conn:
        now = 1_700_000_000
        with kanban_db.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs "
                "(task_id, profile, status, started_at, ended_at, outcome, summary, metadata, error) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (parent_id, "private-profile", "done", now, now + 1, "completed",
                 "raw private summary", json.dumps({"workspace": "/srv/private/work"}),
                 "secret error"),
            )

    runs = client.get(f"/api/plugins/kanban/v1/tasks/{parent_id}/runs")
    assert runs.status_code == 200
    # ``profile`` is part of the external contract (execution attribution —
    # profile names already travel on task.assignee); summary, metadata,
    # error, and claim/PID machinery stay private.
    matching = [
        r for r in runs.json()["runs"] if r["profile"] == "private-profile"
    ]
    assert len(matching) == 1, runs.text
    run = matching[0]
    assert set(run) == {
        "id", "profile", "status", "outcome", "started_at", "ended_at",
    }
    assert "raw private summary" not in runs.text
    assert "secret error" not in runs.text
    assert "/srv/private/work" not in runs.text

    log_path = kanban_db.worker_log_path(parent_id, board="default")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "working in /srv/private/worktree\nAuthorization: Bearer secret-value-1234567890\n"
        "retrying with Bearer bare-opaque-token-42\n"
        "GET https://example.test?access_token=query-token-42\n"
        "cat /secret.txt from /workspace\n"
        "copy \\\\fileserver\\private\\report.txt C:\\Users\\John Doe\\private\\ledger.txt\n"
        "  hermes --resume 20261004_101112_abc123\n",
        encoding="utf-8",
    )
    log_response = client.get(f"/api/plugins/kanban/v1/tasks/{parent_id}/log")
    assert log_response.status_code == 200
    log_body = log_response.json()
    assert "path" not in log_body
    assert "/srv/private" not in log_body["excerpt"]
    assert "secret-value-1234567890" not in log_body["excerpt"]
    assert "bare-opaque-token-42" not in log_body["excerpt"]
    assert "query-token-42" not in log_body["excerpt"]
    assert "/secret.txt" not in log_body["excerpt"] and "/workspace" not in log_body["excerpt"]
    for leaked in ("fileserver", "report.txt", "Doe", "ledger.txt", "20261004_101112_abc123"):
        assert leaked not in log_body["excerpt"], leaked

    unlinked = client.delete(f"/api/plugins/kanban/v1/tasks/{parent_id}/links/{child_id}")
    assert unlinked.status_code == 200
    assert unlinked.json()["removed"] is True

    blocked = client.post(
        f"/api/plugins/kanban/v1/tasks/{child_id}/block",
        json={"reason": "private blocker", "kind": "needs_input"},
    )
    assert blocked.status_code == 200
    assert blocked.json()["task"]["status"] == "blocked"
    assert "private blocker" not in blocked.text

    assert client.post(f"/api/plugins/kanban/v1/tasks/{child_id}/unblock").status_code == 200
    assert client.post(f"/api/plugins/kanban/v1/tasks/{child_id}/archive").status_code == 200


def test_operator_routes_keep_their_paths_beside_v1() -> None:
    """Existing dashboard/desktop clients keep working: the sanitized API is added under
    ``/v1`` and never shadows an operator route."""
    plugin_path = Path(__file__).parents[2] / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    module_name = "test_hermes_dashboard_plugin_kanban"
    spec = importlib.util.spec_from_file_location(module_name, plugin_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        paths = {getattr(route, "path", "") for route in module.router.routes}
    finally:
        sys.modules.pop(module_name, None)

    assert {"/v1/health", "/v1/tasks", "/v1/tasks/{task_id}"} <= paths
    assert {"/board", "/tasks", "/tasks/{task_id}", "/events"} <= paths


def test_idempotency_key_is_unique_among_live_tasks(client: TestClient) -> None:
    """A duplicate live idempotency_key insert is rejected by the UNIQUE index.

    Simulates the TOCTOU race by inserting a second row directly (bypassing
    the SELECT-before-INSERT guard) — the partial UNIQUE index must refuse it.
    """
    created = _create(client, idempotency_key="race-key")
    task_id = created["task"]["id"]

    with kbc.connect(board="default") as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        with pytest.raises(sqlite3.IntegrityError):
            with kanban_db.write_txn(conn):
                conn.execute(
                    "INSERT INTO tasks "
                    "(id, title, status, created_at, workspace_kind, idempotency_key) "
                    "VALUES (?, ?, ?, ?, 'scratch', ?)",
                    ("t_duplicate", "dup", row["status"], row["created_at"], "race-key"),
                )

    # Archiving the live task frees the key: the partial index excludes
    # archived rows, so a fresh create with the same key is allowed (200/false
    # is only for a *live* duplicate; here a brand-new task is created).
    assert client.post(f"/api/plugins/kanban/v1/tasks/{task_id}/archive").status_code == 200
    reused = client.post(
        "/api/plugins/kanban/v1/tasks",
        json={"title": "after archive", "idempotency_key": "race-key"},
    )
    assert reused.status_code == 201, reused.text
    assert reused.json()["created"] is True
    assert reused.json()["task"]["id"] != task_id


def test_create_task_returns_existing_on_idempotency_race(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A create deduped by the storage layer answers 200 / created=false.

    ``kanban_db.create_task_idempotent`` absorbs the UNIQUE-index race (see
    its tests in test_kanban_db.py) and reports ``created=False`` for the
    loser; the adapter must map that to HTTP 200 with the winner task, never
    201 or a 500.
    """
    original = _create(client, idempotency_key="winner-key")
    original_id = original["task"]["id"]

    monkeypatch.setattr(
        kanban_db,
        "create_task_idempotent",
        lambda conn, **kwargs: (original_id, False),
    )

    raced = client.post(
        "/api/plugins/kanban/v1/tasks",
        json={"title": "raced", "idempotency_key": "winner-key"},
    )
    assert raced.status_code == 200, raced.text
    assert raced.json()["created"] is False
    assert raced.json()["task"]["id"] == original_id


def test_link_missing_task_is_404_not_400(client: TestClient) -> None:
    real_id = _create(client, idempotency_key="link-real")["task"]["id"]

    missing_parent = client.post(
        f"/api/plugins/kanban/v1/tasks/t_nope/links/{real_id}"
    )
    assert missing_parent.status_code == 404, missing_parent.text

    missing_child = client.post(
        f"/api/plugins/kanban/v1/tasks/{real_id}/links/t_nope"
    )
    assert missing_child.status_code == 404, missing_child.text

    # Unlink stays 404 for a missing endpoint (unchanged behaviour).
    missing_unlink = client.delete(
        f"/api/plugins/kanban/v1/tasks/{real_id}/links/t_nope"
    )
    assert missing_unlink.status_code == 404, missing_unlink.text


def test_complete_without_evidence_is_a_client_error(client: TestClient) -> None:
    """The storage layer's empty-completion guard surfaces as a 400, not a 500."""
    task_id = _create(client, idempotency_key="empty-complete")["task"]["id"]

    rejected = client.post(f"/api/plugins/kanban/v1/tasks/{task_id}/complete")

    assert rejected.status_code == 400, rejected.text
    assert "no result or summary evidence" in rejected.json()["detail"]
    assert client.get(f"/api/plugins/kanban/v1/tasks/{task_id}").json()["task"][
        "status"
    ] != "done"


def test_patch_rejects_edits_to_completed_task(client: TestClient) -> None:
    task_id = _create(client, idempotency_key="done-edit")["task"]["id"]
    assert client.post(
        f"/api/plugins/kanban/v1/tasks/{task_id}/complete", json={"summary": "shipped"}
    ).status_code == 200

    rejected = client.patch(
        f"/api/plugins/kanban/v1/tasks/{task_id}", json={"title": "too late"}
    )
    assert rejected.status_code == 409, rejected.text
    # The card text is unchanged.
    assert client.get(f"/api/plugins/kanban/v1/tasks/{task_id}").json()["task"][
        "title"
    ] == "External operation"


def test_patch_rejects_edits_to_archived_task(client: TestClient) -> None:
    task_id = _create(client, idempotency_key="archived-edit")["task"]["id"]
    assert client.post(
        f"/api/plugins/kanban/v1/tasks/{task_id}/complete", json={"summary": "shipped"}
    ).status_code == 200
    assert client.post(f"/api/plugins/kanban/v1/tasks/{task_id}/archive").status_code == 200

    rejected = client.patch(
        f"/api/plugins/kanban/v1/tasks/{task_id}", json={"body": "amend history"}
    )
    assert rejected.status_code == 409, rejected.text


def test_patch_mixing_assignee_and_edit_is_all_or_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal transition landing between the assignee write and the title guard
    must roll the assignee back too — never 409 with the reassignment persisted."""
    task_id = _create(client, idempotency_key="atomic-patch")["task"]["id"]
    real_append = kanban_db._append_event

    def racing_append(conn, tid, kind, payload=None, **kwargs):
        real_append(conn, tid, kind, payload, **kwargs)
        if kind == "assigned":  # another actor completes the task right here
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (tid,))

    monkeypatch.setattr(kanban_db, "_append_event", racing_append)
    rejected = client.patch(
        f"/api/plugins/kanban/v1/tasks/{task_id}",
        json={"assignee": "worker-two", "title": "renamed mid-flight"},
    )
    assert rejected.status_code == 409, rejected.text

    monkeypatch.setattr(kanban_db, "_append_event", real_append)
    task = client.get(f"/api/plugins/kanban/v1/tasks/{task_id}").json()["task"]
    assert task["assignee"] is None
    assert task["title"] == "External operation"
    assert task["status"] != "done"


def test_events_and_runs_limit_returns_most_recent_in_order(client: TestClient) -> None:
    task_id = _create(client, idempotency_key="limit-key")["task"]["id"]

    with kbc.connect(board="default") as conn:
        with kanban_db.write_txn(conn):
            for i in range(5):
                conn.execute(
                    "INSERT INTO task_events "
                    "(task_id, run_id, kind, payload, created_at) "
                    "VALUES (?, NULL, ?, NULL, ?)",
                    (task_id, f"evt{i}", 2_000_000_000 + i),
                )
            for i in range(4):
                conn.execute(
                    "INSERT INTO task_runs "
                    "(task_id, status, started_at, ended_at, outcome) "
                    "VALUES (?, 'done', ?, ?, 'completed')",
                    (task_id, 2_000_000_000 + i, 2_000_000_000 + i + 1),
                )

    events = client.get(
        f"/api/plugins/kanban/v1/tasks/{task_id}/events", params={"limit": 3}
    )
    assert events.status_code == 200
    kinds = [e["kind"] for e in events.json()["events"]]
    # The three newest events, oldest-first.
    assert kinds == ["evt2", "evt3", "evt4"]
    assert events.json()["count"] == 3

    runs = client.get(
        f"/api/plugins/kanban/v1/tasks/{task_id}/runs", params={"limit": 2}
    )
    assert runs.status_code == 200
    starts = [r["started_at"] for r in runs.json()["runs"]]
    assert starts == [2_000_000_002, 2_000_000_003]
    assert runs.json()["count"] == 2


def test_error_detail_is_sanitized(client: TestClient) -> None:
    # An invalid board slug's raw ValueError text (regex description) must not
    # reach the client; a stable generic detail is returned instead.
    bad_board = client.get(
        "/api/plugins/kanban/v1/tasks", params={"board": "Bad Slug!!"}
    )
    assert bad_board.status_code == 400, bad_board.text
    assert bad_board.json()["detail"] == "invalid board id"
    assert "1-64 chars" not in bad_board.text

    # A known-safe validation message is still surfaced verbatim (useful to
    # the caller, carries no internal detail).
    unknown_parent = client.post(
        "/api/plugins/kanban/v1/tasks",
        json={"title": "orphan", "parents": ["t_missing"]},
    )
    assert unknown_parent.status_code == 400, unknown_parent.text
    assert "unknown parent task" in unknown_parent.json()["detail"]


def test_task_dto_carries_created_by_attribution(client: TestClient) -> None:
    """created_by lets an external control plane attribute fan-out: it names
    the creating profile/surface, not just the assigned executor."""
    created = _create(client, idempotency_key="attribution-key")
    assert created["task"]["created_by"].startswith("api:")


def test_api_actor_can_never_be_a_profile_name(client: TestClient) -> None:
    """A worker skips comments authored under its own profile name, so the API's
    provenance must not be a name a profile could carry."""
    from hermes_cli.profiles import validate_profile_name

    created_by = _create(client, idempotency_key="actor-key")["task"]["created_by"]
    with pytest.raises(ValueError):
        validate_profile_name(created_by)


def test_runs_expose_executing_profile(client: TestClient) -> None:
    task_id = _create(
        client, idempotency_key="run-profile-key", assignee="worker-a"
    )["task"]["id"]
    with kbc.connect(board="default") as conn:
        with kanban_db.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs "
                "(task_id, profile, status, started_at, ended_at, outcome) "
                "VALUES (?, 'worker-a', 'done', 1, 2, 'completed')",
                (task_id,),
            )
    runs = client.get(f"/api/plugins/kanban/v1/tasks/{task_id}/runs")
    assert runs.status_code == 200
    assert runs.json()["runs"][0]["profile"] == "worker-a"


def test_profiles_roster_is_sanitized_and_sorted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from hermes_cli import profiles as profiles_mod

    fake = [
        SimpleNamespace(
            name="zeta",
            description="  Runs deploys.  ",
            model="secret-model",
            path="/private/profile/dir",
        ),
        SimpleNamespace(name="alpha", description="", model=None, path="/x"),
    ]
    monkeypatch.setattr(profiles_mod, "list_profiles", lambda: fake)

    resp = client.get("/api/plugins/kanban/v1/profiles")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 2
    # Name + description only, sorted by name; model/path never leak.
    assert body["profiles"] == [
        {"name": "alpha", "description": "", "has_description": False},
        {"name": "zeta", "description": "Runs deploys.", "has_description": True},
    ]
    assert "secret-model" not in resp.text
    assert "/private/profile/dir" not in resp.text

    caps = client.get("/api/plugins/kanban/v1/capabilities").json()
    assert caps["profiles_api"] is True
    assert caps["profile_execution"] is False


def test_profiles_roster_unavailable_is_503(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli import profiles as profiles_mod

    def boom():
        raise RuntimeError("disk exploded at /private/some/path")

    monkeypatch.setattr(profiles_mod, "list_profiles", boom)
    resp = client.get("/api/plugins/kanban/v1/profiles")
    assert resp.status_code == 503
    assert resp.json()["detail"] == "profiles unavailable"
    assert "disk exploded" not in resp.text


def test_patch_edits_notify_observers_like_the_dashboard(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REST edits go through ``edit_task``'s write path: same events, and the
    ``on_kanban_task_updated`` observer fires once after commit with the board."""
    task_id = _create(client, idempotency_key="notify-edit")["task"]["id"]
    notified = []
    monkeypatch.setattr(
        kanban_db, "notify_task_updated",
        lambda conn, tid, fields, board=None: notified.append((tid, sorted(fields), board)),
    )
    patched = client.patch(
        f"/api/plugins/kanban/v1/tasks/{task_id}",
        json={"assignee": "worker-two", "title": "renamed", "priority": 7},
    )
    assert patched.status_code == 200, patched.text
    assert notified == [(task_id, ["assignee", "priority", "title"], kanban_db.get_current_board())]
    kinds = [e["kind"] for e in client.get(f"/api/plugins/kanban/v1/tasks/{task_id}/events").json()["events"]]
    assert {"assigned", "reprioritized", "edited"} <= set(kinds)


def test_transcript_streams_worker_session_while_running(
    client: TestClient, transcripts_on: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_state import SessionDB

    profile_home = _worker_home(tmp_path, monkeypatch)

    task_id = _create(client, title="привіт", idempotency_key="transcript-1")["task"]["id"]
    empty = client.get(f"/api/plugins/kanban/v1/tasks/{task_id}/transcript").json()
    assert empty["run_id"] is None and empty["messages"] == []

    with kbc.connect_closing() as conn:
        kanban_db.claim_task(conn, task_id)
        run = kanban_db.latest_run(conn, task_id)
        assert run is not None
        # Before the worker links its session: run known, nothing to show yet.
        pending = client.get(f"/api/plugins/kanban/v1/tasks/{task_id}/transcript").json()
        assert pending["run_status"] == "running" and pending["messages"] == []
        assert kanban_db.set_run_worker_session(conn, run.id, task_id, "sess-1")
        # First writer wins (background review / delegated agents).
        assert not kanban_db.set_run_worker_session(conn, run.id, task_id, "sess-2")

    db = SessionDB(profile_home / "state.db")
    db.create_session("sess-1", "kanban")
    db.append_message("sess-1", "system", content="secret system prompt")
    db.append_message("sess-1", "user", content="work kanban task")
    db.append_message(
        "sess-1", "assistant", content=None, reasoning="Користувач вітається",
        tool_calls=[{"id": "c1", "type": "function", "function": {
            "name": "kanban_show", "arguments": '{"key": "sk-abcdefghijklmnopqrstuvwxyz123456"}'}}],
    )
    db.append_message("sess-1", "tool", content="x" * 5000, tool_name="kanban_show", tool_call_id="c1")
    db.append_message("sess-1", "assistant", content="Привіт! <script>")
    db.close()

    url = f"/api/plugins/kanban/v1/tasks/{task_id}/transcript"
    body = client.get(url, params={"limit": 3}).json()
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]  # system dropped
    assert body["has_more"] is True
    call = body["messages"][1]
    assert call["reasoning"] == "Користувач вітається"
    assert call["tool_calls"][0]["name"] == "kanban_show"
    assert "sk-abcdefghijklmnop" not in call["tool_calls"][0]["arguments"]

    rest = client.get(url, params={"after_id": body["next_after_id"]}).json()
    assert [m["role"] for m in rest["messages"]] == ["tool", "assistant"]
    assert rest["messages"][0]["truncated"] is True
    assert len(rest["messages"][0]["content"]) == 4000
    assert rest["messages"][1]["content"] == "Привіт! <script>"
    assert rest["has_more"] is False

    tail = client.get(url, params={"after_id": rest["next_after_id"]}).json()
    assert tail["messages"] == [] and tail["next_after_id"] == rest["next_after_id"]

    # In-place compaction soft-archives the summarized steps; they stay in the transcript.
    db = SessionDB(profile_home / "state.db")
    db.archive_and_compact("sess-1", [{"role": "user", "content": "[summary]"}])
    db.close()
    for params in ({"limit": 500}, {"latest": True, "limit": 500}):
        contents = [m["content"] for m in client.get(url, params=params).json()["messages"]]
        assert "Привіт! <script>" in contents
        assert contents[-1] == "[summary]"

    assert client.get(url, params={"run_id": 999}).status_code == 404
    assert client.get("/api/plugins/kanban/v1/tasks/missing/transcript").status_code == 404


def test_agent_init_links_kanban_run_session(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.agent_init import _link_kanban_run_session

    task_id = _create(client, idempotency_key="transcript-link")["task"]["id"]
    with kbc.connect_closing() as conn:
        kanban_db.claim_task(conn, task_id)
        run_id = kanban_db.latest_run(conn, task_id).id

    _link_kanban_run_session("sess-outside")  # no kanban env → no-op
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    _link_kanban_run_session("sess-main")
    _link_kanban_run_session("sess-review")  # later agent in the same worker

    with kbc.connect_closing() as conn:
        assert kanban_db.get_run(conn, run_id).worker_session_id == "sess-main"


def test_transcript_latest_returns_newest_steps(
    client: TestClient, transcripts_on: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_state import SessionDB

    home = _worker_home(tmp_path, monkeypatch)
    task_id = _create(client, idempotency_key="transcript-latest")["task"]["id"]
    with kbc.connect_closing() as conn:
        kanban_db.claim_task(conn, task_id)
        run = kanban_db.latest_run(conn, task_id)
        kanban_db.set_run_worker_session(conn, run.id, task_id, "s-root")

    db = SessionDB(home / "state.db")
    db.create_session("s-root", "kanban")
    for i in range(3):
        db.append_message("s-root", "assistant", content=f"root {i}")
    db.end_session("s-root", "compression")
    db.create_session("s-cont", "kanban", parent_session_id="s-root")
    db.append_message("s-cont", "assistant", content="cont 0")
    db.close()

    url = f"/api/plugins/kanban/v1/tasks/{task_id}/transcript"
    body = client.get(url, params={"latest": "true", "limit": 2}).json()
    assert [m["content"] for m in body["messages"]] == ["root 2", "cont 0"]
    assert body["has_more"] is True
    full = client.get(url).json()
    assert [m["content"] for m in full["messages"]] == ["root 0", "root 1", "root 2", "cont 0"]


def test_transcript_ignores_caller_supplied_metadata_session(
    client: TestClient, transcripts_on: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_state import SessionDB

    home = _worker_home(tmp_path, monkeypatch)
    db = SessionDB(home / "state.db")
    db.create_session("private-chat", "cli")
    db.append_message("private-chat", "user", content="not a kanban run")
    db.close()

    task_id = _create(client, idempotency_key="transcript-metadata")["task"]["id"]
    with kbc.connect_closing() as conn:
        kanban_db.claim_task(conn, task_id)
        kanban_db.complete_task(
            conn, task_id, summary="done", metadata={"worker_session_id": "private-chat"})

    body = client.get(f"/api/plugins/kanban/v1/tasks/{task_id}/transcript").json()
    assert body["messages"] == [] and "session_id" not in body


def test_transcript_is_off_by_default(client: TestClient) -> None:
    task_id = _create(client, idempotency_key="transcript-off")["task"]["id"]
    assert client.get(f"/api/plugins/kanban/v1/tasks/{task_id}/transcript").status_code == 404
    caps = client.get("/api/plugins/kanban/v1/capabilities").json()
    assert "transcript" not in caps["observability"]


def test_log_tail_is_cut_after_redaction(client: TestClient) -> None:
    """A tail window opening inside a secret hands the redactor a credential without the
    context it keys on: the ``Bearer`` prefix, or a PEM block's ``BEGIN`` line."""
    task_id = _create(client, idempotency_key="log-tail")["task"]["id"]
    log_path = kanban_db.worker_log_path(task_id, board="default")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("Authorization: Bearer opaque-credential-0123456789", encoding="utf-8")
    for tail_bytes in (10, 30, 40):
        body = client.get(
            f"/api/plugins/kanban/v1/tasks/{task_id}/log", params={"tail_bytes": tail_bytes}
        ).json()
        assert "0123456789" not in body["excerpt"], tail_bytes


def test_actions_fire_lifecycle_hooks_for_the_requested_board(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
    key_body = "".join(f"KEYBODY{line:02d}{'A' * 55}\n" for line in range(20))
    log_path.write_text(
        "starting\n-----BEGIN PRIVATE KEY-----\n" + key_body + "-----END PRIVATE KEY-----\ndone\n",
        encoding="utf-8")
    for tail_bytes in (40, 200, 700, 1300):
        body = client.get(
            f"/api/plugins/kanban/v1/tasks/{task_id}/log", params={"tail_bytes": tail_bytes}
        ).json()
        assert "KEYBODY" not in body["excerpt"], tail_bytes
        assert len(body["excerpt"].encode()) <= tail_bytes

) -> None:
    kanban_db.create_board("ops")
    fired: list[tuple[str, object]] = []
    monkeypatch.setattr(
        kanban_db, "_fire_kanban_lifecycle_hook",
        lambda event, task_id, **fields: fired.append((event, fields.get("board"))),
    )
    base = "/api/plugins/kanban/v1/tasks"
    done = client.post(f"{base}?board=ops", json={"title": "to complete"}).json()["task"]["id"]
    stuck = client.post(f"{base}?board=ops", json={"title": "to block"}).json()["task"]["id"]
    assert client.post(f"{base}/{done}/complete?board=ops", json={"summary": "ok"}).status_code == 200
    assert client.post(f"{base}/{stuck}/block?board=ops", json={"reason": "wait"}).status_code == 200
    assert fired and {board for _event, board in fired} == {"ops"}


def _running_task_with_session(client: TestClient, key: str, session_id: str) -> str:
    task_id = _create(client, idempotency_key=key)["task"]["id"]
    with kbc.connect_closing() as conn:
        kanban_db.claim_task(conn, task_id)
        run = kanban_db.latest_run(conn, task_id)
        kanban_db.set_run_worker_session(conn, run.id, task_id, session_id)
    return task_id


def test_transcript_needs_the_worker_profiles_own_opt_in(
    client: TestClient, transcripts_on: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opt-in is read from the profile whose session is served, never from the profile the
    request selected: ``?profile=`` must not unlock another profile's worker."""
    from hermes_state import SessionDB

    home = _worker_home(tmp_path, monkeypatch, expose=False)
    task_id = _running_task_with_session(client, "transcript-owner", "s1")
    db = SessionDB(home / "state.db")
    db.create_session("s1", "kanban")
    db.append_message("s1", "assistant", content="worker output")
    db.close()

    url = f"/api/plugins/kanban/v1/tasks/{task_id}/transcript"
    assert client.get(url).status_code == 404
    (home / "config.yaml").write_text(_EXPOSE_TRANSCRIPTS)
    assert [m["content"] for m in client.get(url).json()["messages"]] == ["worker output"]


def test_transcript_steps_keep_their_uid_when_compaction_renumbers_them(
    client: TestClient, transcripts_on: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compaction re-inserts the steps it carries forward under fresh ids, so a poller gets a
    step it already read again; ``uid`` is what lets it recognise the repeat."""
    from hermes_state import SessionDB

    home = _worker_home(tmp_path, monkeypatch)
    task_id = _running_task_with_session(client, "transcript-uid", "s1")
    db = SessionDB(home / "state.db")
    db.create_session("s1", "kanban")
    db.append_message("s1", "assistant", content="step 1")
    db.append_message("s1", "assistant", content="step 2")

    url = f"/api/plugins/kanban/v1/tasks/{task_id}/transcript"
    first = client.get(url).json()
    seen = {m["uid"]: m["content"] for m in first["messages"]}
    assert len(seen) == 2 and all(seen)

    # "step 2" arrived during the slow summary call: it is re-sequenced after the summary.
    db.archive_and_compact(
        "s1", [{"role": "user", "content": "[summary]"}], watermark=first["messages"][0]["id"])
    db.close()
    again = client.get(url, params={"after_id": first["next_after_id"]}).json()["messages"]
    repeat = [m for m in again if m["content"] == "step 2"]
    assert repeat and seen[repeat[0]["uid"]] == "step 2"
