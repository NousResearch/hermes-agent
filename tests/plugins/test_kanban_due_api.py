"""Kanban dashboard API: task due dates, through the mounted plugin router.

The dashboard UI sends exactly three shapes, exercised here as the contract: its
``datetime-local`` value as ``due_at`` on create and edit, and ``clear_due: true`` to clear.
Rendering is server-side (``due_state`` / ``due_at_local`` / ``due_at_iso``), so the bundle
does no date math. A refused PATCH must change nothing, however many fields it carries.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from contextlib import closing as kbc_closing
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import hermes_time
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

REPO = Path(__file__).resolve().parents[2]
API = "/api/plugins/kanban"
# 2030-12-01 09:30 in New York (EST, UTC-5) == 14:30Z (far enough ahead to stay "later").
DEC1_0930_NY = 1922365800


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_TIMEZONE", "America/New_York")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    hermes_time.reset_cache()
    kb.init_db()
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_due_test", REPO / "plugins" / "kanban" / "dashboard" / "plugin_api.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    app = FastAPI()
    app.include_router(mod.router, prefix=API)
    yield TestClient(app)
    hermes_time.reset_cache()


def _snapshot(task_id: str):
    """The whole task row plus its event and run counts."""
    with kbc.connect_closing() as conn:
        row = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone())
        events = conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,)).fetchone()[0]
        runs = conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (task_id,)).fetchone()[0]
    return row, events, runs


# --- UI wiring: create / edit / clear ------------------------------------------

def test_ui_create_edit_clear_round_trip(client):
    # Create dialog: the datetime-local value as typed, read in the configured zone.
    task = client.post(f"{API}/tasks", json={"title": "dentist", "due_at": "2030-12-01T09:30"}).json()["task"]
    assert task["due_at"] == DEC1_0930_NY
    assert task["due_at_local"] == "2030-12-01T09:30"  # what the drawer editor is prefilled with
    assert task["due_at_iso"] == "2030-12-01T09:30:00-05:00"
    assert task["due_state"] == "later"
    tid = task["id"]

    # Drawer editor "Save": the edited datetime-local value.
    task = client.patch(f"{API}/tasks/{tid}", json={"due_at": "2030-12-02T08:00"}).json()["task"]
    assert (task["due_at"], task["due_at_local"]) == (DEC1_0930_NY + 22.5 * 3600, "2030-12-02T08:00")

    # A PATCH without due fields leaves the date alone.
    assert client.patch(f"{API}/tasks/{tid}", json={"priority": 2}).json()["task"]["due_at_local"] == "2030-12-02T08:00"

    # Drawer editor "Clear" (or Save with an emptied input).
    task = client.patch(f"{API}/tasks/{tid}", json={"clear_due": True}).json()["task"]
    assert [task[k] for k in ("due_at", "due_at_local", "due_at_iso", "due_state")] == [None] * 4
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).due_at is None


def test_board_cards_carry_server_computed_due_state(client):
    now = int(time.time())
    late = client.post(f"{API}/tasks", json={"title": "late", "due_at": now - 60}).json()["task"]["id"]
    done = client.post(f"{API}/tasks", json={"title": "done late", "due_at": now - 3 * 86400}).json()["task"]["id"]
    client.patch(f"{API}/tasks/{done}", json={"status": "done", "result": "ok"})
    client.post(f"{API}/tasks", json={"title": "later", "due_at": now + 40 * 86400})
    client.post(f"{API}/tasks", json={"title": "undated"})
    cards = {t["title"]: t for col in client.get(f"{API}/board").json()["columns"] for t in col["tasks"]}
    assert {k: c["due_state"] for k, c in cards.items()} == {
        "late": "overdue", "done late": "later", "later": "later", "undated": None}
    assert client.get(f"{API}/tasks/{late}").json()["task"]["due_state"] == "overdue"


# --- Input validation: nothing is written on refusal ---------------------------

@pytest.mark.parametrize("bad", [
    True, False, 1.5, int(time.time() * 1000), "next week", "2026-03-08T02:30", "in 1e308d", "in 1e305d"])
def test_bad_due_on_create_is_refused_and_creates_nothing(client, bad):
    """``true`` is not epoch 1 (strict types, 422); a JS millisecond timestamp, prose, and a
    New York wall time skipped by spring-forward are 400s."""
    r = client.post(f"{API}/tasks", json={"title": "x", "due_at": bad})
    assert r.status_code in (400, 422), r.text
    with kbc.connect_closing() as conn:
        assert kb.list_tasks(conn) == []


def test_ambiguous_fall_back_time_needs_an_offset(client):
    r = client.post(f"{API}/tasks", json={"title": "x", "due_at": "2026-11-01T01:30"})
    assert r.status_code == 400 and "occurs twice" in r.json()["detail"]
    r = client.post(f"{API}/tasks", json={"title": "x", "due_at": "2026-11-01T01:30-05:00"})
    assert r.status_code == 200 and r.json()["task"]["due_at_iso"] == "2026-11-01T01:30:00-05:00"


def _full_patch(**overrides):
    body = {"title": "renamed", "body": "new body", "priority": 7, "assignee": "someone-else",
            "due_at": "2026-12-24T18:00", "model_override": "some-model", "reasoning_effort": "high"}
    body.update(overrides)
    return body


@pytest.fixture
def gated_task(client):
    """A todo card behind an open parent, carrying a value in every field a PATCH touches."""
    parent = client.post(f"{API}/tasks", json={"title": "parent"}).json()["task"]["id"]
    tid = client.post(f"{API}/tasks", json={
        "title": "child", "body": "original", "priority": 1, "assignee": "owner", "parents": [parent],
        "due_at": "2026-12-01T09:30", "model_override": "orig-model", "reasoning_effort": "low",
    }).json()["task"]["id"]
    return tid


@pytest.mark.parametrize("patch, code", [
    # Refused by live state (open parent) after preflight: the status step runs first.
    (_full_patch(status="ready"), 409),
    # Refused by preflight, with every other field valid.
    (_full_patch(status="running"), 400),
    (_full_patch(status="sideways"), 400),
    (_full_patch(title="   "), 400),
    (_full_patch(due_at="2026-03-08T02:30"), 400),
    (_full_patch(due_at="whenever"), 400),
    (_full_patch(due_at="in 1e308d"), 400),
    (_full_patch(reasoning_effort="extreme"), 400),
    (_full_patch(model_override="  ", provider_override="openrouter"), 400),
    (_full_patch(status="archived"), 400),  # overrides on a card being archived
    (_full_patch(due_at=True), 422),
])
def test_refused_combined_patch_changes_nothing(client, gated_task, patch, code):
    before = _snapshot(gated_task)
    r = client.patch(f"{API}/tasks/{gated_task}", json=patch)
    assert r.status_code == code, r.text
    assert _snapshot(gated_task) == before


def test_reassigning_a_claimed_card_is_refused_before_any_write(client):
    tid = client.post(f"{API}/tasks", json={"title": "busy", "assignee": "worker"}).json()["task"]["id"]
    with kbc.connect_closing() as conn:
        assert kb.claim_task(conn, tid)
    before = _snapshot(tid)
    r = client.patch(f"{API}/tasks/{tid}", json=_full_patch(model_override=None, reasoning_effort=None))
    assert r.status_code == 409, r.text
    assert _snapshot(tid) == before


def test_accepted_combined_patch_applies_every_field(client, gated_task):
    r = client.patch(f"{API}/tasks/{gated_task}", json=_full_patch(status="triage"))
    assert r.status_code == 200, r.text
    with kbc.connect_closing() as conn:
        t = kb.get_task(conn, gated_task)
    assert (t.status, t.title, t.body, t.priority, t.assignee, t.model_override, t.reasoning_effort) == (
        "triage", "renamed", "new body", 7, "someone-else", "some-model", "high")
    assert r.json()["task"]["due_at_local"] == "2026-12-24T18:00"


# --- Reassign + dispatchable move: no tick ever sees the card ready under the old assignee ---

@pytest.fixture
def dispatch_at_every_boundary(client, monkeypatch):
    """Run a real dispatcher tick, on its own connection, right before EVERY transaction the
    dashboard request opens (a SQLite trace on the request's connection fires at each
    ``BEGIN``), so any state committed between two of the request's writes is dispatchable
    exactly as a racing gateway tick would see it. Yields the ``(task id, assignee)`` spawns."""
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in ("alice", "bob"))
    spawned: list[tuple[str, str]] = []
    errors: list[BaseException] = []
    real_connect = kbc.connect

    def tick(statement: str) -> None:
        if not statement.lstrip().upper().startswith("BEGIN"):
            return
        # SQLite swallows exceptions raised in a trace callback: record them instead. A fresh
        # connection per tick, because the endpoint runs on a worker thread.
        try:
            with kbc_closing(real_connect()) as dispatcher:
                kbd.dispatch_once(dispatcher, dry_run=False,
                                  spawn_fn=lambda task, ws, **_kw: spawned.append((task.id, task.assignee)) or 4242)
        except BaseException as exc:
            errors.append(exc)

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(tick)
        return conn

    monkeypatch.setattr(kbc, "connect", traced_connect)
    yield spawned
    assert not errors, errors


def _triage_card(**extra) -> str:
    with kbc.connect_closing() as conn:
        return kb.create_task(conn, title="errand", assignee="alice", triage=True, **extra)


def test_reassign_and_ready_never_dispatches_old_assignee(client, dispatch_at_every_boundary):
    tid = _triage_card()
    r = client.patch(f"{API}/tasks/{tid}", json={"status": "ready", "assignee": "bob", "due_at": "2030-12-01T09:30"})
    assert r.status_code == 200, r.text
    assert (r.json()["task"]["status"], r.json()["task"]["assignee"]) in {("ready", "bob"), ("running", "bob")}
    # Ticks ran at every boundary; the first one after the PATCH committed spawned bob, none alice.
    assert dispatch_at_every_boundary == [(tid, "bob")]


def test_refused_reassign_and_ready_rolls_back_whole_patch(client, dispatch_at_every_boundary):
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="nobody-profile")
    tid = _triage_card(parents=(parent,))
    with kbc.connect_closing() as conn:
        kb.edit_task(conn, tid, due_at=DEC1_0930_NY)
        conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (tid,))
    before = _snapshot(tid)
    r = client.patch(f"{API}/tasks/{tid}", json={
        "status": "ready", "assignee": "bob", "title": "renamed", "due_at": "2030-12-24T18:00"})
    assert r.status_code == 409, r.text
    assert "parent" in r.json()["detail"]
    assert _snapshot(tid) == before  # assignee still alice, nothing else written
    assert all(task_id != tid for task_id, _who in dispatch_at_every_boundary)


# --- The shipped bundle's create / edit / clear wiring, executed (not parsed) -------------

BUNDLE = REPO / "plugins" / "kanban" / "dashboard" / "dist" / "index.js"
RUNTIME = Path(__file__).parent / "fixtures" / "kanban_bundle_runtime.js"


def _run_bundle(client, tmp_path, scenario: str, task_id: str = "") -> list[dict]:
    """Mount the real bundle on real GET payloads from this router, drive ``scenario``, and
    return the mutating requests it sent (see the runtime's header)."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    paths = ["/board", "/config", "/boards", "/assignees", "/model-options"]
    if task_id:
        paths.append(f"/tasks/{task_id}")
    get = {f"{API}{p}": r.json() for p in paths if (r := client.get(f"{API}{p}")).status_code == 200}
    responses = tmp_path / "responses.json"
    responses.write_text(json.dumps({"get": get, "mutation": {}}))
    out = subprocess.run([node, str(RUNTIME), str(BUNDLE), str(responses), scenario],
                         capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0, out.stderr
    result = json.loads(out.stdout)
    assert result["errors"] == [], "\n".join(result["errors"])
    return result["requests"]


def _replay(client, request: dict) -> dict:
    r = client.request(request["method"], request["path"], json=request["body"])
    assert r.status_code == 200, r.text
    return r.json()["task"]


def test_bundle_create_dialog_sends_typed_due_date(client, tmp_path):
    sent = _run_bundle(client, tmp_path, "create|todo|Pay rent|2030-12-01T09:30")
    assert [(r["method"], r["path"]) for r in sent] == [("POST", f"{API}/tasks")]
    assert sent[0]["body"]["title"] == "Pay rent"
    assert sent[0]["body"]["due_at"] == "2030-12-01T09:30"
    task = _replay(client, sent[0])
    assert (task["due_at"], task["due_at_local"]) == (DEC1_0930_NY, "2030-12-01T09:30")


def test_bundle_drawer_edits_and_clears_due_date(client, tmp_path):
    tid = client.post(f"{API}/tasks", json={"title": "dentist", "due_at": "2030-12-01T09:30"}).json()["task"]["id"]

    sent = _run_bundle(client, tmp_path, f"edit|{tid}|2030-12-24T18:00", tid)
    assert sent == [{"method": "PATCH", "path": f"{API}/tasks/{tid}", "body": {"due_at": "2030-12-24T18:00"}}]
    assert _replay(client, sent[0])["due_at_local"] == "2030-12-24T18:00"

    for scenario in (f"clear|{tid}", f"empty-save|{tid}"):  # Clear button, or Save on an emptied input
        sent = _run_bundle(client, tmp_path, scenario, tid)
        assert sent == [{"method": "PATCH", "path": f"{API}/tasks/{tid}", "body": {"clear_due": True}}]
        if scenario.startswith("clear"):
            assert _replay(client, sent[0])["due_at"] is None
            client.patch(f"{API}/tasks/{tid}", json={"due_at": "2030-12-01T09:30"})  # restore for the next run
