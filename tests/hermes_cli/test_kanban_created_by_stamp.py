"""Regression tests: ``tasks.created_by`` is never NULL on kanban task creation.

Every creation surface either stamps an explicit author (dashboard session
identity, desktop-via-CLI profile, worker profile identity) or the DB layer
stamps a non-human machine-vocabulary value so unattributed cards stay
distinguishable from human-authored ones. Same server-stamped-identity
family as the comment-author fix (#102693 / #102716).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _created_by(task_id: str) -> str | None:
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None, f"task vanished: {task_id}"
        return task.created_by


def _load_dashboard_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_createdby_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Programmatic create (the raw insert path: scripts, fixtures, any surface
# that calls kanban_db.create_task without knowing the author)
# ---------------------------------------------------------------------------


def test_programmatic_create_without_created_by_never_inserts_null(kanban_home):
    """Omitted author must not produce an unattributable NULL row."""
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="no author given")
    assert _created_by(task_id) == "unattributed"


def test_programmatic_create_blank_created_by_stamps_unattributed(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="blank author", created_by="")
    assert _created_by(task_id) == "unattributed"


def test_programmatic_create_whitespace_created_by_stamps_unattributed(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="blank author", created_by="   ")
    assert _created_by(task_id) == "unattributed"


def test_unattributed_stamp_is_machine_vocabulary(kanban_home):
    """The fallback label must never collide with the session-authenticated
    human vocabulary ('user', 'dashboard'): it exists precisely so surfaces
    that did NOT hold session identity are tellable from ones that did."""
    with kbc.connect_closing() as conn:
        plain = kb.create_task(conn, title="plain")
    assert _created_by(plain) not in {"user", "dashboard"}


# ---------------------------------------------------------------------------
# Explicit authors pass through untouched (no behavior change)
# ---------------------------------------------------------------------------


def test_explicit_author_passes_through_unchanged(kanban_home):
    """Surfaces holding session identity keep stamping it — the fallback must
    not overwrite an explicit author."""
    with kbc.connect_closing() as conn:
        dash = kb.create_task(conn, title="dashboard card", created_by="dashboard")
        human = kb.create_task(conn, title="desktop card", created_by="user")
    assert _created_by(dash) == "dashboard"
    assert _created_by(human) == "user"


def test_desktop_cli_path_still_stamps_user(kanban_home, monkeypatch):
    """No behavior change for the desktop path: it shells through the CLI,
    whose parser defaults --created-by to 'user' (hermes_cli/kanban_parser.py);
    the stamp is the literal 'user' regardless of the driving profile's env."""
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    from hermes_cli import kanban as kc

    raw = kc.run_slash("create 'desktop path card'")
    task_id = raw.split()[1]
    assert _created_by(task_id) == "user"


def test_dashboard_quick_add_stamps_dashboard_author(kanban_home):
    """The dashboard POST /tasks endpoint (the web quick-add surface) stamps
    its own server identity regardless of the client payload."""
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    mod = _load_dashboard_router()
    app = FastAPI()
    app.include_router(mod.router)
    client = TestClient(app)

    resp = client.post("/tasks", json={"title": "quick add card"})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    task_id = payload["task"]["id"]
    assert payload["task"]["created_by"] == "dashboard"
    assert _created_by(task_id) == "dashboard"
