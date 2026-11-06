"""Behavior tests for the verified Desktop Project binding carried into a turn."""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from gateway.session_context import get_session_desktop_project
from gateway.session_policy import LocalSessionPolicy, policy_scope
from hermes_cli import projects_db
from hermes_state_runtime import RuntimeStoreError


def _policy(root, project_id):
    return LocalSessionPolicy(
        source="gui",
        platform="desktop",
        cwd=str(root),
        model="test-model",
        toolsets=(),
        config_json="{}",
        request_json=json.dumps({"source": "gui", "desktop_project_id": project_id}),
        terminal_json=json.dumps({"TERMINAL_ENV": "local", "TERMINAL_CWD": str(root)}),
    )


def test_turn_context_exposes_only_the_backend_verified_project_and_clears_it(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    profile = tmp_path / "profile"
    profile.mkdir()
    state_db = profile / "state.db"
    projects_path = profile / "projects.db"
    with projects_db.connect_closing(db_path=projects_path) as connection:
        project_id = projects_db.create_project(
            connection, name="Research", primary_path=str(root), folders=[str(root / "docs")]
        )

    from gateway import session_local_editor, session_local_mcp

    monkeypatch.setattr(session_local_editor, "editor_scope", lambda _policy: nullcontext())
    monkeypatch.setattr(session_local_mcp, "editor_mcp_scope", lambda *_args: nullcontext())
    authority = SimpleNamespace(db=SimpleNamespace(db_path=str(state_db)))

    with policy_scope(_policy(root, project_id), authority=authority):
        assert get_session_desktop_project() == {
            "project_id": project_id,
            "primary_path": str(root.resolve()),
        }

    assert get_session_desktop_project() is None


def test_turn_refuses_a_project_whose_host_root_changed(tmp_path, monkeypatch):
    root = tmp_path / "project"
    changed = tmp_path / "changed"
    root.mkdir()
    changed.mkdir()
    profile = tmp_path / "profile"
    profile.mkdir()
    projects_path = profile / "projects.db"
    with projects_db.connect_closing(db_path=projects_path) as connection:
        project_id = projects_db.create_project(
            connection, name="Research", primary_path=str(root)
        )
        connection.execute("UPDATE projects SET primary_path = ? WHERE id = ?", (str(changed), project_id))
        connection.commit()

    authority = SimpleNamespace(db=SimpleNamespace(db_path=str(profile / "state.db")))
    with pytest.raises(RuntimeStoreError) as error:
        with policy_scope(_policy(root, project_id), authority=authority):
            pytest.fail("a stale project root must not enter turn execution")
    assert error.value.field == "desktop_project_id"
