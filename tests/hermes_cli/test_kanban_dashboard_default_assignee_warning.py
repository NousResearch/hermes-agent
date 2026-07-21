"""Dashboard create-task warning coverage for kanban.default_assignee."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


@pytest.fixture()
def isolated_dashboard_api(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for mod in list(sys.modules):
        if (
            mod == "hermes_constants"
            or mod == "hermes_state"
            or mod.startswith("hermes_cli")
            or mod.startswith("plugins.kanban.dashboard")
        ):
            del sys.modules[mod]

    # Add a second profile so the dashboard's "sole profile" auto-assign
    # shortcut does not hide the configured default-assignee behavior.
    worker_profile = tmp_path / "profiles" / "worker"
    worker_profile.mkdir(parents=True)
    (worker_profile / "config.yaml").write_text("{}\n")

    from plugins.kanban.dashboard import plugin_api
    yield plugin_api, tmp_path


def _write_config(home: Path, default_assignee: str) -> None:
    (home / "config.yaml").write_text(
        "kanban:\n"
        f"  default_assignee: {default_assignee}\n"
    )


def test_create_ready_unassigned_does_not_warn_when_default_assignee_resolves(
    isolated_dashboard_api, monkeypatch,
):
    plugin_api, home = isolated_dashboard_api
    _write_config(home, "default")
    monkeypatch.setattr(plugin_api, "_dispatcher_presence_warning", lambda: None)

    result = plugin_api.create_task(
        plugin_api.CreateTaskBody(title="route through configured fallback"),
        board=None,
    )

    assert "warning" not in result
    assert result["task"]["assignee"] is None
    assert plugin_api._configured_dispatch_default_assignee() == "default"


def test_create_ready_unassigned_warns_when_no_valid_default_assignee(
    isolated_dashboard_api, monkeypatch,
):
    plugin_api, home = isolated_dashboard_api
    _write_config(home, "missing-worker")
    monkeypatch.setattr(plugin_api, "_dispatcher_presence_warning", lambda: None)

    result = plugin_api.create_task(
        plugin_api.CreateTaskBody(title="no valid configured fallback"),
        board=None,
    )

    assert result["task"]["assignee"] is None
    assert result["warning"] == (
        "Task is ready but has no assignee, so the Kanban dispatcher "
        "will not pick it up until a worker profile is assigned."
    )
    assert plugin_api._configured_dispatch_default_assignee() is None
