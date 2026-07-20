"""Contract tests for the shared Kanban test-isolation guard.

Every mutation-capable Kanban test fixture routes through
``tests/conftest.py::_isolate_kanban_root`` (exposed as the
``isolate_kanban_root`` fixture). The guard must, for a deliberately
*poisoned* inherited environment — one where a stray developer-shell /
worker Kanban path pin points at a live board OUTSIDE the per-test temp
root — fail CLOSED: it must clear every inherited Kanban path pin before
resolution and guarantee that every mutation-capable path (home, DB,
workspaces, attachments, logs) resolves strictly beneath the temp root.

These tests exercise the guard's *behavior* (resolved paths / raising),
never its source text.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb

# Every inherited pin that can redirect a mutation-capable Kanban path at a
# live board. Each is set to an absolute location OUTSIDE the per-test temp
# root to simulate a leaked developer-shell / dispatched-worker environment.
_POISON_PINS = (
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_HOME",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_ATTACHMENTS_ROOT",
    "HERMES_KANBAN_LOGS_ROOT",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_WORKSPACE",
)


def _poison_environment(monkeypatch) -> None:
    """Set every Kanban path pin at a live location outside any temp root.

    Runs from the test body — i.e. AFTER the autouse hermetic layer has
    already blanked these — so it faithfully models a pin that bypasses
    that first layer and reaches the fixture-local guard.
    """
    live = "/nonexistent/live-kanban"
    monkeypatch.setenv("HERMES_KANBAN_DB", f"{live}/kanban.db")
    monkeypatch.setenv("HERMES_KANBAN_HOME", live)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "poisonboard")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "poison-task")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", f"{live}/workspaces")
    monkeypatch.setenv("HERMES_KANBAN_ATTACHMENTS_ROOT", f"{live}/attachments")
    monkeypatch.setenv("HERMES_KANBAN_LOGS_ROOT", f"{live}/logs")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "poison-run")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", f"{live}/ws")


def test_guard_contains_every_mutation_path_under_poison(
    tmp_path, monkeypatch, isolate_kanban_root
):
    """Poisoned inherited pins must not leak into ANY mutation-capable path.

    After the guard runs, the home, DB, workspaces, attachments and log
    dirs must all resolve strictly beneath the per-test temp root — proving
    every inherited pin was cleared before path resolution.
    """
    _poison_environment(monkeypatch)

    isolate_kanban_root(tmp_path, monkeypatch)

    root = tmp_path.resolve()
    resolved = {
        "home": kb.kanban_home().resolve(),
        "db": kb.kanban_db_path().resolve(),
        "workspaces": kb.workspaces_root().resolve(),
        "attachments": kb.attachments_root().resolve(),
        "logs": kb.worker_logs_dir().resolve(),
    }
    escaped = {
        name: path
        for name, path in resolved.items()
        if not path.is_relative_to(root)
    }
    assert not escaped, f"mutation-capable Kanban paths escaped temp root: {escaped}"


def test_guard_aborts_when_resolution_escapes_temp_root(
    tmp_path, monkeypatch, isolate_kanban_root
):
    """If resolution ever lands outside the temp root, the guard aborts.

    Fail-closed backstop: simulate a resolution path the clearing step
    could not neutralize (here, ``kanban_db_path`` forced outside root) and
    assert the guard raises rather than letting a live DB be mutated.
    """
    monkeypatch.setattr(
        kb, "kanban_db_path", lambda *a, **k: __import__("pathlib").Path("/nonexistent/live-kanban/kanban.db")
    )
    with pytest.raises(AssertionError):
        isolate_kanban_root(tmp_path, monkeypatch)
