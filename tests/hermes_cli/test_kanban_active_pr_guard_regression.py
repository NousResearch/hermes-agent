"""Regression tests for Kanban active-PR guard false positives."""
from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_diagnostics as diagnostics
from hermes_cli import kanban_ops


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task_with_pr_comment(conn, *, title: str, url: str) -> str:
    task_id = kb.create_task(conn, title=title, assignee="worker")
    kb.add_comment(conn, task_id, author="worker", body=f"Evidence: {url}")
    return task_id


def test_merged_pr_comment_does_not_hold_ready_task(kanban_home, monkeypatch):
    """A historical merged PR URL must not make a ready task unspawnable."""
    with kbc.connect() as conn:
        task_id = _task_with_pr_comment(
            conn,
            title="baseline gate",
            url="https://github.com/example/repo/pull/88",
        )

        calls = []

        def fake_run(argv, **kwargs):
            calls.append((argv, kwargs))
            return type("Completed", (), {"returncode": 0, "stdout": '{"state":"MERGED"}'})()

        monkeypatch.setattr(kbd.subprocess, "run", fake_run)

        assert kbd.check_respawn_guard(conn, task_id) is None
        assert calls, "the guard must resolve the referenced PR state"
        assert calls[0][0] == [
            "gh", "pr", "view", "88", "--repo", "example/repo", "--json", "state"
        ]


def test_open_pr_comment_still_holds_ready_task(kanban_home, monkeypatch):
    """An open PR URL remains a duplicate-work guard."""
    with kbc.connect() as conn:
        task_id = _task_with_pr_comment(
            conn,
            title="already PRed",
            url="https://github.com/example/repo/pull/89",
        )

        monkeypatch.setattr(
            kbd.subprocess,
            "run",
            lambda *args, **kwargs: type(
                "Completed", (), {"returncode": 0, "stdout": '{"state":"OPEN"}'}
            )(),
        )

        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


def test_unresolved_pr_state_keeps_ready_task_guarded(kanban_home, monkeypatch):
    """A lookup failure must fail closed rather than allow duplicate work."""
    with kbc.connect() as conn:
        task_id = _task_with_pr_comment(
            conn,
            title="unknown PR state",
            url="https://github.com/example/repo/pull/90",
        )

        monkeypatch.setattr(
            kbd.subprocess,
            "run",
            lambda *args, **kwargs: type(
                "Completed", (), {"returncode": 1, "stdout": ""}
            )(),
        )

        assert kbd.check_respawn_guard(conn, task_id) == "active_pr"


def test_dispatch_json_and_text_expose_respawn_guarded():
    """Operators must see why dispatch spawned zero workers."""
    result = kbd.DispatchResult(respawn_guarded=[("t_demo", "active_pr")])
    args = argparse.Namespace(
        dry_run=False, max=None, failure_limit=2, json=True,
    )
    with (
        patch("hermes_cli.config.load_config", return_value={"kanban": {}}),
        patch.object(kbc, "connect_closing", return_value=nullcontext(object())),
        patch.object(kbd, "dispatch_once", return_value=result),
    ):
        output = StringIO()
        with patch("sys.stdout", output):
            assert kanban_ops._cmd_dispatch(args) == 0
    payload = json.loads(output.getvalue())
    assert payload["respawn_guarded"] == [{"task_id": "t_demo", "reason": "active_pr"}]

    args.json = False
    with (
        patch("hermes_cli.config.load_config", return_value={"kanban": {}}),
        patch.object(kbc, "connect_closing", return_value=nullcontext(object())),
        patch.object(kbd, "dispatch_once", return_value=result),
    ):
        output = StringIO()
        with patch("sys.stdout", output):
            assert kanban_ops._cmd_dispatch(args) == 0
    assert "Deferred (respawn guard): t_demo (active_pr)" in output.getvalue()


def test_ready_diagnostics_identify_respawn_guard(kanban_home):
    """Diagnostics must distinguish a guarded ready card from a missing worker."""
    now = 100_000
    task = {
        "id": "t_demo",
        "status": "ready",
        "assignee": "worker",
        "claim_lock": None,
        "created_at": now - 3600,
    }
    events = [
        {"kind": "created", "created_at": now - 3600, "payload": None},
        {
            "kind": "respawn_guarded",
            "created_at": now - 30,
            "payload": '{"reason":"active_pr"}',
        },
    ]

    stranded = [
        item
        for item in diagnostics.compute_task_diagnostics(task, events, [], now=now)
        if item.kind == "stranded_in_ready"
    ]
    assert len(stranded) == 1
    assert "respawn guard" in stranded[0].title.lower()
    assert "active_pr" in stranded[0].title
    assert stranded[0].data["guard_reason"] == "active_pr"
