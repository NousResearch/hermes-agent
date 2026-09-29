"""Kanban dashboard plugin: timestamps serialized to the host ``timeAgo``
milliseconds contract (#94025).

The kanban SQL store keeps ``created_at`` in Unix SECONDS (``int(time.time())``
in ``hermes_cli/kanban_db.py``), but the drawer's host ``timeAgo``
(``apps/desktop/src/lib/time.ts``: ``formatAgo(fromMs, nowMs=Date.now())``)
expects MILLISECONDS. Without the conversion the drawer renders "NaNd ago".
Regression guard: every serialization path that feeds the drawer must expose
``created_at`` in ms (idempotent for already-ms values).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def _load_plugin_module():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_kanban_plugin_epoch_ms_test", plugin_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def plugin(kanban_home):
    return _load_plugin_module()


def test_epoch_ms_converts_seconds_and_is_idempotent(plugin):
    now_s = int(time.time())
    assert plugin._epoch_ms(now_s) == now_s * 1000
    assert plugin._epoch_ms(now_s * 1000) == now_s * 1000  # already ms
    assert plugin._epoch_ms(None) is None
    assert plugin._epoch_ms("not_a_number") == "not_a_number"


def test_task_dict_created_at_in_ms(plugin, kanban_home):
    kb.create_board("board-ms")
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="t1", body="b")
        task = kb.get_task(conn, tid)
    d = plugin._task_dict(task)
    assert d["created_at"] == int(task.created_at) * 1000


def test_comment_dict_created_at_in_ms(plugin, kanban_home):
    kb.create_board("board-ms")
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="t1")
        kb.add_comment(conn, tid, "alice", "hello")
        comment = kb.list_comments(conn, tid)[0]
    d = plugin._comment_dict(comment)
    assert d["created_at"] == int(comment.created_at) * 1000


def test_run_dict_timestamps_in_ms(plugin):
    run = kb.Run(
        id="run_1",
        task_id="t1",
        profile="default",
        step_key="execute",
        status="completed",
        claim_lock="lock",
        claim_expires=1700000000,
        worker_pid=1234,
        max_runtime_seconds=300,
        last_heartbeat_at=1700000000,
        started_at=1700000000,
        ended_at=1700000100,
        outcome="success",
        summary="done",
        error=None,
        metadata=None,
    )
    rd = plugin._run_dict(run)
    assert rd["started_at"] == 1700000000 * 1000
    assert rd["ended_at"] == 1700000100 * 1000
    assert rd["last_heartbeat_at"] == 1700000000 * 1000
