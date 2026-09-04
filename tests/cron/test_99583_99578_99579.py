"""Tests for issue fixes:

* #99583 — ``hermes cron create --script`` validates the script exists at
  creation (profile-scoped HERMES_HOME) and prints the resolved path.
* #99578 — ``hermes kanban show`` names the number of truncated events and
  offers ``--all``.
* #99579 — ``cron list``/``cron status`` name the profile being inspected.
"""

import argparse
import importlib
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture()
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    (home / "cron").mkdir()
    (home / "cron" / "output").mkdir()
    (home / "scripts" / "watch.sh").write_text("#!/bin/bash\necho alert\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "default")

    import hermes_constants

    importlib.reload(hermes_constants)
    yield home


class _Args(argparse.Namespace):
    """Minimal args object with the attributes cron_create reads."""

    def __init__(self, **kw):
        defaults = dict(
            schedule="every 5m",
            prompt="do a thing",
            name=None,
            deliver=None,
            repeat=None,
            skill=None,
            skills=None,
            script=None,
            workdir=None,
            model=None,
            model_provider=None,
            no_agent=False,
            monitor_script=None,
            monitor_url=None,
            continuity=None,
            reasoning_effort=None,
        )
        defaults.update(kw)
        super().__init__(**defaults)


# ---------------------------------------------------------------- #99583


def test_cron_create_rejects_missing_script(hermes_env, capsys):
    from hermes_cli.cron import cron_create

    rc = cron_create(_Args(script="does_not_exist.py"))
    out = capsys.readouterr().out
    assert rc == 1
    assert "Failed to create job" in out
    assert "does_not_exist.py" in out
    assert "Script not found" in out


def test_cron_create_rejects_out_of_tree_absolute_script(hermes_env, capsys):
    from hermes_cli.cron import cron_create

    outside = hermes_env.parent / "outside.py"
    outside.write_text("print('hi')\n")
    rc = cron_create(_Args(script=str(outside)))
    out = capsys.readouterr().out
    assert rc == 1
    assert "outside the scripts directory" in out


def test_cron_create_accepts_existing_script_and_prints_resolved(hermes_env, capsys, monkeypatch):
    from hermes_cli.cron import cron_create

    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    rc = cron_create(_Args(script="watch.sh"))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Created job" in out
    m = re.search(r"Resolved: (\S+watch\.sh)", out)
    assert m, f"expected Resolved line in output:\n{out}"
    assert Path(m.group(1)).resolve() == (hermes_env / "scripts" / "watch.sh").resolve()


def test_cron_create_rejects_missing_monitor_script(hermes_env, capsys):
    from hermes_cli.cron import cron_create

    rc = cron_create(_Args(monitor_script="no_monitor.py"))
    out = capsys.readouterr().out
    assert rc == 1
    assert "monitor-script" in out


# ---------------------------------------------------------------- #99578


def _make_kanban_task_with_events(tmp_path, n_events: int):
    from hermes_cli import kanban_db as kb

    db = tmp_path / "kb.db"
    with kb.connect_closing(db) as conn:
        task_id = kb.create_task(conn, title="t")
    with kb.connect_closing(db) as conn:
        now = int(time.time())
        for i in range(n_events):
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (task_id, None, f"evt{i}", None, now + i),
            )
    return db, task_id


def test_kanban_show_truncation_notice(tmp_path, capsys, monkeypatch):
    from hermes_cli.kanban import _cmd_show

    db, task_id = _make_kanban_task_with_events(tmp_path, 25)
    args = _Args(task_id=task_id, json=False, all=False)
    monkeypatch.setattr("hermes_cli.kanban_db.kanban_db_path", lambda board=None: db)

    rc = _cmd_show(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Events (26):" in out
    assert "6 earlier event(s) not shown" in out
    assert "--all" in out

    rc = _cmd_show(_Args(task_id=task_id, json=False, all=True))
    out2 = capsys.readouterr().out
    assert rc == 0
    assert "Events (26):" in out2
    assert "earlier event(s) not shown" not in out2
    assert "evt0" in out2


# ---------------------------------------------------------------- #99579


def test_cron_list_names_profile(hermes_env, capsys):
    from hermes_cli.cron import cron_list

    cron_list(show_all=False)
    out = capsys.readouterr().out
    assert "profile" in out.lower()
    # With no jobs the label names the profile explicitly.
    assert re.search(r"profile '[^']+'", out)


def test_cron_status_names_profile(hermes_env, capsys, monkeypatch):
    from hermes_cli import gateway
    from hermes_cli.cron import cron_status

    # Force the liveness probe to "not running" so the ✗ branch prints.
    monkeypatch.setattr(
        "hermes_cli.cron._builtin_gateway_liveness",
        lambda: False,
    )
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda: [])
    monkeypatch.setattr(
        "hermes_cli.cron._active_cron_provider_name",
        lambda: "builtin",
    )
    from cron import jobs as cron_jobs

    monkeypatch.setattr(cron_jobs, "get_ticker_heartbeat_age", lambda: None)
    monkeypatch.setattr(cron_jobs, "get_ticker_success_age", lambda: None)
    monkeypatch.setattr(cron_jobs, "get_ticker_last_error", lambda: None)
    cron_status()
    out = capsys.readouterr().out
    assert "profile" in out.lower()
