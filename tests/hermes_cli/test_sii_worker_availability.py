from types import SimpleNamespace
from pathlib import Path

import pytest

from hermes_cli import sii_worker_availability as availability
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(**overrides):
    values = {
        "id": "t_worker",
        "assignee": "sii-worker",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_non_sii_worker_is_not_gated(monkeypatch):
    monkeypatch.setenv("HERMES_SII_AI1_ASSIGNEE", "sii-worker")
    called = False

    def probe(*_args, **_kwargs):
        nonlocal called
        called = True
        return False

    monkeypatch.setattr(availability, "_probe", probe)
    availability.ensure_ai1_available(_task(assignee="other-worker"))
    assert not called


def test_ready_ai1_does_not_send_wake(monkeypatch):
    monkeypatch.setenv("HERMES_SII_AI1_ASSIGNEE", "sii-worker")
    monkeypatch.setattr(availability, "_probe", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        availability.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("wake must not run")),
    )
    availability.ensure_ai1_available(_task())


def test_offline_ai1_wakes_and_waits_until_ready(monkeypatch):
    monkeypatch.setenv("HERMES_SII_AI1_ASSIGNEE", "sii-worker")
    monkeypatch.setenv("HERMES_SII_AI1_WOL_MAC", "b0:82:e2:ac:d2:21")
    monkeypatch.setenv("HERMES_SII_AI1_WAKE_TIMEOUT_SECONDS", "3")
    answers = iter([False, False, True])
    monkeypatch.setattr(availability, "_probe", lambda *_args, **_kwargs: next(answers))
    calls = []
    monkeypatch.setattr(
        availability.subprocess,
        "run",
        lambda args, **kwargs: calls.append(args) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(availability.time, "sleep", lambda _seconds: None)

    availability.ensure_ai1_available(_task())

    assert calls == [["wakeonlan", "b0:82:e2:ac:d2:21"]]


def test_unavailable_ai1_fails_before_worker_spawn(monkeypatch):
    monkeypatch.setenv("HERMES_SII_AI1_ASSIGNEE", "sii-worker")
    monkeypatch.setenv("HERMES_SII_AI1_WOL_MAC", "")
    monkeypatch.setattr(availability, "_probe", lambda *_args, **_kwargs: False)

    try:
        availability.ensure_ai1_available(_task())
    except availability.AI1Unavailable as exc:
        assert "AI1" in str(exc)
        assert "no alternate host" in str(exc)
    else:
        raise AssertionError("unavailable AI1 must block dispatch")


def test_dispatch_blocks_sii_worker_after_configured_failure_limit(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_SII_AI1_ASSIGNEE", "sii-worker")
    monkeypatch.setenv("HERMES_SII_AI1_WOL_MAC", "")
    monkeypatch.setattr(availability, "_probe", lambda *_args, **_kwargs: False)
    spawned = []

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="SII work",
            assignee="sii-worker",
            max_runtime_seconds=60,
        )
        conn.execute("UPDATE tasks SET max_retries = 1 WHERE id = ?", (tid,))
        monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
        first = kbd.dispatch_once(
            conn,
            spawn_fn=lambda *_args, **_kwargs: spawned.append(True) or 42,
            failure_limit=1,
        )
        task = kb.get_task(conn, tid)

    assert spawned == []
    assert first.auto_blocked == [tid]
    assert task is not None
    assert task.status == "blocked"
    assert "AI1 is unavailable" in (task.last_failure_error or "")