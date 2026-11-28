"""Tests: a TTL reclaim of a pid-less claim whose host-local claimer is gone
does not charge the breaker.

``hermes kanban claim <id>`` from a terminal records ``<host>:<cli pid>`` and
the CLI exits at once; no worker ever binds. When the TTL expired,
``release_stale_claims`` booked the reclaim as a failure (#111306), so with
``max_retries=1`` the card went ``blocked`` + ``gave_up`` and subscribers were
told it failed although nothing had run. The reclaim is now booked like a host
spawn refusal (``infrastructure``, #114720): back to ``ready``, the counter
neither charged nor reset. A live claimer, another host's lock, a non-numeric
claimer, a bound worker and a run with any sign that a worker started (a
``spawned`` event without a recorded pid: the worker died in the handshake)
keep the #111306 accounting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _host() -> str:
    return kb._claimer_id().split(":", 1)[0]


def _claim_and_expire(conn, task_id: str, claimer: str) -> None:
    assert kb.claim_task(conn, task_id, claimer=claimer) is not None
    conn.execute(
        "UPDATE tasks SET claim_expires = ? WHERE id = ?",
        (int(time.time()) - 3600, task_id),
    )


def _state(conn, task_id: str) -> dict:
    row = conn.execute(
        "SELECT status, consecutive_failures, claim_lock FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    return dict(row)


def _reclaimed_payload(conn, task_id: str) -> dict:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'reclaimed' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    assert row is not None
    return json.loads(row["payload"])


def test_dead_cli_claimer_returns_card_to_ready_without_breaker(kanban_home, monkeypatch):
    monkeypatch.setattr(kbd, "_pid_alive", lambda _pid: False)
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="cli claim", assignee="ext-lane", max_retries=1)
        _claim_and_expire(conn, t, f"{_host()}:424242")

        assert kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None) == 1

        assert _state(conn, t) == {"status": "ready", "consecutive_failures": 0, "claim_lock": None}
        kinds = [e.kind for e in kb.list_events(conn, t)]
        assert kinds[-1] == "reclaimed"
        assert "gave_up" not in kinds
        # Audit trail: the reclaim is still recorded, and says why it was not charged.
        payload = _reclaimed_payload(conn, t)
        assert payload["claimer_gone"] is True
        assert payload["stale_lock"] == f"{_host()}:424242"
        run = conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (t,),
        ).fetchone()
        assert run["outcome"] == "reclaimed"


def test_dead_cli_claimer_keeps_earlier_failures(kanban_home, monkeypatch):
    """Not charging is not forgiving: real failures before the orphaned claim still count."""
    monkeypatch.setattr(kbd, "_pid_alive", lambda _pid: False)
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="cli claim", assignee="ext-lane", max_retries=3)
        conn.execute("UPDATE tasks SET consecutive_failures = 2 WHERE id = ?", (t,))
        _claim_and_expire(conn, t, f"{_host()}:424242")

        assert kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None) == 1

        assert _state(conn, t)["consecutive_failures"] == 2
        assert _state(conn, t)["status"] == "ready"


def test_live_claimer_still_counts_toward_breaker(kanban_home):
    """A live claimer that never spawned (dispatcher-style) keeps #111306: charged, trips."""
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="never spawned", assignee="a", max_retries=1)
        # This test process is the claimer, and it is alive.
        _claim_and_expire(conn, t, kb._claimer_id())

        assert kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None) == 1

        assert _state(conn, t)["status"] == "blocked"
        assert _state(conn, t)["consecutive_failures"] == 1
        assert [e.kind for e in kb.list_events(conn, t)][-2:] == ["reclaimed", "gave_up"]
        assert _reclaimed_payload(conn, t)["claimer_gone"] is False


def test_foreign_host_lock_is_unchanged(kanban_home, monkeypatch):
    """A pid from another host proves nothing here, so the reclaim is charged as before."""
    monkeypatch.setattr(kbd, "_pid_alive", lambda _pid: False)
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="remote claim", assignee="a", max_retries=1)
        _claim_and_expire(conn, t, f"not-{_host()}:424242")

        assert kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None) == 1

        assert _state(conn, t)["status"] == "blocked"
        assert [e.kind for e in kb.list_events(conn, t)][-2:] == ["reclaimed", "gave_up"]


@pytest.mark.parametrize("suffix", ["worker", "", "0", "-5", "12:34", "12 ", "１２"])
def test_malformed_lock_is_unchanged(kanban_home, monkeypatch, suffix):
    """Only ``<host>:<positive ascii pid>`` is probed; any other claimer keeps the charge."""
    monkeypatch.setattr(kbd, "_pid_alive", lambda _pid: False)
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="odd claimer", assignee="a", max_retries=1)
        _claim_and_expire(conn, t, f"{_host()}:{suffix}")

        assert kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None) == 1

        assert _state(conn, t)["status"] == "blocked"
        assert _state(conn, t)["consecutive_failures"] == 1


def test_bound_worker_with_dead_claimer_still_counts(kanban_home, monkeypatch):
    """A worker bound to the run means something ran: the exemption needs ``worker_pid IS NULL``."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kbd, "_pid_alive", lambda _pid: False)
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="worker ran", assignee="a", max_retries=1)
        _claim_and_expire(conn, t, f"{_host()}:424242")
        kbd._set_worker_pid(conn, t, 12345)

        assert kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None) == 1

        assert _state(conn, t)["status"] == "blocked"
        assert _state(conn, t)["consecutive_failures"] == 1


# ``owner_admitted`` is covered by test_kanban_managed_worker_loss (the owner fence decides first).
@pytest.mark.parametrize("kind", ["spawned", "worker_registered", "heartbeat"])
def test_started_worker_without_pid_still_counts(kanban_home, monkeypatch, kind):
    """Spawn evidence on the run without a recorded pid (worker died in the adopt handshake) is a
    lost attempt, not an orphaned claim: still charged."""
    monkeypatch.setattr(kbd, "_pid_alive", lambda _pid: False)
    with kbc.connect() as conn:
        t = kb.create_task(conn, title="died in handshake", assignee="a", max_retries=1)
        _claim_and_expire(conn, t, f"{_host()}:424242")
        with kb.write_txn(conn):
            kb._append_event(conn, t, kind, {}, run_id=kb._current_run_id(conn, t))

        assert kb.release_stale_claims(conn, signal_fn=lambda _p, _s: None) == 1

        assert _state(conn, t)["status"] == "blocked"
        assert _state(conn, t)["consecutive_failures"] == 1
        assert _reclaimed_payload(conn, t)["claimer_gone"] is False
