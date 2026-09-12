import time
from types import SimpleNamespace

import pytest

import cli
from hermes_state import SessionDB


@pytest.fixture(autouse=True)
def reset_single_query_finalize_state(monkeypatch):
    monkeypatch.setattr(cli, "_single_query_finalize_attempted_session_ids", set())
    monkeypatch.setattr(cli, "_cleanup_done", False)


def test_finalize_single_query_closes_agent_after_cleanup_before_release(monkeypatch):
    calls = []
    fake_agent = SimpleNamespace(close=lambda: calls.append(("close", {})))
    fake_cli = SimpleNamespace(
        agent=fake_agent,
        _release_active_session=lambda: calls.append(("release", {})),
    )

    monkeypatch.setattr(
        cli,
        "_notify_single_query_session_finalize",
        lambda _cli: calls.append(("finalize", {})),
    )
    monkeypatch.setattr(
        cli,
        "_run_cleanup",
        lambda **kwargs: calls.append(("cleanup", kwargs)),
    )

    cli._finalize_single_query(fake_cli)

    assert calls == [
        ("finalize", {}),
        ("cleanup", {"notify_session_finalize": False}),
        ("close", {}),
        ("release", {}),
    ]


def test_finalize_single_query_closes_agent_and_releases_lease_when_cleanup_fails(
    monkeypatch,
):
    calls = []
    fake_agent = SimpleNamespace(close=lambda: calls.append("close"))
    fake_cli = SimpleNamespace(
        agent=fake_agent,
        _release_active_session=lambda: calls.append("release"),
    )

    def cleanup(**_kwargs):
        calls.append("cleanup")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _cli: None)
    monkeypatch.setattr(cli, "_run_cleanup", cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        cli._finalize_single_query(fake_cli)

    assert calls == ["cleanup", "close", "release"]


def test_finalize_single_query_ends_old_worker_session_so_retention_can_prune(
    monkeypatch, tmp_path
):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "completed-kanban-worker"
    old_started_at = time.time() - (8 * 86400)
    with monkeypatch.context() as time_patch:
        time_patch.setattr("hermes_state.time.time", lambda: old_started_at)
        session_db.create_session(session_id, source="cli")

    assert session_db.prune_sessions(older_than_days=7) == 0

    fake_agent = SimpleNamespace(
        session_id=session_id,
        platform="cli",
        close=lambda: session_db.end_session(session_id, "agent_close"),
    )
    fake_cli = SimpleNamespace(
        agent=fake_agent,
        session_id=session_id,
        _release_active_session=lambda: None,
    )
    monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _cli: None)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **_kwargs: None)

    cli._finalize_single_query(fake_cli)

    ended_session = session_db.get_session(session_id)
    assert ended_session is not None
    assert ended_session["ended_at"] is not None
    assert session_db.prune_sessions(older_than_days=7) == 1
    assert session_db.get_session(session_id) is None
    session_db.close()
