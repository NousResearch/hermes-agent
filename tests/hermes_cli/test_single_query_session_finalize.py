from types import SimpleNamespace
from typing import Any

import pytest

import cli


@pytest.fixture(autouse=True)
def reset_single_query_finalize_state(monkeypatch):
    monkeypatch.setattr(cli, "_single_query_finalize_attempted_session_ids", set())
    monkeypatch.setattr(cli, "_cleanup_done", False)




def test_finalize_single_query_releases_lease_before_cleanup(monkeypatch):
    """Settlement (finalize hook) runs before the release, the release before
    cleanup; a cleanup failure can never skip or precede the release."""
    calls = []
    fake_cli = SimpleNamespace(_release_active_session=lambda: calls.append("release"))

    def cleanup(**kwargs):
        calls.append("cleanup")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        cli,
        "_notify_single_query_session_finalize",
        lambda _cli: calls.append("finalize"),
    )
    monkeypatch.setattr(cli, "_run_cleanup", cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        cli._finalize_single_query(fake_cli)

    assert calls == ["finalize", "release", "cleanup"]


def test_finalize_single_query_runs_cleanup_when_finalize_hook_fails(monkeypatch):
    calls = []
    fake_agent = SimpleNamespace(session_id="agent-session", platform="cli")
    fake_cli = SimpleNamespace(
        agent=fake_agent,
        session_id="cli-session",
        _release_active_session=lambda: calls.append("release"),
    )

    def invoke_hook(name, **kwargs):
        calls.append("finalize")
        raise RuntimeError("hook failed")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **kwargs: calls.append("cleanup"))

    cli._finalize_single_query(fake_cli)

    assert calls == ["finalize", "release", "cleanup"]


def test_finalize_settles_session_before_a_successor_can_take_over(tmp_path, monkeypatch):
    """P1 handoff race: a successor that acquires the session DURING the
    predecessor's exit linger must never receive the predecessor's stale
    ``cli_close`` end-stamp — settlement (flush incl. end_session) must complete
    before the lease is released.

    Exercises the real seam with a real SessionDB and a real lease: the linger is
    patched to play the successor (acquire the freed lease, reopen the row), and
    the final assertion is that the successor's open interval survives the
    predecessor's whole finalize. On the unsafe order (release → linger → flush)
    the flush's unconditional ``end_session`` stamps the successor's interval and
    this test fails."""
    from hermes_cli import active_sessions
    from hermes_state import SessionDB

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "s1"
    db.create_session(sid, "cli")
    lease_a, message = active_sessions.try_acquire_active_session(
        session_id=sid, surface="cli", config={}, metadata={"live_session_id": sid}
    )
    assert lease_a is not None, message

    agent_a = SimpleNamespace(
        session_id=sid, _session_db=db, _session_messages=[],
        _persist_disabled=False, _persist_session=lambda *a, **k: None,
    )
    cli_a = SimpleNamespace(
        agent=agent_a, session_id=sid, conversation_history=[], _session_db=db,
        _release_active_session=lease_a.release,
    )

    taken_over: dict[str, Any] = {}

    def linger_with_takeover(_cli):
        # The successor arrives mid-linger. Early admission must hold (the point of
        # the PR)…
        lease_b, msg_b = active_sessions.try_acquire_active_session(
            session_id=sid, surface="cli", config={}, metadata={"live_session_id": sid}
        )
        assert lease_b is not None, f"successor could not acquire during the linger: {msg_b}"
        taken_over["lease_b"] = lease_b
        # …and it resumes the row (the predecessor's settlement already ended it).
        db.reopen_session(sid)

    monkeypatch.setattr(cli, "_wait_for_oneshot_background_completions", linger_with_takeover)
    monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _c: None)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **_k: None)

    cli._finalize_single_query(cli_a)

    row = db.get_session(sid)
    assert row is not None and row["ended_at"] is None, (
        "successor's open interval was end-stamped by the predecessor's finalize"
    )
    taken_over["lease_b"].release()


def test_notify_single_query_session_finalize_uses_agent_session(monkeypatch):
    calls = []
    fake_agent = SimpleNamespace(session_id="agent-session", platform="cli")
    fake_cli = SimpleNamespace(agent=fake_agent, session_id="cli-session")

    def invoke_hook(name, **kwargs):
        calls.append((name, kwargs))

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)

    cli._notify_single_query_session_finalize(fake_cli)

    assert calls == [
        (
            "on_session_finalize",
            {
                "session_id": "agent-session",
                "platform": "cli",
                "reason": "shutdown",
            },
        )
    ]
