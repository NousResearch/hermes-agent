from cli import HermesCLI
from hermes_cli.active_sessions import (
    active_session_registry_snapshot,
    try_acquire_active_session,
)


def test_cli_claim_active_session_respects_global_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cfg = {"max_concurrent_sessions": 1}
    held, message = try_acquire_active_session(
        session_id="held-session",
        surface="tui",
        config=cfg,
    )
    assert message is None
    assert held is not None

    cli = object.__new__(HermesCLI)
    cli.session_id = "new-cli-session"
    cli.config = cfg
    cli._active_session_lease = None
    printed: list[str] = []
    cli._console_print = lambda text: printed.append(text)

    try:
        assert cli._claim_active_session("cli") is False
        assert len(printed) == 1
        assert "active session limit (1/1)" in printed[0]
        # Names the holding surface ("tui"), not the blocked one.
        assert "Held by: tui" in printed[0]

        held.release()

        assert cli._claim_active_session("cli") is True
        assert [entry["session_id"] for entry in active_session_registry_snapshot()] == [
            "new-cli-session"
        ]
    finally:
        held.release()
        cli._release_active_session()


def _cli_on(session_id, cfg=None):
    cli = object.__new__(HermesCLI)
    cli.session_id = session_id
    cli.config = cfg if cfg is not None else {}
    cli._active_session_lease = None
    cli._console_print = lambda text: None
    return cli


def test_lease_follows_the_session_the_cli_moves_to(tmp_path, monkeypatch):
    """/new and every auto-compression rotation reassign ``self.session_id``. The lease has to
    move with it: otherwise the session we left stays fenced against every other surface for the
    life of the process, and the session we are writing holds no slot at all."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cli = _cli_on("parent-session")
    try:
        assert cli._claim_active_session("cli") is True
        assert [e["session_id"] for e in active_session_registry_snapshot()] == ["parent-session"]

        cli.session_id = "child-session"  # /new, or a compression child
        cli._claim_active_session("cli")

        assert [e["session_id"] for e in active_session_registry_snapshot()] == ["child-session"]
    finally:
        cli._release_active_session()


def test_left_session_is_reopenable_by_another_surface(tmp_path, monkeypatch):
    """The user-visible half: after the CLI moves on, the session it left must be openable
    elsewhere. Per-session exclusivity is documented as unconditional correctness, so a lease
    naming a session nobody writes is a refusal with no way out but killing the CLI."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cli = _cli_on("parent-session")
    other = None
    try:
        assert cli._claim_active_session("cli") is True
        cli.session_id = "child-session"
        cli._claim_active_session("cli")

        # A DIFFERENT writer: identity is (pid, live_session_id), so the stand-in has to carry
        # its own live id — reusing "parent-session" here would be read as this process
        # re-claiming its own row and would pass even with the bug present.
        other, message = try_acquire_active_session(
            session_id="parent-session", surface="desktop", config={},
            metadata={"live_session_id": "desktop-live-session"})
        assert message is None, f"the abandoned session is still fenced: {message}"
        assert other is not None
    finally:
        if other is not None:
            other.release()
        cli._release_active_session()


def test_unchanged_session_does_not_churn_the_lease(tmp_path, monkeypatch):
    """Re-claiming for the same id keeps the same lease — the fast path must not drop and
    re-acquire the slot on every turn."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cli = _cli_on("only-session")
    try:
        assert cli._claim_active_session("cli") is True
        first = cli._active_session_lease
        assert cli._claim_active_session("cli") is True
        assert cli._active_session_lease is first
        assert [e["session_id"] for e in active_session_registry_snapshot()] == ["only-session"]
    finally:
        cli._release_active_session()
