from cli import HermesCLI
from hermes_cli.active_sessions import (
    SESSION_NOT_OWNED,
    active_session_registry_snapshot,
    try_acquire_active_session,
)


def test_cli_claim_idle_owner_does_not_consume_global_limit(tmp_path, monkeypatch):
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
        # The CLI is a viewer/proxy: the gateway accounts for its real turn.
        assert cli._claim_active_session("cli") is True
        assert printed == []
        entries = {entry["session_id"]: entry for entry in active_session_registry_snapshot()}
        assert entries["new-cli-session"]["capacity_active"] is False
        assert entries["held-session"]["capacity_active"] is True
        held.release()
        next_turn, refusal = try_acquire_active_session(
            session_id="next-turn", surface="gateway:telegram", config=cfg,
        )
        assert refusal is None and next_turn is not None
        next_turn.release()
        # Idle still means an exclusive writer, not a free-for-all.
        foreign, refusal = try_acquire_active_session(
            session_id=cli.session_id, surface="desktop", config=cfg,
            metadata={"live_session_id": "different-writer"}, capacity_active=False,
        )
        assert foreign is None and getattr(refusal, "reason", None) == SESSION_NOT_OWNED
        assert cli._claim_active_session("cli") is True
        assert [entry["session_id"] for entry in active_session_registry_snapshot()] == [
            "new-cli-session"
        ]
    finally:
        held.release()
        cli._release_active_session()
