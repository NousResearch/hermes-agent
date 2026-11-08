from cli import HermesCLI
from hermes_cli.active_sessions import (
    active_session_registry_snapshot,
    try_acquire_active_session,
)


def test_idle_cli_claim_preserves_writer_exclusion_without_spending_execution_capacity(tmp_path, monkeypatch):
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
        assert cli._claim_active_session("cli") is True
        assert printed == []
        assert {entry["session_id"] for entry in active_session_registry_snapshot()} == {
            "held-session", "new-cli-session"}
        duplicate, refusal = try_acquire_active_session(
            session_id="new-cli-session", surface="gateway", config=cfg)
        assert duplicate is None
        assert refusal.reason == "SESSION_NOT_OWNED"
    finally:
        held.release()
        cli._release_active_session()
