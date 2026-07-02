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
        assert printed == [
            "[bold red]Hermes is at the active session limit (1/1). "
            "Try again when another session finishes.[/]"
        ]

        held.release()

        assert cli._claim_active_session("cli") is True
        assert [entry["session_id"] for entry in active_session_registry_snapshot()] == [
            "new-cli-session"
        ]
    finally:
        held.release()
        cli._release_active_session()


def test_cli_claim_active_session_fails_closed_when_registry_lock_errors(monkeypatch):
    def fail_acquire(**_kwargs):
        raise RuntimeError("active session file lock timed out")

    monkeypatch.setattr(
        "hermes_cli.active_sessions.try_acquire_active_session",
        fail_acquire,
    )

    cli = object.__new__(HermesCLI)
    cli.session_id = "new-cli-session"
    cli.config = {"max_concurrent_sessions": 1}
    cli._active_session_lease = None
    printed: list[str] = []
    cli._console_print = lambda text: printed.append(text)

    assert cli._claim_active_session("cli") is False
    assert printed == [
        "[bold red]Hermes could not claim an active session slot. "
        "Try again shortly or run `hermes runtime active-sessions status`.[/]"
    ]
    assert cli._active_session_lease is None
