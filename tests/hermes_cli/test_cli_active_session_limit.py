import signal

import pytest

import cli as cli_mod
from cli import HermesCLI
from hermes_cli.active_sessions import (
    active_session_registry_snapshot,
    try_acquire_active_session,
)
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


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


def _install_rejecting_cli(monkeypatch):
    class RejectingCLI:
        def __init__(self, **_kwargs):
            self.session_id = "rejected-session"

        def _claim_active_session(self, surface, *, stderr=False):
            assert surface == "cli"
            assert stderr is True
            return False

    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {})
    monkeypatch.setattr(cli_mod, "HermesCLI", RejectingCLI)
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(signal, "signal", lambda *_args, **_kwargs: None)


def test_kanban_worker_session_limit_is_temporary_backpressure(monkeypatch):
    _install_rejecting_cli(monkeypatch)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_backpressure")

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="work kanban task", quiet=True, toolsets="terminal")

    assert exc_info.value.code == KANBAN_RATE_LIMIT_EXIT_CODE


def test_human_cli_session_limit_remains_a_regular_failure(monkeypatch):
    _install_rejecting_cli(monkeypatch)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hello", quiet=True, toolsets="terminal")

    assert exc_info.value.code == 1
