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


def test_cli_shutdown_releases_root_lease_even_without_active_session_slot():
    from unittest.mock import Mock

    cli = object.__new__(HermesCLI)
    cli._active_session_lease = None
    failed_lease = Mock()
    root_lease = Mock()
    cli._failed_conversation_root_leases = [failed_lease]
    cli._conversation_root_lease = root_lease

    cli._release_active_session()

    failed_lease.release.assert_called_once_with()
    root_lease.release.assert_called_once_with()
    assert cli._conversation_root_lease is None
    assert cli._failed_conversation_root_leases == []
