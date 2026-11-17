"""``/handoff desktop`` hands a CLI session to the Desktop app through its deep link.

Invariants: a launched link releases the CLI's ownership (lease + finalize skip) and exits;
a link the OS could not open leaves the CLI session untouched; an empty session gets its
state.db row before any handoff target is told to resume it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import cli_handoff_desktop as mod


def _cli(session_id: str = "20260927_180000_ab12cd"):
    cli = MagicMock()
    cli.session_id = session_id
    cli._should_exit = False
    cli._handoff_prepare_session.return_value = "weekly-review"
    return cli


@pytest.fixture(autouse=True)
def _isolate_handed_off_set():
    import cli as cli_mod

    before = set(cli_mod._handed_off_session_ids)
    cli_mod._handed_off_session_ids.clear()
    yield
    cli_mod._handed_off_session_ids.clear()
    cli_mod._handed_off_session_ids.update(before)


def test_launched_link_hands_ownership_to_desktop_and_exits():
    import cli as cli_mod

    cli = _cli()
    launched: list[str] = []
    with patch.object(mod, "launch_os_url", side_effect=lambda url: launched.append(url)), \
            patch("hermes_cli.profiles.current_profile_name", return_value="work"), \
            patch.object(mod, "_cp"):
        assert mod.handoff_to_desktop(cli) is False

    assert launched == ["hermes://session/20260927_180000_ab12cd?profile=work"]
    cli._release_active_session.assert_called_once_with()
    assert cli.session_id in cli_mod._handed_off_session_ids  # cleanup must not stamp end_reason
    assert cli._should_exit is True


def test_unopenable_link_keeps_the_cli_session():
    import cli as cli_mod

    cli = _cli()
    with patch.object(mod, "launch_os_url", return_value="xdg-open is not installed"), \
            patch("hermes_cli.profiles.current_profile_name", return_value="custom"), \
            patch.object(mod, "_cp") as cp:
        assert mod.handoff_to_desktop(cli) is True

    cli._release_active_session.assert_not_called()
    assert cli.session_id not in cli_mod._handed_off_session_ids
    assert cli._should_exit is False
    printed = " ".join(str(a) for call in cp.call_args_list for a in call.args)
    # The unroutable "custom" profile is dropped from the link the user is told to paste.
    assert "hermes://session/20260927_180000_ab12cd" in printed and "profile=" not in printed


def test_prepare_creates_the_row_for_an_empty_session(tmp_path):
    """Desktop (and a gateway switch_session) resume a ROW; an untouched session has never
    flushed one, and set_session_title no longer inserts it."""
    from types import SimpleNamespace

    from hermes_cli.cli_commands_mixin import CLICommandsMixin
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    cli = SimpleNamespace(_agent_running=False, _session_db=db, session_id="20260927_180000_ab12cd", model="m")
    try:
        assert db.get_session(cli.session_id) is None
        assert CLICommandsMixin._handoff_prepare_session(cli) == cli.session_id[:8]
        assert (db.get_session(cli.session_id) or {}).get("source") == "cli"
    finally:
        db.close()
