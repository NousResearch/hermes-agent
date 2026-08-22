"""Regression tests for #88235 — state.db corruption must surface a warning
to the user's messaging platform, not stay silently in the logs.

When SessionDB init fails at gateway startup (corruption, NFS/SMB locks,
disk errors), the gateway sets _session_db = None and logs a warning — but
the user never sees it.  Messages may flow but nothing is persisted, and the
user only discovers the breakage when /resume or session_search comes back
empty.

The fix adds:
1. _session_db_init_error attribute on GatewayRunner, set when init fails
2. _send_session_db_warning_notifications() — broadcasts a recovery-guidance
   message to all home channels after the gateway connects
3. Improved "corrupt" cause wording in _format_turn_completion_explanation
   with the full recovery path (hermes doctor, sqlite3 .recover, backups)
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_corruption_warning_uses_active_profile_home(tmp_path, monkeypatch):
    """Recovery commands must target the profile whose database failed."""
    from gateway.config import Platform
    from gateway.run import GatewayRunner
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "isolated"
    default_home.mkdir()
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    send = AsyncMock(return_value=None)
    adapter = SimpleNamespace(send=send)
    transport = SimpleNamespace(adapter=adapter, is_relay=False, send=send)
    home = SimpleNamespace(
        chat_id="home-chat",
        thread_id=None,
        user_id=None,
        scope_id=None,
    )
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        platforms={Platform.DISCORD: SimpleNamespace(home_channel=home)}
    )
    runner.adapters = {Platform.DISCORD: adapter}
    runner._session_db_init_error = "database disk image is malformed"
    runner._thread_metadata_for_target = lambda *args, **kwargs: None

    token = set_hermes_home_override(profile_home)
    try:
        with patch("gateway.run.resolve_delivery_transport", return_value=transport):
            await runner._send_session_db_warning_notifications()
    finally:
        reset_hermes_home_override(token)

    message = send.await_args.args[2]
    assert str(profile_home / "state.db") in message
    assert str(profile_home / "backups") in message
    assert str(default_home) not in message
    assert "~/.hermes" not in message


def test_format_turn_completion_corrupt_includes_recovery_options():
    """The 'corrupt' persistence cause must list all recovery options."""
    from run_agent import AIAgent

    explanation = AIAgent._format_turn_completion_explanation(
        "session_persistence_failed", "corrupt"
    )
    assert "hermes doctor" in explanation
    assert ".recover" in explanation
    assert "backups" in explanation
    assert "Freeing disk space will not help" in explanation


def test_format_turn_completion_corrupt_uses_active_profile_home(tmp_path):
    """Formatter recovery paths must target the active profile."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from run_agent import AIAgent

    profile_home = tmp_path / "profiles" / "isolated"
    token = set_hermes_home_override(profile_home)
    try:
        explanation = AIAgent._format_turn_completion_explanation(
            "session_persistence_failed", "corrupt"
        )
    finally:
        reset_hermes_home_override(token)

    state_db_path = profile_home / "state.db"
    backups_path = profile_home / "backups"
    assert f'sqlite3 "{state_db_path}" ".recover"' in explanation
    assert f"{backups_path}/" in explanation
    assert "~/.hermes" not in explanation


def test_format_turn_completion_disk_still_advises_space():
    """The 'disk' cause still gives disk-space advice (unchanged)."""
    from run_agent import AIAgent

    explanation = AIAgent._format_turn_completion_explanation(
        "session_persistence_failed", "disk"
    )
    assert "free some space" in explanation


def test_format_turn_completion_locked_still_advises_retry():
    """The 'locked' cause still advises retrying (unchanged)."""
    from run_agent import AIAgent

    explanation = AIAgent._format_turn_completion_explanation(
        "session_persistence_failed", "locked"
    )
    assert "busy" in explanation
    assert "send it again" in explanation
