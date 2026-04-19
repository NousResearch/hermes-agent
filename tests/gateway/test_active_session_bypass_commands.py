"""Regression tests for ACTIVE_SESSION_BYPASS_COMMANDS.

Bug (root cause)
----------------
When an agent is actively running on a gateway platform (e.g. Telegram), the
adapter-layer dispatch in ``gateway/platforms/base.py`` checks
``should_bypass_active_session(cmd)`` BEFORE any of the per-command logic in
``gateway/run.py`` runs.  If the command is not in the bypass allowlist, the
adapter hands the message to ``_busy_session_handler`` which calls
``running_agent.interrupt(event.text)`` and replies with the
``⚡ Interrupting current task...`` ack.  The command text never reaches its
handler and is instead delivered to the agent as interrupt text.

Symptom
-------
Sending ``/model`` (or ``/verbose``, ``/fast``, ``/reasoning``, ``/yolo``,
``/personality``, ``/provider``, ``/voice``) while the agent is running on
Telegram produces ``⚡ Interrupting current task...`` instead of executing the
settings command.

Fix
---
These commands only touch config / are read-only — they never mutate the
running agent's state.  They are therefore safe to dispatch while an agent is
running and must be included in
``hermes_cli.commands.ACTIVE_SESSION_BYPASS_COMMANDS``.

See also: ``gateway/platforms/base.py:1617`` — the adapter-layer bypass check
that this allowlist drives.
"""

from __future__ import annotations

import pytest

from hermes_cli.commands import (
    ACTIVE_SESSION_BYPASS_COMMANDS,
    should_bypass_active_session,
)


# Settings / info commands added in the fix.  Each handler in gateway/run.py
# was audited and confirmed NOT to touch ``running_agent`` / ``_running_agents``
# or call ``.interrupt(...)``.
SETTINGS_COMMANDS_TO_BYPASS = (
    "model",
    "provider",
    "verbose",
    "fast",
    "reasoning",
    "yolo",
    "personality",
    "voice",
)


# Pre-existing bypass commands — guard against accidental regression if
# someone narrows the allowlist without considering these.
EXISTING_BYPASS_COMMANDS = (
    "agents",
    "approve",
    "background",
    "commands",
    "deny",
    "help",
    "new",
    "profile",
    "queue",
    "restart",
    "status",
    "stop",
    "update",
)


@pytest.mark.parametrize("cmd", SETTINGS_COMMANDS_TO_BYPASS)
def test_settings_command_bypasses_active_session(cmd: str) -> None:
    """Settings/info commands must bypass the active-session busy handler.

    Otherwise they are intercepted by the adapter-layer busy handler (see
    gateway/platforms/base.py:1636) and fed to the running agent as
    interrupt text — producing the ``⚡ Interrupting current task...`` ack
    instead of executing the command.
    """
    assert cmd in ACTIVE_SESSION_BYPASS_COMMANDS, (
        f"/{cmd} is missing from ACTIVE_SESSION_BYPASS_COMMANDS — it will be "
        f"intercepted by the active-session busy handler and delivered to the "
        f"running agent as interrupt text."
    )
    assert should_bypass_active_session(cmd) is True


@pytest.mark.parametrize("cmd", EXISTING_BYPASS_COMMANDS)
def test_existing_bypass_commands_still_bypass(cmd: str) -> None:
    """Regression guard: pre-existing bypass commands must keep bypassing."""
    assert cmd in ACTIVE_SESSION_BYPASS_COMMANDS
    assert should_bypass_active_session(cmd) is True


def test_registered_but_excluded_command_does_not_bypass() -> None:
    """Sanity check: a real registered slash command NOT in the allowlist returns False.

    Distinguishes the "known-but-excluded" branch from
    ``test_unknown_command_does_not_bypass`` below (which exercises the
    ``resolve_command() is None`` branch).  ``/skills`` is a real top-level
    command (see ``hermes_cli/commands.py``) that deliberately is NOT in the
    active-session bypass allowlist.
    """
    from hermes_cli.commands import resolve_command

    assert resolve_command("skills") is not None, (
        "Precondition: /skills must be a registered command for this test "
        "to meaningfully exercise the known-but-excluded branch."
    )
    assert "skills" not in ACTIVE_SESSION_BYPASS_COMMANDS
    assert should_bypass_active_session("skills") is False


def test_unknown_command_does_not_bypass() -> None:
    """Unknown / None inputs must not bypass (defensive default)."""
    assert should_bypass_active_session(None) is False
    assert should_bypass_active_session("definitely-not-a-command") is False
