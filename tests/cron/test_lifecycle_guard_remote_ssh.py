"""Remote-host lifecycle commands are not this process's problem.

The hard block exists because a restart launched from inside the gateway gets SIGTERM'd with it,
and the supervisor's respawn re-runs the turn (SIGTERM loop). A payload handed to ``ssh`` runs
under a separate sshd session — outside this process's children — so the same mechanism cannot
apply. Before this exemption, restarting the other nodes of the fleet from one box was refused
with an explanation that was factually wrong for those commands (observed 2026-10-28).

The exemption is segment-scoped: a local lifecycle command anywhere else in the same input is
still blocked, and anything that cannot be tokenized keeps the conservative reading.
"""
from __future__ import annotations

import pytest

from cron.lifecycle_guard import contains_gateway_lifecycle_command

REMOTE_OK = [
    "ssh -p 2222 deploy@build-host 'systemctl --user restart hermes-gateway'",
    "ssh build-host 'launchctl kickstart -k gui/501/com.example.gateway'",
    "ssh host 'hermes gateway restart'",
    "sudo ssh host 'systemctl --user restart hermes-gateway'",
    "tailscale ssh host 'systemctl restart hermes-gateway'",
    "ssh -o BatchMode=yes host 'systemctl --user restart hermes-gateway' ; echo done",
]

STILL_BLOCKED = [
    "systemctl --user restart hermes-gateway",
    "launchctl kickstart -k gui/501/com.example.gateway",
    "hermes gateway restart",
    # a remote segment does not launder a local one in the same input
    "ssh host uptime; systemctl --user restart hermes-gateway",
    "ssh host 'uptime' && pkill -f hermes-gateway",
]


@pytest.mark.parametrize("cmd", REMOTE_OK)
def test_remote_lifecycle_is_allowed(cmd: str) -> None:
    assert contains_gateway_lifecycle_command(cmd) is False, cmd


@pytest.mark.parametrize("cmd", STILL_BLOCKED)
def test_local_lifecycle_is_still_blocked(cmd: str) -> None:
    assert contains_gateway_lifecycle_command(cmd) is True, cmd


def test_untokenizable_input_fails_closed() -> None:
    # Unbalanced quote: the strip pass cannot parse it, so the guard keeps the raw reading.
    assert contains_gateway_lifecycle_command(
        "ssh host 'systemctl --user restart hermes-gateway"
    ) is True
