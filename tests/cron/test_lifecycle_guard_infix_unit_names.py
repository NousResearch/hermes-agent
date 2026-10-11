"""A supervised unit named ``hermes-<role>-gateway`` is still the gateway: systemctl/launchctl
lifecycle verbs on it must be refused, while read-only verbs and unrelated hermes units stay open."""

import pytest

from cron.lifecycle_guard import contains_gateway_lifecycle_command


@pytest.mark.parametrize(
    "command",
    [
        "systemctl --user restart hermes-fleet-gateway.service",
        "systemctl --user stop hermes-codex-gateway",
        "systemctl stop hermes-local-gateway.service",
        "launchctl kickstart -k gui/501/ai.hermes.fleet.gateway",
        "systemctl --user restart hermes-gateway.service",
    ],
)
def test_infix_unit_lifecycle_is_blocked(command):
    assert contains_gateway_lifecycle_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "systemctl --user status hermes-fleet-gateway.service",
        "systemctl --user restart hermes-memory.service",
        "cat /docs/hermes-fleet-gateway-notes.md",
    ],
)
def test_non_lifecycle_or_unrelated_unit_is_allowed(command):
    assert not contains_gateway_lifecycle_command(command)
