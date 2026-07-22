from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.slash_commands import GatewaySlashCommandsMixin
from hermes_cli.write_approval_commands import approval_profile_name, approval_toggle_allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_name", ["_handle_memory_command", "_handle_skills_command"])
async def test_household_gateway_cannot_disable_write_approval(handler_name):
    handler = GatewaySlashCommandsMixin.__new__(GatewaySlashCommandsMixin)
    event = SimpleNamespace(
        get_command_args=lambda: "approval off",
        source=SimpleNamespace(chat_id="test", profile="family"),
    )

    with patch(
        "hermes_cli.profiles.get_active_profile_name",
        return_value="dev",
    ):
        result = await getattr(handler, handler_name)(event)

    assert result == "Approval gate changes are operator-managed for this profile."


@pytest.mark.parametrize(
    ("source_profile", "active_profile", "expected"),
    [
        ("family", "dev", "family"),
        ("dev", "family", "dev"),
        (None, "family", "family"),
        (None, None, ""),
    ],
)
def test_routed_profile_wins_over_active_profile(
    source_profile, active_profile, expected
):
    resolved = approval_profile_name(source_profile, active_profile)
    assert resolved == expected
    assert approval_toggle_allowed(resolved) is (expected != "family")
