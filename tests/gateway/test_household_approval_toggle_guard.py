from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.slash_commands import GatewaySlashCommandsMixin


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_name", ["_handle_memory_command", "_handle_skills_command"])
async def test_household_gateway_cannot_disable_write_approval(handler_name):
    handler = GatewaySlashCommandsMixin.__new__(GatewaySlashCommandsMixin)
    event = SimpleNamespace(
        get_command_args=lambda: "approval off",
        source=SimpleNamespace(chat_id="test"),
    )

    with patch(
        "hermes_cli.profiles.get_active_profile_name",
        return_value="family",
    ):
        result = await getattr(handler, handler_name)(event)

    assert result == "Approval gate changes are operator-managed for this profile."
