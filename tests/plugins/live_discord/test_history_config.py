"""Read free-response history policy through actual profiles and the gateway loader."""

import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent.secret_scope import reset_multiplex_context, reset_secret_scope, set_multiplex_context, set_secret_scope
from gateway.config import Platform, load_gateway_config
from hermes_cli.config import atomic_config_write
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from plugins.platforms.discord.adapter import DISCORD_AVAILABLE, DiscordAdapter


@pytest.mark.parametrize("configured,environment,expected", [(None, None, True), (False, None, False), (True, "false", False), (False, "true", True), (False, "", False)])
def test_owning_profile_history_switch_and_default(tmp_path, configured, environment, expected):
    home_token = set_hermes_home_override(tmp_path)
    multiplex_token = set_multiplex_context(True)
    setting = "DISCORD_HISTORY_BACKFILL_FREE_RESPONSE"
    previous = os.environ.get(setting)
    scope_token = set_secret_scope({} if environment is None else {setting: environment}, profile_home=str(tmp_path))
    try:
        discord = {"enabled": False, "history_backfill": False}
        if configured is not None:
            discord["history_backfill_free_response"] = configured
        atomic_config_write(tmp_path / "config.yaml", {"platforms": {"discord": discord}})
        config = load_gateway_config().platforms[Platform.DISCORD]
        assert DISCORD_AVAILABLE
        adapter = DiscordAdapter(config)
        assert adapter._discord_history_backfill_free_response() is expected
        assert adapter._discord_history_backfill() is False
        assert os.environ.get(setting) == previous
    finally:
        reset_secret_scope(scope_token)
        reset_multiplex_context(multiplex_token)
        reset_hermes_home_override(home_token)
