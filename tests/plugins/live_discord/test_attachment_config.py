"""Resolve attachment limits through the real gateway loader and profile scope."""

from contextlib import contextmanager
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


@contextmanager
def discord_home(home, section, settings):
    home.mkdir()
    home_token = set_hermes_home_override(home)
    multiplex_token = set_multiplex_context(True)
    scope_token = set_secret_scope(settings, profile_home=str(home))
    try:
        atomic_config_write(home / "config.yaml", section)
        config = load_gateway_config().platforms[Platform.DISCORD]
        assert DISCORD_AVAILABLE
        yield DiscordAdapter(config)
    finally:
        reset_secret_scope(scope_token)
        reset_multiplex_context(multiplex_token)
        reset_hermes_home_override(home_token)


@pytest.mark.parametrize("wrapper", ["platforms", "gateway"])
def test_scoped_attachment_limit_beats_nested_yaml(tmp_path, wrapper):
    block = {"platforms": {"discord": {"enabled": False, "max_attachment_bytes": 4096, "inline_text_attachments": False}}}
    section = block if wrapper == "platforms" else {"gateway": block}
    with discord_home(tmp_path / wrapper, section, {"DISCORD_MAX_ATTACHMENT_BYTES": "999"}) as adapter:
        assert adapter.config.extra["inline_text_attachments"] is False
        assert adapter._discord_max_attachment_bytes() == 999


@pytest.mark.parametrize("environment,configured,expected", [("0", 4096, 0), ("", 0, 0), ("", "", 32 * 1024 * 1024)])
def test_zero_and_unset_keep_their_attachment_limit_meanings(tmp_path, environment, configured, expected):
    section = {"platforms": {"discord": {"enabled": False, "max_attachment_bytes": configured}}}
    with discord_home(tmp_path / "scope", section, {"DISCORD_MAX_ATTACHMENT_BYTES": environment}) as adapter:
        assert adapter._discord_max_attachment_bytes() == expected
