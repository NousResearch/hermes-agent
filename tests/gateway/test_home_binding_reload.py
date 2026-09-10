"""Canonical home identity survives matching legacy delivery settings."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    _apply_env_overrides,
)
from tests.gateway.test_hosted_room_messaging import _event, _runner


@pytest.mark.parametrize(
    "platform", [Platform.TELEGRAM, Platform.SIGNAL, Platform.WHATSAPP]
)
@pytest.mark.parametrize("target", ["same", "other-chat", "other-thread"])
def test_env_delivery_override_preserves_only_the_same_binding(
    monkeypatch, platform, target
):
    monkeypatch.setenv(
        f"{platform.value.upper()}_HOME_CHANNEL",
        "other" if target == "other-chat" else "selected",
    )
    monkeypatch.setenv(
        f"{platform.value.upper()}_HOME_CHANNEL_THREAD_ID",
        "other" if target == "other-thread" else "topic",
    )
    home = HomeChannel(
        platform=platform,
        chat_id="selected",
        name="Home",
        thread_id="topic",
        user_id="owner",
        scope_id="selected-scope",
    )
    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True, home_channel=home)}
    )
    _apply_env_overrides(config)
    loaded = config.get_home_channel(platform)
    assert loaded is not None
    if target == "same":
        assert (loaded.user_id, loaded.scope_id) == ("owner", "selected-scope")
    else:
        assert loaded.user_id is None
        assert loaded.scope_id is None


_COLD_LOAD = """
import json
from pathlib import Path
from hermes_constants import get_hermes_home
from hermes_cli.env_loader import load_hermes_dotenv
load_hermes_dotenv(hermes_home=get_hermes_home(), project_env=Path('/nonexistent/hermes-env'), load_external_secrets=False)
from gateway.config import load_gateway_config, Platform
print('HOME_BINDING=' + json.dumps(load_gateway_config().get_home_channel(Platform.TELEGRAM).to_dict()))
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_target", ["same", "different", "absent"])
@pytest.mark.parametrize("scope", [None, "scope-a"])
async def test_sethome_binding_survives_real_reload(
    tmp_path, monkeypatch, legacy_target, scope
):
    from agent import secret_scope
    from gateway.config import load_gateway_config
    from hermes_cli.env_loader import load_hermes_dotenv

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home_dir))
    (home_dir / "config.yaml").write_text(
        yaml.safe_dump({
            "gateway": {"multiplex_profiles": False},
            "platforms": {
                "telegram": {
                    "enabled": True,
                    "extra": {
                        "allow_from": ["user-1", "user-2"],
                        "group_allow_admin_from": ["user-1"],
                    },
                }
            },
        })
    )
    env_path = home_dir / ".env"
    env_path.write_text(
        "TELEGRAM_ALLOWED_USERS=user-1,user-2\nTELEGRAM_HOME_CHANNEL=chat-telegram\n"
    )
    prior = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(False)
    try:
        load_hermes_dotenv(
            hermes_home=home_dir,
            project_env=home_dir / "unused",
            load_external_secrets=False,
        )
        runner = _runner(platform=Platform.TELEGRAM, extra={})
        runner.config = load_gateway_config()
        runner.adapters[Platform.TELEGRAM].config = runner.config.platforms[Platform.TELEGRAM]
        runner.pairing_store = None
        event = _event(
            "/sethome",
            platform=Platform.TELEGRAM,
            chat_type="group",
            is_one_to_one=False,
        )
        event.source.scope_id = scope
        assert runner.config.get_home_channel(Platform.TELEGRAM).user_id is None
        await runner._handle_set_home_command(event)
        persisted = yaml.safe_load((home_dir / "config.yaml").read_text())["platforms"][
            "telegram"
        ]["home_channel"]
        assert persisted["user_id"] == "user-1"
        assert runner._can_control_group_chats(event, require_audience=False)
        assert not runner._can_control_group_chats(event)

        if legacy_target != "same":
            target = (
                "TELEGRAM_HOME_CHANNEL=foreign-chat\n"
                if legacy_target == "different"
                else ""
            )
            env_path.write_text("TELEGRAM_ALLOWED_USERS=user-1,user-2\n" + target)
        child = subprocess.run(
            [sys.executable, "-c", _COLD_LOAD],
            cwd=home_dir,
            env={
                "PATH": os.environ["PATH"],
                "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
                "HOME": str(home_dir),
                "HERMES_HOME": str(home_dir),
                "HERMES_TEST_ISOLATION": str(home_dir),
            },
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        loaded = json.loads(
            next(
                line.removeprefix("HOME_BINDING=")
                for line in child.stdout.splitlines()
                if line.startswith("HOME_BINDING=")
            )
        )
        if legacy_target == "different":
            assert loaded["chat_id"] == "foreign-chat"
            assert "user_id" not in loaded and "scope_id" not in loaded
        else:
            assert loaded["chat_id"] == event.source.chat_id
            assert loaded["user_id"] == "user-1"
            assert loaded.get("scope_id") == scope
    finally:
        secret_scope.set_multiplex_active(prior)
