"""A config saved while a profile connects must remain pending for the next scan."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner


@pytest.mark.asyncio
@pytest.mark.parametrize("startup", [False, True])
async def test_config_saved_during_connect_is_rescanned(tmp_path, monkeypatch, startup):
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "default"
    )
    (profile / "config.yaml").write_text("model: {default: test}\n", encoding="utf-8")
    secrets = profile / ".env"
    secrets.write_text("DISCORD_BOT_TOKEN=discord-test\n", encoding="utf-8")
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._running = True
    runner._primary_profile_name = "default"
    (
        runner.adapters,
        runner._profile_adapters,
        runner._failed_platforms,
        runner._profile_failed_platforms,
    ) = {}, {}, {}, {}
    runner.pairing_store, runner.pairing_stores = MagicMock(), {}
    runner._busy_text_modes_by_profile, runner._busy_input_modes_by_profile = {}, {}
    runner._register_config_hooks = lambda *a, **kw: None
    runner._configure_profile_adapter = lambda *a: None
    runner._sync_voice_mode_state_to_adapter = lambda *a: None
    runner._restore_secondary_completion_ledgers = lambda *a: None
    runner._adapter_credential_claim = lambda *a: None
    runner._adapter_listener_claim = lambda *a: None
    runner._create_adapter = lambda platform, config: SimpleNamespace(platform=platform)
    runner._note_served_profiles([("default", home)])
    connected = []

    async def connect(adapter, platform):
        connected.append(platform)
        if platform == Platform.DISCORD:
            # Configuration was already read; a second setup operation finishes while
            # the first adapter is awaiting its transport handshake.
            secrets.write_text(
                "DISCORD_BOT_TOKEN=discord-test\nTELEGRAM_BOT_TOKEN=telegram-test\n",
                encoding="utf-8",
            )
        return True

    async def after_added(profiles):
        pass

    runner._connect_initial_adapter_with_timeout = connect
    runner._after_profiles_added = after_added
    if startup:
        await runner._start_secondary_profile_adapters()
    else:
        await runner.reconcile_served_profiles()
    assert connected == [Platform.DISCORD]
    await runner.reconcile_served_profiles()
    assert connected == [Platform.DISCORD, Platform.TELEGRAM]
    await runner.reconcile_served_profiles()
    assert connected == [Platform.DISCORD, Platform.TELEGRAM]
