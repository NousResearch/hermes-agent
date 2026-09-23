"""Per-profile isolation for the remaining process-global sinks under gateway.multiplex_profiles:
Yuanbao auto-sethome env write, the config ``terminal.env_passthrough`` allowlist, and the runner-level
Slack ignored-channel fail-safe (which only had the DEFAULT profile's GatewayConfig)."""
from __future__ import annotations

import os

from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _under_secondary(home, fn):
    set_multiplex_active(True)
    home_token = set_hermes_home_override(str(home))
    secret_token = set_secret_scope({})
    try:
        return fn()
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)


def test_yuanbao_auto_sethome_from_secondary_stays_in_its_config(tmp_path, monkeypatch):
    from gateway.platforms.yuanbao import AutoSetHomeMiddleware

    monkeypatch.delenv("YUANBAO_HOME_CHANNEL", raising=False)
    secondary = tmp_path / "profiles" / "b2"
    secondary.mkdir(parents=True)
    (secondary / "config.yaml").write_text("{}\n")

    class Adapter:
        name = "yuanbao-b2"

    class Ctx:
        chat_id = "dm:tenant-b2"
        chat_name = "b2"

    _under_secondary(secondary, lambda: AutoSetHomeMiddleware._persist_home(Adapter(), Ctx()))

    assert "YUANBAO_HOME_CHANNEL" not in os.environ
    assert "dm:tenant-b2" in (secondary / "config.yaml").read_text()


def test_env_passthrough_allowlist_follows_the_active_profile(tmp_path, monkeypatch):
    import tools.env_passthrough as ep

    default_home = tmp_path / "hermes"
    secondary = default_home / "profiles" / "b2"
    secondary.mkdir(parents=True)
    (default_home / "config.yaml").write_text("terminal:\n  env_passthrough: [FOO_DEFAULT]\n")
    (secondary / "config.yaml").write_text("terminal:\n  env_passthrough: [FOO_B2]\n")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    ep._config_passthrough.clear()

    assert ep._load_config_passthrough() == {"FOO_DEFAULT"}
    assert _under_secondary(secondary, ep._load_config_passthrough) == {"FOO_B2"}
    assert ep._load_config_passthrough() == {"FOO_DEFAULT"}


def test_slack_ignored_channels_use_the_routed_adapters_list(monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import _is_slack_ignored_channel

    monkeypatch.delenv("SLACK_IGNORED_CHANNELS", raising=False)
    default_cfg = GatewayConfig(platforms={Platform.SLACK: PlatformConfig(enabled=True, extra={"ignored_channels": ["C_DEFAULT"]})})

    class SecondaryAdapter:
        config = PlatformConfig(enabled=True, extra={"ignored_channels": ["C_B2"]})

    assert _is_slack_ignored_channel(default_cfg, "C_B2", SecondaryAdapter())
    assert not _is_slack_ignored_channel(default_cfg, "C_DEFAULT", SecondaryAdapter())
    # No routed adapter (legacy callers): the process-level config still rules.
    assert _is_slack_ignored_channel(default_cfg, "C_DEFAULT")
