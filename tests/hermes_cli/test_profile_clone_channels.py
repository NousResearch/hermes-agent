"""``hermes profile create --clone`` leaves messaging channels behind (``hermes_cli.profile_channels``).

Invariant, not snapshot: the clone's credential fingerprint set — computed by the gateway's own
``_adapter_credential_fingerprint`` through the migrate preflight — is DISJOINT from the source's,
while provider/tool keys and general config survive; ``--clone-channels`` restores the copy.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import hermes_constants
from hermes_cli import gateway_migrate as gm
from hermes_cli.profile_channels import (
    channel_platforms_configured, shared_channel_credentials, strip_channel_env_file,
)
from hermes_cli.profiles import create_profile

_SOURCE_ENV = (
    "OPENAI_API_KEY=sk-model-key\n"
    "FIRECRAWL_API_KEY=fc-tool-key\n"
    "# telegram\n"
    "TELEGRAM_BOT_TOKEN=111111:default-telegram-token\n"
    "TELEGRAM_ALLOWED_USERS=12345\n"
    "TELEGRAM_GROUP_ALLOWED_CHATS=-100999\n"
    "DISCORD_BOT_TOKEN=default-discord-token-abcdef\n"
    "DISCORD_ALLOWED_USERS=777\n"
    "WHATSAPP_ENABLED=true\n"
    "API_SERVER_KEY=default-api-server-key-0123456789\n"
)
_SOURCE_CONFIG = {
    "model": {"default": "gpt-5", "provider": "openai"},
    "memory": {"provider": "builtin"},
    "platforms": {"telegram": {"enabled": True, "token": "111111:default-telegram-token"},
                  "discord": {"enabled": True}},
    "telegram": {"reactions": True, "allowed_chats": "-100999"},
    "discord": {"require_mention": False, "dm_role_auth_guild": "42"},
    "gateway": {"multiplex_profiles": True, "profile_routes": [{"profile": "x", "platform": "telegram"}],
                "platform_connect_timeout": 45},
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    for name in ("TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "API_SERVER_KEY", "WHATSAPP_ENABLED",
                 "GATEWAY_MULTIPLEX_PROFILES", "TELEGRAM_ALLOWED_USERS"):
        monkeypatch.delenv(name, raising=False)
    (root / ".env").write_text(_SOURCE_ENV, encoding="utf-8")
    (root / "config.yaml").write_text(yaml.safe_dump(_SOURCE_CONFIG), encoding="utf-8")
    (root / "SOUL.md").write_text("Be helpful.", encoding="utf-8")
    monkeypatch.setattr(gm, "_installed_service", lambda home: None)
    monkeypatch.setattr(gm, "_live_gateway_pid", lambda home: None)
    return root


def _fingerprints(profile_home: Path) -> set:
    """``(platform, fingerprint)`` claims exactly as the migrate preflight / multiplexer see them."""
    with gm._multiplex_read_mode():
        return set(gm._credential_claims(gm._profile_gateway_config(profile_home)))


def test_clone_strips_every_channel_credential_but_keeps_model_and_tool_keys(home):
    source_claims = _fingerprints(home)
    assert {p for p, _ in source_claims} >= {"telegram", "discord"}

    profile_dir = create_profile("bot2", clone_config=True, no_alias=True)

    assert _fingerprints(profile_dir).isdisjoint(source_claims)
    assert shared_channel_credentials(profile_dir, home) == []
    env_text = (profile_dir / ".env").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-model-key" in env_text and "FIRECRAWL_API_KEY=fc-tool-key" in env_text
    assert "TELEGRAM" not in env_text and "DISCORD" not in env_text
    assert "WHATSAPP_ENABLED" not in env_text and "API_SERVER_KEY" not in env_text
    cfg = yaml.safe_load((profile_dir / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["model"] == _SOURCE_CONFIG["model"] and cfg["memory"] == _SOURCE_CONFIG["memory"]
    assert (profile_dir / "SOUL.md").read_text(encoding="utf-8") == "Be helpful."
    for section in ("platforms", "telegram", "discord"):
        assert section not in cfg
    # The clone must not think it is the host's multiplexer, but unrelated gateway knobs survive.
    assert "multiplex_profiles" not in cfg["gateway"] and "profile_routes" not in cfg["gateway"]
    assert cfg["gateway"]["platform_connect_timeout"] == 45
    # The migrate preflight, which blocked with a duplicate finding per platform, is now clean.
    plan = gm.build_migration_plan()
    assert not plan.blocked, plan.blockers


def test_clone_channels_opt_in_keeps_the_source_channels(home):
    profile_dir = create_profile("twin", clone_config=True, no_alias=True, clone_channels=True)
    assert _fingerprints(profile_dir) == _fingerprints(home)
    assert set(shared_channel_credentials(profile_dir, home)) >= {"telegram", "discord"}
    assert set(channel_platforms_configured(profile_dir)) >= {"telegram", "discord", "whatsapp", "api_server"}
    assert gm.build_migration_plan().blocked


def test_clone_all_drops_pairing_and_platform_state(home):
    (home / "platforms" / "pairing").mkdir(parents=True)
    (home / "platforms" / "pairing" / "telegram_approved.json").write_text("{}", encoding="utf-8")
    (home / "discord_threads.json").write_text("{}", encoding="utf-8")
    (home / "memories").mkdir()
    (home / "memories" / "MEMORY.md").write_text("remember", encoding="utf-8")

    profile_dir = create_profile("full", clone_all=True, no_alias=True)

    assert not (profile_dir / "platforms").exists() and not (profile_dir / "discord_threads.json").exists()
    assert (profile_dir / "memories" / "MEMORY.md").read_text(encoding="utf-8") == "remember"
    assert _fingerprints(profile_dir) == set()


def test_strip_env_file_keeps_comments_and_unknown_keys_verbatim(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# header\nexport OPENAI_API_KEY=abc\n\nTELEGRAM_BOT_TOKEN=1:x\nMY_CUSTOM_THING=1\n", encoding="utf-8")
    removed = strip_channel_env_file(env)
    assert removed == {"telegram": ["TELEGRAM_BOT_TOKEN"]}
    assert env.read_text(encoding="utf-8") == "# header\nexport OPENAI_API_KEY=abc\n\nMY_CUSTOM_THING=1\n"
