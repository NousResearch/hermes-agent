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
    monkeypatch.setattr(gm, "_installed_services", lambda home: [])
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


# --- post-merge review of #109502 (gaoanze888): clone safety + ownership-based inventory ---------


def test_clone_all_never_writes_through_a_symlinked_source_env(home, tmp_path):
    """A source whose ``.env`` is a symlink (shared secrets file) must keep its bot token: the clone
    materializes the link before stripping instead of editing the target through it."""
    shared = tmp_path / "shared-secrets.env"
    shared.write_text(_SOURCE_ENV, encoding="utf-8")
    (home / ".env").unlink()
    (home / ".env").symlink_to(shared)

    profile_dir = create_profile("full", clone_all=True, no_alias=True)

    assert shared.read_text(encoding="utf-8") == _SOURCE_ENV
    assert not (profile_dir / ".env").is_symlink()
    assert "TELEGRAM_BOT_TOKEN" not in (profile_dir / ".env").read_text(encoding="utf-8")


def test_clone_is_published_atomically_after_stripping(home, monkeypatch):
    """The multiplexer enumerates ``profiles/`` while a clone is built; the final directory must not
    exist (and no listable profile may appear) until the channel strip has run."""
    from hermes_cli import profile_channels, profiles
    seen = {}
    real_strip = profile_channels.strip_channel_settings

    def _observing_strip(profile_dir, **kw):
        seen["final_exists"] = (home / "profiles" / "bot2").exists()
        seen["served"] = [n for n, _ in profiles.profiles_to_serve(multiplex=True)]
        seen["work_dir_hidden"] = profile_dir.name.startswith(".")
        return real_strip(profile_dir, **kw)

    monkeypatch.setattr(profile_channels, "strip_channel_settings", _observing_strip)
    profile_dir = create_profile("bot2", clone_config=True, no_alias=True)

    assert seen == {"final_exists": False, "served": ["default"], "work_dir_hidden": True}
    assert profile_dir.is_dir() and [n for n, _ in profiles.profiles_to_serve(multiplex=True)] == ["default", "bot2"]
    assert not [p for p in (home / "profiles").iterdir() if p.name.startswith(".")]


def test_clone_channels_refusal_lives_in_create_profile(home, monkeypatch):
    """REST/TUI call ``create_profile`` directly: the live-multiplexer refusal must fire there, not
    only in the CLI, and ``--clone-channels`` without a clone source is an error, not a no-op."""
    from hermes_cli import gateway_multiplex_served as served_mod
    monkeypatch.setattr(served_mod, "recorded_served_profiles", lambda root=None: ["default", "other"])
    with pytest.raises(ValueError, match="already serves"):
        create_profile("twin", clone_config=True, no_alias=True, clone_channels=True)
    assert not (home / "profiles" / "twin").exists()
    with pytest.raises(ValueError, match="only applies to a clone"):
        create_profile("plain", no_alias=True, clone_channels=True)
    # Without --clone-channels the same clone succeeds (the refusal is about the copy, not the clone).
    assert create_profile("twin", clone_config=True, no_alias=True).is_dir()


_SHARED_ENV = (
    "OPENAI_API_KEY=sk\n"
    "GATEWAY_ALLOW_ALL_USERS=true\nGATEWAY_ALLOWED_USERS=1,2\n"
    "GATEWAY_RELAY_ID=gw-1\nGATEWAY_RELAY_SECRET=s\nGATEWAY_RELAY_DELIVERY_KEY=k\n"
    "WECOM_DM_POLICY=open\nSMS_WEBHOOK_PORT=8700\n"
    "HASS_TOKEN=hass-tool-token\nHASS_URL=http://ha.local\n"
    "TWILIO_ACCOUNT_SID=AC1\nTWILIO_AUTH_TOKEN=tw\nTWILIO_PHONE_NUMBER=+1\n"
    "EMAIL_ADDRESS=a@b\nEMAIL_PASSWORD=p\nEMAIL_SMTP_HOST=smtp\nEMAIL_IMAP_HOST=imap\nEMAIL_ALLOWED_USERS=x@y\n"
)


def test_ownership_inventory_strips_policy_relay_and_aliases_but_keeps_tool_credentials(home):
    """Gateway-wide policy, relay identity and alias-prefixed keys are channel settings. HASS/TWILIO/EMAIL
    credentials are shared with tools: they leave with the channel only when the source's gateway would
    run that adapter (email here — complete creds, not disabled); an explicitly disabled channel means
    the key is a tool credential and stays. Its policy keys (allowlists, ports) go regardless."""
    (home / ".env").write_text(_SHARED_ENV, encoding="utf-8")
    (home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "gpt-5", "provider": "openai"},
                        "platforms": {"homeassistant": {"enabled": False}, "sms": {"enabled": False}}}),
        encoding="utf-8")

    profile_dir = create_profile("bot3", clone_config=True, no_alias=True)

    keys = {line.split("=", 1)[0] for line in (profile_dir / ".env").read_text(encoding="utf-8").splitlines()
            if "=" in line and not line.startswith("#")}
    assert keys == {"OPENAI_API_KEY", "HASS_TOKEN", "HASS_URL",
                    "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER"}


def test_string_false_adapter_flag_keeps_shared_tool_credentials(home):
    """Quoted false is the same disabled state the gateway applies at runtime, so HASS credentials
    remain tool-owned instead of being stripped as Home Assistant channel identity."""
    (home / ".env").write_text(_SHARED_ENV, encoding="utf-8")
    (home / "config.yaml").write_text(
        'model:\n  default: gpt-5\nplatforms:\n  homeassistant:\n    enabled: "false"\n',
        encoding="utf-8",
    )

    profile_dir = create_profile("tool-home", clone_config=True, no_alias=True)

    env_text = (profile_dir / ".env").read_text(encoding="utf-8")
    assert "HASS_TOKEN=hass-tool-token" in env_text
    assert "HASS_URL=http://ha.local" in env_text


@pytest.mark.parametrize(
    ("source_enabled", "ambient_enabled", "credentials_survive"),
    [("true", "false", False), ("false", "true", True), (None, None, False)],
    ids=["source-true", "source-false", "unresolved-conservative"],
)
def test_clone_resolves_shared_adapter_flag_in_the_source_secret_scope(
    home, monkeypatch, source_enabled, ambient_enabled, credentials_survive,
):
    """A source-local ref wins over a conflicting ambient value; an unresolved ref strips
    conservatively so a later resolution cannot make the clone reuse channel credentials."""
    source_env = _SHARED_ENV
    if source_enabled is not None:
        source_env += f"HA_CHANNEL_ENABLED={source_enabled}\n"
        monkeypatch.setenv("HA_CHANNEL_ENABLED", ambient_enabled)
    else:
        monkeypatch.delenv("HA_CHANNEL_ENABLED", raising=False)
    (home / ".env").write_text(source_env, encoding="utf-8")
    (home / "config.yaml").write_text(
        "model:\n  default: gpt-5\nplatforms:\n  homeassistant:\n"
        "    enabled: ${HA_CHANNEL_ENABLED}\n",
        encoding="utf-8",
    )

    profile_dir = create_profile("scoped-home", clone_config=True, no_alias=True)

    env_text = (profile_dir / ".env").read_text(encoding="utf-8")
    assert ("HASS_TOKEN=hass-tool-token" in env_text) is credentials_survive
    if not credentials_survive:
        assert not {claim for claim in _fingerprints(profile_dir) if claim[0] == "homeassistant"}


@pytest.mark.parametrize(("enabled", "credentials_survive"), [(True, False), (False, True)])
def test_clone_honors_managed_shared_adapter_ownership(home, tmp_path, monkeypatch, enabled, credentials_survive):
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(
        yaml.safe_dump({"platforms": {"homeassistant": {"enabled": enabled}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    (home / ".env").write_text(_SHARED_ENV, encoding="utf-8")
    (home / "config.yaml").write_text("model:\n  default: gpt-5\n", encoding="utf-8")

    profile_dir = create_profile("managed-home", clone_config=True, no_alias=True)

    env_text = (profile_dir / ".env").read_text(encoding="utf-8")
    assert ("HASS_TOKEN=hass-tool-token" in env_text) is credentials_survive


def test_clone_all_drops_directory_shaped_channel_state(home):
    (home / "google_chat_user_tokens").mkdir()
    (home / "google_chat_user_tokens" / "u@x.json").write_text("{}", encoding="utf-8")
    profile_dir = create_profile("full", clone_all=True, no_alias=True)
    assert not (profile_dir / "google_chat_user_tokens").exists()
