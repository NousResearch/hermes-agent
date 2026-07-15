from __future__ import annotations

import json
from pathlib import Path

from gateway.discord_history_authority import reviewed_cron_history_targets_json


ROOT = Path(__file__).resolve().parents[3]


def test_connector_unit_is_immutable_hardened_and_only_token_owner() -> None:
    unit = (ROOT / "ops/muncho/systemd/muncho-discord-connector.service.in").read_text(
        encoding="utf-8"
    )
    gateway = (
        ROOT / "ops/muncho/systemd/hermes-cloud-gateway.discord-connector.conf"
    ).read_text(encoding="utf-8")
    assert "User=muncho-discord-connector" in unit
    assert "@EXACT_12_CHAR_SHA@" in unit
    assert "/opt/muncho-releases/" not in unit
    assert (
        "/opt/adventico-ai-platform/hermes-agent-releases/"
        "hermes-agent-@EXACT_12_CHAR_SHA@/.venv/bin/python"
    ) in unit
    assert (
        "Environment=PYTHONPATH=/opt/adventico-ai-platform/"
        "hermes-agent-releases/hermes-agent-@EXACT_12_CHAR_SHA@"
    ) in unit
    assert " -B -P -s -m gateway.discord_connector_bootstrap" in unit
    assert "-m gateway.discord_connector_bootstrap" in unit
    assert "Type=notify" in unit
    assert "NotifyAccess=main" in unit
    assert "Restart=on-failure" in unit
    assert "RestartSec=5s" in unit
    assert "RuntimeMaxSec=" not in unit
    assert "TimeoutStartSec=150s" in unit
    assert "TimeoutStopSec=60s" in unit
    assert "NoNewPrivileges=yes" in unit
    assert "CapabilityBoundingSet=" in unit
    assert "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6" in unit
    assert "discord-connector-credentials/bot-token" in unit
    assert "discord-connector-credentials/bot-token" not in gateway
    assert "BindsTo=muncho-discord-connector.service" in gateway
    assert (
        "GATEWAY_RELAY_URL=unix:///run/muncho-discord-connector/connector.sock"
        in gateway
    )
    assert "UnsetEnvironment=DISCORD_BOT_TOKEN DISCORD_TOKEN" in gateway


def test_package_config_has_production_guild_acl_policy_and_no_embedded_token() -> None:
    config = (ROOT / "ops/muncho/systemd/discord-public-connector.json.in").read_text(
        encoding="utf-8"
    )
    assert (
        '"token_file": "/etc/muncho/discord-connector-credentials/bot-token"' in config
    )
    assert '"allowed_guild_ids"' in config
    assert '"allowed_channel_ids"' in config
    assert '"allowed_user_ids"' in config
    assert '"free_response_channel_ids"' in config
    assert '"public_only": false' in config
    assert '"author_policy": "guild_acl"' in config
    assert '"1504852355588423801"' in config
    assert '"1505499746939174993"' in config
    assert "1526870121677848636" not in config
    assert '"allow_bot_authors": false' in config
    assert '"token":' not in config
    assert '"bot_token":' not in config

    rendered = config
    for placeholder in ("@GATEWAY_UID@", "@CONNECTOR_UID@", "@CONNECTOR_GID@"):
        rendered = rendered.replace(placeholder, "1000")
    parsed = json.loads(rendered)
    assert "canary_history_reader" not in parsed["service"]
    assert parsed["discord"]["reviewed_cron_history_targets"] == (
        reviewed_cron_history_targets_json()
    )
