import json
import sqlite3

from gateway.config import DiscordNativeMultibotConfig
from hermes_cli import discord_native


TOKEN_VALUE = "MTIz.fake.discord-token-value"


def _config() -> DiscordNativeMultibotConfig:
    return DiscordNativeMultibotConfig.from_dict(
        {
            "enabled": False,
            "mode": "off",
            "guild_allowlist": ["guild-1"],
            "default_intake_agent_id": "bohumil",
            "identities": [
                {
                    "agent_id": "bohumil",
                    "hermes_profile": "default",
                    "discord_application_id": "1111",
                    "discord_bot_user_id": "2222",
                    "token_secret_ref": "secret://discord/bohumil-token",
                    "capabilities": ["intake", "chat"],
                    "allowed_scopes": {"guild_ids": ["guild-1"]},
                    "enabled": True,
                },
                {
                    "agent_id": "reviewer",
                    "hermes_profile": "reviewer-profile",
                    "discord_application_id": "3333",
                    "discord_bot_user_id": "4444",
                    "token_secret_ref": "secret://discord/reviewer-token",
                    "capabilities": ["review"],
                    "allowed_scopes": {"guild_ids": ["guild-1"]},
                    "enabled": True,
                },
            ],
        }
    )


class FakePresence:
    def __init__(self, present_refs=()):
        self.present_refs = set(present_refs)

    def has_secret(self, ref: str) -> bool:
        return ref in self.present_refs


class FakeHandoffProvider:
    def __init__(self):
        self.calls = []
        self.internal_token_value = TOKEN_VALUE

    def create_handoff(self, *, identity, redacted_secret_ref):
        self.calls.append(
            {
                "agent_id": identity.agent_id,
                "raw_ref_seen_by_provider": identity.token_secret_ref,
                "redacted_secret_ref": redacted_secret_ref,
            }
        )
        return {
            "agent_id": identity.agent_id,
            "token_secret_ref": redacted_secret_ref,
            "handoff_ref": f"fake-handoff://{identity.agent_id}",
            "handoff_url": f"https://handoff.example/{identity.agent_id}",
        }


class FakeDiscordRest:
    def __init__(self):
        self.verified = []
        self.reconciled = []

    def verify_identity(self, identity):
        self.verified.append(identity["agent_id"])
        return {
            "agent_id": identity["agent_id"],
            "status": "ok",
            "bot_user_id_matches": True,
            "token": TOKEN_VALUE,
        }

    def list_guild_bots(self, guild_id):
        self.reconciled.append(guild_id)
        return [{"id": "2222"}, {"id": "9999"}]


def test_plan_snapshot_redacts_refs_and_lists_missing_secrets():
    plan = discord_native.build_plan(
        _config(),
        presence_provider=FakePresence({"secret://discord/reviewer-token"}),
    )

    assert plan == {
        "schema_version": 1,
        "command": "plan",
        "enabled": False,
        "mode": "off",
        "guild_allowlist": ["guild-1"],
        "default_intake_agent_id": "bohumil",
        "desired_identities": [
            {
                "agent_id": "bohumil",
                "hermes_profile": "default",
                "discord_application_id": "1111",
                "discord_bot_user_id": "2222",
                "token_secret_ref": "secret://<redacted>",
                "capabilities": ["intake", "chat"],
                "allowed_scopes": {"guild_ids": ["guild-1"]},
                "enabled": True,
                "will_activate": False,
            },
            {
                "agent_id": "reviewer",
                "hermes_profile": "reviewer-profile",
                "discord_application_id": "3333",
                "discord_bot_user_id": "4444",
                "token_secret_ref": "secret://<redacted>",
                "capabilities": ["review"],
                "allowed_scopes": {"guild_ids": ["guild-1"]},
                "enabled": True,
                "will_activate": False,
            },
        ],
        "missing_secrets": [
            {
                "agent_id": "bohumil",
                "token_secret_ref": "secret://<redacted>",
                "status": "missing_or_unchecked",
            }
        ],
        "network": "not_used",
        "active_gateway": "not_connected",
    }


def test_sync_persists_safe_identity_metadata_only(tmp_path):
    store_path = tmp_path / "discord-v2.sqlite3"

    result = discord_native.sync_metadata(_config(), store_path=store_path)

    assert result["command"] == "sync"
    assert result["synced_identity_count"] == 2
    with sqlite3.connect(store_path) as conn:
        rows = conn.execute(
            "SELECT agent_id, token_secret_ref FROM identity_registry ORDER BY agent_id"
        ).fetchall()
    assert rows == [
        ("bohumil", "secret://discord/bohumil-token"),
        ("reviewer", "secret://discord/reviewer-token"),
    ]
    assert TOKEN_VALUE not in store_path.read_bytes().decode("latin1", errors="ignore")


def test_handoff_missing_secrets_outputs_urls_refs_only_never_tokens(capsys):
    provider = FakeHandoffProvider()

    result = discord_native.create_missing_secret_handoffs(
        _config(),
        provider=provider,
        presence_provider=FakePresence({"secret://discord/reviewer-token"}),
    )
    discord_native._print_json(result)
    output = capsys.readouterr().out

    assert result["handoff_count"] == 1
    assert result["handoffs"][0]["handoff_url"] == "https://handoff.example/bohumil"
    assert result["handoffs"][0]["token_secret_ref"] == "secret://<redacted>"
    assert TOKEN_VALUE not in output
    assert "discord/bohumil-token" not in output
    assert "paste" not in output.lower()
    assert provider.calls == [
        {
            "agent_id": "bohumil",
            "raw_ref_seen_by_provider": "secret://discord/bohumil-token",
            "redacted_secret_ref": "secret://<redacted>",
        }
    ]


def test_verify_uses_injected_fake_client_and_sanitizes_token_output():
    client = FakeDiscordRest()

    result = discord_native.verify_identities(_config(), client=client)

    assert client.verified == ["bohumil", "reviewer"]
    assert result["network"] == "fake_client"
    assert all(item["status"] == "ok" for item in result["results"])
    assert all(item["token"] == "<redacted>" for item in result["results"])
    assert TOKEN_VALUE not in json.dumps(result)


def test_verify_without_client_is_explicitly_not_implemented_no_network():
    result = discord_native.verify_identities(_config())

    assert result["network"] == "not_used"
    assert result["operator_network"] == "not_requested"
    assert [item["status"] for item in result["results"]] == [
        "not_implemented_without_client",
        "not_implemented_without_client",
    ]


def test_operator_network_flag_is_gated_and_does_not_network():
    result = discord_native.verify_identities(_config(), operator_network=True)

    assert result["network"] == "not_used"
    assert result["operator_network"] == "requested_not_implemented"
    assert result["results"][0]["status"] == "operator_network_not_implemented"
    assert result["results"][0]["network_gate"] == "explicit_operator_flag_present_but_no_live_provider"


def test_verify_fake_client_reports_common_operator_failures():
    class FailingDiscordRest:
        def verify_identity(self, identity):
            return {
                "agent_id": identity["agent_id"],
                "bot_user_id": "wrong-bot-id",
                "guild_ids": [],
                "message_content_intent": False,
                "permissions_ok": False,
                "scopes": ["bot"],
                "token_secret_ref": "secret://discord/leaked-full-path",
            }

        def list_guild_bots(self, guild_id):  # pragma: no cover - protocol stub
            return []

    result = discord_native.verify_identities(
        _config(),
        client=FailingDiscordRest(),
        presence_provider=FakePresence(),
    )
    first = result["results"][0]

    assert first["status"] == "failed"
    assert first["checks"]["bot_user_id"]["ok"] is False
    assert first["checks"]["guild_membership"]["missing_guild_ids"] == ["guild-1"]
    assert first["checks"]["message_content_intent"]["ok"] is False
    assert first["checks"]["permissions"]["ok"] is False
    assert first["checks"]["scopes"]["missing"] == ["applications.commands"]
    assert "missing_secret" in first["problems"]
    assert "secret://discord/leaked-full-path" not in json.dumps(result)


def test_invite_links_builds_oauth_urls_from_application_ids():
    result = discord_native.build_invite_links(_config())

    assert result["network"] == "not_used"
    assert [link["agent_id"] for link in result["links"]] == ["bohumil", "reviewer"]
    assert result["links"][0]["url"] == (
        "https://discord.com/oauth2/authorize?"
        "client_id=1111&permissions=0&scope=bot+applications.commands&guild_id=guild-1"
    )


def test_reconcile_guild_compares_registry_with_injected_fake_client():
    client = FakeDiscordRest()

    result = discord_native.reconcile_guild(_config(), client=client)
    diagnostics = result.pop("diagnostics")

    assert client.reconciled == ["guild-1"]
    assert result == {
        "schema_version": 1,
        "command": "reconcile-guild",
        "guilds": [
            {
                "guild_id": "guild-1",
                "status": "compared_with_fake_client",
                "present_agent_ids": ["bohumil"],
                "missing_agent_ids": ["reviewer"],
                "extra_bot_user_ids": ["9999"],
                "guild_bot_user_ids": ["2222", "9999"],
            }
        ],
        "network": "fake_client",
        "operator_network": "not_requested",
    }
    assert diagnostics["schema_version"] == 1
    assert diagnostics["component"] == "discord_protocol_v2"
    assert diagnostics["identities"]["total"] == 2
    assert diagnostics["identities"]["connected"] == 0
    assert diagnostics["secret_refs"]["missing_or_unresolved"] == 2
    assert diagnostics["network"] == "not_used"
    assert diagnostics["token_resolution"] == "<redacted>"
    assert TOKEN_VALUE not in json.dumps(diagnostics)
    assert "secret://discord/bohumil-token" not in json.dumps(diagnostics)
    assert "secret://<redacted>" in json.dumps(diagnostics)


def _write_config_file(path):
    path.write_text(
        """
discord_native_multibot:
  enabled: false
  mode: off
  guild_allowlist: [guild-1]
  default_intake_agent_id: bohumil
  identities:
    - agent_id: bohumil
      hermes_profile: default
      discord_application_id: "1111"
      discord_bot_user_id: "2222"
      token_secret_ref: secret://discord/bohumil-token
      capabilities: [intake]
      allowed_scopes:
        guild_ids: [guild-1]
""",
        encoding="utf-8",
    )


def test_module_cli_accepts_fake_config_file_and_prints_json(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    _write_config_file(config_path)

    rc = discord_native.main(["plan", "--config", str(config_path)])
    output = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert output["command"] == "plan"
    assert output["desired_identities"][0]["agent_id"] == "bohumil"
    assert output["desired_identities"][0]["token_secret_ref"] == "secret://<redacted>"


def test_module_cli_accepts_parent_config_before_subcommand(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    _write_config_file(config_path)

    rc = discord_native.main(["--config", str(config_path), "plan"])
    output = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert output["command"] == "plan"
    assert output["desired_identities"][0]["agent_id"] == "bohumil"
    assert output["desired_identities"][0]["token_secret_ref"] == "secret://<redacted>"
