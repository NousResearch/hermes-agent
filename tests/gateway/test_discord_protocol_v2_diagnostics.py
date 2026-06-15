"""Diagnostics contract tests for Discord protocol v2."""

from __future__ import annotations

import json

from gateway.config import (
    DiscordNativeMultibotConfig,
    DiscordNativeMultibotIdentityConfig,
)
from gateway.discord_protocol_v2_diagnostics import (
    build_health_snapshot,
    sanitize_diagnostics,
)
from gateway.discord_protocol_v2_store import DiscordProtocolV2Store
from hermes_cli.discord_native import reconcile_guild, verify_identities


RAW_TOKEN = "abc.def.ghijklmnopqrstuvwxyz"


class SecretPresence:
    def __init__(self, present: set[str]) -> None:
        self.present = present
        self.calls: list[str] = []

    def has_secret(self, ref: str) -> bool:
        self.calls.append(ref)
        return ref in self.present


class FakeDiscordClient:
    def verify_identity(self, identity: dict[str, object]) -> dict[str, object]:
        return {
            "agent_id": identity["agent_id"],
            "bot_user_id": identity["discord_bot_user_id"],
            "status": "ok",
            "token": RAW_TOKEN,
            "nested": {"secret_ref": "secret://hermes/discord/leaked"},
        }

    def list_guild_bots(self, guild_id: str) -> list[dict[str, str]]:
        assert guild_id == "333333333333333333"
        return [
            {"id": "222222222222222222", "token": RAW_TOKEN},
            {"id": "999999999999999999"},
        ]


def _config() -> DiscordNativeMultibotConfig:
    return DiscordNativeMultibotConfig(
        enabled=True,
        mode="listen_only",
        guild_allowlist=["333333333333333333"],
        default_intake_agent_id="bohumil",
        identities=[
            DiscordNativeMultibotIdentityConfig(
                agent_id="bohumil",
                hermes_profile="default",
                discord_application_id="111111111111111111",
                discord_bot_user_id="222222222222222222",
                token_secret_ref="secret://hermes/discord/bohumil-token",
                capabilities=["intake", "reply"],
                enabled=True,
            ),
            DiscordNativeMultibotIdentityConfig(
                agent_id="karel",
                hermes_profile="default",
                discord_application_id="444444444444444444",
                discord_bot_user_id="555555555555555555",
                token_secret_ref="secret://hermes/discord/karel-token",
                capabilities=["consult"],
                enabled=True,
            ),
        ],
    )


def _store(tmp_path) -> DiscordProtocolV2Store:
    return DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3")


def _seed_store(store: DiscordProtocolV2Store) -> None:
    store.upsert_identity(
        agent_id="bohumil",
        hermes_profile="default",
        discord_application_id="111111111111111111",
        discord_bot_user_id="222222222222222222",
        token_secret_ref="secret://hermes/discord/bohumil-token",
        capabilities=["intake", "reply"],
        scopes={"guild_ids": ["333333333333333333"]},
        enabled=True,
    )
    store.upsert_identity(
        agent_id="karel",
        hermes_profile="default",
        discord_application_id="444444444444444444",
        discord_bot_user_id="555555555555555555",
        token_secret_ref="secret://hermes/discord/karel-token",
        capabilities=["consult"],
        scopes={"guild_ids": ["333333333333333333"]},
        enabled=True,
    )
    store.upsert_topic(
        topic_id="topic-1",
        guild_id="333333333333333333",
        channel_id="777777777777777777",
        title="incident triage",
    )
    deliveries = store.create_discord_inbound_deliveries(
        discord_message_id="888888888888888888",
        guild_id="333333333333333333",
        channel_id="777777777777777777",
        topic_id="topic-1",
        author_id="121212121212121212",
        author_kind="human",
        target_agent_ids=["bohumil"],
        route_reason="mention",
        payload={"content": "hello"},
    )
    assert deliveries
    leased = store.lease_next_inbound(lease_owner="worker", lease_seconds=60)
    assert leased is not None
    store.conn.execute(
        "UPDATE inbound_deliveries SET lease_until = ? WHERE delivery_key = ?",
        ("2000-01-01T00:00:00+00:00", leased["delivery_key"]),
    )
    outbox = store.create_outbox_delivery(
        idempotency_key="response:diagnostics:bohumil",
        target_agent_id="bohumil",
        topic_id="topic-1",
        channel_id="777777777777777777",
        delivery_kind="response",
        payload={"content": "reply"},
    )
    leased_outbox = store.lease_next_outbox(lease_owner="sender", lease_seconds=60)
    assert leased_outbox is not None
    store.mark_outbox_uncertain(outbox["outbox_delivery_id"])
    store.create_reconciliation_run(
        reconciliation_run_id="reconcile:test:latest",
        outbox_delivery_id=outbox["outbox_delivery_id"],
        status="retry_exhausted",
        payload={"token": RAW_TOKEN, "secret_ref": "secret://hermes/discord/leaked"},
    )


def _as_json(data: dict[str, object]) -> str:
    return json.dumps(data, sort_keys=True)


def test_health_snapshot_contains_ids_counts_and_redacts_secrets(tmp_path):
    config = _config()
    presence = SecretPresence({"secret://hermes/discord/bohumil-token"})
    with _store(tmp_path) as store:
        _seed_store(store)
        snapshot = build_health_snapshot(
            config,
            store=store,
            presence_provider=presence,
            runtime_identity_state={
                "bohumil": {"connected": True, "status": "ready"},
                "karel": {"unhealthy": True, "last_error": f"token={RAW_TOKEN}"},
            },
        )

    assert snapshot["identities"]["connected"] == 1
    assert snapshot["identities"]["unhealthy"] == 1
    assert snapshot["state_counts"]["pending_inbound"] == 0
    assert snapshot["state_counts"]["outbox"]["uncertain"] == 1
    assert snapshot["expired_leases"]["inbound"] == 1
    assert snapshot["uncertain_sends"]["count"] == 1
    assert snapshot["last_reconciliation"]["status"] == "retry_exhausted"
    assert snapshot["secret_refs"]["missing_or_unresolved"] == 1

    rendered = _as_json(snapshot)
    assert "bohumil" in rendered
    assert "karel" in rendered
    assert "222222222222222222" in rendered
    assert "555555555555555555" in rendered
    assert RAW_TOKEN not in rendered
    assert "secret://hermes/discord/bohumil-token" not in rendered
    assert "secret://<redacted>" in rendered
    assert presence.calls == [
        "secret://hermes/discord/bohumil-token",
        "secret://hermes/discord/karel-token",
    ]


def test_cli_verify_and_reconcile_use_shared_diagnostics_core(tmp_path):
    config = _config()
    client = FakeDiscordClient()
    with _store(tmp_path) as store:
        _seed_store(store)
        verify = verify_identities(config, client=client, store=store)
        reconcile = reconcile_guild(config, client=client, store=store)

    assert verify["diagnostics"]["component"] == "discord_protocol_v2"
    assert reconcile["diagnostics"]["component"] == "discord_protocol_v2"
    assert verify["diagnostics"]["identities"]["total"] == 2
    assert reconcile["diagnostics"]["uncertain_sends"]["count"] == 1

    rendered = _as_json({"verify": verify, "reconcile": reconcile})
    assert "bohumil" in rendered
    assert "222222222222222222" in rendered
    assert RAW_TOKEN not in rendered
    assert "secret://hermes/discord/leaked" not in rendered


def test_sanitize_diagnostics_redacts_token_like_values_and_secret_refs():
    sanitized = sanitize_diagnostics(
        {
            "message": f"failed with token={RAW_TOKEN}",
            "token_secret_ref": "secret://hermes/discord/bohumil-token",
            "idempotency_key": "response:diagnostics:bohumil",
            "scope_key": "guild:333333333333333333",
        }
    )

    rendered = _as_json(sanitized)
    assert RAW_TOKEN not in rendered
    assert "secret://hermes/discord/bohumil-token" not in rendered
    assert sanitized["idempotency_key"] == "response:diagnostics:bohumil"
    assert sanitized["scope_key"] == "guild:333333333333333333"
