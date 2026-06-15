from __future__ import annotations

import json
from types import SimpleNamespace

from gateway.config import DiscordNativeMultibotConfig, DiscordNativeMultibotIdentityConfig
from gateway.discord_identity_registry import DiscordIdentityRegistry
from gateway.discord_protocol_v2_ingest import DiscordProtocolV2Ingestor
from gateway.discord_protocol_v2_store import DISCORD_SOURCE_TYPE, DiscordProtocolV2Store
from gateway.secret_refs import SensitiveToken


class ExplodingResolver:
    calls = 0

    def resolve(self, ref: str) -> SensitiveToken:  # pragma: no cover - safety guard
        self.calls += 1
        raise AssertionError(f"shadow replay must not resolve token refs: {ref}")


def _native_config() -> DiscordNativeMultibotConfig:
    return DiscordNativeMultibotConfig(
        enabled=True,
        mode="shadow",
        guild_allowlist=["guild-shadow"],
        default_intake_agent_id="bohumil",
        identities=[
            DiscordNativeMultibotIdentityConfig(
                agent_id="bohumil",
                hermes_profile="bohumil-profile",
                discord_application_id="app-bohumil",
                discord_bot_user_id="bot-bohumil",
                token_secret_ref="secret://discord/bohumil-token",
                capabilities=["intake", "chat"],
                enabled=True,
            ),
            DiscordNativeMultibotIdentityConfig(
                agent_id="reviewer",
                hermes_profile="reviewer-profile",
                discord_application_id="app-reviewer",
                discord_bot_user_id="bot-reviewer",
                token_secret_ref="secret://discord/reviewer-token",
                capabilities=["chat"],
                enabled=True,
            ),
        ],
    )


def _ingestor(store: DiscordProtocolV2Store, resolver: ExplodingResolver) -> DiscordProtocolV2Ingestor:
    cfg = _native_config()
    registry = DiscordIdentityRegistry.load(cfg, store, secret_resolver=resolver)
    return DiscordProtocolV2Ingestor(
        store=store,
        identity_registry=registry,
        default_intake_agent_id=cfg.default_intake_agent_id,
        guild_allowlist=cfg.guild_allowlist,
        mode=cfg.mode,
    )


def _mention(user_id: str, *, bot: bool = True) -> SimpleNamespace:
    return SimpleNamespace(id=user_id, bot=bot)


def _message(
    *,
    message_id: str,
    author_id: str = "human-shadow",
    author_bot: bool = False,
    mentions: list[SimpleNamespace] | None = None,
    content: str = "please route this",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        guild=SimpleNamespace(id="guild-shadow"),
        channel=SimpleNamespace(id="channel-shadow", name="staging", parent_id=None),
        author=SimpleNamespace(id=author_id, bot=author_bot),
        mentions=list(mentions or []),
        webhook_id=None,
        type="default",
        content=content,
        attachments=[],
    )


def _targets(decision: dict) -> list[str]:
    return json.loads(decision["target_agent_ids_json"])


def test_shadow_replay_records_route_decisions_without_deliveries_outbox_or_tokens(tmp_path):
    resolver = ExplodingResolver()
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        ingestor = _ingestor(store, resolver)

        explicit = ingestor.ingest_message(
            source_client_agent_id="bohumil",
            message=_message(
                message_id="shadow-explicit",
                mentions=[_mention("bot-reviewer")],
                content="reviewer please inspect",
            ),
        )
        default = ingestor.ingest_message(
            source_client_agent_id="reviewer",
            message=_message(message_id="shadow-default"),
        )
        bot_loop = ingestor.ingest_message(
            source_client_agent_id="bohumil",
            message=_message(
                message_id="shadow-bot-loop",
                author_id="bot-bohumil",
                author_bot=True,
                mentions=[_mention("bot-reviewer")],
            ),
        )

        assert explicit.deliveries == []
        assert default.deliveries == []
        assert bot_loop.deliveries == []
        assert resolver.calls == 0
        assert store.count_rows("topics") == 1
        assert store.count_rows("message_map") == 3
        assert store.count_rows("route_decisions") == 3
        assert store.count_rows("inbound_deliveries") == 0
        assert store.count_rows("outbox_deliveries") == 0
        assert store.count_rows("outbox_parts") == 0

        explicit_decision = store.route_decisions_for(DISCORD_SOURCE_TYPE, "shadow-explicit")[0]
        assert explicit_decision["decision"] == "delivered"
        assert explicit_decision["reason"] == "explicit_mention"
        assert _targets(explicit_decision) == ["reviewer"]
        explicit_payload = json.loads(explicit_decision["payload_json"])
        assert explicit_payload["shadow_replay"] is True
        assert explicit_payload["shadow_no_delivery"] is True

        default_decision = store.route_decisions_for(DISCORD_SOURCE_TYPE, "shadow-default")[0]
        assert default_decision["decision"] == "delivered"
        assert default_decision["reason"] == "default_intake"
        assert _targets(default_decision) == ["bohumil"]

        loop_decision = store.route_decisions_for(DISCORD_SOURCE_TYPE, "shadow-bot-loop")[0]
        assert loop_decision["decision"] == "zero_delivery"
        assert loop_decision["reason"] == "non_human_author:registered_bot"
        assert _targets(loop_decision) == []

        mapped = store.get_message_map("shadow-explicit")
        assert mapped is not None
        assert mapped["direction"] == "inbound"
        assert mapped["source_client_agent_id"] == "bohumil"
        mapped_payload = json.loads(mapped["payload_json"])
        assert mapped_payload["shadow_replay"] is True
        assert mapped_payload["shadow_target_agent_ids"] == ["reviewer"]


def test_shadow_replay_guild_allowlist_ignores_unscoped_messages_without_rows(tmp_path):
    resolver = ExplodingResolver()
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        result = _ingestor(store, resolver).ingest_message(
            source_client_agent_id="bohumil",
            message=SimpleNamespace(
                id="shadow-other-guild",
                guild=SimpleNamespace(id="guild-other"),
                channel=SimpleNamespace(id="channel-shadow", name="staging", parent_id=None),
                author=SimpleNamespace(id="human-shadow", bot=False),
                mentions=[_mention("bot-bohumil")],
                webhook_id=None,
                type="default",
                content="out of scope",
                attachments=[],
            ),
        )

        assert result.ignored is True
        assert result.reason == "guild_not_allowed"
        assert resolver.calls == 0
        assert store.count_rows("topics") == 0
        assert store.count_rows("message_map") == 0
        assert store.count_rows("route_decisions") == 0
        assert store.count_rows("inbound_deliveries") == 0
        assert store.count_rows("outbox_deliveries") == 0
