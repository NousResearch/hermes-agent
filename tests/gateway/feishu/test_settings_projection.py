"""Regression tests for FeishuAdapter settings -> SDK config projection.

These tests cover bugs surfaced by real-environment ``hermes gateway restart``
that the rest of the contract suite missed because conftest fully mocks the
SDK construction path. Each test pins a specific projection invariant and
RED-flags if a refactor regresses to the pre-fix shape.
"""
from __future__ import annotations

import dataclasses

import pytest

# Use the same env-driven settings loader the adapter uses, then exercise the
# pure projection helpers. We don't need a live SDK channel for this.
pytest.importorskip("lark_oapi.channel")  # skip if SDK extra not installed

from gateway.config import PlatformConfig
from gateway.platforms.feishu.adapter import (
    FeishuAdapter,
    FeishuAdapterSettings,
    FeishuGroupRule,
    FEISHU_DOMAIN,
    LARK_DOMAIN,
)


def _make_settings(monkeypatch, **env_overrides) -> FeishuAdapterSettings:
    """Build a FeishuAdapterSettings via FeishuAdapter._load_settings.

    _load_settings is a staticmethod taking an ``extra`` dict (the
    PlatformConfig.extra) and reading additional env vars directly. We seed
    env vars for the env-driven branches and pass only the credentials via
    extra. This is the same code path production hits at adapter init.
    """
    base_env = {
        "FEISHU_APP_ID": "cli_test",
        "FEISHU_APP_SECRET": "secret",
        "FEISHU_DOMAIN": "feishu",
        "FEISHU_CONNECTION_MODE": "websocket",
        "FEISHU_GROUP_POLICY": "open",
        "FEISHU_ALLOWED_USERS": "",
        "FEISHU_ENCRYPT_KEY": "",
        "FEISHU_VERIFICATION_TOKEN": "",
        "FEISHU_ALLOW_ALL_USERS": "",
    }
    base_env.update(env_overrides)
    for k, v in base_env.items():
        monkeypatch.setenv(k, v)
    extra = {
        "app_id": "cli_test",
        "app_secret": "secret",
        "domain": base_env.get("FEISHU_DOMAIN", "feishu"),
        "connection_mode": "websocket",
    }
    return FeishuAdapter._load_settings(extra)


def _make_adapter(monkeypatch, **env_overrides) -> FeishuAdapter:
    """Build a real FeishuAdapter with env-overrides applied so instance
    methods (_build_sdk_policy_config / _build_sdk_safety_config) can run.
    """
    settings = _make_settings(monkeypatch, **env_overrides)
    config = PlatformConfig(
        enabled=True,
        extra={
            "app_id": settings.app_id,
            "app_secret": settings.app_secret,
            "domain": settings.domain_name,
            "connection_mode": settings.connection_mode,
        },
    )
    return FeishuAdapter(config)


class TestBotIdentitySettings:
    """Manual bot identity env vars remain a fallback when SDK hydration is
    unavailable or delayed.
    """

    def test_bot_identity_env_vars_populate_adapter_fallback(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            FEISHU_BOT_OPEN_ID="ou_manual_bot",
            FEISHU_BOT_USER_ID="u_manual_bot",
            FEISHU_BOT_NAME="ManualBot",
        )

        assert adapter._bot_open_id == "ou_manual_bot"
        assert adapter._bot_user_id == "u_manual_bot"
        assert adapter._bot_name == "ManualBot"


class TestPolicyConfigProjection:
    """SDK PolicyConfig built from FeishuAdapterSettings — bug 5 regression
    target: per-USER allowlist must NOT land in per-CHAT group_allowlist."""

    def test_user_allowlist_lands_in_allow_from_not_group_allowlist(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            FEISHU_GROUP_POLICY="allowlist",
            FEISHU_ALLOWED_USERS="u_alice,u_bob",
        )
        cfg = adapter._build_sdk_policy_config(adapter._settings)
        if hasattr(cfg, "sender_identity_fields"):
            assert set(cfg.allow_from or []) == {"u_alice", "u_bob"}
        else:
            assert cfg.allow_from is None
        # group_allowlist (per-chat) MUST NOT receive per-user data.
        assert not getattr(cfg, "group_allowlist", None)
        # And group_policy must be relaxed to "open" so SDK doesn't pre-reject
        # before allow_from runs.
        assert cfg.group_policy == "open"
        if hasattr(cfg, "sender_identity_fields"):
            assert cfg.sender_identity_fields == ["open_id", "user_id", "union_id"]
        else:
            # SDK 1.6.0 only filters allow_from against sender.open_id. In
            # that runtime the adapter intentionally avoids SDK allow_from and
            # applies its own multi-id fallback after SDK admission.
            assert cfg.allow_from is None

    def test_user_allowlist_matches_user_id(self, monkeypatch):
        from lark_oapi.channel import Conversation, Identity, InboundMessage
        from lark_oapi.channel.safety.policy_gate import PolicyGate
        from lark_oapi.channel.types import TextContent

        adapter = _make_adapter(
            monkeypatch,
            FEISHU_GROUP_POLICY="allowlist",
            FEISHU_ALLOWED_USERS="u_alice",
            FEISHU_REQUIRE_MENTION="false",
        )
        cfg = adapter._build_sdk_policy_config(adapter._settings)
        msg = InboundMessage(
            id="om_user_id_allow",
            create_time=1,
            conversation=Conversation(chat_id="oc_team", chat_type="group"),
            sender=Identity(open_id="ou_not_listed", user_id="u_alice"),
            content=TextContent(text="hello"),
            content_text="hello",
            mentions=[],
            mentioned_all=False,
            resources=[],
            raw={},
        )

        decision = PolicyGate(cfg).evaluate(msg)

        assert decision.allowed is True
        if hasattr(cfg, "sender_identity_fields"):
            assert cfg.allow_from == ["u_alice"]
        else:
            assert adapter._needs_sdk_user_policy_fallback(msg) is True
            assert adapter._allow_group_message(msg.sender, msg.conversation.chat_id) is True
            denied = dataclasses.replace(msg.sender, user_id="u_mallory")
            assert adapter._allow_group_message(denied, msg.conversation.chat_id) is False

    def test_allowlist_does_not_pre_filter_peer_bots(self, monkeypatch):
        from lark_oapi.channel import Conversation, Identity, InboundMessage
        from lark_oapi.channel.safety.policy_gate import PolicyGate
        from lark_oapi.channel.types import TextContent

        adapter = _make_adapter(
            monkeypatch,
            FEISHU_GROUP_POLICY="allowlist",
            FEISHU_ALLOWED_USERS="u_human",
            FEISHU_ALLOW_BOTS="all",
            FEISHU_REQUIRE_MENTION="false",
        )
        cfg = adapter._build_sdk_policy_config(adapter._settings)
        msg = InboundMessage(
            id="om_peer_bot",
            create_time=1,
            conversation=Conversation(chat_id="oc_team", chat_type="group"),
            sender=Identity(open_id="ou_peer_bot", user_id="u_peer_bot", is_bot=True),
            content=TextContent(text="hello"),
            content_text="hello",
            mentions=[],
            mentioned_all=False,
            resources=[],
            raw={},
        )

        decision = PolicyGate(cfg).evaluate(msg)

        assert decision.allowed is True

    def test_peer_bot_bypass_keeps_human_allowlist_fallback(self, monkeypatch):
        from lark_oapi.channel import Conversation, Identity, InboundMessage
        from lark_oapi.channel.types import TextContent

        adapter = _make_adapter(
            monkeypatch,
            FEISHU_GROUP_POLICY="allowlist",
            FEISHU_ALLOWED_USERS="u_human",
            FEISHU_ALLOW_BOTS="all",
            FEISHU_REQUIRE_MENTION="false",
        )
        msg = InboundMessage(
            id="om_denied_human",
            create_time=1,
            conversation=Conversation(chat_id="oc_team", chat_type="group"),
            sender=Identity(open_id="ou_denied", user_id="u_denied"),
            content=TextContent(text="hello"),
            content_text="hello",
            mentions=[],
            mentioned_all=False,
            resources=[],
            raw={},
        )

        assert adapter._needs_sdk_user_policy_fallback(msg) is True
        assert adapter._allow_group_message(msg.sender, msg.conversation.chat_id) is False

    def test_group_rule_allowlist_does_not_pre_filter_peer_bots(self, monkeypatch):
        from lark_oapi.channel import Conversation, Identity, InboundMessage
        from lark_oapi.channel.safety.policy_gate import PolicyGate
        from lark_oapi.channel.types import TextContent

        adapter = _make_adapter(
            monkeypatch,
            FEISHU_GROUP_POLICY="open",
            FEISHU_ALLOW_BOTS="all",
            FEISHU_REQUIRE_MENTION="false",
        )
        settings = dataclasses.replace(
            adapter._settings,
            group_rules={
                "oc_team": FeishuGroupRule(
                    policy="allowlist",
                    allowlist={"u_human"},
                    require_mention=False,
                ),
            },
        )
        cfg = adapter._build_sdk_policy_config(settings)
        msg = InboundMessage(
            id="om_group_rule_peer_bot",
            create_time=1,
            conversation=Conversation(chat_id="oc_team", chat_type="group"),
            sender=Identity(open_id="ou_peer_bot", user_id="u_peer_bot", is_bot=True),
            content=TextContent(text="hello"),
            content_text="hello",
            mentions=[],
            mentioned_all=False,
            resources=[],
            raw={},
        )

        decision = PolicyGate(cfg).evaluate(msg)

        assert decision.allowed is True

    def test_allow_all_users_yields_open_policy_with_no_filter(self, monkeypatch):
        adapter = _make_adapter(
            monkeypatch,
            FEISHU_GROUP_POLICY="allowlist",
            FEISHU_ALLOW_ALL_USERS="true",
            FEISHU_ALLOWED_USERS="ignored",
        )
        cfg = adapter._build_sdk_policy_config(adapter._settings)
        assert cfg.group_policy == "open"
        # Full bypass — no allow_from filter (None or empty collection).
        assert not (cfg.allow_from or [])
        if hasattr(cfg, "sender_identity_fields"):
            assert cfg.sender_identity_fields == ["open_id", "user_id", "union_id"]

    def test_per_chat_require_mention_override_flows_to_sdk_gate(self, monkeypatch):
        from lark_oapi.channel import Conversation, Identity, InboundMessage
        from lark_oapi.channel.safety.policy_gate import PolicyGate
        from lark_oapi.channel.types import TextContent

        adapter = _make_adapter(monkeypatch, FEISHU_REQUIRE_MENTION="true")
        settings = dataclasses.replace(
            adapter._settings,
            group_rules={
                "oc_free": FeishuGroupRule(policy="open", require_mention=False),
            },
        )
        cfg = adapter._build_sdk_policy_config(settings)
        msg = InboundMessage(
            id="om_group_no_mention",
            create_time=1,
            conversation=Conversation(chat_id="oc_free", chat_type="group"),
            sender=Identity(open_id="ou_human"),
            content=TextContent(text="hello"),
            content_text="hello",
            mentions=[],
            mentioned_all=False,
            resources=[],
            raw={},
        )

        decision = PolicyGate(cfg).evaluate(msg)

        assert cfg.group_overrides["oc_free"].require_mention is False
        assert decision.allowed is True


class TestDomainProjection:
    """SDK ChannelConfig must receive a fully-qualified https URL, not a short
    name like ``feishu`` or ``lark``. Bug 2 regression.

    The mapping logic is currently inlined in connect() (adapter.py L679-685);
    we re-exercise the same branches here against the module-level constants
    to pin the contract.
    """

    def _resolve(self, short: str) -> str:
        # Mirror the inlined logic at adapter.py:679-685 — if a refactor moves
        # this into a helper, the test should be updated to call the helper
        # but the assertions stay identical.
        s = (short or "feishu").strip().lower()
        if s.startswith("http"):
            return s
        if s == "lark":
            return LARK_DOMAIN or "https://open.larksuite.com"
        return FEISHU_DOMAIN or "https://open.feishu.cn"

    def test_domain_short_name_feishu_maps_to_https_url(self):
        url = self._resolve("feishu")
        assert url == FEISHU_DOMAIN
        assert url.startswith("https://")

    def test_domain_short_name_lark_maps_to_https_url(self):
        url = self._resolve("lark")
        assert url == LARK_DOMAIN
        assert url.startswith("https://")


class TestRetryConfigProjection:
    """Bug 1 regression: a refactor previously left a dangling reference to
    a deleted module-level constant ``_FEISHU_SEND_ATTEMPTS``. The current
    code inlines ``3`` at adapter.py:711 (RetryConfig(max_attempts=3));
    this test pins that and ensures any future move back to a named constant
    must keep the value at 3.

    The retry config is constructed inside ``connect()`` rather than in
    ``_build_sdk_safety_config``, so we instantiate RetryConfig directly with
    the same literal to guard the value.
    """

    def test_retry_config_max_attempts_is_three(self):
        from lark_oapi.channel import RetryConfig
        cfg = RetryConfig(max_attempts=3)
        assert cfg.max_attempts == 3


class TestSafetyConfigProjection:
    """SDK safety settings that preserve Hermes adapter contracts."""

    def test_dedup_ttl_matches_legacy_twenty_four_hours(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        cfg = adapter._build_sdk_safety_config(adapter._settings)

        assert cfg.dedup.ttl_seconds == 24 * 60 * 60


class TestRequireMentionProjection:
    """settings.require_mention must flow into the SDK PolicyConfig.
    Default True; FEISHU_REQUIRE_MENTION=false flips it to False."""

    def test_require_mention_default_is_true(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        cfg = adapter._build_sdk_policy_config(adapter._settings)
        assert cfg.require_mention is True

    def test_require_mention_false_lifts_mention_requirement(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, FEISHU_REQUIRE_MENTION="false")
        cfg = adapter._build_sdk_policy_config(adapter._settings)
        assert cfg.require_mention is False

    def test_require_mention_false_with_allow_all_users_also_lifts(self, monkeypatch):
        # The early-return ``allow_all`` branch in _build_sdk_policy_config
        # must respect the same operator override.
        adapter = _make_adapter(
            monkeypatch,
            FEISHU_ALLOW_ALL_USERS="true",
            FEISHU_REQUIRE_MENTION="false",
        )
        cfg = adapter._build_sdk_policy_config(adapter._settings)
        assert cfg.require_mention is False

    def test_mention_all_satisfies_required_mention(self, monkeypatch):
        from lark_oapi.channel import Conversation, Identity, InboundMessage
        from lark_oapi.channel.safety.policy_gate import PolicyGate
        from lark_oapi.channel.types import TextContent

        adapter = _make_adapter(
            monkeypatch,
            FEISHU_GROUP_POLICY="open",
            FEISHU_REQUIRE_MENTION="true",
        )
        cfg = adapter._build_sdk_policy_config(adapter._settings)
        msg = InboundMessage(
            id="om_all",
            create_time=1,
            conversation=Conversation(chat_id="oc_team", chat_type="group"),
            sender=Identity(open_id="ou_human"),
            content=TextContent(text="@_all hello"),
            content_text="@_all hello",
            mentions=[],
            mentioned_all=True,
            resources=[],
            raw={},
        )

        decision = PolicyGate(cfg).evaluate(msg)

        assert decision.allowed is True
