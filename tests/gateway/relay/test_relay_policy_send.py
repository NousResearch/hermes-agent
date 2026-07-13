"""Unit tests for the gateway-side relay relevance-policy declaration (Phase 6 ζ).

Covers gateway.relay.relay_relevance_policy() (the projection of the agent's
mention-gating / free-response / allow-bots config into the connector's generic
vocabulary) and send_relay_policy() (the boot-time POST to /relay/policy). The
connector HTTP POST is monkeypatched; the cross-repo E2E (connector repo,
gateway_policy_driver.py) exercises the real route. These prove the PROJECTION
mapping, the auth/skip logic, and the fail-soft boot behaviour.
"""

from __future__ import annotations

import pytest

import gateway.relay as relay


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        "GATEWAY_RELAY_URL",
        "GATEWAY_RELAY_ID",
        "GATEWAY_RELAY_SECRET",
        "GATEWAY_RELAY_PLATFORM",
        "GATEWAY_RELAY_BOT_ID",
        "GATEWAY_RELAY_PLATFORMS",
        "GATEWAY_RELAY_BOT_IDS",
        "DISCORD_ALLOW_BOTS",
        "DISCORD_ALLOWED_BOT_IDS",
        "SLACK_ALLOW_BOTS",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {}, raising=False)


# --------------------------------------------------------------------------
# relay_relevance_policy() — the projection
# --------------------------------------------------------------------------

def test_projection_maps_require_mention_and_free_response(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"discord": {"require_mention": True, "free_response_channels": ["c-support", "c-help"]}},
        raising=False,
    )
    pol = relay.relay_relevance_policy()
    assert pol == {
        "platform": "discord",
        "requireAddress": True,
        "freeResponseScopes": ["c-support", "c-help"],
        "allowOtherBots": False,
    }


def test_projection_discord_refuses_other_bots_even_when_enabled_locally(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")
    monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")
    monkeypatch.setenv("DISCORD_ALLOWED_BOT_IDS", "123456789012345678")
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"discord": {"require_mention": True}},
        raising=False,
    )
    pol = relay.relay_relevance_policy()
    assert pol is not None and pol["allowOtherBots"] is False


def test_projection_non_discord_platform_preserves_allow_other_bots(monkeypatch):
    monkeypatch.setenv("SLACK_ALLOW_BOTS", "all")
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"slack": {"require_mention": True}},
        raising=False,
    )
    pol = relay.relay_relevance_policy("slack")
    assert pol is not None and pol["allowOtherBots"] is True


def test_projection_comma_string_free_response(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"discord": {"free_response_channels": "c1, c2 ,c3"}},
        raising=False,
    )
    pol = relay.relay_relevance_policy()
    assert pol is not None and pol["freeResponseScopes"] == ["c1", "c2", "c3"]


def test_projection_falls_back_to_top_level_require_mention(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"require_mention": True},  # top-level, no discord: block
        raising=False,
    )
    pol = relay.relay_relevance_policy()
    assert pol is not None and pol["requireAddress"] is True


def test_projection_discord_declares_fail_closed_default(monkeypatch):
    # Discord must actively overwrite a stale connector policy that once
    # allowed bots, even when every local relevance knob is now at default.
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"discord": {}}, raising=False)
    assert relay.relay_relevance_policy() == {
        "platform": "discord",
        "requireAddress": False,
        "freeResponseScopes": [],
        "allowOtherBots": False,
    }


def test_projection_none_when_non_discord_all_default(monkeypatch):
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"slack": {}}, raising=False)
    assert relay.relay_relevance_policy("slack") is None


def test_projection_none_when_platform_unresolved(monkeypatch):
    # Default platform "relay" ⇒ no concrete fronted platform ⇒ nothing to project.
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"discord": {"require_mention": True}},
        raising=False,
    )
    assert relay.relay_relevance_policy() is None


# --------------------------------------------------------------------------
# send_relay_policy() — the boot-time declaration
# --------------------------------------------------------------------------

def _arm(monkeypatch, *, url="wss://connector.example/relay"):
    monkeypatch.setenv("GATEWAY_RELAY_URL", url)
    monkeypatch.setenv("GATEWAY_RELAY_ID", "gw-x")
    monkeypatch.setenv("GATEWAY_RELAY_SECRET", "s" * 48)
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")


def test_send_posts_projected_policy_with_token(monkeypatch):
    _arm(monkeypatch)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"discord": {"require_mention": True, "free_response_channels": ["c-support"]}},
        raising=False,
    )
    captured = {}

    def _fake_post(*, policy_url, token, policy, timeout=15.0):
        captured["policy_url"] = policy_url
        captured["token"] = token
        captured["policy"] = policy
        return 200

    monkeypatch.setattr(relay, "_post_policy", _fake_post)
    assert relay.send_relay_policy() is True
    assert captured["policy_url"] == "https://connector.example/relay/policy"
    assert captured["token"]  # a real upgrade token was minted
    assert captured["policy"]["requireAddress"] is True
    assert captured["policy"]["freeResponseScopes"] == ["c-support"]


def test_send_skips_when_no_secret(monkeypatch):
    monkeypatch.setenv("GATEWAY_RELAY_URL", "wss://connector.example/relay")
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "discord")
    # no GATEWAY_RELAY_ID / SECRET
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"discord": {"require_mention": True}},
        raising=False,
    )
    called = {"n": 0}
    monkeypatch.setattr(relay, "_post_policy", lambda **k: called.__setitem__("n", called["n"] + 1) or 200)
    assert relay.send_relay_policy() is False
    assert called["n"] == 0  # never attempted without a secret to auth with


def test_send_posts_discord_fail_closed_default(monkeypatch):
    _arm(monkeypatch)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"discord": {}}, raising=False)
    captured = {}

    def _fake_post(**kwargs):
        captured.update(kwargs)
        return 200

    monkeypatch.setattr(relay, "_post_policy", _fake_post)
    assert relay.send_relay_policy() is True
    assert captured["policy"]["allowOtherBots"] is False


def test_send_fail_soft_on_transport_error(monkeypatch):
    _arm(monkeypatch)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"discord": {"require_mention": True}},
        raising=False,
    )

    def _boom(**kwargs):
        raise RuntimeError("connector unreachable")

    monkeypatch.setattr(relay, "_post_policy", _boom)
    # Never raises; returns False so boot proceeds.
    assert relay.send_relay_policy() is False


def test_send_fail_soft_on_non_200(monkeypatch):
    _arm(monkeypatch)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"discord": {"require_mention": True}},
        raising=False,
    )
    monkeypatch.setattr(relay, "_post_policy", lambda **k: 401)
    assert relay.send_relay_policy() is False


def test_send_skips_when_relay_unconfigured(monkeypatch):
    # No GATEWAY_RELAY_URL ⇒ relay not configured ⇒ no-op.
    monkeypatch.setattr(relay, "_post_policy", lambda **k: 200)
    assert relay.send_relay_policy() is False
