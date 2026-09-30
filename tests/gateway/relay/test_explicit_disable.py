"""``platforms.relay.enabled: false`` beats a deployment-injected relay URL.

Real profile files under a temp HERMES_HOME, the production startup hook and the
standalone predicate. Only the connector HTTP calls are replaced.
"""
from unittest.mock import Mock
import json
import os

import pytest
import hermes_yaml as yaml

import gateway.relay as relay
from gateway.config import Platform, load_gateway_config
from gateway.platform_registry import platform_registry
from gateway.run_startup import GatewayStartupMixin

URL = "wss://connector.example/relay"
DISABLE = {"relay": {"enabled": False}}

# Every place the loader accepts a relay block, plus the managed overlay. The verdict must be
# the same whichever one the operator used.
SPELLINGS = {
    "top": {"platforms": DISABLE},
    "nested": {"gateway": {"platforms": DISABLE}},
    "string-false": {"platforms": {"relay": {"enabled": "false"}}},
    "top-beats-nested": {"platforms": DISABLE, "gateway": {"platforms": {"relay": {"enabled": True}}}},
    "managed": None,  # user YAML absent; disable comes from HERMES_MANAGED_DIR
}


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in list(os.environ):
        if key.startswith("GATEWAY_RELAY_"):
            monkeypatch.delenv(key)
    # A deployment stamp: URL + identity provider, no pinned secret, so an
    # un-vetoed boot would resolve a token and provision credentials.
    monkeypatch.setenv("GATEWAY_RELAY_URL", URL)
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "slack")
    monkeypatch.setenv("GATEWAY_RELAY_IDP_TOKEN_URL", "https://identity.example/token")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "native-test-token")
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(GatewayStartupMixin, "_register_config_hooks", lambda *a, **k: None)
    platform_registry.unregister("relay")
    yield tmp_path
    platform_registry.unregister("relay")


def write_disable(home, spelling, monkeypatch):
    doc = SPELLINGS[spelling]
    if doc is None:
        managed = home / "managed"
        managed.mkdir()
        (managed / "config.yaml").write_text(yaml.safe_dump({"platforms": DISABLE}))
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        return
    (home / "config.yaml").write_text(yaml.safe_dump(doc))


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_disabled_relay_startup_hook_has_no_side_effects(profile, monkeypatch, spelling):
    write_disable(profile, spelling, monkeypatch)
    http = Mock(side_effect=AssertionError("disabled relay reached the connector"))
    monkeypatch.setattr("urllib.request.urlopen", http)
    monkeypatch.setattr(relay, "_resolve_relay_identity_token", http)
    before = {k: v for k, v in os.environ.items() if k.startswith("GATEWAY_RELAY_")}

    GatewayStartupMixin._start_register_plugins_relay_hooks()

    http.assert_not_called()
    assert not platform_registry.is_registered("relay")
    assert {k: v for k, v in os.environ.items() if k.startswith("GATEWAY_RELAY_")} == before
    # The test seam and an explicit URL do not override the operator either.
    assert relay.register_relay_adapter(force=True, url=URL) is False
    # Standalone routing (cron preflight) agrees: nothing is relay-fronted.
    assert relay.relay_fronted_platforms() == set()


def test_url_only_activation_is_unchanged(profile, monkeypatch):
    """Control: with no ``enabled`` key the injected URL still activates relay."""
    (profile / "config.yaml").write_text(yaml.safe_dump({"platforms": {"relay": {"extra": {"note": "kept"}}}}))
    provision = Mock(return_value={"gatewayId": "g", "secret": "b" * 64, "deliveryKey": "d"})
    monkeypatch.setattr(relay, "_resolve_relay_identity_token", Mock(return_value="identity-token"))
    monkeypatch.setattr(relay, "_post_provision", provision)
    monkeypatch.setattr(relay, "_post_policy", Mock(return_value=200))

    GatewayStartupMixin._start_register_plugins_relay_hooks()

    provision.assert_called_once()
    assert platform_registry.is_registered("relay")
    assert relay.relay_fronted_platforms() == {"slack"}


@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_predicate_agrees_with_loader_and_keeps_native_adapters(profile, monkeypatch, spelling):
    """The standalone predicate and ``load_gateway_config()`` are two readers of one rule, and an
    opted-out relay does not own the profile's native connections."""
    write_disable(profile, spelling, monkeypatch)
    config = load_gateway_config()

    assert relay.relay_explicitly_disabled() is True
    assert config.platforms[Platform.RELAY].enabled is False
    # Without the veto GATEWAY_RELAY_URL disables every directly-connected platform.
    assert config.platforms[Platform.SLACK].enabled is True

    from cron.scheduler_delivery import _resolve_target_transport

    resolved, error = _resolve_target_transport(
        {"id": "job"}, Platform.SLACK, "slack", {"chat_id": "channel"}, {}, config,
    )
    assert error is None
    assert resolved[1] is config.platforms[Platform.SLACK]


def test_legacy_json_disable_is_advisory_like_other_platforms(profile, monkeypatch):
    (profile / "gateway.json").write_text(json.dumps({"platforms": {
        "relay": {"enabled": False},
        "telegram": {"enabled": False},
    }}))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "native-telegram-test-token")
    monkeypatch.setenv("GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS", "true")
    config = load_gateway_config()

    # Only a YAML ``enabled`` key sets ``_enabled_explicit``; env presence (the URL, the token)
    # re-enables a gateway.json disable for relay exactly as it does for telegram.
    assert relay.relay_explicitly_disabled() is False
    assert config.platforms[Platform.RELAY].enabled is True
    assert config.platforms[Platform.TELEGRAM].enabled is True


def test_malformed_user_yaml_drops_the_yaml_layer_for_both_readers(profile, monkeypatch):
    """The loader falls back to env + gateway.json WITHOUT the managed layer on a malformed user
    file; the predicate must fall back the same way or startup suppresses native adapters for a
    relay it then refuses to register."""
    (profile / "config.yaml").write_text("platforms: [\n")
    write_disable(profile, "managed", monkeypatch)
    config = load_gateway_config()

    assert relay.relay_explicitly_disabled() is False
    assert config.platforms[Platform.RELAY].enabled is True
    assert relay.relay_url() == URL
