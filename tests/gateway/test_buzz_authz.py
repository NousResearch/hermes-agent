"""Authorization integration contracts for the Buzz plugin."""

import pytest

from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform
from gateway.session import SessionSource
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


SELF_PUBKEY = "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
SELF_NPUB = "npub1nl2u0wnd8mezfknc74q7pl9ec58h9nrrakce4tnk434qgaxl4psqe5twr6"
_buzz_mod = load_plugin_adapter("buzz")


class _AuthHarness(GatewayAuthorizationMixin):
    pairing_store = None
    pairing_stores = {}
    adapters = {}
    config = None


def test_buzz_npub_allowlist_authorizes_hex_sender(monkeypatch):
    from gateway.platform_registry import PlatformEntry, platform_registry

    previous_entry = platform_registry._entries.get("buzz")
    previous_deferred = platform_registry._deferred.get("buzz")
    platform_registry.unregister("buzz")
    platform_registry.register(PlatformEntry(
        name="buzz",
        label="Buzz",
        adapter_factory=lambda config: None,
        check_fn=lambda: True,
        allowed_users_env="BUZZ_ALLOWED_USERS",
        allow_all_env="BUZZ_ALLOW_ALL_USERS",
        auth_identity_normalizer=_buzz_mod._normalize_user_ref,
    ))
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", SELF_NPUB)
    monkeypatch.setenv("BUZZ_ALLOW_ALL_USERS", "false")

    try:
        source = SessionSource(
            platform=Platform("buzz"),
            chat_id="channel-1",
            chat_type="group",
            user_id=SELF_PUBKEY,
        )
        assert _AuthHarness()._is_user_authorized(source) is True
    finally:
        platform_registry.unregister("buzz")
        if previous_entry is not None:
            platform_registry.register(previous_entry)
        elif previous_deferred is not None:
            platform_registry.register_deferred("buzz", previous_deferred)


@pytest.mark.parametrize("scope", [{}, {"BUZZ_ALLOW_ALL_USERS": ""}])
def test_buzz_multiplex_scope_does_not_inherit_process_allow_all(monkeypatch, scope):
    from agent import secret_scope as ss
    from gateway.config import PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry

    previous_entry = platform_registry._entries.get("buzz")
    previous_deferred = platform_registry._deferred.get("buzz")
    platform_registry.unregister("buzz")
    platform_registry.register(PlatformEntry(
        name="buzz",
        label="Buzz",
        adapter_factory=lambda config: None,
        check_fn=lambda: True,
        allowed_users_env="BUZZ_ALLOWED_USERS",
        allow_all_env="BUZZ_ALLOW_ALL_USERS",
        auth_identity_normalizer=_buzz_mod._normalize_user_ref,
    ))
    monkeypatch.setenv("BUZZ_ALLOW_ALL_USERS", "true")
    monkeypatch.delenv("BUZZ_ALLOWED_USERS", raising=False)
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope(scope)

    try:
        adapter = _buzz_mod.BuzzAdapter(PlatformConfig(
            enabled=True,
            extra={"allow_all_users": False, "allowed_users": []},
        ))
        harness = _AuthHarness()
        harness.adapters = {Platform("buzz"): adapter}
        source = SessionSource(
            platform=Platform("buzz"),
            chat_id="channel-1",
            chat_type="group",
            user_id=SELF_PUBKEY,
        )
        assert harness._is_user_authorized(source) is False
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)
        platform_registry.unregister("buzz")
        if previous_entry is not None:
            platform_registry.register(previous_entry)
        elif previous_deferred is not None:
            platform_registry.register_deferred("buzz", previous_deferred)


@pytest.mark.parametrize(
    ("scope", "expected"),
    [({}, True), ({"BUZZ_ALLOWED_USERS": ""}, False)],
)
def test_buzz_multiplex_scope_uses_adapter_allowlist_not_process_writer(
    monkeypatch, scope, expected
):
    from agent import secret_scope as ss
    from gateway.config import PlatformConfig
    from gateway.platform_registry import PlatformEntry, platform_registry

    previous_entry = platform_registry._entries.get("buzz")
    previous_deferred = platform_registry._deferred.get("buzz")
    platform_registry.unregister("buzz")
    platform_registry.register(PlatformEntry(
        name="buzz",
        label="Buzz",
        adapter_factory=lambda config: None,
        check_fn=lambda: True,
        allowed_users_env="BUZZ_ALLOWED_USERS",
        allow_all_env="BUZZ_ALLOW_ALL_USERS",
        auth_identity_normalizer=_buzz_mod._normalize_user_ref,
    ))
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", "a" * 64)
    monkeypatch.setenv("BUZZ_ALLOW_ALL_USERS", "false")
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope(scope)

    try:
        adapter = _buzz_mod.BuzzAdapter(PlatformConfig(
            enabled=True,
            extra={"allowed_users": [SELF_NPUB], "allow_all_users": False},
        ))
        harness = _AuthHarness()
        harness.adapters = {Platform("buzz"): adapter}
        source = SessionSource(
            platform=Platform("buzz"),
            chat_id="channel-1",
            chat_type="group",
            user_id=SELF_PUBKEY,
        )
        assert harness._is_user_authorized(source) is expected
    finally:
        ss.reset_secret_scope(token)
        ss.set_multiplex_active(False)
        platform_registry.unregister("buzz")
        if previous_entry is not None:
            platform_registry.register(previous_entry)
        elif previous_deferred is not None:
            platform_registry.register_deferred("buzz", previous_deferred)
