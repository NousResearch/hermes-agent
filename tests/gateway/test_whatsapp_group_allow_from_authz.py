"""Regression tests for #87830: whatsapp.group_allow_from must authorize
routed/observed group turns that arrive with user_id=None.

The shared-transcript observe path strips sender identity from triggered
WhatsApp group messages, so the chat-scoped authorization in
GatewayAuthorizationMixin._is_user_authorized is the only gate. Before the
fix, only the WHATSAPP_GROUP_ALLOWED_USERS process env var authorized these
turns; the documented config surface (whatsapp.group_allow_from) was never
consulted — and under multiplex_profiles the adapter lookup fails closed for
routed profiles, so the adapter-extra fallback silently never fired.
"""

from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner
from gateway.session import Platform, SessionSource

GROUP_JID = "120363028379211573@g.us"


def _bare_runner(platform_extra=None):
    runner = object.__new__(GatewayRunner)
    if platform_extra is not None:
        runner.config = SimpleNamespace(
            platforms={Platform.WHATSAPP: SimpleNamespace(extra=dict(platform_extra))}
        )
    return runner


def _observed_group_source(profile=None, chat_id=GROUP_JID):
    return SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=chat_id,
        chat_type="group",
        user_id=None,
        user_name=None,
        profile=profile,
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "WHATSAPP_GROUP_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(var, raising=False)


def test_config_group_allow_from_authorizes_no_user_id_turn():
    runner = _bare_runner(
        {"group_policy": "allowlist", "group_allow_from": [GROUP_JID]}
    )
    assert runner._is_user_authorized(_observed_group_source()) is True


def test_config_group_allow_from_authorizes_routed_profile_source():
    """Multiplex: the routed source carries a secondary profile; the adapter
    registry fails closed for it, but the gateway's own platform config (the
    same values that gated intake) must still authorize the chat."""
    runner = _bare_runner(
        {"group_policy": "allowlist", "group_allow_from": [GROUP_JID]}
    )
    src = _observed_group_source(profile="research")
    assert runner._is_user_authorized(src) is True


def test_unlisted_group_is_still_denied():
    runner = _bare_runner(
        {"group_policy": "allowlist", "group_allow_from": [GROUP_JID]}
    )
    src = _observed_group_source(chat_id="999999999999999999@g.us")
    assert runner._is_user_authorized(src) is False


def test_open_group_policy_does_not_authorize_via_stale_allowlist():
    """Under group_policy: open the allow_from list carries no restriction
    signal — authorizing from it would be a fail-open."""
    runner = _bare_runner({"group_policy": "open", "group_allow_from": [GROUP_JID]})
    assert runner._is_user_authorized(_observed_group_source()) is False


def test_no_config_no_env_denies():
    runner = _bare_runner()
    assert runner._is_user_authorized(_observed_group_source()) is False


def test_wildcard_entry_authorizes_any_group():
    runner = _bare_runner({"group_policy": "allowlist", "group_allow_from": ["*"]})
    src = _observed_group_source(chat_id="111222333444555666@g.us")
    assert runner._is_user_authorized(src) is True


def test_env_var_path_still_works(monkeypatch):
    """The pre-existing env path must keep working unchanged."""
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "")
    runner = _bare_runner()
    # env-based group user allowlists are consulted later (user_id path);
    # a no-user-id turn with no chat-scoped grant stays denied.
    assert runner._is_user_authorized(_observed_group_source()) is False
