"""Tests for relay upstream-enforced authorization at the gateway layer.

Background: the relay adapter fronts the Team Gateway connector over a
per-instance-authenticated WebSocket. The connector performs owner-only
author-binding resolution BEFORE delivering an inbound event — a message only
reaches this gateway because the connector resolved it to THIS instance's bound
user (``user_instance_binding``, keyed on the connector-observed author id,
never a gateway claim). So a relay inbound is already authorized by a trusted,
authenticated upstream.

Before this fix, ``_is_user_authorized`` had no notion of upstream
authorization: ``Platform.RELAY`` matched no ``*_ALLOWED_USERS`` allowlist and
isn't in the HA/WEBHOOK always-authorized set, so every relay user hit the
default-deny ("No user allowlists configured. All unauthorized users will be
denied.") and the agent never saw the message. This was the live staging bug:
the message routed correctly through the connector to the instance, then the
instance's authz layer dropped it as ``Unauthorized user``.

The fix adds a generic ``BasePlatformAdapter.authorization_is_upstream``
capability (default ``False``) that the relay adapter overrides to ``True``,
plus a dedicated trusted branch in ``_is_user_authorized``. It is delegation to
a trusted upstream, NOT a fail-open: it fires only for an adapter that
explicitly declares the flag; every direct network-exposed adapter leaves it
``False`` and the env-allowlist default-deny is unchanged.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "DISCORD_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "DISCORD_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_runner(*, platform: Platform, authorization_is_upstream: bool):
    """Build a bare GatewayRunner with one adapter for *platform*.

    ``authorization_is_upstream`` controls whether that adapter declares the
    upstream-authz capability.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(
        send=AsyncMock(),
        authorization_is_upstream=authorization_is_upstream,
        enforces_own_access_policy=False,
    )
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    return runner, adapter


def _relay_source(**kw) -> SessionSource:
    base = dict(
        platform=Platform.RELAY,
        user_id="428014785045725184",
        chat_id="1400724139874058314",
        user_name="definitely_not_cthulhu",
        chat_type="group",
    )
    base.update(kw)
    return SessionSource(**base)


# ---------------------------------------------------------------------------
# Capability contract
# ---------------------------------------------------------------------------


def test_base_adapter_defaults_to_not_upstream_authorized():
    """The base property is False — direct adapters keep env default-deny."""
    from gateway.platforms.base import BasePlatformAdapter

    assert BasePlatformAdapter.authorization_is_upstream.fget(object()) is False


def test_relay_adapter_declares_upstream_authz():
    """The relay adapter overrides the capability to True (static capability)."""
    from gateway.relay.adapter import RelayAdapter

    # Property reflects a static capability, independent of instance config.
    assert RelayAdapter.authorization_is_upstream.fget(object()) is True


# ---------------------------------------------------------------------------
# Authorization behavior
# ---------------------------------------------------------------------------


def test_relay_user_authorized_with_no_env_allowlist(monkeypatch):
    """A relay user is authorized even with NO env allowlist configured.

    This is the staging-bug regression guard: the connector already authorized
    the author via owner-only binding, so the instance must not default-deny.
    """
    _clear_auth_env(monkeypatch)
    runner, _ = _make_runner(platform=Platform.RELAY, authorization_is_upstream=True)
    assert runner._is_user_authorized(_relay_source()) is True


def test_relay_dm_authorized_with_no_env_allowlist(monkeypatch):
    """The /link DM path is also authorized (DMs are upstream-bound too)."""
    _clear_auth_env(monkeypatch)
    runner, _ = _make_runner(platform=Platform.RELAY, authorization_is_upstream=True)
    assert runner._is_user_authorized(_relay_source(chat_type="dm")) is True


def test_non_upstream_adapter_still_default_denies(monkeypatch):
    """A direct adapter that does NOT declare the flag still default-denies.

    Guards against the fix becoming a blanket fail-open: an adapter with
    authorization_is_upstream=False and no env allowlist must remain denied.
    """
    _clear_auth_env(monkeypatch)
    runner, _ = _make_runner(platform=Platform.DISCORD, authorization_is_upstream=False)
    src = SessionSource(
        platform=Platform.DISCORD,
        user_id="123",
        chat_id="456",
        user_name="someone",
        chat_type="dm",
    )
    assert runner._is_user_authorized(src) is False


def test_upstream_authz_helper_false_for_unknown_platform(monkeypatch):
    """The helper returns False when there's no adapter for the platform."""
    _clear_auth_env(monkeypatch)
    runner, _ = _make_runner(platform=Platform.RELAY, authorization_is_upstream=True)
    # A platform with no registered adapter must not be treated as upstream-authz.
    assert runner._adapter_authorization_is_upstream(Platform.DISCORD) is False
    assert runner._adapter_authorization_is_upstream(None) is False


# ---------------------------------------------------------------------------
# The underlying-platform regression: a relay *message* inbound carries the
# UNDERLYING platform (source.platform == Platform.DISCORD), not Platform.RELAY,
# because the connector's wire payload sets platform="discord" and
# ws_transport._event_from_wire maps it straight onto SessionSource. The relay
# adapter is registered ONLY under Platform.RELAY, so keying upstream-authz off
# source.platform misses and the user hits default-deny ("Unauthorized user
# <id> (<name>) on discord"). The authentic trust signal is that the event was
# delivered over the per-instance-authenticated relay WS — carried by
# source.delivered_via_upstream_relay, set by the relay transport, NOT by
# source.platform.
# ---------------------------------------------------------------------------


def test_relay_message_with_underlying_discord_platform_authorized(monkeypatch):
    """The live message path: source.platform=DISCORD + relay-delivered marker.

    Reproduces the staging bug exactly — a relay-delivered Discord message whose
    source.platform is the underlying "discord" (not "relay"). The relay adapter
    is registered only under Platform.RELAY, so the OLD source.platform-keyed
    check missed and default-denied. Authorization must come from the
    relay-delivery marker instead.
    """
    _clear_auth_env(monkeypatch)
    runner, _ = _make_runner(platform=Platform.RELAY, authorization_is_upstream=True)
    src = SessionSource(
        platform=Platform.DISCORD,  # underlying platform off the wire
        user_id="267171776755269633",
        chat_id="1400724139874058314",
        user_name="rewbs",
        chat_type="dm",
        delivered_via_upstream_relay=True,
    )
    assert runner._is_user_authorized(src) is True


def test_direct_discord_event_not_authorized_by_relay_presence(monkeypatch):
    """A DIRECT Discord event must NOT be authorized just because a relay adapter
    is registered (multiplexing gateway: direct Discord adapter + relay adapter).

    Without the delivery marker, the relay's upstream-authz must not leak onto a
    direct Discord inbound — that would be a fail-open. Only events the relay
    transport actually delivered carry delivered_via_upstream_relay=True.
    """
    _clear_auth_env(monkeypatch)
    runner, _ = _make_runner(platform=Platform.RELAY, authorization_is_upstream=True)
    src = SessionSource(
        platform=Platform.DISCORD,
        user_id="999",
        chat_id="456",
        user_name="direct_discord_user",
        chat_type="dm",
        # delivered_via_upstream_relay defaults to False (direct delivery)
    )
    assert runner._is_user_authorized(src) is False


def test_relay_delivery_marker_is_wire_invisible():
    """delivered_via_upstream_relay is an INTERNAL trust signal, never serialized.

    It must not appear in to_dict() (the wire/persistence surface) — it is set
    locally by the relay transport from the authenticated socket, never trusted
    off the wire.
    """
    src = SessionSource(
        platform=Platform.DISCORD,
        chat_id="1",
        user_id="2",
        delivered_via_upstream_relay=True,
    )
    assert "delivered_via_upstream_relay" not in src.to_dict()
    # And it does not survive a wire round-trip (a peer can't forge it).
    assert SessionSource.from_dict(src.to_dict()).delivered_via_upstream_relay is False


def test_event_from_wire_sets_relay_delivery_marker():
    """The relay transport stamps the marker on every event it rebuilds.

    This is the authentic injection point: _event_from_wire only runs for frames
    that arrived over the per-instance-authenticated relay WS.
    """
    from gateway.relay.ws_transport import _event_from_wire

    event = _event_from_wire(
        {
            "text": "hello!",
            "source": {
                "platform": "discord",
                "chat_id": "123",
                "chat_type": "dm",
                "user_id": "267171776755269633",
                "user_name": "rewbs",
            },
        }
    )
    assert event.source.platform is Platform.DISCORD
    assert event.source.delivered_via_upstream_relay is True


def test_event_from_wire_stamps_routed_profile():
    """A connector-routed profile on the wire source lands on SessionSource.

    In multiplex mode the connector resolves the target HERMES profile for a
    Team-Gateway message and stamps ``profile`` on the wire source. The relay
    transport must carry it through so build_session_key namespaces the session
    and the agent turn resolves that profile's config/credentials.
    """
    from gateway.relay.ws_transport import _event_from_wire

    event = _event_from_wire(
        {
            "text": "hello!",
            "source": {
                "platform": "discord",
                "chat_id": "123",
                "chat_type": "dm",
                "user_id": "267171776755269633",
                "user_name": "rewbs",
                "profile": "reviewer",
            },
        }
    )
    assert event.source.profile == "reviewer"


def test_event_from_wire_profile_absent_is_none():
    """No ``profile`` on the wire (single-profile gateway) → None.

    Back-compat: a connector that never sets ``profile`` yields the legacy
    behaviour, and session keys stay in the ``agent:main`` namespace.
    """
    from gateway.relay.ws_transport import _event_from_wire

    event = _event_from_wire(
        {
            "text": "hi",
            "source": {
                "platform": "discord",
                "chat_id": "123",
                "chat_type": "dm",
                "user_id": "1",
            },
        }
    )
    assert event.source.profile is None
