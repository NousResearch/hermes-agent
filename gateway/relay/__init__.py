"""Relay/connector support package for the Hermes gateway.

EXPERIMENTAL. This package implements the gateway side of the "Gateway Gateway"
relay design: a generic ``RelayAdapter`` plus the wire-serializable
``CapabilityDescriptor`` the connector hands it at handshake time, and the
production ``WebSocketRelayTransport`` that dials the connector. The public API
(module names, descriptor field set, transport protocol) MAY CHANGE without a
deprecation cycle until at least two real Class-1 platforms (Discord + Telegram)
have shaken out the schema.

See ``docs/relay-connector-contract.md`` for the formal cross-repo interface.

Activation is driven by configuration, not a separate feature flag: the relay
platform is registered when a connector relay URL is configured
(``GATEWAY_RELAY_URL`` env or ``gateway.relay_url`` in config.yaml). Deployments
that don't set it are unaffected — exactly the same shape as ``gateway.proxy_url``.
"""

from __future__ import annotations

import os
from typing import Optional


def relay_url() -> Optional[str]:
    """The connector relay endpoint URL, or None when relay is not configured.

    Checks ``GATEWAY_RELAY_URL`` (convenient for Docker) first, then
    ``gateway.relay_url`` in config.yaml. A non-empty value activates the relay
    platform; absence means a normal direct/single-tenant gateway.
    """
    url = os.environ.get("GATEWAY_RELAY_URL", "").strip()
    if url:
        return url.rstrip("/")
    try:
        from gateway.run import _load_gateway_config  # late import to avoid cycle

        cfg = _load_gateway_config()
        url = (cfg.get("gateway") or {}).get("relay_url", "").strip()
        if url:
            return url.rstrip("/")
    except Exception:  # noqa: BLE001 - config absence/parse must never crash registration
        pass
    return None


def relay_platform_identity() -> tuple[str, str]:
    """Platform + bot id this gateway fronts over the relay (for the handshake hello).

    Defaults to ``("relay", "")``; overridable via ``GATEWAY_RELAY_PLATFORM`` /
    ``GATEWAY_RELAY_BOT_ID`` so one connector can front several platforms.
    """
    platform = os.environ.get("GATEWAY_RELAY_PLATFORM", "relay").strip() or "relay"
    bot_id = os.environ.get("GATEWAY_RELAY_BOT_ID", "").strip()
    return platform, bot_id


def relay_connection_auth() -> tuple[Optional[str], Optional[str]]:
    """The (gateway_id, upgrade_secret) this gateway authenticates the WS upgrade with.

    Both come from enrollment (``hermes gateway enroll`` writes them to
    ``~/.hermes/.env``): ``GATEWAY_RELAY_ID`` identifies the enrolled instance,
    ``GATEWAY_RELAY_SECRET`` is the per-gateway signing secret. Either absent ->
    ``(None, None)`` and the transport dials unauthenticated (dev/test, or a
    connector that doesn't enforce auth). Checks env first (Docker), then
    ``gateway.relay_id`` / ``gateway.relay_secret`` in config.yaml.
    """
    gateway_id = os.environ.get("GATEWAY_RELAY_ID", "").strip()
    secret = os.environ.get("GATEWAY_RELAY_SECRET", "").strip()
    if not (gateway_id and secret):
        try:
            from gateway.run import _load_gateway_config  # late import to avoid cycle

            cfg = (_load_gateway_config().get("gateway") or {})
            gateway_id = gateway_id or str(cfg.get("relay_id", "") or "").strip()
            secret = secret or str(cfg.get("relay_secret", "") or "").strip()
        except Exception:  # noqa: BLE001 - config absence/parse must never crash registration
            pass
    return (gateway_id or None, secret or None)


def relay_inbound_config() -> tuple[Optional[str], Optional[str], int]:
    """Resolve (delivery_key, bind_host, bind_port) for the inbound receiver.

    The connector delivers normalized inbound events to this gateway over a
    SIGNED HTTP POST (not the outbound WS), verified with the per-tenant delivery
    key issued at enrollment (``GATEWAY_RELAY_DELIVERY_KEY``). The receiver only
    starts when a delivery key AND a bind port are configured — a gateway with no
    public inbound URL (e.g. a purely outbound dev run) simply doesn't run it.

    Env first (Docker), then ``gateway.relay_delivery_key`` /
    ``gateway.relay_inbound_host`` / ``gateway.relay_inbound_port`` in config.yaml.
    Port 0 (default/unset) -> receiver disabled.
    """
    key = os.environ.get("GATEWAY_RELAY_DELIVERY_KEY", "").strip()
    host = os.environ.get("GATEWAY_RELAY_INBOUND_HOST", "").strip()
    port_raw = os.environ.get("GATEWAY_RELAY_INBOUND_PORT", "").strip()
    if not (key and port_raw):
        try:
            from gateway.run import _load_gateway_config  # late import to avoid cycle

            cfg = (_load_gateway_config().get("gateway") or {})
            key = key or str(cfg.get("relay_delivery_key", "") or "").strip()
            host = host or str(cfg.get("relay_inbound_host", "") or "").strip()
            if not port_raw:
                port_raw = str(cfg.get("relay_inbound_port", "") or "").strip()
        except Exception:  # noqa: BLE001 - config absence/parse must never crash registration
            pass
    try:
        port = int(port_raw) if port_raw else 0
    except ValueError:
        port = 0
    return (key or None, host or "0.0.0.0", port)


def register_relay_adapter(force: bool = False, url: Optional[str] = None) -> bool:
    """Register the generic ``relay`` platform via the platform registry.

    Registers when a relay URL is configured (or ``force=True`` for tests, which
    builds a transport-less adapter — the unit-test posture). Returns True if
    registration happened. Additive: uses the same registry path as plugin
    adapters, so no core dispatch changes are needed.

    When a URL is present the factory builds a live ``WebSocketRelayTransport``;
    the ``RelayAdapter`` negotiates the real ``CapabilityDescriptor`` at
    ``connect()`` time via ``transport.handshake()``.
    """
    resolved_url = url if url is not None else relay_url()
    if not (force or resolved_url):
        return False

    from gateway.platform_registry import PlatformEntry, platform_registry
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

    platform, bot_id = relay_platform_identity()

    def _factory(config):
        # Placeholder descriptor; replaced by the negotiated one at connect time
        # when a transport is present. With no URL (force/test) the adapter is
        # transport-less and keeps the placeholder.
        placeholder = CapabilityDescriptor(
            contract_version=CONTRACT_VERSION,
            platform=platform,
            label="Relay",
            max_message_length=4096,
            supports_draft_streaming=False,
            supports_edit=True,
            supports_threads=False,
            markdown_dialect="plain",
            len_unit="chars",
        )
        transport = None
        if resolved_url:
            from gateway.relay.ws_transport import WebSocketRelayTransport

            gateway_id, upgrade_secret = relay_connection_auth()
            transport = WebSocketRelayTransport(
                resolved_url,
                platform,
                bot_id,
                gateway_id=gateway_id,
                upgrade_secret=upgrade_secret,
            )
        return RelayAdapter(config, placeholder, transport=transport)

    platform_registry.register(
        PlatformEntry(
            name="relay",
            label="Relay",
            adapter_factory=_factory,
            check_fn=lambda: True,
            source="builtin",
            emoji="\U0001f50c",
        )
    )
    return True
