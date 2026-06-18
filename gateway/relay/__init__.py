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


def relay_endpoint() -> Optional[str]:
    """The gateway's own PUBLIC inbound URL, asserted to the connector at provision.

    The connector delivers signed inbound POSTs to this URL and stores it on the
    tenant's route rows. It is gateway-asserted (the connector scopes it to the
    verified tenant, so a dishonest gateway can only misdirect its OWN inbound).
    The *source* of the value differs by deployment but the code path is uniform:
    a self-hosted operator sets ``GATEWAY_RELAY_ENDPOINT`` (mirrors how they set
    ``HERMES_DASHBOARD_PUBLIC_URL``); a hosted/NAS container has the same var
    stamped in (NAS knows the public URL only in that case). Absent -> the
    gateway provisions outbound-only (no inbound routes written).

    Env first (Docker), then ``gateway.relay_endpoint`` in config.yaml.
    """
    url = os.environ.get("GATEWAY_RELAY_ENDPOINT", "").strip()
    if not url:
        try:
            from gateway.run import _load_gateway_config  # late import to avoid cycle

            cfg = (_load_gateway_config().get("gateway") or {})
            url = str(cfg.get("relay_endpoint", "") or "").strip()
        except Exception:  # noqa: BLE001 - config absence/parse must never crash boot
            url = ""
    return url.rstrip("/") or None


def relay_route_keys() -> list[str]:
    """Discriminators (guild_ids / chat_ids / paths) this gateway's tenant owns.

    Gateway-provided config, paired with ``relay_endpoint()``: the connector
    writes one route row per (routeKey -> tenant, endpoint), so route keys only
    take effect alongside an endpoint. Empty -> outbound-only provisioning (the
    connector accepts an empty set and writes no route rows).

    ``GATEWAY_RELAY_ROUTE_KEYS`` is comma-separated; config.yaml
    ``gateway.relay_route_keys`` may be a list or a comma string.
    """
    raw = os.environ.get("GATEWAY_RELAY_ROUTE_KEYS", "").strip()
    if not raw:
        try:
            from gateway.run import _load_gateway_config  # late import to avoid cycle

            cfg = (_load_gateway_config().get("gateway") or {})
            val = cfg.get("relay_route_keys", "")
            if isinstance(val, (list, tuple)):
                return [str(k).strip() for k in val if str(k).strip()]
            raw = str(val or "").strip()
        except Exception:  # noqa: BLE001
            raw = ""
    return [k.strip() for k in raw.split(",") if k.strip()]


def _provision_url(relay_dial_url: str) -> str:
    """Map the ``ws(s)://…/relay`` dial URL to the ``http(s)://…/relay/provision`` POST URL."""
    raw = relay_dial_url.rstrip("/")
    if raw.startswith("ws://"):
        raw = "http://" + raw[len("ws://"):]
    elif raw.startswith("wss://"):
        raw = "https://" + raw[len("wss://"):]
    if raw.endswith("/relay"):
        raw = raw[: -len("/relay")]
    return f"{raw}/relay/provision"


def _post_provision(
    *,
    provision_url: str,
    access_token: str,
    gateway_id: str,
    platform: str,
    bot_id: str,
    gateway_endpoint: Optional[str],
    route_keys: list[str],
    timeout: float = 15.0,
) -> dict:
    """POST to the connector's ``/relay/provision`` and return the JSON body.

    The connector validates ``access_token`` against NAS, derives the
    authoritative tenant, mints the per-gateway secret + per-tenant delivery key,
    upserts the tenant's route rows, and returns
    ``{secret, deliveryKey, tenant, gatewayId, routeKeys}``. Raises RuntimeError
    with a user-facing message on any non-2xx / transport failure.
    """
    import json
    import urllib.error
    import urllib.request

    body: dict = {
        "gatewayId": gateway_id,
        "platform": platform,
        "botId": bot_id,
        "gatewayEndpoint": gateway_endpoint or "",
        "routeKeys": route_keys,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        provision_url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = (json.loads(exc.read().decode()) or {}).get("error", "")
        except Exception:
            pass
        raise RuntimeError(
            f"connector returned HTTP {exc.code}" + (f": {detail}" if detail else "")
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach connector: {exc.reason}") from exc

    if not isinstance(payload, dict) or not payload.get("secret"):
        raise RuntimeError("connector returned an unexpected response (no secret)")
    return payload


def self_provision_if_managed() -> bool:
    """Managed-boot self-provision: mint relay creds in-process, no human, no disk.

    Fires only on a MANAGED boot (``is_managed()``) with relay configured
    (``relay_url()`` set) and NO per-gateway secret already present. In that case
    the runtime resolves the agent's own Nous access token (the same
    ``resolve_nous_access_token()`` the enroll CLI / dashboard register use),
    POSTs ``/relay/provision`` asserting its own endpoint + route keys, and sets
    ``GATEWAY_RELAY_ID`` / ``GATEWAY_RELAY_SECRET`` / ``GATEWAY_RELAY_DELIVERY_KEY``
    into ``os.environ`` so the subsequent ``register_relay_adapter()`` picks them
    up. The creds live ONLY in process memory — never written to ``~/.hermes/.env``
    (``save_env_value`` refuses under managed anyway, and keeping the secret off
    any volume is the stronger posture).

    Stateless: process-env creds don't survive a restart, so a managed container
    re-provisions every boot; the connector's rotation window covers a still-
    connected prior instance. An explicitly-pinned ``GATEWAY_RELAY_SECRET`` (env
    or config) is RESPECTED — self-provision skips so an operator pin isn't
    stomped.

    Returns True if it provisioned, False otherwise. NEVER raises: a provision
    failure logs and returns False so the gateway still boots (and
    ``register_relay_adapter`` will simply dial unauthenticated / be rejected,
    rather than the whole gateway crashing).
    """
    import logging

    logger = logging.getLogger("gateway.relay")

    try:
        from hermes_cli.config import is_managed
    except Exception:  # noqa: BLE001
        return False

    if not is_managed():
        return False
    dial_url = relay_url()
    if not dial_url:
        return False

    # Respect an already-present (pinned/stamped) secret — don't stomp it.
    existing_id, existing_secret = relay_connection_auth()
    if existing_id and existing_secret:
        logger.info("relay self-provision skipped: GATEWAY_RELAY_SECRET already set")
        return False

    try:
        from hermes_cli.auth import resolve_nous_access_token

        access_token = resolve_nous_access_token()
    except Exception as exc:  # noqa: BLE001 - boot must survive a token failure
        logger.warning("relay self-provision skipped: could not resolve Nous token (%s)", exc)
        return False

    platform, bot_id = relay_platform_identity()
    # gatewayId default mirrors the enroll CLI's hostname-based slug.
    import socket

    try:
        host = socket.gethostname().strip()
    except Exception:  # noqa: BLE001
        host = ""
    gateway_id = os.environ.get("GATEWAY_RELAY_ID", "").strip() or f"gw-{host or 'hermes'}"
    endpoint = relay_endpoint()
    route_keys = relay_route_keys()

    try:
        result = _post_provision(
            provision_url=_provision_url(dial_url),
            access_token=access_token,
            gateway_id=gateway_id,
            platform=platform,
            bot_id=bot_id,
            gateway_endpoint=endpoint,
            route_keys=route_keys,
        )
    except RuntimeError as exc:
        logger.warning("relay self-provision failed (%s); gateway will boot without relay auth", exc)
        return False

    # Set creds in-process so register_relay_adapter() + relay_inbound_config()
    # read them from os.environ. Never logged.
    os.environ["GATEWAY_RELAY_ID"] = str(result.get("gatewayId") or gateway_id)
    os.environ["GATEWAY_RELAY_SECRET"] = str(result.get("secret") or "")
    os.environ["GATEWAY_RELAY_DELIVERY_KEY"] = str(result.get("deliveryKey") or "")
    tenant = str(result.get("tenant") or "")
    logger.info(
        "relay self-provisioned (gateway_id=%s tenant=%s routes=%d inbound=%s)",
        os.environ["GATEWAY_RELAY_ID"],
        tenant or "?",
        len(route_keys),
        "yes" if endpoint else "outbound-only",
    )
    return True


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
