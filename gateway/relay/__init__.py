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
        url = (cfg.get("gateway") or {}).get("relay_url")
        url = (url or "").strip()
        if url:
            return url.rstrip("/")
    except Exception:  # noqa: BLE001 - config absence/parse must never crash registration
        pass
    return None


def relay_platform_identities() -> list[tuple[str, str]]:
    """The (platform, bot_id) pairs this gateway fronts over the relay (Phase 1.5).

    Shape A (multi-platform-per-agent, D-Q1.5c — CUT OVER, no scalar fallback):
    one gateway fronts a SET of platforms on one WS connection. The set is the
    env-stamped deploy config:

      - ``GATEWAY_RELAY_PLATFORMS`` — comma-sep list (e.g. ``discord,telegram``).
      - ``GATEWAY_RELAY_BOT_IDS`` — JSON keyed map
        ``{"discord": {"botId": "..."}, "telegram": {"botId": "...", "username": "..."}}``.

    Returns the ordered list of ``(platform, bot_id)`` pairs (the FIRST is the
    default the handshake/descriptor falls back to). The connector accepts N
    hellos accumulating into its advertised set; outbound frames discriminate
    per-frame on the platform (gateway-gateway D-Q1.5b.1). A platform present in
    the list but absent from the ids map resolves with an empty bot_id (the
    connector rejects an unprovisioned platform with a structured failure).

    Defaults to ``[("relay", "")]`` when nothing is configured (the generic
    single-plane fallback for a connector that didn't stamp a platform set).
    """
    platforms_raw = os.environ.get("GATEWAY_RELAY_PLATFORMS", "").strip()
    platforms = [p.strip() for p in platforms_raw.split(",") if p.strip()]
    if not platforms:
        return [("relay", "")]
    ids = _relay_bot_ids_map()
    out: list[tuple[str, str]] = []
    for platform in platforms:
        entry = ids.get(platform) or {}
        bot_id = str(entry.get("botId", "")).strip() if isinstance(entry, dict) else ""
        out.append((platform, bot_id))
    return out


def _relay_bot_ids_map() -> dict:
    """Parse ``GATEWAY_RELAY_BOT_IDS`` (JSON keyed map). Never raises — a malformed
    map yields ``{}`` so a bad config degrades to empty bot ids (the connector
    rejects an unprovisioned platform) rather than crashing boot."""
    import json
    import logging

    raw = os.environ.get("GATEWAY_RELAY_BOT_IDS", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:  # noqa: BLE001 - a bad map must not crash boot
        logging.getLogger("gateway.relay").warning(
            "GATEWAY_RELAY_BOT_IDS is not valid JSON; treating as empty"
        )
        return {}


def relay_bot_username(platform: str) -> Optional[str]:
    """The bot's deep-link username/handle for a platform (e.g. Telegram's
    ``@handle`` for ``t.me/<handle>``), read from the per-platform entry in
    ``GATEWAY_RELAY_BOT_IDS``. None when absent (most platforms don't need one).
    """
    entry = _relay_bot_ids_map().get(platform)
    if isinstance(entry, dict):
        username = entry.get("username")
        if username:
            return str(username).lstrip("@")
    return None


def relay_platform_identity() -> tuple[str, str]:
    """The PRIMARY (platform, bot_id) — the first identity in the configured set.

    Kept for call sites that need a single representative identity (the default
    descriptor platform, the policy projection's primary). The full set is
    ``relay_platform_identities()``. Defaults to ``("relay", "")``.
    """
    return relay_platform_identities()[0]


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
    """Discriminators (scope_ids / chat_ids / paths) this gateway's tenant owns.

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


def relay_instance_id() -> Optional[str]:
    """Stable per-instance id this gateway forwards at provision (Phase 6 Unit α).

    Binds the connector's ``gatewayId -> instanceId`` so the connector can route
    inbound per-instance (not tenant-broadcast) once Phase 6 delivery lands. The
    value is the NAS ``AgentInstance.id`` for a managed agent (NAS stamps
    ``GATEWAY_RELAY_INSTANCE_ID`` into the container env, beside
    ``GATEWAY_RELAY_URL``); a self-hosted operator may set it explicitly. It is
    gateway-asserted but safely scoped: the org/tenant stays token-verified, so a
    dishonest gateway can only bind ITS OWN tenant's instance — the same posture
    as ``relay_endpoint()``. Absent -> the connector stores null and per-instance
    routing simply has no binding for this connection yet (back-compat).

    Env first (Docker/NAS), then ``gateway.relay_instance_id`` in config.yaml.
    """
    value = os.environ.get("GATEWAY_RELAY_INSTANCE_ID", "").strip()
    if not value:
        try:
            from gateway.run import _load_gateway_config  # late import to avoid cycle

            cfg = (_load_gateway_config().get("gateway") or {})
            value = str(cfg.get("relay_instance_id", "") or "").strip()
        except Exception:  # noqa: BLE001 - config absence/parse must never crash boot
            value = ""
    return value or None


def relay_wake_url() -> Optional[str]:
    """The gateway's WAKE URL, forwarded at provision (Phase 5 §5.2 wake PRIMITIVE).

    A poke target the connector issues a payload-free GET to when a buffered-only
    (going-idle) destination for this instance receives its first buffered event,
    so a suspended gateway wakes, reconnects its relay WS, and drains its
    delivery-leg backlog. The value's *source* differs by deployment but the code
    path is uniform: a managed/NAS container has ``GATEWAY_RELAY_WAKE_URL`` stamped
    in (NAS knows the Fly autostart / dashboard hostname); a self-hosted operator
    sets it explicitly (or passes ``--wake-url`` to ``hermes gateway enroll``).

    Gateway-asserted but safely scoped: the org/tenant stays token-verified, so a
    dishonest gateway can only register a wake target for ITS OWN instance — the
    same posture as ``relay_instance_id()`` / the retired ``relay_endpoint()``.
    Absent -> the connector stores null and simply can't wake this instance
    (buffering still works; the gateway drains whenever it next reconnects).

    Env first (Docker/NAS), then ``gateway.relay_wake_url`` in config.yaml.
    """
    value = os.environ.get("GATEWAY_RELAY_WAKE_URL", "").strip()
    if not value:
        try:
            from gateway.run import _load_gateway_config  # late import to avoid cycle

            cfg = (_load_gateway_config().get("gateway") or {})
            value = str(cfg.get("relay_wake_url", "") or "").strip()
        except Exception:  # noqa: BLE001 - config absence/parse must never crash boot
            value = ""
    return value.rstrip("/") or None


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


def _policy_url(relay_dial_url: str) -> str:
    """Map the ``ws(s)://…/relay`` dial URL to the ``http(s)://…/relay/policy`` POST URL.

    Same host derivation as ``_provision_url``; the connector mounts the
    relevance-policy update channel at ``/relay/policy`` (Phase 6 Unit ζ).
    """
    raw = relay_dial_url.rstrip("/")
    if raw.startswith("ws://"):
        raw = "http://" + raw[len("ws://"):]
    elif raw.startswith("wss://"):
        raw = "https://" + raw[len("wss://"):]
    if raw.endswith("/relay"):
        raw = raw[: -len("/relay")]
    return f"{raw}/relay/policy"


def relay_relevance_policy(platform: Optional[str] = None) -> Optional[dict]:
    """Project a fronted platform's RELEVANCE config into the connector's generic vocabulary.

    The connector's relevance gate (Phase 6 Unit ζ) reasons over a
    platform-agnostic policy — ``requireAddress`` / ``freeResponseScopes`` /
    ``allowOtherBots`` — NOT over Discord/Telegram words. This is the gateway
    side of that contract: it reads the agent's existing relevance knobs and
    emits the generic shape the connector stores per-instance (Phase 1.5: the
    connector keys the policy by ``(tenant, platform, instanceId)``, so each
    fronted platform gets its own row — pass its name here).

    Mapping (the connector vocabulary ← the gateway's existing config):
      - ``requireAddress``     ← the platform's ``require_mention`` (the agent
        only engages a non-owner message that @mentions it / replies to it).
      - ``freeResponseScopes`` ← the platform's ``free_response_channels`` (the
        channel/scope ids where ``require_mention`` is waived — same scope
        vocabulary the connector's δ scope grants + ε floor use).
      - ``allowOtherBots``     ← ``{PLATFORM}_ALLOW_BOTS`` in {"mentions","all"}
        for platforms whose full bot policy fits this broad flag. Discord is
        always projected as ``False`` because its exact-ID allowlist and
        mentions/all distinction cannot be represented by the relay contract.

    Read from the relay platform's config block (the platform the connector
    fronts, e.g. ``discord:``), falling back to the bridged top-level keys, then
    the ``{PLATFORM}_*`` env. ``platform`` defaults to the PRIMARY fronted
    platform (back-compat). Returns the generic dict, or None when relay isn't
    configured or the platform exposes no relevance knobs (⇒ the connector's
    quiet default already matches, so there's nothing to declare).
    """
    if platform is None:
        platform, _bot_id = relay_platform_identity()
    if not platform or platform == "relay":
        # No concrete fronted platform resolved ⇒ nothing platform-specific to project.
        return None

    # Resolve the platform's config block + the bridged top-level keys.
    require_mention = None
    free_response: list[str] = []
    try:
        from gateway.run import _load_gateway_config  # late import to avoid cycle

        cfg = _load_gateway_config() or {}
        plat_cfg = cfg.get(platform)
        if not isinstance(plat_cfg, dict):
            plat_cfg = ((cfg.get("gateway") or {}).get("platforms") or {}).get(platform)
        if not isinstance(plat_cfg, dict):
            plat_cfg = (cfg.get("platforms") or {}).get(platform)
        plat_cfg = plat_cfg if isinstance(plat_cfg, dict) else {}

        if "require_mention" in plat_cfg:
            require_mention = plat_cfg.get("require_mention")
        elif cfg.get("require_mention") is not None:
            require_mention = cfg.get("require_mention")

        frc = plat_cfg.get("free_response_channels")
        if frc is None:
            frc = cfg.get("free_response_channels")
        if isinstance(frc, (list, tuple)):
            free_response = [str(c).strip() for c in frc if str(c).strip()]
        elif isinstance(frc, str) and frc.strip():
            free_response = [c.strip() for c in frc.split(",") if c.strip()]
    except Exception:  # noqa: BLE001 - config absence/parse must never crash boot
        pass

    # Discord bot relay is deliberately fail-closed. The connector's generic
    # policy can carry only a broad boolean, not DISCORD_ALLOWED_BOT_IDS or the
    # mentions/all distinction. Direct Discord intake still supports exact bot
    # IDs; relayed Discord bot input remains off until the wire contract can
    # preserve both controls end to end.
    allow_bots_env = os.environ.get(f"{platform.upper()}_ALLOW_BOTS", "").lower().strip()
    allow_other_bots = platform != "discord" and allow_bots_env in {"mentions", "all"}

    require_address = bool(require_mention) if require_mention is not None else False

    # Non-Discord platforms can omit an all-default row. Discord must actively
    # send allowOtherBots=false so an older permissive connector policy cannot
    # survive an upgrade as stale state.
    if platform != "discord" and not require_address and not free_response and not allow_other_bots:
        return None

    return {
        "platform": platform,
        "requireAddress": require_address,
        "freeResponseScopes": free_response,
        "allowOtherBots": allow_other_bots,
    }


def _post_provision(
    *,
    provision_url: str,
    access_token: str,
    gateway_id: str,
    platform: str,
    bot_id: str,
    gateway_endpoint: Optional[str],
    route_keys: list[str],
    instance_id: Optional[str] = None,
    wake_url: Optional[str] = None,
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
    # Only send instanceId when we actually have one — omitting it lets the
    # connector store null (back-compat) rather than binding an empty string.
    if instance_id:
        body["instanceId"] = instance_id
    # Same for the wake URL (Phase 5 §5.2): omit when absent so the connector
    # stores null and simply can't wake this instance (buffering still works).
    if wake_url:
        body["wakeUrl"] = wake_url
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


def _resolve_relay_identity_token() -> str:
    """Resolve the caller-identity bearer token the connector introspects to a tenant.

    Canonical resolver shared by the runtime self-provision path and the
    ``hermes gateway enroll`` CLI. Two modes, in precedence order:

      1. **Generic OIDC client-credentials** (air-gapped / self-hosted-IdP, NO
         Nous Portal): when ``gateway.idp.token_url`` (or
         ``GATEWAY_RELAY_IDP_TOKEN_URL``) is configured, obtain a workload access
         token via the OAuth2 ``client_credentials`` grant against the operator's
         own IdP (Entra; Authentik in the sandbox). The connector's Seam-A OIDC
         verifier reads a claim (default ``tid``) off it as the tenant.
      2. **Nous Portal** (default): ``resolve_nous_access_token()`` — existing
         managed/hosted behaviour.

    Raises on failure; callers decide whether that's fatal (enroll CLI) or a
    graceful boot no-op (self-provision).
    """
    token_url = os.environ.get("GATEWAY_RELAY_IDP_TOKEN_URL", "").strip()
    client_id = os.environ.get("GATEWAY_RELAY_IDP_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GATEWAY_RELAY_IDP_CLIENT_SECRET", "").strip()
    scope = os.environ.get("GATEWAY_RELAY_IDP_SCOPE", "").strip()
    if not token_url:
        try:
            from gateway.run import _load_gateway_config  # late import to avoid cycle

            idp = ((_load_gateway_config().get("gateway") or {}).get("idp") or {})
            token_url = str(idp.get("token_url", "") or "").strip()
            client_id = client_id or str(idp.get("client_id", "") or "").strip()
            client_secret = client_secret or str(idp.get("client_secret", "") or "").strip()
            scope = scope or str(idp.get("scope", "") or "").strip()
        except Exception:  # noqa: BLE001 - config absence must not crash
            token_url = token_url or ""

    if not token_url:
        # Mode 2 — Nous Portal (default, unchanged behaviour).
        from hermes_cli.auth import resolve_nous_access_token

        return resolve_nous_access_token()

    # Mode 1 — generic OAuth2 client_credentials grant.
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    if not client_id or not client_secret:
        raise RuntimeError(
            "gateway.idp.token_url configured but client_id/client_secret missing"
        )
    form = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    if scope:
        form["scope"] = scope
    req = urllib.request.Request(
        token_url,
        data=urllib.parse.urlencode(form).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15.0) as resp:
        payload = json.loads(resp.read().decode())
    access_token = (payload or {}).get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise RuntimeError("IdP client_credentials response had no access_token")
    return access_token.strip()


def self_provision_relay() -> bool:
    """Boot-time relay self-provision: mint relay creds in-process, no human, no disk.

    Fires when relay is configured (``relay_url()`` set) and NO per-gateway secret
    is already present, AND the agent can resolve its own Nous access token. In
    that case the runtime resolves the agent's own Nous access token (the same
    ``resolve_nous_access_token()`` the enroll CLI / dashboard register use),
    POSTs ``/relay/provision`` asserting its own endpoint + route keys, and sets
    ``GATEWAY_RELAY_ID`` / ``GATEWAY_RELAY_SECRET`` / ``GATEWAY_RELAY_DELIVERY_KEY``
    into ``os.environ`` so the subsequent ``register_relay_adapter()`` picks them
    up. The creds live ONLY in process memory — never written to ``~/.hermes/.env``.

    The trigger is deliberately NOT ``is_managed()``: that means
    "package-manager/NixOS-managed" and is False on a NAS-hosted Fly agent (which
    sets neither ``HERMES_MANAGED`` nor a ``.managed`` marker), so gating on it
    blocked the exact hosted case this is for. The real signal is "you pointed me
    at a connector and didn't pin a secret" — which is both NAS-independent and
    self-guarding:

      - A NAS-hosted agent: has ``GATEWAY_RELAY_URL``, no pinned secret, and a
        bootstrapped NAS token -> self-provisions.
      - A self-hosted operator who ran ``hermes gateway enroll``: has a PINNED
        ``GATEWAY_RELAY_SECRET`` -> skipped (the secret-present guard below).
      - A self-hosted box with a relay URL but no NAS identity:
        ``resolve_nous_access_token()`` fails -> graceful no-op.

    Stateless: process-env creds don't survive a restart, so a hosted container
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

    dial_url = relay_url()
    if not dial_url:
        return False

    # Respect an already-present (pinned/stamped) secret — don't stomp it. This
    # is also what makes a self-hosted, enrolled gateway skip self-provision.
    existing_id, existing_secret = relay_connection_auth()
    if existing_id and existing_secret:
        logger.info("relay self-provision skipped: GATEWAY_RELAY_SECRET already set")
        return False

    try:
        access_token = _resolve_relay_identity_token()
    except Exception as exc:  # noqa: BLE001 - boot must survive a token failure
        # No resolvable identity (e.g. a self-hosted box that hasn't enrolled and
        # configured no IdP) -> nothing to provision with; skip quietly and boot.
        logger.warning("relay self-provision skipped: could not resolve identity token (%s)", exc)
        return False

    identities = relay_platform_identities()
    # gatewayId default mirrors the enroll CLI's hostname-based slug.
    import socket

    try:
        host = socket.gethostname().strip()
    except Exception:  # noqa: BLE001
        host = ""
    gateway_id = os.environ.get("GATEWAY_RELAY_ID", "").strip() or f"gw-{host or 'hermes'}"
    endpoint = relay_endpoint()
    route_keys = relay_route_keys()
    instance_id = relay_instance_id()
    wake_url = relay_wake_url()

    # Phase 1.5 (D-Q1.5c): provision EACH fronted platform under the SAME
    # gatewayId + the SAME (platform-less) per-gateway secret. The connector's
    # secret record is (gatewayId -> tenant) only; platform/botId live on the
    # per-platform route rows (relayProvision.ts:124/148), so N provision POSTs
    # with one gatewayId add N platforms' routes under one secret. The loop is
    # PARTIAL-FAILURE-TOLERANT: a platform that fails to provision is logged and
    # skipped (it just isn't fronted) — the others still come up. The FIRST
    # successful provision sets the in-process creds; later platforms re-provision
    # against the same gatewayId (idempotent on the secret, additive on routes).
    provisioned: list[str] = []
    result: dict = {}
    for platform, bot_id in identities:
        try:
            result = _post_provision(
                provision_url=_provision_url(dial_url),
                access_token=access_token,
                gateway_id=gateway_id,
                platform=platform,
                bot_id=bot_id,
                gateway_endpoint=endpoint,
                route_keys=route_keys,
                instance_id=instance_id,
                wake_url=wake_url,
            )
        except RuntimeError as exc:
            logger.warning(
                "relay self-provision failed for platform=%s (%s); continuing with the rest",
                platform,
                exc,
            )
            continue
        provisioned.append(platform)
        # Set creds in-process on the FIRST success so register_relay_adapter()
        # reads them from os.environ (the per-gateway secret authenticates the
        # outbound WS upgrade). Subsequent platforms share the same gatewayId +
        # secret (the connector returns the same record for the same gatewayId).
        # Never logged.
        if "GATEWAY_RELAY_SECRET" not in os.environ or not os.environ.get("GATEWAY_RELAY_SECRET"):
            os.environ["GATEWAY_RELAY_ID"] = str(result.get("gatewayId") or gateway_id)
            os.environ["GATEWAY_RELAY_SECRET"] = str(result.get("secret") or "")
            os.environ["GATEWAY_RELAY_DELIVERY_KEY"] = str(result.get("deliveryKey") or "")

    if not provisioned:
        logger.warning(
            "relay self-provision failed for ALL platforms (%s); gateway will boot without relay auth",
            ",".join(p for p, _ in identities),
        )
        return False

    tenant = str(result.get("tenant") or "")
    logger.info(
        "relay self-provisioned (gateway_id=%s tenant=%s platforms=%s routes=%d inbound=%s instance=%s wake=%s)",
        os.environ.get("GATEWAY_RELAY_ID", gateway_id),
        tenant or "?",
        ",".join(provisioned),
        len(route_keys),
        "yes" if endpoint else "outbound-only",
        instance_id or "unbound",
        "yes" if wake_url else "none",
    )
    return True


def _post_policy(*, policy_url: str, token: str, policy: dict, timeout: float = 15.0) -> int:
    """POST the relevance policy to the connector's ``/relay/policy``; return the HTTP status.

    Authenticated with the gateway's own per-gateway upgrade token (the SAME
    bearer shape as the WS upgrade — ``make_upgrade_token``), so the connector
    resolves ``{tenant, instanceId}`` from its stored secret record, never the
    body. Raises RuntimeError on transport failure (the caller treats any
    failure as non-fatal — relevance is an optimization, not a boot dependency).
    """
    import json
    import urllib.error
    import urllib.request

    data = json.dumps(policy).encode("utf-8")
    req = urllib.request.Request(
        policy_url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach connector: {exc.reason}") from exc


def send_relay_policy() -> bool:
    """Declare this gateway's relevance policy to the connector (Phase 6 Unit ζ).

    Runs at boot AFTER the per-gateway secret is resolved (self-provisioned or
    pinned), projecting the agent's relevance config into the generic vocabulary
    (``relay_relevance_policy``) and POSTing it to ``/relay/policy`` with the
    gateway's own upgrade token. The connector stores it per-instance and the
    relevance gate enforces it on delivery — so the SAME mention-gating /
    free-response / allow-bots behavior the agent applies directly also governs
    relay delivery, and excluded traffic never wakes a scaled-to-zero agent.

    Self-healing: the agent is the source of truth and re-declares every boot
    (mirrors the ``routeKeys`` upsert at provision). Idempotent — a full replace.

    NEVER raises and NEVER blocks boot: relevance is an optimization layered on
    the δ/ε authorization gate (which already protects isolation), so a failed
    declaration just means the connector keeps the prior/quiet policy. Returns
    True iff the connector accepted the policy (HTTP 200).
    """
    import logging

    logger = logging.getLogger("gateway.relay")

    dial_url = relay_url()
    if not dial_url:
        return False

    gateway_id, secret = relay_connection_auth()
    if not gateway_id or not secret:
        # No resolved per-gateway secret (unenrolled / provision failed) ⇒ we
        # can't authenticate the policy POST; skip quietly (the WS upgrade would
        # be unauthenticated too, so there's no instance to attach a policy to).
        return False

    # Phase 1.5: declare a policy PER fronted platform — the connector keys the
    # relevance policy by (tenant, platform, instanceId), so each platform this
    # gateway fronts gets its own row. Per-platform, partial-tolerant: a platform
    # with nothing non-default to declare is skipped; a failed POST for one
    # platform doesn't block the others. A single-platform gateway declares one
    # policy exactly as before.
    try:
        from gateway.relay.auth import make_upgrade_token

        token = make_upgrade_token(gateway_id, secret)
    except Exception as exc:  # noqa: BLE001 - boot must survive a token-build failure
        logger.warning("relay policy declaration failed to build token (%s); connector keeps prior policy", exc)
        return False

    any_declared = False
    for platform, _bot_id in relay_platform_identities():
        policy = relay_relevance_policy(platform)
        if policy is None:
            # Nothing non-default to declare for this platform ⇒ the connector's
            # quiet default already matches; don't write a redundant row.
            continue
        try:
            status = _post_policy(policy_url=_policy_url(dial_url), token=token, policy=policy)
        except Exception as exc:  # noqa: BLE001 - boot must survive a policy-declare failure
            logger.warning(
                "relay policy declaration failed for platform=%s (%s); continuing", platform, exc
            )
            continue
        if status == 200:
            any_declared = True
            logger.info(
                "relay policy declared (platform=%s require_address=%s free_scopes=%d allow_bots=%s)",
                policy.get("platform"),
                policy.get("requireAddress"),
                len(policy.get("freeResponseScopes") or []),
                policy.get("allowOtherBots"),
            )
        else:
            logger.warning(
                "relay policy declaration for platform=%s returned HTTP %s; connector keeps prior/default policy",
                platform,
                status,
            )
    return any_declared


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
                # Phase 1.5: the full SET of (platform, bot_id) this gateway fronts.
                # The transport sends one hello per identity (the connector
                # accumulates them) and resolves the per-frame egress botId from
                # this set. A single-platform deploy passes a 1-element list, so
                # behaviour is byte-identical to before.
                identities=relay_platform_identities(),
                gateway_id=gateway_id,
                upgrade_secret=upgrade_secret,
                # Phase 5 §5.3: re-dial + re-handshake after an unexpected socket
                # close so a gateway that went idle/suspended re-establishes its
                # relay socket — which triggers the connector's buffered-flip drain
                # (the delivery-leg onResume) on the new handshake.
                reconnect=True,
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
