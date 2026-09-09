"""Relay/connector support package for the Hermes gateway.

EXPERIMENTAL gateway side of the "Gateway Gateway" relay design: a generic
``RelayAdapter`` plus the wire-serializable ``CapabilityDescriptor`` the connector
hands it at handshake, and the production ``WebSocketRelayTransport``. The public
API MAY CHANGE without a deprecation cycle until >=2 real Class-1 platforms have
shaken out the schema (``docs/relay-connector-contract.md``). Activation is
config-driven: the relay platform is registered when a connector relay URL is set
(``GATEWAY_RELAY_URL`` env or ``gateway.relay_url``), like ``gateway.proxy_url``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

logger = logging.getLogger("gateway.relay")

# Shape gate for ambient-endpoint token bodies (mode 1b in
# _resolve_relay_identity_token): a JWT (3+ dotted segments) or a long opaque
# base64url token (>= 32 chars). Short bare words — 'unauthorized', 'error',
# 'null' — match the alphabet but are plain-text error bodies and must fail closed.
_AMBIENT_TOKEN_SHAPE = re.compile(
    r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+){2,}"  # JWT-like: 3+ dotted segments
    r"|[A-Za-z0-9_-]{32,}"  # long opaque bearer token
)

_HTTP_TIMEOUT_S = 15.0

# gateway.idp.* keys (and GATEWAY_RELAY_IDP_<KEY> env mirrors) read by the identity resolver.
_IDP_KEYS = ("token_url", "client_id", "client_secret", "scope")


# ─────────────────────────── config access ───────────────────────────


def _load_cfg() -> dict:
    """The full gateway config, or ``{}`` — config absence/parse must never crash boot."""
    try:
        from gateway.run import _load_gateway_config  # late import to avoid cycle

        cfg = _load_gateway_config()
    except Exception:  # noqa: BLE001
        return {}
    return cfg if isinstance(cfg, dict) else {}


def _gateway_cfg() -> dict:
    """The ``gateway:`` block of config.yaml, or ``{}``."""
    block = _load_cfg().get("gateway")
    return block if isinstance(block, dict) else {}


def _env_or_cfg(env_var: str, cfg_key: str) -> str:
    """Env var first (Docker/NAS stamp), then ``gateway.<cfg_key>``; stripped, ``""`` when unset."""
    value = os.environ.get(env_var, "").strip()
    return value or str(_gateway_cfg().get(cfg_key, "") or "").strip()


def _env_or_cfg_url(env_var: str, cfg_key: str) -> Optional[str]:
    """``_env_or_cfg`` for URL-shaped values: trailing slash stripped, None when unset."""
    return _env_or_cfg(env_var, cfg_key).rstrip("/") or None


def relay_url() -> Optional[str]:
    """The connector relay endpoint URL, or None. A non-empty value activates the relay platform."""
    return _env_or_cfg_url("GATEWAY_RELAY_URL", "relay_url")


def relay_platform_identities() -> list[tuple[str, str]]:
    """The ordered (platform, bot_id) pairs this gateway fronts on one WS connection,
    from ``GATEWAY_RELAY_PLATFORMS`` (comma-sep) and ``GATEWAY_RELAY_BOT_IDS`` (JSON
    ``{"discord": {"botId": "..."}}``). The FIRST pair is the handshake/descriptor
    default; a platform absent from the ids map gets an empty bot_id (the connector
    rejects it with a structured failure). ``[("relay", "")]`` when nothing is set."""
    platforms = [p.strip() for p in os.environ.get("GATEWAY_RELAY_PLATFORMS", "").split(",") if p.strip()]
    if not platforms:
        return [("relay", "")]
    ids = _relay_bot_ids_map()
    out: list[tuple[str, str]] = []
    for platform in platforms:
        entry = ids.get(platform) or {}
        bot_id = str(entry.get("botId", "")).strip() if isinstance(entry, dict) else ""
        out.append((platform, bot_id))
    return out


def relay_fronted_platforms() -> set[str]:
    """Logical platform names the relay fronts, minus the generic ``relay`` fallback.
    Same env source the live adapter's identity set comes from, so config-time
    validation (cron delivery preflight) and fire-time routing can never disagree —
    and it needs no live adapter, so a standalone scheduler can use it."""
    return {p for p, _ in relay_platform_identities() if p != "relay"}


def _relay_bot_ids_map() -> dict:
    """Parse ``GATEWAY_RELAY_BOT_IDS``; a malformed map yields ``{}`` rather than crashing boot."""
    raw = os.environ.get("GATEWAY_RELAY_BOT_IDS", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:  # noqa: BLE001
        logger.warning("GATEWAY_RELAY_BOT_IDS is not valid JSON; treating as empty")
        return {}


def relay_platform_identity() -> tuple[str, str]:
    """The PRIMARY (platform, bot_id) — first of ``relay_platform_identities()``."""
    return relay_platform_identities()[0]


def relay_connection_auth() -> tuple[Optional[str], Optional[str]]:
    """The (gateway_id, upgrade_secret) from enrollment (``GATEWAY_RELAY_ID`` /
    ``GATEWAY_RELAY_SECRET``, then ``gateway.relay_id`` / ``gateway.relay_secret``).
    Either absent -> ``(None, None)`` and the transport dials unauthenticated."""
    gateway_id = _env_or_cfg("GATEWAY_RELAY_ID", "relay_id")
    secret = _env_or_cfg("GATEWAY_RELAY_SECRET", "relay_secret")
    return (gateway_id or None, secret or None)


def relay_endpoint() -> Optional[str]:
    """The gateway's own PUBLIC inbound URL, asserted to the connector at provision
    (``GATEWAY_RELAY_ENDPOINT`` / ``gateway.relay_endpoint``). Stored on the tenant's
    route rows; gateway-asserted but tenant-scoped, so a dishonest gateway can only
    misdirect its OWN inbound. Absent -> outbound-only (no inbound routes written)."""
    return _env_or_cfg_url("GATEWAY_RELAY_ENDPOINT", "relay_endpoint")


def relay_route_keys() -> list[str]:
    """Discriminators (scope_ids / chat_ids / paths) this gateway's tenant owns. The
    connector writes one route row per (routeKey -> tenant, endpoint), so keys only
    take effect alongside ``relay_endpoint()``. ``GATEWAY_RELAY_ROUTE_KEYS`` is
    comma-separated; ``gateway.relay_route_keys`` may be a list or a comma string."""
    raw = os.environ.get("GATEWAY_RELAY_ROUTE_KEYS", "").strip()
    if not raw:
        val = _gateway_cfg().get("relay_route_keys", "")
        if isinstance(val, (list, tuple)):
            return [str(k).strip() for k in val if str(k).strip()]
        raw = str(val or "").strip()
    return [k.strip() for k in raw.split(",") if k.strip()]


def relay_instance_id() -> Optional[str]:
    """Stable per-instance id forwarded at provision (``GATEWAY_RELAY_INSTANCE_ID`` /
    ``gateway.relay_instance_id``): binds the connector's ``gatewayId -> instanceId``
    so inbound routes per-instance rather than tenant-broadcast (NAS stamps its
    ``AgentInstance.id``). Tenant-scoped like ``relay_endpoint()``; absent -> null."""
    return _env_or_cfg("GATEWAY_RELAY_INSTANCE_ID", "relay_instance_id") or None


def relay_wake_url() -> Optional[str]:
    """The gateway's WAKE URL forwarded at provision (``GATEWAY_RELAY_WAKE_URL`` /
    ``gateway.relay_wake_url``): a payload-free poke the connector GETs when a
    going-idle destination receives its first buffered event, so a suspended gateway
    wakes, reconnects and drains. Absent -> no wake (buffering still works)."""
    return _env_or_cfg_url("GATEWAY_RELAY_WAKE_URL", "relay_wake_url")


def relay_display_name() -> Optional[str]:
    """The human-facing agent display name forwarded at provision — the connector's
    multi-agent reply-attribution prefix (``**<displayName>:** ``).
    ``GATEWAY_RELAY_DISPLAY_NAME`` env, then the skin's branded agent name (a skin
    rename propagates on the next boot). Absent -> connector's linked-owner fallback.

    The PRIMARY source for the connector's multi-agent reply-attribution prefix (gateway-gateway #171): in a
    multi-agent scope the shared bot prepends ``**<displayName>:** `` to this instance's replies.
    Gateway-asserted but safely scoped exactly like ``relay_instance_id()`` / ``relay_wake_url()`` — the
    tenant stays token-verified, so a dishonest gateway can only label its OWN instance. Absent -> the
    connector stores null and attribution falls back to the instance's linked-owner identity, else skips the
    prefix.
    """
    value = os.environ.get("GATEWAY_RELAY_DISPLAY_NAME", "").strip()
    if not value:
        try:
            from hermes_cli.skin_engine import get_active_skin  # late import: boot-safe

            value = str(get_active_skin().get_branding("agent_name", "") or "").strip()
        except Exception:  # noqa: BLE001 - branding absence must never crash boot
            value = ""
        # The stock brand is identical on every default install: forwarding it would
        # prefix every reply "**Hermes Agent:**" and shadow the connector's
        # linked-owner fallback, which actually disambiguates. Only a customized
        # name is forwarded.
        if value == "Hermes Agent":
            value = ""
    # Mirror the connector's ingest sanitization (trim + 64-char cap).
    return value[:64] or None


# ─────────────────────────── connector HTTP ───────────────────────────


def _connector_url(relay_dial_url: str, path: str) -> str:
    """``ws(s)://…/relay`` dial URL -> ``http(s)://…{path}`` connector route."""
    from gateway.relay.media import media_base_url

    return f"{media_base_url(relay_dial_url)}{path}"


def _provision_url(relay_dial_url: str) -> str:
    return _connector_url(relay_dial_url, "/relay/provision")


def _json_post(url: str, token: str, body: dict, timeout: float):
    """Bearer-authenticated JSON POST; returns the open response (caller reads it)."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    return urllib.request.urlopen(req, timeout=timeout)


def relay_relevance_policy(platform: Optional[str] = None) -> Optional[dict]:
    """Project a fronted platform's RELEVANCE config into the connector's generic vocabulary.

    The connector's relevance gate reasons over a platform-agnostic policy keyed by
    ``(tenant, platform, instanceId)``: ``requireAddress`` <- ``require_mention``,
    ``freeResponseScopes`` <- ``free_response_channels``, ``allowOtherBots`` <-
    ``{PLATFORM}_ALLOW_BOTS`` in {"mentions","all"}. Read from the platform's config
    block (``discord:``), falling back to the bridged top-level keys, then env.
    ``platform`` defaults to the PRIMARY fronted platform. Returns None when relay
    isn't configured or nothing is CONFIGURED to declare, so the connector's default
    (mention-gated) applies. The condition is "require_mention is unset", NOT "falsy":
    an EXPLICIT ``require_mention: false`` is a non-default choice that MUST be
    declared or the connector would mention-gate an agent configured to free-respond.
    """
    if platform is None:
        platform, _bot_id = relay_platform_identity()
    if not platform or platform == "relay":
        return None

    require_mention = None
    free_response: list[str] = []
    try:
        cfg = _load_cfg()
        # Platform block lookup order: top-level ``<platform>:``, then
        # ``gateway.platforms.<platform>``, then ``platforms.<platform>``.
        def _candidates():
            yield cfg.get(platform)
            gw_platforms = (cfg.get("gateway") or {}).get("platforms") or {}
            yield gw_platforms.get(platform) if isinstance(gw_platforms, dict) else None
            yield (cfg.get("platforms") or {}).get(platform)

        plat_cfg = next((c for c in _candidates() if isinstance(c, dict)), {})

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

    # Same gate as the gateway's own authz_mixin DISCORD_ALLOW_BOTS bypass.
    allow_bots_env = os.environ.get(f"{platform.upper()}_ALLOW_BOTS", "").lower().strip()
    allow_other_bots = allow_bots_env in {"mentions", "all"}

    if require_mention is None and not free_response and not allow_other_bots:
        return None
    return {
        "platform": platform,
        "requireAddress": bool(require_mention),
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
    display_name: Optional[str] = None,
    timeout: float = _HTTP_TIMEOUT_S,
) -> dict:
    """POST ``/relay/provision``; return ``{secret, deliveryKey, tenant, gatewayId,
    routeKeys}`` (the connector validates ``access_token`` against NAS, derives the
    tenant, mints the per-gateway secret and upserts route rows). Raises RuntimeError
    with a user-facing message on any non-2xx / transport failure."""
    body: dict = {
        "gatewayId": gateway_id,
        "platform": platform,
        "botId": bot_id,
        "gatewayEndpoint": gateway_endpoint or "",
        "routeKeys": route_keys,
    }
    # Optional fields are OMITTED when absent so the connector stores null
    # (back-compat) rather than binding an empty string.
    for key, value in (("instanceId", instance_id), ("wakeUrl", wake_url), ("displayName", display_name)):
        if value:
            body[key] = value
    try:
        with _json_post(provision_url, access_token, body, timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = (json.loads(exc.read().decode()) or {}).get("error", "")
        except Exception:  # noqa: BLE001
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

    Canonical resolver shared by runtime self-provision and ``hermes gateway enroll``.
    Modes, in precedence order:
      1.  Generic OIDC client-credentials (self-hosted IdP): ``gateway.idp.token_url``
          (``GATEWAY_RELAY_IDP_TOKEN_URL``) set together with client id + secret ->
          POST the OAuth2 ``client_credentials`` grant.
      1b. Ambient token endpoint: ``token_url`` with NEITHER client_id nor
          client_secret is a metadata-server-style endpoint (e.g. Domino's
          ``$DOMINO_API_PROXY/access-token``): a plain GET whose body IS the token,
          raw JWT or a JSON envelope with ``access_token``. Possession of the
          (typically loopback) endpoint is the credential.
      2.  Nous Portal (default): ``resolve_nous_access_token()``.

    Raises on failure; callers decide whether that's fatal (enroll CLI) or a graceful
    boot no-op (self-provision).
    """
    env = {k: os.environ.get(f"GATEWAY_RELAY_IDP_{k.upper()}", "").strip() for k in _IDP_KEYS}
    if not env["token_url"]:
        # Env token_url absent -> the whole idp block comes from config (env id/secret
        # still win per key). A malformed gateway.idp degrades to "no token_url".
        idp = _gateway_cfg().get("idp")
        idp = idp if isinstance(idp, dict) else {}
        for k in _IDP_KEYS:
            env[k] = env[k] or str(idp.get(k, "") or "").strip()
    token_url, client_id, client_secret, scope = (env[k] for k in _IDP_KEYS)

    if not token_url:
        from hermes_cli.auth import resolve_nous_access_token

        return resolve_nous_access_token()

    if not client_id and not client_secret:
        # Mode 1b — plain GET; the body is the token, raw or JSON-enveloped.
        req = urllib.request.Request(
            token_url, method="GET", headers={"Accept": "application/json, text/plain"}
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            body = resp.read().decode().strip()
        token = ""
        if body.startswith("{"):
            try:
                envelope_token = (json.loads(body) or {}).get("access_token")
            except ValueError:
                envelope_token = None
            # No shape gate on an envelope: it is a deliberate token response, and
            # opaque tokens may use the standard-base64 alphabet the raw gate rejects.
            if isinstance(envelope_token, str):
                token = envelope_token.strip()
        elif _AMBIENT_TOKEN_SHAPE.fullmatch(body):
            token = body
        if not token:
            raise RuntimeError(
                "no client_id/client_secret configured, so gateway.idp.token_url was "
                "treated as an ambient token endpoint (GET), but the response body "
                "was not a token. For the OAuth2 client_credentials grant, configure "
                "client_id and client_secret alongside token_url."
            )
        return token

    if not client_id or not client_secret:
        # Exactly one credential: a mistyped client_credentials setup, not an
        # ambient endpoint. Fail loud; never GET the IdP.
        missing = "client_secret" if client_id else "client_id"
        raise RuntimeError(
            f"gateway.idp.token_url is configured with a partial client credential "
            f"({missing} missing). Configure both client_id and client_secret for "
            f"the OAuth2 client_credentials grant, or neither to treat token_url "
            f"as an ambient token endpoint (plain GET returning the token)."
        )

    # Mode 1 — OAuth2 client_credentials grant.
    form = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
    if scope:
        form["scope"] = scope
    req = urllib.request.Request(
        token_url, data=urllib.parse.urlencode(form).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
        payload = json.loads(resp.read().decode())
    access_token = (payload or {}).get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise RuntimeError("IdP client_credentials response had no access_token")
    return access_token.strip()


def self_provision_relay() -> bool:
    """Boot-time relay self-provision: mint relay creds in-process, no human, no disk.

    Fires when ``relay_url()`` is set and NO per-gateway secret is pinned: resolves the
    identity token, POSTs ``/relay/provision`` for EACH fronted platform and sets
    ``GATEWAY_RELAY_ID`` / ``_SECRET`` / ``_DELIVERY_KEY`` in ``os.environ`` for
    ``register_relay_adapter()``. Creds live ONLY in process memory (never
    ``~/.hermes/.env``), so a hosted container re-provisions every boot; the
    connector's rotation window covers a still-connected prior instance. The trigger
    is deliberately NOT ``is_managed()`` (False on a NAS-hosted Fly agent): "pointed
    at a connector without a pinned secret" is the real signal and self-guards (an
    enrolled gateway has a PINNED secret -> skipped; no resolvable identity -> no-op).
    Returns True iff it provisioned. NEVER raises: a failure logs and returns False so
    the gateway still boots (the adapter then dials unauthenticated / is rejected).
    """
    dial_url = relay_url()
    if not dial_url:
        return False

    existing_id, existing_secret = relay_connection_auth()
    if existing_id and existing_secret:
        logger.info("relay self-provision skipped: GATEWAY_RELAY_SECRET already set")
        return False

    try:
        access_token = _resolve_relay_identity_token()
    except Exception as exc:  # noqa: BLE001 - boot must survive a token failure
        logger.warning("relay self-provision skipped: could not resolve identity token (%s)", exc)
        return False

    identities = relay_platform_identities()
    # gatewayId default mirrors the enroll CLI's hostname-based slug.
    try:
        host = socket.gethostname().strip()
    except Exception:  # noqa: BLE001
        host = ""
    gateway_id = os.environ.get("GATEWAY_RELAY_ID", "").strip() or f"gw-{host or 'hermes'}"
    endpoint = relay_endpoint()
    route_keys = relay_route_keys()
    instance_id = relay_instance_id()
    wake_url = relay_wake_url()
    display_name = relay_display_name()

    # Provision EACH fronted platform under the SAME gatewayId + the SAME per-gateway
    # secret: the connector's secret record is (gatewayId -> tenant) only, platform/
    # botId live on per-platform route rows, so N POSTs with one gatewayId are
    # idempotent on the secret and additive on routes. PARTIAL-FAILURE-TOLERANT: a
    # platform that fails is logged and skipped (just not fronted); the others come up.
    provisioned: list[str] = []
    result: dict = {}
    for platform, bot_id in identities:
        try:
            result = _post_provision(
                provision_url=_provision_url(dial_url), access_token=access_token,
                gateway_id=gateway_id, platform=platform, bot_id=bot_id, gateway_endpoint=endpoint,
                route_keys=route_keys, instance_id=instance_id, wake_url=wake_url,
                display_name=display_name,
            )
        except RuntimeError as exc:
            logger.warning(
                "relay self-provision failed for platform=%s (%s); continuing with the rest",
                platform, exc,
            )
            continue
        provisioned.append(platform)
        # Set creds in-process on the FIRST success (the per-gateway secret
        # authenticates the outbound WS upgrade). Never logged.
        if not os.environ.get("GATEWAY_RELAY_SECRET"):
            os.environ["GATEWAY_RELAY_ID"] = str(result.get("gatewayId") or gateway_id)
            os.environ["GATEWAY_RELAY_SECRET"] = str(result.get("secret") or "")
            os.environ["GATEWAY_RELAY_DELIVERY_KEY"] = str(result.get("deliveryKey") or "")

    if not provisioned:
        logger.warning(
            "relay self-provision failed for ALL platforms (%s); gateway will boot without relay auth",
            ",".join(p for p, _ in identities),
        )
        return False

    logger.info(
        "relay self-provisioned (gateway_id=%s tenant=%s platforms=%s routes=%d inbound=%s instance=%s wake=%s)",
        os.environ.get("GATEWAY_RELAY_ID", gateway_id),
        str(result.get("tenant") or "") or "?",
        ",".join(provisioned),
        len(route_keys),
        "yes" if endpoint else "outbound-only",
        instance_id or "unbound",
        "yes" if wake_url else "none",
    )
    return True


def _post_policy(*, policy_url: str, token: str, policy: dict, timeout: float = _HTTP_TIMEOUT_S) -> int:
    """POST the relevance policy to ``/relay/policy``; return the HTTP status.
    Authenticated with the gateway's own upgrade token, so the connector resolves
    ``{tenant, instanceId}`` from its stored secret record, never the body. Raises
    RuntimeError on transport failure."""
    try:
        with _json_post(policy_url, token, policy, timeout) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach connector: {exc.reason}") from exc


def send_relay_policy() -> bool:
    """Declare this gateway's relevance policy to the connector, per fronted platform,
    at boot AFTER the per-gateway secret is resolved. The connector enforces it on
    delivery, so the SAME mention-gating / free-response / allow-bots behavior the
    agent applies directly also governs relay delivery, and excluded traffic never
    wakes a scaled-to-zero agent. Re-declared every boot (idempotent full replace); a
    platform with nothing non-default is skipped; one failed POST doesn't block the
    others. NEVER raises / blocks boot: relevance is an optimization layered on the
    authorization gate, so a failure just leaves the connector's prior policy.
    Returns True iff the connector accepted at least one policy (HTTP 200)."""
    dial_url = relay_url()
    if not dial_url:
        return False

    gateway_id, secret = relay_connection_auth()
    if not gateway_id or not secret:
        # Can't authenticate the POST (and there's no instance to attach a policy to).
        return False

    try:
        from gateway.relay.auth import make_upgrade_token

        token = make_upgrade_token(gateway_id, secret)
    except Exception as exc:  # noqa: BLE001 - boot must survive a token-build failure
        logger.warning("relay policy declaration failed to build token (%s); connector keeps prior policy", exc)
        return False

    policy_url = _connector_url(dial_url, "/relay/policy")
    any_declared = False
    for platform, _bot_id in relay_platform_identities():
        policy = relay_relevance_policy(platform)
        if policy is None:
            continue
        try:
            status = _post_policy(policy_url=policy_url, token=token, policy=policy)
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
    """Register the generic ``relay`` platform when a relay URL is configured (or
    ``force=True`` for tests: transport-less adapter). Returns True if registered.
    With a URL the factory builds a live ``WebSocketRelayTransport``; the adapter
    negotiates the real ``CapabilityDescriptor`` at ``connect()``."""
    resolved_url = url if url is not None else relay_url()
    if not (force or resolved_url):
        return False

    from gateway.platform_registry import PlatformEntry, platform_registry
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

    platform, bot_id = relay_platform_identity()

    def _factory(config):
        # Placeholder descriptor; replaced by the negotiated one at connect time.
        # With no URL (force/test) the adapter is transport-less and keeps it.
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
                # The full SET of identities: one hello per identity (the connector
                # accumulates them) and the per-frame egress botId resolves from it.
                identities=relay_platform_identities(),
                gateway_id=gateway_id,
                upgrade_secret=upgrade_secret,
                # Re-dial + re-handshake after an unexpected close so a gateway that
                # went idle re-establishes its socket, which triggers the connector's
                # buffered-flip drain on the new handshake.
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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

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
# ---- END PLUGIN-COMPAT ----
