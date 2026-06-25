"""``hermes gateway enroll`` — enroll a self-hosted gateway with a relay connector.

The connector⇄gateway channel is authenticated (the gateway may be
customer-managed and internet-exposed). This command is the gateway half of the
zero-touch enrollment in the connector repo's
``docs/connector-gateway-auth-design.md``:

  1. Resolve a fresh Nous Portal access token from the existing login
     (``~/.hermes/auth.json``) — the same path ``hermes dashboard register``
     uses (``resolve_nous_access_token``). This proves *which Nous org (tenant)*
     the caller owns; the connector derives the authoritative tenant from it via
     ``GET /api/oauth/account`` (never from anything the gateway asserts).
  2. POST ``{enrollmentToken, gatewayId}`` to the connector's ``/relay/enroll``
     with that token in the ``Authorization`` header, over TLS.
  3. The connector verifies the enrollment token (signature + single-use +
     tenant match), mints a per-gateway secret, get-or-creates the per-tenant
     delivery key, and returns both ONCE.
  4. Persist ``GATEWAY_RELAY_ID`` / ``GATEWAY_RELAY_SECRET`` /
     ``GATEWAY_RELAY_DELIVERY_KEY`` (+ ``GATEWAY_RELAY_URL`` if supplied) into
     ``~/.hermes/.env``. The per-gateway secret authenticates the WS upgrade;
     the per-tenant delivery key verifies signed inbound deliveries.

Managed/hosted installs do NOT self-enroll: the orchestrator (NAS) mints the
secret directly and stamps it into the container env, so this command refuses to
run under ``is_managed()`` (mirrors ``dashboard register``).

EXPERIMENTAL: the relay auth scheme may change without a deprecation cycle until
≥2 Class-1 platforms validate the contract.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import urllib.error
import urllib.request
from typing import Optional


def _default_gateway_id() -> str:
    """A stable-ish default gateway instance id: ``<hostname>-<pid-free slug>``.

    The gatewayId identifies this enrolled instance for kill-switch granularity
    (the connector indexes its secret verify list by it). Default to the host
    name so a human can recognize it; overridable via ``--gateway-id``.
    """
    host = ""
    try:
        host = socket.gethostname().strip()
    except Exception:
        host = ""
    return f"gw-{host or 'hermes'}"


def _resolve_connector_url(override: Optional[str]) -> Optional[str]:
    """Resolve the connector base URL (no trailing slash) for enrollment.

    Precedence: explicit ``--connector-url`` flag > ``GATEWAY_RELAY_URL`` env >
    ``gateway.relay_url`` in config.yaml. The relay URL is a ``ws(s)://`` dial
    target; enrollment is an ``http(s)://`` POST to the same host, so we map the
    scheme. Returns None when nothing is configured (the user must supply one).
    """
    raw = (override or os.environ.get("GATEWAY_RELAY_URL", "")).strip()
    if not raw:
        try:
            from gateway.run import _load_gateway_config  # late import to avoid cycle

            cfg = (_load_gateway_config().get("gateway") or {})
            raw = str(cfg.get("relay_url", "") or "").strip()
        except Exception:
            raw = ""
    if not raw:
        return None
    raw = raw.rstrip("/")
    # The relay dial URL is ws(s)://…/relay; enrollment posts to http(s)://…/relay/enroll.
    if raw.startswith("ws://"):
        raw = "http://" + raw[len("ws://"):]
    elif raw.startswith("wss://"):
        raw = "https://" + raw[len("wss://"):]
    # Strip a trailing /relay path segment if the user pasted the dial URL.
    if raw.endswith("/relay"):
        raw = raw[: -len("/relay")]
    return raw


def _post_enroll(
    *,
    connector_base_url: str,
    access_token: str,
    enrollment_token: str,
    gateway_id: str,
    timeout: float = 15.0,
) -> dict:
    """POST to the connector's ``/relay/enroll`` and return the JSON body.

    Raises RuntimeError with a user-facing message on any non-2xx / transport
    failure. The connector returns ``{secret, deliveryKey, tenant, gatewayId}``
    on success, ``{error}`` at 400/401/403.
    """
    url = f"{connector_base_url.rstrip('/')}/relay/enroll"
    data = json.dumps({"enrollmentToken": enrollment_token, "gatewayId": gateway_id}).encode("utf-8")
    req = urllib.request.Request(
        url,
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
        if exc.code == 401:
            raise RuntimeError(
                "Connector rejected the caller identity (401). Your Nous Portal "
                "token could not be verified — try `hermes auth login nous` and retry."
            ) from exc
        if exc.code == 403:
            raise RuntimeError(
                detail
                or "Enrollment token invalid, expired, already used, or tenant mismatch (403)."
            ) from exc
        raise RuntimeError(
            f"Connector returned HTTP {exc.code}" + (f": {detail}" if detail else "")
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach the connector at {connector_base_url}: {exc.reason}"
        ) from exc

    if not isinstance(payload, dict) or not payload.get("secret"):
        raise RuntimeError("Connector returned an unexpected response (no secret).")
    return payload


def cmd_gateway_enroll(args) -> None:
    """Enroll this gateway with a relay connector; persist the auth creds to .env."""
    from hermes_cli.auth import AuthError, resolve_nous_access_token
    from hermes_cli.config import is_managed, save_env_value

    # Managed installs get GATEWAY_RELAY_* stamped in by the orchestrator (NAS
    # mints the secret directly per the design's managed shape). Self-enrolling
    # from inside such a container is a mistake — and save_env_value refuses to
    # write anyway.
    if is_managed():
        print(
            "✗ `hermes gateway enroll` is not available in a managed/hosted install.\n"
            "  The relay gateway secret is provisioned by the hosting platform."
        )
        sys.exit(1)

    enrollment_token = (getattr(args, "token", None) or os.environ.get("GATEWAY_RELAY_ENROLL_TOKEN", "")).strip()
    if not enrollment_token:
        print(
            "✗ No enrollment token. Pass --token <token> (or set "
            "GATEWAY_RELAY_ENROLL_TOKEN).\n"
            "  The connector mints this single-use token when your tenant's route "
            "is provisioned; it is delivered with your gateway config."
        )
        sys.exit(1)

    connector_base_url = _resolve_connector_url(getattr(args, "connector_url", None))
    if not connector_base_url:
        print(
            "✗ No connector URL. Pass --connector-url <url> (or set GATEWAY_RELAY_URL "
            "/ gateway.relay_url in config.yaml)."
        )
        sys.exit(1)

    gateway_id = (getattr(args, "gateway_id", None) or _default_gateway_id()).strip()

    # 1. Resolve a fresh Nous access token (the tenant-proving identity).
    try:
        access_token = resolve_nous_access_token()
    except AuthError as exc:
        if getattr(exc, "relogin_required", False):
            print("✗ You're not logged into Nous Portal.")
            print("  Run `hermes setup` (or `hermes auth login nous`) first, then retry.")
        else:
            print(f"✗ Could not resolve a Nous Portal access token: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"✗ Could not resolve a Nous Portal access token: {exc}")
        sys.exit(1)

    # 2-3. Redeem the enrollment token at the connector.
    try:
        result = _post_enroll(
            connector_base_url=connector_base_url,
            access_token=access_token,
            enrollment_token=enrollment_token,
            gateway_id=gateway_id,
        )
    except RuntimeError as exc:
        print(f"✗ Enrollment failed: {exc}")
        sys.exit(1)

    secret = str(result.get("secret") or "")
    delivery_key = str(result.get("deliveryKey") or "")
    tenant = str(result.get("tenant") or "")
    resolved_gateway_id = str(result.get("gatewayId") or gateway_id)

    # 4. Persist the creds idempotently. The secret + delivery key are sensitive;
    #    save_env_value writes them to ~/.hermes/.env (0600 dir) and never logs.
    to_write = {
        "GATEWAY_RELAY_ID": resolved_gateway_id,
        "GATEWAY_RELAY_SECRET": secret,
        "GATEWAY_RELAY_DELIVERY_KEY": delivery_key,
    }
    # Persist the connector URL too (as the ws(s):// dial target) when supplied
    # explicitly, so the runtime can dial without re-specifying it.
    explicit_url = (getattr(args, "connector_url", None) or "").strip()
    if explicit_url:
        to_write["GATEWAY_RELAY_URL"] = explicit_url.rstrip("/")

    # Phase 5 §5.2: persist the wake URL so self_provision_relay forwards it to
    # the connector (which pokes it to wake this gateway when buffered work
    # arrives while it's idle). Optional — omitted ⇒ the connector can't wake it,
    # but the gateway still drains on its next reconnect.
    explicit_wake_url = (getattr(args, "wake_url", None) or "").strip()
    if explicit_wake_url:
        to_write["GATEWAY_RELAY_WAKE_URL"] = explicit_wake_url.rstrip("/")

    for key, value in to_write.items():
        if not value:
            continue
        try:
            save_env_value(key, value)
        except Exception as exc:
            print(f"✗ Failed to write {key} to .env: {exc}")
            sys.exit(1)

    from hermes_cli.config import get_env_path

    print(f'✓ Enrolled gateway "{resolved_gateway_id}"' + (f" for tenant {tenant}" if tenant else ""))
    print()
    print(f"  Wrote to {get_env_path()}:")
    print(f"    GATEWAY_RELAY_ID={resolved_gateway_id}")
    print("    GATEWAY_RELAY_SECRET=<hidden>")
    print("    GATEWAY_RELAY_DELIVERY_KEY=<hidden>")
    if explicit_url:
        print(f"    GATEWAY_RELAY_URL={explicit_url.rstrip('/')}")
    if explicit_wake_url:
        print(f"    GATEWAY_RELAY_WAKE_URL={explicit_wake_url.rstrip('/')}")
    print()
    print(
        "  The gateway now authenticates its relay WS upgrade with the per-gateway\n"
        "  secret and verifies signed inbound deliveries with the tenant delivery\n"
        "  key. Restart the gateway to pick up the new env."
    )
