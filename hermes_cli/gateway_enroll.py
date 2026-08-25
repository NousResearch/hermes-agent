"""``hermes gateway enroll`` — enroll a self-hosted gateway with a relay connector.

Managed/hosted installs do NOT self-enroll: the orchestrator mints the secret and stamps it into the
container env, so this refuses to run under ``is_managed()`` (mirrors ``dashboard register``).
EXPERIMENTAL: the relay auth scheme may change without a deprecation cycle.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


def _default_gateway_id() -> str:
    """``gw-<hostname>``: the gatewayId is the connector's kill-switch granularity, so default to
    the host name for recognizability; override via ``--gateway-id``."""
    try:
        host = socket.gethostname().strip()
    except Exception:
        host = ""
    return f"gw-{host or 'hermes'}"


def _resolve_connector_url(override: Optional[str]) -> Optional[str]:
    """Connector base URL (no trailing slash): ``--connector-url`` > ``GATEWAY_RELAY_URL`` >
    ``gateway.relay_url``. The relay URL is a ``ws(s)://…/relay`` dial target; enrollment POSTs to
    ``http(s)://`` on the same host, so map the scheme and strip a pasted ``/relay`` suffix."""
    raw = (override or os.environ.get("GATEWAY_RELAY_URL", "")).strip()
    if not raw:
        try:
            from gateway.run import _load_gateway_config  # late import to avoid cycle

            raw = str((_load_gateway_config().get("gateway") or {}).get("relay_url", "") or "").strip()
        except Exception:
            raw = ""
    if not raw:
        return None
    for ws_scheme, http_scheme in (("ws://", "http://"), ("wss://", "https://")):
        if raw.startswith(ws_scheme):
            raw = http_scheme + raw[len(ws_scheme):]
            break
    return raw.rstrip("/").removesuffix("/relay")


def _post_enroll(
    *,
    connector_base_url: str,
    access_token: str,
    enrollment_token: str,
    gateway_id: str,
    timeout: float = 15.0,
) -> dict:
    """POST to the connector's ``/relay/enroll``; return ``{secret, deliveryKey, tenant, gatewayId}``.
    Raises RuntimeError with a user-facing message on any non-2xx / transport failure."""
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
            message = (
                "Connector rejected the caller identity (401). Your Nous Portal "
                "token could not be verified — try `hermes auth add nous` and retry."
            )
        elif exc.code == 403:
            message = detail or "Enrollment token invalid, expired, already used, or tenant mismatch (403)."
        else:
            message = f"Connector returned HTTP {exc.code}" + (f": {detail}" if detail else "")
        raise RuntimeError(message) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach the connector at {connector_base_url}: {exc.reason}"
        ) from exc

    if not isinstance(payload, dict) or not payload.get("secret"):
        raise RuntimeError("Connector returned an unexpected response (no secret).")
    return payload


def _fail(*lines: str) -> None:
    for line in lines:
        print(line)
    sys.exit(1)


def cmd_gateway_enroll(args) -> None:
    """Enroll this gateway with a relay connector; persist the auth creds to .env."""
    from hermes_cli.auth import AuthError
    from hermes_cli.config import is_managed, save_env_value

    # Managed installs get GATEWAY_RELAY_* stamped in by the orchestrator; save_env_value refuses
    # to write there anyway.
    if is_managed():
        _fail(
            "✗ `hermes gateway enroll` is not available in a managed/hosted install.\n"
            "  The relay gateway secret is provisioned by the hosting platform."
        )

    enrollment_token = (getattr(args, "token", None) or os.environ.get("GATEWAY_RELAY_ENROLL_TOKEN", "")).strip()
    if not enrollment_token:
        _fail(
            "✗ No enrollment token. Pass --token <token> (or set "
            "GATEWAY_RELAY_ENROLL_TOKEN).\n"
            "  The connector mints this single-use token when your tenant's route "
            "is provisioned; it is delivered with your gateway config."
        )

    connector_base_url = _resolve_connector_url(getattr(args, "connector_url", None))
    if not connector_base_url:
        _fail(
            "✗ No connector URL. Pass --connector-url <url> (or set GATEWAY_RELAY_URL "
            "/ gateway.relay_url in config.yaml)."
        )

    gateway_id = (getattr(args, "gateway_id", None) or _default_gateway_id()).strip()

    # Caller-identity token (proves the tenant). ``gateway.relay`` owns the ONE resolver shared with
    # the runtime self-provision path: generic OIDC client-credentials when ``gateway.idp.token_url``
    # is set (air-gapped / self-hosted IdP), otherwise Nous Portal.
    try:
        from gateway.relay import _resolve_relay_identity_token

        access_token = _resolve_relay_identity_token()
    except AuthError as exc:
        if getattr(exc, "relogin_required", False):
            _fail(
                "✗ You're not logged into Nous Portal.",
                "  Run `hermes setup` (or `hermes auth add nous`) first, then retry.",
            )
        _fail(f"✗ Could not resolve a Nous Portal access token: {exc}")
    except Exception as exc:
        _fail(f"✗ Could not resolve a caller-identity token: {exc}")

    try:
        result = _post_enroll(
            connector_base_url=connector_base_url,
            access_token=access_token,
            enrollment_token=enrollment_token,
            gateway_id=gateway_id,
        )
    except RuntimeError as exc:
        _fail(f"✗ Enrollment failed: {exc}")

    tenant = str(result.get("tenant") or "")
    resolved_gateway_id = str(result.get("gatewayId") or gateway_id)

    # Persist idempotently; save_env_value writes the sensitive values to ~/.hermes/.env and never
    # logs them. Explicitly supplied URLs are persisted too: the ws(s):// dial target so the runtime
    # needn't re-specify it, and the wake URL so self_provision_relay forwards it to the connector
    # (which pokes it when buffered work arrives while idle; omitted ⇒ drains on next reconnect).
    to_write = {
        "GATEWAY_RELAY_ID": resolved_gateway_id,
        "GATEWAY_RELAY_SECRET": str(result.get("secret") or ""),
        "GATEWAY_RELAY_DELIVERY_KEY": str(result.get("deliveryKey") or ""),
    }
    explicit_urls = {
        env_key: (getattr(args, arg, None) or "").strip()
        for arg, env_key in (("connector_url", "GATEWAY_RELAY_URL"), ("wake_url", "GATEWAY_RELAY_WAKE_URL"))
    }
    to_write.update({k: v.rstrip("/") for k, v in explicit_urls.items() if v})

    for key, value in to_write.items():
        if not value:
            continue
        try:
            save_env_value(key, value)
        except Exception as exc:
            _fail(f"✗ Failed to write {key} to .env: {exc}")

    from hermes_cli.config import get_env_path

    print(f'✓ Enrolled gateway "{resolved_gateway_id}"' + (f" for tenant {tenant}" if tenant else ""))
    print()
    print(f"  Wrote to {get_env_path()}:")
    for key, value in to_write.items():
        shown = "<hidden>" if key in ("GATEWAY_RELAY_SECRET", "GATEWAY_RELAY_DELIVERY_KEY") else value
        print(f"    {key}={shown}")
    print()
    # GATEWAY_RELAY_URL / GATEWAY_RELAY_WAKE_URL are process-global deployment
    # stamps (agent/secret_scope.py): a multiplexed gateway resolves them from
    # the PROCESS environment only, never from a secondary profile's .env
    # (which is loaded into an isolated secret scope, not exported). The .env
    # write above works for a single-profile gateway and for the profile the
    # process is launched under (load_hermes_dotenv exports that .env), so
    # warn rather than refuse — but don't let a secondary-profile enroll claim
    # a config that will silently never activate. Emitted BEFORE the generic
    # restart line so the two don't contradict each other.
    warned_secondary = False
    if explicit_url or explicit_wake_url:
        warned_secondary = _warn_if_secondary_multiplex_profile()
    if not warned_secondary:
        print(
            "  The gateway now authenticates its relay WS upgrade with the per-gateway\n"
            "  secret and verifies signed inbound deliveries with the tenant delivery\n"
            "  key. Restart the gateway to pick up the new env."
        )


def _warn_if_secondary_multiplex_profile() -> bool:
    """Warn when relay routing stamps were written to a secondary profile's
    .env that a multiplexed gateway will never read them from. Returns True
    when the warning fired (the caller suppresses the generic restart text).

    The topology decision is owned by the DEFAULT root, not the active
    profile home: ``multiplex_profiles`` normally lives in
    ``<default_root>/config.yaml`` (or the GATEWAY_MULTIPLEX_PROFILES env
    override), and the secondary check is the resolved-path relationship to
    ``<default_root>/profiles/`` — mirroring the multiplexer-conflict guard
    in hermes_cli/gateway.py. Best-effort: any failure to determine the
    topology stays silent (the credential write itself succeeded).
    """
    try:
        from hermes_constants import get_default_hermes_root
        from hermes_cli.config import get_hermes_home

        default_root = Path(get_default_hermes_root()).resolve()
        home = Path(get_hermes_home()).resolve()
        try:
            home.relative_to(default_root / "profiles")
        except ValueError:
            return False  # default profile or custom layout — not a secondary

        # Multiplex flag precedence mirrors gateway.config: recognized env
        # override wins, else the DEFAULT root's config.yaml (raw read — the
        # active profile's load_gateway_config() is the wrong owner AND runs
        # the full enablement pass, including the relay-exclusive sweep's own
        # log output, which has no place in enroll output).
        from gateway.config import _env_multiplex_profiles_override
        env_multiplex = _env_multiplex_profiles_override()
        if env_multiplex is False:
            return False
        if env_multiplex is not True:
            cfg_path = default_root / "config.yaml"
            if not cfg_path.exists():
                return False
            from hermes_cli.config import read_user_config_raw
            cfg = read_user_config_raw(cfg_path) or {}
            if not bool(
                cfg.get("multiplex_profiles")
                or (cfg.get("gateway", {}) or {}).get("multiplex_profiles")
            ):
                return False

        print(
            "  ⚠ This profile is a SECONDARY profile of a multiplexed gateway.\n"
            "    GATEWAY_RELAY_URL / GATEWAY_RELAY_WAKE_URL are process-level\n"
            "    deployment settings: the gateway reads them from the process\n"
            "    environment (or the default profile's .env), not from this\n"
            "    profile's .env. Set them in the environment the gateway process\n"
            "    is launched with, or enroll from the default profile. The\n"
            "    relay credentials written above are valid either way."
        )
        return True
    except Exception:
        return False
