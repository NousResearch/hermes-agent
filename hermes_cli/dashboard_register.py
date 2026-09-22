"""``hermes dashboard register`` — register a self-hosted dashboard OAuth client.

Automates the Nous Portal ``/local-dashboards`` flow: resolve a fresh Nous access token, POST
``{portal}/api/oauth/self-hosted-client`` (the ``agent:`` prefix is applied server-side),
write ``HERMES_DASHBOARD_OAUTH_CLIENT_ID`` (+ portal/public URL when warranted) into ``.env``
idempotently, then print the gate-engagement hint.
"""

from __future__ import annotations

import json
import os
import random
import sys
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import urlparse

_DEFAULT_PORTAL = "https://portal.nousresearch.com"

# Docker-style adjective_noun names; the portal keys on row id, so collisions are harmless.
_NAME_ADJECTIVES = (
    "amber", "bold", "brave", "bright", "calm", "clever", "cosmic", "crisp",
    "dreamy", "eager", "electric", "fancy", "gentle", "golden", "happy",
    "hidden", "jolly", "keen", "lively", "lucid", "lunar", "mellow", "merry",
    "mighty", "nimble", "noble", "polished", "quiet", "quirky", "rapid",
    "serene", "sharp", "shiny", "silent", "snappy", "solar", "spry", "stellar",
    "sunny", "swift", "tidy", "vivid", "vibrant", "witty", "zesty")

_NAME_NOUNS = (
    "albatross", "antelope", "badger", "beacon", "comet", "condor", "cypress",
    "dolphin", "ember", "falcon", "ferret", "galaxy", "glacier", "harbor",
    "heron", "ibex", "jaguar", "kestrel", "lantern", "lynx", "meadow", "nebula",
    "ocelot", "orchid", "otter", "panther", "petrel", "quasar", "raven", "reef",
    "sparrow", "summit", "tundra", "vortex", "walrus", "willow", "yarrow",
    "kepler", "tesla", "curie", "hopper", "turing", "lovelace")


def _generate_dashboard_name() -> str:
    return f"{random.choice(_NAME_ADJECTIVES)}_{random.choice(_NAME_NOUNS)}"


def _resolve_portal_base_url(override: Optional[str] = None) -> str:
    """Portal base URL: explicit *override* (must be the token's issuer), then the login's stored
    ``portal_base_url``, then production."""
    if isinstance(override, str) and override.strip():
        return override.rstrip("/")
    try:
        from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL, get_provider_auth_state
        base = (get_provider_auth_state("nous") or {}).get("portal_base_url")
        chosen = base if isinstance(base, str) and base.strip() else str(DEFAULT_NOUS_PORTAL_URL)
        return chosen.rstrip("/")
    except Exception:
        return _DEFAULT_PORTAL


def _register_self_hosted_client(
    *, access_token: str, portal_base_url: str, name: Optional[str], custom_redirect_uri: Optional[str],
    existing_client_id: Optional[str] = None, timeout: float = 15.0) -> dict:
    """POST to the portal's self-hosted-client endpoint and return the JSON body.

    ``existing_client_id`` makes the portal update that record in place (idempotent re-runs;
    the portal mints a fresh client if the id no longer resolves, so passing it is always safe).
    ``name`` is ``None`` on the update path without ``--name`` (portal keeps the stored name).
    Raises RuntimeError with a user-facing message on non-2xx or transport failure.
    """
    fields = (("name", name), ("custom_redirect_uri", custom_redirect_uri),
              ("client_id", existing_client_id))
    req = urllib.request.Request(
        f"{portal_base_url.rstrip('/')}/api/oauth/self-hosted-client",
        data=json.dumps({k: v for k, v in fields if v}).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json",
                 "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        try:  # structured JSON errors: {error, error_description}
            err_body = json.loads(exc.read().decode())
            detail = err_body.get("error_description") or err_body.get("error") or ""
        except Exception:
            detail = ""
        if exc.code == 401:
            message = ("Nous Portal rejected the access token (401). "
                       "Try `hermes auth add nous` to re-authenticate.")
        elif exc.code == 403:
            message = detail or "Your account is not permitted to register a self-hosted dashboard."
        else:
            message = f"Portal returned HTTP {exc.code}" + (f": {detail}" if detail else "")
        raise RuntimeError(message) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Nous Portal at {portal_base_url}: {exc.reason}") from exc
    if not isinstance(payload, dict) or not payload.get("client_id"):
        raise RuntimeError("Portal returned an unexpected response (no client_id).")
    return payload


def _print_post_register_hint(
    *, client_id: str, portal_base_url: str, custom_redirect_uri: Optional[str],
    wrote_portal_url: bool, public_url: str = "") -> None:
    """Print the success summary + the gate-engagement caveat."""
    from hermes_cli.config import get_env_path
    print(f"\n  Wrote to {get_env_path()}:\n    HERMES_DASHBOARD_OAUTH_CLIENT_ID={client_id}")
    if wrote_portal_url:
        print("    HERMES_DASHBOARD_PORTAL_URL=" + str(portal_base_url))
    if public_url:
        print("    HERMES_DASHBOARD_PUBLIC_URL=" + str(public_url))
    print(
        "\n  Heads up — Nous login only *engages* on a non-loopback bind. A plain\n"
        "  `hermes dashboard` (localhost) leaves the gate off and serves locally\n"
        "  without auth, which is fine for your own machine.\n")
    if custom_redirect_uri:
        try:  # example host matches the one the user registered
            host = urlparse(custom_redirect_uri).hostname or "your-host"
        except Exception:
            host = "your-host"
        print(
            "  To require Nous login on your registered host, run the dashboard\n"
            f"  bound publicly (it must be reachable at https://{host}) and log in\n"
            "  at its /login page.")
    else:
        print(
            "  To require Nous login (e.g. exposing on your LAN or a public host):\n"
            "    hermes dashboard --host 0.0.0.0\n"
            "  …then log in at the dashboard's /login page.")
    print(
        "\n  If the dashboard is already running, restart it to pick up the new env.\n"
        f"  Manage or revoke this dashboard at {portal_base_url}/local-dashboards")


def _env_value(key: str) -> Optional[str]:
    """Stored ``.env`` value, or ``None`` on any read failure."""
    from hermes_cli.config import get_env_value
    try:
        return get_env_value(key)
    except Exception:
        return None


def _save_env_quietly(key: str, value: str) -> bool:
    """Persist *key*; False on failure (non-fatal: only client_id is load-bearing)."""
    from hermes_cli.config import save_env_value
    try:
        save_env_value(key, value)
        return True
    except Exception:
        return False


def _public_url_from_redirect(redirect_uri: Optional[str]) -> str:
    """Origin (``scheme://host[:port]``) of *redirect_uri*, or ``""`` — the runtime appends
    ``/auth/callback`` to HERMES_DASHBOARD_PUBLIC_URL, so the raw URI would double the path."""
    try:
        parsed = urlparse(redirect_uri or "")
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        pass
    return ""


def cmd_dashboard_register(args) -> None:
    """Register a self-hosted dashboard OAuth client with Nous Portal."""
    from hermes_cli.auth import AuthError, resolve_nous_access_token
    from hermes_cli.config import is_managed, save_env_value
    # Managed installs get the client id stamped in by the orchestrator (save_env_value refuses).
    if is_managed():
        print("✗ `hermes dashboard register` is not available in a managed/hosted install.\n"
              "  The dashboard OAuth client is provisioned by the hosting platform.")
        sys.exit(1)

    try:
        access_token = resolve_nous_access_token()
    except Exception as exc:
        if isinstance(exc, AuthError) and getattr(exc, "relogin_required", False):
            print("✗ You're not logged into Nous Portal.\n"
                  "  Run `hermes setup` (or `hermes auth add nous`) first, then retry.")
        else:
            print(f"✗ Could not resolve a Nous Portal access token: {exc}")
        sys.exit(1)
    # An explicitly supplied portal (flag or env) is persisted in place; an inferred one is
    # written only if absent so .env isn't cluttered for the common production case.
    portal_override = getattr(args, "portal_url", None) or os.environ.get("HERMES_DASHBOARD_PORTAL_URL")
    custom_portal_supplied = bool(isinstance(portal_override, str) and portal_override.strip())
    portal_base_url = _resolve_portal_base_url(portal_override)
    # Re-sending a locally held client_id makes the portal UPDATE that record (idempotent).
    stored = _env_value("HERMES_DASHBOARD_OAUTH_CLIENT_ID")
    existing_client_id = (stored.strip() or None) if isinstance(stored, str) else None
    # Auto-name ONLY a first registration; a re-run without --name keeps the stored name.
    name = getattr(args, "name", None) or (None if existing_client_id else _generate_dashboard_name())
    custom_redirect_uri = getattr(args, "redirect_uri", None)
    try:
        result = _register_self_hosted_client(
            access_token=access_token, portal_base_url=portal_base_url, name=name,
            custom_redirect_uri=custom_redirect_uri, existing_client_id=existing_client_id)
    except RuntimeError as exc:
        print(f"✗ Registration failed: {exc}")
        sys.exit(1)

    client_id = str(result["client_id"])
    registered_name = str(result.get("name") or name or "")
    # The portal echoes back the same client_id when it updated in place.
    verb = "Updated" if existing_client_id and client_id == existing_client_id else "Registered"
    print(f'✓ {verb} dashboard "{registered_name}"')
    try:  # client_id is load-bearing: fatal on failure
        save_env_value("HERMES_DASHBOARD_OAUTH_CLIENT_ID", client_id)
    except Exception as exc:
        print(f"✗ Failed to write HERMES_DASHBOARD_OAUTH_CLIENT_ID to .env: {exc}\n"
              f"  Set it manually:  HERMES_DASHBOARD_OAUTH_CLIENT_ID={client_id}")
        sys.exit(1)
    # Explicit portal → always persist (the user asked); inferred → only if unset AND non-default.
    existing_portal = _env_value("HERMES_DASHBOARD_PORTAL_URL")
    should_write_portal = (
        existing_portal != portal_base_url
        if custom_portal_supplied
        else not existing_portal and portal_base_url.rstrip("/") != _DEFAULT_PORTAL)
    wrote_portal_url = should_write_portal and _save_env_quietly("HERMES_DASHBOARD_PORTAL_URL", portal_base_url)
    # Public URL from --redirect-uri: written when supplied and different; never localhost-only.
    public_url = _public_url_from_redirect(custom_redirect_uri)
    wrote_public_url = bool(
        public_url
        and _env_value("HERMES_DASHBOARD_PUBLIC_URL") != public_url
        and _save_env_quietly("HERMES_DASHBOARD_PUBLIC_URL", public_url))
    _print_post_register_hint(
        client_id=client_id, portal_base_url=portal_base_url, custom_redirect_uri=custom_redirect_uri,
        wrote_portal_url=wrote_portal_url, public_url=public_url if wrote_public_url else "")
