"""``hermes dashboard register`` — register a self-hosted dashboard OAuth client.

Automates what a user otherwise does by hand: open the Nous Portal
``/local-dashboards`` page in a browser, click "register", copy the
resulting ``agent:{id}`` OAuth client ID, and paste it into ``~/.hermes/.env``
as ``HERMES_DASHBOARD_OAUTH_CLIENT_ID``.

This command:
  1. Resolves a fresh Nous Portal access token from the existing login
     (``~/.hermes/auth.json``), refreshing it if needed. Fails fast with a
     "run `hermes setup`" hint when the user isn't logged in.
  2. POSTs to ``{portal}/api/oauth/self-hosted-client`` with that bearer
     token, which creates a SELF_HOSTED agent client owned by the caller's
     org and returns the fully-formed ``agent:{id}`` client_id.
  3. Writes ``HERMES_DASHBOARD_OAUTH_CLIENT_ID`` and (if absent)
     ``HERMES_DASHBOARD_PORTAL_URL`` into ``~/.hermes/.env`` idempotently.
  4. Prints a post-register hint explaining that the OAuth gate only engages
     on a non-loopback bind.

The portal endpoint is the NAS half of this feature (POST
/api/oauth/self-hosted-client). The ``agent:`` prefix is applied server-side,
so this client never needs to know the namespace convention.
"""

from __future__ import annotations

import json
import os
import random
import sys
import urllib.error
import urllib.request
from typing import Optional


# Docker-style name generator. Same vibe as Docker's adjective_surname, but
# adjective_noun with a space-free underscore join so it drops cleanly into a
# label field. There is NO uniqueness constraint on the portal side (the row
# id is the key), so collisions are harmless and we don't retry.
_NAME_ADJECTIVES = (
    "amber", "bold", "brave", "bright", "calm", "clever", "cosmic", "crisp",
    "dreamy", "eager", "electric", "fancy", "gentle", "golden", "happy",
    "hidden", "jolly", "keen", "lively", "lucid", "lunar", "mellow", "merry",
    "mighty", "nimble", "noble", "polished", "quiet", "quirky", "rapid",
    "serene", "sharp", "shiny", "silent", "snappy", "solar", "spry", "stellar",
    "sunny", "swift", "tidy", "vivid", "vibrant", "witty", "zesty",
)

_NAME_NOUNS = (
    "albatross", "antelope", "badger", "beacon", "comet", "condor", "cypress",
    "dolphin", "ember", "falcon", "ferret", "galaxy", "glacier", "harbor",
    "heron", "ibex", "jaguar", "kestrel", "lantern", "lynx", "meadow", "nebula",
    "ocelot", "orchid", "otter", "panther", "petrel", "quasar", "raven", "reef",
    "sparrow", "summit", "tundra", "vortex", "walrus", "willow", "yarrow",
    # A couple of scientist surnames in the Docker spirit.
    "kepler", "tesla", "curie", "hopper", "turing", "lovelace",
)


def _generate_dashboard_name() -> str:
    """Return a human-readable ``adjective_noun`` name (Docker-style)."""
    return f"{random.choice(_NAME_ADJECTIVES)}_{random.choice(_NAME_NOUNS)}"


def _resolve_portal_base_url(override: Optional[str] = None) -> str:
    """Resolve the portal base URL for the registration request.

    Precedence:
      1. ``override`` — explicit ``--portal-url`` flag or
         ``HERMES_DASHBOARD_PORTAL_URL`` env (used for testing against a
         preview/staging portal). NOTE: the access token must be valid at
         this portal — it's minted by whatever portal you logged into, so an
         override only works if the token's issuer matches (e.g. you logged
         into the same staging/preview portal).
      2. The ``portal_base_url`` stored on the Nous login — this is the
         portal that issued the token, so it's the correct default target.
      3. The production default.
    """
    if isinstance(override, str) and override.strip():
        return override.rstrip("/")
    try:
        from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL, get_provider_auth_state

        state = get_provider_auth_state("nous") or {}
        base = state.get("portal_base_url")
        if isinstance(base, str) and base.strip():
            return base.rstrip("/")
        return str(DEFAULT_NOUS_PORTAL_URL).rstrip("/")
    except Exception:
        return "https://portal.nousresearch.com"


def _register_self_hosted_client(
    *,
    access_token: str,
    portal_base_url: str,
    name: Optional[str],
    custom_redirect_uri: Optional[str],
    existing_client_id: Optional[str] = None,
    timeout: float = 15.0,
) -> dict:
    """POST to the portal's self-hosted-client endpoint and return the JSON body.

    When ``existing_client_id`` is provided (the client_id this install
    persisted on a prior run), it is sent so the portal updates that existing
    dashboard record in place instead of minting a duplicate — this is what
    makes re-running ``hermes dashboard register`` idempotent. The portal
    falls back to creating a fresh client if the id no longer resolves to a row
    in the caller's org (stale/deleted), so passing it is always safe.

    ``name`` may be ``None`` on the idempotent update path (re-run without an
    explicit ``--name``): omitting it tells the portal to keep the name it
    already stored rather than overwriting it. It is required on the create
    path; the caller guarantees a value there.

    Raises RuntimeError with a user-facing message on any non-2xx response or
    transport failure.
    """
    url = f"{portal_base_url.rstrip('/')}/api/oauth/self-hosted-client"
    body: dict[str, str] = {}
    if name:
        body["name"] = name
    if custom_redirect_uri:
        body["custom_redirect_uri"] = custom_redirect_uri
    if existing_client_id:
        body["client_id"] = existing_client_id

    data = json.dumps(body).encode("utf-8")
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
        # The endpoint returns structured JSON errors ({error, error_description}).
        detail = ""
        try:
            err_body = json.loads(exc.read().decode())
            detail = (
                err_body.get("error_description")
                or err_body.get("error")
                or ""
            )
        except Exception:
            pass
        if exc.code == 401:
            raise RuntimeError(
                "Nous Portal rejected the access token (401). "
                "Try `hermes auth login nous` to re-authenticate."
            ) from exc
        if exc.code == 403:
            raise RuntimeError(
                detail
                or "Your account is not permitted to register a self-hosted dashboard."
            ) from exc
        raise RuntimeError(
            f"Portal returned HTTP {exc.code}"
            + (f": {detail}" if detail else "")
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Nous Portal at {portal_base_url}: {exc.reason}"
        ) from exc

    if not isinstance(payload, dict) or not payload.get("client_id"):
        raise RuntimeError("Portal returned an unexpected response (no client_id).")
    return payload


def _print_post_register_hint(
    *,
    client_id: str,
    portal_base_url: str,
    custom_redirect_uri: Optional[str],
    wrote_portal_url: bool,
) -> None:
    """Print the success summary + the gate-engagement caveat."""
    from hermes_cli.config import get_env_path

    env_path = get_env_path()
    print()
    print(f"  Wrote to {env_path}:")
    print(f"    HERMES_DASHBOARD_OAUTH_CLIENT_ID={client_id}")
    if wrote_portal_url:
        print(f"    HERMES_DASHBOARD_PORTAL_URL={portal_base_url}")
    print()
    print(
        "  Heads up — Nous login only *engages* on a non-loopback bind. A plain\n"
        "  `hermes dashboard` (localhost) leaves the gate off and serves locally\n"
        "  without auth, which is fine for your own machine."
    )
    print()
    if custom_redirect_uri:
        # Derive the host the user registered so the example matches it.
        try:
            from urllib.parse import urlparse

            host = urlparse(custom_redirect_uri).hostname or "your-host"
        except Exception:
            host = "your-host"
        print("  To require Nous login on your registered host, run the dashboard")
        print(f"  bound publicly (it must be reachable at https://{host}) and log in")
        print("  at its /login page.")
    else:
        print("  To require Nous login (e.g. exposing on your LAN or a public host):")
        print("    hermes dashboard --host 0.0.0.0")
        print("  …then log in at the dashboard's /login page.")
    print()
    print(
        "  If the dashboard is already running, restart it to pick up the new env."
    )
    print(
        f"  Manage or revoke this dashboard at {portal_base_url}/local-dashboards"
    )


def cmd_dashboard_register(args) -> None:
    """Register a self-hosted dashboard OAuth client with Nous Portal."""
    from hermes_cli.auth import AuthError, resolve_nous_access_token
    from hermes_cli.config import get_env_value, is_managed, save_env_value

    # Managed (Docker/hosted) installs get their dashboard OAuth client_id
    # stamped in by the orchestrator (NAS sets HERMES_DASHBOARD_OAUTH_CLIENT_ID
    # via buildContainerEnvVars). Registering from inside such a container is a
    # mistake — and save_env_value refuses to write anyway.
    if is_managed():
        print(
            "✗ `hermes dashboard register` is not available in a managed/hosted "
            "install.\n"
            "  The dashboard OAuth client is provisioned by the hosting platform."
        )
        sys.exit(1)

    # 1. Resolve a fresh Nous access token (refreshes if near expiry). Fail fast
    #    with a setup hint when the user isn't logged in.
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

    # Portal override: explicit --portal-url flag wins, else the
    # HERMES_DASHBOARD_PORTAL_URL env var, else the stored login's portal.
    portal_override = getattr(args, "portal_url", None) or os.environ.get(
        "HERMES_DASHBOARD_PORTAL_URL"
    )
    portal_base_url = _resolve_portal_base_url(portal_override)

    # Idempotency: if this install already registered a dashboard, we hold its
    # client_id locally (HERMES_DASHBOARD_OAUTH_CLIENT_ID). Re-send it so the
    # portal UPDATES that existing record instead of creating a duplicate. No
    # stored client_id -> this is a first registration -> create a fresh one
    # (the original behavior). This mirrors the portal's rule: no client id =
    # new dashboard; client id present = the stable key of the row to modify.
    existing_client_id = None
    try:
        existing_client_id = get_env_value("HERMES_DASHBOARD_OAUTH_CLIENT_ID")
    except Exception:
        existing_client_id = None
    if isinstance(existing_client_id, str):
        existing_client_id = existing_client_id.strip() or None
    else:
        existing_client_id = None

    explicit_name = getattr(args, "name", None)
    # Auto-generate a random name ONLY for a first registration. On a re-run
    # (we hold a client_id) without an explicit --name, keep the name the
    # portal already stored rather than churning it to a new random value
    # every time — so leave `name` unset and let the portal preserve it.
    if explicit_name:
        name = explicit_name
    elif existing_client_id:
        name = None
    else:
        name = _generate_dashboard_name()
    custom_redirect_uri = getattr(args, "redirect_uri", None)

    # 2. Register with the portal.
    try:
        result = _register_self_hosted_client(
            access_token=access_token,
            portal_base_url=portal_base_url,
            name=name,
            custom_redirect_uri=custom_redirect_uri,
            existing_client_id=existing_client_id,
        )
    except RuntimeError as exc:
        print(f"✗ Registration failed: {exc}")
        sys.exit(1)

    client_id = str(result["client_id"])
    registered_name = str(result.get("name") or name or "")

    # Distinguish create vs update for the user: the portal echoes back the
    # same client_id we sent when it updated in place.
    updated_existing = bool(
        existing_client_id and client_id == existing_client_id
    )
    if updated_existing:
        print(f'✓ Updated dashboard "{registered_name}"')
    else:
        print(f'✓ Registered dashboard "{registered_name}"')

    # 3. Write env vars idempotently. Always set the client_id. Only set the
    #    portal URL when it isn't already configured (env or config) AND differs
    #    from the production default, so we don't clutter .env for the common case
    #    but DO persist a non-default portal (e.g. a preview deploy used in dev).
    try:
        save_env_value("HERMES_DASHBOARD_OAUTH_CLIENT_ID", client_id)
    except Exception as exc:
        print(f"✗ Failed to write HERMES_DASHBOARD_OAUTH_CLIENT_ID to .env: {exc}")
        print(f"  Set it manually:  HERMES_DASHBOARD_OAUTH_CLIENT_ID={client_id}")
        sys.exit(1)

    wrote_portal_url = False
    default_portal = "https://portal.nousresearch.com"
    existing_portal = None
    try:
        existing_portal = get_env_value("HERMES_DASHBOARD_PORTAL_URL")
    except Exception:
        existing_portal = None
    if not existing_portal and portal_base_url.rstrip("/") != default_portal:
        try:
            save_env_value("HERMES_DASHBOARD_PORTAL_URL", portal_base_url)
            wrote_portal_url = True
        except Exception:
            # Non-fatal: the client_id is the load-bearing value.
            pass

    # 4. Hint.
    _print_post_register_hint(
        client_id=client_id,
        portal_base_url=portal_base_url,
        custom_redirect_uri=custom_redirect_uri,
        wrote_portal_url=wrote_portal_url,
    )
