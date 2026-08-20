"""hermes webhook — manage dynamic webhook subscriptions from the CLI.

Usage:
    hermes webhook subscribe <name> [options]
    hermes webhook list
    hermes webhook remove <name>
    hermes webhook test <name> [--payload '{"key": "value"}']

Subscriptions persist to the effective profile webhook route store and are
hot-reloaded by the webhook adapter without a gateway restart.
"""

import json
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Dict

from hermes_constants import display_hermes_home
from utils import atomic_replace


def _effective_webhook_config():
    """Return the unified runtime webhook configuration."""
    from gateway.webhook_config import resolve_effective_webhook_config

    return resolve_effective_webhook_config()


_SUBSCRIPTIONS_FILENAME = "webhook_subscriptions.json"
_SUBSCRIPTIONS_FILE_MODE = 0o600


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _subscriptions_path() -> Path:
    try:
        return _effective_webhook_config().routes_path
    except Exception:
        return _hermes_home() / _SUBSCRIPTIONS_FILENAME


def _load_subscriptions() -> Dict[str, dict]:
    path = _subscriptions_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_subscriptions(subs: Dict[str, dict]) -> None:
    path = _subscriptions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reference-only routes should not normally contain plaintext secrets, but
    # keep the route store private during incremental migration as well.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(subs, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_path, _SUBSCRIPTIONS_FILE_MODE)
        atomic_replace(tmp_path, path)
        os.chmod(path, _SUBSCRIPTIONS_FILE_MODE)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _store_route_secret(name: str, value: str) -> str:
    """Store a route secret through the Task 8 canonical persistence seam."""
    from hermes_cli.migrations.webhook_secret_refs import store_webhook_secret

    ref = "WEBHOOK_ROUTE_" + re.sub(r"[^A-Za-z0-9_]", "_", name.upper())
    store_webhook_secret(ref, value)
    return ref


def _resolve_route_secret(route: dict) -> str:
    """Resolve a route through the same helper used by migration/runtime."""
    ref = route.get("secret_ref")
    if not ref:
        return str(route.get("secret", "") or "")
    from hermes_cli.migrations.webhook_secret_refs import resolve_webhook_secret

    return str(resolve_webhook_secret(str(ref)) or "")


def _get_webhook_config() -> dict:
    """Return the legacy dict shape backed by effective webhook config."""
    try:
        effective = _effective_webhook_config()
        return {
            "enabled": effective.enabled,
            "extra": {
                "host": effective.host,
                "port": effective.port,
            },
        }
    except Exception:
        return {}


def _is_webhook_enabled() -> bool:
    try:
        return _effective_webhook_config().enabled
    except Exception:
        return bool(_get_webhook_config().get("enabled"))


def _get_webhook_base_url() -> str:
    wh = _get_webhook_config().get("extra", {})
    host = wh.get("host")
    port = wh.get("port", 8644)
    display_host = "localhost" if not host or host in {"0.0.0.0", "::"} else host
    if ":" in display_host and not display_host.startswith("["):
        display_host = f"[{display_host}]"
    return f"http://{display_host}:{port}"


def _setup_hint() -> str:
    _dhh = display_hermes_home()
    return f"""
  Webhook platform is not enabled. To set it up:

  1. Run the gateway setup wizard:
     hermes gateway setup

  2. Or manually add to {_dhh}/config.yaml:
     platforms:
       webhook:
         enabled: true
         extra:
           port: 8644
           secret_ref: WEBHOOK_SECRET

  3. Or configure the profile secret backend with WEBHOOK_SECRET.

  Then start the gateway: hermes gateway run
"""


def _require_webhook_enabled() -> bool:
    if _is_webhook_enabled():
        return True
    print(_setup_hint())
    return False


def webhook_command(args):
    """Entry point for 'hermes webhook' subcommand."""
    sub = getattr(args, "webhook_action", None)

    if not sub:
        print("Usage: hermes webhook {subscribe|list|remove|test|migrate-secrets}")
        print("Run 'hermes webhook --help' for details.")
        return

    # Migration must remain available when a broken legacy route prevents the
    # runtime platform from becoming enabled.
    if sub == "migrate-secrets":
        _cmd_migrate_secrets(args)
        return

    if not _require_webhook_enabled():
        return

    if sub in {"subscribe", "add"}:
        _cmd_subscribe(args)
    elif sub in {"list", "ls"}:
        _cmd_list(args)
    elif sub in {"remove", "rm"}:
        _cmd_remove(args)
    elif sub == "test":
        _cmd_test(args)


def _cmd_subscribe(args):
    name = args.name.strip().lower().replace(" ", "-")
    if not re.match(r'^[a-z0-9][a-z0-9_-]*$', name):
        print(f"Error: Invalid name '{name}'. Use lowercase alphanumeric with hyphens/underscores.")
        return

    subs = _load_subscriptions()
    is_update = name in subs
    existing_route = subs.get(name) if is_update else None
    supplied_secret = bool(args.secret)
    secret = args.secret or ("" if is_update else secrets.token_urlsafe(32))
    events = [e.strip() for e in args.events.split(",")] if args.events else []

    secret_ref = None
    if is_update and not supplied_secret and isinstance(existing_route, dict):
        secret_ref = existing_route.get("secret_ref")
        if not secret_ref:
            secret = str(existing_route.get("secret", "") or "")
            if not secret:
                # A previously malformed/no-secret route must never be saved
                # back into an unusable state. Mint a fresh credential and
                # surface it once just like a new subscription.
                secret = secrets.token_urlsafe(32)
                supplied_secret = True
            secret_ref = _store_route_secret(name, secret)
    else:
        secret_ref = _store_route_secret(name, secret)

    route = {
        "description": args.description or f"Agent-created subscription: {name}",
        "events": events,
        "prompt": args.prompt or "",
        "skills": [s.strip() for s in args.skills.split(",")] if args.skills else [],
        "deliver": args.deliver or "log",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if secret_ref:
        route["secret_ref"] = secret_ref
    else:
        # Fail closed rather than persisting a route that will die at startup.
        print("Error: webhook secret persistence did not return a reference")
        return

    if getattr(args, "deliver_only", False):
        if route["deliver"] == "log":
            print(
                "Error: --deliver-only requires --deliver to be a real target "
                "(telegram, discord, slack, github_comment, etc.) — not 'log'."
            )
            return
        route["deliver_only"] = True

    script = getattr(args, "script", "") or ""
    if script.strip():
        route["script"] = script.strip()

    if args.deliver_chat_id:
        route["deliver_extra"] = {"chat_id": args.deliver_chat_id}

    subs[name] = route
    _save_subscriptions(subs)

    base_url = _get_webhook_base_url()
    status = "Updated" if is_update else "Created"

    print(f"\n  {status} webhook subscription: {name}")
    print(f"  URL:    {base_url}/webhooks/{name}")
    if not is_update or supplied_secret:
        print(f"  Secret: {secret}")
    else:
        print("  Secret: (unchanged; not displayed)")
    if events:
        print(f"  Events: {', '.join(events)}")
    else:
        print("  Events: (all)")
    print(f"  Deliver: {route['deliver']}")
    if route.get("deliver_only"):
        print("  Mode: direct delivery (no agent, zero LLM cost)")
    if route.get("prompt"):
        prompt_preview = route["prompt"][:80] + ("..." if len(route["prompt"]) > 80 else "")
        label = "Message" if route.get("deliver_only") else "Prompt"
        print(f"  {label}: {prompt_preview}")
    if route.get("script"):
        print(f"  Script: {route['script']}")
    print("\n  Configure your service to POST to the URL above.")
    print("  Use the secret for HMAC-SHA256 signature validation.")
    print("  The gateway must be running to receive events (hermes gateway run).\n")


def _cmd_list(args):
    subs = _load_subscriptions()
    if not subs:
        print("  No dynamic webhook subscriptions.")
        print("  Create one with: hermes webhook subscribe <name>")
        return

    base_url = _get_webhook_base_url()
    print(f"\n  {len(subs)} webhook subscription(s):\n")
    for name, route in subs.items():
        events = ", ".join(route.get("events", [])) or "(all)"
        deliver = route.get("deliver", "log")
        if route.get("deliver_only"):
            deliver = f"{deliver} (direct — no agent)"
        desc = route.get("description", "")
        print(f"  ◆ {name}")
        if desc:
            print(f"    {desc}")
        print(f"    URL:     {base_url}/webhooks/{name}")
        print(f"    Events:  {events}")
        print(f"    Deliver: {deliver}")
        if route.get("script"):
            print(f"    Script:  {route['script']}")
        print()


def _cmd_remove(args):
    name = args.name.strip().lower()
    subs = _load_subscriptions()

    if name not in subs:
        print(f"  No subscription named '{name}'.")
        print("  Note: Static routes from config.yaml cannot be removed here.")
        return

    del subs[name]
    _save_subscriptions(subs)
    print(f"  Removed webhook subscription: {name}")


def _cmd_test(args):
    """Send a test POST to a webhook route."""
    name = args.name.strip().lower()
    subs = _load_subscriptions()

    if name not in subs:
        print(f"  No subscription named '{name}'.")
        return

    route = subs[name]
    secret = _resolve_route_secret(route)
    if not secret:
        print("  Error: webhook secret reference could not be resolved")
        return
    base_url = _get_webhook_base_url()
    url = f"{base_url}/webhooks/{name}"
    payload = args.payload or '{"test": true, "event_type": "test", "message": "Hello from hermes webhook test"}'

    import hmac
    import hashlib
    sig = "sha256=" + hmac.new(
        secret.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()

    print(f"  Sending test POST to {url}")
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            data=payload.encode(),
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
                "X-GitHub-Event": "test",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            print(f"  Response ({resp.status}): {body}")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Is the gateway running? (hermes gateway run)")


def _cmd_migrate_secrets(args):
    """Migrate legacy webhook secrets, returning value-free receipts."""
    from hermes_cli.migrations.webhook_secret_refs import (
        migrate_webhook_config,
        migrate_webhook_routes,
    )

    route_path = _subscriptions_path()
    route_result = {
        "migrated_routes": [],
        "receipts": [],
        "scrubbed_backups": [],
    }
    if route_path.exists():
        backups = tuple(route_path.parent.glob(route_path.name + ".bak*"))
        route_result = migrate_webhook_routes(route_path, backup_paths=backups)

    config_path = _hermes_home() / "config.yaml"
    config_result = {"migrated": False, "receipts": []}
    if config_path.exists():
        config_result = migrate_webhook_config(config_path)

    result = {"routes": route_result, "config": config_result}
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "Webhook secret migration complete: "
            f"{len(route_result.get('migrated_routes', []))} route(s), "
            f"config={'migrated' if config_result.get('migrated') else 'unchanged'}."
        )
    return result
