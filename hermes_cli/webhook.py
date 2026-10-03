"""hermes webhook — manage dynamic webhook subscriptions from the CLI."""

import hashlib
import hmac
import json
import re
import secrets
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict

from hermes_constants import display_hermes_home
from utils import atomic_json_write


_SUBSCRIPTIONS_FILENAME = "webhook_subscriptions.json"
_SUBSCRIPTIONS_FILE_MODE = 0o600
_SUBSCRIPTIONS_LOCK_FILENAME = ".webhook_subscriptions.lock"
_SUBSCRIPTIONS_THREAD_LOCK = threading.Lock()
_MISSING_SUBSCRIPTION = object()


class SubscriptionMutationConflict(RuntimeError):
    """The route changed after a caller took the snapshot it intended to replace."""


class WebhookSubscriptionsError(ValueError):
    """The subscriptions store exists but cannot be read as a JSON object."""


def _subscriptions_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / _SUBSCRIPTIONS_FILENAME


def _subscriptions_lock_path() -> Path:
    return _subscriptions_path().with_name(_SUBSCRIPTIONS_LOCK_FILENAME)


@contextmanager
def _subscriptions_lock():
    """Serialize subscription snapshots and mutations across threads and processes."""
    from hermes_cli.active_sessions import _FileLock

    with _SUBSCRIPTIONS_THREAD_LOCK:
        with _FileLock(_subscriptions_lock_path()):
            yield


def _load_subscriptions_unlocked() -> Dict[str, dict]:
    """Read one complete store snapshot; only a MISSING store reads as empty.

    An unreadable or malformed one raises: read as {}, the next create would publish a store
    holding only its own route — every other route and its HMAC secret gone (the docs send
    users into this file to hand-edit ``toolsets``)."""
    path = _subscriptions_path()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise WebhookSubscriptionsError(
            f"{display_hermes_home()}/{_SUBSCRIPTIONS_FILENAME} is not readable ({exc}); "
            "fix it or move it aside — refusing to overwrite it") from exc
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except ValueError as exc:  # JSONDecodeError / UnicodeDecodeError are ValueErrors
        data = exc
    if not isinstance(data, dict):
        reason = data if isinstance(data, Exception) else f"top level is {type(data).__name__}, not an object"
        raise WebhookSubscriptionsError(
            f"{display_hermes_home()}/{_SUBSCRIPTIONS_FILENAME} is not a valid subscriptions file "
            f"({reason}); fix it or move it aside — refusing to overwrite it")
    return data


def _load_subscriptions() -> Dict[str, dict]:
    """Lock-free read: writers publish by atomic rename, so a reader always sees one whole
    snapshot, and read-only homes (no lock file can be created) keep working."""
    return _load_subscriptions_unlocked()


def _save_subscriptions_unlocked(subs: Dict[str, dict]) -> None:
    # The file holds per-route HMAC secrets: atomic_json_write fchmods the temp file 0o600 BEFORE the
    # rename (no umask window) and re-asserts the mode on the destination afterwards.
    atomic_json_write(_subscriptions_path(), subs, mode=_SUBSCRIPTIONS_FILE_MODE)


def _mutate_subscriptions(mutate: Callable[[Dict[str, dict]], Any]) -> Any:
    """Lock, re-read, mutate and atomically publish one complete store snapshot."""
    with _subscriptions_lock():
        subscriptions = _load_subscriptions_unlocked()
        result = mutate(subscriptions)
        _save_subscriptions_unlocked(subscriptions)
        return result


def _replace_subscription(name: str, route: dict, expected: object) -> dict:
    """CAS one route so a stale update cannot recreate a removed or disabled record.

    A general update is not an enable operation: an explicit ``enabled: False`` on the
    replaced record survives (only the dashboard's dedicated enabled endpoint lifts it).
    Returns the route as published.
    """
    def replace(subscriptions: Dict[str, dict]) -> dict:
        current = subscriptions.get(name, _MISSING_SUBSCRIPTION)
        expected_missing = expected is _MISSING_SUBSCRIPTION
        if (current is _MISSING_SUBSCRIPTION) != expected_missing or (
            not expected_missing and current != expected
        ):
            raise SubscriptionMutationConflict(
                f"Webhook subscription '{name}' changed concurrently; retry the operation."
            )
        published = route
        if isinstance(current, dict) and current.get("enabled") is False:
            published = {**route, "enabled": False}
        subscriptions[name] = published
        return published

    return _mutate_subscriptions(replace)


def _get_webhook_config() -> dict:
    """The webhook platform as the gateway resolves it (``{"enabled", "extra"}``), else {}.

    Resolved through ``load_gateway_config`` — not config.yaml alone — so WEBHOOK_ENABLED /
    WEBHOOK_PORT (what ``hermes gateway setup`` and the dashboard write to .env) enable the CLI
    and set its URLs exactly as they enable and bind the adapter; an explicit yaml ``enabled:
    false`` still wins, as it does for the gateway."""
    try:
        from gateway.config import Platform, load_gateway_config
        pc = load_gateway_config().platforms.get(Platform.WEBHOOK)
    except Exception:
        return {}
    return {"enabled": bool(pc.enabled), "extra": dict(pc.extra or {})} if pc else {}


def _is_webhook_enabled() -> bool:
    return bool(_get_webhook_config().get("enabled"))


def _get_webhook_base_url() -> str:
    wh = _get_webhook_config().get("extra", {})
    host = wh.get("host")
    display_host = "localhost" if not host or host in {"0.0.0.0", "::"} else host
    if ":" in display_host and not display_host.startswith("["):
        display_host = f"[{display_host}]"
    return f"http://{display_host}:{wh.get('port', 8644)}"


def _route_url(name: str, route: dict) -> str:
    profile = route.get("profile", "default")
    prefix = f"/p/{profile}" if profile != "default" else ""
    return f"{_get_webhook_base_url()}{prefix}/webhooks/{name}"


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
           secret: "your-global-hmac-secret"

  3. Or set environment variables in {_dhh}/.env:
     WEBHOOK_ENABLED=true
     WEBHOOK_PORT=8644
     WEBHOOK_SECRET=your-global-secret

  Then start the gateway: hermes gateway run
"""


def webhook_command(args):
    """Entry point for 'hermes webhook' subcommand."""
    sub = getattr(args, "webhook_action", None)
    if not sub:
        print("Usage: hermes webhook {subscribe|list|remove|test}")
        print("Run 'hermes webhook --help' for details.")
        return
    if not _is_webhook_enabled():
        print(_setup_hint())
        return
    handler = _ACTIONS.get(sub)
    if handler is not None:
        try:
            handler(args)
        except WebhookSubscriptionsError as exc:
            print(f"Error: {exc}")
            sys.exit(1)


def _cmd_subscribe(args):
    name = args.name.strip().lower().replace(" ", "-")
    if not re.match(r'^[a-z0-9][a-z0-9_-]*$', name):
        print(f"Error: Invalid name '{name}'. Use lowercase alphanumeric with hyphens/underscores.")
        return

    subs = _load_subscriptions()
    expected = subs.get(name, _MISSING_SUBSCRIPTION)
    is_update = expected is not _MISSING_SUBSCRIPTION
    existing = expected if isinstance(expected, dict) else {}
    profile_arg = getattr(args, "route_profile", None)
    if profile_arg is None:
        profile = existing.get("profile", "default")
    else:
        from hermes_cli.profiles import normalize_profile_name, profile_exists, validate_profile_name
        try:
            profile = normalize_profile_name(profile_arg)
            validate_profile_name(profile)
        except ValueError as exc:
            print(f"Error: {exc}")
            return
        if not profile_exists(profile):
            print(f"Error: Profile '{profile}' does not exist.")
            return
    existing_profile = existing.get("profile", "default")
    profile_changed = is_update and profile != existing_profile
    if profile_changed and args.secret and args.secret == existing.get("secret"):
        print(
            "Error: Rebinding a webhook route requires a new HMAC secret. "
            "Omit --secret to generate one automatically, or provide a different secret."
        )
        return
    secret = args.secret or (None if profile_changed else existing.get("secret")) or secrets.token_urlsafe(32)
    events = [e.strip() for e in args.events.split(",")] if args.events else []
    route = {
        "description": args.description or f"Agent-created subscription: {name}",
        "events": events,
        "secret": secret,
        "prompt": args.prompt or "",
        "skills": [s.strip() for s in args.skills.split(",")] if args.skills else [],
        "deliver": args.deliver or "log",
        "profile": profile,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if getattr(args, "deliver_only", False):
        if getattr(args, "cron_job", ""):
            print(
                "Error: --deliver-only and --cron-job are mutually exclusive. "
                "--deliver-only pushes the rendered template as a message; "
                "--cron-job fires an existing cron job (which handles its own "
                "delivery)."
            )
            return
        if route["deliver"] == "log":
            print(
                "Error: --deliver-only requires --deliver to be a real target "
                "(telegram, discord, slack, github_comment, etc.) — not 'log'.")
            return
        route["deliver_only"] = True
    if getattr(args, "mirror_to_session", False):
        route["mirror_to_session"] = True
    cron_job = (getattr(args, "cron_job", "") or "").strip()
    if cron_job:
        # Validate the reference up-front so a typo surfaces here, not on the first inbound event.
        from cron.jobs import AmbiguousJobReference, resolve_job_ref
        try:
            job = resolve_job_ref(cron_job)
        except AmbiguousJobReference as e:
            print(f"Error: {e}")
            return
        if job is None:
            print(f"Error: no cron job matches '{cron_job}'. List jobs with: hermes cron list")
            return
        route["cron_job"] = job["id"]
    script = (getattr(args, "script", "") or "").strip()
    if script:
        route["script"] = script
    if args.deliver_chat_id:
        route["deliver_extra"] = {"chat_id": args.deliver_chat_id}
    try:
        route = _replace_subscription(name, route, expected)
    except SubscriptionMutationConflict as exc:
        print(f"Error: {exc}")
        return

    print(f"\n  {'Updated' if is_update else 'Created'} webhook subscription: {name}")
    print(f"  URL:    {_route_url(name, route)}")
    print(f"  Profile: {profile}")
    print(f"  Secret: {secret}")
    print(f"  Events: {', '.join(events) or '(all)'}")
    print(f"  Deliver: {route['deliver']}")
    if route.get("deliver_only"):
        print("  Mode: direct delivery (no agent, zero LLM cost)")
    if route.get("mirror_to_session"):
        print("  Replies: each delivery is mirrored into the target chat's session")
    if route.get("cron_job"):
        print(f"  Mode: cron-job trigger — fires job '{route['cron_job']}' on each event")
    if route.get("prompt"):
        prompt_preview = route["prompt"][:80] + ("..." if len(route["prompt"]) > 80 else "")
        print(f"  {'Message' if route.get('deliver_only') else 'Prompt'}: {prompt_preview}")
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

    print(f"\n  {len(subs)} webhook subscription(s):\n")
    for name, route in subs.items():
        events = ", ".join(route.get("events", [])) or "(all)"
        deliver = route.get("deliver", "log")
        if route.get("deliver_only"):
            deliver = f"{deliver} (direct — no agent)"
        if route.get("cron_job"):
            deliver = f"cron job '{route['cron_job']}'"
        desc = route.get("description", "")
        print(f"  ◆ {name}")
        if desc:
            print(f"    {desc}")
        profile = route.get("profile", "default")
        print(f"    URL:     {_route_url(name, route)}")
        print(f"    Profile: {profile}")
        print(f"    Events:  {events}")
        print(f"    Deliver: {deliver}")
        if route.get("script"):
            print(f"    Script:  {route['script']}")
        print()


def _cmd_remove(args):
    name = args.name.strip().lower()

    def remove(subscriptions: Dict[str, dict]) -> None:
        if name not in subscriptions:
            raise KeyError(name)
        del subscriptions[name]

    try:
        _mutate_subscriptions(remove)
    except KeyError:
        print(f"  No subscription named '{name}'.")
        print("  Note: Static routes from config.yaml cannot be removed here.")
        return
    print(f"  Removed webhook subscription: {name}")


def _cmd_test(args):
    """Send a test POST to a webhook route."""
    name = args.name.strip().lower()
    subs = _load_subscriptions()
    if name not in subs:
        print(f"  No subscription named '{name}'.")
        return
    secret = subs[name].get("secret", "")
    url = _route_url(name, subs[name])
    payload = args.payload or '{"test": true, "event_type": "test", "message": "Hello from hermes webhook test"}'
    sig = "sha256=" + hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    print(f"  Sending test POST to {url}")
    try:
        req = urllib.request.Request(
            url,
            data=payload.encode(),
            headers={"Content-Type": "application/json", "X-Hub-Signature-256": sig, "X-GitHub-Event": "test"},
            method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            print(f"  Response ({resp.status}): {body}")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Is the gateway running? (hermes gateway run)")


_ACTIONS = {
    "subscribe": _cmd_subscribe, "add": _cmd_subscribe,
    "list": _cmd_list, "ls": _cmd_list,
    "remove": _cmd_remove, "rm": _cmd_remove,
    "test": _cmd_test}
