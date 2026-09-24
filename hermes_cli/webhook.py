"""hermes webhook — manage dynamic webhook subscriptions from the CLI."""

import errno
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import tempfile
import time
import urllib.request
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

from hermes_constants import display_hermes_home
from hermes_cli.config import cfg_get


_SUBSCRIPTIONS_FILENAME = "webhook_subscriptions.json"
_SUBSCRIPTIONS_FILE_MODE = 0o600


class PublishedButNotDurable(OSError):
    """The registry rename succeeded, but directory durability was not confirmed."""

    published = True


# Replacement routes keep plugin/custom metadata, but omitted optional fields
# from the old command/form must not remain active after an update.
_ROUTE_FORM_FIELDS = frozenset({
    "description", "events", "prompt", "skills", "deliver", "created_at",
    "secret", "profile", "deliver_only", "mirror_to_session", "cron_job",
    "script", "deliver_extra",
})


def _subscriptions_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / _SUBSCRIPTIONS_FILENAME


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
    _replace_registry(_subscriptions_path(), subs)


def _replace_registry(path: Path, subs: Dict[str, dict]) -> None:
    """Publish a fully synced private registry, never using the shared copy fallback.

    EBUSY/EXDEV on a bind-mounted registry cannot safely fall back to an
    in-place rewrite: the plugin and gateway can observe a partial JSON file.
    Transactions hold the persistent sibling lock through this operation.
    """
    from hermes_constants import mkdir_under_hermes_home
    mkdir_under_hermes_home(path.parent)
    payload = json.dumps(subs, indent=2, ensure_ascii=True).encode("utf-8")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            if hasattr(os, "fchmod"):
                os.fchmod(stream.fileno(), _SUBSCRIPTIONS_FILE_MODE)
            else:
                os.chmod(tmp, _SUBSCRIPTIONS_FILE_MODE)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # No fallback: only a successful rename can publish these secrets.
        os.replace(tmp, path)
        # The temp starts 0600 even under a permissive umask; its final
        # private permissions are set before publication on every platform.
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def _existing_route(subs: dict, name: str) -> dict:
    """A writer must not silently replace a damaged route it is targeting."""
    route = subs[name]
    if not isinstance(route, dict):
        raise ValueError(f"Webhook subscription '{name}' is not an object.")
    for field in ("events", "skills"):
        value = route.get(field)
        if value is not None and (not isinstance(value, list)
                                  or any(not isinstance(item, str) for item in value)):
            raise ValueError(f"Webhook subscription '{name}' has invalid {field}.")
    return route


def _read_subscriptions_strict() -> Dict[str, dict]:
    """Administrative reads must not turn corruption into an empty registry."""
    try:
        data = json.loads(_subscriptions_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("Webhook subscriptions registry is invalid JSON.") from exc
    if not isinstance(data, dict):
        raise ValueError("Webhook subscriptions registry is not an object.")
    return data

def _replace_route(existing: dict, route: dict) -> dict:
    return {**{key: value for key, value in existing.items()
               if key not in _ROUTE_FORM_FIELDS}, **route}


def _sync_registry_directory(path: Path) -> None:
    """Surface POSIX parent-directory fsync errors to the transaction writer.

    The shared atomic_json_write(fsync_dir=True) helper is best-effort.
    Windows has no equivalent directory open here; its file fsync still applies.
    """
    if os.name == "nt":
        return
    fd = os.open(path.resolve(strict=False).parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _subscription_transaction():
    """Serialize the entire mutation with plugin writers on a persistent sibling inode.

    Readers retain their permissive legacy loader; writers must never interpret a
    damaged or unreadable registry as empty and then overwrite it.
    """
    path = _subscriptions_path()
    from hermes_constants import mkdir_under_hermes_home
    mkdir_under_hermes_home(path.parent)
    lock_path = path.with_name(path.name + ".lock")
    # The POSIX plugin uses this persistent sibling inode. Windows locks only
    # coordinate native writers; the plugin's fcntl lock is POSIX-only.
    # Never follow a planted symlink or chmod a multiply-linked regular file.
    if not hasattr(os, "O_NOFOLLOW") and lock_path.is_symlink():
        raise ValueError("Webhook subscriptions lock must not be a symlink.")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        lock_stat = os.fstat(fd)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise ValueError("Webhook subscriptions lock must be a singly-linked regular file.")
        if os.name == "nt":
            import msvcrt
            # LK_LOCK retries only ten times. Retry the nonblocking primitive
            # until ownership is available, matching POSIX flock's blocking
            # behavior without imposing an arbitrary writer timeout.
            while True:
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN) and getattr(exc, "winerror", None) not in (33, 36):
                        raise
                    time.sleep(0.05)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.fchmod(fd, 0o600)
        try:
            data = _read_subscriptions_strict()
            original = deepcopy(data)
            yield data
            if data != original:
                _replace_registry(path, data)
                try:
                    _sync_registry_directory(path)
                except OSError as exc:
                    raise PublishedButNotDurable(
                        f"Webhook registry was published, but directory sync failed: {exc}"
                    ) from exc
        finally:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _get_webhook_config() -> dict:
    """Load webhook platform config. Returns {} if not configured."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg_get(cfg, "platforms", "webhook", default={})
    except Exception:
        return {}


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
        return 1
    handler = _ACTIONS.get(sub)
    if handler is not None:
        return handler(args)


def _cmd_subscribe(args):
    name = args.name.strip().lower().replace(" ", "-")
    if not re.match(r'^[a-z0-9][a-z0-9_-]*$', name):
        print(f"Error: Invalid name '{name}'. Use lowercase alphanumeric with hyphens/underscores.")
        return 1

    profile_arg = getattr(args, "route_profile", None)
    profile = "default"
    if profile_arg is not None:
        from hermes_cli.profiles import normalize_profile_name, profile_exists, validate_profile_name
        try:
            profile = normalize_profile_name(profile_arg)
            validate_profile_name(profile)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1
        if not profile_exists(profile):
            print(f"Error: Profile '{profile}' does not exist.")
            return 1
    events = [e.strip() for e in args.events.split(",")] if args.events else []
    route = {
        "description": args.description or f"Agent-created subscription: {name}",
        "events": events,
        "prompt": args.prompt or "",
        "skills": [s.strip() for s in args.skills.split(",")] if args.skills else [],
        "deliver": args.deliver or "log",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    if getattr(args, "deliver_only", False):
        if getattr(args, "cron_job", ""):
            print(
                "Error: --deliver-only and --cron-job are mutually exclusive. "
                "--deliver-only pushes the rendered template as a message; "
                "--cron-job fires an existing cron job (which handles its own "
                "delivery)."
            )
            return 1
        if route["deliver"] == "log":
            print(
                "Error: --deliver-only requires --deliver to be a real target "
                "(telegram, discord, slack, github_comment, etc.) — not 'log'.")
            return 1
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
            return 1
        if job is None:
            print(f"Error: no cron job matches '{cron_job}'. List jobs with: hermes cron list")
            return 1
        route["cron_job"] = job["id"]
    script = (getattr(args, "script", "") or "").strip()
    if script:
        route["script"] = script
    if args.deliver_chat_id:
        route["deliver_extra"] = {"chat_id": args.deliver_chat_id}
    is_update = False
    secret = ""
    try:
        with _subscription_transaction() as subs:
            is_update = name in subs
            existing = _existing_route(subs, name) if is_update else {}
            profile = existing.get("profile", "default") if profile_arg is None else profile
            secret = args.secret or existing.get("secret") or secrets.token_urlsafe(32)
            route["profile"] = profile
            route["secret"] = secret
            subs[name] = _replace_route(existing, route)
    except PublishedButNotDurable:
        durable = False
    except (ValueError, OSError) as exc:
        print(f"Error: Could not update webhook subscriptions: {exc}")
        return 1
    else:
        durable = True

    print(f"\n  {'Updated' if is_update else 'Created'} webhook subscription: {name}")
    print(f"  URL:    {_route_url(name, route)}")
    print(f"  Profile: {profile}")
    print(f"  Secret: {secret}")
    if not durable:
        print("  WARNING: Route was published, but directory sync failed; crash durability is unconfirmed.")
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
    try:
        subs = _read_subscriptions_strict()
        for name in subs:
            _existing_route(subs, name)
    except (ValueError, OSError) as exc:
        print(f"Error: Could not read webhook subscriptions: {exc}")
        return 1
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
    try:
        with _subscription_transaction() as subs:
            if name not in subs:
                print(f"  No subscription named '{name}'.")
                print("  Note: Static routes from config.yaml cannot be removed here.")
                return
            _existing_route(subs, name)
            del subs[name]
    except PublishedButNotDurable:
        print(f"  Removed webhook subscription: {name}")
        print("  WARNING: Removal was published, but directory sync failed; crash durability is unconfirmed.")
        return 0
    except (ValueError, OSError) as exc:
        print(f"Error: Could not update webhook subscriptions: {exc}")
        return 1
    print(f"  Removed webhook subscription: {name}")


def _cmd_test(args):
    """Send a test POST to a webhook route."""
    name = args.name.strip().lower()
    try:
        subs = _read_subscriptions_strict()
    except (ValueError, OSError) as exc:
        print(f"Error: Could not read webhook subscriptions: {exc}")
        return 1
    if name not in subs:
        print(f"  No subscription named '{name}'.")
        return 1
    try:
        route = _existing_route(subs, name)
        secret = route.get("secret", "")
        if not isinstance(secret, str):
            raise ValueError(f"Webhook subscription '{name}' has an invalid secret.")
        url = _route_url(name, route)
    except ValueError as exc:
        print(f"Error: Could not read webhook subscriptions: {exc}")
        return 1
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
        return 1


_ACTIONS = {
    "subscribe": _cmd_subscribe, "add": _cmd_subscribe,
    "list": _cmd_list, "ls": _cmd_list,
    "remove": _cmd_remove, "rm": _cmd_remove,
    "test": _cmd_test}


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
import tempfile  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'atomic_replace': ('utils', 'atomic_replace'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
