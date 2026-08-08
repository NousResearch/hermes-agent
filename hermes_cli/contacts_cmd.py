"""Profile-scoped contact and outbound-route registry CLI.

This module deliberately resolves routes without sending.  Reachability,
authorization, and a user's route preference are separate concerns; callers
must perform an authority check and the actual send through the existing
messaging surface.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from hermes_constants import get_hermes_home

SCHEMA_VERSION = 1
_ROUTE_STATES = {"verified", "unverified", "stale"}
_DIRECTORY_MAX_AGE_SECONDS = 10 * 60
_NON_DIRECTORY_PLATFORMS = {"email"}


class ContactRegistryError(ValueError):
    """Raised when a contact registry does not satisfy the v1 contract."""


def registry_path() -> Path:
    """Return the active profile's contact-registry path."""
    return get_hermes_home() / "contacts.yaml"


def directory_path() -> Path:
    """Return the active profile's generated channel-directory path."""
    return get_hermes_home() / "channel_directory.json"


def _norm(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value)).casefold().strip()


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContactRegistryError(f"{field} must be a list of strings")
    return value


def _contact_names(contact: dict[str, Any]) -> Iterable[str]:
    yield str(contact.get("id", ""))
    yield str(contact.get("display_name", ""))
    yield from _string_list(contact.get("aliases"), "contact.aliases")


def validate_registry(data: Any) -> dict[str, Any]:
    """Validate and return a v1 registry.

    Validation intentionally permits extra fields so people can keep provenance,
    notes, identity evidence, and platform-specific constraints without forcing a
    migration for every extension.
    """
    if not isinstance(data, dict):
        raise ContactRegistryError("registry root must be a mapping")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ContactRegistryError(
            f"schema_version must be {SCHEMA_VERSION}; got {data.get('schema_version')!r}"
        )
    contacts = data.get("contacts")
    if not isinstance(contacts, list):
        raise ContactRegistryError("contacts must be a list")

    contact_ids: set[str] = set()
    name_owners: dict[str, str] = {}
    for index, contact in enumerate(contacts):
        prefix = f"contacts[{index}]"
        if not isinstance(contact, dict):
            raise ContactRegistryError(f"{prefix} must be a mapping")
        contact_id = contact.get("id")
        display_name = contact.get("display_name")
        if not isinstance(contact_id, str) or not contact_id.strip():
            raise ContactRegistryError(f"{prefix}.id must be a non-empty string")
        if not isinstance(display_name, str) or not display_name.strip():
            raise ContactRegistryError(f"{prefix}.display_name must be a non-empty string")
        if contact_id in contact_ids:
            raise ContactRegistryError(f"duplicate contact id: {contact_id}")
        contact_ids.add(contact_id)

        for name in _contact_names(contact):
            normalized = _norm(name)
            if not normalized:
                raise ContactRegistryError(f"{prefix} contains an empty name or alias")
            owner = name_owners.get(normalized)
            if owner and owner != contact_id:
                raise ContactRegistryError(
                    f"ambiguous contact name or alias {name!r}: {owner} and {contact_id}"
                )
            name_owners[normalized] = contact_id

        routes = contact.get("routes", [])
        if not isinstance(routes, list):
            raise ContactRegistryError(f"{prefix}.routes must be a list")
        route_keys: set[str] = set()
        for route_index, route in enumerate(routes):
            route_prefix = f"{prefix}.routes[{route_index}]"
            if not isinstance(route, dict):
                raise ContactRegistryError(f"{route_prefix} must be a mapping")
            key = route.get("key")
            platform = route.get("platform")
            destination = route.get("destination")
            status = route.get("status", "unverified")
            if not isinstance(key, str) or not key.strip():
                raise ContactRegistryError(f"{route_prefix}.key must be a non-empty string")
            if key in route_keys:
                raise ContactRegistryError(f"duplicate route key for {contact_id}: {key}")
            route_keys.add(key)
            if not isinstance(platform, str) or not platform.strip():
                raise ContactRegistryError(f"{route_prefix}.platform must be a non-empty string")
            if not isinstance(destination, str) or not destination.strip():
                raise ContactRegistryError(f"{route_prefix}.destination must be a non-empty string")
            if status not in _ROUTE_STATES:
                raise ContactRegistryError(
                    f"{route_prefix}.status must be one of {sorted(_ROUTE_STATES)}"
                )
            if "sendable" in route and not isinstance(route["sendable"], bool):
                raise ContactRegistryError(f"{route_prefix}.sendable must be a boolean")
            _string_list(route.get("preferred_for"), f"{route_prefix}.preferred_for")
            _string_list(route.get("constraints"), f"{route_prefix}.constraints")

    policy = data.get("policy", {})
    if policy is not None and not isinstance(policy, dict):
        raise ContactRegistryError("policy must be a mapping")
    return data


def load_registry(path: Path | None = None) -> dict[str, Any]:
    target = path or registry_path()
    if not target.exists():
        raise ContactRegistryError(
            f"contact registry not found: {target}; run `hermes contacts init`"
        )
    if os.name != "nt" and stat.S_IMODE(target.stat().st_mode) & 0o077:
        raise ContactRegistryError(
            f"contact registry permissions are too broad: {target}; run `chmod 600 {target}`"
        )
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContactRegistryError(f"could not read contact registry {target}: {exc}") from exc
    return validate_registry(raw)


def _safe_write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Atomically write a registry and enforce owner-only permissions on POSIX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        if os.name != "nt":
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        if os.name != "nt":
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        tmp.unlink(missing_ok=True)
        raise


def init_registry(path: Path | None = None) -> Path:
    target = path or registry_path()
    if target.exists():
        raise ContactRegistryError(f"contact registry already exists: {target}")
    data = {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "default_send": "deny",
        },
        "contacts": [],
    }
    _safe_write_yaml(target, data)
    return target


def find_contact(contacts: list[dict[str, Any]], query: str) -> tuple[str, dict[str, Any] | None]:
    normalized = _norm(query)
    matches = [
        contact
        for contact in contacts
        if normalized and normalized in {_norm(name) for name in _contact_names(contact)}
    ]
    if not matches:
        return "unknown_contact", None
    if len(matches) > 1:
        return "ambiguous_contact", None
    return "ok", matches[0]


def choose_route(
    contact: dict[str, Any], *, purpose: str | None = None, route_key: str | None = None
) -> tuple[str, dict[str, Any] | None]:
    routes = contact.get("routes") or []
    if route_key:
        matches = [route for route in routes if route.get("key") == route_key]
    elif purpose:
        matches = [
            route for route in routes if purpose in (route.get("preferred_for") or [])
        ]
    else:
        return "route_selector_required", None
    if not matches:
        return "no_preferred_route", None
    if len(matches) > 1:
        return "ambiguous_route", None
    return "ok", matches[0]


def load_directory(path: Path | None = None) -> dict[str, Any]:
    target = path or directory_path()
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def directory_has(directory: dict[str, Any], platform: str, destination: str) -> bool:
    entries = (directory.get("platforms") or {}).get(platform) or []
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("id", ""))
        if entry_id == destination:
            return True
        # Some adapters expose a stable chat_id separately from a composite id.
        if str(entry.get("chat_id", "")) == destination:
            return True
    return False


def directory_is_fresh(directory: dict[str, Any]) -> bool:
    """Return whether the generated directory is recent enough to use.

    The gateway refreshes this cache every five minutes. A ten-minute ceiling
    tolerates one missed refresh while preventing a stopped or reconfigured
    gateway from leaving indefinitely trusted reachability evidence.
    """
    updated_at = directory.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at.strip():
        return False
    try:
        timestamp = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.astimezone()
        age_seconds = (datetime.now(timestamp.tzinfo) - timestamp).total_seconds()
    except (TypeError, ValueError, OSError):
        return False
    return -60 <= age_seconds <= _DIRECTORY_MAX_AGE_SECONDS


def resolve_contact(
    data: dict[str, Any],
    query: str,
    *,
    purpose: str | None = None,
    route_key: str | None = None,
    directory: dict[str, Any] | None = None,
    show_destination: bool = False,
) -> tuple[int, dict[str, Any]]:
    """Resolve a route without sending or claiming authorization."""
    status, contact = find_contact(data["contacts"], query)
    if status != "ok" or contact is None:
        return 2, {"status": status, "query": query, "send_performed": False}

    status, route = choose_route(contact, purpose=purpose, route_key=route_key)
    result: dict[str, Any] = {
        "contact_id": contact.get("id"),
        "display_name": contact.get("display_name"),
        "purpose": purpose,
        "send_performed": False,
        "authorization_check": "required",
    }
    if status != "ok" or route is None:
        return 3, {**result, "status": status}

    route_status = str(route.get("status", "unverified"))
    result.update(
        {
            "route_key": route.get("key"),
            "platform": route.get("platform"),
            "destination_type": route.get("destination_type"),
            "route_status": route_status,
            "last_verified": route.get("last_verified"),
            "constraints": route.get("constraints") or [],
        }
    )
    if show_destination:
        result["destination"] = route.get("destination")
        if route.get("mention"):
            result["mention"] = route.get("mention")

    if route_status == "stale":
        return 4, {**result, "status": "stale_destination"}
    if route_status != "verified":
        return 4, {**result, "status": "unverified_destination"}
    if route.get("sendable") is False:
        return 4, {**result, "status": "not_sendable"}

    platform = str(route.get("platform", "")).casefold()
    destination = str(route.get("destination", ""))
    if platform in _NON_DIRECTORY_PLATFORMS:
        # Email requires an account-specific check outside the gateway directory.
        # Return the selected route but do not report a send-ready resolution.
        return 4, {
            **result,
            "status": "live_check_unavailable",
            "live_check": "unsupported_for_platform",
        }
    if directory is None or not directory_is_fresh(directory):
        return 4, {
            **result,
            "status": "stale_channel_directory",
            "live_check": "directory_missing_or_stale",
        }
    if not directory_has(directory, platform, destination):
        return 4, {
            **result,
            "status": "destination_not_in_live_directory",
            "live_check": "failed",
        }
    result["live_check"] = "fresh_directory_match"

    return 0, {**result, "status": "ok"}


def _json_print(payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _redacted_contact(contact: dict[str, Any], *, show_destinations: bool = False) -> dict[str, Any]:
    result = {
        "id": contact.get("id"),
        "display_name": contact.get("display_name"),
        "aliases": contact.get("aliases") or [],
        "routes": [],
    }
    for route in contact.get("routes") or []:
        item = {
            "key": route.get("key"),
            "platform": route.get("platform"),
            "preferred_for": route.get("preferred_for") or [],
            "status": route.get("status", "unverified"),
            "last_verified": route.get("last_verified"),
        }
        if show_destinations:
            item["destination"] = route.get("destination")
        result["routes"].append(item)
    return result


def contacts_command(args: Any) -> int:
    action = getattr(args, "contacts_action", None)
    path = Path(args.file).expanduser() if getattr(args, "file", None) else registry_path()
    try:
        if action == "path":
            print(path)
            return 0
        if action == "init":
            created = init_registry(path)
            _json_print({"status": "created", "path": str(created)})
            return 0

        data = load_registry(path)
        if action == "validate":
            _json_print(
                {
                    "status": "ok",
                    "path": str(path),
                    "schema_version": data["schema_version"],
                    "contact_count": len(data["contacts"]),
                }
            )
            return 0
        if action in ("list", "ls"):
            _json_print(
                [
                    _redacted_contact(contact, show_destinations=bool(args.show_destinations))
                    for contact in data["contacts"]
                ]
            )
            return 0
        if action == "show":
            status, contact = find_contact(data["contacts"], args.contact)
            if status != "ok" or contact is None:
                _json_print({"status": status, "query": args.contact})
                return 2
            _json_print(
                _redacted_contact(contact, show_destinations=bool(args.show_destinations))
            )
            return 0
        if action == "resolve":
            directory = load_directory(
                Path(args.directory).expanduser() if args.directory else directory_path()
            )
            code, result = resolve_contact(
                data,
                args.contact,
                purpose=args.purpose,
                route_key=args.route,
                directory=directory,
                show_destination=bool(args.show_destination),
            )
            _json_print(result)
            return code

        args._contacts_parser.print_help()
        return 0
    except ContactRegistryError as exc:
        _json_print({"status": "registry_error", "detail": str(exc), "send_performed": False})
        return 5


def build_parser(subparsers: Any) -> Any:
    parser = subparsers.add_parser(
        "contacts",
        help="Manage the profile-scoped contact and outbound-route registry",
        description=(
            "Manage a user-owned contact registry under $HERMES_HOME. "
            "Resolution is non-sending and does not grant messaging authority."
        ),
    )
    parser.add_argument(
        "--file",
        help="Registry path (default: $HERMES_HOME/contacts.yaml)",
    )
    actions = parser.add_subparsers(dest="contacts_action")

    actions.add_parser("init", help="Create an empty owner-readable registry")
    actions.add_parser("path", help="Print the active registry path")
    actions.add_parser("validate", help="Validate schema and identity/route uniqueness")

    list_parser = actions.add_parser("list", aliases=["ls"], help="List contacts without destinations")
    list_parser.add_argument(
        "--show-destinations",
        action="store_true",
        help="Include endpoint values in output",
    )

    show = actions.add_parser("show", help="Show one contact without destinations")
    show.add_argument("contact")
    show.add_argument("--show-destinations", action="store_true")

    resolve = actions.add_parser("resolve", help="Resolve one route without sending")
    resolve.add_argument("contact")
    selector = resolve.add_mutually_exclusive_group(required=True)
    selector.add_argument("--purpose", help="Purpose-specific preferred route")
    selector.add_argument("--route", help="Explicit route key")
    resolve.add_argument(
        "--directory",
        help="Generated channel directory (default: $HERMES_HOME/channel_directory.json)",
    )
    resolve.add_argument(
        "--show-destination",
        action="store_true",
        help="Include the endpoint value in output",
    )

    parser.set_defaults(func=contacts_command, _contacts_parser=parser)
    return parser
