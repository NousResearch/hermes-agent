"""Named Google Workspace OAuth auth contexts.

An auth context is a named OAuth credential bundle: account hint, requested
services/scopes, client secret, token, and pending PKCE session. Contexts let one
Hermes profile route different Google operations through different least-
privilege grants without splitting agent memory across Hermes profiles.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from _hermes_home import get_hermes_home

LEGACY_TOKEN_NAME = "google_token.json"
LEGACY_CLIENT_SECRET_NAME = "google_client_secret.json"
LEGACY_PENDING_NAME = "google_oauth_pending.json"
AUTH_CONTEXTS_NAME = "google_workspace_auth_contexts.json"
_LEGACY_MIGRATION_KEY = "legacy_default_migrated"

_CONTEXT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

GMAIL_READONLY = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_SEND = "https://www.googleapis.com/auth/gmail.send"
GMAIL_MODIFY = "https://www.googleapis.com/auth/gmail.modify"
CALENDAR = "https://www.googleapis.com/auth/calendar"
CALENDAR_READONLY = "https://www.googleapis.com/auth/calendar.readonly"
DRIVE = "https://www.googleapis.com/auth/drive"
DRIVE_READONLY = "https://www.googleapis.com/auth/drive.readonly"
DRIVE_FILE = "https://www.googleapis.com/auth/drive.file"
CONTACTS_READONLY = "https://www.googleapis.com/auth/contacts.readonly"
SHEETS = "https://www.googleapis.com/auth/spreadsheets"
SHEETS_READONLY = "https://www.googleapis.com/auth/spreadsheets.readonly"
DOCS = "https://www.googleapis.com/auth/documents"
DOCS_READONLY = "https://www.googleapis.com/auth/documents.readonly"

SERVICE_SCOPE_MAP: dict[str, list[str]] = {
    "gmail-readonly": [GMAIL_READONLY],
    "gmail-send": [GMAIL_SEND],
    "gmail-modify": [GMAIL_MODIFY],
    # Legacy/default Gmail service remains read/send/modify for backward compatibility.
    "gmail": [GMAIL_READONLY, GMAIL_SEND, GMAIL_MODIFY],
    "email": [GMAIL_READONLY, GMAIL_SEND, GMAIL_MODIFY],
    "calendar-readonly": [CALENDAR_READONLY],
    "calendar": [CALENDAR],
    "drive-readonly": [DRIVE_READONLY],
    "drive-file": [DRIVE_FILE],
    "drive": [DRIVE],
    "contacts-readonly": [CONTACTS_READONLY],
    "contacts": [CONTACTS_READONLY],
    "sheets-readonly": [SHEETS_READONLY],
    "sheets": [SHEETS],
    "docs-readonly": [DOCS_READONLY],
    "docs": [DOCS],
}

LEGACY_DEFAULT_SERVICES = ["gmail", "calendar", "drive", "contacts", "sheets", "docs"]
SCOPES = [scope for service in LEGACY_DEFAULT_SERVICES for scope in SERVICE_SCOPE_MAP[service]]

# Full write scopes imply narrower read scopes for local command gating.
SCOPE_IMPLICATIONS: dict[str, set[str]] = {
    GMAIL_MODIFY: {GMAIL_READONLY},
    CALENDAR: {CALENDAR_READONLY},
    DRIVE: {DRIVE_READONLY, DRIVE_FILE},
    SHEETS: {SHEETS_READONLY},
    DOCS: {DOCS_READONLY},
}

# Each set is an OR group; multiple groups for one command are conjunctive.
COMMAND_SCOPE_REQUIREMENTS: dict[tuple[str, str], list[set[str]]] = {
    ("gmail", "search"): [{GMAIL_READONLY, GMAIL_MODIFY}],
    ("gmail", "get"): [{GMAIL_READONLY, GMAIL_MODIFY}],
    ("gmail", "labels"): [{GMAIL_READONLY, GMAIL_MODIFY}],
    ("gmail", "send"): [{GMAIL_SEND}],
    ("gmail", "reply"): [{GMAIL_SEND}, {GMAIL_READONLY, GMAIL_MODIFY}],
    ("gmail", "modify"): [{GMAIL_MODIFY}],
    ("calendar", "list"): [{CALENDAR_READONLY, CALENDAR}],
    ("calendar", "create"): [{CALENDAR}],
    ("calendar", "delete"): [{CALENDAR}],
    ("drive", "search"): [{DRIVE_READONLY, DRIVE_FILE, DRIVE}],
    ("drive", "get"): [{DRIVE_READONLY, DRIVE_FILE, DRIVE}],
    ("drive", "download"): [{DRIVE_READONLY, DRIVE_FILE, DRIVE}],
    ("drive", "upload"): [{DRIVE_FILE, DRIVE}],
    ("drive", "create-folder"): [{DRIVE_FILE, DRIVE}],
    ("drive", "share"): [{DRIVE}],
    ("drive", "delete"): [{DRIVE}],
    ("contacts", "list"): [{CONTACTS_READONLY}],
    ("sheets", "get"): [{SHEETS_READONLY, SHEETS}],
    ("sheets", "create"): [{SHEETS}],
    ("sheets", "update"): [{SHEETS}],
    ("sheets", "append"): [{SHEETS}],
    ("docs", "get"): [{DOCS_READONLY, DOCS}],
    ("docs", "create"): [{DOCS}],
    ("docs", "append"): [{DOCS}],
}


def hermes_home() -> Path:
    return get_hermes_home()


def store_path() -> Path:
    return hermes_home() / AUTH_CONTEXTS_NAME


def legacy_token_path() -> Path:
    return hermes_home() / LEGACY_TOKEN_NAME


def legacy_client_secret_path() -> Path:
    return hermes_home() / LEGACY_CLIENT_SECRET_NAME


def legacy_pending_path() -> Path:
    return hermes_home() / LEGACY_PENDING_NAME


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_legacy_default(data: dict[str, Any]) -> None:
    """Snapshot legacy credentials into ``contexts.default`` exactly once.

    Creating a named context must not make a working legacy default disappear.
    Once the store is initialized, it remains authoritative and later changes
    to stale legacy files are ignored, including after revoke/delete actions.
    """
    if data.get(_LEGACY_MIGRATION_KEY) is True:
        return
    legacy_payloads = {
        "token": _read_json_file(legacy_token_path()),
        "client_secret": _read_json_file(legacy_client_secret_path()),
        "pending_auth": _read_json_file(legacy_pending_path()),
    }
    if any(legacy_payloads.values()):
        contexts = data.setdefault("contexts", {})
        default = contexts.setdefault("default", {"name": "default"})
        default.setdefault("name", "default")
        for key, payload in legacy_payloads.items():
            if payload and key not in default:
                default[key] = payload
    data[_LEGACY_MIGRATION_KEY] = True


def validate_context_name(name: str | None) -> str:
    if name is None:
        name = "default"
    if not isinstance(name, str) or name != name.strip() or not _CONTEXT_RE.fullmatch(name) or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(
            "Auth context names must be 1-64 chars: letters, numbers, '.', '_', '-' "
            "and may not contain path separators"
        )
    return name


def _empty_store() -> dict[str, Any]:
    return {"version": 1, "default_contexts": {}, "contexts": {}}


def load_store() -> dict[str, Any]:
    path = store_path()
    if not path.exists():
        return _empty_store()
    try:
        data = json.loads(path.read_text())
    except Exception:
        return _empty_store()
    if not isinstance(data, dict):
        return _empty_store()
    data.setdefault("version", 1)
    data.setdefault("default_contexts", {})
    data.setdefault("contexts", {})
    _merge_legacy_default(data)
    return data


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    try:
        tmp.chmod(0o600)
    except OSError:
        pass
    os.replace(tmp, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def save_store(data: dict[str, Any]) -> None:
    data.setdefault("version", 1)
    data.setdefault("default_contexts", {})
    data.setdefault("contexts", {})
    _merge_legacy_default(data)
    _write_private_json(store_path(), data)


def _context_entry(store: dict[str, Any], context: str) -> dict[str, Any]:
    context = validate_context_name(context)
    contexts = store.setdefault("contexts", {})
    entry = contexts.setdefault(context, {})
    entry.setdefault("name", context)
    return entry


def context_exists(context: str = "default") -> bool:
    context = validate_context_name(context)
    if context == "default" and (legacy_token_path().exists() or legacy_client_secret_path().exists()):
        return True
    return context in load_store().get("contexts", {})


def list_contexts() -> list[str]:
    names = set(load_store().get("contexts", {}).keys())
    if legacy_token_path().exists() or legacy_client_secret_path().exists():
        names.add("default")
    return sorted(names)


def scope_list(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return [s.strip() for s in raw.split() if s.strip()]
    return [str(s).strip() for s in raw if str(s).strip()]


def expand_scopes(scopes: list[str]) -> set[str]:
    expanded = set(scopes)
    for scope in list(expanded):
        expanded.update(SCOPE_IMPLICATIONS.get(scope, set()))
    return expanded


def resolve_scopes(*, services: str | None = None, scopes: str | None = None) -> list[str]:
    if scopes:
        return sorted(set(scope_list(scopes.replace(",", " "))))
    selected = (services or "all").strip().lower()
    service_names = LEGACY_DEFAULT_SERVICES if selected in {"", "all"} else [s.strip().lower() for s in selected.split(",") if s.strip()]
    unknown = sorted(set(service_names) - set(SERVICE_SCOPE_MAP))
    if unknown:
        raise ValueError(f"Unknown Google service preset(s): {', '.join(unknown)}")
    out: list[str] = []
    for service in service_names:
        for scope in SERVICE_SCOPE_MAP[service]:
            if scope not in out:
                out.append(scope)
    return out


def get_token_payload(context: str = "default") -> dict[str, Any]:
    context = validate_context_name(context)
    store = load_store()
    token = store.get("contexts", {}).get(context, {}).get("token")
    if isinstance(token, dict):
        return dict(token)
    if context == "default" and legacy_token_path().exists() and not store_path().exists():
        try:
            return json.loads(legacy_token_path().read_text())
        except Exception:
            return {}
    return {}


def set_token_payload(context: str, token_payload: dict[str, Any], *, services: list[str] | None = None, requested_scopes: list[str] | None = None) -> None:
    context = validate_context_name(context)
    if context == "default" and not store_path().exists():
        _write_private_json(legacy_token_path(), token_payload)
        return
    store = load_store()
    entry = _context_entry(store, context)
    entry["token"] = token_payload
    if services is not None:
        entry["services"] = services
    if requested_scopes is not None:
        entry["requested_scopes"] = requested_scopes
    save_store(store)


def get_client_secret(context: str = "default") -> dict[str, Any]:
    context = validate_context_name(context)
    store = load_store()
    secret = store.get("contexts", {}).get(context, {}).get("client_secret")
    if isinstance(secret, dict):
        return dict(secret)
    if context == "default" and legacy_client_secret_path().exists() and not store_path().exists():
        try:
            return json.loads(legacy_client_secret_path().read_text())
        except Exception:
            return {}
    return {}


def set_client_secret(context: str, payload: dict[str, Any], *, account_hint: str = "") -> None:
    context = validate_context_name(context)
    if context == "default" and not store_path().exists():
        _write_private_json(legacy_client_secret_path(), payload)
        return
    store = load_store()
    entry = _context_entry(store, context)
    entry["client_secret"] = payload
    if account_hint:
        entry["account_hint"] = account_hint
    save_store(store)


def get_pending_auth(context: str = "default") -> dict[str, Any]:
    context = validate_context_name(context)
    store = load_store()
    pending = store.get("contexts", {}).get(context, {}).get("pending_auth")
    if isinstance(pending, dict):
        return dict(pending)
    if context == "default" and legacy_pending_path().exists() and not store_path().exists():
        try:
            return json.loads(legacy_pending_path().read_text())
        except Exception:
            return {}
    return {}


def set_pending_auth(context: str, payload: dict[str, Any]) -> None:
    context = validate_context_name(context)
    if context == "default" and not store_path().exists():
        _write_private_json(legacy_pending_path(), payload)
        return
    store = load_store()
    entry = _context_entry(store, context)
    entry["pending_auth"] = payload
    save_store(store)


def clear_pending_auth(context: str = "default") -> None:
    context = validate_context_name(context)
    if context == "default":
        # Remove the legacy source before loading/saving the store so a
        # pre-marker store cannot migrate the pending state back in.
        legacy_pending_path().unlink(missing_ok=True)
    store = load_store()
    if context in store.get("contexts", {}):
        store["contexts"][context].pop("pending_auth", None)
        save_store(store)


def delete_token(context: str = "default") -> None:
    context = validate_context_name(context)
    if context == "default":
        # Destructive intent wins over legacy migration. Unlink sources first
        # so old stores without the one-time marker cannot resurrect them.
        legacy_token_path().unlink(missing_ok=True)
        legacy_pending_path().unlink(missing_ok=True)
    store = load_store()
    if context in store.get("contexts", {}):
        store["contexts"][context].pop("token", None)
        store["contexts"][context].pop("pending_auth", None)
        save_store(store)
    _materialized_token_path(context).unlink(missing_ok=True)


def set_default_for_services(context: str, services_csv: str) -> None:
    context = validate_context_name(context)
    services = [s.strip().lower() for s in services_csv.split(",") if s.strip()]
    valid_api_services = {"gmail", "calendar", "drive", "contacts", "sheets", "docs"}
    unknown = sorted(set(services) - valid_api_services)
    if unknown:
        raise ValueError(f"Unknown API service(s) for defaults: {', '.join(unknown)}")
    store = load_store()
    _context_entry(store, context)
    defaults = store.setdefault("default_contexts", {})
    for service in services:
        defaults[service] = context
    save_store(store)


def default_context_for_service(service: str) -> str:
    return load_store().get("default_contexts", {}).get(service, "default")


def _materialized_token_path(context: str) -> Path:
    return hermes_home() / ".cache" / "google-workspace" / "contexts" / context / LEGACY_TOKEN_NAME


def _materialized_client_secret_path(context: str) -> Path:
    return hermes_home() / ".cache" / "google-workspace" / "contexts" / context / LEGACY_CLIENT_SECRET_NAME


def materialize_token_file(context: str = "default") -> Path:
    context = validate_context_name(context)
    if context == "default" and legacy_token_path().exists() and not store_path().exists():
        return legacy_token_path()
    payload = get_token_payload(context)
    path = _materialized_token_path(context)
    if not payload:
        path.unlink(missing_ok=True)
        return path
    _write_private_json(path, payload)
    return path


def materialize_client_secret_file(context: str = "default") -> Path:
    context = validate_context_name(context)
    if context == "default" and legacy_client_secret_path().exists() and not store_path().exists():
        return legacy_client_secret_path()
    payload = get_client_secret(context)
    path = _materialized_client_secret_path(context)
    if not payload:
        path.unlink(missing_ok=True)
        return path
    _write_private_json(path, payload)
    return path


def granted_scopes_for_context(context: str = "default") -> list[str]:
    context = validate_context_name(context)
    payload = get_token_payload(context)
    scopes = scope_list(payload.get("scopes") or payload.get("scope"))
    if scopes:
        return scopes
    # Legacy default tokens predate stored scope metadata; preserve compatibility
    # only for that legacy layout. Named/store-backed contexts fail closed.
    if context == "default" and legacy_token_path().exists() and not store_path().exists():
        return list(SCOPES)
    return []


def missing_scopes(payload: dict[str, Any], required_scopes: list[str] | None = None) -> list[str]:
    required = required_scopes or list(SCOPES)
    granted = expand_scopes(scope_list(payload.get("scopes") or payload.get("scope")))
    if not granted:
        return [] if required_scopes is None else sorted(required)
    return sorted(scope for scope in required if scope not in granted)


def assert_command_allowed(context: str, service: str, action: str) -> None:
    required_groups = COMMAND_SCOPE_REQUIREMENTS.get((service, action))
    if not required_groups:
        return
    granted = expand_scopes(granted_scopes_for_context(context))
    if all(granted.intersection(group) for group in required_groups):
        return
    needed = " AND ".join(
        next(iter(group)) if len(group) == 1 else f"({' OR '.join(sorted(group))})"
        for group in required_groups
    )
    raise PermissionError(
        f"Auth context '{context}' lacks OAuth scope for {service} {action}. "
        f"Need: {needed}. Granted: {', '.join(sorted(granted)) or '(none)'}"
    )
