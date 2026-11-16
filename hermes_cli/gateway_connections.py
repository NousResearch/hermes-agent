"""The machine's saved gateway connections: ONE list that native Desktop and every browser on this
machine's Webapp read and write.

The file is ``<root>/gateway-connections.json`` beside the machine's UI preferences store (same
root rule as :func:`hermes_cli.ui_preferences.store_path`: a profile home names its root), mode
0600. Its document is Electron's v2 connection registry (``apps/desktop/electron/connection-registry.ts``)
plus a ``revision`` counter and ``importedFrom`` (the Desktop userData directories already merged in,
so a connection removed here is never re-imported). Writers share the ``<store>.lock`` directory lock,
replace the file whole by temp + rename, keep an unparseable file aside as ``.corrupt-<ms>``, and
treat only a missing file as "no store". The Electron twin is ``electron/gateway-connections-store.ts``.

Secrets are ``{"encoding": "plain", "value": ...}`` envelopes. The one exception is a secret Desktop
could not decrypt when it imported its old store: it stays the original ``safeStorage`` envelope (and
is named in ``importNotices``) rather than being dropped, reads here as unset, and Desktop converts it
to plain on its next write once its keychain answers.

OAuth sign-ins for oauth-mode entries live in ``<root>/gateway-connection-tokens.json`` under the same
lock contract: ``{"version": 1, "revision": N, "tokens": {<connection id>: NativeTokenSet},
"unbound": {<base url>: NativeTokenSet}}`` (``unbound`` holds sign-ins for a URL no entry names yet).
Each set carries the ``baseUrl`` it was issued by: a set read for an entry that names another URL now
reads as signed out, and a registry write that removes an entry or changes its ``url``/``authMode``
drops that entry's set.
"""

from __future__ import annotations

import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from hermes_cli import ui_preferences
from hermes_cli.gateway_connections_normalize import (
    ConnectionStoreBusy,
    ConnectionStoreError,
    DuplicateConnection,
    NoSuchConnection,
    connection_dial_fields_changed,
    merge_connection_input,
    normalize_connection_input,
    normalize_registry,
    normalize_remote_base_url,
    remove_connection as _remove,
    sanitize_header_value,
    set_connection_launch_mode,
    set_last_used_connection,
    set_primary_connection,
    upsert_connection,
)
from utils import atomic_json_write

__all__ = [
    "STORE_NAME",
    "TOKENS_NAME",
    "ConnectionStoreBusy",
    "ConnectionStoreError",
    "DuplicateConnection",
    "NoSuchConnection",
    "connection_secret",
    "dial_snapshot",
    "drop_tokens",
    "public_registry",
    "read_registry",
    "read_tokens",
    "remove_connection",
    "save_connection",
    "set_last_used",
    "set_launch_mode",
    "set_primary",
    "store_path",
    "tokens_path",
    "write_tokens",
]

STORE_NAME = "gateway-connections.json"
TOKENS_NAME = "gateway-connection-tokens.json"
_REGISTRY_KEYS = ("version", "primary", "launchMode", "lastUsed", "connections", "quarantined")
#: ``quarantinedOverflow``: quarantined entries past the normalizer's cap that Desktop's import merged
#: in; carried verbatim so no quarantined user data is lost (``importNotices`` names it).
_CARRIED_KEYS = ("importedFrom", "importNotices", "quarantinedOverflow")


def _root(root: Optional[Path]) -> Path:
    return Path(root) if root is not None else ui_preferences.store_path().parent


def store_path(root: Optional[Path] = None) -> Path:
    return _root(root) / STORE_NAME


def tokens_path(root: Optional[Path] = None) -> Path:
    return _root(root) / TOKENS_NAME


def _read_doc(path: Path) -> tuple[Optional[dict], bool]:
    """``(document, corrupt)``; ``(None, False)`` for a missing file. Any other read failure says
    nothing about the content, so it refuses rather than reading as empty."""
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None, False
    except OSError as exc:
        raise ConnectionStoreBusy(f"gateway connections store is unreadable: {exc}") from exc
    try:
        doc = json.loads(raw)
    except ValueError:
        doc = None
    return (doc, False) if isinstance(doc, dict) else (None, True)


def _revision(doc: Optional[dict]) -> int:
    value = (doc or {}).get("revision")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


@contextmanager
def _locked(path: Path) -> Iterator[Callable[[], bool]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ui_preferences._store_lock(path) as still_ours:
            yield still_ours
    except ui_preferences.PreferencesBusy as exc:
        raise ConnectionStoreBusy(str(exc).replace("preferences", "gateway connections")) from exc


def _replace(path: Path, doc: dict, corrupt: bool, still_ours: Callable[[], bool]) -> None:
    if not still_ours():
        # Held past the stale limit and broken by another writer: replacing now could drop its save.
        raise ConnectionStoreBusy("gateway connections store lock was taken over")
    if corrupt:
        shutil.copy2(path, path.with_name(f"{path.name}.corrupt-{int(time.time() * 1000)}"))
    atomic_json_write(path, doc, mode=0o600)


def read_registry(path: Optional[Path] = None) -> dict:
    """The normalized registry, secrets included (server-internal), plus ``revision``."""
    doc, _corrupt = _read_doc(path or store_path())
    return {**normalize_registry(doc), "revision": _revision(doc)}


def _signed_in_ids_gone(before: dict, after: dict) -> list[str]:
    """Ids whose sign-in no longer belongs to them: removed, or re-pointed (``url``/``authMode``)."""
    now = {c["id"]: c for c in after["connections"]}
    return [c["id"] for c in before["connections"]
            if c["id"] not in now or any(c.get(k) != now[c["id"]].get(k) for k in ("url", "authMode"))]


def _mutate(change: Callable[[dict], Any], path: Optional[Path] = None) -> tuple[dict, Any]:
    """Apply *change* to the fresh registry under the lock and land the result; *change* returns
    ``(registry, extra)``. An unchanged registry writes nothing."""
    path = path or store_path()
    with _locked(path) as still_ours:
        doc, corrupt = _read_doc(path)
        current = normalize_registry(doc)
        updated, extra = change(current)
        updated = normalize_registry(updated)
        revision = _revision(doc)
        gone: list[str] = []
        if updated != current or doc is None:
            revision += 1
            carried = {k: doc[k] for k in _CARRIED_KEYS if doc and k in doc}
            _replace(path, {"version": 2, "revision": revision, **carried,
                            **{k: updated[k] for k in _REGISTRY_KEYS if k in updated}}, corrupt, still_ours)
            gone = _signed_in_ids_gone(current, updated)
    # After the store lock (one process cannot hold both); read_tokens' URL binding covers the gap.
    drop_tokens(gone, path.parent)
    return {**updated, "revision": revision}, extra


def _plain(secret: Any) -> str:
    """The plaintext of a stored envelope; a ``safeStorage`` blob only Desktop's keychain opens
    reads as unset."""
    if isinstance(secret, str):
        return secret
    if not isinstance(secret, dict) or secret.get("encoding") == "safeStorage":
        return ""
    value = secret.get("value")
    return value if isinstance(value, str) else ""


def _kept_envelope(token: Any, existing: Optional[dict]) -> bool:
    """*token* is the stored ``safeStorage`` envelope Desktop could not decrypt on import, kept as is
    by an edit that did not touch the token: still a token (Desktop opens it once its keychain answers)."""
    return (isinstance(token, dict) and token.get("encoding") == "safeStorage" and bool(token.get("value"))
            and existing is not None and existing.get("token") == token)


def _envelope(value: str) -> dict:
    return {"encoding": "plain", "value": value}


def _token_preview(value: str) -> Optional[str]:
    if not value:
        return None
    return "set" if len(value) <= 8 else f"...{value[-6:]}"


def _public_connection(entry: dict) -> dict:
    rest = {k: v for k, v in entry.items() if k not in ("token", "headers")}
    token = _plain(entry.get("token"))
    headers = entry.get("headers")
    return {**rest, "tokenSet": bool(token), "tokenPreview": _token_preview(token),
            "headerNames": list(headers) if isinstance(headers, dict) else []}


def public_registry(reg: dict) -> dict:
    """Electron's ``sanitizeConnectionsRegistry`` shape: never a token or header value."""
    return {
        "version": reg["version"],
        "primary": reg["primary"],
        "launchMode": reg["launchMode"],
        "lastUsed": reg["lastUsed"],
        "secureTokenStorage": True,
        "connections": [_public_connection(c) for c in reg["connections"]],
        "quarantined": [
            {"reason": str(q.get("reason") or "unknown"),
             "label": q["entry"]["label"] if isinstance(q.get("entry"), dict)
             and isinstance(q["entry"].get("label"), str) else ""}
            for q in reg.get("quarantined") or [] if isinstance(q, dict)
        ],
    }


def _incoming_headers(raw: dict, existing: Any) -> dict:
    """``encryptIncomingRemoteHeaders``: a string is a new value, ``None`` keeps the stored one,
    an envelope is stored verbatim; a name missing from *raw* is cleared."""
    stored = existing if isinstance(existing, dict) else {}
    out: dict = {}
    for name, value in raw.items():
        key = str(name or "").strip()
        if not key:
            continue
        if isinstance(value, str):
            if value.strip():
                out[key] = _envelope(value.strip())
        elif value is None:
            if key in stored:
                out[key] = stored[key]
        elif isinstance(value, dict):
            out[key] = value
    return out


def save_connection(payload: dict) -> tuple[dict, dict]:
    """Create or edit one connection: Electron ``saveRegistryConnection``. Returns
    ``(public connection, registry)``. Raises :class:`DuplicateConnection` on a label, URL or SSH
    target collision and :class:`ConnectionStoreError` on any other invalid payload."""
    if not isinstance(payload, dict):
        raise ConnectionStoreError("A connection save needs a JSON object.")

    def change(registry: dict):
        existing = next((c for c in registry["connections"] if payload.get("id") and c["id"] == payload["id"]), None)
        data = dict(payload)
        data.pop("token", None)
        incoming = payload.get("token").strip() if isinstance(payload.get("token"), str) else ""
        token = _envelope(incoming) if incoming else (existing or {}).get("token")
        if token is not None:
            data["token"] = token
        if isinstance(payload.get("headers"), dict):
            data["headers"] = _incoming_headers(payload["headers"], (existing or {}).get("headers"))
        entry = normalize_connection_input(merge_connection_input(data, existing), registry)
        if (entry["kind"] == "remote" and entry.get("authMode") != "oauth" and not _plain(entry.get("token"))
                and not _kept_envelope(entry.get("token"), existing)):
            raise ConnectionStoreError("Remote gateway session token is required.")
        changed = bool(existing) and connection_dial_fields_changed(existing, entry)
        return upsert_connection(registry, entry), (entry, changed)

    registry, (entry, _changed) = _mutate(change)
    return _public_connection(entry), registry


def remove_connection(connection_id: str) -> dict:
    if not any(c["id"] == connection_id for c in read_registry()["connections"]):
        raise NoSuchConnection(f'No connection with id "{connection_id}".')
    return _mutate(lambda reg: (_remove(reg, connection_id), None))[0]


def set_primary(connection_id: str) -> dict:
    return _mutate(lambda reg: (set_primary_connection(reg, connection_id), None))[0]


def set_launch_mode(mode: str) -> dict:
    return _mutate(lambda reg: (set_connection_launch_mode(reg, mode), None))[0]


def set_last_used(connection_id: str) -> dict:
    return _mutate(lambda reg: (set_last_used_connection(reg, connection_id), None))[0]


def _entry(connection_id: str) -> dict:
    entry = next((c for c in read_registry()["connections"] if c["id"] == connection_id), None)
    if entry is None:
        raise NoSuchConnection(f'No connection with id "{connection_id}".')
    return entry


def connection_secret(connection_id: str) -> dict:
    """``{"token": str | None, "headers": {name: value}}`` for dialing *connection_id* (the proxy)."""
    return _secret_of(_entry(connection_id))


def _secret_of(entry: dict) -> dict:
    headers = {}
    for name, secret in (entry.get("headers") or {}).items():
        value = sanitize_header_value(_plain(secret))
        if value:
            headers[name] = value
    return {"token": _plain(entry.get("token")) or None, "headers": headers}


# ── OAuth token sets ─────────────────────────────────────────────────────────


def _token_doc(path: Path) -> tuple[dict, bool, bool]:
    doc, corrupt = _read_doc(path)
    exists = doc is not None
    doc = doc or {}
    return ({"version": 1, "revision": _revision(doc),
             **{k: doc[k] if isinstance(doc.get(k), dict) else {} for k in ("tokens", "unbound", "undecrypted")}},
            corrupt, exists)


def _entry_url(entry: dict) -> Optional[str]:
    try:
        return normalize_remote_base_url(entry["url"]) if entry.get("url") else None
    except ConnectionStoreError:
        return None


def _bound_url(token_set: Any) -> Optional[str]:
    try:
        return normalize_remote_base_url(token_set["baseUrl"]) if isinstance(token_set.get("baseUrl"), str) else None
    except ConnectionStoreError:
        return None


def read_tokens(connection_id: str) -> Optional[dict]:
    """The NativeTokenSet signed in for *connection_id*, or ``None``. A set issued by another URL
    than the entry names now (it was re-pointed) is never handed out: it reads as signed out."""
    return _tokens_of(_entry(connection_id))


def _tokens_of(entry: dict) -> Optional[dict]:
    connection_id = entry["id"]
    url = _entry_url(entry)
    doc, _corrupt, _exists = _token_doc(tokens_path())
    found = doc["tokens"].get(connection_id)
    if found is None and url:
        found = doc["unbound"].get(url)
        found = {**found, "baseUrl": url} if isinstance(found, dict) else None
    if not (isinstance(found, dict) and found.get("accessToken")) or url is None or _bound_url(found) != url:
        return None
    return found


def dial_snapshot(connection_id: str) -> dict:
    """Everything dialing *connection_id* needs, from ONE registry read: ``{entry, url, token,
    headers, tokens}``. ``url`` is the normalized address (``None`` for local/ssh), and ``tokens``
    is the OAuth set only when it was issued by that same ``url``. Callers dial ``url`` and nothing
    else: credentials read by id in a second lookup can belong to an address the entry was
    re-pointed to meanwhile."""
    entry = _entry(connection_id)
    secret = _secret_of(entry)
    return {"entry": entry, "url": _entry_url(entry), "token": secret["token"], "headers": secret["headers"],
            "tokens": _tokens_of(entry) if entry.get("authMode") == "oauth" else None}


def write_tokens(connection_id: str, token_set: Optional[dict], *, issued_by: Optional[str] = None) -> None:
    """Store (or with ``None`` drop) *connection_id*'s sign-in, bound to the URL it names now.

    *issued_by* is the URL the set came from (a refresh or a completed sign-in). The entry is read
    again under the tokens lock and a set whose issuer it no longer names is refused, so a re-point
    landing while the remote answered never binds one host's tokens to another."""
    if token_set is not None and not (isinstance(token_set, dict) and token_set.get("accessToken")):
        raise ConnectionStoreError("A token set needs an accessToken.")
    path = tokens_path()
    with _locked(path) as still_ours:
        entry = _entry(connection_id)
        url = _entry_url(entry)
        if token_set is not None and url is None:
            raise ConnectionStoreError(f'"{entry.get("label")}" has no gateway address to sign in to.')
        if token_set is not None and issued_by is not None and _bound_url({"baseUrl": issued_by}) != url:
            raise ConnectionStoreError(f'"{entry.get("label")}" changed address while signing in; sign in again.')
        doc, corrupt, _exists = _token_doc(path)
        if url:
            doc["unbound"].pop(url, None)
        if token_set is None:
            doc["tokens"].pop(connection_id, None)
        else:
            doc["tokens"][connection_id] = {**token_set, "baseUrl": url}
        doc["revision"] += 1
        _replace(path, doc, corrupt, still_ours)


def drop_tokens(connection_ids: list[str], root: Optional[Path] = None) -> None:
    """Forget the sign-ins kept under *connection_ids* (entries removed or re-pointed); no entry
    lookup, so it works after the entry is gone."""
    path = tokens_path(root)
    if not connection_ids or not path.exists():
        return
    with _locked(path) as still_ours:
        doc, corrupt, _exists = _token_doc(path)
        if not any(i in doc["tokens"] for i in connection_ids):
            return
        for connection_id in connection_ids:
            doc["tokens"].pop(connection_id, None)
        doc["revision"] += 1
        _replace(path, doc, corrupt, still_ours)
