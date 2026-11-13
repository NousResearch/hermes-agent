"""Pure normalizers for the machine's saved-gateways registry: the Python twin of
``apps/desktop/electron/connection-registry.ts`` (``normalizeRegistry``, ``normalizeConnectionInput``,
``mergeConnectionInput`` and the registry operations) and the ``connection-config.ts`` helpers they
use. The TypeScript module stays the reference; ``tests/fixtures/gateway_connections/normalize_cases.json``
is run through both and must normalize the same inputs to the same output. Change one, change both.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

REGISTRY_VERSION = 2
LOCAL_CONNECTION_ID = "local"
REGISTRY_QUARANTINE_CAP = 20
LABEL_MAX = 64
KINDS = ("local", "remote", "cloud", "ssh")
_RESERVED_REMOTE_PROFILES = {"hermes", "test", "tmp", "root", "sudo"}
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_FORBIDDEN_HEADERS = {
    "authorization", "connection", "content-length", "content-type", "cookie", "host", "origin",
    "referer", "te", "trailer", "transfer-encoding", "upgrade", "x-hermes-session-token",
}
_PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ConnectionStoreError(ValueError):
    """A save/mutation the registry refuses; the route answers 400."""


class ConnectionStoreBusy(RuntimeError):
    """The store lock was held past the wait, or the file exists but cannot be read; 503."""


class NoSuchConnection(KeyError):
    """No saved connection has the id; 404."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "no such connection"


class DuplicateConnection(ConnectionStoreError):
    """The save collides with a saved connection (label, URL or SSH target)."""

    def __init__(self, message: str, label: str):
        super().__init__(message)
        self.label = label


def js_string(value: Any) -> str:
    """JavaScript ``String(value)`` for parsed JSON values."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, dict):
        return "[object Object]"
    if isinstance(value, list):
        return ",".join("" if v is None else js_string(v) for v in value)
    return str(value)


def js_falsy(value: Any) -> bool:
    return value is None or value is False or value == "" or (
        isinstance(value, (int, float)) and not isinstance(value, bool) and (value == 0 or value != value))


def js_str(value: Any) -> str:
    """``String(value || '')``, the way the TS normalizers read loose JSON."""
    return "" if js_falsy(value) else js_string(value)


def label_key(label: Any) -> str:
    return js_str(label).strip().lower()


def unique_label(candidate: Any, taken: Iterable[str]) -> str:
    used = {label_key(t) for t in taken}
    base = js_str(candidate).strip()[: LABEL_MAX - 4]
    if label_key(base) not in used:
        return base
    n = 2
    while label_key(f"{base} {n}") in used:
        n += 1
    return f"{base} {n}"


def label_slug(label: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", js_str(label).strip().lower()).strip("-")[:48]
    return slug or "connection"


def connection_id_for_label(label: Any, taken: Iterable[str]) -> str:
    used = set(taken)
    base = label_slug(label)
    if base not in used and base != LOCAL_CONNECTION_ID:
        return base
    n = 2
    while f"{base}-{n}" in used or f"{base}-{n}" == LOCAL_CONNECTION_ID:
        n += 1
    return f"{base}-{n}"


def norm_auth_mode(mode: Any) -> str:
    return "oauth" if mode == "oauth" else "token"


def stored_auth_mode(kind: str, auth_mode: Any, token: Any) -> str:
    return "oauth" if kind == "cloud" and not token else norm_auth_mode(auth_mode)


def _url_host(parts) -> str:
    host = (parts.hostname or "").lower()
    return f"[{host}]" if ":" in host else host


def normalize_remote_base_url(raw: Any) -> str:
    """``normalizeRemoteBaseUrl``: scheme-less input gets ``http://``; query, fragment and
    trailing slashes go; only http(s) is accepted."""
    value = js_str(raw).strip()
    if not value:
        raise ConnectionStoreError("Remote gateway URL is required.")
    if not re.match(r"^[a-z][a-z0-9+.-]*://", value, re.IGNORECASE):
        value = f"http://{value}"
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise ConnectionStoreError(f"Remote gateway URL is not valid: {exc}") from exc
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise ConnectionStoreError(f"Remote gateway URL must be http:// or https://, got {scheme}:")
    host = _url_host(parts)
    if not host:
        raise ConnectionStoreError("Remote gateway URL is not valid: Invalid URL")
    userinfo = parts.netloc.rsplit("@", 1)[0] + "@" if "@" in parts.netloc else ""
    default_port = 80 if scheme == "http" else 443
    port_part = f":{port}" if port is not None and port != default_port else ""
    path = parts.path.rstrip("/")
    return f"{scheme}://{userinfo}{host}{port_part}{path}".rstrip("/")


def host_label_from_base_url(base_url: Any) -> Optional[str]:
    raw = js_str(base_url).strip()
    if not raw:
        return None
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    host = _url_host(parts)
    default_port = {"http": 80, "https": 443, "ws": 80, "wss": 443}.get(parts.scheme.lower())
    if port is not None and port != default_port and port not in (80, 443):
        return f"{host}:{port}"
    return host


def sanitize_header_value(value: Any) -> str:
    return re.sub(r"[\r\n]+", "", js_str(value)).strip()


def normalize_remote_headers(raw: Any) -> dict:
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for name, secret in raw.items():
        header = js_str(name).strip()
        if not header or not _HEADER_NAME_RE.match(header) or header.lower() in _FORBIDDEN_HEADERS:
            continue
        if isinstance(secret, str):
            value = sanitize_header_value(secret)
            if value:
                out[header] = {"encoding": "plain", "value": value}
        elif isinstance(secret, dict):
            encoding = js_str(secret.get("encoding"))
            value = js_str(secret.get("value"))
            if value and encoding in ("safeStorage", "plain", ""):
                out[header] = {"encoding": encoding or "plain", "value": value}
    return out


def _split_ssh_host(host: str) -> tuple[str, str, Optional[int]]:
    user, port = "", None
    at = host.find("@")
    if at > 0:
        user, host = host[:at], host[at + 1:]
    bracketed = re.match(r"^\[([^\]]+)](?::(\d+))?$", host)
    if bracketed:
        host = bracketed.group(1)
        if bracketed.group(2):
            port = int(bracketed.group(2))
    elif host.count(":") == 1:
        name, raw_port = host.split(":")
        if re.match(r"^\d+$", raw_port):
            host, port = name, int(raw_port)
    return host, user, port


def _explicit_port(raw: Any) -> Optional[int]:
    text = "" if raw is None else js_string(raw).strip()  # String(port ?? '')
    return int(text) if re.fullmatch(r"\d+", text) else None


def normalize_ssh_config(entry: Any) -> Optional[dict]:
    if not isinstance(entry, dict) or entry.get("mode") != "ssh":
        return None
    host = re.sub(r"^ssh\s+", "", js_str(entry.get("host")).strip(), flags=re.IGNORECASE).strip()
    if not host:
        return None
    host, parsed_user, parsed_port = _split_ssh_host(host)
    if not host:
        return None
    out: dict = {"mode": "ssh", "host": host}
    user = js_str(entry.get("user")).strip() or parsed_user
    if user:
        out["user"] = user
    explicit = _explicit_port(entry.get("port"))
    port = explicit if explicit is not None else parsed_port
    if isinstance(port, int) and 0 < port <= 65535 and port != 22:
        out["port"] = port
    for field in ("keyPath", "remoteHermesPath"):
        value = js_str(entry.get(field)).strip()
        if value:
            out[field] = value
    remote_profile = js_str(entry.get("remoteProfile")).strip()
    if _PROFILE_RE.match(remote_profile) and remote_profile not in _RESERVED_REMOTE_PROFILES:
        out["remoteProfile"] = remote_profile
    return out


def _ssh_fields(ssh: dict) -> dict:
    return {k: v for k, v in ssh.items() if k != "mode"}


def _local_entry(label: str = "This device") -> dict:
    return {"id": LOCAL_CONNECTION_ID, "kind": "local", "label": label}


def _remote_like_clean(clean: dict, entry: dict, kind: str) -> Optional[str]:
    url = js_str(entry.get("url")).strip()
    if not url:
        return "entry-missing-url"
    clean["url"] = url
    clean["authMode"] = stored_auth_mode(kind, entry.get("authMode"), entry.get("token"))
    if "token" in entry:
        clean["token"] = entry["token"]
    headers = normalize_remote_headers(entry.get("headers"))
    if headers:
        clean["headers"] = headers
    name = js_str(entry.get("name")).strip()
    if kind in ("remote", "cloud") and name:
        clean["name"] = name
    org = js_str(entry.get("org")).strip()
    if kind == "cloud" and org:
        clean["org"] = org
    return None


def _ssh_clean(clean: dict, entry: dict) -> Optional[str]:
    ssh = normalize_ssh_config({**entry, "mode": "ssh"})
    if not ssh:
        return "entry-missing-ssh-host"
    clean.update(_ssh_fields(ssh))
    if "token" in entry:
        clean["token"] = entry["token"]
    return None


class _Normalizer:
    def __init__(self):
        self.labels: set[str] = set()
        self.ids: set[str] = set()
        self.connections: list[dict] = []
        self.quarantined: list[dict] = []

    def quarantine(self, reason: str, entry: Any) -> None:
        if len(self.quarantined) < REGISTRY_QUARANTINE_CAP:
            self.quarantined.append({"reason": reason, "entry": entry})

    def add(self, item: Any) -> None:
        if js_falsy(item):
            return  # null/false/'' carry no user data
        if isinstance(item, list):
            self.quarantine("entry-unrecognized-kind", item)  # a JS array is an object without a kind
            return
        if not isinstance(item, dict):
            self.quarantine("entry-malformed", item)
            return
        try:
            self._add_entry(item)
        except (AttributeError, TypeError, ValueError):  # one bad entry never aborts the load (#94246)
            try:
                copy = json.loads(json.dumps(item))
            except (TypeError, ValueError):
                copy = {"unserializable": True}
            self.quarantine("entry-normalization-failed", copy)

    def _add_entry(self, entry: dict) -> None:
        kind = entry.get("kind")
        if kind not in KINDS:
            self.quarantine("entry-unrecognized-kind", entry)
            return
        label = js_str(entry.get("label")).strip()
        if not label:
            label = (js_str(entry.get("host") or "ssh") if kind == "ssh"
                     else host_label_from_base_url(js_str(entry.get("url"))) or kind)
        label = unique_label(label, self.labels)
        cid = LOCAL_CONNECTION_ID if kind == "local" else js_str(entry.get("id")).strip()
        if not cid or (cid in self.ids and kind != "local"):
            cid = connection_id_for_label(label, self.ids)
        if cid in self.ids:
            return
        self.labels.add(label_key(label))
        self.ids.add(cid)
        clean = {"id": cid, "kind": kind, "label": label}
        reason = None
        if kind in ("remote", "cloud"):
            reason = _remote_like_clean(clean, entry, kind)
        elif kind == "ssh":
            reason = _ssh_clean(clean, entry)
        if reason:
            self.quarantine(reason, entry)
            return
        self.connections.append(clean)


def normalize_registry(raw: Any) -> dict:
    """``normalizeRegistry``: exactly one ``local`` entry, labels uniqued (suffixed, never dropped),
    primary/lastUsed always naming an entry, malformed entries quarantined as user data."""
    parsed = raw if isinstance(raw, dict) else {}
    norm = _Normalizer()
    prior = parsed.get("quarantined")
    for item in prior if isinstance(prior, list) else []:
        if isinstance(item, dict) and "entry" in item:
            norm.quarantine(js_str(item.get("reason")) or "unknown", item["entry"])
    raw_connections = parsed.get("connections")
    for item in raw_connections if isinstance(raw_connections, list) else []:
        norm.add(item)
    connections = norm.connections
    if not any(c["kind"] == "local" for c in connections):
        connections.insert(0, _local_entry())
    ids = {c["id"] for c in connections}
    stored_primary = js_str(parsed.get("primary")).strip()
    primary = stored_primary if stored_primary in ids else LOCAL_CONNECTION_ID
    stored_last = js_str(parsed.get("lastUsed")).strip()
    out = {
        "version": REGISTRY_VERSION,
        "primary": primary,
        "launchMode": "last-used" if parsed.get("launchMode") == "last-used" else "primary",
        "lastUsed": stored_last if stored_last in ids else primary,
        "connections": connections,
    }
    if norm.quarantined:
        out["quarantined"] = norm.quarantined
    return out


def _url_key(value: str) -> str:
    return value.strip().rstrip("/").lower()


def _ssh_key(c: dict) -> str:
    return (f"{(c.get('user') or '').lower()}@{(c.get('host') or '').lower()}:{c.get('port') or 22}"
            f"::{(c.get('remoteProfile') or '').strip()}")


def _check_label(label: str, input_id: Any, registry: dict) -> None:
    if not label:
        raise ConnectionStoreError(
            'Every connection needs a name. Give this instance a device name (e.g. "Homelab", "Work laptop").')
    if len(label) > LABEL_MAX:
        raise ConnectionStoreError(f"Connection name is too long (max {LABEL_MAX} characters).")
    key = label_key(label)
    for c in registry["connections"]:
        if label_key(c["label"]) == key and c["id"] != input_id:
            raise DuplicateConnection(
                f'A connection named "{c["label"]}" already exists. Connection names must be unique.', c["label"])


def _ssh_input(data: dict, cid: str, registry: dict) -> dict:
    ssh = normalize_ssh_config({"mode": "ssh", **{k: data.get(k) for k in (
        "host", "user", "port", "keyPath", "remoteHermesPath", "remoteProfile")}})
    if not ssh:
        raise ConnectionStoreError("SSH connections need a host.")
    fields = _ssh_fields(ssh)
    for c in registry["connections"]:
        if c["kind"] == "ssh" and c["id"] != cid and _ssh_key(c) == _ssh_key(fields):
            raise DuplicateConnection(f'A connection to this SSH host already exists ("{c["label"]}").', c["label"])
    entry = {"id": cid, "kind": "ssh", "label": data["_label"], **fields}
    if "token" in data:
        entry["token"] = data["token"]
    return entry


def _remote_input(data: dict, cid: str, kind: str, registry: dict) -> dict:
    url = normalize_remote_base_url(data.get("url"))
    for c in registry["connections"]:
        if c["kind"] in ("remote", "cloud") and c["id"] != cid and _url_key(c.get("url") or "") == _url_key(url):
            raise DuplicateConnection(f'A connection to this gateway URL already exists ("{c["label"]}").', c["label"])
    auth_mode = stored_auth_mode(kind, data.get("authMode"), None)
    entry = {"id": cid, "kind": kind, "label": data["_label"], "url": url, "authMode": auth_mode}
    if "token" in data and kind == "remote" and auth_mode == "token":
        entry["token"] = data["token"]
    if "headers" in data:
        headers = normalize_remote_headers(data["headers"])
        if headers:
            entry["headers"] = headers
    name = js_str(data.get("name")).strip()
    if kind in ("remote", "cloud") and name:
        entry["name"] = name
    org = js_str(data.get("org")).strip()
    if kind == "cloud" and org:
        entry["org"] = org
    return entry


def normalize_connection_input(data: dict, registry: dict) -> dict:
    """``normalizeConnectionInput``: validate a save payload into a registry entry. Keys absent from
    *data* are TS ``undefined``."""
    label = js_str(data.get("label")).strip()
    input_id = data.get("id") or None
    _check_label(label, input_id, registry)
    kind = data.get("kind")
    if kind == "local":
        return {"id": LOCAL_CONNECTION_ID, "kind": "local", "label": label}
    if input_id == LOCAL_CONNECTION_ID:
        raise ConnectionStoreError('The id "local" is reserved for the local connection.')
    cid = input_id or connection_id_for_label(label, [c["id"] for c in registry["connections"]])
    data = {**data, "_label": label}
    if kind == "ssh":
        return _ssh_input(data, cid, registry)
    if kind in ("remote", "cloud"):
        return _remote_input(data, cid, kind, registry)
    raise ConnectionStoreError(f"Unknown connection kind: {js_str(kind) if kind is not None else 'undefined'}")


def merge_connection_input(data: dict, existing: Optional[dict]) -> dict:
    """``mergeConnectionInput``: fields the editor does not carry survive an edit."""
    if not existing or existing.get("kind") != data.get("kind"):
        return data
    merged = dict(data)

    def inherit(field: str) -> None:  # an absent key is TS `undefined`
        if field not in merged and field in existing:
            merged[field] = existing[field]

    for field in ("url", "authMode", "org"):
        inherit(field)
    if data.get("kind") in ("remote", "cloud") and ("url" not in data or _same_url(data["url"], existing.get("url"))):
        inherit("name")
    for field in ("host", "keyPath", "remoteHermesPath", "remoteProfile", "headers"):
        inherit(field)
    if "host" not in data or not js_string(data["host"]).strip():
        inherit("user")
        inherit("port")
    return merged


def _same_url(a: Any, b: Any) -> bool:
    return normalize_remote_base_url(a) == normalize_remote_base_url(b)


def connection_dial_fields_changed(before: dict, after: dict) -> bool:
    if before.get("kind") != after.get("kind"):
        return True
    for field in ("url", "authMode", "org", "host", "user", "port", "keyPath", "remoteHermesPath", "remoteProfile"):
        if before.get(field) != after.get(field):
            return True
    return before.get("token") != after.get("token") or before.get("headers") != after.get("headers")


def upsert_connection(registry: dict, entry: dict) -> dict:
    ids = [c["id"] for c in registry["connections"]]
    connections = ([entry if c["id"] == entry["id"] else c for c in registry["connections"]]
                   if entry["id"] in ids else [*registry["connections"], entry])
    return {**registry, "connections": connections}


def remove_connection(registry: dict, cid: str) -> dict:
    target = next((c for c in registry["connections"] if c["id"] == cid), None)
    if target is None:
        return registry
    if target["kind"] == "local":
        raise ConnectionStoreError("The local connection cannot be removed.")
    primary = LOCAL_CONNECTION_ID if registry["primary"] == cid else registry["primary"]
    return {**registry, "primary": primary,
            "lastUsed": primary if registry["lastUsed"] == cid else registry["lastUsed"],
            "connections": [c for c in registry["connections"] if c["id"] != cid]}


def _require(registry: dict, cid: str) -> None:
    if not any(c["id"] == cid for c in registry["connections"]):
        raise NoSuchConnection(f'No connection with id "{cid}".')


def set_primary_connection(registry: dict, cid: str) -> dict:
    _require(registry, cid)
    return {**registry, "primary": cid}


def set_last_used_connection(registry: dict, cid: str) -> dict:
    _require(registry, cid)
    return {**registry, "lastUsed": cid}


def set_connection_launch_mode(registry: dict, mode: str) -> dict:
    if mode not in ("last-used", "primary"):
        raise ConnectionStoreError(f'Unknown connection launch mode "{mode}".')
    return {**registry, "launchMode": mode}
