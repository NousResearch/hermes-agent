"""Private local store for user-managed radio stream stations.

This module performs filesystem operations only; it never contacts a stream or
Sonos device. Station URLs are validated before they are persisted.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import secrets
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from hermes_constants import get_hermes_home
from integrations.file_locks import file_lock

MAX_STATIONS = 50
MAX_NAME_LENGTH = 80
MAX_URL_LENGTH = 2048

_SECRET_QUERY_KEY = re.compile(
    r"(?:token|secret|password|passwd|api.?key|access.?key|auth|credential|signature|private.?key|client.?secret)",
    re.IGNORECASE,
)


def _default_path() -> Path:
    return get_hermes_home() / "sonos-brain-state" / "radio-stations.json"


def _store_lock(path: Path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    lock_path = path.with_suffix(".lock")
    lock_path.touch(mode=0o600, exist_ok=True)
    lock_path.chmod(0o600)
    return file_lock(lock_path)


def _store_path(path: str | os.PathLike[str] | None) -> Path:
    return _default_path() if path is None else Path(path).expanduser()


def _valid_station_url(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_URL_LENGTH:
        raise ValueError("station URL must be a non-empty HTTP(S) URL within the length limit")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        # Accessing .port also catches malformed port strings.
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("invalid station URL") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc or not hostname:
        raise ValueError("station URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("station URL must not contain credentials")
    if parsed.fragment:
        raise ValueError("station URL must not contain a fragment")

    normalized_host = hostname.rstrip(".").lower()
    if normalized_host == "localhost" or normalized_host.endswith((".localhost", ".local", ".private")):
        raise ValueError("station URL host must be publicly routable")
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        address = None
    if address is not None and (not address.is_global or address.is_multicast):
        raise ValueError("station URL IP address must be globally routable")

    for key, _query_value in parse_qsl(parsed.query, keep_blank_values=True):
        if _SECRET_QUERY_KEY.search(key):
            raise ValueError("station URL must not contain secret-like query parameters")
    return value


def _validate_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("station name must be a string")
    if not value.strip() or len(value) > MAX_NAME_LENGTH or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("station name must be non-empty, printable, and within the length limit")
    name = value.strip()
    if not name:
        raise ValueError("station name must be non-empty, printable, and within the length limit")
    return name


def _load(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid radio station store") from exc
    if not isinstance(data, dict) or set(data) != {"stations"} or not isinstance(data["stations"], list):
        raise ValueError("invalid radio station store")
    stations = data["stations"]
    if len(stations) > MAX_STATIONS:
        raise ValueError("invalid radio station store: station limit exceeded")
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    seen_urls: set[str] = set()
    for item in stations:
        if not isinstance(item, dict) or set(item) != {"id", "name", "url"}:
            raise ValueError("invalid radio station store")
        station_id = item["id"]
        if not isinstance(station_id, str) or not station_id:
            raise ValueError("invalid radio station store")
        name = _validate_name(item["name"])
        url = _valid_station_url(item["url"])
        if name != item["name"] or station_id in seen_ids or name.casefold() in seen_names or url.casefold() in seen_urls:
            raise ValueError("invalid radio station store: duplicate or malformed station")
        seen_ids.add(station_id)
        seen_names.add(name.casefold())
        seen_urls.add(url.casefold())
    return [dict(item) for item in stations]


def _write(path: Path, stations: list[dict[str, str]]) -> None:
    parent = path.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent.chmod(0o700)
    payload = json.dumps({"stations": stations}, ensure_ascii=False, separators=(",", ":")) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def list_stations(path: str | os.PathLike[str] | None = None) -> list[dict[str, str]]:
    """Return all stations; an absent store is treated as an empty list."""
    return _load(_store_path(path))


def get_station(station_id: str, path: str | os.PathLike[str] | None = None) -> dict[str, str] | None:
    """Return a station by id, or ``None`` when no station has that id."""
    if not isinstance(station_id, str) or not station_id:
        raise ValueError("station id must be a non-empty string")
    return next((station for station in _load(_store_path(path)) if station["id"] == station_id), None)


def add_station(name: str, url: str, path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Validate and persist a station, returning its id, name, and stream URL."""
    station_name = _validate_name(name)
    station_url = _valid_station_url(url)
    target = _store_path(path)
    with _store_lock(target):
        stations = _load(target)
        if len(stations) >= MAX_STATIONS:
            raise ValueError("radio station limit reached")
        if any(item["name"].casefold() == station_name.casefold() for item in stations):
            raise ValueError("duplicate station name")
        if any(item["url"].casefold() == station_url.casefold() for item in stations):
            raise ValueError("duplicate station URL")
        station_id = secrets.token_urlsafe(6)
        while any(item["id"] == station_id for item in stations):
            station_id = secrets.token_urlsafe(6)
        station = {"id": station_id, "name": station_name, "url": station_url}
        _write(target, [*stations, station])
    return dict(station)


def remove_station(station_id: str, path: str | os.PathLike[str] | None = None) -> bool:
    """Remove a station by id; return False when the id is not present."""
    if not isinstance(station_id, str) or not station_id:
        raise ValueError("station id must be a non-empty string")
    target = _store_path(path)
    with _store_lock(target):
        stations = _load(target)
        remaining = [station for station in stations if station["id"] != station_id]
        if len(remaining) == len(stations):
            return False
        _write(target, remaining)
    return True
