"""On-disk state for sharing a gateway over tailcat (``<home>/tailcat/``).

Everything the CLI and the running backend must agree on lives here, so a code
minted by ``hermes share code`` in one process is redeemable by the backend in
another. Secrets are stored only as SHA-256 digests: a pending connection code
and a paired device's token are both unguessable random strings, so a digest is
enough to recognise them and useless to anyone who reads the files.

Files (all 0600):
  server.key.json  tailcat node key; its public half IS the stable address
  share.json       the share listener's loopback port + last published address
  devices.json     paired devices {id, name, token_sha256, created_at, last_seen_at}
  codes.json       unredeemed one-time codes {sha256, expires_at}
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_constants import get_routing_process_hermes_home
from utils import atomic_json_write, read_json_or_empty

CODE_SCHEME = "hermes-tailcat:"
CODE_TTL_S = 300
_NAME_MAX = 64
# Serializes read-modify-write of the JSON files within one process; the files
# are small and writes are atomic, so cross-process races lose at most a
# last_seen stamp, never a device or a code.
_LOCK = threading.RLock()


def share_dir(home: Optional[Path] = None) -> Path:
    """The launch profile owns the share: the listener serves the whole backend."""
    return (home or get_routing_process_hermes_home()) / "tailcat"


def _digest(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def address_fingerprint(address: str) -> str:
    """Short, distinguishing label for an address. Tailcat addresses share a long
    fixed prefix and suffix, so truncating the address itself labels them all alike."""
    return hashlib.sha256(address.encode("utf-8")).hexdigest()[:8] if address else ""


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, data, mode=0o600)


# ── Connection codes ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConnectionCode:
    address: str
    port: int
    secret: str

    def render(self) -> str:
        return f"{CODE_SCHEME}{self.address}:{self.port}:{self.secret}"


def parse_code(text: str) -> Optional[ConnectionCode]:
    """Inverse of :meth:`ConnectionCode.render`; None for anything malformed."""
    raw = (text or "").strip()
    if not raw.startswith(CODE_SCHEME):
        return None
    parts = raw[len(CODE_SCHEME):].split(":")
    if len(parts) != 3 or not all(parts):
        return None
    address, port, secret = parts
    if not address.startswith("tc") or not port.isdigit() or not 0 < int(port) < 65536:
        return None
    return ConnectionCode(address, int(port), secret)


def mint_code(address: str, port: int, *, home: Optional[Path] = None, now: Optional[float] = None) -> ConnectionCode:
    """Record a fresh single-use code; expired codes are pruned on the way."""
    now = time.time() if now is None else now
    code = ConnectionCode(address, int(port), secrets.token_urlsafe(24))
    path = share_dir(home) / "codes.json"
    with _LOCK:
        live = [c for c in read_json_or_empty(path).get("codes", []) if c.get("expires_at", 0) > now]
        live.append({"sha256": _digest(code.secret), "expires_at": now + CODE_TTL_S})
        _write(path, {"codes": live})
    return code


def consume_code(secret: str, *, home: Optional[Path] = None, now: Optional[float] = None) -> bool:
    """True exactly once for a live code; the code is gone afterwards either way."""
    now = time.time() if now is None else now
    want = _digest(secret or "")
    path = share_dir(home) / "codes.json"
    with _LOCK:
        codes = read_json_or_empty(path).get("codes", [])
        hit = next((c for c in codes if hmac.compare_digest(str(c.get("sha256", "")), want)), None)
        live = [c for c in codes if c is not hit and c.get("expires_at", 0) > now]
        if hit is not None or len(live) != len(codes):
            _write(path, {"codes": live})
    return hit is not None and hit.get("expires_at", 0) > now


# ── Paired devices ───────────────────────────────────────────────────────────


def _clean_name(name: str) -> str:
    text = " ".join(str(name or "").split())[:_NAME_MAX]
    return text or "Unnamed device"


def load_devices(home: Optional[Path] = None) -> list[dict]:
    return [d for d in read_json_or_empty(share_dir(home) / "devices.json").get("devices", []) if isinstance(d, dict)]


def pair_device(name: str, *, home: Optional[Path] = None, now: Optional[float] = None) -> tuple[dict, str]:
    """Add a device; returns (public record, bearer token). The token is shown once."""
    now = time.time() if now is None else now
    token = secrets.token_urlsafe(32)
    record = {
        "id": secrets.token_hex(6),
        "name": _clean_name(name),
        "token_sha256": _digest(token),
        "created_at": now,
        "last_seen_at": now,
    }
    path = share_dir(home) / "devices.json"
    with _LOCK:
        devices = load_devices(home)
        devices.append(record)
        _write(path, {"devices": devices})
    return public_device(record), token


def device_for_token(token: str, *, home: Optional[Path] = None) -> Optional[dict]:
    """The paired device owning ``token``, or None. Constant-time per record."""
    if not token:
        return None
    want = _digest(token)
    for device in load_devices(home):
        if hmac.compare_digest(str(device.get("token_sha256", "")), want):
            return device
    return None


def touch_device(device_id: str, *, home: Optional[Path] = None, now: Optional[float] = None,
                 min_interval_s: float = 60.0) -> None:
    """Stamp last_seen_at, at most once a minute per device (it is a list hint, not an audit log)."""
    now = time.time() if now is None else now
    path = share_dir(home) / "devices.json"
    with _LOCK:
        devices = load_devices(home)
        for device in devices:
            if device.get("id") == device_id:
                if now - float(device.get("last_seen_at") or 0) < min_interval_s:
                    return
                device["last_seen_at"] = now
                _write(path, {"devices": devices})
                return


def revoke_device(device_id: str, *, home: Optional[Path] = None) -> bool:
    path = share_dir(home) / "devices.json"
    with _LOCK:
        devices = load_devices(home)
        kept = [d for d in devices if d.get("id") != device_id]
        if len(kept) == len(devices):
            return False
        _write(path, {"devices": kept})
    return True


def public_device(record: dict) -> dict:
    return {k: record.get(k) for k in ("id", "name", "created_at", "last_seen_at")}


# ── Listener + identity ──────────────────────────────────────────────────────


def server_key_path(home: Optional[Path] = None) -> Path:
    return share_dir(home) / "server.key.json"


def load_share_state(home: Optional[Path] = None) -> dict:
    return read_json_or_empty(share_dir(home) / "share.json")


def save_share_state(state: dict, *, home: Optional[Path] = None) -> None:
    with _LOCK:
        _write(share_dir(home) / "share.json", state)


def forget_identity(home: Optional[Path] = None) -> None:
    """Regenerate: drop the node key (new address) and every paired device and code."""
    base = share_dir(home)
    with _LOCK:
        for name in ("server.key.json", "devices.json", "codes.json"):
            (base / name).unlink(missing_ok=True)
        state = load_share_state(home)
        state.pop("address", None)
        if state:
            _write(base / "share.json", state)
