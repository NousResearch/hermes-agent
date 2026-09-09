"""Operator settings lock: named config paths no writer may change until it is unlocked.

Every config write in the tree funnels through :func:`hermes_cli.config.save_config` — the CLI's
``hermes config set``, the desktop's ``config.set`` RPC, and the web Config page all end up there.
So the lock is enforced at that one chokepoint rather than in each writer, and a new writer added
tomorrow is covered without knowing this module exists.

The spec is read from the SHARED ROOT ``config.yaml`` only, never the active profile's: a
per-profile copy must not be able to unlock its own profile, for the same reason
``bots.force_private`` is root-only.

What this does and does not defend against, stated plainly because it matters:

* It stops a UI, an RPC client, ``hermes config set``, and an agent calling any of those from
  changing a locked setting. That covers accidents, well-meaning changes, and an agent that
  reaches for the documented command.
* It is NOT a security boundary against anyone who can write ``config.yaml`` directly. They can
  edit the locked value, or delete the lock stanza, in one line. Back it with file ownership
  (run the gateway as a user that cannot write its own config) if that is the threat.

Unlike the fail-OPEN Bot Mode flags, an unusable spec fails CLOSED. A typo in a visibility flag
must never quietly remove a working teammate; a typo in a lock must never quietly stop protecting
what the operator asked to protect. Refusing writes is recoverable by editing the file, which
already requires the access the lock does not claim to stop.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

LOCK_SECTION = "settings_lock"
UNLOCK_FILENAME = ".settings-unlock"
DEFAULT_UNLOCK_SECONDS = 900

# scrypt parameters. n=2**14 keeps an interactive unlock well under a second on the machines
# Hermes runs on while costing a brute-forcer real memory; r/p are the usual defaults.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1
_HASH_SCHEME = "scrypt"


class SettingsLockError(RuntimeError):
    """A write was refused. ``paths`` names the locked paths it would have changed."""

    def __init__(self, message: str, paths: tuple[str, ...] = ()):
        super().__init__(message)
        self.paths = paths


def hermes_root(home: Path | str | None = None) -> Path:
    """The shared root for a profile home or the root itself (``<root>/profiles/<name>`` → root)."""
    base = Path(home) if home is not None else Path(
        os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
    return base.parent.parent if base.parent.name == "profiles" else base


def _read_root_yaml(root: Path) -> dict:
    """The root config as raw YAML, or {} — read directly, never through the config loader.

    Deliberately not ``load_config()``: this runs inside ``save_config``'s ``_CONFIG_LOCK``, and
    re-entering the loader (which touches the dotenv/secrets locks) is the shape of the
    lock-order deadlock in #105405.
    """
    path = root / "config.yaml"
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    if LOCK_SECTION not in raw:  # cheap precheck: the dominant install has no lock at all
        return {}
    try:
        import yaml

        data = yaml.safe_load(raw)
    except Exception:
        logger.warning("settings lock: root config.yaml could not be parsed", exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def lock_spec(home: Path | str | None = None) -> dict:
    """The ``settings_lock`` mapping from the ROOT config, or {} when absent/unusable."""
    section = _read_root_yaml(hermes_root(home)).get(LOCK_SECTION)
    return section if isinstance(section, dict) else {}


def is_enabled(spec: dict) -> bool:
    value = spec.get("enabled")
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    return isinstance(value, str) and value.strip().lower() in ("1", "true", "yes", "on")


# The lock always protects itself. Without this, the front doors it guards would each be one
# `settings_lock.enabled false` away from turning it off, which is no lock at all.
SELF_PATTERN = f"{LOCK_SECTION}.*"


def locked_patterns(spec: dict) -> tuple[str, ...]:
    """The configured path patterns, trimmed, plus the implicit self-protecting one.

    Unusable entries are dropped rather than guessed at. ``settings_lock`` itself is always
    included while the lock is on, so disabling the lock needs the unlock window (and its
    password) like every other locked change.
    """
    raw = spec.get("keys")
    configured = tuple(entry.strip() for entry in raw
                       if isinstance(entry, str) and entry.strip()) if isinstance(raw, list) else ()
    if not configured:
        return ()
    return configured + (SELF_PATTERN,)


def spec_is_unusable(spec: dict) -> bool:
    """True when the operator asked for a lock but named nothing usable to lock."""
    raw = spec.get("keys")
    configured = [entry for entry in raw
                  if isinstance(entry, str) and entry.strip()] if isinstance(raw, list) else []
    return is_enabled(spec) and not configured


def path_matches(path: str, pattern: str) -> bool:
    """``a.b`` matches itself, and ``a.*`` matches ``a`` and everything beneath it."""
    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        return path == prefix or path.startswith(prefix + ".")
    return path == pattern


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Config as dotted leaf paths. Lists are leaves: order and length are part of the value."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, sub in value.items():
            out.update(_flatten(sub, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: value} if prefix else {}


def changed_paths(before: Any, after: Any) -> tuple[str, ...]:
    """Dotted paths whose value differs, in either direction (added, removed, or altered)."""
    flat_before, flat_after = _flatten(before or {}), _flatten(after or {})
    keys = set(flat_before) | set(flat_after)
    sentinel = object()
    return tuple(sorted(k for k in keys
                        if flat_before.get(k, sentinel) != flat_after.get(k, sentinel)))


def violations(before: Any, after: Any, spec: dict) -> tuple[str, ...]:
    """Locked paths this write would change. Empty when the lock is off or nothing locked moved."""
    if not is_enabled(spec):
        return ()
    patterns = locked_patterns(spec)
    if not patterns:
        return ()
    return tuple(path for path in changed_paths(before, after)
                 if any(path_matches(path, pattern) for pattern in patterns))


# ── password ─────────────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    """``scrypt$n$r$p$salt$hash`` — the only form ever written to disk."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                            n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    b64 = lambda raw: base64.b64encode(raw).decode("ascii")  # noqa: E731 — local shorthand
    return f"{_HASH_SCHEME}${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: object) -> bool:
    """Constant-time check against a stored hash. Any malformed/absent hash verifies as False."""
    if not isinstance(stored, str) or not password:
        return False
    parts = stored.split("$")
    if len(parts) != 6 or parts[0] != _HASH_SCHEME:
        return False
    try:
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt, expected = base64.b64decode(parts[4]), base64.b64decode(parts[5])
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
                                dklen=len(expected))
    except Exception:
        return False
    return hmac.compare_digest(actual, expected)


def has_password(spec: dict) -> bool:
    return isinstance(spec.get("password"), str) and bool(spec["password"].strip())


# ── the unlock window ────────────────────────────────────────────────────────


def unlock_path(home: Path | str | None = None) -> Path:
    return hermes_root(home) / UNLOCK_FILENAME


def unlock_expiry(home: Path | str | None = None) -> Optional[float]:
    """Expiry of the live unlock window, or None when there is none (or it lapsed)."""
    try:
        data = json.loads(unlock_path(home).read_text(encoding="utf-8"))
        expires = float(data.get("expires_at") or 0)
    except (OSError, ValueError, TypeError):
        return None
    return expires if expires > time.time() else None


def is_unlocked(home: Path | str | None = None) -> bool:
    return unlock_expiry(home) is not None


def begin_unlock(home: Path | str | None = None, seconds: float = DEFAULT_UNLOCK_SECONDS) -> float:
    """Open a time-boxed unlock window and return its expiry. Caller verifies the password first."""
    from utils import atomic_json_write

    expires = time.time() + max(1.0, float(seconds))
    path = unlock_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, {"expires_at": expires}, mode=0o600)
    return expires


def end_unlock(home: Path | str | None = None) -> None:
    try:
        unlock_path(home).unlink()
    except OSError:
        pass


# ── the gate ─────────────────────────────────────────────────────────────────


def check_write(before: Any, after: Any, home: Path | str | None = None) -> None:
    """Raise :class:`SettingsLockError` when this write would change a locked path.

    Never raises while an unlock window is live, and never for a write that leaves every locked
    path exactly as it was.
    """
    spec = lock_spec(home)
    if not is_enabled(spec) or is_unlocked(home):
        return
    if spec_is_unusable(spec):
        raise SettingsLockError(
            f"settings are locked but {LOCK_SECTION}.keys is empty or not a list, so the lock "
            f"cannot be applied — fix {LOCK_SECTION} in the root config.yaml, or set "
            f"{LOCK_SECTION}.enabled: false")
    offending = violations(before, after, spec)
    if not offending:
        return
    raise SettingsLockError(
        "settings are locked: " + ", ".join(offending)
        + ". Run `hermes config unlock` to open a time-boxed window"
        + (" (a password is required)." if has_password(spec) else "."),
        offending)


def describe(home: Path | str | None = None) -> dict:
    """Status for the CLI and the desktop: is it on, what is locked, is a window open."""
    spec = lock_spec(home)
    expires = unlock_expiry(home)
    return {
        "enabled": is_enabled(spec),
        "keys": list(locked_patterns(spec)),
        "password_required": has_password(spec),
        "unusable": spec_is_unusable(spec),
        "unlocked": expires is not None,
        "unlocked_until": expires,
    }


def locked_leaf_paths(config: Any, home: Path | str | None = None) -> tuple[str, ...]:
    """Which existing config leaves are currently locked — what a UI greys out."""
    spec = lock_spec(home)
    if not is_enabled(spec):
        return ()
    patterns = locked_patterns(spec)
    return tuple(sorted(path for path in _flatten(config or {})
                        if any(path_matches(path, pattern) for pattern in patterns)))
