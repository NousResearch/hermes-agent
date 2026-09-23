"""Operator settings lock: named config paths no writer may change until it is unlocked.

The lock is enforced at the seam where a ``config.yaml`` document is realised on disk, not in each
front door: the whole-document primitives — :func:`hermes_cli.config.atomic_config_write` and the
two ruamel round-trip writers :func:`utils.atomic_roundtrip_yaml_save` /
:func:`utils.atomic_roundtrip_yaml_update` — each call :func:`check_config_write` against the
document they are about to replace. ``save_config``, ``hermes config set``/``unset``, the TUI's and
the desktop's model switch, the desktop's ``config.set`` RPC, the web Config page, the gateway
slash commands, the credential lifecycle, the ``hermes auth`` provider switch, ``hermes agent
import`` and the post-update restore all end in one of those — and a writer added tomorrow is
covered by using any of them. Writers with an EARLIER side effect (a ``.env`` rotation, an
``auth.json`` switch) ask first so a refusal never leaves a half-applied change.


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

So the stanza is parsed into exactly three states by :func:`lock_state`, and every decision reads
that instead of re-deriving "is it on?" from an ambient value:

* ``off`` — no stanza, or ``enabled`` is a recognised false spelling. Writes proceed.
* ``valid`` — ``enabled`` is a recognised true spelling, ``keys`` names at least one path, and a
  ``password``, if present, is a hash this build can actually verify.
* ``unusable`` — the operator asked for a lock this build cannot apply as written: an unrecognised
  ``enabled`` value, no usable ``keys``, a ``password`` that is not a supported hash, or a stanza
  that is not a mapping. Every guarded write is refused, **including while an unlock window is
  open**: a spec that cannot be normalised cannot be reasoned about, and a window opened against
  one cannot be shown to have authorised anything. Recovery is editing the root ``config.yaml``.

An unlock window is therefore authority over ONE lock generation, not over "the lock" in general.
The receipt carries a fingerprint of the exact normalised spec (patterns + password hash) it was
opened against, and lapses the moment that changes — so unlocking lock A and then replacing it with
lock B does not leave B unlocked with B's password never verified.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


def _read_root_yaml(root: Path) -> Optional[dict]:
    """The root config as raw YAML, {} when it names no lock, or ``None`` when it names one but
    does not parse to a mapping — read directly, never through the config loader.

    ``None`` is not "no lock": a typo anywhere in the root file would otherwise switch the lock off
    for every profile, whose own config.yaml still parses and writes normally.

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
        return None
    return data if isinstance(data, dict) else None


def lock_spec(home: Path | str | None = None) -> dict:
    """The ``settings_lock`` mapping from the ROOT config, or {} when absent/unusable."""
    section = (_read_root_yaml(hermes_root(home)) or {}).get(LOCK_SECTION)
    return section if isinstance(section, dict) else {}


# The YAML spellings an operator may reasonably write. Anything else is a typo, and a typo in a
# lock is not a decision to switch it off — see ``_enabled_state``.
_TRUE_WORDS = ("1", "true", "yes", "on")
_FALSE_WORDS = ("0", "false", "no", "off")


def _enabled_state(value: Any) -> str:
    """``"on"`` | ``"off"`` | ``"invalid"`` — never collapse an unrecognised value to "off".

    ``enabled: maybe`` used to be indistinguishable from an explicit disable, which made a typo in
    the one field that arms the lock silently unarm it. A value this function cannot recognise is
    reported as ``invalid`` so the caller can refuse writes instead of proceeding.
    """
    if value is None:
        return "off"
    if isinstance(value, bool):
        return "on" if value else "off"
    # bool is a subclass of int, so this only sees real integers.
    if isinstance(value, int):
        return "on" if value == 1 else ("off" if value == 0 else "invalid")
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return "on"
        if word in _FALSE_WORDS or not word:
            return "off"
    return "invalid"


def is_enabled(spec: dict) -> bool:
    """Whether the lock is armed. ``False`` for both "off" and "unusable" — callers deciding
    whether to REFUSE a write must use :func:`lock_state`, which separates the two."""
    return _enabled_state(spec.get("enabled")) == "on"


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


def _unusable_reason(spec: dict) -> str:
    """Why an armed lock cannot be applied as written, or ``""`` when it can."""
    raw = spec.get("keys")
    configured = [entry for entry in raw
                  if isinstance(entry, str) and entry.strip()] if isinstance(raw, list) else []
    if not configured:
        return f"{LOCK_SECTION}.keys is empty or not a list of paths"
    if _password_state(spec.get("password")) == "invalid":
        # A password this build cannot verify must never degrade to "no password required".
        return (f"{LOCK_SECTION}.password is not a hash this build can verify "
                f"(expected `{_HASH_SCHEME}$n$r$p$salt$hash`, as written by `hermes config lock`)")
    return ""


def spec_is_unusable(spec: dict) -> bool:
    """True when the operator asked for a lock this build cannot apply as written."""
    state = _enabled_state(spec.get("enabled"))
    if state == "invalid":
        return True
    return state == "on" and bool(_unusable_reason(spec))


@dataclass(frozen=True)
class LockState:
    """The parsed stanza: ``status`` is ``"off"``, ``"valid"`` or ``"unusable"``."""

    status: str
    spec: dict
    reason: str = ""


def lock_state(home: Path | str | None = None) -> LockState:
    """Parse the root stanza into its one authoritative state.

    Every policy decision reads this — never ``is_enabled`` alone, which cannot tell an explicit
    disable from a value it failed to recognise.
    """
    root = hermes_root(home)
    data = _read_root_yaml(root)
    if data is None:
        return LockState("unusable", {},
                         f"the root config.yaml mentions {LOCK_SECTION} but does not parse as a YAML mapping")
    section = data.get(LOCK_SECTION)
    if section is None:
        return LockState("off", {})
    if not isinstance(section, dict):
        return LockState("unusable", {},
                         f"{LOCK_SECTION} is not a mapping of settings (got {type(section).__name__})")
    enabled = _enabled_state(section.get("enabled"))
    if enabled == "off":
        return LockState("off", section)
    if enabled == "invalid":
        return LockState("unusable", section,
                         f"{LOCK_SECTION}.enabled is {section.get('enabled')!r}, which is neither "
                         f"{' / '.join(_TRUE_WORDS)} nor {' / '.join(_FALSE_WORDS)}")
    reason = _unusable_reason(section)
    return LockState("unusable", section, reason) if reason else LockState("valid", section)


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


def _parse_hash(stored: object) -> Optional[tuple[int, int, int, bytes, bytes]]:
    """``(n, r, p, salt, digest)`` for a hash this build can verify, else ``None``.

    The single parser for the stored form: :func:`verify_password` and the validity check in
    :func:`_unusable_reason` must agree on what "a usable password" means, or a hash one of them
    rejects would be treated as "no password set" by the other.
    """
    if not isinstance(stored, str):
        return None
    parts = stored.strip().split("$")
    if len(parts) != 6 or parts[0] != _HASH_SCHEME:
        return None
    try:
        n, r, p = int(parts[1]), int(parts[2]), int(parts[3])
        salt, digest = base64.b64decode(parts[4], validate=True), base64.b64decode(parts[5], validate=True)
    except Exception:
        return None
    if n < 2 or r < 1 or p < 1 or not salt or not digest:
        return None
    return n, r, p, salt, digest


def _password_state(stored: object) -> str:
    """``"absent"`` | ``"valid"`` | ``"invalid"``.

    A value that is present but not a hash we can verify is ``invalid`` — never ``absent``. The
    unlock doors only ask for a password when one is "required", so reporting a malformed hash as
    "no password configured" would open the window to anyone who asked.
    """
    if stored is None:
        return "absent"
    if isinstance(stored, str) and not stored.strip():
        return "absent"
    return "valid" if _parse_hash(stored) is not None else "invalid"


def verify_password(password: str, stored: object) -> bool:
    """Constant-time check against a stored hash. Any malformed/absent hash verifies as False."""
    parsed = _parse_hash(stored)
    if parsed is None or not password:
        return False
    n, r, p, salt, expected = parsed
    try:
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
                                dklen=len(expected))
    except Exception:
        return False
    return hmac.compare_digest(actual, expected)


def has_password(spec: dict) -> bool:
    """Whether unlocking this spec must present a password.

    Only ever consulted for a spec :func:`lock_state` called ``valid``, where a present password is
    a verifiable hash; a malformed one makes the whole spec unusable rather than passwordless.
    """
    return _password_state(spec.get("password")) == "valid"


# ── the unlock window ────────────────────────────────────────────────────────


def unlock_path(home: Path | str | None = None) -> Path:
    return hermes_root(home) / UNLOCK_FILENAME


def spec_fingerprint(spec: dict) -> str:
    """A stable id for one lock generation: its patterns and its password hash.

    An unlock window authorises changes to the lock it was opened against — not to whatever lock
    happens to be in the file later. Binding the receipt to this fingerprint means replacing the
    lock (new keys, new password, cleared and recreated) lapses the window instead of handing the
    new lock an authority nobody proved. The password hash is folded in as a SHA-256 input, so the
    receipt never carries the stored hash itself.
    """
    payload = json.dumps({"keys": sorted(locked_patterns(spec)),
                          "password": spec.get("password") if isinstance(spec.get("password"), str) else ""},
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unlock_expiry(home: Path | str | None = None, *, spec: dict | None = None) -> Optional[float]:
    """Expiry of the live unlock window for *spec*, or None when there is none.

    None when the receipt is missing, unreadable, lapsed, or was opened against a DIFFERENT lock
    generation — including a receipt written before this binding existed, which cannot be shown to
    belong to any spec and so is never honoured.
    """
    if spec is None:
        spec = lock_spec(home)
    try:
        data = json.loads(unlock_path(home).read_text(encoding="utf-8"))
        expires = float(data.get("expires_at") or 0)
    except (OSError, ValueError, TypeError):
        return None
    if expires <= time.time():
        return None
    return expires if data.get("lock") == spec_fingerprint(spec) else None


def is_unlocked(home: Path | str | None = None, *, spec: dict | None = None) -> bool:
    return unlock_expiry(home, spec=spec) is not None


def begin_unlock(home: Path | str | None = None, seconds: float = DEFAULT_UNLOCK_SECONDS,
                 *, spec: dict | None = None) -> float:
    """Open a time-boxed unlock window over ONE lock generation and return its expiry.

    The caller verifies the password first. *spec* is the lock that authority was proven against;
    the window lapses if it is replaced.
    """
    from utils import atomic_json_write

    if spec is None:
        spec = lock_spec(home)
    expires = time.time() + max(1.0, float(seconds))
    path = unlock_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, {"expires_at": expires, "lock": spec_fingerprint(spec)}, mode=0o600)
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
    state = lock_state(home)
    if state.status == "off":
        return
    if state.status == "unusable":
        # No unlock-window escape here on purpose: a stanza that cannot be normalised cannot be
        # reasoned about, and a window opened against one cannot be shown to have authorised
        # anything (it has no fingerprint to match). Recovery is editing the root config.yaml,
        # which this lock never claimed to stop.
        raise SettingsLockError(
            f"settings are locked but the {LOCK_SECTION} stanza cannot be applied as written: "
            f"{state.reason}. Fix {LOCK_SECTION} in the root config.yaml (or set "
            f"{LOCK_SECTION}.enabled: false there) — every config write is refused until you do.")
    spec = state.spec
    if is_unlocked(home, spec=spec):
        return
    offending = violations(before, after, spec)
    if not offending:
        return
    raise SettingsLockError(
        "settings are locked: " + ", ".join(offending)
        + ". Run `hermes config unlock` to open a time-boxed window"
        + (" (a password is required)." if has_password(spec) else "."),
        offending)


def check_config_write(config_path: Path | str, before: Any, after: Any) -> None:
    """The seam every ``config.yaml`` writer passes through: :func:`check_write` for the document at
    *config_path*, with the lock read from the root that owns it (``profiles/<name>`` → root).

    Called by the whole-document primitives — ``hermes_cli.config.atomic_config_write``,
    ``utils.atomic_roundtrip_yaml_save`` and ``utils.atomic_roundtrip_yaml_update`` — and by
    writers that must refuse BEFORE an earlier side effect (a ``.env`` rotation whose config.yaml
    mirror is locked, an ``auth.json`` provider switch).
    """
    check_write(before, after, Path(config_path).parent)


def describe(home: Path | str | None = None) -> dict:
    """Status for the CLI and the desktop: is it on, what is locked, is a window open."""
    state = lock_state(home)
    spec = state.spec
    expires = unlock_expiry(home, spec=spec) if state.status == "valid" else None
    return {
        "enabled": state.status != "off",
        "keys": list(locked_patterns(spec)),
        "password_required": has_password(spec),
        "unusable": state.status == "unusable",
        "reason": state.reason,
        "unlocked": expires is not None,
        "unlocked_until": expires,
    }


def locked_leaf_paths(config: Any, home: Path | str | None = None) -> tuple[str, ...]:
    """Which existing config leaves are currently locked — what a UI greys out."""
    state = lock_state(home)
    if state.status != "valid":
        return ()
    patterns = locked_patterns(state.spec)
    return tuple(sorted(path for path in _flatten(config or {})
                        if any(path_matches(path, pattern) for pattern in patterns)))
