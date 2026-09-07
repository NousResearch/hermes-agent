"""North Forge tier / pinned-edition access control.

North Forge ships as a *chassis* (the generic public agent) plus optional
*editions* — verticals such as Kyocera, Penny Pincher, Sales, Pine Barron Farms.
Mechanically **an edition is a Hermes profile**: its own ``SOUL.md``, ``skills/``,
``mcp.json``, ``config.yaml`` and state under ``<nf-root>/profiles/<edition>/``.
The North Forge generic chassis is the root profile (``"default"``).

A *deployed* drive carries a provisioning record written **once, at Setup Run**
by an admin (``scripts/nf-setup.ps1``), never by the recipient:

    <nf-root>/north-forge/provisioning.json   {schema,tier,pinned_edition,...,sig}
    <nf-root>/north-forge/.nf-key             32 random bytes, hex, mode 0600 — HMAC key
    <nf-root>/north-forge/.nf-admin           pbkdf2 hash of the admin passcode (gates re-provision)

``<nf-root>`` is :func:`hermes_constants.get_default_hermes_root` — the folder
``north-forge.cmd`` pins ``HERMES_HOME`` to. For a dev checkout / plain upstream
Hermes it is ``~/.hermes`` and no provisioning file exists, so this module is
inert.

Two tiers, no third:

* **Full** — Kenneth (admin) + trusted engineers. ``pinned_edition`` is only the
  default landing profile; every edition switch path stays open.
* **Basic** — everyone else. ``pinned_edition`` is the **only** reachable edition.
  Any attempt to select another — ``-p``/``--profile``, a hand-edited
  ``active_profile``, ``hermes profile use``, the dashboard, ``/edition`` — fails
  cleanly and ``HERMES_HOME`` never moves.

States:

* ``unprovisioned`` — no record. No lock. Behaves exactly like upstream Hermes.
* ``active`` — record present and its signature verifies. Tier logic applies.
* ``tampered`` — record present but the signature or key is missing/invalid.
  **Fail closed:** callers at process entry refuse to start.

The signature is HMAC-SHA256 with a key stored on the drive. That is
*tamper-evident*, not tamper-proof: it stops a casual edit of
``provisioning.json``; it is not a defence against an operator who will script a
re-sign. The real containment for proprietary edition content is that it is not
shipped to a Basic drive at all. See
``logs/ledger/decisions/DECISION-LOG.md`` DECISION-2026-09-07-003.

Import-light on purpose — :func:`hermes_cli.main._apply_profile_override` calls
this before argparse and before any hermes module is imported. Standard library
only; no hermes imports at module scope.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = 1

TIER_FULL = "full"
TIER_BASIC = "basic"
_VALID_TIERS = frozenset({TIER_FULL, TIER_BASIC})

STATE_UNPROVISIONED = "unprovisioned"
STATE_ACTIVE = "active"
STATE_TAMPERED = "tampered"

_DIR_NAME = "north-forge"
_RECORD_NAME = "provisioning.json"
_KEY_NAME = ".nf-key"
_ADMIN_NAME = ".nf-admin"

# Spellings that all mean "the generic North Forge chassis" == the root profile.
_ROOT_ALIASES = frozenset({"", "default", "north-forge", "north_forge", "northforge", "generic", "chassis"})

_PBKDF2_ITERS = 210_000

# Fields covered by the signature (order-independent; canonicalised with sort_keys).
_SIGNED_FIELDS = ("schema", "tier", "pinned_edition", "installed_editions",
                  "provisioned_at", "provisioned_by", "note")


class NfTierError(ValueError):
    """A tier policy refused an edition selection, or provisioning is tampered.

    Subclasses ``ValueError`` so the existing ``except ValueError`` arms in
    ``main._apply_profile_override`` / ``profiles.resolve_profile_env`` print the
    message and exit non-zero instead of falling through to a generic warning.
    """


@dataclass(frozen=True)
class Provisioning:
    state: str
    tier: str = ""
    pinned_edition: str = ""          # canonical; "" == the root chassis
    installed_editions: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)
    path: str = ""
    error: str = ""                  # populated when state == tampered

    @property
    def locked(self) -> bool:
        """True when this drive confines the agent to a single edition."""
        return self.state == STATE_ACTIVE and self.tier == TIER_BASIC

    @property
    def provisioned(self) -> bool:
        return self.state == STATE_ACTIVE


# --------------------------------------------------------------------------- paths


def normalize_edition(name: str | None) -> str:
    """Canonical edition id: lowercase/trim; every root alias collapses to ``"default"``.

    Mirrors ``hermes_cli.profiles.normalize_profile_name`` for named editions but
    is self-contained (no import) for the pre-argparse hot path.
    """
    s = ("" if name is None else str(name)).strip()
    if s.casefold() in _ROOT_ALIASES:
        return "default"
    return s.lower()


def nf_dir(root: Path | str | None = None) -> Path:
    """``<nf-root>/north-forge`` — where the provisioning record lives."""
    if root is not None:
        return Path(root) / _DIR_NAME
    try:
        from hermes_constants import get_default_hermes_root
        return get_default_hermes_root() / _DIR_NAME
    except Exception:
        # get_default_hermes_root is import-light and already used pre-argparse in
        # main.py; this fallback only fires if hermes_constants itself is broken.
        env = os.environ.get("HERMES_HOME", "").strip()
        base = Path(env) if env else Path.home() / ".hermes"
        return base / _DIR_NAME


def record_path(root: Path | str | None = None) -> Path:
    return nf_dir(root) / _RECORD_NAME


def key_path(root: Path | str | None = None) -> Path:
    return nf_dir(root) / _KEY_NAME


def admin_path(root: Path | str | None = None) -> Path:
    return nf_dir(root) / _ADMIN_NAME


# ------------------------------------------------------------------- sign / verify


def _canonical_signed_bytes(rec: dict[str, Any]) -> bytes:
    """Deterministic byte image of the signed subset of *rec*."""
    payload = {k: rec.get(k) for k in _SIGNED_FIELDS if k in rec}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _compute_sig(key: bytes, rec: dict[str, Any]) -> str:
    return hmac.new(key, _canonical_signed_bytes(rec), hashlib.sha256).hexdigest()


def _read_key(root: Path | str | None) -> bytes | None:
    try:
        raw = key_path(root).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    try:
        key = bytes.fromhex(raw)
    except ValueError:
        return None
    return key if len(key) >= 16 else None


# ------------------------------------------------------------------------- loading

_CACHE: dict[str, Provisioning] = {}


def load(root: Path | str | None = None, *, use_cache: bool = True) -> Provisioning:
    """Read + verify the provisioning record for *root* (default: the live nf-root)."""
    rp = record_path(root)
    cache_key = str(rp)
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]
    result = _load_uncached(rp, root)
    if use_cache:
        _CACHE[cache_key] = result
    return result


def _load_uncached(rp: Path, root: Path | str | None) -> Provisioning:
    if not rp.is_file():
        return Provisioning(state=STATE_UNPROVISIONED, path=str(rp))

    try:
        rec = json.loads(rp.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return Provisioning(state=STATE_TAMPERED, path=str(rp),
                            error=f"provisioning.json is unreadable ({exc})")
    if not isinstance(rec, dict):
        return Provisioning(state=STATE_TAMPERED, path=str(rp),
                            error="provisioning.json is not a JSON object")

    sig = rec.get("sig")
    key = _read_key(root)
    if not isinstance(sig, str) or not sig:
        return Provisioning(state=STATE_TAMPERED, path=str(rp),
                            error="provisioning.json has no signature")
    if key is None:
        return Provisioning(state=STATE_TAMPERED, path=str(rp),
                            error=f"signing key {key_path(root).name} is missing or invalid")
    if not hmac.compare_digest(sig, _compute_sig(key, rec)):
        return Provisioning(state=STATE_TAMPERED, path=str(rp),
                            error="provisioning.json signature does not match "
                                  "(record was modified after Setup Run)")

    tier = str(rec.get("tier", "")).strip().lower()
    if tier not in _VALID_TIERS:
        return Provisioning(state=STATE_TAMPERED, path=str(rp),
                            error=f"unknown tier {tier!r}")
    if int(rec.get("schema", 0)) != SCHEMA:
        return Provisioning(state=STATE_TAMPERED, path=str(rp),
                            error=f"unsupported provisioning schema {rec.get('schema')!r}")

    pin = normalize_edition(rec.get("pinned_edition"))
    installed_raw = rec.get("installed_editions") or []
    if not isinstance(installed_raw, list):
        installed_raw = []
    installed = tuple(sorted({normalize_edition(x) for x in installed_raw if str(x).strip()}))

    if tier == TIER_BASIC and pin == "default" and not str(rec.get("pinned_edition", "")).strip():
        # A Basic drive with no pin at all would confine the agent to the generic
        # chassis, which is a valid deployment (a plain free-chassis drive). Allow it.
        pass

    return Provisioning(state=STATE_ACTIVE, tier=tier, pinned_edition=pin,
                        installed_editions=installed, raw=rec, path=str(rp))


def clear_cache() -> None:
    _CACHE.clear()


# ------------------------------------------------------------------- policy checks


def default_edition(root: Path | str | None = None) -> str | None:
    """The edition to land on when nothing else is selected, or ``None`` for the root."""
    p = load(root)
    if p.state != STATE_ACTIVE:
        return None
    return None if p.pinned_edition == "default" else p.pinned_edition


def allowed_editions(root: Path | str | None = None) -> set[str] | None:
    """Editions this drive may run. ``None`` == no restriction (unprovisioned / Full)."""
    p = load(root)
    if p.locked:
        return {p.pinned_edition or "default"}
    return None


def edition_allowed(name: str | None, root: Path | str | None = None) -> bool:
    allowed = allowed_editions(root)
    return True if allowed is None else normalize_edition(name) in allowed


def assert_edition_allowed(name: str | None, *, action: str = "select",
                           root: Path | str | None = None) -> None:
    """Raise :class:`NfTierError` when *name* is not reachable on this drive.

    Returns cleanly (no-op) for an unprovisioned drive or a Full-tier drive.
    ``action`` is a verb for the message ("switch", "resolve", "start in").
    """
    p = load(root)
    if p.state == STATE_TAMPERED:
        raise NfTierError(_tampered_message(p))
    if not p.locked:
        return
    if normalize_edition(name) in {p.pinned_edition or "default"}:
        return
    pin_label = p.pinned_edition or "the North Forge chassis"
    raise NfTierError(
        f"This drive is provisioned Basic-tier, pinned to '{pin_label}'. "
        f"You cannot {action} the '{name}' edition here — it is not available on "
        f"this drive. (Full-tier drives can switch editions; this one cannot.)"
    )


def _tampered_message(p: Provisioning) -> str:
    return (
        f"North Forge provisioning at {p.path} is invalid or was modified after "
        f"Setup Run: {p.error}. The drive will not start until an admin repairs it "
        f"with  scripts\\nf-setup.ps1 --repair ."
    )


def enforce_startup_profile(requested: str | None, *, from_flag: bool,
                            root: Path | str | None = None) -> str | None:
    """Resolve the profile/edition to actually launch, applying tier policy.

    * ``requested`` — the edition asked for (``-p`` value, or a sticky
      ``active_profile`` entry, or ``None`` for a bare launch).
    * ``from_flag`` — True when ``requested`` came from an explicit ``-p`` /
      ``--profile`` on the command line (a deliberate ask ⇒ a forbidden one is a
      hard error), False when it came from the ``active_profile`` file (a stale
      sticky value ⇒ silently ignored, not an error).

    Returns the profile name to hand to ``resolve_profile_env`` (``None`` == the
    root chassis). Raises :class:`NfTierError` for a tampered record, or for an
    explicit ``-p`` naming a forbidden edition.
    """
    p = load(root)
    if p.state == STATE_TAMPERED:
        raise NfTierError(_tampered_message(p))
    if p.state == STATE_UNPROVISIONED:
        return requested
    # active
    if p.tier == TIER_FULL:
        if requested is None:
            return None if p.pinned_edition == "default" else p.pinned_edition
        return requested
    # tier == basic
    pin = None if p.pinned_edition == "default" else p.pinned_edition
    if requested is not None and normalize_edition(requested) != (pin or "default"):
        if from_flag:
            pin_label = pin or "the North Forge chassis"
            raise NfTierError(
                f"This drive is provisioned for '{pin_label}' only (Basic tier). "
                f"The '{requested}' edition is not available here."
            )
        # stale active_profile → ignore, fall through to the pin
    return pin


# --------------------------------------------------------------- provisioning (admin)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_key(root: Path | str | None = None) -> bytes:
    """Return the drive's HMAC key, creating a fresh 32-byte one on first call."""
    existing = _read_key(root)
    if existing is not None:
        return existing
    kp = key_path(root)
    kp.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    kp.write_text(key.hex() + "\n", encoding="utf-8")
    _chmod_600(kp)
    return key


def _chmod_600(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # Windows / filesystem without POSIX modes — the dir is under nf-root anyway


def write_provisioning(*, tier: str, pinned_edition: str, installed_editions: list[str] | None = None,
                       note: str = "", provisioned_by: str = "", root: Path | str | None = None,
                       overwrite: bool = False) -> Provisioning:
    """Write (and sign) ``provisioning.json``. Admin-side; called by ``nf-setup.ps1``."""
    tier = str(tier).strip().lower()
    if tier not in _VALID_TIERS:
        raise NfTierError(f"tier must be 'full' or 'basic', not {tier!r}")
    pin = normalize_edition(pinned_edition)
    if tier == TIER_BASIC and pin == "default" and str(pinned_edition).strip().casefold() not in _ROOT_ALIASES:
        raise NfTierError(f"pinned edition {pinned_edition!r} did not normalise to a usable name")

    rp = record_path(root)
    if rp.exists() and not overwrite:
        raise NfTierError(f"{rp} already exists — pass --force to re-provision this drive")

    installed = sorted({normalize_edition(x) for x in (installed_editions or []) if str(x).strip()})
    if tier == TIER_BASIC:
        # A Basic drive can only ever run its pin; record that honestly.
        installed = [pin]
    elif pin not in installed and pin != "default":
        installed.append(pin)
        installed = sorted(set(installed))

    rec: dict[str, Any] = {
        "schema": SCHEMA,
        "tier": tier,
        "pinned_edition": "" if pin == "default" and str(pinned_edition).strip().casefold() in _ROOT_ALIASES else pin,
        "installed_editions": installed,
        "provisioned_at": _now_iso(),
        "provisioned_by": provisioned_by or _whoami(),
        "note": note or "",
    }
    key = ensure_key(root)
    rec["sig"] = _compute_sig(key, rec)

    rp.parent.mkdir(parents=True, exist_ok=True)
    tmp = rp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(rp)
    clear_cache()
    return load(root, use_cache=False)


def _whoami() -> str:
    user = os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"
    host = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or ""
    return f"{host}\\{user}" if host else user


# ---- admin passcode (gates re-provisioning; NOT the signature key) -------------


def set_admin_passcode(passcode: str, root: Path | str | None = None, *, rotate: bool = False) -> None:
    ap = admin_path(root)
    if ap.exists() and not rotate:
        raise NfTierError(f"{ap} already exists — pass --rotate to change the admin passcode")
    if not passcode or len(passcode) < 6:
        raise NfTierError("admin passcode must be at least 6 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", passcode.encode("utf-8"), salt, _PBKDF2_ITERS)
    ap.parent.mkdir(parents=True, exist_ok=True)
    ap.write_text(json.dumps({
        "salt": salt.hex(), "hash": digest.hex(), "iterations": _PBKDF2_ITERS,
        "created_at": _now_iso(),
    }, indent=2) + "\n", encoding="utf-8")
    _chmod_600(ap)


def verify_admin_passcode(passcode: str, root: Path | str | None = None) -> bool:
    """True when *passcode* matches the stored hash. True also when no passcode is
    set yet (a fresh drive family — the first ``nf-setup`` run establishes it)."""
    ap = admin_path(root)
    if not ap.is_file():
        return True
    try:
        data = json.loads(ap.read_text(encoding="utf-8"))
        salt = bytes.fromhex(data["salt"])
        want = bytes.fromhex(data["hash"])
        iters = int(data.get("iterations", _PBKDF2_ITERS))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    got = hashlib.pbkdf2_hmac("sha256", (passcode or "").encode("utf-8"), salt, iters)
    return hmac.compare_digest(got, want)


def admin_passcode_is_set(root: Path | str | None = None) -> bool:
    return admin_path(root).is_file()


# ------------------------------------------------------------------------- __main__


def _print_human(p: Provisioning) -> None:
    print(f"state            : {p.state}")
    if p.state == STATE_TAMPERED:
        print(f"error            : {p.error}")
        return
    if p.state == STATE_UNPROVISIONED:
        print(f"record           : {p.path} (absent — drive is not provisioned)")
        return
    print(f"tier             : {p.tier}")
    print(f"pinned edition   : {p.pinned_edition or '(North Forge chassis / root)'}")
    print(f"installed        : {', '.join(p.installed_editions) or '(none recorded)'}")
    print(f"locked           : {'yes — only the pin is reachable' if p.locked else 'no — editions are switchable'}")
    print(f"record           : {p.path}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m hermes_cli.nf_tier",
        description="Inspect or write the North Forge tier / pinned-edition provisioning record.")
    ap.add_argument("--root", default=None,
                    help="nf-root override (default: hermes_constants.get_default_hermes_root())")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show", help="print the current provisioning state")

    v = sub.add_parser("verify", help="exit 0 if usable (active/unprovisioned), 2 if tampered")
    v.add_argument("--json", action="store_true", help="emit the state as JSON")

    pr = sub.add_parser("provision", help="write and sign provisioning.json (admin — Setup Run)")
    pr.add_argument("--tier", required=True, choices=[TIER_FULL, TIER_BASIC])
    pr.add_argument("--pin", required=True, help="pinned edition (profile name; 'default' = the chassis)")
    pr.add_argument("--installed", default="", help="comma-separated editions installed on this drive")
    pr.add_argument("--note", default="")
    pr.add_argument("--by", default="", help="who provisioned it (default: host\\user)")
    pr.add_argument("--force", action="store_true", help="overwrite an existing record")
    pr.add_argument("--passcode-stdin", action="store_true",
                    help="read the admin passcode from stdin and check it before writing")

    sa = sub.add_parser("set-admin", help="establish or rotate the admin passcode (reads it from stdin)")
    sa.add_argument("--rotate", action="store_true")

    args = ap.parse_args(argv)
    root = args.root

    if args.cmd == "show":
        _print_human(load(root, use_cache=False))
        return 0

    if args.cmd == "verify":
        p = load(root, use_cache=False)
        if args.json:
            print(json.dumps({"state": p.state, "tier": p.tier,
                              "pinned_edition": p.pinned_edition, "error": p.error}))
        else:
            _print_human(p)
        return 2 if p.state == STATE_TAMPERED else 0

    if args.cmd == "set-admin":
        import sys
        pc = sys.stdin.readline().rstrip("\n")
        try:
            set_admin_passcode(pc, root, rotate=args.rotate)
        except NfTierError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"admin passcode {'rotated' if args.rotate else 'set'} at {admin_path(root)}")
        return 0

    if args.cmd == "provision":
        import sys
        if args.passcode_stdin:
            pc = sys.stdin.readline().rstrip("\n")
            if not verify_admin_passcode(pc, root):
                print("error: admin passcode does not match this drive", file=sys.stderr)
                return 1
        installed = [s for s in (x.strip() for x in args.installed.split(",")) if s]
        try:
            p = write_provisioning(tier=args.tier, pinned_edition=args.pin,
                                   installed_editions=installed, note=args.note,
                                   provisioned_by=args.by, root=root, overwrite=args.force)
        except NfTierError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("wrote", record_path(root))
        _print_human(p)
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
