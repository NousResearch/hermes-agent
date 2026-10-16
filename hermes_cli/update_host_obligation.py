"""Host-scoped update→restart obligation for ``hermes update``.

Multiplex-only (Teknium ruling): exactly ONE ``hermes gateway run`` per host serves every
profile, so "this pull still owes the fleet a restart" is a property of the HOST, not of one
profile's ``HERMES_HOME``. The legacy ``$HERMES_HOME/fleet_restart_pending`` marker was
per-home: ``hermes -p coder update`` armed and cleared coder's copy while restarting the
SHARED process, and every other profile's CLI could neither see nor discharge that obligation
— it simply armed its own and re-killed the same host process.

The record therefore lives beside the host rendezvous record, in
:func:`gateway.host_rendezvous.host_state_dir` (``$HERMES_GATEWAY_LOCK_DIR`` else
``$XDG_STATE_HOME/hermes/gateway-locks``) — the one cross-profile, per-OS-user state root the
tree already has. It is written once per host and installation (keyed by the checkout), read by
every profile's CLI, and cleared once.

The same "one host process, not one per profile" identity is what
:func:`collapse_units_to_host_processes` applies to enumerated systemd units: leftover
per-profile ``hermes-gateway-<p>.service`` units on a multiplexed host all point at the same
live ``MainPID``, so restarting each one restarts the host process N times.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("hermes_cli.update_cmd")

#: The record before it carried an install key, beside ``host-gateway.json`` / ``host-serve.json``.
#: Still read, and cleared by a discharge that judged it, so a debt an older release armed survives
#: the upgrade; it names no install, so it keeps its old meaning: owed by whichever install reads it.
HOST_OBLIGATION_NAME = "host-update-restart.json"

#: The per-``HERMES_HOME`` breadcrumb an unwritable host record falls back to; every reader honours it.
PROFILE_MARKER_NAME = "fleet_restart_pending"

_RECORD_VERSION = 1


def host_obligation_path() -> Path:
    """This installation's host obligation record.

    Keyed by the checkout this module runs from: every profile of one install arms, reads and
    clears the same record, while a second install of the same OS user (another checkout, a ZIP
    install) owes its own, so its completed restart never erases this install's debt (review S3).
    """
    # Not ``gateway.host_rendezvous.host_state_dir``: ``gateway.status`` imports ``utils`` -> ruamel,
    # absent from the historical interpreter an old updater's takeover arms this record in; the
    # recovery module's stdlib copy of the rule is drift-tested against the gateway resolver.
    from hermes_cli.update_restart_recovery import _host_state_dir

    install = os.path.normcase(str(Path(__file__).resolve().parents[1]))
    key = hashlib.sha256(install.encode("utf-8")).hexdigest()[:16]
    return Path(_host_state_dir()) / f"{Path(HOST_OBLIGATION_NAME).stem}-{key}.json"


def _owed_path() -> Path:
    """The record a reader honours: this install's, else an older release's unkeyed one."""
    path = host_obligation_path()
    legacy = path.with_name(HOST_OBLIGATION_NAME)
    return legacy if not os.path.exists(path) and os.path.exists(legacy) else path  # never raises


def _write_record(path: Path, record: dict) -> None:
    """``utils.atomic_json_write(path, record, mode=0o600)`` in stdlib only (see above)."""
    replace_bytes(path, json.dumps(record, indent=2, ensure_ascii=False).encode("utf-8"))


def read_host_obligation() -> Optional[dict]:
    """The published obligation record, or ``None`` when absent/corrupt/foreign-versioned."""
    return _read_record(_owed_path())


def _read_record(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != _RECORD_VERSION:
        return None
    return payload


def host_obligation_present() -> bool:
    """True when the record FILE exists, parseable or not.

    Fail-closed: a corrupt record is an obligation whose terms are unknown, never a discharged
    one — the restart is still owed and the reader falls back to "no recorded inventory".
    """
    try:
        return _owed_path().is_file()
    except OSError:
        return False


def amend_host_obligation(**fields: Any) -> None:
    """Merge ``fields`` into the armed record (test/diagnostic surface). Never raises."""
    _update_record(lambda record: record.update(fields), "amend")


def write_host_obligation(
    *, expected_sha: str = "", runtimes: Optional[list] = None, profile: str = "", owner: str = ""
) -> bool:
    """Arm the host obligation. True when it was written. Never raises.

    Re-arming from a second profile for the SAME pulled SHA keeps the existing record (and its
    ``restarted`` proof) instead of resetting it: the host owes one restart, not one per profile.
    ``owner`` (an update run's commit-point token) joins the record's ``owners``: a run that
    fails before its move hands back only its own stake (``release_host_obligation``), never
    another run's debt for the same SHA. The first owner also stores what it found there
    (``found``), which the last owner to leave puts back; an owner retargeting to another SHA
    keeps that baseline (``_baseline_for``).
    """
    path = host_obligation_path()
    try:
        with _record_mutex(path):
            return _write_locked(path, expected_sha=expected_sha, runtimes=runtimes, profile=profile, owner=owner)
    except OSError as exc:  # the mutex busy or unopenable: the caller falls back to the per-home marker
        logger.debug("Could not lock the host update-restart obligation: %s", exc)
        return False


def arm_host_obligation(
    marker: Path, *, expected_sha: str = "", runtimes: Optional[list] = None, profile: str = "", owner: str = ""
) -> bool:
    """Arm the host record, else ``marker`` (the arming profile's per-home breadcrumb). Never raises.

    The one fallback for every arm (``hermes update``'s commit point, a historical updater's
    takeover). An unwritable host state dir (``HERMES_GATEWAY_LOCK_DIR`` on a read-only mount, a
    container UID that does not own ``$HOME``) must never disarm the obligation (#117275), but the
    marker is visible to the arming profile only: it settles the debt only on an install PROVEN to
    have no other profile, and an unreadable profile roster proves nothing (review S2 residual).
    Otherwise the marker is still left for this profile, the gap is said out loud, and the result
    is False: the commit point refuses to move; a takeover, whose tree already moved, keeps the warning.
    """
    if write_host_obligation(expected_sha=expected_sha, runtimes=runtimes, profile=profile, owner=owner):
        return True
    lines = [f"started={time.time()}", f"pid={os.getpid()}"] + ([f"expected_sha={expected_sha}"] if expected_sha else [])
    if runtimes is not None:
        lines.append("inventory=" + json.dumps({"version": 1, "runtimes": runtimes}))
    hidden = ""
    try:
        replace_bytes(marker, ("\n".join(lines) + "\n").encode("utf-8"))
    except OSError as exc:
        hidden = f" or {marker} ({exc})"
    if not hidden:
        try:
            if _named_profiles_exist():
                hidden = f" ({marker} hides it from the other profiles)"
        except OSError as exc:
            hidden = f" ({marker} may hide it from other profiles: the profile list is unreadable: {exc})"
    if not hidden:
        logger.warning("Host update-restart obligation (%s) is unwritable; armed the per-home marker %s instead.",
                       host_obligation_path(), marker)
        return True
    logger.error("Could not arm the update-restart obligation in %s%s; an interrupted update will not warn.",
                 host_obligation_path(), hidden)
    # ASCII: a historical updater's takeover may print through a legacy console code page.
    print("  Warning: could not record the pending gateway-restart obligation where every profile sees it "
          "(state dir not writable) - restart gateways with `hermes gateway restart` if this update is "
          "interrupted.", file=sys.stderr)
    return False


def _named_profiles_exist() -> bool:
    """True when the install has a live named profile beside ``default``, read from a COMPLETE roster.

    ``profiles.list_profile_names`` suppresses enumeration errors and reads an unreadable roster as
    "default only"; here any error listing the roster or reading an entry's identity raises instead.
    Same identity rule as ``hermes_constants.named_profile_is_live`` (stdlib only: see above), and
    the same tombstone as ``named_profile_is_deleted``: the canonical marker, so an alias of a
    deleted home is deleted too, read here with errors raised where ``is_file()`` would hide them.
    """
    from hermes_constants import _PROFILE_IDENTITY_MARKERS, PROFILE_ID_RE, get_default_hermes_root, profile_deletion_marker_path

    try:
        entries = list((get_default_hermes_root() / "profiles").iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return False
    for home in entries:
        if home.name == "default" or not PROFILE_ID_RE.match(home.name) or not stat.S_ISDIR(_mode(home, os.stat)):
            continue
        if not any((mode := _mode(home / name, os.lstat)) and not stat.S_ISDIR(mode)
                   for name in _PROFILE_IDENTITY_MARKERS):
            continue
        marker = profile_deletion_marker_path(home)
        if marker is None or not stat.S_ISREG(_mode(marker, os.stat)):
            return True
    return False


def _mode(path: Path, probe: Callable[[Path], os.stat_result]) -> int:
    """``probe(path).st_mode``; 0 when absent. Any other error raises: unreadable is unknown, not absent."""
    try:
        return probe(path).st_mode
    except (FileNotFoundError, NotADirectoryError):
        return 0


def _write_locked(path: Path, *, expected_sha: str, runtimes: Optional[list], profile: str, owner: str) -> bool:
    existing = _read_record(path)
    try:
        found = _baseline_for(owner, existing, path) if owner else None
    except OSError as exc:  # unreadable is not absent: a guessed ``found`` would delete it later
        logger.debug("Could not read the host update-restart obligation: %s", exc)
        return False
    if existing is not None and expected_sha and existing.get("expected_sha") == expected_sha:
        # Same pull, second profile: the host owes ONE restart, so keep the standing record (and
        # any proof that the restart already happened) rather than resetting it. A later arm that
        # carries the owed inventory still upgrades it — an inventory-less record owes no set.
        fields: dict[str, Any] = {}
        owners = _owners(existing)
        if owner and owner not in owners:
            fields["owners"] = [*owners, owner]
            if not owners:
                fields["found"] = found
        inventory = {"version": 1, "runtimes": runtimes}
        if runtimes is not None and existing.get("inventory") != inventory:
            fields["inventory"] = inventory
        if not fields:
            return True
        try:
            _write_record(path, {**existing, **fields})
        except Exception as exc:  # health: allow BLE001 -- never raises: the caller falls back to the per-home marker
            logger.debug("Could not amend host update-restart obligation: %s", exc)
            return False
        return True
    payload: dict[str, Any] = {
        "version": _RECORD_VERSION,
        "started": time.time(),
        "pid": os.getpid(),
        "armed_by_profile": profile or "",
        "expected_sha": expected_sha or "",
    }
    if owner:
        payload["owners"] = [owner]
        payload["found"] = found
    if runtimes is not None:
        payload["inventory"] = {"version": 1, "runtimes": runtimes}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_record(path, payload)
    except Exception as exc:  # health: allow BLE001 -- never raises: the caller falls back to the per-home marker
        logger.debug("Could not write host update-restart obligation: %s", exc)
        return False
    return True


@contextlib.contextmanager
def _record_mutex(path: Path):
    """Serialize every read-judge-write of the record across processes (``update_lock.marker_mutex``).

    The sidecar sits in a subdirectory: the state dir is the gateway lock dir, whose ``--replace``
    cleanup unlinks every top-level ``*.lock``, and a deleted sidecar splits the lock in two.
    """
    # One guard for every install's record (and the unkeyed one an older release still locks).
    guard = path.parent / f".{Path(HOST_OBLIGATION_NAME).stem}.mutex"
    guard.mkdir(parents=True, exist_ok=True)
    with _marker_mutex()(guard / "record"):
        yield


def _marker_mutex():
    """``update_lock.marker_mutex``. A historical updater imported ``update_lock`` before its pull and
    then lazily imports this pulled module, so its in-memory copy predates the mutex: load the pulled
    ``update_lock`` (stdlib-only) beside it, which locks the same sidecar every current process does."""
    import importlib.util
    import sys

    from hermes_cli import update_lock

    if hasattr(update_lock, "marker_mutex"):
        return update_lock.marker_mutex
    name = "hermes_cli._update_lock_pulled"
    pulled = sys.modules.get(name)
    if pulled is None:
        spec = importlib.util.spec_from_file_location(name, update_lock.__file__)
        pulled = importlib.util.module_from_spec(spec)
        sys.modules[name] = pulled  # before exec: dataclasses resolve the module by name
        spec.loader.exec_module(pulled)
    return pulled.marker_mutex


def _owners(record: dict) -> list[str]:
    owners = record.get("owners")
    return [str(o) for o in owners] if isinstance(owners, list) else []


def _baseline_for(owner: str, existing: Optional[dict], path: Path) -> Optional[str]:
    """What ``owner``'s release must put back (base64; ``None`` = absent): the record as it stood
    before ``owner``'s first stake in it (review R1).

    A run re-arms with one owner as it retargets (CP0 branch, origin pull, upstream fork ff), so
    "the record already has owners" never means "nothing was here": a new owner keeps the bytes
    there now, other owners' stakes included; an owner already in the record carries the baseline
    it saved then, or, when it shared the record with others, the shared record minus its own stake.
    """
    owners = _owners(existing or {})
    if owner not in owners:
        return _found_field(path)
    others = [o for o in owners if o != owner]
    if not others:
        return (existing or {}).get("found")
    shared = json.dumps({**(existing or {}), "owners": others}, indent=2, ensure_ascii=False)
    return base64.b64encode(shared.encode("utf-8")).decode("ascii")


def _found_field(path: Path) -> Optional[str]:
    """The record's current bytes (base64), ``None`` when absent: what a first owner puts back.
    Any other read error raises: custody is never guessed."""
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except FileNotFoundError:
        return None


def release_host_obligation(owner: str) -> None:
    """Hand back ``owner``'s stake in the record. When no other run still owes through it, the
    record goes back to what its first owner found (``found``; absent = unlinked).

    A record without ``owner`` was rewritten since (another run's newer pull): it is theirs and
    stays. Raises OSError when the record cannot be rewritten (the caller keeps the debt armed).
    Judged and rewritten under ``_record_mutex``, which every arm takes too, so no arm can land
    between the look and the write and be undone by it (kshitijk4poor F22/N05).
    """
    path = host_obligation_path()
    if not owner or owner not in _owners(_read_record(path) or {}):
        return  # nothing of ours to hand back: no lock taken, no state dir created
    with _record_mutex(path):
        _release_locked(path, owner)


def _release_locked(path: Path, owner: str) -> None:
    record = _read_record(path)
    owners = _owners(record) if record is not None else []
    if owner not in owners:
        return  # rewritten since the unlocked look (another run's arm): theirs now
    remaining = [o for o in owners if o != owner]
    if remaining:
        _write_record(path, {**record, "owners": remaining})
        return
    try:
        found = base64.b64decode(record["found"], validate=True) if record.get("found") is not None else None
    except (TypeError, ValueError):
        return  # a damaged ``found`` cannot be put back: the debt stays armed, never guessed away
    if found is None:
        path.unlink(missing_ok=True)
    else:
        replace_bytes(path, found)


def replace_bytes(path: Path, data: bytes) -> None:
    """``data`` at ``path`` as one durable record: a fresh 0600 ``mkstemp`` file (never written
    through a planted alias), fsynced, renamed over ``path``, then the rename itself made durable."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}_", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    if os.name != "nt":  # Windows opens no directory handle; NTFS journals the rename itself
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def clear_host_obligation() -> None:
    """Discharge the record this install's readers judged (``_owed_path``). Never raises.

    That is this install's own record whenever it has one. An older release's unkeyed record names
    no install: while this install has its own record, the unkeyed one may be another
    installation's standing debt that nothing here judged, so it stays (review S3 residual). It goes
    only when it was the record judged (this install armed none): it keeps its old meaning, owed by
    whichever install reads it, and an upgrade's own interrupted debt must stay dischargeable.
    """
    try:
        _owed_path().unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("Could not clear host update-restart obligation: %s", exc)


def obligation_fields() -> Optional[dict[str, str]]:
    """The obligation in the legacy ``key=value`` field shape, or ``None`` when unarmed.

    Keeps one parser for both sources: the fields a reader needs (``expected_sha``, the
    serialized ``inventory``) are identical whether they came from the host record or from an
    in-flight legacy per-home marker.
    """
    record = read_host_obligation()
    if record is None:
        return None
    fields = {"expected_sha": str(record.get("expected_sha") or "")}
    inventory = record.get("inventory")
    if inventory is not None:
        fields["inventory"] = json.dumps(inventory)
    return fields


def mark_host_restart_completed(sha: str) -> None:
    """Record that the host process was restarted onto ``sha``. Never raises."""
    _update_record(
        lambda record: record.update(restarted={"sha": sha or "", "pid": os.getpid(), "at": time.time()}),
        "stamp restart completion on")


def _update_record(change: Callable[[dict], None], what: str) -> None:
    """Read, ``change`` and rewrite an armed record under ``_record_mutex`` (an arm's new owner
    stake is never overwritten by a stale copy). Never raises; no record, no write."""
    path = _owed_path()
    if _read_record(path) is None:
        return
    try:
        with _record_mutex(path):
            record = _read_record(path)
            if record is None:
                return
            change(record)
            _write_record(path, record)
    except Exception as exc:  # health: allow BLE001 -- never raises: a lost stamp costs a repeat restart, never the debt
        logger.debug("Could not %s host update-restart obligation: %s", what, exc)


def host_restart_already_completed(sha: Optional[str]) -> bool:
    """True when THIS host obligation was already restarted onto ``sha``.

    The guard that makes the catch-up restart idempotent per host: a second profile running
    ``hermes update`` must attach to the first restart's outcome, never kill the shared
    multiplexer again.
    """
    record = read_host_obligation()
    if record is None or not sha:
        return False
    restarted = record.get("restarted")
    return isinstance(restarted, dict) and str(restarted.get("sha") or "") == sha


def collapse_units_to_host_processes(
    units: Iterable[str], main_pid: Callable[[str], int]
) -> tuple[list[str], dict[str, str]]:
    """Split enumerated units into ``(restart, {legacy_unit: covering_unit})``.

    Units resolving to the same live ``MainPID`` are ONE host process; restarting each of them
    restarts that process N times, which on a multiplexed host is an N-fold outage triggered by
    leftover per-profile units. A unit with no readable main PID (inactive, unprivileged scope)
    keeps its own restart: identity that cannot be proved is never collapsed away.
    """
    restart: list[str] = []
    covered: dict[str, str] = {}
    owner_by_pid: dict[int, str] = {}
    for unit in units:
        try:
            pid = int(main_pid(unit) or 0)
        except Exception:
            # Identity that cannot be proved keeps its own restart; a probe failure of any kind
            # must never abort the whole pass.
            pid = 0
        if pid <= 0:
            restart.append(unit)
            continue
        owner = owner_by_pid.get(pid)
        if owner is None:
            owner_by_pid[pid] = unit
            restart.append(unit)
        else:
            covered[unit] = owner
    return restart, covered
