"""Machine-wide model-residency admission: a pure coordination primitive.

The store arbitrates *who may hold one heavy residency slot per resource*
across every Hermes profile and controller on the machine. It lives in the
machine-owned Hermes root (``get_default_hermes_root()``, deliberately not
profile-scoped) at ``<root>/local_runtime/admission.sqlite3``, created
lazily on first write; reads never create files.

Coordination style mirrors ``model_settings.py``: every mutation runs one
``BEGIN IMMEDIATE`` transaction with a busy timeout and compare-and-swap on
a per-resource monotonic fencing epoch, so separate OS processes serialise
correctly. A newer controller instance always obtains a strictly larger
epoch; any mutating call carrying an older epoch raises :class:`FencedError`
and changes nothing.

Guarantees that shape the semantics:

* An expired deadline is NOT proof GPU memory was released. A new controller
  is refused until ``recover()`` runs; without ``verified_exit=True`` the
  record is quarantined (surfaced as unavailable), never silently adopted.
* A different OS-boot identity quarantines old-boot records; they are never
  adopted as live grants in the new boot.
* An ambiguous process identity (same pid, different caller-supplied
  creation identity) quarantines the record and refuses the grant with an
  actionable reason. This module never terminates, inspects, or otherwise
  touches OS processes, and never opens a network connection — verification
  of worker exit is the caller's responsibility, passed in as
  ``verified_exit``.
* Lease creation/release and job-terminal transitions are idempotent.

Existing llama.cpp boot locks and router ownership are untouched; this
module does not replace them.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NoReturn, TypeVar

_DB_RELATIVE = ("local_runtime", "admission.sqlite3")
_BUSY_TIMEOUT_MS = 30_000
STORE_SCHEMA_VERSION = 1
DEFAULT_LEASE_SECONDS = 30.0

#: Residency states of a record. ``loading``/``resident``/``draining`` are
#: live ownership; ``released`` is terminal-clean; ``quarantined`` means the
#: record is unavailable pending verified exit of the recorded process.
LIVE_RESIDENCY = ("loading", "resident", "draining")
RESIDENCY_STATES = (*LIVE_RESIDENCY, "released", "quarantined")

_T = TypeVar("_T")

# The heavy engines on a machine — the llama.cpp router and the media sidecar — are one residency slot, not two. They arrived at admission at
# different times under different names, and simply renaming the media key would let both boot side by side for anyone already holding a lease
# under the old one. So the names are canonicalised here, at the store's own single point where a resource becomes a row lookup: an existing
# record is found whichever name it was written under, and nothing has to be migrated.
SHARED_RESIDENCY_RESOURCE = "gpu:heavy"
LEGACY_MEDIA_RESIDENCY_RESOURCE = "media:studio"
_RESOURCE_ALIASES = {LEGACY_MEDIA_RESIDENCY_RESOURCE: SHARED_RESIDENCY_RESOURCE}


def canonical_resource(resource: str) -> str:
    """The name a resource's record lives under. Aliases share a slot; every other name is its own."""
    if not isinstance(resource, str):
        raise AdmissionError(f"resource must be a string, got {type(resource).__name__}")
    return _RESOURCE_ALIASES.get(resource, resource)



# ``owner_scope`` is stored and echoed back verbatim; this service never
# parses or interprets it. ``process_start_identity`` and ``boot_id`` are
# opaque strings supplied by the caller — the store fabricates neither.


class AdmissionError(Exception):
    """Base class for every admission-domain failure."""


class FencedError(AdmissionError):
    """A mutation carried a stale or invalidated grant; nothing changed."""

    def __init__(self, resource: str, grant_epoch: int, current_epoch: int,
                 why: str = "grant epoch is no longer current") -> None:
        super().__init__(
            f"{why} for {resource!r} (grant epoch {grant_epoch}, "
            f"current epoch {current_epoch})")
        self.resource = resource
        self.grant_epoch = grant_epoch
        self.current_epoch = current_epoch


@dataclass(frozen=True)
class Grant:
    """Handle returned by a successful acquire; carries the fencing epoch."""

    resource: str
    owner_scope: str
    controller_id: str
    pid: int
    process_start_identity: str
    engine: str
    model: str
    residency: str
    epoch: int
    boot_id: str
    deadline: float


@dataclass(frozen=True)
class Refused:
    """Result of a denied acquire. ``reason`` is one of: ``busy``,
    ``unload_required``, ``quarantined``, ``boot_mismatch``,
    ``stale_unverified``, ``identity_ambiguous``."""

    reason: str
    retry_after: float | None
    detail: str = ""


@dataclass(frozen=True)
class LeaseResult:
    resource: str
    lease_id: str
    kind: str
    state: str  # active | released | completed
    epoch: int


@dataclass(frozen=True)
class Record:
    """Full durable state of one resource, as persisted."""

    resource: str
    owner_scope: str
    controller_id: str
    pid: int
    process_start_identity: str
    engine: str
    model: str
    residency: str
    epoch: int
    boot_id: str
    deadline: float
    lease_seconds: float
    unload_requested: bool
    quarantine_reason: str | None
    updated_at: float
    active_leases: tuple[str, ...]


_CREATE_RESOURCES = """
CREATE TABLE IF NOT EXISTS admission_resources (
    resource               TEXT PRIMARY KEY,
    schema_version         INTEGER NOT NULL,
    epoch                  INTEGER NOT NULL,
    owner_scope            TEXT NOT NULL,
    controller_id          TEXT NOT NULL,
    pid                    INTEGER NOT NULL,
    process_start_identity TEXT NOT NULL,
    engine                 TEXT NOT NULL,
    model                  TEXT NOT NULL,
    residency              TEXT NOT NULL,
    boot_id                TEXT NOT NULL,
    deadline               REAL NOT NULL,
    lease_seconds          REAL NOT NULL,
    unload_requested       INTEGER NOT NULL DEFAULT 0,
    quarantine_reason      TEXT,
    updated_at             REAL NOT NULL
)
"""

_CREATE_LEASES = """
CREATE TABLE IF NOT EXISTS admission_leases (
    resource   TEXT NOT NULL,
    lease_id   TEXT NOT NULL,
    kind       TEXT NOT NULL,
    state      TEXT NOT NULL,
    epoch      INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (resource, lease_id)
)
"""

_ROW_FIELDS = ("resource", "schema_version", "epoch", "owner_scope",
               "controller_id", "pid", "process_start_identity", "engine",
               "model", "residency", "boot_id", "deadline", "lease_seconds",
               "unload_requested", "quarantine_reason", "updated_at")
_SELECT_ROW = (f"SELECT {', '.join(_ROW_FIELDS)} FROM admission_resources "
               "WHERE resource = ?")


def _fail_for_sqlite(exc: sqlite3.Error) -> NoReturn:
    raise AdmissionError(f"admission store sqlite failure: {exc}") from exc


def _safe_rollback(conn: sqlite3.Connection) -> None:
    try:
        conn.rollback()
    except sqlite3.Error:
        pass


def _resolve_now(now: float | None) -> float:
    """Monotonic seconds within the identified OS boot (caller supplies)."""
    return time.monotonic() if now is None else float(now)


def default_store_path() -> Path:
    """``<machine Hermes root>/local_runtime/admission.sqlite3``.

    Uses ``get_default_hermes_root()`` — the machine-owned location shared
    by every profile, same rule as ``bootstrap.models_dir()``.
    """
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root().joinpath(*_DB_RELATIVE)


class AdmissionStore:
    """Cross-process admission arbiter for heavy residency per resource.

    One instance is cheap and stateless: every call opens its own SQLite
    connection and runs inside ``BEGIN IMMEDIATE``. There are no module or
    instance globals, so any number of controllers may build their own.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        if root is None:
            from hermes_constants import get_default_hermes_root

            root = get_default_hermes_root()
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def path(self) -> Path:
        return self._root.joinpath(*_DB_RELATIVE)

    # -- connections ----------------------------------------------------------

    def _connect(self, create_parent: bool,
                 read_only: bool = False) -> sqlite3.Connection:
        if create_parent:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if read_only:
            uri = f"{self.path.absolute().as_uri()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True,
                                   timeout=_BUSY_TIMEOUT_MS / 1000.0)
        else:
            conn = sqlite3.connect(self.path,
                                   timeout=_BUSY_TIMEOUT_MS / 1000.0)
        try:
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            if not read_only:
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                except sqlite3.Error:
                    pass  # WAL is an optimisation; correctness holds without
        except sqlite3.Error:
            conn.close()
            raise
        return conn

    def _write(self, fn: Callable[[sqlite3.Connection], _T]) -> _T:
        conn = self._connect(create_parent=True)
        try:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(_CREATE_RESOURCES)
                conn.execute(_CREATE_LEASES)
                result = fn(conn)
                conn.execute("COMMIT")
                return result
            except sqlite3.DatabaseError as exc:
                _fail_for_sqlite(exc)
        except BaseException:
            _safe_rollback(conn)
            raise
        finally:
            conn.close()

    def _fetch_row(self, conn: sqlite3.Connection,
                   resource: str) -> dict | None:
        row = conn.execute(_SELECT_ROW, (resource,)).fetchone()
        return None if row is None else dict(zip(_ROW_FIELDS, row))

    def _active_leases(self, conn: sqlite3.Connection,
                       resource: str) -> tuple[str, ...]:
        rows = conn.execute(
            "SELECT lease_id FROM admission_leases "
            "WHERE resource = ? AND state = 'active' ORDER BY rowid",
            (resource,)).fetchall()
        return tuple(r[0] for r in rows)

    # -- status (never creates anything) ---------------------------------------

    def status(self, resource: str) -> Record | None:
        resource = canonical_resource(resource)
        if not self.path.exists():
            return None
        conn = self._connect(create_parent=False, read_only=True)
        try:
            outcome = self._status_read(conn, resource)
        finally:
            conn.close()
        return outcome

    @staticmethod
    def _status_read(conn: sqlite3.Connection,
                     resource: str) -> Record | None:
        try:
            row = conn.execute(_SELECT_ROW, (resource,)).fetchone()
            if row is None:
                return None
            active = tuple(
                r[0] for r in conn.execute(
                    "SELECT lease_id FROM admission_leases "
                    "WHERE resource = ? AND state = 'active' ORDER BY rowid",
                    (resource,)).fetchall())
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return None
            _fail_for_sqlite(exc)
        except sqlite3.DatabaseError as exc:
            _fail_for_sqlite(exc)
        return _record_from_row(dict(zip(_ROW_FIELDS, row)), active)

    # -- admission ---------------------------------------------------------------

    def acquire(self, resource: str, *, owner_scope: str, controller_id: str,
                pid: int, process_start_identity: str, engine: str,
                model: str, boot_id: str, now: float | None = None,
                lease_seconds: float = DEFAULT_LEASE_SECONDS,
                request_unload: bool = False) -> Grant | Refused:
        """Ask for the residency grant on ``resource``.

        At most one live owner exists per resource. A second controller is
        ``Refused`` with ``busy`` (or ``unload_required`` once a drain was
        requested and the incumbent holds no leases), never preempted: the
        incumbent must release first. Identity fields are caller-supplied
        and stored verbatim; nothing here validates them against the OS.
        """
        _require_text("resource", resource)
        resource = canonical_resource(resource)
        _require_text("owner_scope", owner_scope)
        _require_text("controller_id", controller_id)
        _require_text("process_start_identity", process_start_identity)
        _require_text("boot_id", boot_id)
        if not isinstance(pid, int) or pid <= 0:
            raise AdmissionError("pid must be a positive integer")
        if lease_seconds <= 0:
            raise AdmissionError("lease_seconds must be positive")
        stamp = _resolve_now(now)

        def tx(conn: sqlite3.Connection) -> Grant | Refused:
            row = self._fetch_row(conn, resource)
            if row is None:
                return self._insert_new(conn, resource, owner_scope,
                                        controller_id, pid,
                                        process_start_identity, engine,
                                        model, boot_id, stamp,
                                        lease_seconds, epoch=1)
            if row["residency"] == "quarantined":
                return Refused(
                    "quarantined", None,
                    f"record quarantined ({row['quarantine_reason']}); last "
                    f"holder controller {row['controller_id']!r} (pid "
                    f"{row['pid']}, identity {row['process_start_identity']!r})"
                    " — call recover(resource, now, boot_id) with "
                    "verified_exit=True only after verifying that process "
                    "exit and reconciling residency; never act on the "
                    "recorded pid otherwise")
            if row["residency"] == "released":
                return self._insert_new(conn, resource, owner_scope,
                                        controller_id, pid,
                                        process_start_identity, engine,
                                        model, boot_id, stamp,
                                        lease_seconds,
                                        epoch=row["epoch"] + 1)
            return self._live_incumbent(conn, row, owner_scope,
                                        controller_id, pid,
                                        process_start_identity, engine,
                                        model, boot_id, stamp, lease_seconds,
                                        request_unload)

        return self._write(tx)

    # -- acquire internals -------------------------------------------------------

    def _insert_new(self, conn: sqlite3.Connection, resource: str,
                    owner_scope: str, controller_id: str, pid: int,
                    ident: str, engine: str, model: str, boot_id: str,
                    now: float, lease_seconds: float, *,
                    epoch: int) -> Grant:
        deadline = now + lease_seconds
        conn.execute(
            "INSERT OR REPLACE INTO admission_resources (resource, "
            "schema_version, epoch, owner_scope, controller_id, pid, "
            "process_start_identity, engine, model, residency, boot_id, "
            "deadline, lease_seconds, unload_requested, quarantine_reason, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'loading', ?, "
            "?, ?, 0, NULL, ?)",
            (resource, STORE_SCHEMA_VERSION, epoch, owner_scope,
             controller_id, pid, ident, engine, model, boot_id, deadline,
             lease_seconds, now))
        conn.execute("UPDATE admission_leases SET state = 'released', "
                     "updated_at = ? WHERE resource = ? AND state = 'active'",
                     (now, resource))
        return Grant(resource=resource, owner_scope=owner_scope,
                     controller_id=controller_id, pid=pid,
                     process_start_identity=ident, engine=engine,
                     model=model, residency="loading", epoch=epoch,
                     boot_id=boot_id, deadline=deadline)

    def _live_incumbent(self, conn: sqlite3.Connection, row: dict,
                        owner_scope: str, controller_id: str, pid: int,
                        ident: str, engine: str, model: str, boot_id: str,
                        now: float, lease_seconds: float,
                        request_unload: bool) -> Grant | Refused:
        if row["boot_id"] != boot_id:
            return Refused(
                "boot_mismatch", None,
                f"record belongs to boot {row['boot_id']!r}, request cites "
                f"{boot_id!r}; old-boot records are never adopted — call "
                "recover(resource, now, boot_id) first")
        if (row["controller_id"] == controller_id
                and row["pid"] == pid and row["process_start_identity"] == ident):
            return _grant_from_row(row)  # idempotent re-acquire by the holder
        if row["pid"] == pid and row["process_start_identity"] != ident:
            # Same pid, different creation identity: pid reuse is suspected.
            # Quarantine and refuse; this module takes no action on any pid.
            conn.execute(
                "UPDATE admission_resources SET residency='quarantined', "
                "quarantine_reason='identity_ambiguous', updated_at=? "
                "WHERE resource=?", (now, row["resource"]))
            return Refused(
                "identity_ambiguous", None,
                f"pid {pid} carries identity {row['process_start_identity']!r} "
                f"in the record but {ident!r} in this request — pid reuse "
                "cannot be told apart; record quarantined, call "
                "recover(resource, now, boot_id) with verified_exit=True "
                "only after verifying the recorded process is gone")
        if now >= row["deadline"]:
            return Refused(
                "stale_unverified", None,
                f"incumbent deadline expired without proof residency memory "
                f"was released (holder {row['controller_id']!r}); call "
                "recover(resource, now, boot_id) — it quarantines unless "
                "verified_exit=True")
        if request_unload and row["residency"] != "draining":
            conn.execute(
                "UPDATE admission_resources SET residency='draining', "
                "unload_requested=1, updated_at=? WHERE resource=?",
                (now, row["resource"]))
            row = dict(row, residency="draining", unload_requested=1)
        draining = (row["residency"] == "draining"
                    or bool(row["unload_requested"]))
        if draining and not self._active_leases(conn, row["resource"]):
            return Refused(
                "unload_required", None,
                f"incumbent {row['controller_id']!r} was asked to drain and "
                "holds no leases; its engine must complete owned unload and "
                "release before a new grant is possible")
        return Refused("busy", max(0.0, row["deadline"] - now),
                       f"resource held by controller {row['controller_id']!r} "
                       f"until deadline {row['deadline']:.3f}")

    # -- fencing ------------------------------------------------------------------

    def _holder(self, conn: sqlite3.Connection, grant: Grant,
                boot_id: str | None = None) -> dict:
        """Return the row if ``grant`` is the current, un-fenced holder."""
        row = self._fetch_row(conn, grant.resource)
        if row is None:
            raise AdmissionError(f"no admission record for {grant.resource!r}")
        if (row["epoch"] != grant.epoch
                or row["controller_id"] != grant.controller_id):
            raise FencedError(grant.resource, grant.epoch, row["epoch"])
        if boot_id is not None and row["boot_id"] != boot_id:
            raise FencedError(grant.resource, grant.epoch, row["epoch"],
                              why="grant belongs to a different OS boot")
        if row["residency"] == "quarantined":
            raise FencedError(grant.resource, grant.epoch, row["epoch"],
                              why="record was quarantined by recovery")
        return row

    # -- grant lifecycle ------------------------------------------------------------

    def renew(self, grant: Grant, *, now: float | None = None,
              lease_seconds: float | None = None, residency: str | None = None,
              boot_id: str | None = None) -> Grant:
        """Extend the deadline (and optionally advance residency).

        Raises :class:`FencedError` for invalidated grants and
        :class:`AdmissionError` once the lease has already expired — an
        expired lease is recovered via ``recover()``, never renewed.
        """
        stamp = _resolve_now(now)
        if residency is not None and residency not in LIVE_RESIDENCY:
            raise AdmissionError(
                f"residency must be one of {LIVE_RESIDENCY}, got {residency!r}")

        def tx(conn: sqlite3.Connection) -> Grant:
            row = self._holder(conn, grant, boot_id=boot_id)
            if row["residency"] == "released":
                raise AdmissionError(
                    f"grant for {grant.resource!r} was already released")
            if stamp >= row["deadline"]:
                raise AdmissionError(
                    f"lease for {grant.resource!r} expired at "
                    f"{row['deadline']:.3f}; run recover() with "
                    "verified_exit=True after verifying worker exit")
            secs = row["lease_seconds"] if lease_seconds is None \
                else float(lease_seconds)
            if secs <= 0:
                raise AdmissionError("lease_seconds must be positive")
            deadline = stamp + secs
            new_res = row["residency"] if residency is None else residency
            conn.execute(
                "UPDATE admission_resources SET deadline=?, residency=?, "
                "lease_seconds=?, updated_at=? WHERE resource=?",
                (deadline, new_res, secs, stamp, grant.resource))
            return _grant_from_row(dict(row, deadline=deadline,
                                        residency=new_res,
                                        lease_seconds=secs))

        return self._write(tx)

    def release(self, grant: Grant, *, now: float | None = None) -> bool:
        """Terminal clean handback. Idempotent for the current holder."""
        stamp = _resolve_now(now)

        def tx(conn: sqlite3.Connection) -> bool:
            row = self._holder(conn, grant)
            if row["residency"] == "released":
                return True  # repeat release by the same holder: no-op
            conn.execute(
                "UPDATE admission_resources SET residency='released', "
                "unload_requested=0, quarantine_reason=NULL, updated_at=? "
                "WHERE resource=?", (stamp, grant.resource))
            conn.execute(
                "UPDATE admission_leases SET state='released', updated_at=? "
                "WHERE resource=? AND state='active'",
                (stamp, grant.resource))
            return True

        return self._write(tx)

    # -- leases ----------------------------------------------------------------------

    def add_lease(self, grant: Grant, lease_id: str,
                  kind: str = "job") -> LeaseResult:
        """Register a stream/job lease. Repeating returns the same result."""
        _require_text("lease_id", lease_id)
        if kind not in ("job", "stream"):
            raise AdmissionError("kind must be 'job' or 'stream'")

        def tx(conn: sqlite3.Connection) -> LeaseResult:
            row = self._holder(conn, grant)
            if row["residency"] == "released":
                raise AdmissionError(
                    f"grant for {grant.resource!r} was already released")
            if row["residency"] == "draining":
                raise AdmissionError(
                    f"{grant.resource!r} is draining; no new leases are "
                    "admitted while the incumbent unloads")
            existing = conn.execute(
                "SELECT kind, state, epoch FROM admission_leases "
                "WHERE resource=? AND lease_id=?",
                (grant.resource, lease_id)).fetchone()
            if existing is not None and existing[2] == grant.epoch:
                return LeaseResult(grant.resource, lease_id, existing[0],
                                   existing[1], grant.epoch)
            if existing is not None:
                # A row from an OLDER epoch is history, not this grant's
                # lease: a reused id must be tracked afresh or the drain
                # logic would see "no leases" while a job runs.
                conn.execute("DELETE FROM admission_leases WHERE resource=? "
                             "AND lease_id=?", (grant.resource, lease_id))
            conn.execute(
                "INSERT INTO admission_leases (resource, lease_id, kind, "
                "state, epoch, updated_at) VALUES (?, ?, ?, 'active', ?, ?)",
                (grant.resource, lease_id, kind, grant.epoch,
                 time.monotonic()))
            return LeaseResult(grant.resource, lease_id, kind, "active",
                               grant.epoch)

        return self._write(tx)

    def release_lease(self, grant: Grant, lease_id: str) -> LeaseResult:
        """Release a lease. Never raises on repetition or unknown ids."""

        def tx(conn: sqlite3.Connection) -> LeaseResult:
            row = self._holder(conn, grant)
            existing = conn.execute(
                "SELECT kind, state FROM admission_leases "
                "WHERE resource=? AND lease_id=? AND epoch=?",
                (grant.resource, lease_id, grant.epoch)).fetchone()
            if existing is None:
                return LeaseResult(grant.resource, lease_id, "job",
                                   "released", row["epoch"])
            kind, state = existing
            if state == "active":
                conn.execute(
                    "UPDATE admission_leases SET state='released', "
                    "updated_at=? WHERE resource=? AND lease_id=?",
                    (time.monotonic(), grant.resource, lease_id))
                state = "released"
            return LeaseResult(grant.resource, lease_id, kind, state,
                               row["epoch"])

        return self._write(tx)

    def complete_job(self, grant: Grant, lease_id: str) -> LeaseResult:
        """Terminal transition for a job lease; idempotent."""

        def tx(conn: sqlite3.Connection) -> LeaseResult:
            row = self._holder(conn, grant)
            existing = conn.execute(
                "SELECT kind, state FROM admission_leases "
                "WHERE resource=? AND lease_id=? AND epoch=?",
                (grant.resource, lease_id, grant.epoch)).fetchone()
            if existing is None:
                raise AdmissionError(
                    f"no lease {lease_id!r} on {grant.resource!r} under this "
                    "grant")
            kind, state = existing
            if kind != "job":
                raise AdmissionError(
                    f"lease {lease_id!r} is a {kind!r} lease; only jobs "
                    "complete")
            if state != "completed":
                conn.execute(
                    "UPDATE admission_leases SET state='completed', "
                    "updated_at=? WHERE resource=? AND lease_id=?",
                    (time.monotonic(), grant.resource, lease_id))
            return LeaseResult(grant.resource, lease_id, "job", "completed",
                               row["epoch"])

        return self._write(tx)

    # -- stale recovery -----------------------------------------------------------------

    def recover(self, resource: str, now: float, boot_id: str, *,
                verified_exit: bool = False,
                expected_epoch: int | None = None) -> str:
        """Reconcile a record whose holder is suspect. Returns the outcome.

        * no record at all -> ``'none'``
        * already released -> ``'released'`` (terminal no-op)
        * otherwise: without ``verified_exit=True`` the record becomes
          ``'quarantined'`` (reason ``boot_mismatch`` when the recorded OS
          boot differs, else ``stale_unverified``) — an expired deadline is
          never treated as proof residency memory was released. Only a
          caller-verified exit (``verified_exit=True``, supplied by a caller
          that independently confirmed the recorded process identity is gone
          and reconciled residency) transitions to ``'released'``, which is
          what unlocks the next grant at a higher epoch. Records from an old
          boot are never adopted as live grants.

        ``verified_exit`` is an attestation about ONE specific holder, so it
        must carry ``expected_epoch`` — the epoch of the grant whose process
        was verified gone. If the record has since moved to a different epoch
        (a newer holder was legitimately granted between the verification and
        this call) the attestation is stale: the outcome is ``'stale_epoch'``
        and nothing changes. Without that binding a late attestation could
        release a LIVE holder and allow a double residency.

        This function never touches processes; verification is an input.
        """
        stamp = _resolve_now(now)
        _require_text("boot_id", boot_id)
        if verified_exit and (isinstance(expected_epoch, bool)
                              or not isinstance(expected_epoch, int)):
            raise AdmissionError(
                "verified_exit=True requires expected_epoch: the epoch of the "
                "grant whose process exit was verified")

        resource = canonical_resource(resource)

        def tx(conn: sqlite3.Connection) -> str:
            row = self._fetch_row(conn, resource)
            if row is None:
                return "none"
            if verified_exit and row["epoch"] != expected_epoch:
                return "stale_epoch"
            if row["residency"] == "released":
                return "released"
            if row["residency"] == "quarantined" and not verified_exit:
                return "quarantined"
            if not verified_exit:
                reason = ("boot_mismatch" if row["boot_id"] != boot_id
                          else "stale_unverified")
                conn.execute(
                    "UPDATE admission_resources SET residency='quarantined', "
                    "quarantine_reason=?, updated_at=? WHERE resource=?",
                    (reason, stamp, resource))
                return "quarantined"
            conn.execute(
                "UPDATE admission_resources SET residency='released', "
                "unload_requested=0, quarantine_reason=NULL, updated_at=? "
                "WHERE resource=?", (stamp, resource))
            conn.execute(
                "UPDATE admission_leases SET state='released', updated_at=? "
                "WHERE resource=? AND state='active'", (stamp, resource))
            return "released"

        return self._write(tx)


# -- row mapping helpers ---------------------------------------------------------


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise AdmissionError(f"{name} must be a non-empty string")


def _grant_from_row(row: dict) -> Grant:
    return Grant(resource=row["resource"], owner_scope=row["owner_scope"],
                 controller_id=row["controller_id"], pid=row["pid"],
                 process_start_identity=row["process_start_identity"],
                 engine=row["engine"], model=row["model"],
                 residency=row["residency"], epoch=row["epoch"],
                 boot_id=row["boot_id"], deadline=row["deadline"])


def _record_from_row(row: dict,
                     active_leases: tuple[str, ...]) -> Record:
    return Record(resource=row["resource"], owner_scope=row["owner_scope"],
                  controller_id=row["controller_id"], pid=row["pid"],
                  process_start_identity=row["process_start_identity"],
                  engine=row["engine"], model=row["model"],
                  residency=row["residency"], epoch=row["epoch"],
                  boot_id=row["boot_id"], deadline=row["deadline"],
                  lease_seconds=row["lease_seconds"],
                  unload_requested=bool(row["unload_requested"]),
                  quarantine_reason=row["quarantine_reason"],
                  updated_at=row["updated_at"],
                  active_leases=active_leases)


__all__ = [
    "DEFAULT_LEASE_SECONDS",
    "LEGACY_MEDIA_RESIDENCY_RESOURCE",
    "LIVE_RESIDENCY",
    "RESIDENCY_STATES",
    "SHARED_RESIDENCY_RESOURCE",
    "AdmissionError",
    "AdmissionStore",
    "FencedError",
    "Grant",
    "LeaseResult",
    "Record",
    "Refused",
    "canonical_resource",
    "default_store_path",
]
