"""Kanban board lifecycle registry (P0-G-B1: compatibility containment).

Single external registry file that the gateway (and every manual dispatch
path) consults *before* touching a board's ``kanban.db``. It is the
compatibility-preserving alternative to a destructive Kanban root reset:
existing boards keep behaving exactly as they do today (``LEGACY_ACTIVE``),
while every brand-new board defaults to ``INACTIVE`` (fail closed) until an
operator explicitly activates it.

Design authority: ``kanban-containment-proposal.md`` (already reviewed and
PASSED) — this module implements that document's architecture/contracts/
race-handling/integrity-monitoring reasoning, but supersedes its "one
sidecar file per board directory" storage choice with a single external
registry file at ``<HERMES_HOME>/kanban-control/boards.json``, per a newer
KJ decision. Everything else (states, fail-closed semantics, CAS, alerting)
follows the design doc.

Deliberate deviation from a literal reading of the "fingerprint mismatch
excludes a board" instruction: the hot dispatch-eligibility path does NOT
recompute/compare a full-file SHA-256 on every tick. A ``LEGACY_ACTIVE``
production board's ``kanban.db`` mutates on essentially every dispatch (new
events, status changes), so a per-tick content-hash compare against a
migration-time snapshot would make every real board ineligible the moment
it is first touched post-rollout — defeating the entire "zero behavior
change for the 43 real boards" goal of this rollout. Fingerprint
verification is therefore reserved for: (a) migration snapshot generation
and validation, (b) the ``activate`` command's "requested board == actual
board" confirmation, and (c) the dry-run report. The routine per-tick
eligibility check is: registry loadable + board entry present + state in
the dispatch-eligible set. This is documented here and in the P0-G-B1
implementation report so a reviewer can evaluate the tradeoff explicitly.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import sqlite3
import stat as stat_module
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"

STATES = {"LEGACY_ACTIVE", "ACTIVE", "INACTIVE", "QUARANTINED", "ARCHIVED"}
DISPATCH_ELIGIBLE_STATES = {"LEGACY_ACTIVE", "ACTIVE"}
WRITE_FORBIDDEN_STATES = {"QUARANTINED", "ARCHIVED"}

# Allowed lifecycle transitions for the P0-G-B1 round. QUARANTINED and
# ARCHIVED are terminal-ish: leaving QUARANTINED requires the separate,
# harder-to-invoke "restore-approve" path (see restore_quarantined_board);
# leaving ARCHIVED is forbidden entirely this round.
_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "LEGACY_ACTIVE": {"ACTIVE", "INACTIVE", "QUARANTINED"},
    "ACTIVE": {"INACTIVE", "QUARANTINED"},
    "INACTIVE": {"ACTIVE", "QUARANTINED"},
    "QUARANTINED": set(),   # only via restore_quarantined_board()
    "ARCHIVED": set(),      # forbidden entirely this round
}

_ENV_CONTROL_ROOT = "HERMES_KANBAN_CONTROL_ROOT"
_REGISTRY_LOCK_TIMEOUT_SECONDS = 5.0
_REGISTRY_LOCK_POLL_SECONDS = 0.05

WRITER_ROLES = {
    "gateway-dispatcher",
    "manual-dispatch-cli",
    "worker-completion",
    "worker-block",
    "crash-reclaimer",
    "integrity-monitor",
    "backup-preserver",
    "migration",
}


class LifecycleRegistryError(Exception):
    """Registry cannot be trusted: missing, malformed, wrong version, unreadable.

    Any caller catching this MUST fail closed (treat every board as not
    dispatch-eligible) rather than falling back to LEGACY_ACTIVE/ACTIVE.
    """


class LifecycleTransitionError(Exception):
    """Requested state transition is not permitted."""


class LifecycleCasConflictError(Exception):
    """Registry `generation` changed between the writer's read and its write.

    No retry loop — the caller must re-read and re-decide.
    """


class LifecycleLockTimeoutError(Exception):
    """Could not acquire the registry write lock within the deadline.

    Unlike ``_cross_process_init_lock`` (which proceeds unlocked after a
    timeout because init work is idempotent), a registry write is NOT
    idempotent under a race — proceeding without the lock could silently
    drop a concurrent writer's update. Registry writes hard-fail on lock
    timeout instead.
    """


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _safe_control_root() -> Path:
    """Resolve the kanban-control root, constrained under HERMES_HOME.

    ``HERMES_KANBAN_CONTROL_ROOT`` may override the default location for
    tests/staging, but an arbitrary absolute path from the environment is
    never trusted unchecked: it must resolve to a location under the
    current ``HERMES_HOME`` root, else we fall back to the default.
    """
    home = get_hermes_home()
    default_root = home / "kanban-control"
    override = os.environ.get(_ENV_CONTROL_ROOT, "").strip()
    if not override:
        return default_root
    try:
        home_resolved = home.expanduser().resolve()
        candidate = Path(override).expanduser()
        # Resolve without requiring existence (parents may not exist yet).
        candidate_resolved = (
            candidate.resolve() if candidate.exists() else candidate.absolute()
        )
    except OSError:
        logger.warning(
            "%s=%r unusable; using default control root %s",
            _ENV_CONTROL_ROOT, override, default_root,
        )
        return default_root
    try:
        candidate_resolved.relative_to(home_resolved)
    except ValueError:
        logger.warning(
            "%s=%r escapes HERMES_HOME (%s); ignoring override, using default "
            "control root %s",
            _ENV_CONTROL_ROOT, override, home_resolved, default_root,
        )
        return default_root
    return candidate_resolved


def control_root() -> Path:
    return _safe_control_root()


def registry_path() -> Path:
    return control_root() / "boards.json"


def telemetry_path() -> Path:
    return control_root() / "telemetry.jsonl"


def alerts_path() -> Path:
    return control_root() / "alerts.jsonl"


def _reject_symlink(path: Path) -> None:
    """Refuse to use a registry/telemetry/alerts path that is a symlink.

    We check the path itself (not just its parents) via ``os.path.islink``,
    which does not follow the final component — this is deliberately NOT
    ``Path.exists()``/``resolve()`` based, since ``resolve()`` would follow
    the link and silently redirect us to whatever it points at.
    """
    if os.path.islink(path):
        raise LifecycleRegistryError(
            f"refusing to use symlinked registry path: {path}"
        )


def _ensure_control_root(path: Path) -> None:
    root = path.parent
    _reject_symlink(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Registry load (fail-closed)
# ---------------------------------------------------------------------------

@dataclass
class BoardEntry:
    state: str
    purpose: str = "unknown"
    actor: str = "operator"
    reason: str = ""
    updated_at: str = ""
    db_fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "purpose": self.purpose,
            "actor": self.actor,
            "reason": self.reason,
            "updated_at": self.updated_at,
            "db_fingerprint": self.db_fingerprint,
        }


def load_registry_raw() -> dict:
    """Load + validate the registry file. Raises LifecycleRegistryError on
    ANY problem (missing, symlink, unparseable JSON, wrong/missing schema
    version, malformed shape). Never returns a partially-trusted dict."""
    path = registry_path()
    try:
        _reject_symlink(path)
    except LifecycleRegistryError:
        raise
    if not path.exists():
        raise LifecycleRegistryError(f"registry file does not exist: {path}")
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LifecycleRegistryError(f"cannot read registry file {path}: {exc}") from exc
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LifecycleRegistryError(f"registry file {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LifecycleRegistryError(f"registry file {path} root is not an object")
    schema_version = data.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise LifecycleRegistryError(
            f"registry schema_version={schema_version!r} unsupported "
            f"(expected {SCHEMA_VERSION!r})"
        )
    if not isinstance(data.get("generation"), int):
        raise LifecycleRegistryError("registry missing integer 'generation'")
    boards = data.get("boards")
    if not isinstance(boards, dict):
        raise LifecycleRegistryError("registry missing 'boards' object")
    for slug, entry in boards.items():
        if not isinstance(entry, dict) or entry.get("state") not in STATES:
            raise LifecycleRegistryError(
                f"registry board entry {slug!r} malformed or missing valid 'state'"
            )
    return data


def load_registry() -> dict:
    """Public alias of :func:`load_registry_raw` (kept for readability at call sites)."""
    return load_registry_raw()


def try_load_registry() -> "tuple[Optional[dict], Optional[str]]":
    """Non-raising variant: returns (registry_dict_or_None, error_message_or_None)."""
    try:
        return load_registry_raw(), None
    except LifecycleRegistryError as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

@dataclass
class EligibilityResult:
    eligible: bool
    slug: str
    state: Optional[str]
    reason: str
    registry_ok: bool


def check_dispatch_eligibility(slug: str) -> EligibilityResult:
    """Fail-closed dispatch-eligibility check for one board.

    - Registry unreadable/malformed  -> ineligible, registry_ok=False (the
      caller should treat EVERY board as ineligible in this case, since a
      whole-registry failure is not a single-board problem).
    - Board missing from registry    -> ineligible ("no_registry_entry");
      this is the default for brand-new boards created after rollout.
    - Board state not in the
      dispatch-eligible set           -> ineligible ("state=<STATE>").
    - Otherwise                       -> eligible.
    """
    registry, err = try_load_registry()
    if registry is None:
        return EligibilityResult(
            eligible=False, slug=slug, state=None,
            reason=f"registry_unreadable: {err}", registry_ok=False,
        )
    entry = registry["boards"].get(slug)
    if entry is None:
        return EligibilityResult(
            eligible=False, slug=slug, state=None,
            reason="no_registry_entry (defaults to INACTIVE)", registry_ok=True,
        )
    state = entry["state"]
    if state not in DISPATCH_ELIGIBLE_STATES:
        return EligibilityResult(
            eligible=False, slug=slug, state=state,
            reason=f"state={state}", registry_ok=True,
        )
    return EligibilityResult(
        eligible=True, slug=slug, state=state, reason="ok", registry_ok=True,
    )


def check_write_allowed(slug: str, *, operation: str) -> EligibilityResult:
    """Guard for manual reclaim/unblock/complete-style single-task operations.

    Deliberately narrower than :func:`check_dispatch_eligibility`. Per the
    containment design's per-state semantics (design doc §11): QUARANTINED
    forbids retry/unblock/complete explicitly ("dispatch FORBIDDEN (gateway
    AND manual), retry/unblock/complete FORBIDDEN"), and ARCHIVED forbids
    "normal writes" entirely. INACTIVE only forbids *dispatch* — an
    operator recovering a task that's already running/blocked on an
    INACTIVE board (e.g. one just deactivated mid-flight) is not something
    the design doc asks to block, and a board with NO registry entry at
    all defaults to INACTIVE for dispatch purposes but is not thereby
    QUARANTINED, so these single-task recovery operations remain allowed.

    An entirely missing/unreadable registry is treated as "rollout not
    applied yet" for THIS check specifically (unlike dispatch eligibility,
    which fails closed on a missing registry) — deliberately, so that an
    install that hasn't run the P0-G-B1 migration yet keeps today's
    reclaim/unblock/complete behavior unchanged, matching the
    compatibility-preserving goal of this rollout for boards that predate
    it. Once a registry DOES exist, an explicit QUARANTINED/ARCHIVED entry
    is still honoured strictly. This is a deliberate, narrower fail-open
    for this one check only — flagged explicitly in the P0-G-B1
    implementation report as a scoping tradeoff versus a literal reading of
    "same lifecycle-eligibility check everywhere."
    """
    registry, err = try_load_registry()
    if registry is None:
        logger.warning(
            "kanban lifecycle: registry unreadable (%s) while checking %s "
            "for board %s; treating as pre-rollout and allowing (see "
            "check_write_allowed docstring)", err, operation, slug,
        )
        return EligibilityResult(eligible=True, slug=slug, state=None, reason="registry_unreadable_pre_rollout", registry_ok=False)
    entry = registry["boards"].get(slug)
    state = entry["state"] if entry else None
    if state in WRITE_FORBIDDEN_STATES:
        return EligibilityResult(
            eligible=False, slug=slug, state=state,
            reason=f"{operation} forbidden: state={state}", registry_ok=True,
        )
    return EligibilityResult(eligible=True, slug=slug, state=state, reason="ok", registry_ok=True)


# ---------------------------------------------------------------------------
# Fingerprint (read-only file hashing — never a read-write SQLite handle)
# ---------------------------------------------------------------------------

def compute_db_fingerprint(db_path: Path) -> str:
    """Read-only SHA-256 of a board DB file, as ``sha256:<hex>``.

    Uses plain file-level hashing (not a SQLite connection at all, ro or
    otherwise) so this is safe to call even against a CORRUPT board's file
    — no SQLite header/page validation is needed just to hash bytes.
    """
    h = hashlib.sha256()
    try:
        with open(db_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError as exc:
        raise LifecycleRegistryError(f"cannot hash {db_path}: {exc}") from exc
    return f"sha256:{h.hexdigest()}"


def path_hash(path: "str | Path") -> str:
    """SHA-256 of a path string — used in telemetry so raw filesystem paths
    (harmless here, but a good habit) aren't the only join key."""
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Registry write lock (reuses the fcntl/msvcrt pattern already used by
# hermes_cli.kanban_db, but hard-fails on timeout instead of proceeding
# unlocked — a registry write is not idempotent under a race the way
# CREATE TABLE IF NOT EXISTS is).
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _registry_write_lock(path: Path):
    _ensure_control_root(path)
    lock_path = path.with_name(path.name + ".lock")
    handle = lock_path.open("a+b")
    acquired = False
    try:
        deadline = time.monotonic() + _REGISTRY_LOCK_TIMEOUT_SECONDS
        try:
            import fcntl
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(_REGISTRY_LOCK_POLL_SECONDS)
        except ImportError:
            # No fcntl (non-POSIX) — best effort, treat as acquired. Registry
            # writes are rare/operator-driven so this degrade is acceptable.
            acquired = True
        if not acquired:
            raise LifecycleLockTimeoutError(
                f"could not acquire registry write lock {lock_path} within "
                f"{_REGISTRY_LOCK_TIMEOUT_SECONDS}s — hard-failing (no unlocked "
                f"fallback for registry writes)"
            )
        yield
    finally:
        try:
            if acquired:
                try:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except ImportError:
                    pass
        finally:
            handle.close()


def _atomic_write_json(path: Path, data: dict) -> None:
    _ensure_control_root(path)
    _reject_symlink(path)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".boards.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    # Re-validate what actually landed on disk.
    load_registry_raw()


def write_new_registry(boards: dict) -> dict:
    """Create the registry file from scratch (migration entrypoint only).

    Fails if a registry already exists — use ``apply_board_transition`` for
    updates to an existing registry.
    """
    path = registry_path()
    with _registry_write_lock(path):
        if path.exists():
            raise LifecycleRegistryError(
                f"registry already exists at {path}; refusing to overwrite "
                f"via write_new_registry (use apply_board_transition instead)"
            )
        data = {
            "schema_version": SCHEMA_VERSION,
            "generation": 1,
            "boards": boards,
        }
        _atomic_write_json(path, data)
        return data


def apply_board_transition(
    slug: str,
    new_state: str,
    *,
    actor: str,
    reason: str,
    purpose: Optional[str] = None,
    db_fingerprint: Optional[str] = None,
    expected_generation: Optional[int] = None,
    allow_from_quarantine: bool = False,
    updated_at: Optional[str] = None,
) -> dict:
    """Compare-and-swap update of one board's lifecycle state.

    ``expected_generation`` should be the generation the caller observed
    from its own prior ``load_registry()`` call. Immediately before
    writing, this function re-reads the generation under the write lock;
    if it has changed, it raises :class:`LifecycleCasConflictError` with NO
    retry loop (fail closed — the caller must re-read and re-decide).
    """
    if new_state not in STATES:
        raise LifecycleTransitionError(f"unknown target state {new_state!r}")
    if not actor or not actor.strip():
        raise LifecycleTransitionError("actor is required")
    if not reason or not reason.strip():
        raise LifecycleTransitionError("reason is required")

    path = registry_path()
    with _registry_write_lock(path):
        current = load_registry_raw()
        if expected_generation is not None and current["generation"] != expected_generation:
            raise LifecycleCasConflictError(
                f"registry generation changed: expected {expected_generation}, "
                f"found {current['generation']} — re-read and retry the "
                f"transition decision (no automatic retry)"
            )
        boards = current["boards"]
        existing = boards.get(slug)
        current_state = existing["state"] if existing else "INACTIVE"
        if current_state == "QUARANTINED" and not allow_from_quarantine:
            raise LifecycleTransitionError(
                "QUARANTINED -> * requires the restore-approve path "
                "(restore_quarantined_board), not a normal transition"
            )
        if current_state == "ARCHIVED":
            raise LifecycleTransitionError(
                "ARCHIVED -> * is forbidden entirely in this round"
            )
        if not allow_from_quarantine:
            allowed = _ALLOWED_TRANSITIONS.get(current_state, set())
            if new_state not in allowed and new_state != current_state:
                raise LifecycleTransitionError(
                    f"{current_state} -> {new_state} is not an allowed transition"
                )
        new_entry = {
            "state": new_state,
            "purpose": purpose if purpose is not None else (existing or {}).get("purpose", "unknown"),
            "actor": actor,
            "reason": reason,
            "updated_at": updated_at or _now_iso(),
            "db_fingerprint": db_fingerprint if db_fingerprint is not None else (existing or {}).get("db_fingerprint", ""),
        }
        boards[slug] = new_entry
        current["generation"] = current["generation"] + 1
        _atomic_write_json(path, current)
        return {"generation": current["generation"], "entry": new_entry}


def restore_quarantined_board(
    slug: str,
    *,
    actor: str,
    reason: str,
    repair_approved: bool,
    integrity_status: str,
    target_state: str = "INACTIVE",
    expected_generation: Optional[int] = None,
) -> dict:
    """Distinct, harder-to-invoke path out of QUARANTINED.

    Requires an explicit second confirmation (``repair_approved=True``) in
    addition to the normal actor/reason preconditions, and a passing Level-2
    integrity check (``integrity_status == "pass"``). This is deliberately
    NOT reachable via ``apply_board_transition``.
    """
    if not repair_approved:
        raise LifecycleTransitionError(
            "restore from QUARANTINED requires repair_approved=True "
            "(explicit second confirmation)"
        )
    if integrity_status != "pass":
        raise LifecycleTransitionError(
            f"restore from QUARANTINED requires a passing Level-2 integrity "
            f"check; got integrity_status={integrity_status!r}"
        )
    if target_state not in {"INACTIVE", "ACTIVE"}:
        raise LifecycleTransitionError(
            "restore from QUARANTINED may only land on INACTIVE or ACTIVE"
        )
    return apply_board_transition(
        slug, target_state, actor=actor, reason=reason,
        expected_generation=expected_generation, allow_from_quarantine=True,
    )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# Integrity precheck (Level 1: lightweight, read-only, pre-dispatch)
# ---------------------------------------------------------------------------

_EXPECTED_TABLES = {
    "tasks", "task_links", "task_comments", "task_events",
    "task_runs", "task_attachments", "kanban_notify_subs",
}


@dataclass
class IntegrityResult:
    passed: bool
    level: str
    reason: str
    details: dict = field(default_factory=dict)


def _corrupt_backup_markers(db_path: Path) -> list[str]:
    try:
        return sorted(
            str(p) for p in db_path.parent.glob(db_path.name + ".corrupt.*")
        )
    except OSError:
        return []


def integrity_precheck(db_path: Path) -> IntegrityResult:
    """Level 1: cheap, read-only precheck run before every dispatch of a
    LEGACY_ACTIVE/ACTIVE board. Must use a genuinely read-only connection
    (``mode=ro``), never write, never repair. On failure: deny dispatch,
    do not crash, do not auto-repair.
    """
    markers = _corrupt_backup_markers(db_path)
    if markers:
        return IntegrityResult(
            passed=False, level="1", reason="unresolved corrupt-backup marker present",
            details={"markers": markers},
        )
    if not db_path.exists():
        return IntegrityResult(passed=False, level="1", reason="db file does not exist")
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        try:
            row = conn.execute("PRAGMA quick_check(1)").fetchone()
            quick = row[0] if row else "unknown"
            if quick != "ok":
                return IntegrityResult(
                    passed=False, level="1", reason=f"quick_check={quick}",
                )
            tables = {
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            missing = _EXPECTED_TABLES - tables
            if missing:
                return IntegrityResult(
                    passed=False, level="1",
                    reason=f"missing expected tables: {sorted(missing)}",
                )
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        return IntegrityResult(passed=False, level="1", reason=f"sqlite error: {exc}")
    except OSError as exc:
        return IntegrityResult(passed=False, level="1", reason=f"cannot open: {exc}")
    return IntegrityResult(passed=True, level="1", reason="ok")


def integrity_full_check(db_path: Path) -> IntegrityResult:
    """Level 2: full ``PRAGMA integrity_check`` — read-only connection.

    Reserved for explicit activation, the manual ``integrity check``
    command, the post-Level-1-failure escalation, and a low-frequency
    periodic monitor. NOT run before every dispatch tick (too expensive).
    """
    level1 = integrity_precheck(db_path)
    if not level1.passed:
        return IntegrityResult(passed=False, level="2", reason=level1.reason, details=level1.details)
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30)
        try:
            rows = conn.execute("PRAGMA integrity_check").fetchall()
            texts = [r[0] for r in rows]
            if texts == ["ok"]:
                return IntegrityResult(passed=True, level="2", reason="ok")
            return IntegrityResult(
                passed=False, level="2", reason="integrity_check failed",
                details={"integrity_check": texts},
            )
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        return IntegrityResult(passed=False, level="2", reason=f"sqlite error: {exc}")
    except OSError as exc:
        return IntegrityResult(passed=False, level="2", reason=f"cannot open: {exc}")


# ---------------------------------------------------------------------------
# Telemetry (external — must survive the board DB's own potential corruption)
# ---------------------------------------------------------------------------

_TELEMETRY_MAX_BYTES = 5 * 1024 * 1024
_TELEMETRY_KEEP = 3
_disk_full_warned_at = 0.0


def _rotate_if_needed(path: Path, max_bytes: int, keep: int) -> None:
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
    except OSError:
        return
    for i in range(keep - 1, 0, -1):
        src = path.with_name(f"{path.name}.{i}")
        dst = path.with_name(f"{path.name}.{i + 1}")
        if src.exists():
            try:
                os.replace(src, dst)
            except OSError:
                pass
    try:
        os.replace(path, path.with_name(f"{path.name}.1"))
    except OSError:
        pass


def _append_jsonl(path: Path, record: dict) -> "tuple[bool, Optional[str]]":
    """Append one JSON line. Returns (ok, error). Never raises — a
    telemetry-write failure must never crash or block the primary dispatch
    path, but per design doc §12 it must also never be silent: the caller
    is expected to surface ``ok=False`` (e.g. into DispatchResult / CLI
    output / an alert) rather than swallow it.
    """
    global _disk_full_warned_at
    try:
        _ensure_control_root(path)
        _reject_symlink(path)
        _rotate_if_needed(path, _TELEMETRY_MAX_BYTES, _TELEMETRY_KEEP)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return True, None
    except OSError as exc:
        now = time.monotonic()
        if now - _disk_full_warned_at > 60:
            logger.error("kanban lifecycle telemetry write failed for %s: %s", path, exc)
            _disk_full_warned_at = now
        return False, str(exc)


def record_writer_event(
    *,
    writer_id: str,
    writer_role: str,
    board: str,
    db_path: "str | Path",
    operation_category: str,
    txn_id: str,
    phase: str,
    sqlite_result: str = "",
    busy_count: int = 0,
    retry_count: int = 0,
    tick_id: str = "",
    run_id: str = "",
    task_id: str = "",
    pid: Optional[int] = None,
    ppid: Optional[int] = None,
) -> "tuple[bool, Optional[str]]":
    if writer_role not in WRITER_ROLES:
        raise ValueError(f"unknown writer_role {writer_role!r}; must be one of {WRITER_ROLES}")
    record = {
        "ts": _now_iso(),
        "writer_id": writer_id,
        "writer_role": writer_role,
        "pid": pid if pid is not None else os.getpid(),
        "ppid": ppid if ppid is not None else os.getppid(),
        "board": board,
        "db_path_hash": path_hash(db_path),
        "txn_id": txn_id,
        "operation_category": operation_category,
        "phase": phase,  # BEGIN | COMMIT | ROLLBACK
        "sqlite_result": sqlite_result,
        "busy_count": busy_count,
        "retry_count": retry_count,
        "tick_id": tick_id,
        "run_id": run_id,
        "task_id": task_id,
    }
    return _append_jsonl(telemetry_path(), record)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def emit_alert(
    *,
    event_type: str,
    board: str,
    reason: str,
    detector: str,
    db_fingerprint: str = "",
    evidence_path: str = "",
    dispatch_stopped: bool = True,
    operator_action_required: bool = True,
) -> "tuple[bool, Optional[str]]":
    """Machine-readable alert emission. Best-effort file-based sink — this
    repo has no heavyweight alert bus, so a structured JSONL file is the
    lightest thing that satisfies "machine readable" without inventing new
    infra. Alert delivery failure must NEVER be treated as a reason to
    continue dispatching an unsafe board — callers must not gate their own
    fail-closed decision on this function's return value.
    """
    record = {
        "ts": _now_iso(),
        "event_type": event_type,
        "board": board,
        "db_fingerprint": db_fingerprint,
        "detector": detector,
        "reason": reason,
        "evidence_path": evidence_path,
        "dispatch_stopped": dispatch_stopped,
        "operator_action_required": operator_action_required,
    }
    ok, err = _append_jsonl(alerts_path(), record)
    if not ok:
        logger.error("kanban lifecycle: alert emission failed (%s): %s", event_type, err)
    return ok, err
