"""Append-only local action receipt ledger.

One receipt per actually-executed tool call, written to a dedicated SQLite
database. Update and delete triggers guard ordinary writes; configured retention
temporarily removes the delete trigger to prune old rows. Rows are chained by a
local SHA-256 hash so some bounded internal inconsistencies are detectable.

**Security caveat**: the unkeyed local SHA-256 chain detects internal partial
inconsistency but cannot prove originality against coordinated rewriting or
tail truncation. Any actor with write access to the database file can rebuild
a fully valid chain. The chain is a corruption detector, not a cryptographic
proof of custody or tamper-evidence against a privileged local attacker.

**Privacy design**: ``args_hash``, ``output_hash``, and body fingerprints are
HMAC-SHA-256 values keyed by a per-process ephemeral secret that is never
persisted or logged.  The chain still provides corruption detection (the same
key is used consistently within a process lifetime) but the stored values cannot
be used to confirm raw arguments or output content offline.  Cookie and
Set-Cookie headers are redacted at arbitrary nesting depth alongside other
credential-bearing keys.

**DB hardening**: on POSIX, the database is opened through an owner-only
``O_NOFOLLOW`` file descriptor so SQLite cannot follow a last-component
symlink substituted between validation and connection open. The descriptor
also pins the inode across a concurrent pathname replacement. The database
file and WAL/SHM sidecars are hardened to ``0o600`` on every writable open.
Writable creation passes mode ``0o600`` to ``os.open``, eliminating the initial
umask exposure window. Other platforms retain regular-file validation
and immediate owner-only permission hardening.

**Retention / size policy**: ``max_size_mb`` is a deterministic bound on live
data.  SQLite tracks deleted pages in its freelist; the live-data footprint is
``(page_count - freelist_count) * page_size``, which shrinks within a
transaction as rows are deleted (freed pages join the freelist immediately).
``_apply_retention`` loops — deleting batches of oldest rows, preserving at
least ``_SIZE_FLOOR_ROWS`` newest rows — until that metric is at or below
``limit_bytes`` or no more deletable rows exist.  VACUUM is never called inside
a transaction; the allocated *file* size does not shrink without an external
VACUUM, but the live-data measure that governs pruning does converge within the
same transaction.  If the table is already at the floor (``_SIZE_FLOOR_ROWS``
rows or fewer) and still over the limit, no further deletion is attempted; this
exceptional case (a single receipt larger than the entire configured budget) is
documented and tested explicitly.

The ledger is **default-off**. It is reachable only when the user opts in with

.. code-block:: yaml

    observability:
      action_receipts:
        enabled: true

in ``config.yaml``. With the key absent or false — and on any config read
error — :func:`is_enabled` returns ``False`` and nothing here opens a
database, builds an envelope, or touches the filesystem.

Optional retention / rotation settings (all default-off/unlimited):

.. code-block:: yaml

    observability:
      action_receipts:
        enabled: true
        max_rows: 100000          # hard row cap; oldest rows pruned first
        max_age_days: 90          # rows older than this are pruned on open
        max_size_mb: 512          # deterministic bound on live-data pages; see retention note above

The SQLite setup follows the same local-ledger conventions used elsewhere in
Hermes: WAL mode, ``BEGIN IMMEDIATE``, a module lock, and a schema table.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import stat
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from agent.task_envelope import (
    TaskEnvelope,
    build_shadow_envelope,
    canonical_hash,
    canonical_json,
    envelope_hash,
)
from hermes_constants import get_hermes_home


RECEIPT_SCHEMA_VERSION = 1

MAX_ARGS_SUMMARY_CHARS = 1000

# Boring defaults for retention — unset means unlimited.
_DEFAULT_MAX_ROWS: Optional[int] = None
_DEFAULT_MAX_AGE_DAYS: Optional[int] = None
_DEFAULT_MAX_SIZE_MB: Optional[int] = None

_DB_LOCK = threading.Lock()
_REDACTED = "[redacted]"
_BODY_FIELD_KEYS = frozenset(
    {
        "content",
        "code",
        "file_content",
        "information",
        "message",
        "new_string",
        "old_string",
        "patch",
        "prompt",
        "text",
    }
)

# Per-process ephemeral HMAC key — generated once at import time, never stored.
# Used for args_hash, output_hash, and body fingerprints so stored values cannot
# be used to confirm raw content offline (no dictionary/preimage attack).
_HMAC_KEY: bytes = secrets.token_bytes(32)


#: Argument keys whose values are treated as secrets wherever they appear,
#: at any nesting depth.
_SECRET_KEY_RE = re.compile(
    r"api[-_ ]?key|apikey|token|secret|password|passwd|pwd|auth|credential|"
    r"bearer|session[-_ ]?id_secret|private[-_ ]?key|"
    r"cookie|set[-_ ]?cookie",
    re.IGNORECASE,
)

#: Values that look like credentials even under an innocuous key.
_SECRET_VALUE_RE = re.compile(
    r"(?:bearer\s+\S+)|"
    r"(?:authorization\s*[:=]\s*(?:basic|bearer)?\s*\S+)|"
    r"(?:--?(?:api[-_]?key|token|secret|password|passwd|pwd|auth|credential)(?:=|\s+)\S+)|"
    r"(?:https?://[^\s/@:]+:[^\s/@]+@)|"
    r"(?:\b(?:sk|pk|ghp|gho|ghu|ghs|xox[baprs])-[A-Za-z0-9_\-]{8,})|"
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    re.IGNORECASE,
)


def _load_config() -> Any:
    """Read the user config, read-only.

    Isolated behind a module-level function so tests inject a config dict
    instead of touching the operator's real ``config.yaml``.
    """
    from hermes_cli.config import load_config_readonly

    return load_config_readonly()


def _receipts_section() -> dict[str, Any]:
    """Return the ``observability.action_receipts`` config dict, or empty."""
    try:
        config = _load_config()
        if not isinstance(config, dict):
            return {}
        section = config.get("observability")
        if not isinstance(section, dict):
            return {}
        receipts = section.get("action_receipts")
        if not isinstance(receipts, dict):
            return {}
        return receipts
    except Exception:
        return {}


def is_enabled() -> bool:
    """True only when ``observability.action_receipts.enabled`` is exactly true.

    Any missing key, non-mapping section, or config read failure yields
    ``False``: telemetry never fails open into being on.
    """
    try:
        return _receipts_section().get("enabled") is True
    except Exception:
        return False


def _retention_config() -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (max_rows, max_age_days, max_size_mb) from config, or defaults.

    All three default to ``None`` (unlimited). Values <= 0 are treated as
    unlimited so a typo of ``0`` is safe rather than destructive.
    """
    section = _receipts_section()

    def _pos_int(key: str) -> Optional[int]:
        val = section.get(key)
        try:
            n = int(val)
            return n if n > 0 else None
        except (TypeError, ValueError):
            return None

    return (
        _pos_int("max_rows"),
        _pos_int("max_age_days"),
        _pos_int("max_size_mb"),
    )


def default_db_path() -> Path:
    """The ledger's own database — never any existing Hermes store."""
    return get_hermes_home() / "action_receipts.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _receipt_hmac(data: bytes) -> str:
    """Return a per-process HMAC-SHA-256 hex digest.

    The key ``_HMAC_KEY`` is ephemeral and never stored, so the digest cannot
    be used offline to confirm raw content even if the DB is exfiltrated.
    """
    return hmac.new(_HMAC_KEY, data, hashlib.sha256).hexdigest()


def _body_fingerprint(value: Any) -> dict[str, Any]:
    """Represent quarantine-class bodies without retaining their contents.

    Returns ``bytes`` (length) and ``hmac`` (per-process keyed digest).  The
    HMAC cannot confirm raw content offline; it serves only as an internal
    consistency token within one process lifetime.
    """
    if isinstance(value, str):
        encoded = value.encode("utf-8")
    elif value is None or isinstance(value, (bool, int, float)):
        encoded = canonical_json(value).encode("utf-8")
    else:
        try:
            encoded = canonical_json(value).encode("utf-8")
        except Exception:
            return {"type": type(value).__name__, "hmac": _receipt_hmac(b"")}
    return {"bytes": len(encoded), "hmac": _receipt_hmac(encoded)}


def _scrub_envelope_dict(env_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the envelope dict with input_hash replaced by an HMAC.

    ``TaskEnvelope.input_hash`` is SHA-256(canonical_json(raw_args)), which is
    a deterministic unkeyed digest that allows offline confirmation of guessed
    argument content.  We replace it with an ephemeral HMAC before storing so
    the persisted value carries no offline-confirmable information about the
    original arguments.

    ``planner_hash`` and ``policy_hash`` follow the same pattern if non-None;
    we apply the same substitution for consistency.
    """
    out = dict(env_dict)
    for key in ("input_hash", "planner_hash", "policy_hash"):
        value = out.get(key)
        if value is not None and isinstance(value, str):
            out[key] = _receipt_hmac(value.encode("utf-8"))
    return out


def _redact(value: Any, *, depth: int = 0) -> Any:
    """Recursively strip secret-shaped keys and values.

    Redaction happens *before* anything is written, so a secret never reaches
    the database in the first place.
    """
    if depth > 12:
        return "[truncated]"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                key_text = key
            elif key is None:
                key_text = "null"
            elif isinstance(key, bool):
                key_text = "true" if key else "false"
            elif isinstance(key, (int, float)):
                key_text = canonical_json(key)
            else:
                key_text = f"[unsupported-key:{type(key).__name__}]"
            if _SECRET_KEY_RE.search(key_text):
                out[key_text] = _REDACTED
            elif key_text.lower() in _BODY_FIELD_KEYS:
                out[key_text] = _body_fingerprint(item)
            else:
                out[key_text] = _redact(item, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_redact(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(
            value,
            force=True,
            redact_url_credentials=True,
        )
        return _SECRET_VALUE_RE.sub(_REDACTED, redacted)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"[unsupported:{type(value).__name__}]"


def redacted_args_summary(args: Any) -> str:
    """A bounded, secrets-stripped rendering of the call arguments."""
    try:
        summary = canonical_json(_redact(args if args is not None else {}))
    except Exception:
        summary = "[unserializable]"
    if len(summary) > MAX_ARGS_SUMMARY_CHARS:
        return summary[: MAX_ARGS_SUMMARY_CHARS - 3] + "..."
    return summary


def _output_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, (bool, int, float, dict, list, tuple)):
        try:
            return canonical_json(_redact(output))
        except Exception:
            pass
    return f"[unsupported:{type(output).__name__}]"


def _harden_db_permissions(path: Path) -> None:
    """Set DB and any WAL/SHM sidecars to owner-read/write only (mode 0o600).

    Rejects symlinks and non-regular files before chmod so a substitution
    attack cannot redirect the chmod to another target. Failures are silently
    swallowed because receipts are optional observability and must never alter
    tool execution.
    """
    for suffix in ("", "-wal", "-shm"):
        try:
            target = path.parent / (path.name + suffix)
            try:
                st = os.lstat(target)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            os.chmod(target, stat.S_IRUSR | stat.S_IWUSR, follow_symlinks=False)
        except Exception:
            pass


def _assert_regular_file(path: Path) -> None:
    """Raise ``PermissionError`` if *path* is a symlink or non-regular file."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(st.st_mode):
        raise PermissionError(f"action_receipts: DB path is a symlink: {path}")
    if not stat.S_ISREG(st.st_mode):
        raise PermissionError(f"action_receipts: DB path is not a regular file: {path}")


def _open_db_file_0600(path: Path, *, readonly: bool = False) -> Optional[int]:
    """Securely open *path* and return a pinned descriptor on POSIX.

    ``O_NOFOLLOW`` closes the last-component symlink race between validation
    and SQLite's own open. ``fstat`` verifies the object actually opened, and
    ``fchmod`` hardens an existing permissive database before SQLite touches
    it. The caller keeps the descriptor alive for the connection lifetime.

    Windows has no portable descriptor path that SQLite can reopen, so it uses
    the pre-open regular-file check and atomic creation path instead.
    """
    _assert_regular_file(path)
    if os.name != "posix":
        if readonly:
            if not path.exists():
                raise FileNotFoundError(path)
            return None
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
        except FileExistsError:
            pass
        return None

    flags = os.O_RDONLY if readonly else os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise PermissionError(
                f"action_receipts: DB path is not a regular file: {path}"
            )
        if not readonly:
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _descriptor_path(fd: int) -> str:
    """Return the platform descriptor path SQLite can reopen."""
    proc_path = f"/proc/self/fd/{fd}"
    return proc_path if os.path.exists(proc_path) else f"/dev/fd/{fd}"


class _ReceiptConnection(sqlite3.Connection):
    """SQLite connection that owns the descriptor used for its secure open."""

    _action_receipts_fd: Optional[int] = None

    def close(self) -> None:
        fd = self._action_receipts_fd
        self._action_receipts_fd = None
        try:
            super().close()
        finally:
            if fd is not None:
                os.close(fd)




class ActionReceiptLedger:
    """Hash-chained ledger of executed actions.

    Table triggers reject ordinary row updates and deletes. Configured retention
    has the only built-in delete path: it temporarily removes the delete trigger
    inside the append transaction and records the resulting chain boundary.

    The local SHA-256 chain detects internal partial inconsistency (e.g. a
    corrupted write) but cannot prove originality against coordinated
    rewriting or tail truncation. See module docstring for the full caveat.
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()

    # ── connection / schema ────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        fd = _open_db_file_0600(self.db_path)
        sqlite_path: str | Path = _descriptor_path(fd) if fd is not None else self.db_path
        try:
            conn = sqlite3.connect(
                sqlite_path,
                isolation_level=None,
                factory=_ReceiptConnection,
            )
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise
        conn._action_receipts_fd = fd
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            _harden_db_permissions(self.db_path)
            return conn
        except BaseException:
            conn.close()
            raise

    def _connect_readonly(self) -> sqlite3.Connection:
        fd = _open_db_file_0600(self.db_path, readonly=True)
        sqlite_path: str | Path = _descriptor_path(fd) if fd is not None else self.db_path
        try:
            conn = sqlite3.connect(
                sqlite_path,
                isolation_level=None,
                factory=_ReceiptConnection,
            )
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise
        conn._action_receipts_fd = fd
        try:
            conn.execute("PRAGMA query_only=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            return conn
        except BaseException:
            conn.close()
            raise

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS action_receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                receipt_version INTEGER NOT NULL,
                receipt_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                action_id TEXT NOT NULL,
                session_id TEXT,
                task_id TEXT,
                tool_call_id TEXT,
                envelope_id TEXT,
                actor_lane TEXT NOT NULL,
                execution_surface TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                exit_status TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                cwd TEXT,
                args_hash TEXT NOT NULL,
                args_redacted_summary TEXT NOT NULL,
                output_hash TEXT,
                output_bytes INTEGER NOT NULL,
                envelope_hash TEXT,
                envelope_json TEXT,
                prev_receipt_hash TEXT,
                receipt_hash TEXT NOT NULL
            )
            """
        )
        # Append-only guards. RAISE(ABORT) surfaces to Python as
        # sqlite3.IntegrityError, so a tamper attempt is loud, not silent.
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS action_receipts_no_update
            BEFORE UPDATE ON action_receipts
            BEGIN
                SELECT RAISE(ABORT, 'action_receipts is append-only');
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS action_receipts_no_delete
            BEFORE DELETE ON action_receipts
            BEGIN
                SELECT RAISE(ABORT, 'action_receipts is append-only');
            END
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(RECEIPT_SCHEMA_VERSION),),
        )

    # ── retention / rotation ───────────────────────────────────────────
    def _apply_retention(self, conn: sqlite3.Connection) -> None:
        """Prune the oldest rows to satisfy configured retention bounds.

        Pruning is done inside the caller's BEGIN IMMEDIATE transaction via a
        temporary trigger-bypass: the append-only triggers block DELETE so we
        drop and recreate them around the prune, then restore them. This is
        the single legitimate delete path in the module.

        Bounds (from config):
        - ``max_rows``: keep at most this many rows (newest first).
        - ``max_age_days``: remove rows with ``created_at`` older than N days.
        - ``max_size_mb``: deterministic live-data bound.  The live-data
          footprint is ``(page_count - freelist_count) * page_size``; freed
          pages join SQLite's freelist immediately within the transaction, so
          this metric decreases with each DELETE batch without requiring a
          VACUUM.  Pruning loops — deleting batches of oldest rows, preserving
          at least ``_SIZE_FLOOR_ROWS`` newest rows — until the live-data
          footprint is at or below ``limit_bytes`` or no more rows are
          deletable.  If the floor is reached while still over-limit (a single
          receipt larger than the entire budget) the loop stops; that
          exceptional case is documented and tested.  VACUUM is never called
          inside a transaction; the on-disk allocated-file size does not shrink
          until an external VACUUM is run, but the live-data measure that
          governs pruning converges deterministically within this transaction.

        All three are best-effort. A failure here is swallowed.
        """
        max_rows, max_age_days, max_size_mb = _retention_config()
        if max_rows is None and max_age_days is None and max_size_mb is None:
            return

        try:
            # Temporarily disable the no-delete trigger for pruning.
            conn.execute("DROP TRIGGER IF EXISTS action_receipts_no_delete")
            try:
                if max_age_days is not None:
                    cutoff = (
                        datetime.now(timezone.utc) - timedelta(days=max_age_days)
                    ).isoformat()
                    conn.execute(
                        "DELETE FROM action_receipts WHERE created_at < ?", (cutoff,)
                    )

                if max_rows is not None:
                    # Retention runs after the incoming receipt is inserted, so
                    # pruning to max_rows yields the configured post-append bound.
                    count = conn.execute(
                        "SELECT COUNT(*) FROM action_receipts"
                    ).fetchone()[0]
                    excess = count - max_rows
                    if excess > 0:
                        conn.execute(
                            "DELETE FROM action_receipts WHERE id IN "
                            "(SELECT id FROM action_receipts ORDER BY id ASC LIMIT ?)",
                            (excess,),
                        )

                if max_size_mb is not None:
                    limit_bytes = max_size_mb * 1024 * 1024
                    # _SIZE_FLOOR_ROWS: the minimum number of newest rows we
                    # will never delete regardless of the size limit.  This
                    # bounds the exceptional case where a single receipt
                    # exceeds the entire configured budget.
                    _SIZE_FLOOR_ROWS = 50
                    # Loop until live-data fits or no more rows are deletable.
                    # (page_count - freelist_count) * page_size measures only
                    # the pages actually holding live data; freed pages join
                    # the freelist immediately within this transaction, so the
                    # metric decreases with each batch without requiring VACUUM.
                    while True:
                        live_row = conn.execute(
                            "SELECT (page_count - freelist_count) * page_size"
                            " FROM pragma_page_count(), pragma_freelist_count(),"
                            " pragma_page_size()"
                        ).fetchone()
                        if live_row is None or live_row[0] <= limit_bytes:
                            break
                        total = conn.execute(
                            "SELECT COUNT(*) FROM action_receipts"
                        ).fetchone()[0]
                        deletable = max(0, total - _SIZE_FLOOR_ROWS)
                        if deletable == 0:
                            # Floor reached with live-data still over limit:
                            # the newest rows alone exceed the budget (single
                            # oversized receipt).  Stop; cannot prune further.
                            break
                        # Delete at least 25 % of deletable rows per pass so
                        # the loop converges in O(log n) iterations.
                        batch = max(1, min(deletable, max(deletable // 4, 100)))
                        conn.execute(
                            "DELETE FROM action_receipts WHERE id IN "
                            "(SELECT id FROM action_receipts ORDER BY id ASC LIMIT ?)",
                            (batch,),
                        )

                # After all deletes, record the predecessor hash expected by the
                # first surviving row. This lets verify_chain() distinguish the
                # legitimate prune boundary from a broken chain link caused by
                # corruption.  Written atomically inside the same transaction.
                first = conn.execute(
                    "SELECT prev_receipt_hash FROM action_receipts ORDER BY id ASC LIMIT 1"
                ).fetchone()
                if first is not None:
                    boundary_hash = first["prev_receipt_hash"] if first["prev_receipt_hash"] else ""
                    conn.execute(
                        "INSERT OR REPLACE INTO meta(key, value) VALUES ('prune_boundary_prev_hash', ?)",
                        (boundary_hash,),
                    )
                else:
                    # Table is empty after pruning — clear any stale boundary.
                    conn.execute(
                        "DELETE FROM meta WHERE key='prune_boundary_prev_hash'"
                    )
            finally:
                # Always restore the no-delete trigger.
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS action_receipts_no_delete
                    BEFORE DELETE ON action_receipts
                    BEGIN
                        SELECT RAISE(ABORT, 'action_receipts is append-only');
                    END
                    """
                )
        except Exception:
            pass

    # ── hashing ────────────────────────────────────────────────────────
    @staticmethod
    def _chain_payload(row: Any) -> dict[str, Any]:
        """The exact field set covered by ``receipt_hash``."""
        keys = (
            "receipt_version",
            "receipt_id",
            "created_at",
            "kind",
            "action_id",
            "session_id",
            "task_id",
            "tool_call_id",
            "envelope_id",
            "actor_lane",
            "execution_surface",
            "tool_name",
            "exit_status",
            "duration_ms",
            "cwd",
            "args_hash",
            "args_redacted_summary",
            "output_hash",
            "output_bytes",
            "envelope_hash",
            "envelope_json",
            "prev_receipt_hash",
        )
        return {key: row[key] for key in keys}

    # ── write ──────────────────────────────────────────────────────────
    def record_receipt(
        self,
        *,
        tool_name: str,
        args: Any = None,
        output: Any = None,
        exit_status: str = "ok",
        duration_ms: int = 0,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        cwd: Optional[str] = None,
        kind: str = "tool_exec",
        envelope: Optional[TaskEnvelope] = None,
        receipt_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> str:
        """Append exactly one receipt and return its ``receipt_id``.

        The tail read, hash computation, and insert all happen inside one
        ``BEGIN IMMEDIATE`` transaction under a process-wide lock, so
        concurrent callers extend a single chain instead of forking it.
        """
        if envelope is None:
            envelope = build_shadow_envelope(
                tool_name=tool_name,
                args=args,
                cwd=cwd,
                session_id=session_id,
                task_id=task_id,
                tool_call_id=tool_call_id,
            )

        output_text = _output_text(output)
        row: dict[str, Any] = {
            "receipt_version": RECEIPT_SCHEMA_VERSION,
            "receipt_id": receipt_id or f"r-{uuid.uuid4().hex}",
            "created_at": created_at or _utc_now(),
            "kind": kind,
            "action_id": envelope.action_id,
            "session_id": session_id,
            "task_id": task_id,
            "tool_call_id": tool_call_id,
            "envelope_id": envelope.envelope_id,
            "actor_lane": envelope.lane,
            "execution_surface": envelope.execution_surface,
            "tool_name": str(tool_name),
            "exit_status": str(exit_status),
            "duration_ms": int(duration_ms),
            "cwd": cwd,
            # args_hash is an HMAC of the canonical arguments; the key is
            # ephemeral (_HMAC_KEY) so the stored value cannot be used offline
            # to confirm specific inputs via dictionary or preimage attack.
            "args_hash": _receipt_hmac(
                canonical_json(args if args is not None else {}).encode("utf-8")
            ),
            "args_redacted_summary": redacted_args_summary(args),
            # output_hash: same privacy property as args_hash.
            "output_hash": (
                _receipt_hmac(output_text.encode("utf-8"))
                if output is not None
                else None
            ),
            "output_bytes": len(output_text.encode("utf-8")),
            # Scrub envelope before persisting: input_hash (SHA-256 of raw args)
            # is replaced with an ephemeral HMAC so the stored JSON cannot be
            # used offline to confirm argument content.
            "envelope_hash": canonical_hash(
                _scrub_envelope_dict(envelope.to_dict())
            ),
            "envelope_json": canonical_json(
                _scrub_envelope_dict(envelope.to_dict())
            ),
        }

        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                tail = conn.execute(
                    "SELECT receipt_hash FROM action_receipts ORDER BY id DESC LIMIT 1"
                ).fetchone()
                row["prev_receipt_hash"] = tail["receipt_hash"] if tail else None
                row["receipt_hash"] = canonical_hash(self._chain_payload(row))
                columns = list(row.keys())
                conn.execute(
                    f"INSERT INTO action_receipts({', '.join(columns)}) "
                    f"VALUES ({', '.join('?' for _ in columns)})",
                    [row[c] for c in columns],
                )
                # Retain after insertion so every configured bound observes the
                # incoming receipt and describes the committed ledger state.
                self._apply_retention(conn)
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

        return str(row["receipt_id"])

    # ── read ───────────────────────────────────────────────────────────
    def read_all(self) -> list[dict[str, Any]]:
        """Every receipt, oldest first. Read-only."""
        if not self.db_path.exists():
            return []
        conn = self._connect_readonly()
        try:
            rows = conn.execute(
                "SELECT * FROM action_receipts ORDER BY id ASC"
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]

    def verify_chain(self) -> list[str]:
        """Return one message per chain defect; empty list means intact.

        Detects internal partial inconsistency: a row whose stored hash does
        not match a recomputation of its own fields, or whose prev_receipt_hash
        does not match the previous row's receipt_hash.

        When a legitimate prune has occurred, the first surviving row's
        ``prev_receipt_hash`` will not match any stored row — but it WILL match
        the ``prune_boundary_prev_hash`` entry written atomically during the
        prune. If that entry is absent or mismatched, the broken link is still
        reported as a defect.

        Both ``prune_boundary_prev_hash`` and the receipt rows are read within
        the *same* ``_connect_readonly`` connection so they come from a single
        WAL snapshot; there is no gap between a boundary read and a row read
        that a concurrent writer could slip into.

        This does **not** prove originality. An actor who rewrites the entire
        file — or truncates the tail — can produce a chain that passes
        verification. The chain is a corruption/partial-tamper detector, not
        cryptographic proof of custody.
        """
        problems: list[str] = []
        if not self.db_path.exists():
            return problems

        # Both meta and receipt rows are fetched inside one BEGIN transaction
        # on a single connection so they share the same WAL read-point — no
        # concurrent writer can slip between the two SELECTs.
        try:
            conn = self._connect_readonly()
            try:
                conn.execute("BEGIN")
                meta_row = conn.execute(
                    "SELECT value FROM meta WHERE key='prune_boundary_prev_hash'"
                ).fetchone()
                prune_boundary_prev: Optional[str] = (
                    meta_row[0] or None if meta_row is not None else None
                )
                raw_rows = conn.execute(
                    "SELECT * FROM action_receipts ORDER BY id ASC"
                ).fetchall()
            finally:
                conn.close()
        except Exception:
            return problems

        rows = [dict(r) for r in raw_rows]
        previous: Optional[str] = None
        is_first = True
        for row in rows:
            expected = canonical_hash(self._chain_payload(row))
            if row["receipt_hash"] != expected:
                problems.append(f"receipt {row['receipt_id']}: content hash mismatch")
            if row["prev_receipt_hash"] != previous:
                if is_first and prune_boundary_prev is not None:
                    # First row after a prune: check against the recorded boundary.
                    expected_prev = prune_boundary_prev if prune_boundary_prev else None
                    if row["prev_receipt_hash"] != expected_prev:
                        problems.append(
                            f"receipt {row['receipt_id']}: broken chain link "
                            f"(prune boundary mismatch)"
                        )
                else:
                    problems.append(f"receipt {row['receipt_id']}: broken chain link")
            previous = row["receipt_hash"]
            is_first = False
        return problems


__all__ = [
    "ActionReceiptLedger",
    "MAX_ARGS_SUMMARY_CHARS",
    "RECEIPT_SCHEMA_VERSION",
    "default_db_path",
    "is_enabled",
    "redacted_args_summary",
]
