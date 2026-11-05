#!/usr/bin/env python3
"""SQLite state store for Hermes Agent: session metadata, message history, model
config, FTS5 search. WAL mode (concurrent readers + one writer); compression
splits sessions via parent_session_id chains; sessions are source-tagged
('cli', 'telegram', ...). Batch-runner / RL trajectories live elsewhere.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import threading
from pathlib import Path
from hermes_dir_acl import make_user_dir

from hermes_constants import get_hermes_home
from typing import Any, Optional

from hermes_state_common import (
    TITLE_SOURCE_DERIVED as _TITLE_SOURCE_DERIVED, TITLE_SOURCE_LLM as _TITLE_SOURCE_LLM,
    TITLE_SOURCE_USER as _TITLE_SOURCE_USER,
    escape_like as _escape_like, _placeholders,
)
from hermes_state_pidns import holder_pid_checkable
from hermes_state_dbfile import _connect_tracked_db
from hermes_state_errors import (
    _STATE_DB_CORRUPT_MSG, DeletedWalGenerationError, SessionCompressionInProgressError, StateDbCorruptError,
    StateDbReplacedError, classify_persistence_error, is_malformed_db_error,
)
from hermes_state_sessions import SessionSessionsMixin
from hermes_state_fts import SessionFtsSetupMixin, load_fts5_cjk_extension
from hermes_state_portability import SessionPortabilityMixin
from hermes_state_telegram import SessionTelegramTopicsMixin
from hermes_state_profile_repair import SessionProfileRepairMixin
from hermes_state_schema import SessionSchemaMixin
from hermes_state_messages import SessionMessagesMixin
from hermes_state_prune_rewrite import SessionPruneRewriteMixin
from hermes_state_api_content import SessionApiContentMixin
from hermes_state_coverage import SessionCoverageMixin
from hermes_state_rewind import SessionRewindMixin
from hermes_state_wal import _WAL_INCOMPAT_MARKERS
from hermes_state_repair import preflight_db_writability
from hermes_state_titles import SessionTitlesMixin
from hermes_state_prompt_refresh import SessionPromptRefreshMixin
from hermes_state_usage import SessionUsageMixin
from hermes_state_maintenance import SessionMaintenanceMixin
from hermes_state_gateway import SessionGatewayMixin
from hermes_state_compression import SessionCompressionMixin
from hermes_state_search import SessionSearchMixin
from persistence.sqlite._holder import StoreHolder, _secure_state_db_files
from storage_user_copy import get_last_init_error

try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_MAX_SAFE_MESSAGES = 20_000  # resume/export guard default


def _configured_transcript_limit(key: str, fallback: int = _MAX_SAFE_MESSAGES) -> int:
    """``sessions.<key>`` from config.yaml (lazy import: circular at load), else *fallback*; 0 disables."""
    try:
        from hermes_cli.config import load_config_readonly
        value = (load_config_readonly().get("sessions") or {}).get(key)
        if value is None:
            return fallback
        limit = int(value)
        return limit if limit >= 0 else fallback
    except Exception:
        return fallback


def resolved_max_resume_messages() -> int:
    return _configured_transcript_limit("max_resume_messages")


def resolved_max_export_messages() -> int:
    return _configured_transcript_limit("max_export_messages")


class SessionResumeTooLargeError(ValueError):
    def __init__(
        self, message_count: int, limit: int = _MAX_SAFE_MESSAGES, scope: str = "across its lineage",
    ):
        self.message_count, self.limit = message_count, limit
        self.scope = scope
        super().__init__(
            f"This session is too long to reload safely ({message_count} messages; limit {limit}). "
            "Start a fresh chat and keep a copy with the dashboard Sessions page's Export action or "
            "`hermes sessions export --format md --session-id <id>` (neither is capped), or raise the limit "
            "with `hermes config set sessions.max_resume_messages 0`."
        )


class SessionExportTooLargeError(ValueError):
    def __init__(self, session_id: str, message_count: int, limit: int = _MAX_SAFE_MESSAGES):
        self.session_id, self.message_count, self.limit = session_id, message_count, limit
        # User-facing refusal shared by every in-memory JSON/JSONL export (CLI and console).
        super().__init__(
            f"Session '{session_id}' has more than {limit:,} exportable messages; the JSON/JSONL "
            "backup is built in memory and capped per session. Use the dashboard Sessions page's streaming "
            "Export action, or set sessions.max_export_messages: 0 in config.yaml to disable the guard."
        )


def _compression_lock_holder_process_is_dead(holder: str) -> bool:
    """True only when a ``pid=<n>`` lock holder's local PID is provably gone.
    Reclaim on kernel proof only: unstructured/same-process holders (another
    thread's live lease) and any probe doubt keep the lease until TTL expiry
    (PID reuse must never steal a live lease; a wrongly-kept one self-heals).
    Foreign/unstamped PID namespaces defer to TTL: see ``hermes_state_pidns``."""
    match = re.search(r"(?:^|:)pid=(\d+)(?::|$)", holder or "")
    pid = int(match.group(1)) if match else 0
    if pid <= 0 or pid == os.getpid():
        return False
    if not holder_pid_checkable(holder):
        return False  # foreign / unknown namespace: defer to TTL
    if psutil is not None:
        try:
            return not psutil.pid_exists(pid)  # recycled PIDs read as alive (conservative)
        except Exception:
            return False
    # psutil-less fallback is POSIX-only: on Windows os.kill(pid, 0) maps sig=0 to
    # CTRL_C_EVENT and can kill the target's console group.
    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok — nt early-returns just above
    except ProcessLookupError:
        return True
    except (OSError, OverflowError):  # PermissionError is an OSError: alive but foreign
        return False
    return False


# Billing buckets that aren't a routable provider identity: a session that persisted only
# one of these (never ran /model) falls back to the config default. Shared by
# session_gateway_runtime and tui_gateway.server so they cannot drift.
_BARE_BILLING_PROVIDERS = frozenset({"auto", "custom"})


# Import-time snapshot lets _default_db_path() detect a re-pointed DEFAULT_DB_PATH
# (tests monkeypatch the constant directly).
DEFAULT_DB_PATH = _IMPORT_DEFAULT_DB_PATH = get_hermes_home() / "state.db"


def _default_db_path() -> Path:
    """Default state DB path at CALL time: a re-pointed ``DEFAULT_DB_PATH`` wins, else
    ``get_hermes_home()`` is resolved fresh (a runtime HERMES_HOME redirect works regardless of import)."""
    return DEFAULT_DB_PATH if DEFAULT_DB_PATH != _IMPORT_DEFAULT_DB_PATH else get_hermes_home() / "state.db"



# Openings of the background-review harness prompts (agent/background_review.py).
_REVIEW_HARNESS_PREFIXES = (
    "Review the conversation above and update the skill library",
    "Review the conversation above and consider saving to memory",
)


def _is_background_review_harness_message(msg: dict[str, Any]) -> bool:
    """Persisted harness prompt (older builds wrote the forked curator's turns
    into real sessions; replaying them hijacks the session)."""
    if not isinstance(msg, dict) or msg.get("role") not in {"user", "system"}:
        return False
    content = msg.get("content")
    return isinstance(content, str) and content.lstrip().startswith(_REVIEW_HARNESS_PREFIXES)


def _strip_background_review_harness(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop harness messages and the curator-mode assistant reply that immediately followed each."""
    if not messages:
        return messages
    out: list[dict[str, Any]] = []
    skip_next_assistant = False
    previous_was_harness = False
    for msg in messages:
        if _is_background_review_harness_message(msg):
            # A consecutive harness prompt occupies the preceding prompt's
            # immediate reply slot, so it must not arm another assistant skip.
            skip_next_assistant = not previous_was_harness
            previous_was_harness = True
            continue
        if skip_next_assistant:
            skip_next_assistant = False
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                previous_was_harness = False
                continue  # the curator-mode reply to the harness prompt
        previous_was_harness = False
        out.append(msg)
    return out


# Matches a bare protocol/tool-name marker such as "[memory]" or "[skill_manage]".
_STALE_TOOL_CALL_MARKER_RE = re.compile(r"^\[[A-Za-z_][A-Za-z0-9_.-]*\]$")


def _is_stale_tool_call_marker_message(msg: dict[str, Any]) -> bool:
    """Assistant tool-call turn whose content is a bare ``[marker]`` (an older
    conversation_loop persisted a local template's marker as the final response)."""
    if not isinstance(msg, dict) or msg.get("role") != "assistant" or not msg.get("tool_calls"):
        return False
    content = msg.get("content")
    return isinstance(content, str) and bool(_STALE_TOOL_CALL_MARKER_RE.fullmatch(content.strip()))


def _strip_stale_tool_call_markers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Blank stale ``[marker]`` assistant content (replaying it teaches the model
    to keep emitting it); tool_call/result pairing stays intact."""
    repaired = 0
    for msg in filter(_is_stale_tool_call_marker_message, messages):
        msg["content"] = ""
        repaired += 1
    if repaired:
        logger.info(
            "Cleared %d stale tool-call marker message(s) while restoring session (#78148)", repaired,
        )
    return messages


_SESSION_DB_CONSEQUENCE = "Sessions will not be saved until this is fixed."
_NETWORK_DRIVE_HINT = " If the database lives on a network drive, move it to a local disk."
_NETWORK_DRIVE_GLOSS = "the session database could not be opened; it may be on a network or unsupported drive"
_NETWORK_DRIVE_ACTION = (
    "Move it to a local disk (`hermes {profile_arg}doctor` shows where it is), then start Hermes again."
)


def format_session_db_unavailable(
    prefix: str = "Hermes can't open its session history right now",
    *,
    details: bool = False,
) -> str:
    """User-facing one-liner: ``<prefix>: <gloss>. <consequence> <action>[ network hint]``.

    The cause table lives in ``hermes_state_user_copy`` so CLI, gateway and TUI agree. Chat
    surfaces (gateway, TUI) get the one-liner; ``details=True`` (the CLI banner) appends a
    ``Details: <raw cause>`` line for the raw SQLite text. Network filesystems (NFS/SMB/FUSE/ZFS)
    cannot host SQLite's write-ahead log: when the raw cause carries one of those markers the
    message names the network-drive suspicion, because ``hermes doctor --fix`` cannot repair a
    mount — only moving the file can."""
    from hermes_constants import profile_cli_selector

    profile_arg = profile_cli_selector()
    cause = get_last_init_error()
    if not cause:
        return (
            f"{prefix}. {_SESSION_DB_CONSEQUENCE} Run `hermes {profile_arg}doctor` to check the "
            "storage location."
        )
    from hermes_state_user_copy import describe_storage_failure
    failure = describe_storage_failure(cause)
    gloss, action, hint = failure.gloss, failure.action, ""
    if any(m in cause.lower() for m in _WAL_INCOMPAT_MARKERS):
        if failure.cause == "unknown":
            gloss, action = _NETWORK_DRIVE_GLOSS, _NETWORK_DRIVE_ACTION.replace("{profile_arg}", profile_arg)
        else:
            hint = _NETWORK_DRIVE_HINT
    text = f"{prefix}: {gloss}. {_SESSION_DB_CONSEQUENCE} {action}{hint}"
    if details:
        from hermes_state_user_copy import storage_failure_details
        text += f"\nDetails: {storage_failure_details(cause)}"
    return text


# Auto-repair at most once per DB path per process (no repair loops; serialises concurrent
# web_server / gateway opens on the same malformed file).
_repair_attempted_paths: set[str] = set()
_repair_attempt_lock = threading.Lock()
# Cross-process schema-surgery lock timeout (``_repair_attempt_lock`` covers one interpreter
# only); sized for the slowest legitimate holder (VACUUM, multi-GB DB).
_REPAIR_LOCK_TIMEOUT_SECONDS = 120.0
_IS_WINDOWS = sys.platform == "win32"



def divert_session_transcript_jsonl(session_id: str, messages) -> Optional[Path]:
    """Append pending messages to HERMES_HOME/sessions/<id>.jsonl (state.db was replaced under a
    live process). Returns the path, or None if nothing to write."""
    sid = str(session_id or "").strip()
    if not sid or not messages:
        return None
    sessions_dir = get_hermes_home() / "sessions"
    make_user_dir(sessions_dir, parents=True, exist_ok=True)
    path = sessions_dir / f"{sid}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for msg in messages:
            if msg is not None:
                record = msg if isinstance(msg, dict) else {"content": str(msg)}
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


# Process-wide shared registry (#90837): long-lived in-process callers (gateway, tui_gateway, cron, in-process
# tools) share ONE store holder per resolved path via hermes_state_registry.acquire(); CLI one-shots, recovery
# flows, and read-only cross-profile opens use SessionDB() directly with their own close().


class SessionDB(
    SessionSessionsMixin, SessionFtsSetupMixin, SessionSearchMixin, SessionSchemaMixin,
    SessionPortabilityMixin, SessionTelegramTopicsMixin, SessionCompressionMixin,
    SessionGatewayMixin, SessionMaintenanceMixin, SessionUsageMixin, SessionTitlesMixin,
    SessionMessagesMixin, SessionPruneRewriteMixin, SessionApiContentMixin, SessionCoverageMixin, SessionRewindMixin, SessionProfileRepairMixin,
    SessionPromptRefreshMixin,
):
    """SQLite-backed session storage with FTS5 search; many reader threads, one writer (WAL)."""

    # Only these state-owned producers join automatic stale-open reconciliation; messaging/UI
    # sources have their own lifecycle owners; unknown sources fail closed.
    # See #60609.  `recovered` = placeholders `hermes sessions recover` synthesizes for
    # orphaned messages (no live owner, never stamped ended_at); without it they are immortal.
    _AUTO_PRUNE_STALE_OPEN_SOURCES: tuple[str, ...] = (
        "cli", "cron", "kanban", "acp", "api_server", "subagent", "tool", "recovered",
    )

    # Imports cap lower than exports: an import holds one BEGIN IMMEDIATE.
    _IMPORT_MAX_SESSIONS, _IMPORT_MAX_MESSAGES_PER_SESSION, _IMPORT_MAX_TOTAL_MESSAGES = 500, 10_000, 50_000
    _IMPORT_MAX_SESSION_BYTES, _IMPORT_MAX_TOTAL_BYTES = 5 * 1024 * 1024, 25 * 1024 * 1024

    @staticmethod
    def _store_system_prompt(conn, system_prompt: Optional[str]) -> Optional[str]:
        if system_prompt is None:
            return None
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT OR IGNORE INTO system_prompts (hash, prompt) VALUES (?, ?)",
            (prompt_hash, system_prompt),
        )
        return prompt_hash

    @staticmethod
    def _delete_unreferenced_system_prompts(conn) -> None:
        conn.execute(
            "DELETE FROM system_prompts WHERE NOT EXISTS ("
            "SELECT 1 FROM sessions WHERE sessions.system_prompt_hash = system_prompts.hash) AND NOT EXISTS ("
            "SELECT 1 FROM sessions WHERE sessions.tool_names = system_prompts.hash)"
        )

    @staticmethod
    def _session_row_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for column in ("system_prompt", "tool_names"):
            if f"_{column}_resolved" in data:
                resolved = data.pop(f"_{column}_resolved")
                if column in data:
                    data[column] = resolved
        # Lift /new vs /branch markers out of model_config so list payloads
        # (which strip that heavy field) can still tell a reset sibling from
        # a genuine fork. Keep this a local helper: tests sometimes replace
        # the SessionDB name with a factory lambda.
        if not (data.get("_reset_from") and data.get("_branched_from")):
            raw = data.get("model_config")
            cfg = None
            if isinstance(raw, str) and raw:
                try:
                    cfg = json.loads(raw)
                except (TypeError, ValueError):
                    cfg = None
            elif isinstance(raw, dict):
                cfg = raw
            if isinstance(cfg, dict):
                if not data.get("_reset_from"):
                    value = cfg.get("_reset_from")
                    if isinstance(value, str) and value.strip():
                        data["_reset_from"] = value.strip()
                if not data.get("_branched_from"):
                    value = cfg.get("_branched_from")
                    if isinstance(value, str) and value.strip():
                        data["_branched_from"] = value.strip()
        return data

    def __init__(self, db_path: Path | None = None, read_only: bool = False, *, holder: Optional[StoreHolder] = None):
        """Open *db_path* on a store holder of its own, or serve through *holder* (the registry lends one per file)."""
        # Keep only the opening call site, never a frame (which pins caller locals).
        creation_site = "unknown"
        caller = None
        try:
            caller = sys._getframe(1)
            creation_site = f"{caller.f_globals.get('__name__', '?')}.{caller.f_code.co_name}:{caller.f_lineno}"
        except Exception:
            pass  # Diagnostic metadata must not prevent opening the database.
        finally:
            del caller
        if holder is None:
            holder = StoreHolder(db_path or _default_db_path(), read_only, creation_site=creation_site)
        self._holder = holder

    def __enter__(self) -> "SessionDB":
        """``with SessionDB(path) as db:`` closes on exit; owners must release deterministically.

        Ownership of a SessionDB should be released explicitly. Historically an instance with a started
        token writer pinned ITSELF (bound-method writer target plus a strong ``atexit`` drain hook), so
        ``__del__`` never ran for exactly the instances that leaked descriptors (#88033). The writer now
        retires after an idle window and the atexit hook holds only a weak reference, so abandoned handles
        are eventually collectible — but "eventually, after the idle window and a GC cycle" is not a release
        policy. Call sites owning a handle are still expected to close it deterministically (see the
        ownership comments in ``run_agent.py`` and ``tui_gateway/methods_session.py``).
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False  # never suppress the caller's exception

    def close(self):
        """Close this handle's store holder; on a registry-lent one, release one reference instead."""
        self._holder.close()

    MAX_TITLE_LENGTH = 100

    # Title provenance, lowest to highest authority: auto-titling may only replace a
    # strictly lower-authority title (``derived`` -> ``llm`` once; never a user-typed name).
    TITLE_SOURCE_DERIVED = _TITLE_SOURCE_DERIVED
    TITLE_SOURCE_LLM = _TITLE_SOURCE_LLM
    TITLE_SOURCE_USER = _TITLE_SOURCE_USER
    _TITLE_SOURCE_RANK = {TITLE_SOURCE_DERIVED: 0, TITLE_SOURCE_LLM: 1, TITLE_SOURCE_USER: 2}

    # Bot Mode's canonical chat is resolved by exact-title lookup: the title IS the identity,
    # so _set_session_title refuses renames of a hidden row holding it.
    # Bot Mode's forever-chat registry: the session titled exactly this, on a bot's profile, IS the bot's
    # canonical chat — resolved by exact-title lookup on every open (no session-id pointer exists). See
    # #92473.
    CANONICAL_BOT_CHAT_TITLE = "Bot Chat"

    # ── Message storage constants (SessionMessagesMixin) ──
    # Prefix marking JSON-encoded structured content; NUL cannot collide with text.
    _CONTENT_JSON_PREFIX = "\x00json:"
    #: Reactions live inside ``display_metadata`` so they survive row rewrites.
    REACTIONS_METADATA_KEY = "reactions"
    # Columns every conversation projection decodes; ``active`` rides along so a display read
    # can split compaction-archived rows without a second query. Contract: must include every
    # ``agent.transcript_repair._OWNED_COLUMNS`` column, because replay stamps
    # ``transcript_row_snapshot(row)`` from these rows (token_count is hashed there, not decoded).
    _CONVERSATION_ROW_COLUMNS = (
        "id, role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
        "finish_reason, reasoning, reasoning_content, reasoning_details, "
        "codex_reasoning_items, codex_message_items, platform_message_id, observed, "
        "_compressed_summary, timestamp, token_count, active, api_content, display_kind, display_metadata, message_uid, "
        "absorbed_message_uids, tool_call_uids, tool_call_uid"
    )

    # ── Meta key/value (scheduler bookkeeping) ──

    def get_meta(self, key: str) -> Optional[str]:
        """Read state_meta[key] on self._lock (not _read_ctx): fts_rebuild_step reads progress before its
        write transaction and a WAL reader would not see it."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def set_meta(self, key: str, value: str, *, cursor: Optional[sqlite3.Cursor] = None) -> None:
        """Upsert state_meta[key]; with ``cursor`` the write is inline (the caller already holds a
        transaction — nesting BEGIN IMMEDIATE would deadlock)."""
        sql = (
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )
        if cursor is not None:
            cursor.execute(sql, (key, value))
        else:
            self._write_sql(sql, (key, value))

    def retag_kanban_worker_sessions(self, workspaces_root: str) -> int:
        """Retag legacy kanban worker rows from ``cli`` to ``kanban`` by cwd under the board's workspaces
        root; gated once per root via state_meta. Returns rows retagged."""
        prefix = str(workspaces_root).rstrip("/\\")
        if not prefix:
            return 0
        gate = f"kanban_worker_source_retagged:{prefix}"
        if self.get_meta(gate) == "1":
            return 0
        def _do(conn):
            esc = _escape_like(prefix)
            cursor = conn.execute(
                "UPDATE sessions SET source = 'kanban' "
                "WHERE source = 'cli' AND (cwd = ? OR cwd LIKE ? ESCAPE '\\' "
                "OR cwd LIKE ? ESCAPE '\\')",
                (prefix, f"{esc}/%", f"{esc}\\\\%"),
            )
            # rowcount BEFORE set_meta reuses this cursor for its INSERT.
            retagged = cursor.rowcount or 0
            self.set_meta(gate, "1", cursor=cursor)
            return retagged
        return self._execute_write(_do)

    def is_kanban_owned_session(self, session_id: str) -> bool:
        """True when this session — or any segment of the compression lineage a resume would
        materialize — belongs to the Kanban dispatcher (``source``/``created_source`` =
        ``'kanban'``): the transcript of a worker run, not a human conversation (#68779).

        Both columns are checked: ``source`` is live routing state the dispatcher tags
        (``HERMES_SESSION_SOURCE=kanban``) and the legacy retag rewrites, while immutable
        ``created_source`` survives later surface flips. The lineage is the VERIFIED
        compression chain (``_resume_lineage_ids``) — exactly the rows a resume loads — so a
        worker transcript stays kanban-owned across rotations, while a delegate/branch child
        of a worker (a DIFFERENT conversation) is not swept in. Plain classification, not a
        gate: the resume-time guard (``hermes_cli/kanban_resume_guard.py``) owns the decision
        and the dispatcher-owned exemption."""
        lineage = self._resume_lineage_ids(session_id)
        if not lineage:
            return False
        rows = self._read_all(
            f"SELECT 1 FROM sessions WHERE id IN ({_placeholders(lineage)}) "
            "AND (source = 'kanban' OR created_source = 'kanban') LIMIT 1",
            tuple(lineage))
        return bool(rows)

    def list_meta_prefix(self, prefix: str) -> list[tuple[str, str]]:
        """``[(key, value), ...]`` for state_meta keys starting with the literal
        ``prefix`` (LIKE wildcards escaped) — e.g. ``loop:<session_id>`` rows."""
        if not prefix:
            return []
        rows = self._read_all(
            "SELECT key, value FROM state_meta WHERE key LIKE ? ESCAPE '\\'", (_escape_like(prefix) + "%",),
        )
        return [(row[0], row[1]) for row in rows]


class _HolderMember:
    """A member of the store holder a SessionDB serves through: read, written and deleted on the handle's holder,
    read from StoreHolder on the class."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __get__(self, db, owner=None):
        return getattr(StoreHolder if db is None else db._holder, self.name)

    def __set__(self, db, value) -> None:
        setattr(db._holder, self.name, value)

    def __delete__(self, db) -> None:
        delattr(db._holder, self.name)


# The connection core a SessionDB serves through: writer connection and lock, transactions, the read pool,
# file and WAL-generation guards, quarantine, the accounting queue and the state the schema code keeps there.
for _name in (
    "db_path", "read_only", "_creation_site", "_shared_registry_owned",
    "_WRITE_PATIENCE_S", "_TRANSCRIPT_WRITE_PATIENCE_S", "_ACTIVITY_WRITE_PATIENCE_S", "_COMPRESSION_BUSY_WAIT_S",
    "_WRITE_RETRY_MIN_S", "_WRITE_RETRY_MAX_S", "_WRITE_RETRY_SLOW_AFTER_S", "_WRITE_RETRY_SLOW_MIN_S",
    "_WRITE_RETRY_SLOW_MAX_S", "_CHECKPOINT_EVERY_N_WRITES", "_FTS_MERGE_EVERY_N_WRITES",
    "_FTS_MERGE_MAX_PAGES_PER_INDEX", "_FTS_MERGE_COMMANDS_PER_PASS",
    "_conn", "_lock", "_wal_active", "_write_count", "_wal_lock_guard",
    "_read_pool", "_read_budget", "_read_permits", "_read_conns_lock", "_read_conns_closed", "_read_open_failed_at",
    "_db_file_identity", "_db_file_application_id", "_db_sidecar_identity", "_db_replaced", "_db_wal_generation_lost",
    "_retired_generation_capture", "_retired_capture_lock", "_retire_connection", "_connection_pinned",
    "_db_corrupt", "_db_corrupt_reason",
    "_fts_enabled", "_fts_stale", "_trigram_available", "_fts_cjk_loaded", "_fts_cjk_available",
    "_fts_unavailable_warned", "_trigram_unavailable_warned", "_fts_usermerge_floor_applied",
    "_fts_deferred_holder_pids", "_fts_stale_retry_after", "_fts_stale_retry_interval",
    "_close_connection_quietly", "_close_conn_logged", "_open_writer", "_open_read_only", "_connect_read_only",
    "_handle_quarantine_if_invalid", "_open_writer_conn", "_connect_and_init", "_connect_and_init_with_lock_patience",
    "_get_read_conn", "_evict_one_idle_read_conn", "_close_read_conn", "_checkout_read_conn", "_read_ctx",
    "_reopen_after_close_locked", "_execute_write", "_write_sql", "_write_rowcount", "_read_one", "_read_all",
    "_log_reopen_notice",
    "_read_retrying_ioerr", "_ensure_db_file_generation", "_record_db_file_identity", "_db_file_was_replaced",
    "_wal_generation_was_lost", "_halt_if_db_generation_changed", "_capture_retired_generation",
    "_raise_if_db_replaced", "_is_structural_corruption_error", "_corrupt_error", "_halt_db_corrupt",
    "_disable_close_time_checkpoint", "_pin_connection", "_settle_lost_generation_locked", "_raise_if_db_corrupt",
    "_sleep_before_write_retry", "_foreign_state_db_holders", "_quarantine_reason", "_try_wal_checkpoint",
    "_TOKEN_DELTA_SUM_FIELDS", "_TOKEN_DELTA_COST_FIELDS", "_TOKEN_DELTA_ROUTE_FIELDS", "_TOKEN_WRITER_IDLE_SECONDS",
    "_token_queue", "_token_queue_cond", "_token_writer_thread", "_token_writer_stop", "_token_writer_busy",
    "_token_atexit_hook", "queue_token_counts", "_apply_claimed_batch", "flush_token_counts", "_token_writer_loop",
    "_apply_token_batch", "_coalesce_token_deltas", "_stop_token_writer", "_drain_token_queue_at_exit",
):
    setattr(SessionDB, _name, _HolderMember(_name))
del _name


class AsyncSessionDB:
    """Async door onto SessionDB: every call runs via asyncio.to_thread so a blocking SQLite call
    never freezes the event loop (no method returns a live cursor)."""

    def __init__(self, db: SessionDB) -> None:
        self._db = db

    def __getattr__(self, name: str):
        attr = getattr(self._db, name)
        if not callable(attr):
            return attr
        async def _offloaded(*args, **kwargs):
            return await asyncio.to_thread(attr, *args, **kwargs)
        return _offloaded
