"""LCM Engine — Lossless Context Management.

Implements the ContextEngine ABC. Replaces the built-in ContextCompressor
with a DAG-based summarization system that preserves every message.
"""

import copy
import inspect
import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from agent.context_engine import ContextEngine

from .config import LCMConfig
from .dag import SummaryDAG, SummaryNode
from .escalation import (
    SummaryCircuitBreaker,
    _strip_reasoning_blocks,
    summarize_with_escalation,
)
from .externalize import (
    build_transcript_gc_placeholder,
    extract_externalized_ref,
    find_externalized_payload_for_message,
    load_externalized_payload,
    maybe_externalize_tool_output,
    reassign_externalized_payloads,
)
from .extraction import (
    extract_before_compaction,
    sanitize_pre_compaction_content,
    sanitize_pre_compaction_tool_arguments,
)
from .ingest_protection import (
    _json_has_duplicate_object_keys,
    assistant_output_quarantine_reason,
    extract_ingest_externalized_refs,
    protect_inline_payloads_in_text,
    protect_messages_for_ingest,
    quarantine_suspicious_assistant_messages,
    redact_sensitive_value,
    restore_ingest_payload_placeholders,
    sensitive_pattern_status,
)
from .schemas import (
    LCM_DESCRIBE,
    LCM_DOCTOR,
    LCM_EXPAND,
    LCM_EXPAND_QUERY,
    LCM_GREP,
    LCM_LOAD_SESSION,
    LCM_STATUS,
)
from .session_patterns import (
    build_session_match_keys,
    compile_session_patterns,
    matches_session_pattern,
)
from .message_patterns import compile_message_patterns, matches_message_pattern
from .lifecycle_state import LifecycleStateStore
from .message_content import (
    normalize_content_value,
    stored_text_content_for_pattern_matching,
    text_content_for_pattern_matching,
)
from .store import MessageStore
from .storage_security import plaintext_path_policy, retention_status
from .tokens import count_message_tokens, count_messages_tokens, count_tokens
from . import tools as lcm_tools

logger = logging.getLogger(__name__)

_PLUGIN_ROOT = Path(__file__).resolve().parent
_PLUGIN_METADATA: dict[str, str] | None = None
_SESSION_END_BUSY_TIMEOUT_MS = 50
_FAIL_OPEN_RATE_WINDOW_SECONDS = 300.0


class LCMFailOpenRecoveryError(RuntimeError):
    """Raised when fail-open fallback cannot keep the next request under context."""

    recoverable = True
    compression_exhausted = True


_VISIBLE_TEXT_PART_TYPES = {"text", "input_text", "output_text"}
_INTERNAL_ASSISTANT_PART_TYPES = {
    "analysis",
    "chain_of_thought",
    "internal",
    "reasoning",
    "redacted_thinking",
    "scratchpad",
    "thought",
    "thinking",
}


def _strip_metadata_scalar(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _plugin_metadata() -> dict[str, str]:
    """Return plugin identity from the loaded code tree.

    Always re-read the manifest from disk when available so status tools reflect
    hot-updated plugin checkouts even in long-lived Hermes processes.
    """
    global _PLUGIN_METADATA

    metadata = {"name": "hermes-lcm", "version": "unknown"}
    manifest = _PLUGIN_ROOT / "plugin.yaml"
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            key, sep, raw_value = line.partition(":")
            if not sep:
                continue
            key = key.strip()
            if key in {"name", "version"}:
                metadata[key] = _strip_metadata_scalar(raw_value)
        _PLUGIN_METADATA = metadata
        return dict(metadata)
    except OSError:
        logger.debug("LCM plugin manifest not readable at %s", manifest)

    if _PLUGIN_METADATA is not None:
        return dict(_PLUGIN_METADATA)
    return dict(metadata)


def _git_runtime_identity(root: Path) -> dict[str, Any]:
    """Best-effort git identity for source checkouts.

    Packaged installs may not have a `.git` directory. In that case the fields
    stay empty instead of turning status/doctor into a git dependency.
    """

    if not (root / ".git").exists():
        return {
            "plugin_git_commit": "",
            "plugin_git_branch": "",
            "plugin_git_dirty": None,
            "plugin_git_remote": "",
        }

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=1,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.debug("LCM git identity probe failed at %s: %s", root, exc)
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    dirty_output = _git("status", "--porcelain")
    return {
        "plugin_git_commit": _git("rev-parse", "HEAD") or "",
        "plugin_git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "",
        "plugin_git_dirty": None if dirty_output is None else bool(dirty_output),
        "plugin_git_remote": _git("config", "--get", "remote.origin.url") or "",
    }


def _is_sqlite_locked_error(exc: BaseException) -> bool:
    """Return True when an exception chain represents SQLite lock contention."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        if isinstance(current, sqlite3.Error) and "locked" in message:
            return True
        current = current.__cause__ or current.__context__
    return False


def _sqlite_busy_timeout_ms(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA busy_timeout").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


@contextmanager
def _temporary_sqlite_busy_timeout(
    connections: List[sqlite3.Connection | None],
    timeout_ms: int,
) -> Iterator[None]:
    """Temporarily bound SQLite lock waits for gateway-critical paths."""
    bounded_timeout = max(0, int(timeout_ms))
    originals: list[tuple[sqlite3.Connection, int]] = []
    for conn in connections:
        if conn is None:
            continue
        original = _sqlite_busy_timeout_ms(conn)
        conn.execute(f"PRAGMA busy_timeout={bounded_timeout}")
        originals.append((conn, original))
    try:
        yield
    finally:
        for conn, original in reversed(originals):
            conn.execute(f"PRAGMA busy_timeout={original}")


_SYNTHETIC_ASSISTANT_NOISE = {
    "ack",
    "acknowledged",
    "heartbeat",
    "heartbeat ack",
    "keepalive",
    "keep alive",
    "pong",
}

_PRESERVED_TODO_CONTEXT_PREFIX = "[Your active task list was preserved across context compression]"
_PRESERVED_OBJECTIVE_CONTEXT_PREFIX = "[Current user objective preserved from compacted history]"


def _tool_call_id(tool_call: Any) -> str:
    if not isinstance(tool_call, dict):
        return ""
    value = tool_call.get("id") or tool_call.get("tool_call_id")
    return str(value).strip() if value else ""


def _assistant_tool_call_ids(messages: List[Dict[str, Any]]) -> set[str]:
    call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            call_id = _tool_call_id(tool_call)
            if call_id:
                call_ids.add(call_id)
    return call_ids


def _matched_tool_call_ids(messages: List[Dict[str, Any]]) -> set[str]:
    assistant_call_ids = _assistant_tool_call_ids(messages)
    tool_result_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            tool_call_id = str(msg.get("tool_call_id") or "").strip()
            if tool_call_id:
                tool_result_ids.add(tool_call_id)
    return assistant_call_ids & tool_result_ids


def _is_synthetic_assistant_noise(content: str) -> bool:
    normalized = re.sub(r"\s+", " ", (content or "").strip()).lower()
    if not normalized:
        return True
    normalized = normalized.strip("`*_ ")
    bracketless = normalized.strip("[](){} ")
    return normalized in _SYNTHETIC_ASSISTANT_NOISE or bracketless in _SYNTHETIC_ASSISTANT_NOISE


class LCMEngine(ContextEngine):
    """Lossless Context Management engine.

    Automatic LCM compaction is routine background maintenance. Hosts that
    support user-visible compaction status opt-outs should keep successful
    automatic LCM passes silent unless the user explicitly asks for diagnostics.

    Architecture:
      1. Every message is persisted verbatim in an immutable MessageStore
      2. When context pressure builds, older messages outside the fresh tail
         are summarized into leaf nodes (D0) in a SummaryDAG
      3. When enough nodes accumulate at a depth, they're condensed into
         higher-depth nodes (D1, D2, ...)
      4. The agent gets tools (lcm_grep, lcm_load_session, lcm_describe,
         lcm_expand) to search and drill into compacted history
      5. Active context = system prompt + DAG summaries + fresh tail
    """

    def __init__(self, config: LCMConfig | None = None,
                 hermes_home: str = ""):
        self._config = config or LCMConfig.from_env()
        self._hermes_home = hermes_home

        db_path = self._resolve_db_path(hermes_home)
        self._bind_storage(db_path, hermes_home)

        self._session_id: str = ""
        self._session_platform: str = ""
        # Tracks the most recent non-ignored, non-stateless binding so that
        # user-facing tools (lcm_status, lcm_grep default scope, lcm_describe,
        # lcm_expand_query, lcm_doctor) keep showing the foreground session
        # even while a side-channel session (cron, debug) temporarily owns the
        # engine's _session_id binding. Updated alongside _session_id only
        # when _refresh_session_filters classifies the new session as a real
        # foreground (neither ignored nor stateless). Read via the
        # `current_session_id` / `current_session_platform` properties and
        # `current_session_ignored` / `current_session_stateless` /
        # `side_channel_active` companion predicates.
        self._foreground_session_id: str = ""
        self._foreground_session_platform: str = ""
        self._foreground_conversation_id: str = ""
        self._conversation_id: str = ""
        self._session_match_keys: list[str] = []
        self._session_ignored = False
        self._session_stateless = False
        self._compiled_ignore_session_patterns = compile_session_patterns(
            self._config.ignore_session_patterns
        )
        self._compiled_stateless_session_patterns = compile_session_patterns(
            self._config.stateless_session_patterns
        )
        self._compiled_ignore_message_patterns = compile_message_patterns(
            self._config.ignore_message_patterns
        )
        self._ignored_message_count: int = 0

        # Track which store_ids have been ingested into the DAG
        self._last_compacted_store_id: int = 0

        # Cursor: index in the current messages list up to which all
        # messages have been persisted.  After compress() shortens the
        # list, the cursor resets to len(compressed) so that only
        # genuinely new messages (appended after compaction) get ingested.
        # The cursor is process-local; existing sessions rebound after a
        # gateway restart reconcile it against the durable store on the
        # next ingest.
        self._ingest_cursor: int = 0
        self._ingest_cursor_needs_reconcile = False
        self._last_ingest_reconciliation: Dict[str, Any] = {
            "action": "none",
            "reason": "not run",
        }

        # State required by ContextEngine ABC and run_agent.py compatibility
        self.model = ""
        self.base_url = ""
        self.api_key = ""
        self.provider = ""
        self.api_mode = ""
        self.context_length = 0
        self._context_length_source = ""
        self._update_model_pending_session_start = False
        self.threshold_tokens = 0
        self.threshold_percent = self._config.context_threshold
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_reasoning_tokens = 0
        self.cache_metrics_available = False
        self.compression_count = 0
        # run_agent.py reads these for preflight checks
        self.protect_first_n = 3
        self.protect_last_n = self._config.fresh_tail_count
        # run_agent.py reads these for context probing
        self._context_probed = False
        self._context_probe_persistable = False
        # Host compatibility: LCM treats successful automatic compaction as
        # silent maintenance. Manual /lcm diagnostics and warning/error paths
        # remain explicit.
        self.emit_automatic_compaction_status = False
        self.quiet_mode = True
        self.summary_model = self._config.summary_model
        self._summary_circuit_breaker = SummaryCircuitBreaker(
            failure_threshold=self._config.summary_circuit_breaker_failure_threshold,
            cooldown_seconds=self._config.summary_circuit_breaker_cooldown_seconds,
        )
        self._last_overflow_recovery_failed = False
        self._last_condensation_suppressed_reason = ""
        self._last_compression_status = "idle"
        self._last_compression_noop_reason = ""
        self._degraded = False
        self._last_degraded_reason = ""
        self._last_fail_open_at = 0.0
        self._last_fail_open_error = ""
        self._last_fail_open_error_type = ""
        self._last_fail_open_fallback_error = ""
        self._last_fail_open_fallback_error_type = ""
        self._last_fail_open_raw_tokens = 0
        self._last_fail_open_context_limit = 0
        self._last_fail_open_over_context_limit = False
        self._last_fail_open_fallback_attempted = False
        self._last_fail_open_fallback_status = ""
        self._fail_open_total_count = 0
        self._fail_open_consecutive_count = 0
        self._fail_open_timestamps: list[float] = []
        # Cooldown timestamp to prevent compression cascade after boundary skip.
        # Set when skip-carry-over path is taken in _continue_compression_boundary.
        self._last_boundary_skip_time: float = 0
        # Temporary source window used only while compress() assembles context.
        # _assemble_context also serves tests and recovery paths directly, so
        # keep anchoring opt-in rather than changing its public behavior.
        self._pending_context_anchor_messages: Optional[List[Dict[str, Any]]] = None
        self._logged_filter_config = False
        self._pending_reset_session_id: str = ""
        self._pending_reset_conversation_id: str = ""
        self._pending_reset_frontier_store_id: int = 0
        self._thread_context = threading.local()
        self._auxiliary_session_ids: set[str] = set()
        self._auxiliary_lineage_session_ids: set[str] = set()
        self._auxiliary_session_lock = threading.RLock()

    def clone_for_agent(self) -> "LCMEngine":
        """Return a fresh runtime engine for one AIAgent instance.

        Hermes registers plugin context engines process-wide, while gateway
        runtimes may keep multiple cached AIAgent instances alive at once
        (different platforms, chats, cron jobs, etc.).  LCM stores mutable
        session binding and ingest cursor state on the engine object itself, so
        sharing one registered instance across agents can let one conversation
        rebind another conversation's raw-message ingest and lifecycle state.

        The clone shares the same durable SQLite database path/configuration,
        but gets independent session/cursor/lifecycle runtime state. Hermes core
        calls update_model() on the returned engine before on_session_start().
        """
        return type(self)(
            config=copy.deepcopy(self._config),
            hermes_home=self._hermes_home,
        )

    def _resolve_db_path(self, hermes_home: str = "") -> Path:
        """Resolve the SQLite path for the active Hermes profile/home."""
        if self._config.database_path:
            return Path(self._config.database_path)
        if hermes_home:
            return Path(hermes_home) / "lcm.db"
        return Path.home() / ".hermes" / "lcm.db"

    def _bind_storage(self, db_path: str | Path, hermes_home: str = "") -> None:
        """Bind store/DAG/lifecycle helpers to one SQLite database."""
        self._store = MessageStore(
            db_path,
            ingest_protection_config=self._config,
            hermes_home=hermes_home,
        )
        self._dag = SummaryDAG(db_path, ingest_protection_config=self._config)
        self._lifecycle = LifecycleStateStore(db_path)

    def _close_storage(self) -> None:
        """Best-effort close of currently bound SQLite helpers."""
        for attr in ("_store", "_dag", "_lifecycle"):
            helper = getattr(self, attr, None)
            close = getattr(helper, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("LCM failed closing %s during profile rebind", attr, exc_info=True)

    def _reset_profile_runtime_state(self) -> None:
        """Clear process-local session state that cannot cross profile homes."""
        self._session_id = ""
        self._session_platform = ""
        self._foreground_session_id = ""
        self._foreground_session_platform = ""
        self._foreground_conversation_id = ""
        self._conversation_id = ""
        self._session_match_keys = []
        self._session_ignored = False
        self._session_stateless = False
        self._clear_pending_reset_boundary()
        with self._auxiliary_session_lock:
            self._auxiliary_session_ids.clear()
            self._auxiliary_lineage_session_ids.clear()
        self._clear_thread_context_stateless()
        self._reset_session_scoped_runtime_state()

    def _rebind_storage_for_home(self, hermes_home: str = "") -> bool:
        """Switch SQLite-backed state when a reused engine serves another profile.

        Hermes core passes the active ``hermes_home`` on session start.  Older
        Hermes versions may still reuse the same plugin/context-engine object
        after ``HERMES_HOME`` changes, so the plugin must not assume the store
        captured during ``register()`` is still correct.
        """
        if not hermes_home:
            return False
        if self._config.database_path:
            current_home = str(self._hermes_home or "")
            current_store_home = str(getattr(getattr(self, "_store", None), "_hermes_home", "") or "")
            if current_home == str(hermes_home) and current_store_home == str(hermes_home):
                return False
            self._hermes_home = hermes_home
            store = getattr(self, "_store", None)
            if store is not None:
                store._hermes_home = hermes_home
            self._reset_profile_runtime_state()
            logger.info("LCM rebound Hermes home for configured database path %s", hermes_home)
            return True

        db_path = self._resolve_db_path(hermes_home)
        current_db = Path(getattr(getattr(self, "_store", None), "db_path", ""))
        if current_db == db_path and str(self._hermes_home or "") == str(hermes_home):
            return False

        self._close_storage()
        self._hermes_home = hermes_home
        self._bind_storage(db_path, hermes_home)
        self._reset_profile_runtime_state()
        logger.info("LCM rebound storage for Hermes home %s", hermes_home)
        return True

    def _set_context_length(self, context_length: Any, *, source: str) -> bool:
        try:
            parsed_context_length = int(context_length)
        except (TypeError, ValueError):
            logger.debug("LCM ignored invalid %s context_length: %r", source, context_length)
            return False
        if parsed_context_length <= 0:
            logger.debug(
                "LCM cleared non-positive %s context_length: %r",
                source,
                context_length,
            )
            self.context_length = 0
            self._context_length_source = source
            self.threshold_tokens = 0
            return True
        self.context_length = parsed_context_length
        self._context_length_source = source
        self.threshold_tokens = int(
            parsed_context_length * self._config.context_threshold
        )
        return True

    def _session_metadata_matches_active_runtime(
        self,
        kwargs: Dict[str, Any],
        *,
        ignore_empty_optional: bool = False,
    ) -> bool:
        if "model" in kwargs and str(kwargs.get("model") or "") != self.model:
            return False
        for key in ("provider", "base_url", "api_key", "api_mode"):
            if key not in kwargs:
                continue
            incoming = str(kwargs.get(key) or "")
            if ignore_empty_optional and not incoming:
                continue
            if incoming != str(getattr(self, key, "") or ""):
                return False
        return True

    @property
    def name(self) -> str:
        return "lcm"

    @property
    def current_session_id(self) -> str:
        """User-facing "current session" id surfaced by LCM tools.

        Returns the most recent foreground binding (the last session id that
        ``_refresh_session_filters`` classified as neither ignored nor
        stateless). Falls back to ``_session_id`` when no foreground has
        ever been bound, so unattended cron-only or stateless-only processes
        remain observable via ``lcm_status``.

        Lifecycle paths (compress, ingest, on_session_end, etc.) must keep
        reading ``_session_id`` directly because those paths must follow the
        binding the engine is actually servicing. Only tool-surface code
        paths that report a "current session" view to operators should read
        this property.
        """
        return self._foreground_session_id or self._session_id

    @property
    def current_session_platform(self) -> str:
        """Platform string paired with ``current_session_id``."""
        if self._foreground_session_id:
            return self._foreground_session_platform
        return self._session_platform

    @property
    def current_conversation_id(self) -> str:
        """Conversation id paired with ``current_session_id``."""
        if self._foreground_session_id:
            return self._foreground_conversation_id
        return self._conversation_id

    @property
    def side_channel_active(self) -> bool:
        """True when an ignored or stateless session has temporarily rebound
        ``_session_id`` while a real foreground binding still exists.

        Operators reading lcm_status during this window see the foreground
        session id and counts (because tools read ``current_session_id``)
        but the engine itself is servicing the side channel. This predicate
        lets diagnostic surfaces (lcm_status, /lcm command) make the
        divergence explicit without recomputing the underlying invariant.
        """
        return bool(self._foreground_session_id) and self._foreground_session_id != self._session_id

    @property
    def current_session_ignored(self) -> bool:
        """``_session_ignored`` reported for ``current_session_id``.

        When a side channel is in flight the foreground is by definition
        non-ignored; otherwise this is the bound session's ignore flag.
        """
        if self.side_channel_active:
            return False
        return self._session_ignored

    @property
    def current_session_stateless(self) -> bool:
        """``_session_stateless`` reported for ``current_session_id``.

        When a side channel is in flight the foreground is by definition
        non-stateless; otherwise this is the bound session's stateless flag.
        """
        if self.side_channel_active:
            return False
        return self._session_stateless

    # -- ContextEngine required methods ------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        if self._thread_context_stateless():
            return
        self.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.last_total_tokens = int(usage.get("total_tokens", 0) or 0)

        cache_keys = {"cache_read_tokens", "cache_write_tokens"}
        self.cache_metrics_available = any(key in usage for key in cache_keys)
        self.last_input_tokens = int(usage.get("input_tokens", self.last_prompt_tokens) or 0)
        self.last_output_tokens = int(
            usage.get("output_tokens", self.last_completion_tokens) or 0
        )
        self.last_cache_read_tokens = int(usage.get("cache_read_tokens", 0) or 0)
        self.last_cache_write_tokens = int(usage.get("cache_write_tokens", 0) or 0)
        self.last_reasoning_tokens = int(usage.get("reasoning_tokens", 0) or 0)

    @property
    def cache_read_ratio(self) -> float:
        if self.last_prompt_tokens <= 0:
            return 0.0
        return self.last_cache_read_tokens / self.last_prompt_tokens

    def _compression_boundary_cooldown_active(self) -> bool:
        """Return true while a boundary skip is in its short no-compress window."""
        if self._last_boundary_skip_time <= 0:
            return False
        elapsed = time.time() - self._last_boundary_skip_time
        if elapsed < 60:
            logger.debug(
                "LCM compression cooldown active: %.1f seconds since boundary skip",
                elapsed,
            )
            return True
        self._last_boundary_skip_time = 0
        return False

    def should_compress(self, prompt_tokens: int = None) -> bool:
        if self._session_ignored or self._session_stateless or self._thread_context_stateless():
            return False
        if self._compression_boundary_cooldown_active():
            return False
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if self._should_force_overflow_recovery(observed_tokens=tokens):
            return True
        if self.threshold_tokens <= 0:
            return False
        return tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages):
        """Pre-flight check — also ingests messages into the store."""
        if self._session_ignored or self._session_stateless or self._thread_context_stateless():
            return False
        replay_messages = None
        if self._session_id and messages:
            try:
                replay_messages = self._ingest_messages(messages)
            except Exception as e:
                logger.warning("Ingest during preflight: %s", e)
        if replay_messages is not None and replay_messages != messages:
            if self._compression_boundary_cooldown_active():
                return False
            return True
        if self._compression_boundary_cooldown_active():
            return False
        from .tokens import count_messages_tokens
        rough = count_messages_tokens(messages)
        if self._should_force_overflow_recovery(observed_tokens=rough):
            return True
        if self.threshold_tokens > 0 and rough >= self.threshold_tokens:
            eligible, reason = self._leaf_compaction_candidate_status(messages)
            if eligible:
                return True
            if self._should_run_deferred_maintenance(messages, observed_tokens=rough):
                return True
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = reason
            logger.info("LCM preflight compression no-op: %s", reason)
            return False
        self._refresh_raw_backlog_debt(messages, observed_tokens=rough)
        return self._should_run_deferred_maintenance(messages, observed_tokens=rough)

    def _leaf_compaction_candidate_status(
        self,
        messages: List[Dict[str, Any]],
        *,
        force_overflow: bool = False,
    ) -> tuple[bool, str]:
        """Return whether a normal leaf compaction pass can actually run.

        The host asks ``should_compress_preflight`` before it emits user-visible
        compression status. A session can be over the global context threshold
        while all pressure sits in the protected fresh tail, or while the raw
        backlog outside that tail is still smaller than the configured leaf
        chunk. In that case ``compress()`` would immediately no-op, so preflight
        should not advertise a compaction attempt yet.
        """
        if not messages:
            return False, "empty message list"
        n = len(messages)
        fresh_tail_start = max(0, n - self._config.fresh_tail_count)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return False, "no eligible raw backlog outside fresh tail"

        candidate_raw = messages[leading_anchor_count:fresh_tail_start]
        if not candidate_raw:
            return False, "no eligible raw backlog outside fresh tail"

        if force_overflow:
            return True, "forced overflow recovery"

        raw_tokens_outside_tail = count_messages_tokens(candidate_raw)
        if self._config.dynamic_leaf_chunk_enabled:
            working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens_outside_tail)
        else:
            working_leaf_chunk_tokens = self._config.leaf_chunk_tokens
        if raw_tokens_outside_tail < working_leaf_chunk_tokens:
            return False, "raw backlog outside fresh tail is below leaf chunk threshold"
        return True, "eligible raw backlog outside fresh tail"

    def _working_leaf_chunk_tokens(self, raw_tokens_outside_tail: int) -> int:
        base = max(1, self._config.leaf_chunk_tokens)
        if not self._config.dynamic_leaf_chunk_enabled:
            return base
        ceiling = max(base, self._config.dynamic_leaf_chunk_max)
        working = base
        while working < ceiling and raw_tokens_outside_tail > working * 2:
            working = min(ceiling, working * 2)
        return working

    def _select_oldest_leaf_chunk(
        self,
        candidate_raw: List[Dict[str, Any]],
        working_leaf_chunk_tokens: int,
    ) -> List[Dict[str, Any]]:
        selected: list[Dict[str, Any]] = []
        used = 0
        for msg in candidate_raw:
            msg_tokens = count_message_tokens(msg)
            if used + msg_tokens > working_leaf_chunk_tokens and selected:
                break
            selected.append(msg)
            used += msg_tokens
        return selected

    def _is_retry_worthy_leaf_summary_error(self, exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        message = str(exc).lower()
        retry_markers = (
            "context length",
            "maximum context",
            "max context",
            "too many tokens",
            "token limit",
            "prompt is too long",
            "input too long",
            "request too large",
            "timed out",
            "timeout",
        )
        return any(marker in message for marker in retry_markers)

    def _next_leaf_rescue_chunk(
        self,
        current_chunk: List[Dict[str, Any]],
        current_source_tokens: int,
    ) -> List[Dict[str, Any]]:
        if len(current_chunk) <= 1:
            return []

        floor_tokens = max(1, self._config.leaf_chunk_tokens)
        shrink_targets = [
            max(floor_tokens, int(current_source_tokens * 0.75)),
            max(floor_tokens, int(current_source_tokens * 0.50)),
        ]

        for target in shrink_targets:
            if target >= current_source_tokens:
                continue
            smaller = self._select_oldest_leaf_chunk(current_chunk, target)
            if smaller and len(smaller) < len(current_chunk):
                return smaller

        return current_chunk[:-1]

    def _summarize_leaf_chunk_with_rescue(
        self,
        initial_chunk: List[Dict[str, Any]],
        focus_topic: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], int, str, int, int]:
        attempt_chunk = list(initial_chunk)
        max_attempts = 3
        attempt_number = 0

        while attempt_chunk and attempt_number < max_attempts:
            attempt_number += 1
            source_tokens = count_messages_tokens(attempt_chunk)
            serialized = self._serialize_messages(attempt_chunk)
            token_budget = max(2000, int(source_tokens * 0.20))
            token_budget = min(token_budget, 12000)

            try:
                summary_text, level = summarize_with_escalation(
                    text=serialized,
                    source_tokens=source_tokens,
                    token_budget=token_budget,
                    depth=0,
                    model=self._config.summary_model,
                    fallback_models=self._config.summary_fallback_models,
                    circuit_breaker=self._summary_circuit_breaker,
                    timeout=self._config.summary_timeout_ms / 1000,
                    l2_budget_ratio=self._config.l2_budget_ratio,
                    l3_truncate_tokens=self._config.l3_truncate_tokens,
                    focus_topic=focus_topic or "",
                    custom_instructions=self._config.custom_instructions,
                )
                return attempt_chunk, source_tokens, summary_text, level, attempt_number
            except Exception as exc:
                if attempt_number >= max_attempts or not self._is_retry_worthy_leaf_summary_error(exc):
                    raise
                smaller_chunk = self._next_leaf_rescue_chunk(attempt_chunk, source_tokens)
                if not smaller_chunk or len(smaller_chunk) >= len(attempt_chunk):
                    raise
                logger.warning(
                    "LCM leaf summarization retrying with smaller oldest chunk after retry-worthy failure: %s (attempt %d/%d, %d→%d messages)",
                    exc,
                    attempt_number,
                    max_attempts,
                    len(attempt_chunk),
                    len(smaller_chunk),
                )
                attempt_chunk = smaller_chunk

        raise RuntimeError("adaptive leaf rescue exhausted without a valid chunk")

    def compress(self, messages: List[Dict[str, Any]],
                 current_tokens: Optional[int] = None,
                 focus_topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """Main compaction entry point with fail-open degraded handling."""
        try:
            compressed = self._compress_lossless(
                messages,
                current_tokens=current_tokens,
                focus_topic=focus_topic,
            )
        except LCMFailOpenRecoveryError:
            raise
        except Exception as exc:
            return self._handle_fail_open_compression_exception(
                messages,
                exc,
                current_tokens=current_tokens,
                focus_topic=focus_topic,
            )
        self._mark_compression_not_degraded()
        return compressed

    def _compress_lossless(self, messages: List[Dict[str, Any]],
                           current_tokens: Optional[int] = None,
                           focus_topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """Main compaction implementation.

        1. Ingest any new messages into the store
        2. Identify messages outside the fresh tail
        3. Summarize them into DAG leaf nodes
        4. Check if condensation is needed
        5. Assemble new active context: summaries + fresh tail
        """
        if not messages:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = "empty message list"
            return messages

        self._last_compression_status = "running"
        self._last_compression_noop_reason = ""

        if self._session_ignored or self._session_stateless or self._thread_context_stateless():
            reason = (
                "auxiliary thread context"
                if self._thread_context_stateless()
                else "ignored session"
                if self._session_ignored
                else "stateless session"
            )
            logger.debug(
                "LCM compress bypassed for %s session %s",
                "auxiliary" if self._thread_context_stateless() else "ignored" if self._session_ignored else "stateless",
                self._thread_context_session_id() or self._session_id or "(unknown)",
            )
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = f"bypassed: {reason}"
            return self._redact_active_replay_messages(messages)

        observed_prompt_tokens = current_tokens if current_tokens is not None else None
        force_overflow = self._should_force_overflow_recovery(
            observed_tokens=observed_prompt_tokens,
            messages=messages,
        )
        recovery_assembly_cap = (
            self._overflow_recovery_assembly_cap(
                observed_tokens=observed_prompt_tokens,
                messages=messages,
            )
            if force_overflow
            else None
        )

        # Step 1: Ingest new messages into the immutable store. Work from a
        # replay-safe view so quarantined assistant loops do not enter summaries
        # or provider context after the durable row has been written.
        working_messages = self._ingest_messages(messages)
        anchor_source_messages = list(working_messages)
        pressure_messages = messages if len(messages) == len(working_messages) else working_messages
        leaf_compacted_this_turn = False
        dropped_replayed_scaffold_messages = False
        leaf_passes = 0
        critical_budget_pressure = self._critical_budget_pressure_reached(
            observed_tokens=observed_prompt_tokens,
            messages=working_messages,
        )
        deferred_maintenance_active = (
            not force_overflow
            and self._should_run_deferred_maintenance(
                working_messages,
                observed_tokens=observed_prompt_tokens,
            )
        )
        if deferred_maintenance_active:
            self._lifecycle.record_maintenance_attempt(self._conversation_id)
        base_max_leaf_passes = 4 if self._config.dynamic_leaf_chunk_enabled else 1
        max_leaf_passes = base_max_leaf_passes
        if deferred_maintenance_active:
            max_leaf_passes = max(1, self._config.deferred_maintenance_max_passes)
        estimated_active_tokens = (
            observed_prompt_tokens
            if observed_prompt_tokens is not None and observed_prompt_tokens > 0
            else count_messages_tokens(messages)
        )
        noop_reason = "no eligible raw backlog outside fresh tail"

        while leaf_passes < max_leaf_passes:
            n = len(working_messages)
            fresh_tail_start = max(0, n - self._config.fresh_tail_count)

            # Keep only a real system prompt anchored. Gateway sessions may
            # pass only conversation messages, so index 0 can be an old user
            # turn; that must remain eligible for compaction instead of being
            # replayed forever as fresh-looking intent.
            leading_anchor_count = self._leading_anchor_count(working_messages)
            if fresh_tail_start <= leading_anchor_count:
                noop_reason = "no eligible raw backlog outside fresh tail"
                break

            candidate_start = leading_anchor_count
            while (
                candidate_start < fresh_tail_start
                and self._is_replayed_context_scaffold_message(working_messages[candidate_start])
            ):
                candidate_start += 1
            if candidate_start > leading_anchor_count:
                dropped_replayed_scaffold_messages = True
                working_messages = working_messages[:leading_anchor_count] + working_messages[candidate_start:]
                pressure_messages = pressure_messages[:leading_anchor_count] + pressure_messages[candidate_start:]
                n = len(working_messages)
                fresh_tail_start = max(0, n - self._config.fresh_tail_count)
                if fresh_tail_start <= leading_anchor_count:
                    noop_reason = "selected leaf chunk lacks raw store lineage"
                    break

            candidate_raw = working_messages[leading_anchor_count:fresh_tail_start]
            if not candidate_raw:
                noop_reason = "no eligible raw backlog outside fresh tail"
                break

            pressure_candidate_raw = pressure_messages[leading_anchor_count:fresh_tail_start]
            raw_tokens_outside_tail = count_messages_tokens(pressure_candidate_raw)
            if self._config.dynamic_leaf_chunk_enabled:
                working_leaf_chunk_tokens = self._working_leaf_chunk_tokens(raw_tokens_outside_tail)
                if raw_tokens_outside_tail < working_leaf_chunk_tokens and not force_overflow:
                    if not (deferred_maintenance_active and critical_budget_pressure):
                        noop_reason = (
                            "raw backlog outside fresh tail is below leaf chunk threshold"
                        )
                        break
                if force_overflow:
                    to_compact = candidate_raw
                else:
                    to_compact = self._select_oldest_leaf_chunk(candidate_raw, working_leaf_chunk_tokens)
            else:
                if raw_tokens_outside_tail < self._config.leaf_chunk_tokens and not force_overflow:
                    if not (deferred_maintenance_active and critical_budget_pressure):
                        noop_reason = (
                            "raw backlog outside fresh tail is below leaf chunk threshold"
                        )
                        break
                to_compact = candidate_raw

            if not to_compact:
                noop_reason = "no eligible leaf chunk selected"
                break

            # Pre-compaction extraction: best-effort, never blocks compaction
            if self._config.extraction_enabled:
                self._run_pre_compaction_extraction(to_compact)

            compacted_chunk, source_tokens, summary_text, _level, _rescue_attempts = self._summarize_leaf_chunk_with_rescue(
                to_compact,
                focus_topic=focus_topic,
            )
            remaining_messages = working_messages[leading_anchor_count + len(compacted_chunk):]

            source_store_ids = self._get_store_ids_for_messages(compacted_chunk)
            earliest_at, latest_at = self._store.get_time_bounds(source_store_ids)
            summary_tokens = count_tokens(summary_text)

            node = SummaryNode(
                session_id=self._session_id,
                depth=0,
                summary=summary_text,
                token_count=summary_tokens,
                source_token_count=source_tokens,
                source_ids=source_store_ids,
                source_type="messages",
                created_at=time.time(),
                earliest_at=earliest_at,
                latest_at=latest_at,
                expand_hint=self._extract_expand_hint(summary_text),
            )
            self._dag.add_node(node)
            self._maybe_gc_compacted_tool_results(compacted_chunk, source_store_ids)
            self._last_compacted_store_id = max(source_store_ids) if source_store_ids else 0
            self._persist_frontier_marker()

            pressure_remaining_messages = pressure_messages[leading_anchor_count + len(compacted_chunk):]
            working_messages = working_messages[:leading_anchor_count] + remaining_messages
            pressure_messages = pressure_messages[:leading_anchor_count] + pressure_remaining_messages
            leaf_compacted_this_turn = True
            leaf_passes += 1
            estimated_active_tokens = max(0, estimated_active_tokens - source_tokens + summary_tokens)

            if not self._config.dynamic_leaf_chunk_enabled:
                break

            if not force_overflow:
                if (not deferred_maintenance_active) and self.threshold_tokens > 0 and estimated_active_tokens < self.threshold_tokens:
                    break
                leading_anchor_count = self._leading_anchor_count(working_messages)
                remaining_raw = working_messages[
                    leading_anchor_count:max(0, len(working_messages) - self._config.fresh_tail_count)
                ]
                if not remaining_raw:
                    break
                pressure_remaining_raw = pressure_messages[
                    leading_anchor_count:max(0, len(pressure_messages) - self._config.fresh_tail_count)
                ]
                remaining_raw_tokens = count_messages_tokens(pressure_remaining_raw)
                remaining_threshold = self._working_leaf_chunk_tokens(remaining_raw_tokens)
                if remaining_raw_tokens < remaining_threshold:
                    if not (deferred_maintenance_active and critical_budget_pressure):
                        break

        if not leaf_compacted_this_turn:
            self._refresh_raw_backlog_debt(
                working_messages,
                observed_tokens=observed_prompt_tokens,
            )
            if force_overflow and len(messages) >= 1:
                leading_anchor_count = self._leading_anchor_count(working_messages)
                compressed = self._assemble_overflow_recovery_context(
                    working_messages[0] if leading_anchor_count else None,
                    working_messages[leading_anchor_count:],
                    assembly_cap_override=recovery_assembly_cap,
                )
                return self._finalize_forced_overflow_result(
                    working_messages,
                    compressed,
                    assembly_cap_override=recovery_assembly_cap,
                )
            if dropped_replayed_scaffold_messages:
                leading_anchor_count = self._leading_anchor_count(working_messages)
                anchor_leading_count = self._leading_anchor_count(anchor_source_messages)
                self._pending_context_anchor_messages = anchor_source_messages[anchor_leading_count:]
                try:
                    sanitized_messages = self._assemble_context(
                        working_messages[0] if leading_anchor_count else None,
                        working_messages[leading_anchor_count:],
                        assembly_cap_override=recovery_assembly_cap,
                    )
                finally:
                    self._pending_context_anchor_messages = None
            else:
                sanitized_messages = self._sanitize_active_context_messages(
                    working_messages,
                    insert_missing_tool_stubs=False,
                )
            if sanitized_messages != working_messages:
                # _ingest_messages() already advanced the cursor to the original
                # active-context length. If the host continues from a sanitized
                # or reassembled context, keeping the old cursor could make the
                # next appended messages look already ingested. This applies to
                # content-only cleanup as well as dropped-message cleanup.
                self._ingest_cursor = len(sanitized_messages)
                self._last_compression_status = "sanitized"
                self._last_compression_noop_reason = ""
            else:
                if dropped_replayed_scaffold_messages:
                    # The active context changed even though no new leaf node was
                    # written. Keep the cursor aligned with the returned context
                    # so the next appended turn is ingested instead of skipped.
                    self._ingest_cursor = len(sanitized_messages)
                self._last_compression_status = "noop"
                self._last_compression_noop_reason = noop_reason
                logger.info("LCM compression no-op: %s", noop_reason)
            return sanitized_messages

        # Step 6: Check if condensation is needed
        self._maybe_condense(
            focus_topic=focus_topic,
            leaf_compacted_this_turn=True,
            force_overflow=force_overflow,
            critical_budget_pressure=critical_budget_pressure,
        )

        # Step 7: Assemble new active context
        self._refresh_raw_backlog_debt(
            working_messages,
            observed_tokens=observed_prompt_tokens,
        )
        leading_anchor_count = self._leading_anchor_count(working_messages)
        anchor_leading_count = self._leading_anchor_count(anchor_source_messages)
        self._pending_context_anchor_messages = anchor_source_messages[anchor_leading_count:]
        try:
            compressed = self._assemble_context(
                working_messages[0] if leading_anchor_count else None,
                working_messages[leading_anchor_count:],
                assembly_cap_override=recovery_assembly_cap,
            )
        finally:
            self._pending_context_anchor_messages = None
        self.compression_count += 1
        self._last_compression_status = "compacted"
        self._last_compression_noop_reason = ""
        if recovery_assembly_cap is None:
            self._last_overflow_recovery_failed = False
        else:
            self._last_overflow_recovery_failed = count_messages_tokens(compressed) > recovery_assembly_cap
            if self._last_overflow_recovery_failed:
                logger.warning(
                    "LCM overflow recovery could not get under cap=%d after compaction; returning best-effort context (%d tokens)",
                    recovery_assembly_cap,
                    count_messages_tokens(compressed),
                )
        # Reset cursor to the length of the compressed context so that
        # only messages appended *after* this point get ingested next time.
        self._ingest_cursor = len(compressed)
        self._ingest_cursor_needs_reconcile = False

        logger.info(
            "LCM compaction #%d: %d messages → %d (%d leaf pass%s, %d→%d tokens, %d DAG nodes%s)",
            self.compression_count,
            len(messages),
            len(compressed),
            leaf_passes,
            "es" if leaf_passes != 1 else "",
            count_messages_tokens(messages),
            count_messages_tokens(compressed),
            len(self._dag.get_session_nodes(self._session_id)),
            ", forced overflow recovery" if force_overflow else "",
        )

        # ── Active-context cleanup / tool-pair guardrail (same as _assemble_context) ──
        # compress() output is consumed directly by the main loop in some
        # edge cases (e.g. forced overflow recovery bypassing _assemble_context).
        compressed = self._sanitize_active_context_messages(compressed)

        return compressed

    def _mark_compression_not_degraded(self) -> None:
        self._degraded = False
        self._last_degraded_reason = ""
        self._fail_open_consecutive_count = 0

    def _fail_open_rate_snapshot(self) -> tuple[int, float]:
        now = time.time()
        cutoff = now - _FAIL_OPEN_RATE_WINDOW_SECONDS
        self._fail_open_timestamps = [
            ts for ts in self._fail_open_timestamps if ts >= cutoff
        ]
        window_count = len(self._fail_open_timestamps)
        rate_per_minute = window_count * (60.0 / _FAIL_OPEN_RATE_WINDOW_SECONDS)
        return window_count, rate_per_minute

    def _safe_fail_open_error_text(self, exc: BaseException) -> str:
        redacted = redact_sensitive_value(str(exc), self._config)
        text = str(redacted or "")
        return text or type(exc).__name__

    def _raw_messages_exceed_context_limit(
        self,
        raw_tokens: int,
        context_limit: Optional[int],
    ) -> bool:
        return bool(context_limit and context_limit > 0 and raw_tokens >= context_limit)

    def _record_fail_open_degraded(
        self,
        exc: BaseException,
        *,
        raw_tokens: int,
        context_limit: Optional[int],
        fallback_attempted: bool,
        fallback_status: str,
        fallback_exc: Optional[BaseException] = None,
    ) -> None:
        now = time.time()
        safe_error = self._safe_fail_open_error_text(exc)
        safe_fallback_error = (
            self._safe_fail_open_error_text(fallback_exc)
            if fallback_exc is not None
            else ""
        )

        self._degraded = True
        self._last_degraded_reason = "fail_open"
        self._last_fail_open_at = now
        self._last_fail_open_error = safe_error
        self._last_fail_open_error_type = type(exc).__name__
        self._last_fail_open_fallback_error = safe_fallback_error
        self._last_fail_open_fallback_error_type = (
            type(fallback_exc).__name__ if fallback_exc is not None else ""
        )
        self._last_fail_open_raw_tokens = raw_tokens
        self._last_fail_open_context_limit = int(context_limit or 0)
        self._last_fail_open_over_context_limit = self._raw_messages_exceed_context_limit(
            raw_tokens,
            context_limit,
        )
        self._last_fail_open_fallback_attempted = fallback_attempted
        self._last_fail_open_fallback_status = fallback_status
        self._fail_open_total_count += 1
        self._fail_open_consecutive_count += 1
        self._fail_open_timestamps.append(now)
        window_count, rate_per_minute = self._fail_open_rate_snapshot()

        if fallback_status == "succeeded":
            self._last_compression_status = "degraded_fallback_compressed"
            self._last_compression_noop_reason = ""
        else:
            self._last_compression_status = "degraded_fail_open"
            self._last_compression_noop_reason = (
                f"fail-open: {type(exc).__name__}: {safe_error}"
            )

        logger.warning(
            "event=lcm_fail_open_degraded engine=lcm session_id=%s "
            "error_type=%s error=%r raw_tokens=%d context_limit=%s "
            "over_context_limit=%s fallback_attempted=%s fallback_status=%s "
            "total=%d consecutive=%d window_seconds=%d window_count=%d "
            "rate_per_minute=%.3f fallback_error_type=%s fallback_error=%r",
            self.current_session_id or self._session_id or "",
            type(exc).__name__,
            safe_error,
            raw_tokens,
            int(context_limit or 0) if context_limit else "unknown",
            self._last_fail_open_over_context_limit,
            fallback_attempted,
            fallback_status,
            self._fail_open_total_count,
            self._fail_open_consecutive_count,
            int(_FAIL_OPEN_RATE_WINDOW_SECONDS),
            window_count,
            rate_per_minute,
            type(fallback_exc).__name__ if fallback_exc is not None else "",
            safe_fallback_error,
        )

    def _build_fail_open_fallback_compressor(self) -> Any:
        from agent.context_compressor import ContextCompressor

        return ContextCompressor(
            model=self.model or "unknown",
            threshold_percent=self.threshold_percent,
            protect_first_n=self.protect_first_n,
            protect_last_n=self.protect_last_n,
            summary_target_ratio=0.20,
            quiet_mode=True,
            summary_model_override=self.summary_model or self._config.summary_model,
            base_url=self.base_url,
            api_key=self.api_key,
            config_context_length=self.context_length if self.context_length > 0 else None,
            provider=self.provider,
            api_mode=self.api_mode,
            abort_on_summary_failure=False,
        )

    def _recoverable_fail_open_error(
        self,
        primary_exc: BaseException,
        fallback_exc: BaseException,
        *,
        raw_tokens: int,
        context_limit: Optional[int],
    ) -> LCMFailOpenRecoveryError:
        primary_text = self._safe_fail_open_error_text(primary_exc)
        fallback_text = self._safe_fail_open_error_text(fallback_exc)
        limit_text = str(int(context_limit or 0)) if context_limit else "unknown"
        return LCMFailOpenRecoveryError(
            "recoverable LCM fail-open overflow: primary LCM compression failed "
            f"({type(primary_exc).__name__}: {primary_text}); raw context "
            f"is over provider context limit ({raw_tokens}/{limit_text} tokens); "
            "built-in ContextCompressor fallback failed "
            f"({type(fallback_exc).__name__}: {fallback_text}). "
            "No silent provider-overflow fallback is safe; start a fresh session "
            "or manually compact after fixing LCM."
        )

    def _handle_fail_open_compression_exception(
        self,
        messages: List[Dict[str, Any]],
        exc: Exception,
        *,
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        raw_tokens = (
            current_tokens
            if current_tokens is not None and current_tokens > 0
            else count_messages_tokens(messages)
        )
        request_overhead_tokens = max(0, raw_tokens - count_messages_tokens(messages))
        context_limit = self.context_length if self.context_length > 0 else None

        if not self._raw_messages_exceed_context_limit(raw_tokens, context_limit):
            self._record_fail_open_degraded(
                exc,
                raw_tokens=raw_tokens,
                context_limit=context_limit,
                fallback_attempted=False,
                fallback_status="not_needed_below_limit",
            )
            return messages

        try:
            fallback_compressor = self._build_fail_open_fallback_compressor()
            fallback_messages = fallback_compressor.compress(
                messages,
                current_tokens=raw_tokens,
                focus_topic=focus_topic,
                force=True,
            )
            fallback_tokens = count_messages_tokens(fallback_messages) + request_overhead_tokens
            if self._raw_messages_exceed_context_limit(fallback_tokens, context_limit):
                raise LCMFailOpenRecoveryError(
                    "built-in ContextCompressor fallback did not reduce raw "
                    f"context below provider limit ({fallback_tokens}/"
                    f"{int(context_limit or 0)} tokens)"
                )
        except Exception as fallback_exc:
            self._record_fail_open_degraded(
                exc,
                raw_tokens=raw_tokens,
                context_limit=context_limit,
                fallback_attempted=True,
                fallback_status="failed",
                fallback_exc=fallback_exc,
            )
            raise self._recoverable_fail_open_error(
                exc,
                fallback_exc,
                raw_tokens=raw_tokens,
                context_limit=context_limit,
            ) from fallback_exc

        self._record_fail_open_degraded(
            exc,
            raw_tokens=raw_tokens,
            context_limit=context_limit,
            fallback_attempted=True,
            fallback_status="succeeded",
        )
        self._ingest_cursor = len(fallback_messages)
        self._ingest_cursor_needs_reconcile = False
        return fallback_messages

    # -- ContextEngine optional methods ------------------------------------

    def _bind_lifecycle_state(
        self,
        session_id: str,
        *,
        conversation_id: str | None = None,
    ) -> None:
        state = self._lifecycle.bind_session(session_id, conversation_id=conversation_id)
        self._conversation_id = state.conversation_id
        self._last_compacted_store_id = state.current_frontier_store_id
        if not self._session_ignored and not self._session_stateless:
            self._foreground_session_id = session_id
            self._foreground_session_platform = self._session_platform
            self._foreground_conversation_id = state.conversation_id

        # Garbage-collect empty lifecycle rows when the table exceeds threshold.
        # Gateway restarts, ephemeral cron ticks, and crash-loops all create
        # lifecycle rows that never ingest data — prune them here so they
        # don't accumulate forever.
        if (
            self._config.empty_lifecycle_gc_enabled
            and self._lifecycle.row_count() > self._config.empty_lifecycle_gc_threshold
        ):
            protected = {str(self._session_id)} if self._session_id else None
            max_age = self._config.empty_lifecycle_gc_max_age_hours
            try:
                deleted = self._lifecycle.prune_empty_sessions(
                    protected_session_ids=protected,
                    max_age_hours=max_age,
                )
            except Exception:
                deleted = 0
            if deleted:
                logger.info(
                    "LCM pruned %d lifecycle rows with zero stored data "
                    "(table exceeded threshold of %d rows)",
                    deleted,
                    self._config.empty_lifecycle_gc_threshold,
                )

    def _persist_frontier_marker(self) -> None:
        if not self._session_id or not self._conversation_id:
            return
        self._lifecycle.advance_frontier(
            self._conversation_id,
            self._session_id,
            self._last_compacted_store_id,
        )

    def _thread_context_auxiliary_stack(self) -> list[str]:
        stack = getattr(self._thread_context, "auxiliary_session_stack", None)
        if stack is None:
            current = str(getattr(self._thread_context, "current_auxiliary_session_id", "") or "")
            stack = [current] if current else []
            self._thread_context.auxiliary_session_stack = stack
        return stack

    def _sync_thread_context_current_auxiliary(self) -> list[str]:
        stack = self._thread_context_auxiliary_stack()
        active_ids = self._active_auxiliary_session_ids()
        stack[:] = [session_id for session_id in stack if session_id in active_ids]
        self._thread_context.current_auxiliary_session_id = stack[-1] if stack else ""
        return stack

    def _thread_context_session_id(self) -> str:
        stack = self._sync_thread_context_current_auxiliary()
        stack_session_id = self._in_process_auxiliary_session_id_from_stack()
        if stack_session_id:
            return stack_session_id
        if stack:
            return stack[-1]
        return ""

    def _thread_context_has_auxiliary_session(self, session_id: str) -> bool:
        with self._auxiliary_session_lock:
            return session_id in self._auxiliary_session_ids

    def _active_auxiliary_session_ids(self) -> set[str]:
        with self._auxiliary_session_lock:
            return set(self._auxiliary_session_ids)

    def _known_auxiliary_lineage_session_ids(self) -> set[str]:
        with self._auxiliary_session_lock:
            return set(self._auxiliary_lineage_session_ids)

    def _has_auxiliary_lineage_session(self, session_id: str) -> bool:
        with self._auxiliary_session_lock:
            return session_id in self._auxiliary_lineage_session_ids

    def _thread_context_stateless(self) -> bool:
        return bool(self._thread_context_session_id())

    def _register_auxiliary_session(self, session_id: str) -> None:
        with self._auxiliary_session_lock:
            self._auxiliary_session_ids.add(session_id)
            self._auxiliary_lineage_session_ids.add(session_id)

    def _deactivate_auxiliary_session(self, session_id: str) -> None:
        if not session_id:
            return
        with self._auxiliary_session_lock:
            self._auxiliary_session_ids.discard(session_id)

    def _mark_thread_context_stateless(self, session_id: str) -> None:
        self._register_auxiliary_session(session_id)
        stack = self._thread_context_auxiliary_stack()
        stack[:] = [existing for existing in stack if existing != session_id]
        stack.append(session_id)
        self._thread_context.current_auxiliary_session_id = session_id

    def _clear_thread_context_stateless(self, session_id: str = "") -> None:
        stack = self._thread_context_auxiliary_stack()
        if session_id:
            stack[:] = [existing for existing in stack if existing != session_id]
        else:
            stack.clear()
        self._sync_thread_context_current_auxiliary()

    def _handoff_auxiliary_session(self, old_session_id: str, new_session_id: str) -> None:
        with self._auxiliary_session_lock:
            if old_session_id:
                self._auxiliary_session_ids.discard(old_session_id)
                self._auxiliary_lineage_session_ids.add(old_session_id)
            if new_session_id:
                self._auxiliary_session_ids.add(new_session_id)
                self._auxiliary_lineage_session_ids.add(new_session_id)
        stack = self._thread_context_auxiliary_stack()
        had_thread_marker = old_session_id in stack or new_session_id in stack
        stack[:] = [
            existing
            for existing in stack
            if existing not in {old_session_id, new_session_id}
        ]
        if had_thread_marker and new_session_id:
            stack.append(new_session_id)
        self._sync_thread_context_current_auxiliary()

    def _unmark_thread_context_auxiliary_session(self, session_id: str) -> None:
        with self._auxiliary_session_lock:
            self._auxiliary_session_ids.discard(session_id)
        self._clear_thread_context_stateless(session_id)

    def _get_allowed_hermes_base(self) -> Path | None:
        """Get the allowed base directory for hermes_home, or None if not restricted."""
        env_base = os.environ.get("LCM_HERMES_BASE_DIR")
        if env_base:
            return Path(env_base).expanduser().resolve()
        return None  # No restriction when env var not set

    def _state_db_path(self, kwargs: Dict[str, Any] | None = None) -> Path:
        kwargs = kwargs or {}
        hermes_home = str(kwargs.get("hermes_home") or self._hermes_home or "")
        if hermes_home:
            # Prevent directory traversal by resolving the path
            path = Path(hermes_home).expanduser().resolve()
            # Check containment within allowed base only when restriction is active
            allowed_base = self._get_allowed_hermes_base()
            if allowed_base is not None:
                try:
                    path.relative_to(allowed_base)
                except ValueError:
                    raise ValueError(
                        f"hermes_home {hermes_home} resolves to {path} which is not within allowed base {allowed_base}"
                    )
            return path / "state.db"
        return Path(self._store.db_path).parent / "state.db"

    def _caller_is_auxiliary_agent_frame(self, caller_self: Any) -> bool:
        if caller_self is None:
            return False
        if getattr(caller_self, "_subagent_id", None):
            return True
        if getattr(caller_self, "_parent_subagent_id", None):
            return True
        try:
            if int(getattr(caller_self, "_delegate_depth", 0) or 0) > 0:
                return True
        except (TypeError, ValueError):
            pass
        memory_origin = str(getattr(caller_self, "_memory_write_origin", "") or "")
        memory_context = str(getattr(caller_self, "_memory_write_context", "") or "")
        if memory_origin == "background_review" or memory_context == "background_review":
            return True
        log_prefix = str(getattr(caller_self, "log_prefix", "") or "").strip()
        if log_prefix.startswith("[subagent-"):
            return True
        enabled_toolsets = getattr(caller_self, "enabled_toolsets", None)
        if enabled_toolsets is not None:
            try:
                toolsets = {str(toolset) for toolset in enabled_toolsets}
            except TypeError:
                toolsets = set()
            if toolsets and toolsets <= {"memory", "skills"}:
                return True
        if getattr(caller_self, "ephemeral_system_prompt", None) and log_prefix.startswith("[subagent-"):
            return True
        return False

    def _in_process_parent_session_id(
        self,
        kwargs: Dict[str, Any],
        session_id: str = "",
        include_explicit: bool = True,
    ) -> str:
        explicit = str(kwargs.get("parent_session_id") or "")
        if include_explicit and explicit:
            return explicit
        target_session_id = str(session_id or kwargs.get("session_id") or "")
        frame = inspect.currentframe()
        try:
            frame = frame.f_back if frame is not None else None
            for _ in range(32):
                if frame is None:
                    return ""
                caller_self = frame.f_locals.get("self")
                if not self._caller_is_auxiliary_agent_frame(caller_self):
                    frame = frame.f_back
                    continue
                parent = str(getattr(caller_self, "_parent_session_id", "") or "")
                caller_session = str(getattr(caller_self, "session_id", "") or "")
                if parent and caller_session and (
                    not target_session_id or caller_session == target_session_id
                ):
                    return parent
                frame = frame.f_back
        finally:
            del frame
        return ""

    def _in_process_auxiliary_session_id_from_stack(self) -> str:
        active_ids = self._active_auxiliary_session_ids()
        lineage_ids = self._known_auxiliary_lineage_session_ids()
        if not active_ids and not lineage_ids and not self._session_id:
            return ""
        frame = inspect.currentframe()
        try:
            frame = frame.f_back if frame is not None else None
            for _ in range(32):
                if frame is None:
                    return ""
                caller_self = frame.f_locals.get("self")
                if not self._caller_is_auxiliary_agent_frame(caller_self):
                    frame = frame.f_back
                    continue
                session_id = str(getattr(caller_self, "session_id", "") or "")
                parent_id = str(getattr(caller_self, "_parent_session_id", "") or "")
                if session_id and parent_id and (
                    session_id in active_ids
                    or session_id in lineage_ids
                    or parent_id == self._session_id
                    or parent_id in lineage_ids
                ):
                    return session_id
                frame = frame.f_back
        finally:
            del frame
        return ""

    def _is_live_auxiliary_child_session(
        self,
        session_id: str,
        parent_session_id: str,
        kwargs: Dict[str, Any],
    ) -> bool:
        """Return True when a same-process child agent should not rebind LCM.

        Detect Hermes auxiliary/background child sessions without treating real
        foreground branches as stateless. In-process auxiliary agent frames are
        trusted even when this engine is fresh and has no bound foreground yet.
        Explicit parent metadata by itself is not enough, because legitimate
        foreground branches can also carry parent ids before their state.db row
        is visible to the plugin.
        """
        if not session_id or session_id == parent_session_id:
            return False
        known_auxiliary_ids = self._known_auxiliary_lineage_session_ids()
        explicit_parent_id = str(kwargs.get("parent_session_id") or "")
        in_process_parent_id = self._in_process_parent_session_id(
            kwargs,
            session_id,
            include_explicit=False,
        )
        if in_process_parent_id:
            if not parent_session_id or in_process_parent_id == parent_session_id:
                return True
            if in_process_parent_id in known_auxiliary_ids:
                return True
        if explicit_parent_id:
            if self._thread_context_has_auxiliary_session(explicit_parent_id):
                return True
            if explicit_parent_id in known_auxiliary_ids and explicit_parent_id != self._session_id:
                return True
            return False
        if not parent_session_id:
            return False

        path = self._state_db_path(kwargs)
        if not path.exists():
            return False
        try:
            uri = path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                row = conn.execute(
                    """
                    SELECT
                        child.parent_session_id,
                        child.started_at,
                        child.ended_at,
                        parent.id,
                        parent.ended_at
                    FROM sessions AS child
                    LEFT JOIN sessions AS parent
                        ON parent.id = child.parent_session_id
                    WHERE child.id = ?
                    LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
            finally:
                conn.close()
        except Exception as exc:  # pragma: no cover - defensive against host DB drift
            logger.debug("LCM auxiliary child session probe failed: %s", exc)
            return False
        if not row:
            return False
        child_parent_id, child_started_at, child_ended_at, actual_parent_id, parent_ended_at = row
        if child_ended_at is not None or actual_parent_id is None:
            return False

        active_auxiliary_ids = self._active_auxiliary_session_ids()
        known_auxiliary_ids = self._known_auxiliary_lineage_session_ids()
        if child_parent_id in active_auxiliary_ids:
            return True
        if child_parent_id in known_auxiliary_ids and child_parent_id != self._session_id:
            return True
        if child_parent_id != parent_session_id:
            return self._session_has_auxiliary_ancestor(
                str(child_parent_id or ""),
                known_auxiliary_ids | active_auxiliary_ids,
                path,
            )
        return False

    def _session_has_auxiliary_ancestor(
        self,
        session_id: str,
        auxiliary_lineage_ids: set[str],
        state_db_path: Path,
    ) -> bool:
        if not session_id or not auxiliary_lineage_ids or not state_db_path.exists():
            return False
        visited: set[str] = set()
        current = session_id
        try:
            uri = state_db_path.resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            try:
                for _ in range(32):
                    if not current or current in visited:
                        return False
                    if current in auxiliary_lineage_ids:
                        return True
                    visited.add(current)
                    row = conn.execute(
                        "SELECT parent_session_id FROM sessions WHERE id = ? LIMIT 1",
                        (current,),
                    ).fetchone()
                    if not row:
                        return False
                    current = str(row[0] or "")
            finally:
                conn.close()
        except Exception as exc:  # pragma: no cover - defensive against host DB drift
            logger.debug("LCM auxiliary ancestor probe failed: %s", exc)
            return False
        return False

    def _clear_pending_reset_boundary(self) -> None:
        self._pending_reset_session_id = ""
        self._pending_reset_conversation_id = ""
        self._pending_reset_frontier_store_id = 0

    def _finalize_pending_reset_boundary(self, session_id: str) -> None:
        if not self._pending_reset_session_id:
            return
        if self._pending_reset_session_id != session_id:
            self._clear_pending_reset_boundary()
            return
        if not self._pending_reset_conversation_id:
            self._clear_pending_reset_boundary()
            return
        state = self._lifecycle.get_by_conversation(self._pending_reset_conversation_id)
        frontier_store_id = self._pending_reset_frontier_store_id
        if state is not None and state.current_session_id == session_id:
            frontier_store_id = max(
                frontier_store_id,
                int(state.current_frontier_store_id or 0),
            )
        self._lifecycle.finalize_session(
            self._pending_reset_conversation_id,
            self._pending_reset_session_id,
            frontier_store_id=frontier_store_id,
        )
        self._clear_pending_reset_boundary()

    def _raw_backlog_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        n = len(messages)
        fresh_tail_start = max(0, n - self._config.fresh_tail_count)
        leading_anchor_count = self._leading_anchor_count(messages)
        if fresh_tail_start <= leading_anchor_count:
            return []
        return messages[leading_anchor_count:fresh_tail_start]

    @staticmethod
    def _leading_anchor_count(messages: List[Dict[str, Any]]) -> int:
        """Return the number of non-compactable leading messages.

        Only the system prompt is a safe permanent anchor. Hermes gateway
        sessions can begin with a user message when core passes conversation
        history without a system prompt; preserving that first user turn as raw
        active context lets stale requests look current after later compaction.
        """
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            return 1
        return 0

    def _raw_backlog_tokens(self, messages: List[Dict[str, Any]]) -> int:
        backlog = self._raw_backlog_messages(messages)
        if not backlog:
            return 0
        return count_messages_tokens(backlog)

    def _raw_backlog_threshold(self, raw_tokens: int) -> int:
        if self._config.dynamic_leaf_chunk_enabled:
            return self._working_leaf_chunk_tokens(raw_tokens)
        return max(1, self._config.leaf_chunk_tokens)

    def _has_raw_backlog_debt(self) -> bool:
        if not self._config.deferred_maintenance_enabled or not self._conversation_id:
            return False
        state = self._lifecycle.get_by_conversation(self._conversation_id)
        return bool(state and state.debt_kind == "raw_backlog" and state.debt_size_estimate > 0)

    def _budget_pressure_ratio(
        self,
        *,
        observed_tokens: int | None = None,
        messages: List[Dict[str, Any]] | None = None,
    ) -> float | None:
        if self.context_length <= 0:
            return None
        token_count: int | None = None
        if observed_tokens is not None and observed_tokens > 0:
            token_count = observed_tokens
        elif messages is not None:
            token_count = count_messages_tokens(messages)
        elif self.last_prompt_tokens > 0:
            token_count = self.last_prompt_tokens
        if token_count is None or token_count <= 0:
            return None
        return token_count / self.context_length

    def _critical_budget_pressure_reached(
        self,
        *,
        observed_tokens: int | None = None,
        messages: List[Dict[str, Any]] | None = None,
    ) -> bool:
        threshold = self._config.critical_budget_pressure_ratio
        if threshold <= 0:
            return False
        pressure = self._budget_pressure_ratio(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        return pressure is not None and pressure >= threshold

    def _should_run_deferred_maintenance(
        self,
        messages: List[Dict[str, Any]],
        *,
        observed_tokens: int | None = None,
    ) -> bool:
        if not self._has_raw_backlog_debt():
            return False
        raw_tokens = self._raw_backlog_tokens(messages)
        if raw_tokens <= 0:
            return False
        if raw_tokens >= self._raw_backlog_threshold(raw_tokens):
            return True
        return self._critical_budget_pressure_reached(
            observed_tokens=observed_tokens,
            messages=messages,
        )

    def _refresh_raw_backlog_debt(
        self,
        messages: List[Dict[str, Any]],
        *,
        observed_tokens: int | None = None,
    ) -> None:
        if not self._config.deferred_maintenance_enabled or not self._conversation_id:
            return
        raw_tokens = self._raw_backlog_tokens(messages)
        threshold = self._raw_backlog_threshold(raw_tokens) if raw_tokens > 0 else 0
        keep_under_critical_pressure = (
            raw_tokens > 0
            and self._has_raw_backlog_debt()
            and self._critical_budget_pressure_reached(
                observed_tokens=observed_tokens,
                messages=messages,
            )
        )
        if raw_tokens > 0 and (raw_tokens >= threshold or keep_under_critical_pressure):
            self._lifecycle.record_debt(
                self._conversation_id,
                kind="raw_backlog",
                size_estimate=raw_tokens,
            )
            return
        if self._has_raw_backlog_debt():
            self._lifecycle.clear_debt(self._conversation_id)

    def _reset_session_counters(self) -> None:
        """Reset session-scoped counters and token tracking.

        Safe to call on boundary skip because it does not affect compaction progress.
        """
        self.compression_count = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cache_read_tokens = 0
        self.last_cache_write_tokens = 0
        self.last_reasoning_tokens = 0
        self.cache_metrics_available = False
        self._context_probed = False
        self._context_probe_persistable = False
        self._last_overflow_recovery_failed = False
        self._last_condensation_suppressed_reason = ""
        self._last_compression_status = "idle"
        self._last_compression_noop_reason = ""
        self._degraded = False
        self._last_degraded_reason = ""
        self._last_fail_open_at = 0.0
        self._last_fail_open_error = ""
        self._last_fail_open_error_type = ""
        self._last_fail_open_fallback_error = ""
        self._last_fail_open_fallback_error_type = ""
        self._last_fail_open_raw_tokens = 0
        self._last_fail_open_context_limit = 0
        self._last_fail_open_over_context_limit = False
        self._last_fail_open_fallback_attempted = False
        self._last_fail_open_fallback_status = ""
        self._fail_open_total_count = 0
        self._fail_open_consecutive_count = 0
        self._fail_open_timestamps = []
        self._last_boundary_skip_time = 0

    def _reset_compaction_progress(self) -> None:
        """Reset process-local compaction markers for a fresh/unproven session."""
        self._last_compacted_store_id = 0
        self._ingest_cursor = 0
        self._ingest_cursor_needs_reconcile = False
        self._last_ingest_reconciliation = {"action": "none", "reason": "not run"}

    def _reset_session_scoped_runtime_state(self) -> None:
        """Reset all session-scoped runtime state.

        Calls both _reset_session_counters and _reset_compaction_progress.
        Proven carry-over paths must restore/advance state from the verified
        source lifecycle after rebinding.
        """
        self._reset_session_counters()
        self._reset_compaction_progress()

    def _apply_session_start_metadata(self, session_id: str, kwargs: Dict[str, Any]) -> None:
        self._session_id = session_id
        self._session_platform = str(kwargs.get("platform") or "")
        self._refresh_session_filters()
        # Hold the foreground view stable when the new binding is a side
        # channel (cron tick inside the gateway process, debug probe, etc.).
        # Tools that report "current session" to operators must keep pointing
        # at the real foreground rather than the ignored/stateless session
        # that just stole _session_id. Lifecycle paths still read _session_id
        # directly so cron's compress short-circuits correctly via the
        # _session_ignored / _session_stateless gates.
        if not self._session_ignored and not self._session_stateless:
            self._foreground_session_id = session_id
            self._foreground_session_platform = self._session_platform
        if "hermes_home" in kwargs:
            self._hermes_home = kwargs["hermes_home"]

        update_model_is_authoritative = (
            self._context_length_source == "update_model"
            and self._update_model_pending_session_start
        )

        # Pick up context_length from kwargs if provided, but do not let stale
        # session metadata undo the authoritative runtime update_model() call.
        # Hermes Agent calls update_model() with the resolver output before it
        # binds a fresh agent/session.  Older or buggy host paths can still pass
        # a context_length copied from the previously bound runtime; treating
        # that as authoritative makes /model switches keep compressing against
        # the old model window.
        if "context_length" in kwargs:
            incoming_context_length = kwargs["context_length"]
            try:
                parsed_context_length = int(incoming_context_length)
            except (TypeError, ValueError):
                logger.debug(
                    "LCM ignored invalid session-start context_length: %r",
                    incoming_context_length,
                )
                self._update_model_pending_session_start = False
                return
            if parsed_context_length <= 0:
                if update_model_is_authoritative:
                    if self._session_metadata_matches_active_runtime(
                        kwargs,
                        ignore_empty_optional=True,
                    ):
                        logger.debug(
                            "LCM ignored missing session-start context_length=%r for model=%s; active update_model context_length=%s",
                            incoming_context_length,
                            self.model or str(kwargs.get("model") or ""),
                            self.context_length,
                        )
                    else:
                        logger.warning(
                            "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                            str(kwargs.get("model") or ""),
                            self.model,
                        )
                    self._update_model_pending_session_start = False
                    return
                self._set_context_length(parsed_context_length, source="session_start")
                update_model_is_authoritative = False
            else:
                if (
                    update_model_is_authoritative
                    and parsed_context_length != self.context_length
                ):
                    logger.warning(
                        "LCM ignored stale session-start context_length=%s for model=%s; active update_model context_length=%s",
                        parsed_context_length,
                        self.model or str(kwargs.get("model") or ""),
                        self.context_length,
                    )
                    self._update_model_pending_session_start = False
                    return
                if update_model_is_authoritative:
                    if not self._session_metadata_matches_active_runtime(kwargs):
                        logger.warning(
                            "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                            str(kwargs.get("model") or ""),
                            self.model,
                        )
                        self._update_model_pending_session_start = False
                        return
                else:
                    self._set_context_length(parsed_context_length, source="session_start")
                    update_model_is_authoritative = False
        if (
            update_model_is_authoritative
            and not self._session_metadata_matches_active_runtime(kwargs)
        ):
            logger.warning(
                "LCM ignored stale session-start runtime metadata for model=%s; active update_model model=%s",
                str(kwargs.get("model") or ""),
                self.model,
            )
            self._update_model_pending_session_start = False
            return
        if "model" in kwargs:
            self.model = str(kwargs.get("model") or "")
        for key in ("base_url", "api_key", "provider", "api_mode"):
            if key in kwargs:
                setattr(self, key, str(kwargs.get(key) or ""))
        self._update_model_pending_session_start = False

    def _continue_compression_boundary(
        self,
        session_id: str,
        old_session_id: str,
        kwargs: Dict[str, Any],
    ) -> None:
        previous_session_id = self._session_id
        requested_conversation_id = kwargs.get("conversation_id")
        session_state = self._lifecycle.get_by_session(old_session_id)
        conversation_state = self._lifecycle.get_by_conversation(old_session_id)

        def _state_conversation_matches(state: Any) -> bool:
            return bool(
                state
                and (
                    not requested_conversation_id
                    or state.conversation_id == requested_conversation_id
                )
            )

        def _has_summary_nodes(candidate_session_id: str | None) -> bool:
            return bool(candidate_session_id and self._dag.get_session_nodes(candidate_session_id))

        def _host_source_from_conversation_state(state: Any) -> tuple[str, Any]:
            if not _state_conversation_matches(state):
                return "", None
            if state.current_session_id == old_session_id and _has_summary_nodes(old_session_id):
                return old_session_id, state
            if (
                state.conversation_id == old_session_id
                and state.current_session_id
                and _has_summary_nodes(state.current_session_id)
            ):
                return state.current_session_id, state
            if (
                state.current_session_id is None
                and state.last_finalized_session_id
                and _has_summary_nodes(state.last_finalized_session_id)
            ):
                return state.last_finalized_session_id, state
            return "", None

        def _host_source_from_session_state(state: Any) -> tuple[str, Any]:
            if not _state_conversation_matches(state):
                return "", None
            if state.current_session_id == old_session_id and _has_summary_nodes(old_session_id):
                return old_session_id, state
            if (
                state.current_session_id is None
                and state.last_finalized_session_id == old_session_id
                and _has_summary_nodes(old_session_id)
            ):
                return old_session_id, state
            return "", None

        host_source_session_id, host_source_state = _host_source_from_conversation_state(
            conversation_state
        )
        if not host_source_session_id:
            host_source_session_id, host_source_state = _host_source_from_session_state(
                session_state
            )

        source_session_id = host_source_session_id or old_session_id
        source_state = host_source_state or session_state

        if previous_session_id and previous_session_id != old_session_id:
            # Hermes passes the session that actually crossed the compression
            # boundary as old_session_id. A different bound session can be a
            # short-lived subagent/cron/WebUI side channel that ran after the
            # foreground compaction. Prefer the host-authoritative source when
            # durable lifecycle + DAG evidence proves it belongs to LCM, then
            # fall back to the older bound-session recovery path. When the host
            # old_session_id is the durable conversation id, use that row's
            # current/finalized LCM source instead of unrelated auxiliary rows
            # where the id appears only as last_finalized_session_id.
            if host_source_session_id:
                logger.warning(
                    "LCM compression boundary using host old_session_id %s as carry-over source=%s despite bound session drift=%s",
                    old_session_id,
                    host_source_session_id,
                    previous_session_id,
                )
            else:
                bound_state = self._lifecycle.get_by_session(previous_session_id)
                bound_conversation_matches = bool(
                    bound_state
                    and (not self._conversation_id or bound_state.conversation_id == self._conversation_id)
                    and (
                        not requested_conversation_id
                        or bound_state.conversation_id == requested_conversation_id
                    )
                )
                bound_is_active_source = bool(
                    bound_state and bound_state.current_session_id == previous_session_id
                )
                bound_is_finalized_source = bool(
                    bound_state
                    and bound_state.current_session_id is None
                    and bound_state.last_finalized_session_id == previous_session_id
                )
                bound_has_summary_nodes = bool(self._dag.get_session_nodes(previous_session_id))
                if (
                    bound_conversation_matches
                    and (bound_is_active_source or bound_is_finalized_source)
                    and bound_has_summary_nodes
                ):
                    source_session_id = previous_session_id
                    source_state = bound_state
                    logger.warning(
                        "LCM compression boundary using bound session %s as carry-over source; host old_session_id=%s does not match",
                        previous_session_id,
                        old_session_id,
                    )
                else:
                    # Fallback: sibling chain with zero-DAG parent.
                    # When stale old_session_id has no DAG nodes AND the
                    # bound session belongs to a different conversation_id
                    # but shares the same last_finalized_session_id
                    # (parent) — prefer the bound session despite the
                    # conversation_id mismatch. This handles the lifecycle
                    # fork case where two sessions on the same channel
                    # received different conversation_ids.
                    bound_shares_parent_with_host = bool(
                        bound_state
                        and bound_state.last_finalized_session_id == old_session_id
                    )
                    host_has_no_dag = not bool(
                        self._dag.get_session_nodes(old_session_id)
                    )
                    if (
                        bound_shares_parent_with_host
                        and host_has_no_dag
                        and (bound_is_active_source or bound_is_finalized_source)
                        and bound_has_summary_nodes
                    ):
                        source_session_id = previous_session_id
                        source_state = bound_state
                        logger.warning(
                            "LCM compression boundary using bound session %s on sibling chain as carry-over source; host old_session_id=%s has zero DAG, parent=%s matches",
                            previous_session_id,
                            old_session_id,
                            bound_state.last_finalized_session_id,
                        )
                    else:
                        source_session_id = ""
                        source_state = None

        conversation_id = (
            (source_state.conversation_id if source_state else None)
            or kwargs.get("conversation_id")
            or self._conversation_id
            or source_session_id
            or old_session_id
            or session_id
        )
        process_local_frontier = (
            int(self._last_compacted_store_id or 0)
            if source_session_id and previous_session_id == source_session_id
            else 0
        )
        pending_reset_frontier = int(
            self._pending_reset_frontier_store_id
            if self._pending_reset_session_id
            and self._pending_reset_session_id == source_session_id
            else 0
        )
        frontier = max(
            process_local_frontier,
            int(source_state.current_frontier_store_id if source_state else 0),
            int(source_state.last_finalized_frontier_store_id if source_state else 0),
            pending_reset_frontier,
        )
        can_reassign = bool(
            source_session_id
            and session_id
            and source_session_id != session_id
        )

        if can_reassign:
            self._lifecycle.finalize_session(
                conversation_id,
                source_session_id,
                frontier_store_id=frontier,
            )
            moved_messages = self._store.reassign_session_messages(source_session_id, session_id)
            moved_nodes = self._dag.reassign_session_nodes(source_session_id, session_id)
            moved_payloads = reassign_externalized_payloads(
                source_session_id,
                session_id,
                config=self._config,
                hermes_home=self._hermes_home,
            )
            logger.debug(
                "LCM compression boundary continued %s -> %s: moved %d messages, %d DAG nodes, %d externalized payloads",
                source_session_id,
                session_id,
                moved_messages,
                moved_nodes,
                moved_payloads,
            )
        elif old_session_id:
            logger.warning(
                "LCM compression boundary skipped carry-over: old_session_id=%s does not match bound session=%s",
                old_session_id,
                previous_session_id,
            )
            self._finalize_pending_reset_boundary(previous_session_id)
            self._reset_session_scoped_runtime_state()
            self._last_boundary_skip_time = time.time()
            self._apply_session_start_metadata(session_id, kwargs)
            self._bind_lifecycle_state(
                session_id,
                conversation_id=kwargs.get("conversation_id"),
            )
            self._schedule_ingest_cursor_reconciliation()
            self._clear_pending_reset_boundary()
            self._log_session_filter_diagnostics()
            return

        self._apply_session_start_metadata(session_id, kwargs)
        self._bind_lifecycle_state(session_id, conversation_id=conversation_id)
        if frontier > 0:
            state = self._lifecycle.advance_frontier(
                self._conversation_id,
                session_id,
                frontier,
            )
            if state is not None:
                self._last_compacted_store_id = state.current_frontier_store_id
        self._clear_pending_reset_boundary()
        self._log_session_filter_diagnostics()

    def on_session_start(self, session_id: str, **kwargs) -> None:
        if "hermes_home" in kwargs:
            self._rebind_storage_for_home(str(kwargs.get("hermes_home") or ""))

        boundary_reason = str(kwargs.get("boundary_reason") or "")
        old_session_id = str(kwargs.get("old_session_id") or "")
        previous_session_id = self._session_id
        if boundary_reason == "compression" and old_session_id and old_session_id != session_id:
            if (
                self._has_auxiliary_lineage_session(old_session_id)
                and old_session_id != self._session_id
            ):
                self._handoff_auxiliary_session(old_session_id, session_id)
                logger.info(
                    "LCM auxiliary session %s compressed to %s — keeping boundary stateless",
                    old_session_id,
                    session_id,
                )
                return
            self._clear_thread_context_stateless()
            self._continue_compression_boundary(session_id, old_session_id, kwargs)
            return

        if self._is_live_auxiliary_child_session(session_id, previous_session_id, kwargs):
            self._register_auxiliary_session(session_id)
            logger.info(
                "LCM session %s is a live child of bound session %s — treating it as auxiliary/stateless",
                session_id,
                previous_session_id,
            )
            return
        self._deactivate_auxiliary_session(session_id)
        self._clear_thread_context_stateless()
        if previous_session_id and previous_session_id != session_id:
            self._finalize_pending_reset_boundary(previous_session_id)
            self._reset_session_scoped_runtime_state()
        else:
            self._clear_pending_reset_boundary()
            self._ingest_cursor = 0
            self._last_compacted_store_id = 0
            self._last_overflow_recovery_failed = False
            self._last_condensation_suppressed_reason = ""
        self._apply_session_start_metadata(session_id, kwargs)
        self._bind_lifecycle_state(
            session_id,
            conversation_id=kwargs.get("conversation_id"),
        )
        self._schedule_ingest_cursor_reconciliation()
        self._log_session_filter_diagnostics()

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        if self._has_auxiliary_lineage_session(session_id) and session_id != self._session_id:
            current_thread_session_id = self._thread_context_session_id()
            with self._auxiliary_session_lock:
                self._auxiliary_session_ids.discard(session_id)
            if current_thread_session_id == session_id:
                self._clear_thread_context_stateless(session_id)
            return
        try:
            with _temporary_sqlite_busy_timeout(
                [
                    getattr(self._store, "_conn", None),
                    getattr(self._lifecycle, "_conn", None),
                ],
                _SESSION_END_BUSY_TIMEOUT_MS,
            ):
                try:
                    # Best-effort final flush. Keep this path bounded because
                    # host gateways call session-end hooks from lifecycle paths
                    # that must not wait through SQLite's normal busy timeout.
                    self._ingest_messages(messages)
                except KeyboardInterrupt:
                    logger.warning(
                        "LCM session-end raw-message ingest interrupted; "
                        "final messages may be absent from the plugin-local store"
                    )
                    return
                except Exception as exc:
                    if _is_sqlite_locked_error(exc):
                        logger.warning(
                            "LCM session-end raw-message ingest skipped due to SQLite lock after short wait; "
                            "final messages may be absent from the plugin-local store: %s",
                            exc,
                        )
                        return
                    raise

                try:
                    self._lifecycle.finalize_session(
                        self._conversation_id,
                        session_id,
                        frontier_store_id=self._last_compacted_store_id,
                    )
                except KeyboardInterrupt:
                    logger.warning(
                        "LCM session-end lifecycle finalization interrupted; "
                        "raw messages may be ingested but lifecycle state may be finalized later"
                    )
                    return
                except Exception as exc:
                    if _is_sqlite_locked_error(exc):
                        logger.warning(
                            "LCM session-end lifecycle finalization skipped due to SQLite lock after short wait; "
                            "raw messages were ingested but lifecycle state may be finalized later: %s",
                            exc,
                        )
                        return
                    raise
        except KeyboardInterrupt:
            logger.warning("LCM session-end ingest/finalize interrupted before bounded flush completed")
            return
        except Exception as exc:
            if _is_sqlite_locked_error(exc):
                logger.warning(
                    "LCM session-end ingest/finalize skipped due to SQLite lock before bounded flush: %s",
                    exc,
                )
                return
            raise

    def on_session_reset(self) -> None:
        self._pending_reset_session_id = self._session_id
        self._pending_reset_conversation_id = self._conversation_id
        self._pending_reset_frontier_store_id = self._last_compacted_store_id
        super().on_session_reset()
        self._lifecycle.record_reset(self._conversation_id)
        self._reset_session_scoped_runtime_state()

        # Retain DAG nodes across sessions based on config.
        #   -1  → keep all nodes
        #    0  → delete everything
        #    N  → keep nodes at depth >= N (e.g. 2 keeps d2+)
        retain = self._config.new_session_retain_depth
        if self._session_id and retain != -1:
            if retain == 0:
                self._dag.delete_session_nodes(self._session_id)
            else:
                self._dag.delete_below_depth(self._session_id, retain)

    def carry_over_new_session_context(self, old_session_id: str, new_session_id: str) -> int:
        """Move retained summaries from the old session into the new one.

        This reassigns session ownership for retained summary nodes, but it does
        not rewrite the nodes' descendant raw-message lineage. Retrieval under
        ``session_scope='current'`` may therefore include a carried-over node in
        the new session, while ``source`` filtering still evaluates against the
        node's original descendant message sources.
        """
        if not old_session_id or not new_session_id or old_session_id == new_session_id:
            return 0
        if self._session_ignored and new_session_id == self._session_id:
            logger.debug(
                "LCM carry-over skipped for ignored session %s",
                new_session_id,
            )
            return 0
        return self._dag.reassign_session_nodes(old_session_id, new_session_id)

    def rollover_session(
        self,
        old_session_id: str,
        new_session_id: str,
        previous_messages: List[Dict[str, Any]] | None = None,
        carry_over_context: bool = True,
        **kwargs,
    ) -> int:
        """Complete a Hermes-style `/new` rollover for this engine.

        This is a small helper for host/runtime integrations that need the
        correct lifecycle ordering in one call:
        1. flush old-session messages into the store
        2. prune/reset retained DAG state on the old session
        3. bind the engine to the new session
        4. optionally move retained summaries into the new session
        """
        previous_messages = previous_messages or []
        boundary_reason = str(kwargs.get("boundary_reason") or "")
        conversation_id = self._conversation_id or old_session_id or new_session_id
        bound_session_id = self._session_id
        can_carry_over = bool(
            old_session_id and bound_session_id and old_session_id == bound_session_id
        )

        if carry_over_context and boundary_reason == "compression" and old_session_id and old_session_id != new_session_id:
            before_node_ids = {node.node_id for node in self._dag.get_session_nodes(new_session_id)}
            if can_carry_over:
                self.on_session_end(old_session_id, previous_messages)
            else:
                logger.warning(
                    "LCM compression rollover old_session_id=%s does not match bound session=%s; using boundary handler fallback",
                    old_session_id,
                    bound_session_id,
                )
            self.on_session_start(
                new_session_id,
                old_session_id=old_session_id,
                **kwargs,
            )
            after_node_ids = {node.node_id for node in self._dag.get_session_nodes(new_session_id)}
            return len(after_node_ids - before_node_ids)

        if old_session_id and can_carry_over:
            self.on_session_end(old_session_id, previous_messages)
            self.on_session_reset()
        elif old_session_id and not carry_over_context:
            logger.warning(
                "LCM rollover skipped old-session finalization: old_session_id=%s does not match bound session=%s",
                old_session_id,
                bound_session_id,
            )
        elif old_session_id and not can_carry_over:
            logger.warning(
                "LCM carry-over skipped: old_session_id=%s does not match bound session=%s",
                old_session_id,
                bound_session_id,
            )

        self.on_session_start(new_session_id, conversation_id=conversation_id, **kwargs)

        if not carry_over_context:
            return 0
        if old_session_id and not can_carry_over:
            return 0
        return self.carry_over_new_session_context(old_session_id, new_session_id)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {"type": "function", "function": schema}
            for schema in (
                LCM_GREP,
                LCM_LOAD_SESSION,
                LCM_DESCRIBE,
                LCM_EXPAND,
                LCM_EXPAND_QUERY,
                LCM_STATUS,
                LCM_DOCTOR,
            )
        ]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        # Ingest live messages if passed (enables current-turn search)
        messages = kwargs.get("messages")

        if messages and self._session_id and not (
            self._session_ignored or self._session_stateless or self._thread_context_stateless()
        ):
            try:
                self._ingest_messages(messages)
            except Exception as e:
                logger.warning("Ingest during tool call failed: %s", e)

        handlers = {
            "lcm_grep": lcm_tools.lcm_grep,
            "lcm_load_session": lcm_tools.lcm_load_session,
            "lcm_describe": lcm_tools.lcm_describe,
            "lcm_expand": lcm_tools.lcm_expand,
            "lcm_expand_query": lcm_tools.lcm_expand_query,
            "lcm_status": lcm_tools.lcm_status,
            "lcm_doctor": lcm_tools.lcm_doctor,
        }
        handler = handlers.get(name)
        if handler:
            return handler(args, engine=self)
        return json.dumps({"error": f"Unknown LCM tool: {name}"})

    def _database_path_source(self) -> str:
        if self._config.database_path:
            return "config.database_path"
        if self._hermes_home:
            return "hermes_home"
        return "default_home"

    def get_runtime_identity(self) -> Dict[str, Any]:
        """Return operator-facing identity for the loaded LCM runtime.

        The public identity follows the same foreground-session view as
        ``lcm_status`` and other tools. When a side-channel session is bound,
        the bound session details are still exposed separately for diagnostics.
        """
        metadata = _plugin_metadata()
        git_identity = _git_runtime_identity(_PLUGIN_ROOT)
        session_id = self.current_session_id
        conversation_id = self.current_conversation_id
        lifecycle_state = None
        lifecycle_error = ""
        if conversation_id:
            try:
                lifecycle_state = self._lifecycle.get_by_conversation(conversation_id)
            except Exception as exc:  # pragma: no cover - defensive
                lifecycle_error = str(exc)

        identity: Dict[str, Any] = {
            "engine": self.name,
            "plugin_name": metadata.get("name", "hermes-lcm"),
            "plugin_version": metadata.get("version", "unknown"),
            "plugin_path": str(_PLUGIN_ROOT),
            "module_path": str(Path(__file__).resolve()),
            "hermes_home": str(self._hermes_home or ""),
            "database_path": str(self._store.db_path),
            "database_path_source": self._database_path_source(),
            "session_id": session_id,
            "session_platform": self.current_session_platform,
            "session_bound": bool(session_id),
            "conversation_id": conversation_id,
            "lifecycle_current_session_id": "",
            "lifecycle_last_finalized_session_id": "",
        }
        if self.side_channel_active:
            identity.update({
                "bound_session_id": self._session_id,
                "bound_session_platform": self._session_platform,
                "bound_conversation_id": self._conversation_id,
            })
        identity.update(git_identity)
        if lifecycle_state is not None:
            identity.update({
                "lifecycle_current_session_id": lifecycle_state.current_session_id or "",
                "lifecycle_last_finalized_session_id": lifecycle_state.last_finalized_session_id or "",
            })
        if lifecycle_error:
            identity["lifecycle_error"] = lifecycle_error
        return identity

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        fail_open_window_count, fail_open_rate_per_minute = self._fail_open_rate_snapshot()
        status.update({
            "compression_count": self.compression_count,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_completion_tokens": self.last_completion_tokens,
            "last_total_tokens": self.last_total_tokens,
            "last_input_tokens": self.last_input_tokens,
            "last_output_tokens": self.last_output_tokens,
            "last_cache_read_tokens": self.last_cache_read_tokens,
            "last_cache_write_tokens": self.last_cache_write_tokens,
            "last_reasoning_tokens": self.last_reasoning_tokens,
            "cache_metrics_available": self.cache_metrics_available,
            "cache_read_ratio": round(self.cache_read_ratio, 4),
            "context_length": self.context_length,
            "threshold_tokens": self.threshold_tokens,
            "last_compression_status": self._last_compression_status,
            "last_compression_noop_reason": self._last_compression_noop_reason,
            "degraded": self._degraded,
            "context_engine_degraded": self._degraded,
            "degraded_reason": self._last_degraded_reason,
            "last_fail_open_at": self._last_fail_open_at or None,
            "last_fail_open_error": self._last_fail_open_error,
            "last_fail_open_error_type": self._last_fail_open_error_type,
            "last_fail_open_fallback_error": self._last_fail_open_fallback_error,
            "last_fail_open_fallback_error_type": self._last_fail_open_fallback_error_type,
            "last_fail_open_raw_tokens": self._last_fail_open_raw_tokens,
            "last_fail_open_context_limit": self._last_fail_open_context_limit,
            "last_fail_open_over_context_limit": self._last_fail_open_over_context_limit,
            "last_fail_open_fallback_attempted": self._last_fail_open_fallback_attempted,
            "last_fail_open_fallback_status": self._last_fail_open_fallback_status,
            "fail_open_total_count": self._fail_open_total_count,
            "fail_open_consecutive_count": self._fail_open_consecutive_count,
            "fail_open_window_seconds": int(_FAIL_OPEN_RATE_WINDOW_SECONDS),
            "fail_open_window_count": fail_open_window_count,
            "fail_open_rate_per_minute": round(fail_open_rate_per_minute, 4),
            "model": self.model,
            "provider": self.provider,
            "context_length_source": self._context_length_source,
            "context_threshold": self._config.context_threshold,
        })
        session_id = self.current_session_id
        conversation_id = self.current_conversation_id
        lifecycle_state = self._lifecycle.get_by_conversation(conversation_id) if conversation_id else None
        status["engine"] = "lcm"
        status["runtime_identity"] = self.get_runtime_identity()
        status["ingest_protection"] = sensitive_pattern_status(self._config)
        try:
            status["source_lineage"] = self._store.get_source_stats(session_id or None)
        except Exception as exc:  # pragma: no cover - defensive
            status["source_lineage"] = {"error": str(exc)}
        try:
            state_db_path = (
                Path(self._hermes_home).expanduser() / "state.db"
                if self._hermes_home
                else Path(self._store.db_path).parent / "state.db"
            )
            status["lifecycle_fragmentation"] = self._lifecycle.get_fragmentation_stats(
                state_db_path=state_db_path
            )
        except Exception as exc:  # pragma: no cover - defensive
            status["lifecycle_fragmentation"] = {"error": str(exc), "read_only": True}
        try:
            rotate_backup_path = self.rotate_backup_path()
            status["rotate_backup_path"] = str(rotate_backup_path)
            # Single stat() to avoid a TOCTOU window where the rolling slot
            # could be atomically replaced between separate mtime and size reads.
            try:
                rotate_stat = rotate_backup_path.stat()
            except FileNotFoundError:
                rotate_stat = None
            if rotate_stat is not None:
                status["last_rotate_at"] = rotate_stat.st_mtime
                status["rotate_backup_size"] = rotate_stat.st_size
            else:
                status["last_rotate_at"] = None
                status["rotate_backup_size"] = 0
        except Exception as exc:  # pragma: no cover - defensive
            status["rotate_backup_path"] = None
            status["last_rotate_at"] = None
            status["rotate_backup_size"] = 0
            status["rotate_backup_error"] = str(exc)
        try:
            status["storage_retention"] = retention_status(
                self._store._conn,
                self._store.db_path,
                ttl_days=self._config.retention_ttl_days,
                max_bytes=self._config.retention_max_bytes,
            )
        except Exception as exc:  # pragma: no cover - defensive
            status["storage_retention"] = {"error": str(exc)}
        if session_id:
            status["store_messages"] = self._store.get_session_count(session_id)
            status["dag_nodes"] = len(self._dag.get_session_nodes(session_id))
            status["session_platform"] = self.current_session_platform
            status["session_ignored"] = self.current_session_ignored
            status["session_stateless"] = self.current_session_stateless
            status["ignore_session_patterns"] = list(self._config.ignore_session_patterns)
            status["stateless_session_patterns"] = list(self._config.stateless_session_patterns)
            status["ignore_message_patterns"] = list(self._config.ignore_message_patterns)
            status["ignore_session_patterns_source"] = self._config.ignore_session_patterns_source
            status["stateless_session_patterns_source"] = self._config.stateless_session_patterns_source
            status["ignore_message_patterns_source"] = self._config.ignore_message_patterns_source
            status["ignored_message_count"] = self._ignored_message_count
            status["ingest_reconciliation"] = dict(self._last_ingest_reconciliation)
            status["overflow_recovery_failed"] = self._last_overflow_recovery_failed
            status["condensation_suppressed_reason"] = self._last_condensation_suppressed_reason
            status["conversation_id"] = conversation_id
            if lifecycle_state is not None:
                status["lifecycle"] = {
                    "conversation_id": lifecycle_state.conversation_id,
                    "current_session_id": lifecycle_state.current_session_id,
                    "last_finalized_session_id": lifecycle_state.last_finalized_session_id,
                    "current_frontier_store_id": lifecycle_state.current_frontier_store_id,
                    "last_finalized_frontier_store_id": lifecycle_state.last_finalized_frontier_store_id,
                    "debt_kind": lifecycle_state.debt_kind,
                    "debt_size_estimate": lifecycle_state.debt_size_estimate,
                    "current_bound_at": lifecycle_state.current_bound_at,
                    "last_finalized_at": lifecycle_state.last_finalized_at,
                    "debt_updated_at": lifecycle_state.debt_updated_at,
                    "last_maintenance_attempt_at": lifecycle_state.last_maintenance_attempt_at,
                    "last_rollover_at": lifecycle_state.last_rollover_at,
                    "last_reset_at": lifecycle_state.last_reset_at,
                    "updated_at": lifecycle_state.updated_at,
                }
        return status

    def update_model(self, model: str, context_length: int,
                     base_url: str = "", api_key: str = "",
                     provider: str = "",
                     api_mode: str = "") -> None:
        parent_session_id = self._in_process_parent_session_id({})
        if parent_session_id:
            logger.debug(
                "LCM model update ignored for auxiliary child of %s",
                parent_session_id,
            )
            return
        self.model = str(model or "")
        self.base_url = str(base_url or "")
        self.api_key = str(api_key or "")
        self.provider = str(provider or "")
        self.api_mode = str(api_mode or "")
        self._set_context_length(context_length, source="update_model")
        self._update_model_pending_session_start = True

    def _refresh_session_filters(self) -> None:
        self._session_match_keys = build_session_match_keys(
            self._session_id,
            platform=self._session_platform,
        )
        self._session_ignored = matches_session_pattern(
            self._session_match_keys,
            self._compiled_ignore_session_patterns,
        )
        self._session_stateless = (
            not self._session_ignored
            and matches_session_pattern(
                self._session_match_keys,
                self._compiled_stateless_session_patterns,
            )
        )

    def _log_session_filter_diagnostics(self) -> None:
        if not self._logged_filter_config:
            if self._config.ignore_session_patterns:
                logger.info(
                    "LCM ignore_session_patterns from %s: %s",
                    self._config.ignore_session_patterns_source,
                    ", ".join(self._config.ignore_session_patterns),
                )
            if self._config.stateless_session_patterns:
                logger.info(
                    "LCM stateless_session_patterns from %s: %s",
                    self._config.stateless_session_patterns_source,
                    ", ".join(self._config.stateless_session_patterns),
                )
            if self._config.ignore_message_patterns:
                logger.info(
                    "LCM ignore_message_patterns from %s: %s",
                    self._config.ignore_message_patterns_source,
                    ", ".join(self._config.ignore_message_patterns),
                )
            self._logged_filter_config = True
        if self._session_ignored:
            logger.info(
                "LCM session %s matched ignore_session_patterns via %s — skipping writes and compaction",
                self._session_id,
                ", ".join(self._session_match_keys),
            )
        elif self._session_stateless:
            logger.info(
                "LCM session %s matched stateless_session_patterns via %s — read-only mode (no LCM writes)",
                self._session_id,
                ", ".join(self._session_match_keys),
            )

    # -- Internal: message ingestion ---------------------------------------

    def _schedule_ingest_cursor_reconciliation(self) -> None:
        """Mark existing-session rebinds for cursor repair on next ingest."""
        self._ingest_cursor_needs_reconcile = False
        if not self._session_id or self._session_ignored or self._session_stateless:
            return
        try:
            self._ingest_cursor_needs_reconcile = self._store.get_session_count(self._session_id) > 0
        except Exception as exc:  # pragma: no cover - defensive only
            logger.debug("LCM ingest cursor reconciliation probe failed: %s", exc)
            self._ingest_cursor_needs_reconcile = False

    def _stored_row_externalized_text_for_pattern_matching(self, msg: Dict[str, Any]) -> str:
        content = msg.get("content")
        if not isinstance(content, str):
            return ""
        refs = extract_ingest_externalized_refs(content)
        legacy_ref = extract_externalized_ref(content)
        if legacy_ref and legacy_ref not in refs:
            refs.append(legacy_ref)
        parts: list[str] = []
        session_id = str(msg.get("session_id") or self._session_id or "")
        for ref in refs:
            payload = load_externalized_payload(
                ref,
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if not payload:
                continue
            payload_session_id = str(payload.get("session_id") or "")
            if session_id and payload_session_id and payload_session_id != session_id:
                continue
            payload_content = payload.get("content")
            if isinstance(payload_content, str):
                parts.append(payload_content)
        return "\n".join(parts)

    @staticmethod
    def _is_volatile_ignored_quarantine_placeholder(msg: Dict[str, Any], text: str) -> bool:
        if str(msg.get("role") or "") != "assistant":
            return False
        return bool(
            re.fullmatch(
                r"\[LCM active replay placeholder: assistant output quarantined; "
                r"kind=quarantined_assistant_output; "
                r"reason=[A-Za-z0-9_.:/-]+; "
                r"scope=ignored_message_pattern; field=content; "
                r"chars=\d+; bytes=\d+; "
                r"sha256=[0-9a-f]{16}\]",
                text.strip(),
            )
        )

    def _matches_ignore_message_patterns(self, msg: Dict[str, Any], *, stored_row: bool = False) -> bool:
        if not self._compiled_ignore_message_patterns:
            return False
        content = msg.get("content")
        text = (
            stored_text_content_for_pattern_matching(content)
            if stored_row
            else text_content_for_pattern_matching(content)
        ) or ""
        if matches_message_pattern(text, self._compiled_ignore_message_patterns):
            return True
        if stored_row:
            externalized_text = self._stored_row_externalized_text_for_pattern_matching(msg)
            if externalized_text and externalized_text != text:
                return matches_message_pattern(externalized_text, self._compiled_ignore_message_patterns)
        return False

    def _is_replayed_context_scaffold_message(self, msg: Dict[str, Any]) -> bool:
        """Return true for active-context scaffolding that should not be re-ingested."""
        role = str(msg.get("role") or "")
        content = normalize_content_value(msg.get("content")) or ""
        if role == "system":
            return (
                "[Note: This conversation uses Lossless Context Management (LCM)." in content
                and "Earlier turns have been compacted into hierarchical summaries below." in content
            )
        if content.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX):
            return True
        if "[Expand for details:" not in content:
            return False
        return bool(
            re.search(
                r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]",
                content,
            )
        )

    @staticmethod
    def _canonicalize_tool_call_identity_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: LCMEngine._canonicalize_tool_call_identity_value(val)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [LCMEngine._canonicalize_tool_call_identity_value(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped[0] in "[{":
                if _json_has_duplicate_object_keys(value):
                    return value
                try:
                    parsed = json.loads(value)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return value
                if isinstance(parsed, (dict, list)):
                    canonical = LCMEngine._canonicalize_tool_call_identity_value(parsed)
                    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return value
        return value

    @staticmethod
    def _stable_tool_calls_identity(tool_calls: Any) -> str:
        if not tool_calls:
            return ""
        try:
            canonical = LCMEngine._canonicalize_tool_call_identity_value(tool_calls)
            return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            return str(tool_calls)

    def _restore_ingest_payload_placeholders_in_value(self, value: Any, *, session_id: str) -> Any:
        if isinstance(value, dict):
            return {
                self._restore_ingest_payload_placeholders_in_value(key, session_id=session_id)
                if isinstance(key, str)
                else key: self._restore_ingest_payload_placeholders_in_value(val, session_id=session_id)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [self._restore_ingest_payload_placeholders_in_value(item, session_id=session_id) for item in value]
        if isinstance(value, str):
            return restore_ingest_payload_placeholders(
                value,
                config=self._config,
                hermes_home=self._hermes_home,
                session_id=session_id,
            )
        return value

    def _restore_ingest_payload_placeholders_in_content_identity(self, content: str, *, session_id: str) -> str:
        if not content:
            return content
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return restore_ingest_payload_placeholders(
                content,
                config=self._config,
                hermes_home=self._hermes_home,
                session_id=session_id,
            )
        restore_as_structured = False
        if isinstance(decoded, (dict, list)) and normalize_content_value(decoded) == content:
            for ref in extract_ingest_externalized_refs(content):
                payload = load_externalized_payload(
                    ref,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                payload_session_id = (payload or {}).get("session_id") or ""
                if session_id and payload_session_id and payload_session_id != session_id:
                    continue
                field_path = str((payload or {}).get("field_path") or "")
                if field_path and field_path != "content":
                    restore_as_structured = True
                    break
        if restore_as_structured:
            restored = self._restore_ingest_payload_placeholders_in_value(decoded, session_id=session_id)
            return normalize_content_value(restored) or ""
        return restore_ingest_payload_placeholders(
            content,
            config=self._config,
            hermes_home=self._hermes_home,
            session_id=session_id,
        )

    def _message_replay_identity(self, msg: Dict[str, Any], *, stored_row: bool = False) -> tuple[str, str, str, str]:
        content = normalize_content_value(msg.get("content")) or ""
        tool_calls = msg.get("tool_calls")
        if stored_row:
            session_id = str(msg.get("session_id") or self._session_id or "")
            content = self._restore_ingest_payload_placeholders_in_content_identity(
                content,
                session_id=session_id,
            )
            tool_calls = self._restore_ingest_payload_placeholders_in_value(tool_calls, session_id=session_id)
        ref = extract_externalized_ref(content)
        if ref and "quarantined_assistant_output" not in content:
            payload = load_externalized_payload(
                ref,
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if payload is not None and isinstance(payload.get("content"), str):
                content = payload["content"]
        tool_calls_identity = self._stable_tool_calls_identity(tool_calls)
        return (
            str(msg.get("role") or "unknown"),
            content,
            str(msg.get("tool_call_id") or ""),
            tool_calls_identity,
        )

    @staticmethod
    def _matches_store_tail_suffix(
        stored_tail: list[tuple[str, str, str, str]],
        candidate_prefix: list[tuple[str, str, str, str]],
    ) -> bool:
        if not candidate_prefix:
            return True
        if len(candidate_prefix) > len(stored_tail):
            return False
        return stored_tail[-len(candidate_prefix) :] == candidate_prefix

    @classmethod
    def _identity_content_for_active_cleanup(cls, content: str) -> Any:
        """Decode canonical stored JSON content before active-cleanup checks.

        Structured assistant content is persisted as deterministic JSON. Active
        replay cleanup sees the original list/dict shape, so restart
        reconciliation has to decode the stored identity before deciding whether
        a durable assistant row could be absent from sanitized active context.
        """
        if not isinstance(content, str):
            return content
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return content
        if isinstance(decoded, (list, dict)) and normalize_content_value(decoded) == content:
            return decoded
        return content

    @classmethod
    def _is_active_context_droppable_identity(cls, identity: tuple[str, str, str, str]) -> bool:
        """Return true for durable rows sanitized out of active replay only."""
        role, content, _tool_call_id, tool_calls = identity
        if role != "assistant" or tool_calls:
            return False
        return cls._should_drop_active_assistant_message({
            "role": role,
            "content": cls._identity_content_for_active_cleanup(content),
        })

    @classmethod
    def _active_cleanup_replay_identity(
        cls,
        identity: tuple[str, str, str, str],
    ) -> tuple[str, str, str, str] | None:
        role, content, tool_call_id, tool_calls = identity
        if role != "assistant":
            return identity
        msg: dict[str, Any] = {
            "role": role,
            "content": cls._identity_content_for_active_cleanup(content),
        }
        if tool_calls:
            try:
                decoded_tool_calls = json.loads(tool_calls)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded_tool_calls = tool_calls
            msg["tool_calls"] = decoded_tool_calls
        cleaned = cls._clean_active_assistant_message(msg)
        if cleaned is None:
            return None
        return (
            role,
            normalize_content_value(cleaned.get("content")) or "",
            tool_call_id,
            tool_calls,
        )

    @staticmethod
    def _is_quarantined_assistant_replay_identity(identity: tuple[str, str, str, str]) -> bool:
        role, content, _tool_call_id, _tool_calls = identity
        if role != "assistant":
            return False
        text = str(content or "").strip()
        return bool(
            re.fullmatch(
                r"\[Externalized LCM ingest payload: assistant output quarantined; "
                r"kind=quarantined_assistant_output; "
                r"reason=[A-Za-z0-9_.:/-]+; "
                r"field=[A-Za-z0-9_.:/<>\[\]-]+; "
                r"chars=\d+; bytes=\d+; "
                r"ref=[^\]\s]+\]",
                text,
            )
            or re.fullmatch(
                r"\[LCM active replay placeholder: assistant output quarantined; "
                r"kind=quarantined_assistant_output; "
                r"reason=[A-Za-z0-9_.:/-]+; "
                r"scope=ignored_message_pattern; field=content; "
                r"chars=\d+; bytes=\d+; "
                r"sha256=[0-9a-f]{16}\]",
                text,
            )
        )

    def _ignored_message_is_quarantinable_assistant(self, msg: Dict[str, Any]) -> bool:
        if self._is_volatile_ignored_quarantine_placeholder(
            msg,
            text_content_for_pattern_matching(msg.get("content")) or "",
        ):
            return True
        identity = self._message_replay_identity(msg)
        if self._is_quarantined_assistant_replay_identity(identity):
            return True
        if not self._matches_ignore_message_patterns(msg):
            return False
        if identity[0] != "assistant":
            return False
        content = normalize_content_value(msg.get("content")) or ""
        return assistant_output_quarantine_reason(content) is not None

    def _stored_tail_for_sanitized_active_replay(
        self,
        stored_tail: list[tuple[str, str, str, str]],
    ) -> list[tuple[str, str, str, str]]:
        """Mirror active-context cleanup for restart replay reconciliation.

        Raw storage remains lossless. This view is used only to reconcile a
        restarted process when the host replays sanitized active context where
        assistant rows may be removed or have internal content stripped.
        """
        sanitized_tail: list[tuple[str, str, str, str]] = []
        for identity in stored_tail:
            cleaned_identity = self._active_cleanup_replay_identity(identity)
            if cleaned_identity is not None:
                sanitized_tail.append(cleaned_identity)
        return sanitized_tail

    def _find_reconciled_cursor_for_store_tail(
        self,
        messages: List[Dict[str, Any]],
        stored_tail: list[tuple[str, str, str, str]],
        *,
        allow_empty_prefix: bool,
        session_count: int,
        raw_session_count: int,
    ) -> int | None:
        sanitized_replay_tail = self._stored_tail_for_sanitized_active_replay(stored_tail)
        effective_session_count = len(sanitized_replay_tail)
        sanitized_tail_collapsed = len(sanitized_replay_tail) < len(stored_tail)
        empty_prefix_cursor: int | None = None
        for cursor in range(len(messages), -1, -1):
            candidate_messages = messages[:cursor]
            candidate_visible_messages = [
                msg
                for msg in candidate_messages
                if not self._is_replayed_context_scaffold_message(msg)
                and not self._matches_ignore_message_patterns(msg)
            ]
            candidate_non_placeholder_messages = [
                msg
                for msg in candidate_visible_messages
                if not self._is_volatile_ignored_quarantine_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                and not (
                    self._compiled_ignore_message_patterns
                    and self._is_quarantined_assistant_replay_identity(
                        self._message_replay_identity(msg)
                    )
                    and self._matches_ignore_message_patterns(msg, stored_row=True)
                )
            ]
            filtered_candidate_placeholders = len(candidate_non_placeholder_messages) < len(candidate_visible_messages)
            candidate_identity_messages = (
                candidate_non_placeholder_messages
                if candidate_non_placeholder_messages or filtered_candidate_placeholders
                else candidate_visible_messages
            )
            candidate_prefix = [
                self._message_replay_identity(msg)
                for msg in candidate_identity_messages
            ]
            if not candidate_prefix:
                empty_prefix_cursor = cursor
                if allow_empty_prefix:
                    return cursor
                continue

            matches_sanitized_tail = (
                len(candidate_prefix) <= len(sanitized_replay_tail)
                and self._matches_store_tail_suffix(sanitized_replay_tail, candidate_prefix)
            )
            matches_raw_tail = self._matches_store_tail_suffix(stored_tail, candidate_prefix)
            raw_tail_suffix = stored_tail[-len(candidate_prefix) :] if matches_raw_tail else []
            raw_suffix_needs_cleanup_equivalence = any(
                self._active_cleanup_replay_identity(identity) != identity
                for identity in raw_tail_suffix
            )
            if not matches_sanitized_tail and not matches_raw_tail:
                continue

            # Matching a stored suffix is not enough evidence by itself.  A
            # gateway restart may provide only newly arrived delta messages; if
            # the first delta happens to repeat the durable tail, treating that
            # row as replay silently loses it.  Only advance the cursor when the
            # incoming prefix proves replay by covering the full durable session.
            # A system prompt is a strong anchor. Older/minimal transcripts can
            # start directly with user/assistant turns, so multi-row full replay
            # is accepted only when active cleanup did not collapse the durable
            # tail; otherwise a fresh delta can repeat the remaining visible
            # suffix and must be preserved.
            candidate_has_system = any(identity[0] == "system" for identity in candidate_prefix)
            candidate_dropped_quarantine_replay_placeholder = any(
                self._is_volatile_ignored_quarantine_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                or (
                    self._compiled_ignore_message_patterns
                    and self._is_quarantined_assistant_replay_identity(
                        self._message_replay_identity(msg)
                    )
                    and self._matches_ignore_message_patterns(msg, stored_row=True)
                )
                for msg in candidate_messages
            )
            has_quarantined_singleton_replay = (
                matches_sanitized_tail
                and len(candidate_prefix) == 1
                and effective_session_count == 1
                and self._is_quarantined_assistant_replay_identity(candidate_prefix[0])
                and self._is_quarantined_assistant_replay_identity(sanitized_replay_tail[0])
            )
            has_externalized_singleton_replay = (
                matches_raw_tail
                and len(candidate_prefix) == 1
                and raw_session_count == 1
                and bool(extract_externalized_ref(normalize_content_value(candidate_identity_messages[0].get("content")) or ""))
                and candidate_prefix == stored_tail
            )
            has_filtered_full_replay = (
                matches_sanitized_tail
                and candidate_dropped_quarantine_replay_placeholder
                and len(candidate_prefix) >= effective_session_count
                and effective_session_count > 0
            )
            has_effective_full_replay = matches_sanitized_tail and len(candidate_prefix) >= effective_session_count and (
                candidate_has_system
                or (effective_session_count > 1 and not sanitized_tail_collapsed)
                or has_quarantined_singleton_replay
                or has_filtered_full_replay
            )
            has_scaffold_evidence = any(
                self._is_replayed_context_scaffold_message(msg) for msg in candidate_messages
            )
            has_raw_full_replay = (
                matches_raw_tail
                and not has_scaffold_evidence
                and len(candidate_messages) >= raw_session_count
                and raw_session_count > 1
            )
            has_preserved_objective_scaffold = any(
                str(msg.get("role") or "") != "system"
                and (normalize_content_value(msg.get("content")) or "").lstrip().startswith(
                    _PRESERVED_OBJECTIVE_CONTEXT_PREFIX
                )
                for msg in candidate_messages
            )
            candidate_suffix_has_user_turn = any(identity[0] == "user" for identity in candidate_prefix)
            has_scaffold_suffix_replay = (
                matches_sanitized_tail
                and has_preserved_objective_scaffold
                and not candidate_suffix_has_user_turn
            )
            has_raw_cleanup_replay = (
                matches_raw_tail
                and has_scaffold_evidence
                and cursor < len(messages)
                and len(candidate_prefix) >= max(1, self._config.fresh_tail_count)
                and raw_suffix_needs_cleanup_equivalence
            )
            if (
                has_effective_full_replay
                or has_externalized_singleton_replay
                or has_raw_full_replay
                or has_scaffold_suffix_replay
                or has_raw_cleanup_replay
            ):
                return cursor
        return empty_prefix_cursor if allow_empty_prefix else None

    def _record_ingest_reconciliation(
        self,
        *,
        action: str,
        reason: str,
        cursor: int,
        incoming: int,
        session_count: int,
        stored_tail_count: int,
        effective_incoming: int | None = None,
    ) -> None:
        self._last_ingest_reconciliation = {
            "action": action,
            "reason": reason,
            "cursor": cursor,
            "incoming": incoming,
            "session_count": session_count,
            "stored_tail_count": stored_tail_count,
        }
        if effective_incoming is not None:
            self._last_ingest_reconciliation["effective_incoming"] = effective_incoming

    def _effective_replay_identities(
        self,
        messages: List[Dict[str, Any]],
    ) -> list[tuple[str, str, str, str]]:
        return [
            self._message_replay_identity(msg)
            for msg in messages
            if not self._is_replayed_context_scaffold_message(msg)
            and not self._matches_ignore_message_patterns(msg)
        ]

    def _is_suspicious_stale_no_overlap_snapshot(
        self,
        incoming_identities: list[tuple[str, str, str, str]],
        stored_tail: list[tuple[str, str, str, str]],
        stored_head: list[tuple[str, str, str, str]],
    ) -> bool:
        """Return true for short stale snapshots with no durable-tail overlap.

        A restarted gateway can hand LCM a stale, short in-memory snapshot from
        the beginning of a longer session.  When that snapshot has no overlap
        with the durable tail, appending it as a delta creates duplicate rows.
        Fail closed only when the short batch is proven stale by matching the
        contiguous durable-store prefix; singleton no-overlap deltas remain
        ambiguous and are preserved.
        """
        if len(incoming_identities) <= 1:
            return False
        if incoming_identities[0][0] != "system":
            return False
        if not stored_tail or len(incoming_identities) >= len(stored_tail):
            return False
        if set(incoming_identities).intersection(stored_tail):
            return False
        if len(incoming_identities) > len(stored_head):
            return False
        return stored_head[: len(incoming_identities)] == incoming_identities

    def _reconcile_ingest_cursor_from_store(self, messages: List[Dict[str, Any]]) -> int:
        """Infer the in-memory cursor for an existing session after process restart."""
        if not self._session_id or not messages:
            return 0

        try:
            session_count = self._store.get_session_count(self._session_id)
        except Exception as exc:  # pragma: no cover - defensive only
            logger.debug("LCM ingest cursor reconciliation count failed: %s", exc)
            return 0
        if session_count <= 0:
            return 0

        tail_limit = min(max(len(messages) * 4, 64), session_count)
        stored_rows = self._store.get_session_tail(self._session_id, limit=tail_limit)
        if not stored_rows:
            return 0
        stored_tail = [
            self._message_replay_identity(row, stored_row=True)
            for row in stored_rows
            if not self._matches_ignore_message_patterns(row, stored_row=True)
        ]
        cursor = self._find_reconciled_cursor_for_store_tail(
            messages,
            stored_tail,
            allow_empty_prefix=True,
            session_count=len(stored_tail),
            raw_session_count=session_count,
        )
        if cursor is not None and cursor > 0:
            reason = (
                "skipped scaffold-only prefix"
                if not self._effective_replay_identities(messages[:cursor])
                else "replayed durable tail"
            )
            self._record_ingest_reconciliation(
                action="advanced cursor",
                reason=reason,
                cursor=cursor,
                incoming=len(messages),
                session_count=session_count,
                stored_tail_count=len(stored_tail),
                effective_incoming=len(self._effective_replay_identities(messages)),
            )
            logger.debug(
                "LCM reconciled ingest cursor after existing-session bind: session=%s cursor=%d incoming=%d stored_tail=%d session_count=%d reason=%s",
                self._session_id,
                cursor,
                len(messages),
                len(stored_tail),
                session_count,
                reason,
            )
            return cursor

        incoming_identities = self._effective_replay_identities(messages)
        stored_head_rows = self._store.get_session_messages(
            self._session_id,
            limit=tail_limit,
        )
        stored_head = [self._message_replay_identity(row, stored_row=True) for row in stored_head_rows]
        # Stale-snapshot proof uses the raw durable prefix.  Ignore-message
        # filters may suppress noisy rows for tail reconciliation, but filtered
        # history alone must not create replay evidence for skipping a batch.
        if self._is_suspicious_stale_no_overlap_snapshot(
            incoming_identities,
            stored_tail,
            stored_head,
        ):
            self._record_ingest_reconciliation(
                action="skipped batch",
                reason="skipped stale no-overlap snapshot",
                cursor=len(messages),
                incoming=len(messages),
                session_count=session_count,
                stored_tail_count=len(stored_tail),
                effective_incoming=len(incoming_identities),
            )
            logger.warning(
                "LCM skipped stale no-overlap snapshot after existing-session bind: session=%s incoming=%d effective_incoming=%d stored_tail=%d session_count=%d",
                self._session_id,
                len(messages),
                len(incoming_identities),
                len(stored_tail),
                session_count,
            )
            return len(messages)

        self._record_ingest_reconciliation(
            action="persisted batch",
            reason="persisted ambiguous delta",
            cursor=0,
            incoming=len(messages),
            session_count=session_count,
            stored_tail_count=len(stored_tail),
            effective_incoming=len(incoming_identities),
        )
        return 0

    def _redact_active_replay_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        redacted_replay_messages: list[Dict[str, Any]] = []
        for message in messages:
            redacted_message = dict(message)
            if "content" in redacted_message:
                redacted_message["content"] = redact_sensitive_value(
                    redacted_message.get("content"),
                    self._config,
                    parse_json_strings=False,
                )
            if "tool_calls" in redacted_message:
                redacted_message["tool_calls"] = redact_sensitive_value(
                    redacted_message.get("tool_calls"),
                    self._config,
                    parse_json_strings=True,
                )
            redacted_replay_messages.append(redacted_message)
        return redacted_replay_messages

    def _ingest_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Persist new messages to the store.

        Uses a cursor to track which portion of the current messages list
        has already been persisted.  After compress() shortens the list,
        the cursor is reset to len(compressed), so only messages appended
        after compaction are ingested — regardless of how the store count
        compares to the current list length.

        Returns a replay-safe copy of ``messages`` with obviously broken
        assistant loops replaced by quarantine placeholders. Existing callers may
        ignore the return value when they only need durable persistence.
        """
        if not self._session_id:
            logger.debug("Ingest skipped: no session_id")
            return self._redact_active_replay_messages(messages)

        if self._session_ignored or self._session_stateless:
            logger.debug(
                "Ingest skipped for %s session %s",
                "ignored" if self._session_ignored else "stateless",
                self._session_id,
            )
            return self._redact_active_replay_messages(messages)

        n = len(messages)
        cursor = min(max(self._ingest_cursor, 0), n)
        scan_start = 0 if self._ingest_cursor_needs_reconcile else cursor
        ignored_original_messages = [False] * n
        if self._compiled_ignore_message_patterns:
            for idx in range(scan_start, n):
                ignored_original_messages[idx] = self._matches_ignore_message_patterns(messages[idx])
        externalize_messages = [False] * n
        prefer_existing_externalized = [False] * n
        for idx in range(scan_start, n):
            externalize_messages[idx] = not ignored_original_messages[idx]
        for idx in range(0, scan_start):
            prefer_existing_externalized[idx] = not ignored_original_messages[idx]
        replay_messages = quarantine_suspicious_assistant_messages(
            messages,
            session_id=self._session_id,
            config=self._config,
            hermes_home=self._hermes_home,
            externalize=externalize_messages,
            prefer_existing_externalized=prefer_existing_externalized,
        )
        replay_messages = self._redact_active_replay_messages(replay_messages)
        if self._ingest_cursor_needs_reconcile:
            reconcile_messages = replay_messages
            if self._compiled_ignore_message_patterns:
                reconcile_messages = [
                    original_msg if ignored_original_messages[idx] else replay_msg
                    for idx, (original_msg, replay_msg) in enumerate(zip(messages, replay_messages))
                ]
            self._ingest_cursor = self._reconcile_ingest_cursor_from_store(reconcile_messages)
            self._ingest_cursor_needs_reconcile = False
        cursor = min(max(self._ingest_cursor, 0), n)
        logger.debug(
            "Ingest: session=%s cursor=%d incoming=%d",
            self._session_id, cursor, n,
        )

        new_messages = replay_messages[cursor:] if cursor < n else []
        original_new_messages = messages[cursor:] if cursor < n else []

        if not new_messages:
            return replay_messages

        messages_to_store_with_index: list[tuple[int, Dict[str, Any]]] = [
            (cursor + offset, replay_msg)
            for offset, replay_msg in enumerate(new_messages)
        ]
        if self._compiled_ignore_message_patterns:
            kept: list[tuple[int, Dict[str, Any]]] = []
            for offset, (original_msg, replay_msg) in enumerate(zip(original_new_messages, new_messages)):
                absolute_idx = cursor + offset
                if ignored_original_messages[absolute_idx] or self._is_volatile_ignored_quarantine_placeholder(
                    replay_msg,
                    text_content_for_pattern_matching(replay_msg.get("content")) or "",
                ):
                    self._ignored_message_count += 1
                    text = text_content_for_pattern_matching(original_msg.get("content")) or ""
                    excerpt = text[:80].replace("\n", " ")
                    logger.debug(
                        "LCM ignore_message_patterns dropped %s message: %r",
                        original_msg.get("role", "unknown"),
                        excerpt,
                    )
                    continue
                kept.append((absolute_idx, replay_msg))
            messages_to_store_with_index = kept

        if not messages_to_store_with_index:
            self._ingest_cursor = n
            return replay_messages

        protected_messages = protect_messages_for_ingest(
            [msg for _idx, msg in messages_to_store_with_index],
            session_id=self._session_id,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        active_replay_messages = replay_messages
        for (absolute_idx, _replay_msg), protected_msg in zip(
            messages_to_store_with_index,
            protected_messages,
        ):
            if self._protected_message_uses_raw_payload_active_stub(protected_msg):
                if active_replay_messages is replay_messages:
                    active_replay_messages = [dict(message) for message in replay_messages]
                active_message = dict(active_replay_messages[absolute_idx])
                active_message["content"] = protected_msg["content"]
                active_replay_messages[absolute_idx] = active_message

        estimates = [count_message_tokens(m) for m in protected_messages]
        self._store.append_batch(
            self._session_id,
            protected_messages,
            estimates,
            source=self._session_platform,
        )
        self._ingest_cursor = n
        logger.debug("Ingested %d messages into LCM store", len(messages_to_store_with_index))
        # Most ``protected_messages`` changes are storage-only: inline media,
        # tool results, and data/base64 substrings must stay provider-usable in
        # active replay. Whole-message ``raw_payload`` externalization is the
        # exception: it intentionally returns a compact active stub so the host
        # does not replay huge opaque text while SQLite stores only the stub.
        return active_replay_messages

    @staticmethod
    def _protected_message_uses_raw_payload_active_stub(message: Dict[str, Any]) -> bool:
        content = message.get("content")
        return isinstance(content, str) and content.startswith(
            "[Externalized payload: kind=raw_payload;"
        )

    def _get_store_ids_for_messages(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Map current raw messages back to store_ids in stable store order.

        Matching starts strictly after ``_last_compacted_store_id`` so repeated
        content from older already-compacted history cannot hijack the mapping.
        Synthetic summary messages simply fail to match and are skipped.
        """
        candidates = [
            stored for stored in self._store.get_session_messages(self._session_id)
            if stored["store_id"] > self._last_compacted_store_id
        ]

        ids: list[int] = []
        store_idx = 0
        for msg in messages:
            message_identity = self._message_replay_identity(msg)
            wanted_cleanup_identity = self._active_cleanup_replay_identity(message_identity)
            probe_idx = store_idx
            while probe_idx < len(candidates):
                stored = candidates[probe_idx]
                stored_identity = self._message_replay_identity(stored, stored_row=True)
                if stored_identity == message_identity:
                    ids.append(stored["store_id"])
                    store_idx = probe_idx + 1
                    break
                if (
                    wanted_cleanup_identity is not None
                    and self._active_cleanup_replay_identity(stored_identity) == wanted_cleanup_identity
                ):
                    ids.append(stored["store_id"])
                    store_idx = probe_idx + 1
                    break
                probe_idx += 1

        return ids

    # -- Internal: summarization -------------------------------------------

    def _run_pre_compaction_extraction(self, messages: List[Dict[str, Any]]) -> None:
        """Best-effort extraction of decisions before compaction."""
        try:
            serialized = self._serialize_messages(messages)
            output_path = self._config.extraction_output_path
            if not output_path:
                base = self._hermes_home or os.path.expanduser("~/.hermes")
                output_path = os.path.join(base, "lcm-extractions")
            extraction_model = self._config.extraction_model or self._config.summary_model
            extract_before_compaction(
                serialized_messages=serialized,
                output_path=output_path,
                session_id=self._session_id or "",
                model=extraction_model,
                timeout=self._config.summary_timeout_ms / 1000,
            )
        except Exception as e:
            logger.warning("Pre-compaction extraction failed (non-blocking): %s", e)

    def _maybe_gc_compacted_tool_results(
        self,
        compacted_chunk: List[Dict[str, Any]],
        source_store_ids: List[int],
    ) -> None:
        if not getattr(self._config, "large_output_transcript_gc_enabled", False):
            return
        if not compacted_chunk or not source_store_ids:
            return

        stored_by_id = self._store.get_batch(source_store_ids)
        for store_id in source_store_ids:
            stored = stored_by_id.get(store_id)
            if not stored or stored.get("session_id") != self._session_id:
                continue
            if stored.get("role") != "tool":
                continue
            content = stored.get("content", "") or ""
            tool_call_id = stored.get("tool_call_id", "") or ""
            if not content:
                continue

            ref = extract_externalized_ref(content)
            if ref:
                externalized = load_externalized_payload(
                    ref,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                if externalized is not None and externalized.get("kind", "tool_result") == "tool_result":
                    placeholder = build_transcript_gc_placeholder(externalized)
                    self._store.gc_externalized_tool_result(store_id, placeholder)
                    continue

            lookup_candidates = []
            sanitized_content = sanitize_pre_compaction_content(content)
            if sanitized_content and sanitized_content != content:
                lookup_candidates.append(sanitized_content)
            lookup_candidates.append(content)

            externalized = None
            for candidate in lookup_candidates:
                externalized = find_externalized_payload_for_message(
                    candidate,
                    tool_call_id=tool_call_id,
                    session_id=self._session_id,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                if externalized is not None:
                    break
            if externalized is None:
                continue

            placeholder = build_transcript_gc_placeholder(externalized)
            self._store.gc_externalized_tool_result(store_id, placeholder)

    def _serialize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Serialize messages into labeled text for the summarizer."""
        parts = []
        matched_tool_ids = _matched_tool_call_ids(messages)
        for msg in messages:
            role = msg.get("role", "unknown")
            content = redact_sensitive_value(
                msg.get("content") or "",
                self._config,
                parse_json_strings=False,
            )
            content = sanitize_pre_compaction_content(content)

            if role == "tool":
                tool_id = str(msg.get("tool_call_id") or "").strip()
                externalized = maybe_externalize_tool_output(
                    content,
                    tool_call_id=tool_id,
                    session_id=self._session_id,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                if externalized:
                    content = externalized["placeholder"]
                elif len(content) > 3000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            if role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                matched_tool_calls = [
                    tc for tc in tool_calls
                    if not _tool_call_id(tc) or _tool_call_id(tc) in matched_tool_ids
                ]
                if _is_synthetic_assistant_noise(content):
                    if not matched_tool_calls:
                        continue
                    content = ""
                if len(content) > 3000:
                    content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
                if matched_tool_calls:
                    tc_parts = []
                    for tc in matched_tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function", {})
                            name = fn.get("name", "?")
                            args = fn.get("arguments", "")
                            args = redact_sensitive_value(
                                args,
                                self._config,
                                parse_json_strings=True,
                            )
                            args = sanitize_pre_compaction_tool_arguments(args)
                            if len(args) > 500:
                                args = args[:400] + "..."
                            tc_parts.append(f"  {name}({args})")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            if len(content) > 3000:
                content = content[:2000] + "\n...[truncated]...\n" + content[-800:]
            parts.append(f"[{role.upper()}]: {content}")

        return "\n\n".join(parts)

    # -- Internal: tool-pair sanitization ------------------------------------

    @staticmethod
    def _structured_part_text(part: Dict[str, Any]) -> str:
        for key in ("text", "content", "value"):
            value = part.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                nested = value.get("value")
                if isinstance(nested, str):
                    return nested
                nested = value.get("content")
                if isinstance(nested, str):
                    return nested
        return ""

    @classmethod
    def _structured_part_has_visible_assistant_content(cls, part: Any) -> bool:
        if part is None:
            return False
        if isinstance(part, str):
            return bool(_strip_reasoning_blocks(part).strip())
        if not isinstance(part, dict):
            return bool(str(part).strip())

        part_type = str(part.get("type") or "").strip().lower()
        if part_type in _INTERNAL_ASSISTANT_PART_TYPES:
            return False
        if part_type in _VISIBLE_TEXT_PART_TYPES:
            return bool(_strip_reasoning_blocks(cls._structured_part_text(part)).strip())

        # Unknown non-internal content blocks may be visible (for example
        # images/audio/annotations in provider-specific formats).  Preserve
        # them rather than risk dropping a legitimate assistant turn.
        return True

    @classmethod
    def _assistant_message_has_visible_content(cls, msg: Dict[str, Any]) -> bool:
        content = msg.get("content")
        if content is None:
            return False
        if isinstance(content, str):
            return bool(_strip_reasoning_blocks(content).strip())
        if isinstance(content, list):
            return any(cls._structured_part_has_visible_assistant_content(part) for part in content)
        if isinstance(content, dict):
            return cls._structured_part_has_visible_assistant_content(content)
        return bool(str(content).strip())

    @classmethod
    def _strip_structured_text_part(cls, part: Dict[str, Any]) -> Dict[str, Any] | None:
        cleaned = dict(part)
        for key in ("text", "content", "value"):
            value = cleaned.get(key)
            if isinstance(value, str):
                stripped = _strip_reasoning_blocks(value)
                if not stripped.strip():
                    return None
                cleaned[key] = stripped
                return cleaned
            if isinstance(value, dict):
                nested = dict(value)
                for nested_key in ("value", "content", "text"):
                    nested_value = nested.get(nested_key)
                    if isinstance(nested_value, str):
                        stripped = _strip_reasoning_blocks(nested_value)
                        if not stripped.strip():
                            return None
                        nested[nested_key] = stripped
                        cleaned[key] = nested
                        return cleaned
        return cleaned if cls._structured_part_has_visible_assistant_content(cleaned) else None

    @classmethod
    def _sanitize_active_assistant_content(cls, content: Any) -> Any | None:
        if content is None:
            return None
        if isinstance(content, str):
            stripped = _strip_reasoning_blocks(content)
            return stripped if stripped.strip() else None
        if isinstance(content, list):
            cleaned_parts: list[Any] = []
            for part in content:
                if isinstance(part, str):
                    stripped = _strip_reasoning_blocks(part)
                    if stripped.strip():
                        cleaned_parts.append(stripped)
                    continue
                if isinstance(part, dict):
                    part_type = str(part.get("type") or "").strip().lower()
                    if part_type in _INTERNAL_ASSISTANT_PART_TYPES:
                        continue
                    if part_type in _VISIBLE_TEXT_PART_TYPES:
                        cleaned_part = cls._strip_structured_text_part(part)
                        if cleaned_part is not None:
                            cleaned_parts.append(cleaned_part)
                        continue
                if cls._structured_part_has_visible_assistant_content(part):
                    cleaned_parts.append(part)
            return cleaned_parts or None
        if isinstance(content, dict):
            part_type = str(content.get("type") or "").strip().lower()
            if part_type in _INTERNAL_ASSISTANT_PART_TYPES:
                return None
            if part_type in _VISIBLE_TEXT_PART_TYPES:
                return cls._strip_structured_text_part(content)
            return content if cls._structured_part_has_visible_assistant_content(content) else None
        return content if str(content).strip() else None

    @classmethod
    def _clean_active_assistant_message(cls, msg: Dict[str, Any]) -> Dict[str, Any] | None:
        if msg.get("role") != "assistant":
            return msg
        if "content" not in msg:
            return msg
        cleaned_content = cls._sanitize_active_assistant_content(msg.get("content"))
        if cleaned_content is None:
            if not msg.get("tool_calls"):
                return None
            cleaned_content = ""
        if cleaned_content == msg.get("content"):
            return msg
        cleaned = dict(msg)
        cleaned["content"] = cleaned_content
        return cleaned

    @classmethod
    def _should_drop_active_assistant_message(cls, msg: Dict[str, Any]) -> bool:
        if msg.get("role") != "assistant":
            return False
        if msg.get("tool_calls"):
            return False
        return cls._clean_active_assistant_message(msg) is None

    def _sanitize_active_context_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        insert_missing_tool_stubs: bool = True,
    ) -> List[Dict[str, Any]]:
        """Drop unsafe assistant-only noise, then repair tool sequencing.

        This is intentionally active-context-only: callers pass the selected
        provider replay context, and this helper never mutates stored rows,
        source mappings, or DAG nodes.
        """
        cleaned: list[Dict[str, Any]] = []
        dropped_assistant_messages = 0
        stripped_assistant_messages = 0
        for msg in messages:
            if msg.get("role") == "assistant":
                cleaned_msg = self._clean_active_assistant_message(msg)
                if cleaned_msg is None:
                    dropped_assistant_messages += 1
                    continue
                if cleaned_msg is not msg:
                    stripped_assistant_messages += 1
                cleaned.append(cleaned_msg)
                continue
            cleaned.append(msg)

        if dropped_assistant_messages:
            logger.info(
                "LCM active-context cleanup: dropped %d assistant message(s) with no visible content",
                dropped_assistant_messages,
            )
        if stripped_assistant_messages:
            logger.info(
                "LCM active-context cleanup: stripped internal content from %d assistant message(s)",
                stripped_assistant_messages,
            )

        return self._sanitize_tool_pairs(
            cleaned,
            insert_missing_tool_stubs=insert_missing_tool_stubs,
        )

    def _sanitize_tool_pairs(
        self,
        messages: List[Dict[str, Any]],
        *,
        insert_missing_tool_stubs: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return provider-safe active-context tool-call/result sequencing.

        Raw store and DAG history remain lossless. This guardrail only sanitizes
        the active context emitted back to providers, where assistant tool calls
        must be followed immediately by their contiguous tool results. Late,
        duplicate, out-of-order, and orphan tool results are dropped; missing
        direct results get synthetic stubs.
        """
        sanitized: List[Dict[str, Any]] = []
        dropped_tool_results = 0
        inserted_stub_results = 0

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.get("role") == "tool":
                dropped_tool_results += 1
                i += 1
                continue

            sanitized.append(msg)

            if msg.get("role") == "assistant":
                expected_ids = [
                    call_id
                    for call_id in (_tool_call_id(tool_call) for tool_call in (msg.get("tool_calls") or []))
                    if call_id
                ]

                for expected_id in expected_ids:
                    matched_direct_result = False
                    while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                        next_msg = messages[i + 1]
                        next_id = str(next_msg.get("tool_call_id") or "").strip()
                        if next_id == expected_id:
                            sanitized.append(next_msg)
                            i += 1
                            matched_direct_result = True
                            break
                        dropped_tool_results += 1
                        i += 1

                    if not matched_direct_result and insert_missing_tool_stubs:
                        sanitized.append({
                            "role": "tool",
                            "content": "[Result from earlier conversation — see context summary above]",
                            "tool_call_id": expected_id,
                        })
                        inserted_stub_results += 1

                while i + 1 < len(messages) and messages[i + 1].get("role") == "tool":
                    dropped_tool_results += 1
                    i += 1

            i += 1

        if dropped_tool_results:
            logger.info(
                "LCM tool-pair guardrail: dropped %d late/orphan/duplicate tool result(s)",
                dropped_tool_results,
            )
        if inserted_stub_results:
            logger.info(
                "LCM tool-pair guardrail: inserted %d missing tool-result stub(s)",
                inserted_stub_results,
            )

        return sanitized

    # -- Internal: condensation --------------------------------------------

    def _should_allow_follow_on_condensation(
        self,
        *,
        uncondensed_count: int,
        leaf_compacted_this_turn: bool,
        force_overflow: bool,
        critical_budget_pressure: bool = False,
    ) -> tuple[bool, str]:
        if not leaf_compacted_this_turn:
            return True, ""
        if not self._config.cache_friendly_condensation_enabled:
            return True, ""
        if force_overflow:
            return True, ""
        if critical_budget_pressure:
            return True, ""

        fanin = max(1, self._config.condensation_fanin)
        debt_threshold = fanin * max(1, self._config.cache_friendly_min_debt_groups)
        if uncondensed_count >= debt_threshold:
            return True, ""
        if uncondensed_count == fanin:
            return False, "cache_friendly_single_group"
        return False, "cache_friendly_low_debt"

    def _maybe_condense(
        self,
        focus_topic: Optional[str] = None,
        *,
        leaf_compacted_this_turn: bool = False,
        force_overflow: bool = False,
        critical_budget_pressure: bool = False,
    ) -> None:
        """Check if any depth level has enough nodes for condensation."""
        self._last_condensation_suppressed_reason = ""

        max_depth = self._config.incremental_max_depth
        if max_depth == 0:
            return  # condensation disabled

        # When max_depth is -1 (unlimited), derive the upper bound from
        # the deepest existing node + 1, so condensation can always
        # create the next depth level.
        if max_depth < 0:
            all_nodes = self._dag.get_session_nodes(self._session_id)
            upper = (max(n.depth for n in all_nodes) + 1) if all_nodes else 1
        else:
            upper = max_depth

        condensed_any = False
        suppression_reason = ""

        for depth in range(upper):
            uncondensed = self._dag.get_uncondensed_at_depth(
                self._session_id, depth
            )
            if len(uncondensed) < self._config.condensation_fanin:
                continue

            allow_condense, reason = self._should_allow_follow_on_condensation(
                uncondensed_count=len(uncondensed),
                leaf_compacted_this_turn=leaf_compacted_this_turn,
                force_overflow=force_overflow,
                critical_budget_pressure=critical_budget_pressure,
            )
            if not allow_condense:
                suppression_reason = reason or suppression_reason
                continue

            # Take the first fanin nodes and condense
            to_condense = uncondensed[:self._config.condensation_fanin]
            combined_text = "\n\n---\n\n".join(n.summary for n in to_condense)
            source_tokens = sum(n.token_count for n in to_condense)
            token_budget = max(1000, int(source_tokens * 0.40))

            summary_text, level = summarize_with_escalation(
                text=combined_text,
                source_tokens=source_tokens,
                token_budget=token_budget,
                depth=depth + 1,
                model=self._config.summary_model,
                fallback_models=self._config.summary_fallback_models,
                circuit_breaker=self._summary_circuit_breaker,
                timeout=self._config.summary_timeout_ms / 1000,
                l2_budget_ratio=self._config.l2_budget_ratio,
                l3_truncate_tokens=self._config.l3_truncate_tokens,
                focus_topic=focus_topic or "",
                custom_instructions=self._config.custom_instructions,
            )

            earliest_at, latest_at = self._dag.get_source_time_window([n.node_id for n in to_condense])
            node = SummaryNode(
                session_id=self._session_id,
                depth=depth + 1,
                summary=summary_text,
                token_count=count_tokens(summary_text),
                source_token_count=source_tokens,
                source_ids=[n.node_id for n in to_condense],
                source_type="nodes",
                created_at=time.time(),
                earliest_at=earliest_at,
                latest_at=latest_at,
                expand_hint=self._extract_expand_hint(summary_text),
            )
            self._dag.add_node(node)
            condensed_any = True

            logger.info(
                "LCM condensation: d%d × %d → d%d (L%d, %d→%d tokens)",
                depth, len(to_condense), depth + 1, level,
                source_tokens, count_tokens(summary_text),
            )

            if leaf_compacted_this_turn and self._config.cache_friendly_condensation_enabled:
                break

        if not condensed_any and leaf_compacted_this_turn and self._config.cache_friendly_condensation_enabled:
            self._last_condensation_suppressed_reason = suppression_reason

    # -- Internal: context assembly ----------------------------------------

    @staticmethod
    def _append_lcm_note_to_content(content: Any) -> Any:
        note = (
            "\n\n[Note: This conversation uses Lossless Context Management (LCM). "
            "Earlier turns have been compacted into hierarchical summaries below. "
            "Use lcm_grep to search history, lcm_describe to inspect the DAG, "
            "and lcm_expand to recover original details from any summary.]"
        )
        if isinstance(content, str):
            return content + note
        note_part = {"type": "text", "text": note.lstrip()}
        if content is None:
            return note.lstrip()
        if isinstance(content, list):
            return list(content) + [note_part]
        normalized = normalize_content_value(content) or ""
        return normalized + note

    @staticmethod
    def _is_preserved_todo_context_message(message: Dict[str, Any]) -> bool:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        return content.lstrip().startswith(_PRESERVED_TODO_CONTEXT_PREFIX)

    @staticmethod
    def _preserved_objective_context_content(message: Dict[str, Any]) -> str:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        return content if content.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX) else ""

    def _build_preserved_objective_summary_part(self, message: Dict[str, Any]) -> str:
        content = text_content_for_pattern_matching(message.get("content")) or ""
        content = protect_inline_payloads_in_text(
            content,
            role=str(message.get("role") or "user"),
            session_id=self._session_id,
            field_path="preserved_objective.content",
            config=self._config,
            hermes_home=self._hermes_home,
        )
        return f"{_PRESERVED_OBJECTIVE_CONTEXT_PREFIX}\n{content}"

    def _latest_user_context_anchor(
        self,
        messages: List[Dict[str, Any]],
        selected_tail: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Return a scaffolded newest real user objective omitted from the tail.

        Tool-heavy turns can push the operative user request outside the fresh
        tail while retaining only assistant/tool traces from that turn.  The
        returned text is active-context scaffolding, not raw conversation: it is
        emitted inside the summary block so restart reconciliation ignores it
        instead of ingesting a duplicate non-contiguous user message.

        If a previous compaction already emitted the preserved-objective
        scaffold and no newer real user turn exists, carry that scaffold forward
        as the next anchor source so repeated compaction does not summarize the
        active objective away one compression later.
        """
        selected_tail_messages = [msg for msg in selected_tail if isinstance(msg, dict)]
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            preserved_objective = self._preserved_objective_context_content(message)
            if preserved_objective:
                if any(
                    self._preserved_objective_context_content(selected) == preserved_objective
                    for selected in selected_tail_messages
                ):
                    return None
                return preserved_objective
            if message.get("role") != "user":
                continue
            if self._is_preserved_todo_context_message(message):
                continue
            if any(message == selected for selected in selected_tail_messages):
                return None
            return self._build_preserved_objective_summary_part(message)
        return None

    def _assemble_context(
        self,
        system_msg: Optional[Dict[str, Any]],
        tail_messages: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
        include_lcm_note: bool = True,
    ) -> List[Dict[str, Any]]:
        """Build the active context from DAG summaries + fresh tail.

        Structure:
          [leading anchor, normally system prompt]
          [highest-depth summary nodes first, then lower]
          [fresh tail messages]
        """
        result = []

        # Leading anchor with optional LCM annotation. Only a true system prompt
        # is a safe permanent anchor; gateway sessions can start directly with
        # user messages, and those user turns must remain compactable.
        leading_msg = system_msg.copy() if system_msg is not None else None
        if leading_msg is not None:
            if (
                leading_msg.get("role") == "system"
                and self.compression_count == 0
                and include_lcm_note
            ):
                leading_msg["content"] = self._append_lcm_note_to_content(
                    leading_msg.get("content", "")
                )
            result.append(leading_msg)

        assembly_cap = (
            assembly_cap_override
            if assembly_cap_override is not None
            else self._effective_assembly_token_cap()
        )

        tail_selected = tail_messages
        anchor_source = getattr(self, "_pending_context_anchor_messages", None)
        if anchor_source is None:
            anchor_source = tail_messages
        anchor_part: Optional[str] = None
        summary_budget = None
        if assembly_cap is not None:
            used = count_message_tokens(leading_msg) if leading_msg is not None else 0
            kept_tail_reversed: list[Dict[str, Any]] = []
            tail_token_total = 0
            tail_for_selection = self._sanitize_active_context_messages(
                tail_messages,
                insert_missing_tool_stubs=False,
            )
            skipped_tail_gap = False
            for msg in reversed(tail_for_selection):
                msg_tokens = count_message_tokens(msg)
                if used + tail_token_total + msg_tokens > assembly_cap:
                    if self._is_budget_droppable_tail_message(msg):
                        skipped_tail_gap = True
                        continue
                    break
                if skipped_tail_gap:
                    break
                kept_tail_reversed.append(msg)
                tail_token_total += msg_tokens
            tail_selected = list(reversed(kept_tail_reversed))
            summary_budget = max(0, assembly_cap - used - tail_token_total)
        if anchor_source is not None:
            anchor_part = self._latest_user_context_anchor(anchor_source, tail_selected)

        # Collect DAG summaries — highest depth first for context hierarchy
        summary_parts: list[str] = []
        last_role = result[-1].get("role", "system") if result else "system"
        summary_role = "assistant" if last_role != "assistant" else "user"
        if anchor_part is not None:
            anchor_msg = {"role": summary_role, "content": anchor_part}
            if summary_budget is None or count_message_tokens(anchor_msg) <= summary_budget:
                summary_parts.append(anchor_part)

        all_nodes = self._dag.get_session_nodes(self._session_id)
        if all_nodes:
            # Group by depth, take the most recent uncondensed at each level
            # For active context, we want the highest-level summaries
            # that haven't been condensed into even higher levels
            depths = sorted(set(n.depth for n in all_nodes), reverse=True)
            for d in depths:
                uncondensed = self._dag.get_uncondensed_at_depth(self._session_id, d)
                for node in uncondensed:
                    depth_label = {
                        0: "Recent",
                        1: "Session Arc",
                        2: "Durable",
                    }.get(d, f"Depth-{d}")
                    summary_parts.append(
                        f"[{depth_label} Summary (d{d}, node {node.node_id})]\n"
                        f"{node.summary}\n"
                        f"[Expand for details: {node.expand_hint}]"
                    )

        if summary_parts:
            selected_parts = summary_parts
            if summary_budget is not None:
                selected_parts = []
                for part in summary_parts:
                    candidate = "\n\n---\n\n".join(selected_parts + [part])
                    candidate_msg = {"role": summary_role, "content": candidate}
                    if count_message_tokens(candidate_msg) > summary_budget:
                        if part == anchor_part:
                            continue
                        continue
                    selected_parts.append(part)
            if selected_parts:
                combined = "\n\n---\n\n".join(selected_parts)
                # Structural marker so downstream compaction-stats classification
                # detects this summary row exactly, without depending on the marker
                # text surviving content flattening. ``_``-prefixed → stripped by the
                # transport sanitizer / not copied by the allowlist-rebuild adapters,
                # so it never reaches a provider request or perturbs the prompt cache.
                result.append({"role": summary_role, "content": combined, "_lcm_summary": True})

        # Fresh tail
        result.extend(tail_selected)

        # ── Active-context cleanup / tool-pair guardrail ──
        # Drop assistant turns that carry only blank/internal structured content,
        # then ensure provider-valid tool-call/result sequencing.
        result = self._sanitize_active_context_messages(result)
        if (
            assembly_cap is not None
            and anchor_part is not None
            and count_messages_tokens(result) > assembly_cap
        ):
            trimmed_result: list[Dict[str, Any]] = []
            for msg in result:
                content = normalize_content_value(msg.get("content")) or ""
                if _PRESERVED_OBJECTIVE_CONTEXT_PREFIX not in content:
                    trimmed_result.append(msg)
                    continue
                parts = [
                    part for part in content.split("\n\n---\n\n")
                    if not part.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX)
                ]
                if parts:
                    trimmed = msg.copy()
                    trimmed["content"] = "\n\n---\n\n".join(parts)
                    trimmed_result.append(trimmed)
            result = self._sanitize_active_context_messages(trimmed_result)

        return result

    def _is_budget_droppable_tail_message(self, message: Dict[str, Any]) -> bool:
        """Return whether an over-budget tail message may be evicted.

        User turns are prompt-bearing context and stop tail selection when they
        cannot fit. Assistant/tool turns are derived context; if one bulky turn
        blocks older prompt material, skip it and keep scanning for budgetable
        user intent or compact status that still fits.
        """
        role = message.get("role")
        if role not in {"assistant", "tool"}:
            return False
        content = normalize_content_value(message.get("content")) or ""
        if _PRESERVED_TODO_CONTEXT_PREFIX in content:
            return False
        if _PRESERVED_OBJECTIVE_CONTEXT_PREFIX in content:
            return False
        return True

    def _finalize_forced_overflow_result(
        self,
        original_messages: List[Dict[str, Any]],
        compressed: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if compressed != original_messages:
            self._last_compression_status = "overflow_recovery"
            self._last_compression_noop_reason = ""
            self._ingest_cursor = len(compressed)
            self._ingest_cursor_needs_reconcile = False
            logger.info(
                "LCM assembly guardrail recovery: %d messages → %d (no new summary node)",
                len(original_messages),
                len(compressed),
            )
        else:
            self._last_compression_status = "noop"
            self._last_compression_noop_reason = (
                "forced overflow recovery found no droppable active-context messages"
            )

        effective_cap = (
            assembly_cap_override
            if assembly_cap_override is not None
            else self._effective_assembly_token_cap()
        )
        if effective_cap is None:
            self._last_overflow_recovery_failed = False
        else:
            self._last_overflow_recovery_failed = count_messages_tokens(compressed) > effective_cap
            if self._last_overflow_recovery_failed:
                logger.warning(
                    "LCM overflow recovery could not get under cap=%d; returning best-effort context (%d tokens)",
                    effective_cap,
                    count_messages_tokens(compressed),
                )
        return compressed

    def _should_force_overflow_recovery(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is None:
            return False

        tokens = self._overflow_recovery_signal_tokens(
            observed_tokens=observed_tokens,
            messages=messages,
        )
        if tokens is None:
            return False
        return tokens >= assembly_cap

    def _overflow_recovery_signal_tokens(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        candidates: list[int] = []
        if observed_tokens is not None and observed_tokens > 0:
            candidates.append(observed_tokens)
        if messages is not None:
            candidates.append(count_messages_tokens(messages))
        if not candidates:
            return None
        return max(candidates)

    def _overflow_recovery_assembly_cap(
        self,
        observed_tokens: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        assembly_cap = self._effective_assembly_token_cap()
        if assembly_cap is None:
            return None
        if messages is None or observed_tokens is None or observed_tokens <= 0:
            return assembly_cap

        message_tokens = count_messages_tokens(messages)
        overhead_tokens = max(0, observed_tokens - message_tokens)
        return max(1, assembly_cap - overhead_tokens)

    def _effective_assembly_token_cap(self) -> Optional[int]:
        """Return the active assembly cap, if any.

        Two knobs can constrain the assembled active context:
        - max_assembly_tokens: explicit hard cap
        - reserve_tokens_floor: keep headroom inside context_length
        """
        caps: list[int] = []

        if self._config.max_assembly_tokens > 0:
            caps.append(self._config.max_assembly_tokens)

        if self.context_length > 0 and self._config.reserve_tokens_floor > 0:
            reserve_cap = self.context_length - self._config.reserve_tokens_floor
            if reserve_cap > 0:
                caps.append(reserve_cap)
            else:
                logger.warning(
                    "LCM reserve_tokens_floor=%d disables reserve-based assembly cap because context_length=%d",
                    self._config.reserve_tokens_floor,
                    self.context_length,
                )

        if not caps:
            return None

        return max(1, min(caps))

    # -- Internal: helpers -------------------------------------------------

    def _assemble_overflow_recovery_context(
        self,
        system_msg: Optional[Dict[str, Any]],
        tail_messages: List[Dict[str, Any]],
        assembly_cap_override: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if tail_messages:
            first = tail_messages[0]
            content = first.get("content") or ""
            role = first.get("role") or ""
            if role == "assistant" and self._looks_like_active_summary_blob(content):
                candidate = self._assemble_context(
                    system_msg,
                    tail_messages[1:],
                    assembly_cap_override=assembly_cap_override,
                    include_lcm_note=False,
                )
                if any(
                    (msg.get("content") or "") == content
                    for msg in (candidate[1:] if system_msg is not None else candidate)
                ):
                    return candidate

        candidate = self._assemble_context(
            system_msg,
            tail_messages,
            assembly_cap_override=assembly_cap_override,
            include_lcm_note=False,
        )
        minimum_candidate_len = 1 if system_msg is not None else 0
        if len(candidate) == minimum_candidate_len and tail_messages:
            fallback = ([system_msg] if system_msg is not None else []) + [tail_messages[-1]]
            return self._sanitize_active_context_messages(fallback)
        return candidate

    @staticmethod
    def _looks_like_active_summary_blob(content: str) -> bool:
        if not isinstance(content, str) or not content:
            return False
        block = (
            r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]\n"
            r".*?\n"
            r"\[Expand for details: .*?\]"
        )
        pattern = rf"^{block}(?:\n\n---\n\n{block})*$"
        return re.fullmatch(pattern, content, flags=re.DOTALL) is not None

    @staticmethod
    def _extract_expand_hint(summary: str) -> str:
        """Extract the 'Expand for details about:' line from a summary."""
        marker = "Expand for details about:"
        idx = summary.rfind(marker)
        if idx >= 0:
            hint = summary[idx + len(marker):].strip()
            # Take first line only
            return hint.split("\n")[0].strip()
        return ""

    # -- Rotate ------------------------------------------------------------

    def backup_dir(self) -> Path:
        """Return the directory where LCM backup snapshots are written.

        Centralized so the timestamped ``/lcm backup`` slot and the rolling
        ``/lcm rotate apply`` slot share the same directory derivation.
        """
        db_path = Path(self._store.db_path)
        backup_root = (
            Path(self._hermes_home).expanduser()
            if getattr(self, "_hermes_home", "")
            else db_path.parent
        )
        return backup_root / "backups" / "lcm"

    def rotate_backup_path(self) -> Path:
        """Return the rolling rotate-latest SQLite backup path for this engine.

        Centralized so command.py (which writes the backup) and get_status()
        (which reads its mtime to surface last_rotate_at) cannot drift.
        """
        db_path = Path(self._store.db_path)
        return self.backup_dir() / f"{db_path.stem}-rotate-latest.sqlite3"

    def rotate_active_session(
        self,
        *,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Compact the active session in-place without changing identity.

        Read-only by default (``apply=False``). Returns a preview describing
        what would change. When ``apply=True``, advances the lifecycle frontier
        marker past the pre-tail raw messages so they are no longer replayed
        into active context on subsequent bootstrap. Raw messages remain in
        the SQLite store and are recoverable through ``lcm_load_session`` and
        ``lcm_expand`` — the lossless raw recovery contract is preserved.

        Refuses on sessions that are unbound, ignored, or stateless.

        Two frontier markers are intentionally kept separate:

        - The **persisted lifecycle frontier**
          (``lifecycle_state.current_frontier_store_id``) is the
          bootstrap signal — on next session start, raw rows at or
          below it are not replayed into the active context. Rotate
          advances this marker.
        - The **in-process source-mapping marker**
          (``self._last_compacted_store_id``) tracks raw rows that the
          *current process* has already moved into summary DAG nodes.
          ``_get_store_ids_for_messages`` uses it to filter candidates
          when mapping in-memory active messages back to ``store_id``.
          Rotate deliberately does NOT advance this marker: pre-tail
          raw messages remain in the in-memory active context until
          the host rebuilds it, so a normal ``compress()`` later in
          the same process can still summarize them with correct
          ``source_ids`` lineage. On next process start,
          ``_bind_lifecycle_state`` reads the persisted frontier into
          the in-process marker — at that point the active context is
          being built from scratch, so the contract holds.

        Refusal/no-op reason codes (returned as ``reason``):

        - ``no_active_session``: engine has no bound session or conversation.
        - ``session_ignored``: foreground session matched
          ``LCM_IGNORE_SESSION_PATTERNS``.
        - ``session_stateless``: foreground session matched
          ``LCM_STATELESS_SESSION_PATTERNS``.
        - ``no_pre_tail_content``: total stored messages do not exceed
          ``fresh_tail_count``; nothing to rotate.
        - ``empty_tail``: tail query returned no rows despite a non-zero
          count (concurrent deletion race); rotate cannot compute a boundary.
        - ``frontier_already_ahead``: lifecycle frontier is already at or
          past the proposed new frontier; rotate is a no-op.
        - ``stale_lifecycle_state``: apply requested but lifecycle's
          ``current_session_id`` did not match this engine's session, so
          ``advance_frontier`` did not persist the change.
        """
        session_id = self._session_id
        conversation_id = self._conversation_id

        if not session_id or not conversation_id:
            return {"ok": False, "reason": "no_active_session"}
        if self._session_ignored:
            return {"ok": False, "reason": "session_ignored", "session_id": session_id}
        if self._session_stateless:
            return {"ok": False, "reason": "session_stateless", "session_id": session_id}

        fresh_tail_count = max(1, int(self._config.fresh_tail_count))
        total_count = int(self._store.get_session_count(session_id))

        state = self._lifecycle.get_by_conversation(conversation_id)
        current_frontier = int(state.current_frontier_store_id) if state else 0

        base = {
            "ok": True,
            "session_id": session_id,
            "conversation_id": conversation_id,
            "total_message_count": total_count,
            "fresh_tail_count": fresh_tail_count,
            "current_frontier_store_id": current_frontier,
            "mode": "apply" if apply else "preview",
        }

        if total_count <= fresh_tail_count:
            return {
                **base,
                "noop": True,
                "reason": "no_pre_tail_content",
                "pre_tail_message_count": 0,
                "new_frontier_store_id": current_frontier,
            }

        tail = self._store.get_session_tail(session_id, fresh_tail_count)
        if not tail:
            # Concurrent deletion can empty the tail after the count check.
            # Surface the same shape callers expect for any other no-op so
            # downstream formatters can render it without KeyError.
            return {
                **base,
                "noop": True,
                "reason": "empty_tail",
                "pre_tail_message_count": 0,
                "new_frontier_store_id": current_frontier,
            }

        smallest_tail_store_id = int(tail[0].get("store_id") or 0)
        new_frontier = max(0, smallest_tail_store_id - 1)
        pre_tail_count = max(0, total_count - len(tail))

        is_noop = new_frontier <= current_frontier
        result = {
            **base,
            "pre_tail_message_count": pre_tail_count,
            "new_frontier_store_id": new_frontier,
            "noop": is_noop,
        }
        if is_noop:
            # Set the reason for both preview and apply so downstream
            # formatters can render a stable explanation. Preview previously
            # omitted the reason, which left _rotate_apply_text's preflight
            # check unable to distinguish frontier-already-ahead from other
            # no-ops.
            result["reason"] = "frontier_already_ahead"

        if not apply:
            return result

        if is_noop:
            return result

        new_state = self._lifecycle.advance_frontier(
            conversation_id,
            session_id,
            new_frontier,
        )
        # advance_frontier silently returns the unchanged state when its
        # session_id check fails (lifecycle_state.py:557-559). Detect that
        # by checking whether the persisted frontier actually advanced; only
        # promote the in-process marker on a confirmed persist.
        persisted_frontier = (
            int(new_state.current_frontier_store_id) if new_state else current_frontier
        )
        if persisted_frontier < new_frontier:
            return {
                **{k: v for k, v in result.items() if k != "ok"},
                "ok": False,
                "noop": False,
                "reason": "stale_lifecycle_state",
                "applied_frontier_store_id": persisted_frontier,
            }
        # Deliberately do NOT touch self._last_compacted_store_id here.
        # The in-process source-mapping marker must stay aligned with the
        # in-memory active context the host is still using. Pre-tail raw
        # messages remain in that active context until the host rebuilds
        # it; advancing the marker would make
        # _get_store_ids_for_messages filter out those rows on the next
        # in-process compress(), producing summary nodes whose text
        # covers pre-rotate messages but whose source_ids reference only
        # post-rotate rows. The persisted lifecycle frontier we just
        # advanced is the bootstrap signal for the next process start,
        # where _bind_lifecycle_state will read it into the marker
        # against a freshly-built active context.
        result["applied_frontier_store_id"] = persisted_frontier
        return result

    # -- Lifecycle ---------------------------------------------------------

    def shutdown(self):
        self._store.close()
        self._dag.close()
        self._lifecycle.close()
