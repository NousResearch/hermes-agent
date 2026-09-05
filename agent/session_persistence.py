"""Durable transcript persistence for ``AIAgent`` (mixin; MRO-resolved from ``run_agent``): SQLite flush
with intrinsic ``_DB_PERSISTED_MARKER`` dedup, ephemeral-scaffolding filtering, optional JSON session log,
trajectory export."""
import hashlib
import json
import logging
import re
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    _DB_PERSISTED_MARKER,
    ContextCompressor,
    user_originated_turn_view,
)
from agent.lazy_forward import forward as _forward, forward_static as _forward_static
from agent.memory_manager import sanitize_context
from agent.redact import redact_sensitive_text
from agent.tool_dispatch_helpers import _is_multimodal_tool_result, _multimodal_text_summary
from agent.trajectory import convert_scratchpad_to_think, save_trajectory as _save_trajectory_to_file
from agent.transcript_repair import sync_flushed_message_markers
from utils import atomic_json_write

logger = logging.getLogger("run_agent")  # origin module's name: log records / caplog filters unchanged

# Flags marking ephemeral recovery scaffolding the loop pops before appending the real response.
# Persistence must skip them or a resumed session replays synthetic turns / breaks prefix-cache reuse.
_EPHEMERAL_SCAFFOLDING_FLAGS = (
    "_empty_recovery_synthetic",
    "_empty_terminal_sentinel",
    "_thinking_prefill",
    "_verification_stop_synthetic",  # verify-on-stop nudge; the assistant candidate itself is NOT synthetic
    "_pre_verify_synthetic",
    "_kanban_stop_synthetic",  # kanban worker stop-guard
    "_dropped_toolcall_nudge",  # internal retry instruction; must not replay as user context
)

_IMAGE_PART_TYPES = {"image", "image_url", "input_image"}
# Reasoning/codex fields are role-gated (assistant-only) inside _insert_message_rows.
_ROW_REASONING_KEYS = ("reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items")


def _is_ephemeral_scaffolding(msg: Any) -> bool:
    """True when ``msg`` is internal recovery scaffolding that must never reach the durable transcript."""
    return isinstance(msg, dict) and any(msg.get(flag) for flag in _EPHEMERAL_SCAFFOLDING_FLAGS)


# `_DB_PERSISTED_MARKER` (agent.context_compressor) is the intrinsic "already written to SQLite" marker: an
# id(msg) set can alias a freed dict's address onto a new message, a key on the dict cannot. The `_` prefix is
# mandatory (wire sanitizers strip `_` keys). CONTRACT: the marker asserts the dict's CONTENT is durable as
# written — any in-place mutation that must persist MUST pop it (turn_finalizer, context_compressor).


def _safe_session_filename_component(session_id: str) -> str:
    """Path-safe component for a (possibly untrusted ``X-Hermes-Session-Id``) ID: non ``[A-Za-z0-9_-]`` → ``_``,
    capped, plus a content hash when changed so distinct IDs cannot collide."""
    raw = str(session_id or "").strip()
    sanitized = re.sub(r"[^\w-]", "_", raw).strip("._")[:96] or "session"
    if raw and sanitized == raw:
        return sanitized
    return f"{sanitized}_{hashlib.sha256(raw.encode('utf-8', errors='surrogatepass')).hexdigest()[:12]}"


def _override_replaces_content(msg: Dict, content: Any, override: Any) -> bool:
    """May the persist override replace ``content``? A plain-text override must not replace native image/audio
    blocks (a list override is the clean multimodal payload and does), nor a message MERGED with a compaction
    summary (overwriting would drop the summary)."""
    return (
        override is not None
        and not msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
        and (not isinstance(content, list) or isinstance(override, list))
    )


def _summary_display_kind(msg: Dict) -> Any:
    """Standalone handoffs are hidden so they never occupy the active user slot in retry/undo dispatch;
    merge-into-tail carriers keep their prior visibility."""
    if (
        msg.get(COMPRESSED_SUMMARY_METADATA_KEY)
        and user_originated_turn_view(msg) is None
        and (
            ContextCompressor.classify_summary_content(msg.get("content")) == "standalone"
            or not msg.get("_compressed_summary_has_user_turn")
        )
    ):
        return "hidden"
    return msg.get("display_kind")


def _durable_content(content: Any) -> Any:
    """Text-only DB projection: multimodal envelopes → summary; part lists keep text, images → ``[screenshot]``."""
    if _is_multimodal_tool_result(content):
        return _multimodal_text_summary(content)
    if not isinstance(content, list):
        return content
    txt = [
        str(p.get("text", "")) if p.get("type") == "text" else "[screenshot]"
        for p in content
        if isinstance(p, dict) and (p.get("type") == "text" or p.get("type") in _IMAGE_PART_TYPES)
    ]
    return "\n".join(txt) if txt else None


def _persist_lock(agent):
    """Close and turn-start persistence can run on separate CLI threads: one critical section.

    ``__init__`` always creates ``_session_persist_lock``; only ``object.__new__``-built test stubs lack it
    (they run unlocked, matching the historical ``if persist_lock is None`` branch).
    """
    lock = getattr(agent, "_session_persist_lock", None)
    return nullcontext() if lock is None else lock


# --- flush phases (module-level so the flush also works bound onto duck-typed agents) ---

def _db_flush_seed_ids(agent) -> set:
    """One-shot ``_flushed_db_message_ids`` seed (same session, after a non-empty flush); the scan translates
    it to markers and the flush clears it."""
    current_session_id = getattr(agent, "session_id", None)
    same_session = getattr(agent, "_flushed_db_message_session_id", None) == current_session_id
    seed_ids = getattr(agent, "_flushed_db_message_ids", None) if same_session and agent._last_flushed_db_idx != 0 else None
    agent._flushed_db_message_session_id = current_session_id
    return seed_ids if isinstance(seed_ids, set) else set()


def _db_flush_scan_start(agent, messages: List[Dict]) -> int:
    """Skip the identity-matched, still-marked prefix of the previous flush's snapshot."""
    scan_start = 0
    for prev, cur in zip(getattr(agent, "_db_flush_scan_prefix", None) or (), messages):
        if cur is not prev or not cur.get(_DB_PERSISTED_MARKER):
            break
        scan_start += 1
    return scan_start


def _db_flush_row(agent, msg: Dict, is_current_turn_user: bool) -> Dict[str, Any]:
    """Build the session-db row for ``msg``, applying the persist override to THIS row only."""
    role = msg.get("role", "unknown")
    content = msg.get("content")
    # api_content sidecar: exact bytes sent to the API when they differ from clean content (replay parity).
    api_content = msg.get("api_content") if isinstance(msg.get("api_content"), str) else None
    timestamp = msg.get("timestamp")
    if is_current_turn_user and role == "user":
        override = getattr(agent, "_persist_user_message_override", None)
        if _override_replaces_content(msg, content, override):
            # Live content is what the wire sent, the override is the clean transcript; keep the sent bytes.
            if api_content is None and isinstance(content, str) and content != override:
                api_content = content
            content = override
        ov_timestamp = getattr(agent, "_persist_user_message_timestamp", None)
        timestamp = timestamp if ov_timestamp is None else ov_timestamp
    if api_content == content:
        api_content = None
    # get_messages_as_conversation replays rows through sanitize_context().strip(); capture the sent bytes
    # when they would differ (compared in wire form).
    if (
        api_content is None and role in ("user", "assistant") and isinstance(content, str) and content
        and sanitize_context(content).strip() != content.strip()
    ):
        api_content = content
    # Key order is the divert-JSONL wire order (divert_session_transcript_jsonl).
    row = {
        "role": role, "content": _durable_content(content), "tool_name": msg.get("tool_name"),
        "tool_calls": msg["tool_calls"] if isinstance(msg.get("tool_calls"), list) else None,
        "tool_call_id": msg.get("tool_call_id"), "finish_reason": msg.get("finish_reason"),
        **{k: msg.get(k) for k in _ROW_REASONING_KEYS},
        "_compressed_summary": bool(msg.get(COMPRESSED_SUMMARY_METADATA_KEY)),
        "timestamp": timestamp, "api_content": api_content,
        "display_kind": _summary_display_kind(msg), "display_metadata": msg.get("display_metadata"),
        "platform_message_id": msg.get("platform_message_id"),  # load-bearing for restart drain-window recovery dedup
    }
    if isinstance(msg.get("_row_id"), int):
        row["_row_id"] = msg["_row_id"]
    return row


def _db_flush_collect(agent, messages: List[Dict], conversation_history: Optional[List[Dict]]):
    """Scan for un-flushed messages; returns ``(rows, msgs)`` to write in one transaction."""
    seed_ids = _db_flush_seed_ids(agent)
    history_ids = {id(item) for item in (conversation_history or []) if isinstance(item, dict)}
    ov_idx = getattr(agent, "_persist_user_message_idx", None)
    # Also match the staged CLI dict by identity — the close safety-net may flush a shortened snapshot whose
    # turn index refers to the full history.
    pending_cli_message = getattr(agent, "_pending_cli_user_message", None)
    batch_rows: List[Dict[str, Any]] = []
    batch_msgs: List[Dict] = []
    for msg_idx in range(_db_flush_scan_start(agent, messages), len(messages)):
        msg = messages[msg_idx]
        # Append-only flush: a mid-turn persist of scaffolding would commit a synthetic turn the end-of-turn
        # drop cannot un-write. Skip regardless of position.
        if not isinstance(msg, dict) or _is_ephemeral_scaffolding(msg) or msg.get(_DB_PERSISTED_MARKER):
            continue
        # Already durable (history copy or caller-seeded): stamp so future flushes skip it.
        if id(msg) in history_ids or id(msg) in seed_ids:
            msg[_DB_PERSISTED_MARKER] = True
            continue
        batch_rows.append(_db_flush_row(agent, msg, ov_idx == msg_idx or msg is pending_cli_message))
        batch_msgs.append(msg)
    return batch_rows, batch_msgs


def _db_flush_write(agent, batch_rows: List[Dict[str, Any]], batch_msgs: List[Dict]) -> None:
    """One transaction for the turn's new rows: on failure nothing lands and no markers are stamped."""
    if not batch_rows:
        return
    agent._session_db.append_messages_batch(
        session_id=agent.session_id, messages=batch_rows,
        compression_lock_holder=getattr(agent, "_active_compression_lock_holder", None),
        turn_lease_holder=getattr(agent, "_active_session_turn_lease_holder", None),
        turn_lease_ttl_seconds=getattr(agent, "_active_session_turn_lease_ttl_seconds", 300.0) or 300.0,
    )
    sync_flushed_message_markers(batch_msgs, batch_rows)


def _db_flush_adopt_compression_tip(agent) -> bool:
    """Adopt the live continuation of a compression-closed session. Same-id tip = no continuation; a tip
    whose row is missing or already ended is not adopted either."""
    old_id = agent.session_id
    try:
        tip = agent._session_db.get_compression_tip(old_id)
    except Exception as tip_exc:
        logger.warning("compression tip lookup failed for %s: %s", old_id, tip_exc)
        return False
    if not tip or tip == old_id:
        return False
    try:
        tip_row = agent._session_db.get_session(tip)
    except Exception:
        tip_row = None
    if tip_row is None or tip_row.get("ended_at") is not None:
        return False
    logger.warning("Adopted live compression tip %s for closed session %s; retrying flush once", tip, old_id)
    agent.session_id, agent._flushed_db_message_ids, agent._last_flushed_db_idx = tip, set(), 0
    agent._compression_adoption_failed = False
    return True


def _db_flush_failed(agent, e: Exception, batch_rows: List[Dict[str, Any]], adoption_budget: int) -> bool:
    """Classify a failed flush; True when the caller should retry once on an adopted compression tip."""
    agent._db_flush_scan_prefix = None  # full re-scan next flush: an exception mid-loop leaves mixed dispositions
    # The only place the SQLite error is visible before it becomes a bare False — classify it so the turn-end
    # explanation can distinguish lock contention from disk-full/read-only.
    from hermes_state import StateDbCorruptError, StateDbReplacedError, classify_persistence_error, divert_session_transcript_jsonl
    from hermes_state_errors import CompressionSessionClosedError
    agent._last_persistence_error_cause = classify_persistence_error(e)
    if isinstance(e, (StateDbReplacedError, StateDbCorruptError)):
        # A replaced/quarantined handle will not take this batch again — keep it on disk.
        try:
            divert_session_transcript_jsonl(getattr(agent, "session_id", "") or "", batch_rows)
        except Exception:
            logger.warning("JSONL divert failed after state.db %s for %s",
                           agent._last_persistence_error_cause, getattr(agent, "session_id", None), exc_info=True)
    if isinstance(e, CompressionSessionClosedError):
        # Compression race: another path rotated this session mid-write. Retry exactly once on the live tip; a
        # second closed-parent write fails closed.
        if adoption_budget > 0 and _db_flush_adopt_compression_tip(agent):
            return True
        agent._compression_adoption_failed = True  # lets the turn explanation name rotation, not full-disk advice
    logger.warning("Session DB append_message failed: %s", e)
    return False


def _session_log_entry(agent, msg: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of ``msg`` with scratchpad tags normalised and credentials redacted (honours HERMES_REDACT_SECRETS)."""
    if "content" not in msg:
        return msg
    content = msg["content"]
    if msg.get("role") == "assistant" and content:
        content = agent._clean_session_content(content)
    return {**msg, "content": agent._redact_message_content(content)}


def _existing_log_is_larger(log_file, count: int) -> bool:
    """Never overwrite a larger log with fewer messages (resumed agent with partial history); a corrupted
    existing file allows the overwrite."""
    if not log_file.exists():
        return False
    try:
        existing = json.loads(log_file.read_text(encoding="utf-8"))
        existing_count = existing.get("message_count", len(existing.get("messages", [])))
    except Exception:
        return False
    if existing_count > count:
        logging.debug("Skipping session log overwrite: existing has %d messages, current has %d", existing_count, count)
        return True
    return False


class SessionPersistenceMixin:
    """Session DB flush, session log and trajectory persistence (see module docstring)."""

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """Rewrite the current-turn user message in place: some paths send an API-only variant that must not
        leak into transcripts or resumed history."""
        idx = getattr(self, "_persist_user_message_idx", None)
        override = getattr(self, "_persist_user_message_override", None)
        timestamp = getattr(self, "_persist_user_message_timestamp", None)
        platform_id = getattr(self, "_persist_user_message_platform_id", None)
        if idx is None or (override is None and timestamp is None and platform_id is None):
            return
        msg = messages[idx] if 0 <= idx < len(messages) else None
        if not (isinstance(msg, dict) and msg.get("role") == "user"):
            return
        if _override_replaces_content(msg, msg.get("content"), override):
            msg["content"] = override
        if timestamp is not None:
            msg["timestamp"] = timestamp
        if platform_id is not None:  # load-bearing for restart drain-window recovery dedup (has_platform_message_id)
            msg["platform_message_id"] = platform_id

    def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """Save to JSON log and SQLite on any exit path. Trailing empty-response scaffolding is dropped from
        the live list; the persist override is applied to the DB row only.

        The persist user-message *override* is NOT applied here — it is resolved inside
        ``_flush_messages_to_session_db`` and written only to the DB row, never mutating the live message
        list used by the API call (#48677 is thus closed for every persist caller, not just this one).
        """
        from agent.agent_runtime_helpers import note_turn_persisted
        with _persist_lock(self):
            self._drop_trailing_empty_response_scaffolding(messages)
            self._session_messages = messages
            self._save_session_log(messages)
            self._flush_messages_to_session_db(messages, conversation_history)
            # Drain async token-accounting deltas at every persist point; cheap no-op when nothing queued.
            if self._session_db is not None:
                self._session_db.flush_token_counts()
            note_turn_persisted(self)

    def _drop_trailing_empty_response_scaffolding(self, messages: List[Dict]) -> None:
        """Pop empty-response retry scaffolding from the tail, then (only if any was present) rewind the
        tool-result / assistant(tool_calls) pair the failed iteration left hanging — otherwise the next user
        turn lands as ``...tool, user`` and providers return empty content forever."""
        def tail(*keys: str) -> bool:
            return bool(messages) and isinstance(messages[-1], dict) and any(messages[-1].get(k) for k in keys)

        def tail_role(role: str) -> bool:
            return bool(messages) and isinstance(messages[-1], dict) and messages[-1].get("role") == role

        dropped_scaffolding = False
        while tail("_empty_recovery_synthetic", "_empty_terminal_sentinel"):
            messages.pop()
            dropped_scaffolding = True
        if not dropped_scaffolding:
            return
        while tail_role("tool"):
            messages.pop()
        # Providers reject a dangling assistant(tool_calls) whose results were just popped.
        if tail_role("assistant") and tail("tool_calls"):
            messages.pop()

    _repair_message_sequence = _forward("agent.agent_runtime_helpers", "repair_message_sequence")

    def _flush_messages_to_session_db(self, messages: List[Dict], conversation_history: Optional[List[Dict]] = None):
        """Serialize direct and turn-boundary session flushes per agent."""
        with _persist_lock(self):
            return self._flush_messages_to_session_db_unlocked(messages, conversation_history)

    def _flush_messages_to_session_db_unlocked(
        self, messages: List[Dict], conversation_history: Optional[List[Dict]] = None, _adoption_budget: int = 1,
    ):
        """Persist un-flushed messages to SQLite. Dedup is the intrinsic ``_DB_PERSISTED_MARKER`` on each written
        dict — not positional slices (drift after sequence repair) nor an ``id(msg)`` set (address reuse). The
        persist override touches the written row only. A compression-closed session adopts its live tip and
        retries exactly once.

        Deduplicates via an intrinsic ``_DB_PERSISTED_MARKER`` stamped on each written message dict, so
        repeated calls (from multiple exit paths) only write truly new messages — preventing the
        duplicate-write bug (#860) without relying on positional slices that can drift after
        message-sequence repair, and without a retained ``id(msg)`` set that CPython could alias onto a
        freed-then-reused address (#50372). The ``_flushed_db_message_ids`` attribute is now only a one-shot
        seed (translated to markers, then cleared each flush), not a persisted set.
        """
        # Persistence-isolated agents (background review fork) share the parent's session_id for cache warmth;
        # a write here would land the curator's turn in the user's real history.
        if getattr(self, "_persist_disabled", False) or not self._session_db:
            return None
        batch_rows: List[Dict[str, Any]] = []
        try:
            if not self._session_db_created:  # retry row creation if the earlier attempt failed transiently
                self._ensure_db_session()
            batch_rows, batch_msgs = _db_flush_collect(self, messages, conversation_history)
            _db_flush_write(self, batch_rows, batch_msgs)
            # Markers are now the sole truth; reset the one-shot seed so no id() outlives this flush.
            self._flushed_db_message_ids = set()
            self._last_flushed_db_idx = len(messages)
            # Snapshot for the bounded scan — only on full success, so a partial list is never treated as settled.
            self._db_flush_scan_prefix = messages[:]
            return True
        except Exception as e:
            if _db_flush_failed(self, e, batch_rows, _adoption_budget):
                return self._flush_messages_to_session_db_unlocked(messages, conversation_history, _adoption_budget=0)
            return False

    def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """Messages before the last assistant turn (rollback point for a malformed final answer); all if none."""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                return messages[:i]
        return messages.copy()

    _format_tools_for_system_message = _forward("agent.system_prompt", "format_tools_for_system_message")
    _convert_to_trajectory_format = _forward("agent.agent_runtime_helpers", "convert_to_trajectory_format")

    def _save_trajectory(self, messages: List[Dict[str, Any]], user_query: str, completed: bool):
        """Save conversation trajectory to JSONL file."""
        if not self.save_trajectories:
            return
        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        _save_trajectory_to_file(trajectory, self.model, completed)

    _extract_api_error_context = _forward_static("agent.agent_runtime_helpers", "extract_api_error_context")
    _dump_api_request_debug = _forward("agent.agent_runtime_helpers", "dump_api_request_debug")

    @staticmethod
    def _clean_session_content(content: str) -> str:
        """Convert REASONING_SCRATCHPAD to think tags and clean up whitespace."""
        if not content:
            return content
        content = re.sub(r'\n+(<think>)', r'\n\1', convert_scratchpad_to_think(content))
        return re.sub(r'(</think>)\n+', r'\1\n', content).strip()

    @staticmethod
    def _redact_message_content(content):
        """Redact secrets in str or list-of-parts content (text fields only; honours HERMES_REDACT_SECRETS)."""
        if isinstance(content, str):
            return redact_sensitive_text(content)
        if not isinstance(content, list):
            return content
        return [{**p, **{k: redact_sensitive_text(p[k]) for k in ("text", "content") if isinstance(p.get(k), str)}}
                if isinstance(p, dict) else p for p in content]

    def _save_session_log(self, messages: List[Dict[str, Any]] = None):
        """Optional per-session JSON snapshot (``sessions.write_json_snapshots``, default False) for external
        tooling; state.db is canonical. Rewrites the full list after every persistence point."""
        if not getattr(self, "_session_json_enabled", False):
            return
        messages = messages or self._session_messages
        if not messages:
            return
        try:  # re-derive the path each call so /branch and /compress land in the right file
            log_file = self.logs_dir / f"session_{_safe_session_filename_component(self.session_id)}.json"
        except Exception:
            return
        try:
            # Mirror the SQLite flush: scaffolding is never durable transcript content.
            cleaned = [_session_log_entry(self, msg) for msg in messages if not _is_ephemeral_scaffolding(msg)]
            if _existing_log_is_larger(log_file, len(cleaned)):
                return
            entry = {
                "session_id": self.session_id, "model": self.model, "base_url": self.base_url, "platform": self.platform,
                "session_start": self.session_start.isoformat(), "last_updated": datetime.now().isoformat(),
                "system_prompt": redact_sensitive_text(self._cached_system_prompt or ""), "tools": self.tools or [],
                "message_count": len(cleaned), "messages": cleaned,
            }
            from hermes_cli.config import artifact_file_mode

            atomic_json_write(log_file, entry, indent=2, mode=artifact_file_mode(), default=str)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")
