"""Session row decoding, prompt cleanup and the kanban ownership reads."""


import json
import logging
import os
import re
import sqlite3
from typing import Any
from persistence.sqlite.conversation._store.sql_common import escape_like as _escape_like, _placeholders
from hermes_state_pidns import holder_pid_checkable
try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]
from persistence.sqlite._holder import StoreHolder


logger = logging.getLogger("hermes_state")


_MAX_SAFE_MESSAGES = 20_000  # resume/export guard default


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
        except Exception:  # health: allow BLE001 -- a pid psutil cannot probe reads as alive
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


_AUTO_PRUNE_STALE_OPEN_SOURCES: tuple[str, ...] = (
    "cli", "cron", "kanban", "acp", "api_server", "subagent", "tool", "recovered",
)


_IMPORT_MAX_SESSIONS, _IMPORT_MAX_MESSAGES_PER_SESSION, _IMPORT_MAX_TOTAL_MESSAGES = 500, 10_000, 50_000


_IMPORT_MAX_SESSION_BYTES, _IMPORT_MAX_TOTAL_BYTES = 5 * 1024 * 1024, 25 * 1024 * 1024


def _delete_unreferenced_system_prompts(conn) -> None:
    conn.execute(
        "DELETE FROM system_prompts WHERE NOT EXISTS ("
        "SELECT 1 FROM sessions WHERE sessions.system_prompt_hash = system_prompts.hash) AND NOT EXISTS ("
        "SELECT 1 FROM sessions WHERE sessions.tool_names = system_prompts.hash)"
    )


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


MAX_TITLE_LENGTH = 100


CANONICAL_BOT_CHAT_TITLE = "Bot Chat"


_CONTENT_JSON_PREFIX = "\x00json:"


REACTIONS_METADATA_KEY = "reactions"


_CONVERSATION_ROW_COLUMNS = (
    "id, role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
    "finish_reason, reasoning, reasoning_content, reasoning_details, "
    "codex_reasoning_items, codex_message_items, platform_message_id, observed, "
    "_compressed_summary, timestamp, token_count, active, api_content, display_kind, display_metadata, message_uid, "
    "absorbed_message_uids, tool_call_uids, tool_call_uid"
)


def retag_kanban_worker_sessions(holder: StoreHolder, workspaces_root: str) -> int:
    """Retag legacy kanban worker rows from ``cli`` to ``kanban`` by cwd under the board's workspaces
    root; gated once per root via state_meta. Returns rows retagged."""
    prefix = str(workspaces_root).rstrip("/\\")
    if not prefix:
        return 0
    gate = f"kanban_worker_source_retagged:{prefix}"
    if holder.get_meta(gate) == "1":
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
        holder.set_meta(gate, "1", cursor=cursor)
        return retagged
    return holder._execute_write(_do)


def is_kanban_owned_session(holder: StoreHolder, session_id: str) -> bool:
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
    lineage = _resume_lineage_ids(holder, session_id)
    if not lineage:
        return False
    rows = holder._read_all(
        f"SELECT 1 FROM sessions WHERE id IN ({_placeholders(lineage)}) "
        "AND (source = 'kanban' OR created_source = 'kanban') LIMIT 1",
        tuple(lineage))
    return bool(rows)


# Sibling store modules import this module's copies in turn, so their copies are imported after these.
from persistence.sqlite.conversation._store.messages import _resume_lineage_ids
