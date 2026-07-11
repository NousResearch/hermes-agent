"""Safety helpers for gateway interrupted-turn auto-continuation.

This module supports the existing boot-resume scheduler in ``gateway.run``.  It
classifies persisted tool-call tails and stores the once-ever auto-resume credit;
it does not schedule turns itself.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from agent.replay_cleanup import is_interrupted_tool_result

logger = logging.getLogger(__name__)

AUTO_RESUME_ATTEMPT_TTL_SECONDS = 7 * 24 * 60 * 60
_STORE_VERSION = 1

# Auto-resume is allowlist-based.  Unknown tools fail closed because plugins and
# MCP servers can expose arbitrary side effects under names core cannot classify.
_READ_ONLY_TOOLS = frozenset(
    {
        "browser_get_images",
        "browser_snapshot",
        "ha_get_state",
        "ha_list_entities",
        "ha_list_services",
        "lcm_describe",
        "lcm_doctor",
        "lcm_expand",
        "lcm_expand_query",
        "lcm_grep",
        "lcm_load_session",
        "lcm_status",
        "mem0_profile",
        "mem0_search",
        "read_file",
        "search_files",
        "session_search",
        "skill_view",
        "skills_list",
        "vision_analyze",
        "web_extract",
        "web_search",
    }
)

# These known core surfaces mutate state or can dispatch mutations.  A persisted
# result proves completion and is safe to continue past; a missing result is the
# exact ambiguous tail the mechanical gate must stop.
_MUTATING_TOOLS = frozenset(
    {
        "browser_back",
        "browser_click",
        "browser_navigate",
        "browser_press",
        "browser_scroll",
        "browser_type",
        "clarify",
        "delegate_task",
        "execute_code",
        "ha_call_service",
        "kanban_block",
        "kanban_comment",
        "kanban_complete",
        "kanban_create",
        "kanban_heartbeat",
        "kanban_link",
        "memory",
        "mem0_conclude",
        "patch",
        "process",
        "skill_manage",
        "terminal",
        "todo",
        "write_file",
    }
)


@dataclass(frozen=True)
class InterruptedTurnAssessment:
    """Mechanical auto-resume disposition for one persisted interrupted turn."""

    turn_rowid: int | None
    auto_eligible: bool
    suspect_tool: str | None = None
    reason: str | None = None


def _message_id(row: dict[str, Any]) -> int | None:
    value = row.get("id")
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _is_nonempty_human_message(row: dict[str, Any]) -> bool:
    if row.get("role") != "user":
        return False
    content = row.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    return content not in (None, [], {})


def _turn_segment(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in messages if isinstance(row, dict)]
    start = 0
    for index, row in enumerate(rows):
        if _is_nonempty_human_message(row):
            start = index
    return rows[start:]


def _stable_turn_rowid(rows: list[dict[str, Any]]) -> int | None:
    """Return the original interrupted turn's cross-boot-stable assistant rowid.

    Startup resume notes are API-only and persist an empty user row.  On a second
    interruption, assistant rows have been appended after that synthetic boundary;
    choosing the last assistant would mint a fresh credit.  The assistant immediately
    before the first empty synthetic user remains the original turn's last assistant
    row across every restart.  Before any synthetic boundary exists, the first
    ``interrupt_close`` row is the original interruption marker; otherwise use the
    current last assistant row.
    """

    for index, row in enumerate(rows):
        if row.get("role") == "user" and row.get("content") == "":
            prior = [
                _message_id(candidate)
                for candidate in rows[:index]
                if candidate.get("role") == "assistant"
            ]
            prior = [rowid for rowid in prior if rowid is not None]
            return prior[-1] if prior else None

    interrupted = [
        _message_id(row)
        for row in rows
        if row.get("role") == "assistant"
        and row.get("finish_reason") == "interrupt_close"
    ]
    interrupted = [rowid for rowid in interrupted if rowid is not None]
    if interrupted:
        return interrupted[0]

    assistants = [
        _message_id(row) for row in rows if row.get("role") == "assistant"
    ]
    assistants = [rowid for rowid in assistants if rowid is not None]
    return assistants[-1] if assistants else None


def _tool_name(call: Any) -> tuple[str | None, str | None]:
    if not isinstance(call, dict):
        return None, None
    call_id = call.get("id")
    function = call.get("function")
    if not isinstance(call_id, str) or not call_id:
        return None, None
    if not isinstance(function, dict):
        return None, call_id
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return None, call_id
    return name.strip(), call_id


def assess_interrupted_turn(
    messages: Iterable[dict[str, Any]],
) -> InterruptedTurnAssessment:
    """Classify a persisted interrupted-turn tail conservatively.

    Read-only calls may be incomplete.  Known mutating calls require a matching
    persisted tool-result row.  Unknown or malformed calls and orphaned results are
    ambiguous and therefore prompt-only.
    """

    rows = _turn_segment(messages)
    turn_rowid = _stable_turn_rowid(rows)
    if turn_rowid is None:
        return InterruptedTurnAssessment(
            turn_rowid=None,
            auto_eligible=False,
            reason="missing persisted assistant rowid",
        )

    results: dict[str, list[tuple[int, Any]]] = {}
    for index, row in enumerate(rows):
        if row.get("role") != "tool":
            continue
        call_id = row.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            return InterruptedTurnAssessment(
                turn_rowid=turn_rowid,
                auto_eligible=False,
                reason="unclassifiable tool result",
            )
        results.setdefault(call_id, []).append((index, row.get("content")))

    seen_call_ids: set[str] = set()
    for index, row in enumerate(rows):
        raw_calls = row.get("tool_calls")
        if raw_calls in (None, []):
            continue
        if row.get("role") != "assistant" or not isinstance(raw_calls, list):
            return InterruptedTurnAssessment(
                turn_rowid=turn_rowid,
                auto_eligible=False,
                reason="unclassifiable tool call",
            )
        for call in raw_calls:
            name, call_id = _tool_name(call)
            if name is None or call_id is None:
                return InterruptedTurnAssessment(
                    turn_rowid=turn_rowid,
                    auto_eligible=False,
                    suspect_tool=name,
                    reason="unclassifiable tool call",
                )
            if call_id in seen_call_ids:
                return InterruptedTurnAssessment(
                    turn_rowid=turn_rowid,
                    auto_eligible=False,
                    suspect_tool=name,
                    reason="duplicate tool call id",
                )
            seen_call_ids.add(call_id)
            completed = any(
                position > index and not is_interrupted_tool_result(content)
                for position, content in results.get(call_id, [])
            )
            if name in _READ_ONLY_TOOLS:
                continue
            if name in _MUTATING_TOOLS:
                if completed:
                    continue
                return InterruptedTurnAssessment(
                    turn_rowid=turn_rowid,
                    auto_eligible=False,
                    suspect_tool=name,
                    reason=f"incomplete mutating tool call: {name}",
                )
            return InterruptedTurnAssessment(
                turn_rowid=turn_rowid,
                auto_eligible=False,
                suspect_tool=name,
                reason=f"unclassifiable tool call: {name}",
            )

    orphaned = set(results) - seen_call_ids
    if orphaned:
        return InterruptedTurnAssessment(
            turn_rowid=turn_rowid,
            auto_eligible=False,
            reason="unclassifiable orphaned tool result",
        )

    return InterruptedTurnAssessment(turn_rowid=turn_rowid, auto_eligible=True)


class AutoResumeAttemptStore:
    """Seven-day durable once-ever auto-resume credits.

    A malformed file poisons this store instance closed: every lookup reports an
    existing attempt and exactly one warning is emitted.  This prevents a corrupt
    safety backstop from silently granting fresh auto-resume credits.
    """

    def __init__(
        self,
        path: Path,
        *,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self._now = now
        self._invalid = False
        self._warned = False

    def _warn_invalid(self, exc: Exception | str) -> None:
        if self._warned:
            return
        self._warned = True
        logger.warning(
            "%s is unparseable; interrupted-turn auto-continuation fails closed "
            "to prompt for all sessions: %s",
            self.path.name,
            exc,
        )

    def _validate(self, raw: Any) -> list[dict[str, Any]]:
        if not isinstance(raw, dict) or raw.get("version") != _STORE_VERSION:
            raise ValueError("unsupported or missing store version")
        attempts = raw.get("attempts")
        if not isinstance(attempts, list):
            raise ValueError("attempts must be a list")
        validated: list[dict[str, Any]] = []
        for item in attempts:
            if not isinstance(item, dict):
                raise ValueError("attempt entry must be an object")
            session_key = item.get("session_key")
            rowid = item.get("assistant_rowid")
            attempted_at = item.get("attempted_at")
            if not isinstance(session_key, str) or not session_key:
                raise ValueError("attempt session_key must be a non-empty string")
            if isinstance(rowid, bool) or not isinstance(rowid, int) or rowid <= 0:
                raise ValueError("attempt assistant_rowid must be a positive integer")
            if isinstance(attempted_at, bool) or not isinstance(attempted_at, (int, float)):
                raise ValueError("attempt attempted_at must be numeric")
            validated.append(
                {
                    "session_key": session_key,
                    "assistant_rowid": rowid,
                    "attempted_at": float(attempted_at),
                }
            )
        return validated

    def _load(self) -> list[dict[str, Any]] | None:
        if self._invalid:
            return None
        if not self.path.exists():
            return []
        try:
            attempts = self._validate(json.loads(self.path.read_text(encoding="utf-8")))
            cutoff = self._now() - AUTO_RESUME_ATTEMPT_TTL_SECONDS
            current = [item for item in attempts if item["attempted_at"] >= cutoff]
            if len(current) != len(attempts):
                self._write(current)
            return current
        except Exception as exc:
            self._invalid = True
            self._warn_invalid(exc)
            return None

    def _write(self, attempts: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": _STORE_VERSION, "attempts": attempts},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, 0o600)
            os.replace(temp_path, self.path)
        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def has_attempt(self, session_key: str, assistant_rowid: int) -> bool:
        attempts = self._load()
        if attempts is None:
            return True
        return any(
            item["session_key"] == session_key
            and item["assistant_rowid"] == assistant_rowid
            for item in attempts
        )

    def consume(self, session_key: str, assistant_rowid: int) -> bool:
        """Record a scheduled auto continuation; false means fail closed."""

        attempts = self._load()
        if attempts is None:
            return False
        if any(
            item["session_key"] == session_key
            and item["assistant_rowid"] == assistant_rowid
            for item in attempts
        ):
            return False
        attempts.append(
            {
                "session_key": session_key,
                "assistant_rowid": assistant_rowid,
                "attempted_at": float(self._now()),
            }
        )
        try:
            self._write(attempts)
        except Exception as exc:
            self._invalid = True
            self._warn_invalid(exc)
            return False
        return True
