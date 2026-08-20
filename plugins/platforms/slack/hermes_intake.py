"""Slack message → Hermes Kanban intake mapping and HTTP client."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import aiohttp

from gateway.config import Platform
from gateway.session import SessionSource
from hermes_constants import get_hermes_home

BOARD = "hrms"
MAX_CONTEXT_LENGTH = 2_000
MAX_TITLE_LENGTH = 80

Transport = Callable[
    [str, dict[str, str], dict[str, Any]],
    Awaitable[dict[str, Any]],
]


@dataclass(frozen=True)
class SlackIntakeRecord:
    intake_id: str
    revision: int
    state: str
    profile: str
    team_id: str
    source_channel_id: str
    source_message_ts: str
    source_permalink: str
    submitter_id: str
    invocation_key: str
    dm_channel_id: str | None
    thread_ts: str | None
    session_key: str | None
    session_id: str | None
    first_event_id: str
    promotion_key: str
    card_board: str | None
    card_id: str | None


class SlackIntakeStore:
    """Restart-safe Slack intake lineage with transactional state changes."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or (get_hermes_home() / "slack_intakes.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        for artifact in (self.path, self.path.with_name(self.path.name + "-wal"),
                         self.path.with_name(self.path.name + "-shm")):
            if artifact.exists():
                os.chmod(artifact, 0o600)
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS slack_intakes (
                    intake_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    revision INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL,
                    claim_owner TEXT,
                    claim_pid INTEGER,
                    profile TEXT NOT NULL,
                    team_id TEXT NOT NULL,
                    source_channel_id TEXT NOT NULL,
                    source_message_ts TEXT NOT NULL,
                    source_permalink TEXT NOT NULL DEFAULT '',
                    submitter_id TEXT NOT NULL,
                    invocation_key TEXT NOT NULL UNIQUE,
                    dm_channel_id TEXT,
                    thread_ts TEXT,
                    session_key TEXT,
                    session_id TEXT,
                    first_event_id TEXT NOT NULL UNIQUE,
                    promotion_key TEXT NOT NULL UNIQUE,
                    card_board TEXT,
                    card_id TEXT UNIQUE,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    promoted_at INTEGER
                );
                CREATE TABLE IF NOT EXISTS slack_intake_sources (
                    intake_id TEXT PRIMARY KEY REFERENCES slack_intakes(intake_id),
                    source_json TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_slack_intake_thread
                    ON slack_intakes(team_id, dm_channel_id, thread_ts)
                    WHERE dm_channel_id IS NOT NULL AND thread_ts IS NOT NULL;
                """
            )
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(slack_intakes)")
            }
            if "claim_owner" not in columns:
                conn.execute("ALTER TABLE slack_intakes ADD COLUMN claim_owner TEXT")
            if "claim_pid" not in columns:
                conn.execute("ALTER TABLE slack_intakes ADD COLUMN claim_pid INTEGER")
        os.chmod(self.path, 0o600)

    @staticmethod
    def _record(row: sqlite3.Row) -> SlackIntakeRecord:
        fields = SlackIntakeRecord.__dataclass_fields__
        return SlackIntakeRecord(**{name: row[name] for name in fields})

    def reserve(
        self, invocation_key: str, profile: str, source: Mapping[str, Any]
    ) -> SlackIntakeRecord:
        now = int(time.time())
        intake_id = f"i_{uuid.uuid4().hex}"
        first_event_id = f"slack-intake:{intake_id}:first-turn"
        promotion_key = f"slack-intake:{profile}:{source['team_id']}:{intake_id}"
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM slack_intakes WHERE invocation_key = ?",
                (invocation_key,),
            ).fetchone()
            if existing is not None:
                conn.commit()
                return self._record(existing)
            conn.execute(
                """
                INSERT INTO slack_intakes (
                    intake_id, state, profile, team_id, source_channel_id,
                    source_message_ts, source_permalink, submitter_id,
                    invocation_key, first_event_id, promotion_key,
                    created_at, updated_at
                ) VALUES (?, 'reserved', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intake_id, profile, str(source["team_id"]),
                    str(source["channel_id"]), str(source["message_ts"]),
                    str(source.get("permalink") or ""),
                    str(source["submitter_id"]), invocation_key,
                    first_event_id, promotion_key, now, now,
                ),
            )
            conn.execute(
                "INSERT INTO slack_intake_sources (intake_id, source_json) VALUES (?, ?)",
                (intake_id, json.dumps(dict(source), sort_keys=True)),
            )
            row = conn.execute(
                "SELECT * FROM slack_intakes WHERE intake_id = ?", (intake_id,)
            ).fetchone()
            conn.commit()
        return self._record(row)

    def get(self, intake_id: str) -> SlackIntakeRecord:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM slack_intakes WHERE intake_id = ?", (intake_id,)
            ).fetchone()
        if row is None:
            raise KeyError(intake_id)
        return self._record(row)

    def source(self, intake_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT source_json FROM slack_intake_sources WHERE intake_id = ?",
                (intake_id,),
            ).fetchone()
        if row is None:
            raise KeyError(intake_id)
        return json.loads(row["source_json"])

    def _bind(
        self, intake_id: str, assignments: Mapping[str, Any], state: str
    ) -> SlackIntakeRecord:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT * FROM slack_intakes WHERE intake_id = ?", (intake_id,)
            ).fetchone()
            if current is None:
                raise KeyError(intake_id)
            columns = "".join(f"{name} = ?, " for name in assignments)
            conn.execute(
                f"UPDATE slack_intakes SET {columns}state = ?, "
                "revision = revision + 1, updated_at = ? WHERE intake_id = ?",
                (*assignments.values(), state, int(time.time()), intake_id),
            )
            row = conn.execute(
                "SELECT * FROM slack_intakes WHERE intake_id = ?", (intake_id,)
            ).fetchone()
            conn.commit()
        return self._record(row)

    def bind_thread(
        self, intake_id: str, dm_channel_id: str, thread_ts: str,
        *, owner_token: str | None = None,
    ) -> SlackIntakeRecord:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT * FROM slack_intakes WHERE intake_id = ?", (intake_id,)
            ).fetchone()
            if current is None:
                raise KeyError(intake_id)
            if not current["dm_channel_id"] or not current["thread_ts"]:
                owner_clause = " AND claim_owner = ?" if owner_token else ""
                conn.execute(
                    "UPDATE slack_intakes SET dm_channel_id = ?, thread_ts = ?, "
                    "state = 'thread_bound', claim_owner = NULL, claim_pid = NULL, "
                    "revision = revision + 1, updated_at = ? "
                    "WHERE intake_id = ? AND dm_channel_id IS NULL AND thread_ts IS NULL"
                    + owner_clause,
                    (
                        dm_channel_id, thread_ts, int(time.time()), intake_id,
                        *((owner_token,) if owner_token else ()),
                    ),
                )
            row = conn.execute(
                "SELECT * FROM slack_intakes WHERE intake_id = ?", (intake_id,)
            ).fetchone()
            conn.commit()
        return self._record(row)

    def bind_session(
        self, intake_id: str, session_key: str, session_id: str
    ) -> SlackIntakeRecord:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT * FROM slack_intakes WHERE intake_id = ?", (intake_id,)
            ).fetchone()
            if current is None:
                raise KeyError(intake_id)
            if not current["session_key"] or not current["session_id"]:
                conn.execute(
                    "UPDATE slack_intakes SET session_key = ?, session_id = ?, "
                    "state = 'session_bound', revision = revision + 1, updated_at = ? "
                    "WHERE intake_id = ? AND session_key IS NULL AND session_id IS NULL",
                    (session_key, session_id, int(time.time()), intake_id),
                )
            row = conn.execute(
                "SELECT * FROM slack_intakes WHERE intake_id = ?", (intake_id,)
            ).fetchone()
            conn.commit()
        return self._record(row)

    def set_state(self, intake_id: str, state: str) -> SlackIntakeRecord:
        """Advance lifecycle state without weakening the lineage invariants."""
        return self._bind(intake_id, {}, state)

    def claim_stage(
        self,
        intake_id: str,
        *,
        from_states: tuple[str, ...],
        claimed_state: str,
        stale_after: int = 60,
    ) -> str | None:
        """Durably claim one side-effecting stage, with crash-lease recovery."""
        now = int(time.time())
        owner_token = uuid.uuid4().hex
        placeholders = ",".join("?" for _ in from_states)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT state, updated_at, claim_pid FROM slack_intakes WHERE intake_id = ?",
                (intake_id,),
            ).fetchone()
            if current is None:
                raise KeyError(intake_id)
            stale_owner_dead = (
                current["state"] == claimed_state
                and int(current["updated_at"]) <= now - stale_after
                and not self._pid_is_alive(current["claim_pid"])
            )
            allowed = current["state"] in from_states or stale_owner_dead
            if not allowed:
                conn.commit()
                return None
            cur = conn.execute(
                f"UPDATE slack_intakes SET state = ?, claim_owner = ?, claim_pid = ?, "
                "revision = revision + 1, "
                f"updated_at = ? WHERE intake_id = ? AND (state IN ({placeholders}) "
                "OR (state = ? AND updated_at <= ? AND claim_pid = ?))",
                (
                    claimed_state, owner_token, os.getpid(), now, intake_id, *from_states,
                    claimed_state, now - stale_after, current["claim_pid"],
                ),
            )
            conn.commit()
        return owner_token if cur.rowcount == 1 else None

    def complete_stage(
        self, intake_id: str, *, owner_token: str, state: str
    ) -> bool:
        """Complete a claimed stage only while this worker still owns its lease."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                "UPDATE slack_intakes SET state = ?, claim_owner = NULL, claim_pid = NULL, "
                "revision = revision + 1, updated_at = ? "
                "WHERE intake_id = ? AND claim_owner = ?",
                (state, int(time.time()), intake_id, owner_token),
            )
            conn.commit()
        return cur.rowcount == 1

    @staticmethod
    def _pid_is_alive(pid: Any) -> bool:
        try:
            value = int(pid)
            if value <= 0:
                return False
            os.kill(value, 0)
            return True
        except (TypeError, ValueError, ProcessLookupError):
            return False
        except PermissionError:
            return True

    def bind_profile(self, intake_id: str, profile: str) -> SlackIntakeRecord:
        """Persist the profile selected by the gateway's normal route resolver."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE slack_intakes SET profile = ?, "
                "promotion_key = 'slack-intake:' || ? || ':' || team_id || ':' || intake_id, "
                "revision = revision + 1, "
                "updated_at = ? WHERE intake_id = ? AND profile = ''",
                (profile, profile, int(time.time()), intake_id),
            )
            row = conn.execute(
                "SELECT * FROM slack_intakes WHERE intake_id = ?", (intake_id,)
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(intake_id)
        return self._record(row)

    def bind_card(self, intake_id: str, board: str, card_id: str) -> SlackIntakeRecord:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT * FROM slack_intakes WHERE intake_id = ?", (intake_id,)
            ).fetchone()
            if current is None:
                raise KeyError(intake_id)
            if current["card_id"] is None:
                now = int(time.time())
                conn.execute(
                    "UPDATE slack_intakes SET card_board = ?, card_id = ?, "
                    "state = 'promoted', promoted_at = ?, updated_at = ?, "
                    "revision = revision + 1 WHERE intake_id = ? AND card_id IS NULL",
                    (board, card_id, now, now, intake_id),
                )
            row = conn.execute(
                "SELECT * FROM slack_intakes WHERE intake_id = ?", (intake_id,)
            ).fetchone()
            conn.commit()
        return self._record(row)

    def lineage(self, intake_id: str) -> dict[str, Any]:
        return asdict(self.get(intake_id))


def build_intake_source(
    *, profile: str, team_id: str, dm_channel_id: str, thread_ts: str,
    user_id: str, user_name: str,
) -> SessionSource:
    return SessionSource(
        platform=Platform.SLACK,
        chat_id=dm_channel_id,
        chat_type="dm",
        user_id=user_id,
        user_name=user_name,
        thread_id=thread_ts,
        scope_id=team_id,
        profile=profile,
    )


def build_intake_prompt(source: Mapping[str, Any], *, intake_id: str) -> str:
    permalink = str(source.get("permalink") or "unavailable")
    return (
        "You are beginning a private conversational intake from an authorized "
        "Slack message. Treat the following as an ordinary user-provided source, "
        "help refine it into an execution-ready brief, ask at most one material "
        "question, and do not create a Kanban card from text. End every response "
        "with the current brief using these exact Markdown headings: ## Outcome, "
        "## Why, ## Scope, ## Non-goals, ## Acceptance criteria, ## Decisions, "
        "## Constraints, and ## Unresolved questions. No card exists yet; only "
        "the persistent Create card button can promote this intake.\n\n"
        f"Intake ID: {intake_id}\nSource: {permalink}\n\n"
        f"Selected message:\n{source.get('message_text') or '(no text)'}"
    )


_BRIEF_HEADINGS = {
    "Outcome": "outcome",
    "Why": "why",
    "Scope": "scope",
    "Non-goals": "non_goals",
    "Acceptance criteria": "acceptance_criteria",
    "Decisions": "decisions",
    "Constraints": "constraints",
    "Unresolved questions": "unresolved_questions",
}


def parse_resolved_brief(messages: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Parse the latest assistant's explicit current-draft projection."""
    content = next(
        (
            str(message.get("content") or "")
            for message in reversed(messages)
            if message.get("role") == "assistant" and message.get("content")
        ),
        "",
    )
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in content.splitlines():
        heading = line.removeprefix("## ").strip() if line.startswith("## ") else ""
        if heading in _BRIEF_HEADINGS:
            current = _BRIEF_HEADINGS[heading]
            sections[current] = []
        elif current is not None and line.strip():
            sections[current].append(line.strip())
    if not sections.get("outcome"):
        raise ValueError("resolved intake brief is incomplete")
    result: dict[str, Any] = {}
    for key in _BRIEF_HEADINGS.values():
        values = sections.get(key, [])
        if key in {"outcome", "why"}:
            result[key] = " ".join(value.lstrip("- ") for value in values)
        else:
            result[key] = [
                value.lstrip("- ") for value in values
                if value.lstrip("- ").lower() not in {"none", "not specified"}
            ]
    return result


def _brief_section(title: str, value: Any) -> str:
    if isinstance(value, (list, tuple)):
        rendered = "\n".join(f"- {item}" for item in value) or "- None"
    else:
        rendered = str(value or "Not specified")
    return f"## {title}\n\n{rendered}"


def build_promotion_payload(
    *, intake_id: str, promotion_key: str, source: Mapping[str, Any],
    session_key: str, session_id: str, dm_channel_id: str, thread_ts: str,
    brief: Mapping[str, Any], raw_transcript: str | None = None,
) -> dict[str, Any]:
    del raw_transcript
    outcome = str(brief.get("outcome") or "").strip()
    if not outcome:
        raise ValueError("outcome is required before promotion")
    sections = [
        _brief_section("Outcome", outcome),
        _brief_section("Why", brief.get("why")),
        _brief_section("Scope", brief.get("scope")),
        _brief_section("Non-goals", brief.get("non_goals")),
        _brief_section("Acceptance criteria", brief.get("acceptance_criteria")),
        _brief_section("Decisions", brief.get("decisions")),
        _brief_section("Constraints", brief.get("constraints")),
        _brief_section("Unresolved questions", brief.get("unresolved_questions")),
        "## Lineage\n\n"
        f"- Slack source: {source.get('permalink') or 'unavailable'} "
        f"(`{source.get('team_id')}` / `{source.get('channel_id')}` / `{source.get('message_ts')}`)\n"
        f"- Intake ID: `{intake_id}`\n"
        f"- Hermes session: `{session_key}` / `{session_id}`\n"
        f"- Slack DM thread: `{dm_channel_id}` / `{thread_ts}`\n"
        f"- Promotion key: `{promotion_key}`",
    ]
    return {
        "title": _bounded_title(outcome, str(source.get("author_name") or "")),
        "body": "\n\n".join(sections),
        "triage": True,
        "idempotency_key": promotion_key,
    }


def _clean_label(value: Any, fallback: str) -> str:
    text = " ".join(str(value or "").split())
    return text or fallback


def _bounded_title(message_text: str, author_name: str) -> str:
    first_line = next(
        (" ".join(line.split()) for line in str(message_text).splitlines() if line.strip()),
        "",
    )
    title = first_line or f"Slack follow-up from {_clean_label(author_name, 'unknown author')}"
    if len(title) <= MAX_TITLE_LENGTH:
        return title
    return title[: MAX_TITLE_LENGTH - 1].rstrip() + "…"


def build_task_payload(source: Mapping[str, Any], context: str) -> dict[str, Any]:
    """Build the bounded Kanban request for one authorized Slack message."""
    team_id = str(source.get("team_id") or "").strip()
    channel_id = str(source.get("channel_id") or "").strip()
    message_ts = str(source.get("message_ts") or "").strip()
    message_text = str(source.get("message_text") or "")
    author_id = str(source.get("author_id") or "unknown")
    submitter_id = str(source.get("submitter_id") or "unknown")
    author_name = _clean_label(source.get("author_name"), author_id)
    submitter_name = _clean_label(source.get("submitter_name"), submitter_id)
    channel_name = _clean_label(source.get("channel_name"), "private channel")
    permalink = str(source.get("permalink") or "").strip()
    bounded_context = str(context or "")[:MAX_CONTEXT_LENGTH].strip()

    if not team_id or not channel_id or not message_ts:
        raise ValueError("Slack source identity is incomplete")

    body = "\n".join(
        [
            "## Context from the submitter",
            "",
            bounded_context or "No additional context supplied.",
            "",
            "## Source Slack message",
            "",
            message_text or "(No text content supplied by Slack.)",
            "",
            "## Source metadata",
            "",
            f"- Author: {author_name} (`{author_id}`)",
            f"- Submitted by: {submitter_name} (`{submitter_id}`)",
            f"- Channel: {channel_name} (`{channel_id}`)",
            f"- Message timestamp: `{message_ts}`",
            f"- Permalink: {permalink or 'unavailable to the integration'}",
            f"- Slack workspace: `{team_id}`",
        ]
    )
    return {
        "title": _bounded_title(message_text, author_name),
        "body": body,
        "triage": True,
        "idempotency_key": f"slack-message:{team_id}:{channel_id}:{message_ts}",
    }


class HermesIntakeError(RuntimeError):
    """A safe, user-displayable Hermes intake failure."""


class HermesIntakeClient:
    """Create HRMS Triage tasks with per-promotion serialization."""

    def __init__(
        self,
        *,
        base_url: str,
        session_token: str,
        transport: Transport | None = None,
    ) -> None:
        self._base_url = str(base_url).rstrip("/")
        self._session_token = str(session_token)
        self._transport = transport or self._post
        self._locks: dict[str, tuple[asyncio.Lock, int]] = {}
        self._locks_guard = asyncio.Lock()

    async def _source_lock(self, key: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock, users = self._locks.get(key, (asyncio.Lock(), 0))
            self._locks[key] = (lock, users + 1)
            return lock

    async def _release_source_lock(self, key: str, lock: asyncio.Lock) -> None:
        async with self._locks_guard:
            current, users = self._locks.get(key, (lock, 1))
            if current is not lock:
                return
            if users <= 1:
                self._locks.pop(key, None)
            else:
                self._locks[key] = (lock, users - 1)

    async def create_task(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        key = str(payload.get("idempotency_key") or "")
        if not key:
            raise ValueError("idempotency_key is required")
        lock = await self._source_lock(key)
        try:
            async with lock:
                response = await self._transport(
                    f"{self._base_url}/api/plugins/kanban/tasks?{urlencode({'board': BOARD})}",
                    {"X-Hermes-Session-Token": self._session_token},
                    dict(payload),
                )
        finally:
            await self._release_source_lock(key, lock)

        task = response.get("task") if isinstance(response, dict) else None
        if not isinstance(task, dict) or not task.get("id"):
            raise HermesIntakeError("Hermes returned an invalid task response.")
        return task

    async def _post(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        if response.status in {401, 403}:
                            raise HermesIntakeError(
                                "Hermes rejected the integration credentials."
                            )
                        if response.status == 404:
                            raise HermesIntakeError(
                                "The HRMS Kanban board is unavailable."
                            )
                        if response.status == 429 or response.status >= 500:
                            raise HermesIntakeError(
                                "Hermes is temporarily unavailable; please retry."
                            )
                        raise HermesIntakeError(
                            "Hermes rejected this task; review the source and retry."
                        )
                    data = await response.json(content_type=None)
        except HermesIntakeError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError):
            raise HermesIntakeError(
                "Hermes is temporarily unavailable; please retry."
            ) from None
        except (TypeError, ValueError):
            raise HermesIntakeError("Hermes returned an invalid task response.") from None
        return data
