"""Linear gateway platform adapter.

Treats Linear issues as Hermes sessions and Linear comments as turns.
This is intentionally a gateway adapter, not a cron scheduler: it reuses
Hermes' existing session, queue, steer, interrupt, and delivery semantics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from hermes_constants import get_hermes_home
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionSource, build_session_key
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)

logger = logging.getLogger(__name__)

LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"
DEFAULT_POLL_INTERVAL_SECONDS = 20
DEFAULT_START_STATES = ["Todo"]
DEFAULT_WAITING_STATES = ["Blocked"]
DEFAULT_WORKING_STATE = "Working"
DEFAULT_REVIEW_STATE = "Ready for Review"
DEFAULT_BLOCKED_STATE = "Blocked"
DEFAULT_TERMINAL_STATES = ["Done", "Canceled", "Duplicate"]
LINEAR_STATE_DIRECTIVE_RE = re.compile(r"<!--\s*linear-state\s*:\s*([a-zA-Z _-]+)\s*-->", re.IGNORECASE)
KANBAN_TERMINAL_EVENT_KINDS = {"completed", "blocked", "gave_up", "crashed", "timed_out"}


def check_linear_requirements() -> bool:
    return AIOHTTP_AVAILABLE


def _split_csv(value: Any, default: Optional[List[str]] = None) -> List[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


class LinearGraphQLClient:
    """Small async Linear GraphQL client used by the adapter."""

    def __init__(self, api_key: str, *, endpoint: str = LINEAR_GRAPHQL_URL) -> None:
        self.api_key = api_key
        self.endpoint = endpoint

    async def graphql(self, query: str, variables: Optional[dict] = None) -> dict:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for Linear gateway support")
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"query": query, "variables": variables or {}}
        async with aiohttp.ClientSession() as session:  # type: ignore[union-attr]
            async with session.post(self.endpoint, json=payload, headers=headers) as response:
                text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"Linear GraphQL HTTP {response.status}: {text[:500]}")
                data = json.loads(text)
        if data.get("errors"):
            raise RuntimeError(f"Linear GraphQL errors: {data['errors']}")
        return data.get("data") or {}

    async def comment_create(self, issue_id: str, body: str, *, parent_id: str = "") -> dict:
        variable_defs = ["$issueId: String!", "$body: String!"]
        input_fields = ["issueId: $issueId", "body: $body"]
        variables: dict[str, Any] = {"issueId": issue_id, "body": body}
        parent_id = str(parent_id or "").strip()
        if parent_id:
            variable_defs.append("$parentId: String")
            input_fields.append("parentId: $parentId")
            variables["parentId"] = parent_id
        data = await self.graphql(
            f"""
            mutation CreateHermesLinearComment({', '.join(variable_defs)}) {{
              commentCreate(input: {{{', '.join(input_fields)}}}) {{
                success
                comment {{ id }}
              }}
            }}
            """,
            variables,
        )
        payload = data.get("commentCreate") or {}
        if not payload.get("success"):
            raise RuntimeError("Linear commentCreate did not report success")
        return payload.get("comment") or {}

    async def fetch_workflow_states(
        self,
        *,
        team_key: str = "",
        team_id: str = "",
        first: int = 100,
    ) -> List[dict]:
        filters: list[str] = []
        variable_defs = ["$first: Int!"]
        variables: dict[str, Any] = {"first": int(first)}
        if team_id:
            filters.append("team: { id: { eq: $teamId } }")
            variable_defs.append("$teamId: ID")
            variables["teamId"] = team_id
        elif team_key:
            filters.append("team: { key: { eq: $teamKey } }")
            variable_defs.append("$teamKey: String")
            variables["teamKey"] = team_key
        filter_arg = f"filter: {{ {', '.join(filters)} }}," if filters else ""
        query = f"""
        query HermesLinearWorkflowStates({', '.join(variable_defs)}) {{
          workflowStates({filter_arg} first: $first) {{
            nodes {{ id name type team {{ id key name }} }}
          }}
        }}
        """
        data = await self.graphql(query, variables)
        return ((data.get("workflowStates") or {}).get("nodes") or [])

    async def issue_update_state(self, issue_id: str, state_id: str) -> dict:
        data = await self.graphql(
            """
            mutation UpdateHermesLinearIssueState($issueId: String!, $stateId: String) {
              issueUpdate(id: $issueId, input: {stateId: $stateId}) {
                success
                issue { id identifier state { name } }
              }
            }
            """,
            {"issueId": issue_id, "stateId": state_id},
        )
        payload = data.get("issueUpdate") or {}
        if not payload.get("success"):
            raise RuntimeError("Linear issueUpdate did not report success")
        return payload.get("issue") or {}

    async def set_issue_state_by_name(
        self,
        *,
        issue_id: str,
        state_name: str,
        team_key: str = "",
        team_id: str = "",
    ) -> dict:
        wanted = state_name.strip().lower()
        states = await self.fetch_workflow_states(team_key=team_key, team_id=team_id)
        match = next((state for state in states if str(state.get("name") or "").strip().lower() == wanted), None)
        if not match:
            scope = team_id or team_key or "workspace"
            raise RuntimeError(f"Linear workflow state {state_name!r} not found for {scope}")
        return await self.issue_update_state(issue_id, str(match.get("id") or ""))

    async def fetch_candidate_issues(
        self,
        *,
        team_key: str = "",
        project_id: str = "",
        project_name: str = "",
        state_names: Optional[List[str]] = None,
        first: int = 50,
    ) -> List[dict]:
        """Fetch issue summaries plus recent comments for polling.

        The filter is intentionally conservative. If a project id is configured,
        use it. Otherwise use team + optional state filter and let the adapter
        discard non-matching project names locally.
        """
        filters: list[str] = []
        variable_defs = ["$first: Int!"]
        variables: dict[str, Any] = {"first": int(first)}
        if team_key:
            filters.append("team: { key: { eq: $teamKey } }")
            variable_defs.append("$teamKey: String")
            variables["teamKey"] = team_key
        if project_id:
            filters.append("project: { id: { eq: $projectId } }")
            variable_defs.append("$projectId: ID")
            variables["projectId"] = project_id
        if state_names:
            filters.append("state: { name: { in: $stateNames } }")
            variable_defs.append("$stateNames: [String!]")
            variables["stateNames"] = state_names
        filter_arg = f"filter: {{ {', '.join(filters)} }}," if filters else ""
        query = f"""
        query HermesLinearIssues({', '.join(variable_defs)}) {{
          issues({filter_arg} first: $first, orderBy: updatedAt) {{
            nodes {{
              id
              identifier
              title
              description
              url
              updatedAt
              state {{ name }}
              team {{ id key name }}
              project {{ id name slugId }}
              creator {{ id name email }}
              labels {{ nodes {{ name }} }}
              comments(first: 20, orderBy: createdAt) {{
                nodes {{
                  id
                  body
                  createdAt
                  user {{ id name email }}
                }}
              }}
            }}
          }}
        }}
        """
        data = await self.graphql(
            query,
            variables,
        )
        issues = ((data.get("issues") or {}).get("nodes") or [])
        if project_name:
            wanted = project_name.strip().lower()
            issues = [
                issue for issue in issues
                if ((issue.get("project") or {}).get("name") or "").strip().lower() == wanted
            ]
        return issues


class LinearAdapter(BasePlatformAdapter):
    """Hermes platform adapter for Linear."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.LINEAR)
        extra = config.extra or {}
        self.api_key = (
            config.api_key
            or config.token
            or extra.get("api_key")
            or os.getenv("SYMPHONY_LINEAR_API_KEY")
            or os.getenv("HERMES_LINEAR_API_KEY")
            or ""
        )
        self.team_key = str(extra.get("team_key") or os.getenv("LINEAR_TEAM_KEY") or "").strip()
        self.project_id = str(extra.get("project_id") or os.getenv("LINEAR_PROJECT_ID") or "").strip()
        self.project_slug = str(extra.get("project_slug") or os.getenv("LINEAR_PROJECT_SLUG") or "").strip()
        self.project_name = str(extra.get("project_name") or os.getenv("LINEAR_PROJECT_NAME") or "").strip()
        self.start_states = _split_csv(
            extra.get("start_states")
            or extra.get("active_states")
            or os.getenv("LINEAR_START_STATES")
            or os.getenv("LINEAR_ACTIVE_STATES"),
            DEFAULT_START_STATES,
        )
        self.blocked_state = str(extra.get("blocked_state") or os.getenv("LINEAR_BLOCKED_STATE") or DEFAULT_BLOCKED_STATE).strip()
        self.waiting_states = _split_csv(
            extra.get("waiting_states")
            or os.getenv("LINEAR_WAITING_STATES"),
            [self.blocked_state] if self.blocked_state else DEFAULT_WAITING_STATES,
        )
        self.working_state = str(extra.get("working_state") or os.getenv("LINEAR_WORKING_STATE") or DEFAULT_WORKING_STATE).strip()
        self.review_state = str(extra.get("review_state") or os.getenv("LINEAR_REVIEW_STATE") or DEFAULT_REVIEW_STATE).strip()
        self.active_states = self.start_states  # Backwards-compatible alias for older tests/scripts.
        self.terminal_states = _split_csv(extra.get("terminal_states") or os.getenv("LINEAR_TERMINAL_STATES"), DEFAULT_TERMINAL_STATES)
        self.inbox_states = _split_csv(extra.get("inbox_states") or os.getenv("LINEAR_INBOX_STATES"))
        self.comment_wake_states = _split_csv(
            extra.get("comment_wake_states") or os.getenv("LINEAR_COMMENT_WAKE_STATES"),
            list(dict.fromkeys(self.start_states + self.waiting_states + self.inbox_states + [self.review_state, self.working_state])),
        )
        try:
            self.poll_interval_seconds = int(extra.get("poll_interval_seconds") or os.getenv("LINEAR_POLL_INTERVAL_SECONDS") or DEFAULT_POLL_INTERVAL_SECONDS)
        except (TypeError, ValueError):
            self.poll_interval_seconds = DEFAULT_POLL_INTERVAL_SECONDS
        self.allowed_user_emails = set(_split_csv(extra.get("allowed_user_emails") or os.getenv("LINEAR_ALLOWED_USER_EMAILS")))
        self.bot_user_emails = set(_split_csv(extra.get("bot_user_emails") or os.getenv("LINEAR_BOT_USER_EMAILS")))
        self._client: Any = LinearGraphQLClient(self.api_key) if self.api_key else None
        self._poll_task: Optional[asyncio.Task] = None
        self._seen_issue_ids: set[str] = set()
        self._seen_comment_ids: set[str] = set()
        self._issue_context_by_id: dict[str, dict[str, str]] = {}
        self._explicit_state_by_issue_id: dict[str, str] = {}
        self.execution_mode = str(extra.get("execution_mode") or os.getenv("LINEAR_EXECUTION_MODE") or "direct").strip().lower()
        self.kanban_board = str(extra.get("kanban_board") or os.getenv("LINEAR_KANBAN_BOARD") or "").strip()
        self.kanban_default_assignee = str(
            extra.get("kanban_default_assignee")
            or extra.get("default_assignee")
            or os.getenv("LINEAR_KANBAN_DEFAULT_ASSIGNEE")
            or os.getenv("LINEAR_DEFAULT_ASSIGNEE")
            or "default"
        ).strip()
        self.kanban_workspace_kind = str(extra.get("kanban_workspace_kind") or os.getenv("LINEAR_KANBAN_WORKSPACE_KIND") or "scratch").strip()
        self.kanban_workspace_path = str(extra.get("kanban_workspace_path") or os.getenv("LINEAR_KANBAN_WORKSPACE_PATH") or "").strip() or None
        self.kanban_skills = _split_csv(extra.get("kanban_skills") or os.getenv("LINEAR_KANBAN_SKILLS"))
        self.kanban_terminal_comment_kinds = {
            kind.strip().lower()
            for kind in _split_csv(
                extra.get("kanban_terminal_comment_kinds")
                or os.getenv("LINEAR_KANBAN_TERMINAL_COMMENT_KINDS"),
                ["blocked", "gave_up", "crashed", "timed_out"],
            )
            if kind.strip().lower()
        }
        self.status_issue_id = str(extra.get("status_issue_id") or os.getenv("LINEAR_STATUS_ISSUE_ID") or "").strip()
        self.status_issue_identifier = str(extra.get("status_issue_identifier") or os.getenv("LINEAR_STATUS_ISSUE_IDENTIFIER") or "").strip()
        self._kanban_task_by_issue_id: dict[str, str] = {}
        self._state_path = get_hermes_home() / "linear" / "state.json"

    async def connect(self) -> bool:
        if not self.api_key:
            self._set_fatal_error("missing_token", "Linear token is not configured", retryable=False)
            return False
        if not check_linear_requirements():
            self._set_fatal_error("missing_dependency", "aiohttp is required for Linear gateway support", retryable=False)
            return False
        self._load_state()
        self._mark_connected()
        try:
            await self._reconcile_orphaned_working_issues()
        except Exception as exc:
            logger.warning("[linear] orphaned Working issue reconciliation failed: %s", exc)
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("[linear] connected: team=%s project=%s", self.team_key or "*", self.project_id or self.project_name or self.project_slug or "*")
        return True

    async def disconnect(self) -> None:
        task = self._poll_task
        self._poll_task = None
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._save_state()
        self._mark_disconnected()
        logger.info("[linear] disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        metadata = metadata or {}
        issue_id = str(metadata.get("issue_id") or chat_id)
        clean_content, target_state = self._strip_state_directive(content)
        clean_content = clean_content.strip() or content.strip()
        if not target_state:
            target_state = self._state_for_kanban_event(metadata)
        context = self._issue_context_by_id.get(issue_id, {})
        team_key = str(metadata.get("team_key") or context.get("team_key") or "").strip()
        team_id = str(metadata.get("team_id") or context.get("team_id") or "").strip()
        kanban_event_kind = str(
            metadata.get("kanban_event_kind") or metadata.get("event_kind") or ""
        ).strip().lower()
        if self._is_gateway_system_notice(clean_content, metadata):
            logger.debug("[linear] suppressed gateway system notice for %s", issue_id)
            return SendResult(
                success=True,
                raw_response={"suppressed": True, "reason": "gateway_system_notice"},
            )
        if (
            kanban_event_kind in KANBAN_TERMINAL_EVENT_KINDS
            and kanban_event_kind not in self.kanban_terminal_comment_kinds
        ):
            await self._apply_send_state_transition(
                issue_id=issue_id,
                target_state=target_state,
                metadata=metadata,
                team_key=team_key,
                team_id=team_id,
            )
            logger.debug(
                "[linear] suppressed kanban %s terminal comment for %s",
                kanban_event_kind,
                issue_id,
            )
            return SendResult(
                success=True,
                raw_response={
                    "suppressed": True,
                    "reason": "kanban_terminal_comment",
                    "kanban_event_kind": kanban_event_kind,
                    "target_state": target_state,
                },
            )
        try:
            parent_comment_id = self._parent_comment_id_for_reply(reply_to, metadata)
            if parent_comment_id:
                comment = await self._client.comment_create(
                    issue_id,
                    clean_content,
                    parent_id=parent_comment_id,
                )
            else:
                comment = await self._client.comment_create(issue_id, clean_content)
            comment_id = comment.get("id")
            if comment_id:
                self._seen_comment_ids.add(str(comment_id))
                self._save_state()
            await self._apply_send_state_transition(
                issue_id=issue_id,
                target_state=target_state,
                metadata=metadata,
                team_key=team_key,
                team_id=team_id,
            )
            return SendResult(success=True, message_id=comment_id, raw_response=comment)
        except Exception as exc:
            logger.warning("[linear] commentCreate failed for %s: %s", issue_id, exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def _apply_send_state_transition(
        self,
        *,
        issue_id: str,
        target_state: str,
        metadata: Dict[str, Any],
        team_key: str,
        team_id: str,
    ) -> None:
        if not target_state or metadata.get("linear_skip_state_transition"):
            return
        self._explicit_state_by_issue_id[issue_id] = target_state
        try:
            await self._client.set_issue_state_by_name(
                issue_id=issue_id,
                state_name=target_state,
                team_key=team_key,
                team_id=team_id,
            )
        except Exception as state_exc:
            logger.warning("[linear] state transition to %s failed for %s: %s", target_state, issue_id, state_exc)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": str(chat_id), "type": "thread"}

    async def on_processing_start(self, event: MessageEvent) -> None:
        issue = ((event.raw_message or {}).get("issue") or {}) if isinstance(event.raw_message, dict) else {}
        if not issue or not self._client or self.execution_mode == "kanban" or self._is_status_issue(issue):
            return
        self._remember_issue_context(issue)
        state_name = ((issue.get("state") or {}).get("name") or "").strip()
        if state_name not in set(self.start_states + self.waiting_states + [self.review_state]):
            return
        await self._move_issue_to_state(issue, self.working_state)

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        issue = ((event.raw_message or {}).get("issue") or {}) if isinstance(event.raw_message, dict) else {}
        if not issue or not self._client or self.execution_mode == "kanban" or self._is_status_issue(issue):
            return
        issue_id = str(issue.get("id") or "").strip()
        explicit_state = self._explicit_state_by_issue_id.pop(issue_id, "") if issue_id else ""
        if explicit_state:
            return
        if outcome is ProcessingOutcome.SUCCESS:
            await self._move_issue_to_state(issue, self.review_state)
        elif outcome in (ProcessingOutcome.FAILURE, ProcessingOutcome.CANCELLED):
            await self._move_issue_to_state(issue, self.blocked_state)

    async def _reconcile_orphaned_working_issues(self) -> None:
        """Move interrupted prior-run Linear work out of Working.

        Linear comments are marked seen before the async agent turn finishes.
        If the gateway then restarts or the task is cancelled, the comment will
        not be polled again.  SessionStore marks those interrupted sessions as
        ``resume_pending`` or ``suspended``; on adapter startup/reconnect, use
        that durable flag to stop the issue from looking actively in progress.
        """
        if not self._client or not self.working_state or not self.blocked_state:
            return
        issues = await self._client.fetch_candidate_issues(
            team_key=self.team_key,
            project_id=self.project_id,
            project_name=self.project_name,
            state_names=[self.working_state],
        )
        for issue in issues:
            if self.project_slug and not self._issue_matches_project_slug(issue, self.project_slug):
                continue
            state_name = ((issue.get("state") or {}).get("name") or "").strip()
            if state_name != self.working_state:
                continue
            self._remember_issue_context(issue)
            if not self._issue_has_interrupted_session(issue):
                continue
            identifier = issue.get("identifier") or issue.get("id") or "unknown"
            logger.warning(
                "[linear] moving orphaned Working issue %s to %s after interrupted gateway run",
                identifier,
                self.blocked_state,
            )
            await self._move_issue_to_state(issue, self.blocked_state)

    def _issue_has_interrupted_session(self, issue: dict) -> bool:
        session_key = self._session_key_for_issue(issue)
        if not session_key:
            return False
        if session_key in self._active_sessions and not self._session_task_is_stale(session_key):
            return False
        store = getattr(self, "_session_store", None)
        if store is None:
            return False
        try:
            ensure_loaded = getattr(store, "_ensure_loaded", None)
            if callable(ensure_loaded):
                ensure_loaded()
        except Exception:
            pass
        entries = getattr(store, "_entries", {}) or {}
        entry = entries.get(session_key) if hasattr(entries, "get") else None
        if entry is None:
            return False
        suspended = bool(getattr(entry, "suspended", False))
        resume_pending = bool(getattr(entry, "resume_pending", False))
        if isinstance(entry, dict):
            suspended = suspended or bool(entry.get("suspended"))
            resume_pending = resume_pending or bool(entry.get("resume_pending"))
        return suspended or resume_pending

    def _session_key_for_issue(self, issue: dict) -> str:
        source = self._source_for_issue(issue, user=issue.get("creator") or {})
        return build_session_key(
            source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )

    async def _move_issue_to_state(self, issue: dict, state_name: str) -> None:
        issue_id = str(issue.get("id") or "").strip()
        if not issue_id or not state_name or not self._client:
            return
        team = issue.get("team") or {}
        team_key = str(team.get("key") or "").strip()
        team_id = str(team.get("id") or "").strip()
        try:
            await self._client.set_issue_state_by_name(
                issue_id=issue_id,
                state_name=state_name,
                team_key=team_key,
                team_id=team_id,
            )
        except Exception as exc:
            logger.warning("[linear] state transition to %s failed for %s: %s", state_name, issue_id, exc)

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[linear] poll failed: %s", exc)
            await asyncio.sleep(max(3, self.poll_interval_seconds))

    async def _poll_once(self) -> None:
        if not self._client:
            return
        poll_states = list(dict.fromkeys(self.comment_wake_states))
        issues = await self._client.fetch_candidate_issues(
            team_key=self.team_key,
            project_id=self.project_id,
            project_name=self.project_name,
            state_names=poll_states,
        )
        for issue in issues:
            if self.project_slug and not self._issue_matches_project_slug(issue, self.project_slug):
                continue
            self._remember_issue_context(issue)
            state_name = ((issue.get("state") or {}).get("name") or "").strip()
            if state_name in self.terminal_states or state_name not in poll_states:
                continue
            issue_id = issue.get("id")
            is_status_issue = self._is_status_issue(issue)
            is_inbox_issue = self._is_inbox_issue(issue)
            if issue_id and issue_id not in self._seen_issue_ids and state_name in self.start_states + self.inbox_states:
                self._seen_issue_ids.add(issue_id)
                if is_status_issue or is_inbox_issue or self.execution_mode != "kanban":
                    await self.handle_message(self._event_from_issue(issue))
                else:
                    self._ensure_kanban_task_for_issue(issue)
                    await self._move_issue_to_state(issue, self.working_state)
            for comment in ((issue.get("comments") or {}).get("nodes") or []):
                comment_id = comment.get("id")
                if not comment_id or comment_id in self._seen_comment_ids:
                    continue
                if not self._comment_allowed(comment):
                    self._seen_comment_ids.add(comment_id)
                    continue
                self._seen_comment_ids.add(comment_id)
                if is_status_issue or is_inbox_issue or self.execution_mode != "kanban":
                    await self.handle_message(self._event_from_comment(issue, comment))
                else:
                    await self._handle_kanban_comment(issue, comment)
        self._save_state()

    def _event_from_comment(self, issue: dict, comment: dict) -> MessageEvent:
        user = comment.get("user") or {}
        author = str(user.get("name") or user.get("email") or "Linear user")
        body = str(comment.get("body") or "").strip()
        contract = self._contract_lines_for_issue(issue)
        parts = self._issue_context_lines(issue) + contract + ["", f"Comment from {author}:", body]
        return MessageEvent(
            text="\n".join(part for part in parts if part is not None).strip(),
            message_type=MessageType.TEXT,
            source=self._source_for_issue(issue, user=user, message_id=comment.get("id")),
            raw_message={"issue": issue, "comment": comment},
            message_id=str(comment.get("id") or ""),
        )

    def _event_from_issue(self, issue: dict) -> MessageEvent:
        creator = issue.get("creator") or {}
        description = issue.get("description") or ""
        parts = self._issue_context_lines(issue) + self._contract_lines_for_issue(issue) + ["", description]
        text = "\n".join(part for part in parts if part is not None).strip()
        return MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=self._source_for_issue(issue, user=creator, message_id=f"issue:{issue.get('id')}"),
            raw_message={"issue": issue},
            message_id=f"issue:{issue.get('id')}",
        )

    def _issue_context_lines(self, issue: dict) -> List[str]:
        identifier = issue.get("identifier") or issue.get("id") or "Linear issue"
        title = issue.get("title") or "Untitled"
        state_name = ((issue.get("state") or {}).get("name") or "").strip()
        team = issue.get("team") or {}
        team_key = str(team.get("key") or "").strip()
        team_name = str(team.get("name") or "").strip()
        project = issue.get("project") or {}
        project_name = str(project.get("name") or "").strip()
        project_slug = str(project.get("slugId") or "").strip()
        labels = self._label_names(issue)
        project_text = ""
        if project_name and project_slug:
            project_text = f"Project: {project_name} ({project_slug})"
        elif project_name:
            project_text = f"Project: {project_name}"
        elif project_slug:
            project_text = f"Project: {project_slug}"
        return [
            f"Linear issue {identifier}: {title}",
            f"State: {state_name}" if state_name else "",
            f"Team: {team_key or team_name}" if team_key or team_name else "",
            project_text,
            f"URL: {issue.get('url')}" if issue.get("url") else "",
            f"Labels: {', '.join(labels)}" if labels else "",
        ]

    def _source_for_issue(self, issue: dict, *, user: Optional[dict] = None, message_id: Optional[str] = None) -> SessionSource:
        user = user or {}
        issue_id = str(issue.get("id") or issue.get("identifier") or "")
        identifier = str(issue.get("identifier") or issue_id)
        title = str(issue.get("title") or "")
        chat_name = f"{identifier}: {title}" if title else identifier
        return SessionSource(
            platform=Platform.LINEAR,
            chat_id=issue_id,
            chat_name=chat_name,
            chat_type="thread",
            user_id=str(user.get("id") or user.get("email") or "linear-user"),
            user_name=str(user.get("name") or user.get("email") or "Linear user"),
            thread_id=identifier,
            message_id=message_id,
        )

    def _comment_allowed(self, comment: dict) -> bool:
        user = comment.get("user") or {}
        email = str(user.get("email") or "").strip().lower()
        if self.bot_user_emails and email in {e.lower() for e in self.bot_user_emails}:
            return False
        if self.allowed_user_emails and email not in {e.lower() for e in self.allowed_user_emails}:
            return False
        body = str(comment.get("body") or "")
        if body.startswith("## Hermes Workpad"):
            return False
        return bool(body.strip())

    @staticmethod
    def _label_names(issue: dict) -> List[str]:
        return [
            str(label.get("name"))
            for label in ((issue.get("labels") or {}).get("nodes") or [])
            if label.get("name")
        ]

    @staticmethod
    def _issue_matches_project_slug(issue: dict, slug: str) -> bool:
        project = issue.get("project") or {}
        wanted = slug.strip().lower()
        return wanted in {
            str(project.get("slugId") or "").strip().lower(),
            str(project.get("name") or "").strip().lower(),
        }

    @staticmethod
    def _parent_comment_id_for_reply(reply_to: Optional[str], metadata: Dict[str, Any]) -> str:
        """Return a Linear comment id to nest under, excluding synthetic issue ids."""
        candidate = str(
            metadata.get("parent_comment_id")
            or metadata.get("linear_parent_comment_id")
            or reply_to
            or ""
        ).strip()
        if not candidate or candidate.startswith("issue:"):
            return ""
        return candidate

    def _remember_issue_context(self, issue: dict) -> None:
        issue_id = str(issue.get("id") or "").strip()
        if not issue_id:
            return
        team = issue.get("team") or {}
        self._issue_context_by_id[issue_id] = {
            "team_key": str(team.get("key") or "").strip(),
            "team_id": str(team.get("id") or "").strip(),
            "identifier": str(issue.get("identifier") or "").strip(),
        }

    @staticmethod
    def _is_gateway_system_notice(content: str, metadata: Dict[str, Any]) -> bool:
        """Return True for gateway-generated status/noise, not agent replies.

        Linear issue threads are durable work records. Agent-authored interim
        comments may be useful, but gateway housekeeping ("still working",
        queue acks, restart/drain notices, memory-review pings) should not be
        persisted as issue comments.
        """
        if bool(metadata.get("hermes_system_notice") or metadata.get("gateway_system_notice")):
            return True
        text = (content or "").strip()
        system_prefixes = (
            "⏳ Still working...",
            "⏳ Queued for the next turn",
            "⏳ Gateway ",
            "⏳ Gateway is ",
            "⚠️ Gateway ",
            "⚠️ No activity for ",
            "Session automatically reset",
            "💾 Memory updated",
        )
        return any(text.startswith(prefix) for prefix in system_prefixes)

    def _strip_state_directive(self, content: str) -> tuple[str, str]:
        match = LINEAR_STATE_DIRECTIVE_RE.search(content or "")
        target = ""
        if match:
            requested = match.group(1).strip().lower().replace("_", "-")
            if requested in {"blocked", "block", "waiting", "waiting-for-user", "waiting-for-anton"}:
                target = self.blocked_state
            elif requested in {"ready", "review", "ready-for-review", "human-review"}:
                target = self.review_state
        clean = LINEAR_STATE_DIRECTIVE_RE.sub("", content or "").strip()
        return clean, target

    def _state_for_kanban_event(self, metadata: Dict[str, Any]) -> str:
        kind = str(metadata.get("kanban_event_kind") or metadata.get("event_kind") or "").strip().lower()
        if kind == "completed":
            return self.review_state
        if kind in {"blocked", "gave_up", "crashed", "timed_out"}:
            return self.blocked_state
        return ""

    def _contract_lines_for_issue(self, issue: dict) -> List[str]:
        if self._is_status_issue(issue):
            return self._status_chat_contract_lines()
        if self._is_inbox_issue(issue):
            return self._inbox_chat_contract_lines()
        return self._workflow_contract_lines()

    def _status_chat_contract_lines(self) -> List[str]:
        return [
            "",
            "Persistent Linear status/chat issue:",
            "- This issue is a durable chat/control surface for Hermes, not an implementation task.",
            "- Do not create Kanban tasks from this issue unless Anton explicitly asks.",
            "- For status questions, inspect Hermes Kanban/Linear state and answer concisely.",
            "- Do not mutate Linear workflow state just because this chat received a comment.",
        ]

    def _inbox_chat_contract_lines(self) -> List[str]:
        states = ", ".join(self.inbox_states) or "Inbox"
        return [
            "",
            "Linear inbox/chat issue:",
            f"- Issues in {states} are one-off chat sessions, not implementation tasks.",
            "- Answer directly in the Linear thread unless Anton explicitly asks you to create work.",
            "- Do not create or touch Kanban tasks for this issue by default.",
            "- Do not mutate Linear workflow state just because this chat received a comment.",
        ]

    def _is_status_issue(self, issue: dict) -> bool:
        issue_id = str(issue.get("id") or "").strip()
        identifier = str(issue.get("identifier") or "").strip()
        if self.status_issue_id and issue_id == self.status_issue_id:
            return True
        if self.status_issue_identifier and identifier.lower() == self.status_issue_identifier.lower():
            return True
        return False

    def _is_inbox_issue(self, issue: dict) -> bool:
        state_name = ((issue.get("state") or {}).get("name") or "").strip()
        if not state_name:
            return False
        return state_name.lower() in {state.lower() for state in self.inbox_states}

    def _kanban_conn(self):
        from hermes_cli import kanban_db as kb

        board = self.kanban_board or None
        kb.init_db(board=board)
        return kb.connect(board=board)

    def _ensure_kanban_task_for_issue(
        self,
        issue: dict,
        *,
        followup_comment: Optional[dict] = None,
        parent_task_id: str = "",
    ) -> str:
        from hermes_cli import kanban_db as kb

        issue_id = str(issue.get("id") or "").strip()
        if not issue_id:
            raise ValueError("Linear issue id is required for Kanban task creation")
        identifier = str(issue.get("identifier") or issue_id).strip()
        title = str(issue.get("title") or "Untitled").strip()
        comment_id = str((followup_comment or {}).get("id") or "").strip()
        if followup_comment:
            snippet = self._comment_snippet(followup_comment)
            task_title = f"{identifier} follow-up: {snippet}"
            idempotency_key = f"linear:{issue_id}:comment:{comment_id or snippet}"
            parents = [parent_task_id] if parent_task_id else []
        else:
            task_title = f"{identifier}: {title}"
            idempotency_key = f"linear:{issue_id}"
            parents = []
        body = self._kanban_task_body(issue, followup_comment=followup_comment, parent_task_id=parent_task_id)
        conn = self._kanban_conn()
        try:
            task_id = kb.create_task(
                conn,
                title=task_title,
                body=body,
                assignee=self.kanban_default_assignee or None,
                created_by="linear",
                workspace_kind=self.kanban_workspace_kind or "scratch",
                workspace_path=self.kanban_workspace_path,
                parents=parents,
                idempotency_key=idempotency_key,
                skills=self.kanban_skills or None,
            )
            kb.add_notify_sub(
                conn,
                task_id=task_id,
                platform=Platform.LINEAR.value,
                chat_id=issue_id,
                thread_id=identifier,
            )
        finally:
            conn.close()
        self._kanban_task_by_issue_id[issue_id] = task_id
        self._save_state()
        return task_id

    async def _handle_kanban_comment(self, issue: dict, comment: dict) -> str:
        from hermes_cli import kanban_db as kb

        issue_id = str(issue.get("id") or "").strip()
        if not issue_id:
            return ""
        conn = self._kanban_conn()
        try:
            task_id = self._kanban_task_by_issue_id.get(issue_id, "")
            task = kb.get_task(conn, task_id) if task_id else None
            if task is None:
                task_id = self._ensure_kanban_task_for_issue(issue)
                task = kb.get_task(conn, task_id)
            if task is not None and task.status in {"done", "archived"}:
                followup_id = self._ensure_kanban_task_for_issue(
                    issue,
                    followup_comment=comment,
                    parent_task_id=task.id,
                )
                await self._move_issue_to_state(issue, self.working_state)
                return followup_id
            author = self._comment_author(comment)
            body = self._kanban_comment_body(comment)
            kb.add_comment(conn, task_id, author, body)
            if task is not None and task.status == "blocked":
                kb.unblock_task(conn, task_id)
                await self._move_issue_to_state(issue, self.working_state)
            return task_id
        finally:
            conn.close()

    def _kanban_task_body(
        self,
        issue: dict,
        *,
        followup_comment: Optional[dict] = None,
        parent_task_id: str = "",
    ) -> str:
        lines = self._issue_context_lines(issue)
        description = str(issue.get("description") or "").strip()
        if followup_comment:
            lines += [
                "",
                "Linear follow-up comment:",
                self._kanban_comment_body(followup_comment),
            ]
            if parent_task_id:
                lines += ["", f"Parent Kanban task: {parent_task_id}"]
        elif description:
            lines += ["", "Issue description:", description]
        lines += [
            "",
            "Execution contract:",
            "- Treat Linear as the human-facing control plane; do the work through this Kanban task.",
            "- Back-and-forth Linear comments are appended to the Kanban thread. If blocked, they unblock the task.",
            "- Complete with a concise human-facing summary. Block only when Anton input is required.",
            "- Ignore Linear assignee as a routing signal; Anton may assign issues to himself for iOS convenience.",
        ]
        return "\n".join(line for line in lines if line is not None).strip()

    def _kanban_comment_body(self, comment: dict) -> str:
        body = str(comment.get("body") or "").strip()
        comment_id = str(comment.get("id") or "").strip()
        prefix = f"Linear comment: {comment_id}" if comment_id else "Linear comment"
        return f"{prefix}\n\n{body}".strip()

    def _comment_author(self, comment: dict) -> str:
        user = comment.get("user") or {}
        return str(user.get("name") or user.get("email") or "Linear user").strip()

    def _comment_snippet(self, comment: dict) -> str:
        body = " ".join(str(comment.get("body") or "").strip().split())
        if not body:
            return str(comment.get("id") or "comment")[:80]
        return body[:80]

    def _workflow_contract_lines(self) -> List[str]:
        return [
            "",
            "Linear workflow contract:",
            f"- Backlog is inert. Hermes starts only from: {', '.join(self.start_states)}.",
            f"- When processing begins, the gateway moves this issue to: {self.working_state}.",
            f"- If your final reply is ready for Anton review, omit a state marker; the gateway moves it to: {self.review_state}.",
            f"- Anton comments in {self.review_state}, {self.blocked_state}, or {self.working_state} are treated as new turns; {self.review_state} and {self.blocked_state} move back to {self.working_state} when processing begins.",
            f"- If you need Anton's response to proceed, include this hidden marker at the end of your final reply: <!-- linear-state: blocked -->. The gateway will remove it and move the issue to: {self.blocked_state}.",
            "- You may recommend labels/projects when useful, but do not invent project membership as a hard constraint.",
        ]

    def _load_state(self) -> None:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            self._seen_issue_ids = set(data.get("seen_issue_ids") or [])
            self._seen_comment_ids = set(data.get("seen_comment_ids") or [])
            self._kanban_task_by_issue_id = {
                str(k): str(v)
                for k, v in (data.get("kanban_task_by_issue_id") or {}).items()
                if k and v
            }
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.debug("[linear] could not load state: %s", exc)

    def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "seen_issue_ids": sorted(self._seen_issue_ids)[-1000:],
                "seen_comment_ids": sorted(self._seen_comment_ids)[-3000:],
                "kanban_task_by_issue_id": self._kanban_task_by_issue_id,
            }
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._state_path)
        except Exception as exc:
            logger.debug("[linear] could not save state: %s", exc)
