"""Opt-in living task-status publication for Telegram-backed Kanban work.

The module is deliberately policy-neutral: callers supply card wording and
recommended actions, while the primitive enforces exact task/message routing,
monotonic versions, publication dedup, and callback correlation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence

from hermes_cli import kanban_db as kb


TELEGRAM_CALLBACK_LIMIT_BYTES = 64
_ACTION_RE = re.compile(r"^[a-z][a-z0-9_]{0,15}$")
_TASK_ID_RE = re.compile(r"^t_[A-Za-z0-9_-]+$")


class PublicationKind(str, Enum):
    ROUTINE = "routine"
    DECISION = "decision"
    BLOCKER = "blocker"
    MATERIAL_CHANGE = "material_change"
    ACCEPTED_COMPLETION = "accepted_completion"


class DeliveryRoute(str, Enum):
    CONTROL = "control"
    BRIEFINGS = "briefings"
    PROJECT = "project"
    SYSTEM = "system"


PUSH_KINDS = frozenset(
    {
        PublicationKind.DECISION,
        PublicationKind.BLOCKER,
        PublicationKind.MATERIAL_CHANGE,
        PublicationKind.ACCEPTED_COMPLETION,
    }
)


@dataclass(frozen=True)
class TaskStatusDestination:
    chat_id: str
    message_thread_id: str = ""

    def __post_init__(self) -> None:
        if not str(self.chat_id).strip():
            raise ValueError("task-status destination chat_id is required")
        object.__setattr__(self, "chat_id", str(self.chat_id))
        object.__setattr__(
            self, "message_thread_id", str(self.message_thread_id or "")
        )


@dataclass(frozen=True)
class TaskStatusButton:
    label: str
    action: str


@dataclass(frozen=True)
class TaskStatusPublication:
    operation: str
    kanban_task_id: str
    session_id: str
    lifecycle_state: str
    state_version: int
    content: str
    kind: PublicationKind = PublicationKind.ROUTINE
    destination: Optional[TaskStatusDestination] = None
    linear_issue_key: Optional[str] = None
    recommended_action: str = ""
    buttons: Sequence[TaskStatusButton] = field(default_factory=tuple)
    push_content: Optional[str] = None

    @classmethod
    def from_event_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "TaskStatusPublication":
        """Parse a policy-produced Kanban event without inferring any binding."""
        raw_buttons = payload.get("buttons") or ()
        if not isinstance(raw_buttons, Sequence) or isinstance(
            raw_buttons, (str, bytes)
        ):
            raise ValueError("task-status buttons must be a list")
        buttons: list[TaskStatusButton] = []
        for raw in raw_buttons:
            if not isinstance(raw, Mapping):
                raise ValueError("task-status button entries must be objects")
            buttons.append(
                TaskStatusButton(
                    label=str(raw.get("label") or ""),
                    action=str(raw.get("action") or ""),
                )
            )
        try:
            kind = PublicationKind(str(payload.get("kind") or "routine"))
        except ValueError as exc:
            raise ValueError("unknown task-status publication kind") from exc
        operation = str(payload.get("operation") or "")
        raw_destination = payload.get("destination")
        if not isinstance(raw_destination, Mapping) or (
            "chat_id" not in raw_destination
            or "message_thread_id" not in raw_destination
        ):
            raise ValueError(
                "task-status event requires an exact destination with chat_id "
                "and message_thread_id"
            )
        destination = TaskStatusDestination(
            chat_id=str(raw_destination.get("chat_id") or ""),
            message_thread_id=str(
                raw_destination.get("message_thread_id") or ""
            ),
        )
        return cls(
            operation=operation,
            kanban_task_id=str(payload.get("kanban_task_id") or ""),
            session_id=str(payload.get("session_id") or ""),
            linear_issue_key=(
                str(payload["linear_issue_key"])
                if payload.get("linear_issue_key") is not None
                else None
            ),
            lifecycle_state=str(payload.get("lifecycle_state") or ""),
            state_version=int(payload.get("state_version") or 0),
            content=str(payload.get("content") or ""),
            kind=kind,
            destination=destination,
            recommended_action=str(payload.get("recommended_action") or ""),
            buttons=tuple(buttons),
            push_content=(
                str(payload["push_content"])
                if payload.get("push_content") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PublicationResult:
    ok: bool
    message_id: Optional[str] = None
    deduplicated: bool = False
    pushed: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class CallbackResolution:
    ok: bool
    kanban_task_id: Optional[str] = None
    state_version: Optional[int] = None
    action: Optional[str] = None
    error: Optional[str] = None


def make_task_status_callback_data(
    kanban_task_id: str, state_version: int, action: str
) -> str:
    task_id = str(kanban_task_id)
    normalized_action = str(action).strip().lower()
    if not _TASK_ID_RE.fullmatch(task_id):
        raise ValueError("invalid canonical Kanban task id for callback")
    if not _ACTION_RE.fullmatch(normalized_action):
        raise ValueError("invalid task-status callback action")
    if int(state_version) < 1:
        raise ValueError("task-status callback version must be at least 1")
    data = f"ts:{normalized_action}:{task_id}:{int(state_version)}"
    if len(data.encode("utf-8")) > TELEGRAM_CALLBACK_LIMIT_BYTES:
        raise ValueError("task-status callback_data exceeds Telegram's 64-byte limit")
    return data


def parse_task_status_callback_data(data: str) -> tuple[str, int, str]:
    raw = str(data or "")
    if len(raw.encode("utf-8")) > TELEGRAM_CALLBACK_LIMIT_BYTES:
        raise ValueError("task-status callback_data exceeds Telegram's 64-byte limit")
    parts = raw.split(":")
    if len(parts) != 4 or parts[0] != "ts":
        raise ValueError("invalid task-status callback data")
    action, task_id, version_raw = parts[1], parts[2], parts[3]
    if not _ACTION_RE.fullmatch(action):
        raise ValueError("unknown or invalid task-status callback action")
    if not _TASK_ID_RE.fullmatch(task_id):
        raise ValueError("invalid canonical Kanban task id in callback")
    try:
        version = int(version_raw)
    except ValueError as exc:
        raise ValueError("invalid task-status callback version") from exc
    if version < 1 or str(version) != version_raw:
        raise ValueError("invalid task-status callback version")
    return task_id, version, action


def resolve_task_status_callback(
    conn,
    *,
    callback_data: str,
    chat_id: str,
    message_thread_id: str,
    message_id: str,
    allowed_actions: Sequence[str],
) -> CallbackResolution:
    """Validate and at-most-once claim a consequential task callback."""
    try:
        task_id, version, action = parse_task_status_callback_data(callback_data)
    except ValueError as exc:
        return CallbackResolution(ok=False, error=str(exc))

    allowed = {str(value).strip().lower() for value in allowed_actions}
    if action not in allowed:
        return CallbackResolution(
            ok=False, error=f"unknown task-status action {action!r}"
        )

    binding = kb.get_task_status_binding(conn, task_id)
    if binding is None:
        return CallbackResolution(
            ok=False, error=f"no task-status binding for Kanban task {task_id}"
        )
    if binding.state_version != version:
        return CallbackResolution(
            ok=False,
            error=(
                f"stale task-status callback for {task_id}: "
                f"version {version}, current {binding.state_version}"
            ),
        )
    living_message_matches = (
        binding.message_id == str(message_id)
        and binding.chat_id == str(chat_id)
        and binding.message_thread_id == str(message_thread_id or "")
    )
    callback_message_matches = kb.matches_task_status_callback_message(
        conn,
        kanban_task_id=task_id,
        state_version=version,
        platform="telegram",
        chat_id=str(chat_id),
        message_thread_id=str(message_thread_id or ""),
        message_id=str(message_id),
    )
    if not living_message_matches and not callback_message_matches:
        return CallbackResolution(
            ok=False,
            error=(
                "callback message/chat/topic does not match an exact "
                "task-status callback destination"
            ),
        )
    if not kb.claim_task_status_callback(
        conn,
        kanban_task_id=task_id,
        state_version=version,
        action=action,
    ):
        return CallbackResolution(
            ok=False, error="this task-status decision was already resolved"
        )
    return CallbackResolution(
        ok=True,
        kanban_task_id=task_id,
        state_version=version,
        action=action,
    )


def resolve_task_status_callback_across_boards(
    *,
    callback_data: str,
    chat_id: str,
    message_thread_id: str,
    message_id: str,
    allowed_actions: Sequence[str],
    board: Optional[str] = None,
) -> CallbackResolution:
    """Resolve an exact callback on one board, failing on cross-board ambiguity."""
    try:
        task_id, _version, _action = parse_task_status_callback_data(callback_data)
    except ValueError as exc:
        return CallbackResolution(ok=False, error=str(exc))

    if board:
        if not kb.board_exists(board):
            return CallbackResolution(
                ok=False, error=f"unknown task-status callback board {board!r}"
            )
        conn = kb.connect(board=board)
        try:
            return resolve_task_status_callback(
                conn,
                callback_data=callback_data,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                message_id=message_id,
                allowed_actions=allowed_actions,
            )
        finally:
            conn.close()

    candidates: list[str] = []
    seen_paths: set[str] = set()
    for metadata in kb.list_boards(include_archived=False):
        slug = str(metadata.get("slug") or kb.DEFAULT_BOARD)
        path = str(metadata.get("db_path") or kb.kanban_db_path(slug))
        if path in seen_paths:
            continue
        seen_paths.add(path)
        conn = kb.connect(board=slug)
        try:
            binding = kb.get_task_status_binding(conn, task_id)
            if binding is None:
                continue
            living_message_matches = (
                binding.chat_id == str(chat_id)
                and binding.message_thread_id == str(message_thread_id or "")
                and binding.message_id == str(message_id)
            )
            callback_message_matches = kb.matches_task_status_callback_message(
                conn,
                kanban_task_id=task_id,
                state_version=_version,
                platform="telegram",
                chat_id=str(chat_id),
                message_thread_id=str(message_thread_id or ""),
                message_id=str(message_id),
            )
        finally:
            conn.close()
        if living_message_matches or callback_message_matches:
            candidates.append(slug)

    if not candidates:
        return CallbackResolution(
            ok=False,
            error=(
                f"no exact task-status binding for callback task {task_id} "
                "and message destination"
            ),
        )
    if len(candidates) != 1:
        return CallbackResolution(
            ok=False,
            error=f"ambiguous task-status callback binding across boards: {candidates}",
        )
    conn = kb.connect(board=candidates[0])
    try:
        return resolve_task_status_callback(
            conn,
            callback_data=callback_data,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            message_id=message_id,
            allowed_actions=allowed_actions,
        )
    finally:
        conn.close()


def _dedup_key(
    *,
    task_id: str,
    lifecycle_state: str,
    state_version: int,
    destination: TaskStatusDestination,
    recommended_action: str,
    channel: str,
) -> str:
    raw = json.dumps(
        {
            "task_id": task_id,
            "lifecycle_state": lifecycle_state,
            "state_version": int(state_version),
            "chat_id": destination.chat_id,
            "thread_id": destination.message_thread_id,
            "recommended_action": recommended_action,
            "channel": channel,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _configured_destination(config: Mapping[str, Any], route: str) -> TaskStatusDestination:
    routes = config.get("routes")
    route_cfg = routes.get(route) if isinstance(routes, Mapping) else None
    if not isinstance(route_cfg, Mapping):
        raise ValueError(f"task-status {route} destination is not configured")
    if "chat_id" not in route_cfg or "thread_id" not in route_cfg:
        raise ValueError(
            f"task-status {route} destination requires exact chat_id and thread_id"
        )
    return TaskStatusDestination(
        chat_id=str(route_cfg.get("chat_id") or ""),
        message_thread_id=str(route_cfg.get("thread_id") or ""),
    )


def resolve_exact_task_status_destination(
    route: DeliveryRoute,
    *,
    origin: TaskStatusDestination,
    config: Mapping[str, Any],
) -> TaskStatusDestination:
    """Resolve one explicit route without topic, session, or recency inference."""
    route = DeliveryRoute(route)
    if route == DeliveryRoute.PROJECT:
        return origin
    return _configured_destination(config, route.value)


def resolve_task_status_push_destination(
    kind: PublicationKind,
    *,
    origin: TaskStatusDestination,
    config: Mapping[str, Any],
) -> TaskStatusDestination:
    """Resolve exactly one push destination without title/recency inference."""
    if kind in {PublicationKind.DECISION, PublicationKind.BLOCKER}:
        return resolve_exact_task_status_destination(
            DeliveryRoute.CONTROL, origin=origin, config=config
        )
    if kind in {
        PublicationKind.MATERIAL_CHANGE,
        PublicationKind.ACCEPTED_COMPLETION,
    }:
        return resolve_exact_task_status_destination(
            DeliveryRoute.PROJECT, origin=origin, config=config
        )
    raise ValueError(f"publication kind {kind.value!r} does not permit a fresh push")


class TaskStatusPublisher:
    def __init__(
        self,
        *,
        conn,
        adapter,
        config: Mapping[str, Any],
        profile_name: str,
    ) -> None:
        self.conn = conn
        self.adapter = adapter
        self.config = config
        self.profile_name = str(profile_name or "")

    def _preflight(self) -> Optional[str]:
        if not bool(self.config.get("enabled", False)):
            return "task-status publication is disabled"
        required_profile = str(self.config.get("publisher_profile") or "").strip()
        if not required_profile:
            return "task-status publisher_profile is not configured"
        if self.profile_name != required_profile:
            return (
                f"task-status publisher profile mismatch: expected "
                f"{required_profile!r}, got {self.profile_name!r}"
            )
        for method in (
            "send_task_status",
            "edit_task_status",
            "send_task_status_push",
        ):
            if not callable(getattr(self.adapter, method, None)):
                return f"configured adapter does not support {method}"
        return None

    @staticmethod
    def _buttons(publication: TaskStatusPublication) -> list[dict[str, str]]:
        return [
            {
                "label": str(button.label),
                "callback_data": make_task_status_callback_data(
                    publication.kanban_task_id,
                    publication.state_version,
                    button.action,
                ),
            }
            for button in publication.buttons
        ]

    async def publish(self, publication: TaskStatusPublication) -> PublicationResult:
        error = self._preflight()
        if error:
            return PublicationResult(ok=False, error=error)
        if publication.operation not in {"create", "update"}:
            return PublicationResult(ok=False, error="operation must be create or update")
        if not publication.content or not publication.content.strip():
            return PublicationResult(ok=False, error="task-status content is required")
        if not publication.session_id or not publication.session_id.strip():
            return PublicationResult(ok=False, error="task-status session_id is required")
        try:
            buttons = self._buttons(publication)
        except ValueError as exc:
            return PublicationResult(ok=False, error=str(exc))

        if publication.operation == "create":
            return await self._create(publication, buttons)
        return await self._update(publication, buttons)

    async def _create(
        self, publication: TaskStatusPublication, buttons: list[dict[str, str]]
    ) -> PublicationResult:
        if publication.destination is None:
            return PublicationResult(
                ok=False, error="create requires an exact task-status destination"
            )
        if publication.state_version != 1:
            return PublicationResult(
                ok=False, error="task-status create must use state_version 1"
            )
        task = kb.get_task(self.conn, publication.kanban_task_id)
        if task is None:
            return PublicationResult(
                ok=False,
                error=f"unknown Kanban task {publication.kanban_task_id!r}",
            )
        if task.session_id and task.session_id != publication.session_id:
            return PublicationResult(
                ok=False, error="task-status session_id does not match the Kanban task"
            )
        try:
            kb.reserve_task_status_binding(
                self.conn,
                kanban_task_id=publication.kanban_task_id,
                platform="telegram",
                chat_id=publication.destination.chat_id,
                message_thread_id=publication.destination.message_thread_id,
                session_id=publication.session_id,
                linear_issue_key=publication.linear_issue_key,
                lifecycle_state=publication.lifecycle_state,
                state_version=publication.state_version,
            )
        except ValueError as exc:
            return PublicationResult(ok=False, error=str(exc))

        result = await self.adapter.send_task_status(
            publication.destination.chat_id,
            publication.content,
            buttons=buttons,
            metadata={
                "thread_id": publication.destination.message_thread_id,
                "notify": False,
            },
        )
        if not getattr(result, "success", False) or not getattr(
            result, "message_id", None
        ):
            kb.delete_pending_task_status_binding(
                self.conn, publication.kanban_task_id
            )
            return PublicationResult(
                ok=False,
                error=(
                    getattr(result, "error", None)
                    or "task-status create returned no message_id"
                ),
            )
        try:
            binding = kb.finalize_task_status_binding(
                self.conn,
                kanban_task_id=publication.kanban_task_id,
                message_id=str(result.message_id),
            )
        except ValueError as exc:
            return PublicationResult(ok=False, error=str(exc))
        return PublicationResult(ok=True, message_id=binding.message_id)

    async def _update(
        self, publication: TaskStatusPublication, buttons: list[dict[str, str]]
    ) -> PublicationResult:
        binding = kb.get_task_status_binding(
            self.conn, publication.kanban_task_id
        )
        if binding is None or not binding.message_id:
            return PublicationResult(
                ok=False,
                error=(
                    f"no complete task-status binding for Kanban task "
                    f"{publication.kanban_task_id}"
                ),
            )
        if binding.session_id != publication.session_id:
            return PublicationResult(
                ok=False, error="task-status session_id does not match the binding"
            )
        if publication.destination is not None and (
            publication.destination.chat_id != binding.chat_id
            or publication.destination.message_thread_id
            != binding.message_thread_id
        ):
            return PublicationResult(
                ok=False, error="task-status update destination does not match the binding"
            )
        if publication.state_version < binding.state_version:
            return PublicationResult(
                ok=False,
                error=(
                    f"stale task-status update: version {publication.state_version}, "
                    f"current {binding.state_version}"
                ),
            )
        if publication.state_version > binding.state_version + 1:
            return PublicationResult(
                ok=False, error="task-status update would skip a state version"
            )
        if (
            publication.state_version == binding.state_version
            and publication.lifecycle_state != binding.lifecycle_state
        ):
            return PublicationResult(
                ok=False,
                error="task-status lifecycle change requires a new state version",
            )

        origin = TaskStatusDestination(
            binding.chat_id, binding.message_thread_id
        )
        push_destination: Optional[TaskStatusDestination] = None
        if publication.kind in PUSH_KINDS:
            if not publication.push_content or not publication.push_content.strip():
                return PublicationResult(
                    ok=False,
                    error=(
                        f"{publication.kind.value} requires concise push_content"
                    ),
                )
            try:
                push_destination = resolve_task_status_push_destination(
                    publication.kind,
                    origin=origin,
                    config=self.config,
                )
            except ValueError as exc:
                return PublicationResult(ok=False, error=str(exc))
        edit_key = _dedup_key(
            task_id=publication.kanban_task_id,
            lifecycle_state=publication.lifecycle_state,
            state_version=publication.state_version,
            destination=origin,
            recommended_action=publication.recommended_action,
            channel="living-card",
        )
        window = int(self.config.get("dedup_seconds", 180) or 180)
        edit_claimed = kb.claim_task_status_publication(
            self.conn,
            dedup_key=edit_key,
            kanban_task_id=publication.kanban_task_id,
            window_seconds=window,
        )
        if edit_claimed:
            result = await self.adapter.edit_task_status(
                binding.chat_id,
                binding.message_id,
                publication.content,
                buttons=buttons,
                metadata={
                    "thread_id": binding.message_thread_id,
                    "notify": False,
                },
            )
            if not getattr(result, "success", False):
                kb.release_task_status_publication(self.conn, edit_key)
                return PublicationResult(
                    ok=False,
                    error=(getattr(result, "error", None) or "task-status edit failed"),
                )
            if publication.state_version == binding.state_version + 1:
                try:
                    binding = kb.advance_task_status_binding(
                        self.conn,
                        kanban_task_id=publication.kanban_task_id,
                        expected_version=binding.state_version,
                        lifecycle_state=publication.lifecycle_state,
                        state_version=publication.state_version,
                    )
                except ValueError as exc:
                    return PublicationResult(ok=False, error=str(exc))

        pushed = False
        push_deduped = False
        if publication.kind in PUSH_KINDS:
            assert push_destination is not None
            push_key = _dedup_key(
                task_id=publication.kanban_task_id,
                lifecycle_state=publication.lifecycle_state,
                state_version=publication.state_version,
                destination=push_destination,
                recommended_action=publication.recommended_action,
                channel="push",
            )
            push_claimed = kb.claim_task_status_publication(
                self.conn,
                dedup_key=push_key,
                kanban_task_id=publication.kanban_task_id,
                window_seconds=window,
            )
            if push_claimed:
                push_result = await self.adapter.send_task_status_push(
                    push_destination.chat_id,
                    publication.push_content,
                    buttons=buttons,
                    metadata={
                        "thread_id": push_destination.message_thread_id,
                        "notify": True,
                    },
                )
                if not getattr(push_result, "success", False):
                    kb.release_task_status_publication(self.conn, push_key)
                    return PublicationResult(
                        ok=False,
                        message_id=binding.message_id,
                        error=(
                            getattr(push_result, "error", None)
                            or "task-status push failed"
                        ),
                    )
                if buttons:
                    push_message_id = getattr(push_result, "message_id", None)
                    if not push_message_id:
                        return PublicationResult(
                            ok=False,
                            message_id=binding.message_id,
                            error=(
                                "task-status decision push returned no message_id; "
                                "callbacks were not enabled"
                            ),
                        )
                    try:
                        kb.register_task_status_callback_message(
                            self.conn,
                            kanban_task_id=publication.kanban_task_id,
                            state_version=publication.state_version,
                            platform="telegram",
                            chat_id=push_destination.chat_id,
                            message_thread_id=push_destination.message_thread_id,
                            message_id=str(push_message_id),
                        )
                    except ValueError as exc:
                        return PublicationResult(
                            ok=False,
                            message_id=binding.message_id,
                            error=str(exc),
                        )
                pushed = True
            else:
                push_deduped = True

        return PublicationResult(
            ok=True,
            message_id=binding.message_id,
            deduplicated=(not edit_claimed and (not publication.kind in PUSH_KINDS or push_deduped)),
            pushed=pushed,
        )


async def invoke_task_status_action(
    handler: Callable[..., Any], resolution: CallbackResolution
) -> Any:
    """Invoke an injected consequential handler only after correlation passes."""
    result = handler(
        kanban_task_id=resolution.kanban_task_id,
        state_version=resolution.state_version,
        action=resolution.action,
    )
    if inspect.isawaitable(result):
        return await result
    return result
