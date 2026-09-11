"""JSON-safe canonical events and Router state values."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping


MEETING_STATUSES = frozenset(
    {
        "pending_approval",
        "preflight",
        "awaiting_sensitive_approval",
        "sensitive_lookup_approved",
        "ready",
        "round_open",
        "selecting",
        "publishing",
        "awaiting_continue",
        "completed",
        "stopped",
        "blocked_materials",
        "blocked_preflight_error",
        "blocked_all_error",
        "blocked_report_failed",
        "blocked_publication",
        "blocked_recovery",
    }
)
MEETING_ROUND_STATUSES = frozenset(
    {
        "collecting",
        "closed",
        "discarded",
        "blocked_all_error",
        "blocked_publication",
    }
)
MEETING_CANDIDATE_STATUSES = frozenset(
    {"pending", "submitted", "timeout", "candidate_error", "discarded"}
)


@dataclass(frozen=True)
class CanonicalEvent:
    event_id: str
    channel_id: str
    thread_ts: str
    text: str = ""
    author_kind: str = "human"  # human or bot
    author_profile: str | None = None
    author_user_id: str | None = None
    mentioned_user_ids: tuple[str, ...] = ()
    channel_type: str = "channel"
    task_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.author_kind not in {"human", "bot"}:
            raise ValueError("author_kind must be human or bot")

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "channel_id": self.channel_id,
            "thread_ts": self.thread_ts,
            "text": self.text,
            "author_kind": self.author_kind,
            "author_profile": self.author_profile,
            "author_user_id": self.author_user_id,
            "mentioned_user_ids": list(self.mentioned_user_ids),
            "channel_type": self.channel_type,
            "task_id": self.task_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CanonicalEvent":
        metadata = value.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        return cls(
            event_id=str(value.get("event_id") or ""),
            channel_id=str(value.get("channel_id") or ""),
            thread_ts=str(value.get("thread_ts") or ""),
            text=str(value.get("text") or ""),
            author_kind=str(value.get("author_kind") or "human"),
            author_profile=(str(value["author_profile"]) if value.get("author_profile") else None),
            author_user_id=(str(value["author_user_id"]) if value.get("author_user_id") else None),
            mentioned_user_ids=tuple(str(item) for item in value.get("mentioned_user_ids", ()) or ()),
            channel_type=str(value.get("channel_type") or "channel"),
            task_id=(str(value["task_id"]) if value.get("task_id") else None),
            metadata=metadata,
        )


@dataclass(frozen=True)
class RouterThreadState:
    """The exact logical thread state schema from the Stage 3 design."""

    channel_id: str
    thread_ts: str
    task_id: str | None = None
    state_version: int = 0
    lock_version: int = 0
    mode: str = "work"
    owner: str | None = None
    active_team: str | None = None
    advisor_stack: tuple[str, ...] = ()
    last_human_target: str | None = None
    last_event_id: str | None = None
    updated_at: float = 0.0
    expires_at: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"work", "meeting"}:
            raise ValueError("Router thread mode must be work or meeting")

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "thread_ts": self.thread_ts,
            "task_id": self.task_id,
            "state_version": self.state_version,
            "lock_version": self.lock_version,
            "mode": self.mode,
            "owner": self.owner,
            "active_team": self.active_team,
            "advisor_stack": list(self.advisor_stack),
            "last_human_target": self.last_human_target,
            "last_event_id": self.last_event_id,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RouterThreadState":
        advisor_stack = value.get("advisor_stack", ())
        if not isinstance(advisor_stack, (list, tuple)) or not all(
            isinstance(item, str) for item in advisor_stack
        ):
            raise ValueError("Router thread advisor_stack is invalid")
        return cls(
            channel_id=str(value.get("channel_id") or ""),
            thread_ts=str(value.get("thread_ts") or ""),
            task_id=str(value["task_id"]) if value.get("task_id") else None,
            state_version=int(value.get("state_version") or 0),
            lock_version=int(value.get("lock_version") or 0),
            mode=str(value.get("mode") or "work"),
            owner=str(value["owner"]) if value.get("owner") else None,
            active_team=str(value["active_team"]) if value.get("active_team") else None,
            advisor_stack=tuple(advisor_stack),
            last_human_target=(
                str(value["last_human_target"]) if value.get("last_human_target") else None
            ),
            last_event_id=str(value["last_event_id"]) if value.get("last_event_id") else None,
            updated_at=float(value.get("updated_at") or 0.0),
            expires_at=float(value.get("expires_at") or 0.0),
        )


@dataclass(frozen=True)
class MeetingState:
    meeting_id: str
    channel_id: str
    thread_ts: str
    status: str = "pending_approval"
    generation: int = 1
    current_round: int = 0
    participants: tuple[str, ...] = ()
    timeout_round_streak: int = 0
    all_pass_streak: int = 0
    return_state: RouterThreadState | None = None
    preflight_result_json: str | None = None
    block_reason: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.meeting_id:
            raise ValueError("meeting_id is required")
        if self.status not in MEETING_STATUSES:
            raise ValueError("unknown meeting status")
        if self.generation < 1:
            raise ValueError("meeting generation must be positive")
        if not 0 <= self.current_round <= 5:
            raise ValueError("meeting round must be between zero and five")
        if self.participants and len(self.participants) < 3:
            raise ValueError("meeting participant pool must contain at least three Staff")
        if len(set(self.participants)) != len(self.participants):
            raise ValueError("meeting participants must be unique")
        if self.return_state is not None:
            if self.return_state.mode != "work":
                raise ValueError("meeting return state must be Work Mode")
            if (
                self.return_state.channel_id != self.channel_id
                or self.return_state.thread_ts != self.thread_ts
            ):
                raise ValueError("meeting return state must match the meeting thread")


@dataclass(frozen=True)
class MeetingRoundState:
    meeting_id: str
    round_id: int
    generation: int
    status: str = "collecting"
    packet_json: str = "{}"
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.meeting_id or not 1 <= self.round_id <= 5 or self.generation < 1:
            raise ValueError("meeting round identity is invalid")
        if self.status not in MEETING_ROUND_STATUSES:
            raise ValueError("unknown meeting round status")


@dataclass(frozen=True)
class MeetingCandidate:
    meeting_id: str
    round_id: int
    generation: int
    participant: str
    status: str = "pending"
    reason_to_speak: bool | None = None
    reason_class: str = "none"
    statement: str = ""
    error_class: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if not self.meeting_id or not 1 <= self.round_id <= 5 or self.generation < 1:
            raise ValueError("meeting candidate identity is invalid")
        if not self.participant:
            raise ValueError("meeting candidate participant is required")
        if self.status not in MEETING_CANDIDATE_STATUSES:
            raise ValueError("unknown candidate status")


@dataclass(frozen=True)
class RouteAction:
    kind: str
    target_profile: str | None = None
    response_kind: str = "silence"
    content: str | None = None
    reason: str = ""
    artifact_path: str | None = None
    context: str | None = None


@dataclass(frozen=True)
class WorkExecution:
    directive_id: str
    source_event_id: str
    slack_team_id: str
    channel_id: str
    thread_ts: str
    directive_ts: str
    owner_profile: str
    owner_slack_user_id: str
    state: str
    owner_task_id: str | None = None
    pm_task_id: str | None = None
    owner_run_id: int | None = None
    pm_run_id: int | None = None
    review_generation: int = 0
    approval_required: bool = False
    last_error: str | None = None
    lifecycle_attempt_count: int = 0
    next_lifecycle_attempt_at: float | None = None


@dataclass(frozen=True)
class RouteDecision:
    action: RouteAction
    next_state: RouterThreadState
    handoff_count_delta: int = 0


@dataclass(frozen=True)
class LoungeExecutionIntent:
    kind: str


def classify_lounge_execution_intent(text: str) -> LoungeExecutionIntent:
    """Fail-closed deterministic classifier for TSK-60 Lounge admission."""
    normalized = " ".join((text or "").strip().split())

    execution_phrases = (
        "진행해",
        "착수해",
        "업무요청으로 넘겨",
        "작성해줘",
        "수정해줘",
        "찾아줘",
    )
    discussion_phrases = (
        "해야지",
        "대응해야지",
        "검토해보자",
        "어떻게 생각해",
        "문제 한번 보자",
    )
    ambiguous_execution_phrases = (
        "이거 해",
        "처리 가능",
        "한번 해볼까",
    )

    if any(
        re.search(rf"{re.escape(phrase)}(?![가-힣A-Za-z0-9])", normalized)
        for phrase in execution_phrases
    ):
        return LoungeExecutionIntent(kind="execute")

    if any(phrase in normalized for phrase in discussion_phrases):
        return LoungeExecutionIntent(kind="discussion")

    if any(phrase in normalized for phrase in ambiguous_execution_phrases):
        return LoungeExecutionIntent(kind="clarify")

    return LoungeExecutionIntent(kind="discussion")
