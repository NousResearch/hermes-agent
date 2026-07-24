"""Typed HEGI domain models with stable JSON serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class SourceMessage:
    source_agent: str
    source_db: str
    message_id: int
    session_id: str
    platform_message_id: str | None
    chat_id: str
    chat_type: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: float
    active: bool = True
    compacted: bool = False

    def trace_id(self) -> str:
        return f"{self.source_agent}:{self.session_id}:{self.message_id}"


@dataclass(slots=True)
class MeetingEpisode:
    meeting_id: str
    chat_id: str
    started_at: float
    ended_at: float
    participants: list[str]
    messages: list[SourceMessage]
    episode_hash: str
    topic_hint: str | None = None
    status: Literal[
        "collecting", "quiet", "analyzing", "reported", "archived", "failed"
    ] = "collecting"

    @property
    def source_message_ids(self) -> list[int]:
        return [message.message_id for message in self.messages]

    @property
    def source_session_ids(self) -> list[str]:
        return sorted({message.session_id for message in self.messages})


@dataclass(slots=True)
class DiscussionStage:
    heading: str
    summary: str
    source_message_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class AgentPosition:
    agent: str
    position: str
    contributions: list[str] = field(default_factory=list)
    source_message_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ConceptDefinition:
    name: str
    definition: str
    status: Literal["proposed", "agreed", "uncertain"] = "proposed"
    source_message_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceItem:
    claim: str
    source: str | None
    verification: Literal["검증됨", "추정", "추가 확인 필요"]
    source_message_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ActionItem:
    action_id: str
    title: str
    description: str
    source_message_ids: list[int]
    owner: str | None
    priority: Literal["critical", "high", "medium", "low"]
    status: Literal["open", "in_progress", "blocked", "done", "cancelled"] = "open"
    deadline: str | None = None
    project_id: str | None = None
    rationale: str = ""


@dataclass(slots=True)
class MemoryMatch:
    memory_id: str
    title: str
    summary: str
    relation: Literal["높은 중복", "부분 중복", "낮은 중복", "신규"]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryEvaluation:
    searched_queries: list[str] = field(default_factory=list)
    matched_memories: list[MemoryMatch] = field(default_factory=list)
    duplicate_score: float | None = None
    novelty_score: float | None = None
    significance_score: float | None = None
    durability_score: float | None = None
    recommendation: Literal[
        "no_memory", "merge_existing", "create_stm_draft", "needs_professor_review"
    ] = "needs_professor_review"
    candidate_memory_title: str | None = None
    candidate_memory_summary: str | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MeetingMinutes:
    meeting_id: str
    title: str
    background: str
    agenda: list[str]
    discussion_flow: list[DiscussionStage]
    agent_positions: list[AgentPosition]
    professor_positions: list[str]
    agreements: list[str]
    disagreements: list[str]
    unresolved_questions: list[str]
    new_concepts: list[ConceptDefinition]
    evidence_and_sources: list[EvidenceItem]
    research_direction: list[str]
    action_items: list[ActionItem]
    memory_evaluation: MemoryEvaluation | None
    confidence: float
    warnings: list[str]
    recommendation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def as_jsonable(value: Any) -> Any:
    """Convert dataclass graphs into JSON-compatible values."""
    if hasattr(value, "__dataclass_fields__"):
        return {key: as_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value
