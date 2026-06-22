"""Dataset schemas and fixture loaders for Memory v2 evals."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvalEvent:
    id: str
    session_id: str
    role: str
    text: str
    expected_candidate_type: str = ""


@dataclass(frozen=True)
class EvalQuery:
    id: str
    route: str
    text: str
    expected_answer_contains: list[str] = field(default_factory=list)
    expected_source_refs: list[str] = field(default_factory=list)
    should_retrieve: bool = True


@dataclass(frozen=True)
class EvalDataset:
    version: int
    name: str
    description: str
    events: list[EvalEvent]
    queries: list[EvalQuery]

    def query_by_id(self, query_id: str) -> EvalQuery:
        for query in self.queries:
            if query.id == query_id:
                return query
        raise KeyError(f"query not found: {query_id}")


def load_eval_dataset(path: str | Path) -> EvalDataset:
    payload: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return EvalDataset(
        version=int(payload.get("version") or 1),
        name=str(payload.get("name") or ""),
        description=str(payload.get("description") or ""),
        events=[EvalEvent(**event) for event in payload.get("events", [])],
        queries=[EvalQuery(**query) for query in payload.get("queries", [])],
    )


def load_locomo_sample(path: str | Path) -> EvalDataset:
    """Load a tiny local LoCoMo-shaped JSON sample into Memory v2 eval rows.

    This is intentionally a skeleton importer for tests and adapter development,
    not a downloader and not a vendored copy of the public LoCoMo dataset. The
    expected local JSON shape is explicit:

    - ``dataset_id``/``description`` metadata at the top level.
    - ``conversations`` list with ``conversation_id`` and ``messages``.
    - each message has ``id``, ``speaker`` (or ``role``), and ``text``.
    - ``qa_pairs`` list with ``id``, ``conversation_id``, ``question``, optional
      ``answer_contains``, optional ``source_message_ids``, optional ``route``,
      and optional ``should_retrieve``.

    Message IDs are preserved as ``EvalEvent.id`` and QA ``source_message_ids``
    are preserved as ``EvalQuery.expected_source_refs`` so eval source-recall
    metrics can stay source-grounded.
    """

    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    events: list[EvalEvent] = []
    queries: list[EvalQuery] = []

    for conversation in payload.get("conversations", []):
        session_id = str(conversation["conversation_id"])
        for message in conversation.get("messages", []):
            role = str(message.get("role") or message.get("speaker") or "user")
            events.append(
                EvalEvent(
                    id=str(message["id"]),
                    session_id=session_id,
                    role=role,
                    text=str(message.get("text") or ""),
                )
            )

    for qa_pair in payload.get("qa_pairs", []):
        queries.append(
            EvalQuery(
                id=str(qa_pair["id"]),
                route=str(qa_pair.get("route") or "past_conversation_exact"),
                text=str(qa_pair.get("question") or qa_pair.get("text") or ""),
                expected_answer_contains=[str(item) for item in qa_pair.get("answer_contains", [])],
                expected_source_refs=[str(item) for item in qa_pair.get("source_message_ids", [])],
                should_retrieve=bool(qa_pair.get("should_retrieve", True)),
            )
        )

    return EvalDataset(
        version=int(payload.get("version") or 1),
        name=str(payload.get("dataset_id") or payload.get("name") or "locomo_sample"),
        description=str(payload.get("description") or ""),
        events=events,
        queries=queries,
    )
