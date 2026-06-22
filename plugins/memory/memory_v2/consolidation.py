"""Rule-based promotion/consolidation for Memory v2 candidates.

The v0 consolidator is deliberately cheap and deterministic. It promotes only
candidate shapes that can become source-grounded canonical ``MemoryItem`` records
without an LLM, archives/rejects candidates that belong in other systems, and
handles explicit supersession by updating old records rather than silently
letting contradictions coexist as active memories.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, cast

from .index import MemoryV2Index
from .schemas import (
    CandidateMemory,
    GateDecision,
    MemoryItem,
    MemoryStatus,
    MemoryType,
    ProjectCard,
    ProjectStatus,
    normalize_project_id,
    utc_now_iso,
)
from .store import MemoryV2Store


@dataclass
class ConsolidationReport:
    """Summary of one consolidation pass."""

    considered: int = 0
    promoted: int = 0
    rejected: int = 0
    archived_only: int = 0
    superseded: int = 0
    promoted_ids: List[str] = field(default_factory=list)
    rejected_ids: List[str] = field(default_factory=list)
    archived_ids: List[str] = field(default_factory=list)
    superseded_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "considered": self.considered,
            "promoted": self.promoted,
            "rejected": self.rejected,
            "archived_only": self.archived_only,
            "superseded": self.superseded,
            "promoted_ids": list(self.promoted_ids),
            "rejected_ids": list(self.rejected_ids),
            "archived_ids": list(self.archived_ids),
            "superseded_ids": list(self.superseded_ids),
        }


class RuleBasedConsolidator:
    """Low-compute v0 candidate promoter.

    This class intentionally does not perform semantic LLM summarization. It only
    promotes explicit, already-gated candidates and keeps source refs attached so
    later richer consolidation can audit or improve the canonical records.
    """

    def consolidate(self, store: MemoryV2Store, index: MemoryV2Index) -> ConsolidationReport:
        candidates = store.list_candidates()
        report = ConsolidationReport()
        updated_candidates: List[CandidateMemory] = []

        for candidate in candidates:
            if candidate.gate_decision != GateDecision.PENDING:
                updated_candidates.append(candidate)
                continue

            report.considered += 1
            if not candidate.source_refs and not self._should_reject(candidate) and not self._should_archive_only(candidate):
                rejected = self._with_decision(
                    candidate,
                    GateDecision.REJECTED,
                    "Semantic promotion requires at least one source_refs evidence id.",
                )
                store.append_rejected_candidate(rejected)
                updated_candidates.append(rejected)
                report.rejected += 1
                report.rejected_ids.append(candidate.id)
                index.index_candidate(rejected)
                continue

            dangling_source_refs = self._dangling_source_refs(candidate, store)
            if dangling_source_refs and not self._should_reject(candidate) and not self._should_archive_only(candidate):
                rejected = self._with_decision(
                    candidate,
                    GateDecision.REJECTED,
                    f"Semantic promotion has dangling source_refs: {', '.join(dangling_source_refs)}.",
                )
                store.append_rejected_candidate(rejected)
                updated_candidates.append(rejected)
                report.rejected += 1
                report.rejected_ids.append(candidate.id)
                index.index_candidate(rejected)
                continue

            if self._should_reject(candidate):
                rejected = self._with_decision(
                    candidate,
                    GateDecision.REJECTED,
                    "Procedure candidates require skill authoring/review; not promoted as semantic memory.",
                )
                store.append_rejected_candidate(rejected)
                updated_candidates.append(rejected)
                report.rejected += 1
                report.rejected_ids.append(candidate.id)
                index.index_candidate(rejected)
                continue

            if self._is_open_loop_candidate(candidate):
                loop = store.upsert_open_loop(
                    {
                        "text": candidate.claim,
                        "source_refs": candidate.source_refs,
                        "session_id": self._session_id_from_source(candidate),
                        "candidate_id": candidate.id,
                    }
                )
                archived = self._with_decision(
                    candidate,
                    GateDecision.ARCHIVED_ONLY,
                    f"Routed to working/open_loops.yaml as {loop['id']}; semantic promotion skipped.",
                )
                updated_candidates.append(archived)
                report.archived_only += 1
                report.archived_ids.append(candidate.id)
                index.index_candidate(archived)
                continue

            if self._should_archive_only(candidate):
                archived = self._with_decision(
                    candidate,
                    GateDecision.ARCHIVED_ONLY,
                    f"Candidate belongs in {candidate.proposed_destination}; semantic promotion skipped in v0.",
                )
                updated_candidates.append(archived)
                report.archived_only += 1
                report.archived_ids.append(candidate.id)
                index.index_candidate(archived)
                continue

            if self._is_project_card_candidate(candidate):
                card = self._merge_project_card(candidate, store)
                path = store.write_project_card(card)
                index.index_project_card(card, file_path=path)
                promoted = self._with_decision(candidate, GateDecision.PROMOTED, f"Promoted to ProjectCard {card.id}.")
                updated_candidates.append(promoted)
                index.index_candidate(promoted)
                report.promoted += 1
                report.promoted_ids.append(card.id)
                continue

            item = self._memory_item_from_candidate(candidate)
            superseded = self._superseded_items_for(candidate, item, store)
            if superseded:
                item.supersedes = [old.id for old in superseded]
                for old in superseded:
                    old.status = MemoryStatus.SUPERSEDED
                    old.superseded_by = item.id
                    old.updated_at = utc_now_iso()
                    path = store.write_memory_item(old)
                    index.index_memory_item(old, file_path=path)
                    report.superseded += 1
                    report.superseded_ids.append(old.id)

            path = store.write_memory_item(item)
            index.index_memory_item(item, file_path=path)
            promoted = self._with_decision(candidate, GateDecision.PROMOTED, f"Promoted to canonical MemoryItem {item.id}.")
            updated_candidates.append(promoted)
            index.index_candidate(promoted)
            report.promoted += 1
            report.promoted_ids.append(item.id)

        store.rewrite_candidates(updated_candidates)
        return report

    @staticmethod
    def _with_decision(candidate: CandidateMemory, decision: GateDecision, reason: str) -> CandidateMemory:
        data = candidate.to_dict()
        data["gate_decision"] = cast(GateDecision, decision).value
        data["decision_reason"] = reason
        return CandidateMemory.from_dict(data)

    @staticmethod
    def _should_reject(candidate: CandidateMemory) -> bool:
        candidate_type = cast(MemoryType, candidate.type).value
        return candidate_type == MemoryType.PROCEDURE_REF.value or candidate.proposed_destination == "skills"

    @staticmethod
    def _dangling_source_refs(candidate: CandidateMemory, store: MemoryV2Store) -> List[str]:
        raw_event_ids = {str(event.get("id") or "") for event in store.read_raw_events()}
        missing: List[str] = []
        for source_id in candidate.source_refs:
            if store.read_source_ref(source_id) is None and source_id not in raw_event_ids:
                missing.append(source_id)
        return missing

    @staticmethod
    def _should_archive_only(candidate: CandidateMemory) -> bool:
        destination = str(candidate.proposed_destination or "")
        return destination.startswith("working/") or destination.startswith("episodic/")

    @staticmethod
    def _is_open_loop_candidate(candidate: CandidateMemory) -> bool:
        destination = str(candidate.proposed_destination or "")
        reason = str(candidate.promotion_reason or "").lower()
        return destination == "working/open_loops.yaml" or reason.startswith("open_loop")

    @staticmethod
    def _session_id_from_source(candidate: CandidateMemory) -> str:
        # Raw event metadata can resolve this later; keep the open-loop record cheap and source-grounded.
        return ""

    @staticmethod
    def _is_project_card_candidate(candidate: CandidateMemory) -> bool:
        candidate_type = cast(MemoryType, candidate.type).value
        destination = str(candidate.proposed_destination or "")
        return candidate_type == MemoryType.PROJECT_STATE.value and destination.startswith("semantic/projects/")

    def _merge_project_card(self, candidate: CandidateMemory, store: MemoryV2Store) -> ProjectCard:
        project_id = self._project_id_for(candidate)
        existing = store.read_project_card(project_id)
        card = existing or ProjectCard(id=project_id, name=self._project_name_for(project_id), importance=candidate.importance)
        update_kind = self._project_update_kind(candidate)
        update_text = self._project_update_text(candidate, update_kind)

        if update_kind == "goal":
            card.goal = update_text
        elif update_kind == "why_it_matters":
            card.why_it_matters = update_text
        elif update_kind == "decision":
            card.decisions = self._append_unique(card.decisions, update_text)
        elif update_kind == "open_question":
            card.open_questions = self._append_unique(card.open_questions, update_text)
        elif update_kind == "next_action":
            card.next_actions = self._append_unique(card.next_actions, update_text)
        elif update_kind == "status":
            card.status = self._project_status_from_text(update_text)
        else:
            card.current_state = update_text

        card.importance = max(card.importance, candidate.importance)
        card.source_refs = self._append_unique(card.source_refs, *candidate.source_refs)
        card.updated_at = utc_now_iso()
        return ProjectCard.from_dict(card.to_dict())

    @staticmethod
    def _project_id_for(candidate: CandidateMemory) -> str:
        destination = str(candidate.proposed_destination or "")
        match = re.search(r"semantic/projects/([^/]+?)(?:\.ya?ml)?$", destination)
        if match:
            return normalize_project_id(match.group(1))
        match = re.search(r"\bproject\s+(.+?)\s+(?:current state|decision|open question|next action|status|goal|why it matters)\s*:", candidate.claim, re.IGNORECASE)
        if match:
            return normalize_project_id(match.group(1))
        return normalize_project_id("project")

    @staticmethod
    def _project_name_for(project_id: str) -> str:
        slug = normalize_project_id(project_id).split(":", 1)[1]
        return " ".join(part.upper() if part in {"v2", "api", "ui"} else part.capitalize() for part in slug.split("-"))

    @staticmethod
    def _project_update_kind(candidate: CandidateMemory) -> str:
        text = f"{candidate.promotion_reason}\n{candidate.claim}".lower()
        if "open_question" in text or "open question" in text:
            return "open_question"
        if "next_action" in text or "next action" in text:
            return "next_action"
        if "why_it_matters" in text or "why it matters" in text:
            return "why_it_matters"
        for kind in ("decision", "status", "goal"):
            if re.search(rf"\b{kind}\b", text):
                return kind
        return "current_state"

    @staticmethod
    def _project_update_text(candidate: CandidateMemory, update_kind: str) -> str:
        label = update_kind.replace("_", r"[ _]")
        patterns = [
            rf"^\s*project\s+.+?\s+{label}\s*:\s*(.+?)\s*$",
            rf"^\s*{label}\s*:\s*(.+?)\s*$",
        ]
        if update_kind == "current_state":
            patterns.insert(0, r"^\s*project\s+.+?\s+current\s+state\s*:\s*(.+?)\s*$")
        for pattern in patterns:
            match = re.match(pattern, candidate.claim, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return candidate.claim.strip()

    @staticmethod
    def _project_status_from_text(text: str) -> ProjectStatus:
        lowered = text.lower()
        if "archive" in lowered:
            return ProjectStatus.ARCHIVED
        if "pause" in lowered:
            return ProjectStatus.PAUSED
        return ProjectStatus.ACTIVE

    @staticmethod
    def _append_unique(values: List[str], *new_values: str) -> List[str]:
        merged = list(values)
        seen = {value for value in merged}
        for value in new_values:
            text = str(value or "").strip()
            if text and text not in seen:
                merged.append(text)
                seen.add(text)
        return merged

    def _memory_item_from_candidate(self, candidate: CandidateMemory) -> MemoryItem:
        memory_type = cast(MemoryType, candidate.type).value
        subject, predicate = self._subject_predicate_for(candidate)
        item_id = self._memory_id(memory_type, subject, predicate, candidate.claim)
        return MemoryItem(
            id=item_id,
            type=memory_type,
            subject=subject,
            predicate=predicate,
            value=candidate.claim,
            body=candidate.claim,
            summary=candidate.claim,
            confidence=candidate.confidence,
            importance=candidate.importance,
            source_refs=list(candidate.source_refs),
            tags=[memory_type, "promoted_from_candidate", candidate.id],
        )

    @staticmethod
    def _subject_predicate_for(candidate: CandidateMemory) -> tuple[str, str]:
        memory_type = cast(MemoryType, candidate.type).value
        claim = candidate.claim.strip()
        if memory_type == MemoryType.PREFERENCE.value:
            subject_match = re.match(r"^([A-Z][\w.-]{1,40})\s+prefers\b", claim)
            subject = subject_match.group(1) if subject_match else "user"
            return subject, "prefers"
        if memory_type == MemoryType.ENVIRONMENT.value:
            return "Hermes runtime", "has_environment_fact"
        if memory_type == MemoryType.PROJECT_STATE.value:
            return "project", "has_current_state"
        if memory_type == MemoryType.EPISODE.value:
            return "session", "has_episode"
        return "memory", "states"

    @staticmethod
    def _memory_id(memory_type: str, subject: str, predicate: str, claim: str) -> str:
        normalized = " ".join([memory_type, subject, predicate, claim]).lower().encode("utf-8")
        digest = hashlib.sha256(normalized).hexdigest()[:12]
        return f"mem_{RuleBasedConsolidator._safe_slug(memory_type)}_{digest}"

    @staticmethod
    def _safe_slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return slug or "item"

    def _superseded_items_for(self, candidate: CandidateMemory, new_item: MemoryItem, store: MemoryV2Store) -> List[MemoryItem]:
        if not self._candidate_requests_supersession(candidate):
            return []
        superseded: List[MemoryItem] = []
        new_type = cast(MemoryType, new_item.type).value
        candidate_tokens = self._supersession_tokens(candidate.claim)
        for item in store.list_memory_items(memory_type=new_type, status=MemoryStatus.ACTIVE.value):
            if item.id == new_item.id:
                continue
            if item.subject != new_item.subject or item.predicate != new_item.predicate:
                continue
            item_text = " ".join(str(part or "") for part in (item.value, item.summary, item.body, " ".join(item.tags)))
            if candidate_tokens.intersection(self._supersession_tokens(item_text)):
                superseded.append(item)
        return superseded

    @staticmethod
    def _supersession_tokens(text: str) -> set[str]:
        stopwords = {
            "a",
            "an",
            "and",
            "alex",
            "for",
            "i",
            "is",
            "it",
            "not",
            "now",
            "prefers",
            "prefer",
            "the",
            "to",
            "user",
            "with",
        }
        return {token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower()) if token not in stopwords}

    @staticmethod
    def _candidate_requests_supersession(candidate: CandidateMemory) -> bool:
        text = f"{candidate.promotion_reason}\n{candidate.claim}".lower()
        return any(marker in text for marker in ("supersede_existing", " no longer ", " not ", " instead of "))
