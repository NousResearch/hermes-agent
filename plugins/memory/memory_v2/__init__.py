"""Memory v2 provider.

Local, profile-scoped memory provider for routed, source-grounded,
low-compute long-term memory. It stores raw turn evidence, gated candidates,
semantic/core/episodic records, open loops, and a rebuildable SQLite FTS index.
Dynamic recall is returned through bounded memory packets rather than by
inflating the stable system prompt.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agent.memory_provider import MemoryProvider
from .consolidation import RuleBasedConsolidator
from .daily_consolidation import run_daily_consolidation_report
from .index import MemoryV2Index
from .redaction import redact_data, redact_text
from .retrieval import MemoryPacketComposer
from .schemas import CandidateMemory, GateDecision, MemoryItem, WorkingMemory, utc_now_iso
from .store import MemoryV2Store
from .write_gate import RuleBasedWriteGate

STATUS_SCHEMA = {
    "name": "memory_v2_status",
    "description": "Report Memory v2 provider health, profile-scoped paths, and basic record counts.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

SEARCH_SCHEMA = {
    "name": "memory_v2_search",
    "description": "Keyword search over the local Memory v2 SQLite FTS index.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "limit": {"type": "integer", "description": "Maximum results to return (default 10)."},
        },
        "required": ["query"],
    },
}

CONSOLIDATE_SCHEMA = {
    "name": "memory_v2_consolidate",
    "description": "Run Memory v2 v0 promotion/consolidation over pending write candidates.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

DAILY_REPORT_SCHEMA = {
    "name": "memory_v2_daily_report",
    "description": "Run Memory v2 daily consolidation and write an auditable daily report/episode.",
    "parameters": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Optional report date as YYYY-MM-DD; defaults to current UTC date."},
        },
        "required": [],
    },
}

CANDIDATES_SCHEMA = {
    "name": "memory_v2_candidates",
    "description": "List Memory v2 write candidates, optionally filtering by memory type and gate status.",
    "parameters": {
        "type": "object",
        "properties": {
            "type": {"type": "string", "description": "Optional candidate memory type filter."},
            "status": {"type": "string", "description": "Optional gate decision/status filter."},
            "limit": {"type": "integer", "description": "Maximum candidates to return."},
        },
        "required": [],
    },
}

PROMOTE_SCHEMA = {
    "name": "memory_v2_promote",
    "description": "Manually promote one pending Memory v2 candidate after source validation.",
    "parameters": {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "description": "Candidate id to promote."},
            "force": {"type": "boolean", "description": "Allow promotion despite missing/dangling source refs."},
        },
        "required": ["candidate_id"],
    },
}

REJECT_SCHEMA = {
    "name": "memory_v2_reject",
    "description": "Manually reject one Memory v2 candidate with an audit reason.",
    "parameters": {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "description": "Candidate id to reject."},
            "reason": {"type": "string", "description": "Human-readable rejection reason."},
        },
        "required": ["candidate_id", "reason"],
    },
}

SHOW_SOURCE_SCHEMA = {
    "name": "memory_v2_show_source",
    "description": "Show source evidence for a memory item, candidate, project, source id, or raw event id.",
    "parameters": {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Memory/candidate/source/raw event id."}},
        "required": ["id"],
    },
}

RESOLVE_OPEN_LOOP_SCHEMA = {
    "name": "memory_v2_resolve_open_loop",
    "description": "Update an open-loop status while preserving its history.",
    "parameters": {
        "type": "object",
        "properties": {
            "loop_id": {"type": "string", "description": "Open-loop id."},
            "status": {"type": "string", "description": "resolved, abandoned, blocked, snoozed, or open."},
            "resolution": {"type": "string", "description": "Optional resolution/update note."},
        },
        "required": ["loop_id", "status"],
    },
}

CONTRADICTIONS_SCHEMA = {
    "name": "memory_v2_contradictions",
    "description": "Build a Memory v2 contradiction/supersession dashboard. Default and candidate modes do not mutate memories; auto_supersede=true may mutate only high-confidence explicit corrections with source evidence.",
    "parameters": {
        "type": "object",
        "properties": {
            "create_candidates": {
                "type": "boolean",
                "description": "If true, append pending contradiction review candidates without superseding or mutating memories.",
            },
            "auto_supersede": {
                "type": "boolean",
                "description": "If true, automatically supersede only high-confidence explicit corrections with source evidence and audit metadata.",
            },
            "min_confidence": {
                "type": "number",
                "description": "Minimum dashboard confidence for automatic supersession (default 0.9).",
            },
            "limit": {"type": "integer", "description": "Maximum conflicts to return (default 50)."},
        },
        "required": [],
    },
}


class MemoryV2Provider(MemoryProvider):
    """Local profile-scoped Memory v2 provider."""

    def __init__(self) -> None:
        self._session_id = ""
        self._platform = ""
        self._agent_context = "primary"
        self._hermes_home: Path | None = None
        self._base_dir: Path | None = None
        self._store: MemoryV2Store | None = None
        self._index: MemoryV2Index | None = None
        self._initialized = False

    @property
    def name(self) -> str:
        return "memory_v2"

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def platform(self) -> str:
        return self._platform

    @property
    def base_dir(self) -> Path:
        if self._base_dir is None:
            raise RuntimeError("Memory v2 provider has not been initialized")
        return self._base_dir

    @property
    def store(self) -> MemoryV2Store:
        if self._store is None:
            raise RuntimeError("Memory v2 provider has not been initialized")
        return self._store

    @property
    def index(self) -> MemoryV2Index:
        if self._index is None:
            raise RuntimeError("Memory v2 provider has not been initialized")
        return self._index

    def is_available(self) -> bool:
        """Memory v2 has no external dependencies or credentials in v0."""
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        hermes_home = kwargs.get("hermes_home")
        if not hermes_home:
            raise ValueError("Memory v2 requires hermes_home for profile-scoped storage")

        self._session_id = session_id
        self._platform = str(kwargs.get("platform") or "")
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._hermes_home = Path(hermes_home).expanduser().resolve()
        self._base_dir = self._hermes_home / "memory_v2"
        self._store = MemoryV2Store(self._base_dir)
        self._index = MemoryV2Index(self._base_dir / "indexes" / "memory.sqlite")

        self._store.initialize()
        self._index.initialize()
        self._initialized = True

    def system_prompt_block(self) -> str:
        """Return small stable core memory suitable for prompt caching."""
        records = self.store.list_core_memory_records() if self._store is not None else []
        if not records:
            return (
                "Memory v2 provider is available: use routed, source-grounded recall "
                "when relevant; avoid treating summaries as evidence without sources."
            )
        lines = [
            "Memory v2 core memory (curated, source-grounded, high-confidence; treat as durable but updateable):"
        ]
        for record in records[:12]:
            sources = ",".join(record.source_refs[:3]) if record.source_refs else "none"
            lines.append(
                f"- [{record.category.value} p={record.priority:.2f} c={record.confidence:.2f} source_refs={sources}] {record.statement}"
            )
        return "\n".join(lines)[:1200]

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return a bounded routed memory packet when indexed recall is relevant."""
        composer = MemoryPacketComposer(self.index)
        packet = composer.compose(query)
        rendered = composer.render(packet)
        working = self._working_prefetch_block(query, session_id=session_id)
        if working and rendered:
            return f"{working}\n---\n{rendered}"
        return working or rendered

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        """Refresh current working-memory focus for the active turn."""
        if self._agent_context != "primary":
            return
        focus = {
            "turn_number": int(turn_number),
            "current_user_message": self._redact_sensitive_text(str(message or "").strip()),
            "platform": self._platform,
        }
        for key in ("model", "remaining_tokens", "tool_count"):
            if key in kwargs and kwargs[key] not in (None, ""):
                focus[key] = kwargs[key]
        existing = self.store.read_current_working_memory()
        scratchpad = existing.scratchpad if existing else {}
        self.store.write_current_working_memory(
            WorkingMemory(session_id=self._session_id, focus=focus, scratchpad=scratchpad)
        )

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs: Any,
    ) -> None:
        """Keep provider-local session state aligned with Hermes session switches."""
        self._session_id = str(new_session_id or "")

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Persist a completed turn as raw evidence and conservative candidates.

        v0 intentionally avoids durable promotion. It records the exchange as a
        raw event and only creates a pending candidate for explicit user memory
        requests such as "remember that ...".
        """
        user_text = self._redact_sensitive_text(str(user_content or "").strip())
        assistant_text = self._redact_sensitive_text(str(assistant_content or "").strip())
        if self._agent_context != "primary":
            return
        if not user_text or not assistant_text:
            return

        event = self.store.append_raw_event(
            {
                "type": "turn",
                "session_id": session_id or self._session_id,
                "provider_session_id": self._session_id,
                "platform": self._platform,
                "user_content": user_text,
                "assistant_content": assistant_text,
            }
        )
        self.index.index_raw_event(event)
        if source := self.store.read_source_ref(str(event["id"])):
            self.index.index_source_ref(source)

        candidate = self._candidate_from_turn(user_text, event_id=str(event["id"]))
        if candidate is not None:
            if self._candidate_is_obvious_redacted_secret(candidate):
                candidate = self._candidate_with_decision(
                    candidate,
                    GateDecision.ARCHIVED_ONLY,
                    "Archived automatically: obvious redacted secret candidate; raw evidence retained without pending promotion.",
                )
            duplicate = self._find_duplicate_candidate(candidate)
            if duplicate is None:
                self.store.append_candidate(candidate)
                self.index.index_candidate(candidate)
            else:
                merged_refs = list(duplicate.source_refs)
                for source_ref in candidate.source_refs:
                    if source_ref not in merged_refs:
                        merged_refs.append(source_ref)
                if merged_refs != list(duplicate.source_refs):
                    duplicate.source_refs = merged_refs
                    updated_candidates = [duplicate if existing.id == duplicate.id else existing for existing in self.store.list_candidates()]
                    self.store.rewrite_candidates(updated_candidates)
                    self.index.index_candidate(duplicate)
                candidate = duplicate
        self._update_working_after_turn(user_text, assistant_text, event_id=str(event["id"]), candidate=candidate)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            STATUS_SCHEMA,
            SEARCH_SCHEMA,
            CONSOLIDATE_SCHEMA,
            DAILY_REPORT_SCHEMA,
            CANDIDATES_SCHEMA,
            PROMOTE_SCHEMA,
            REJECT_SCHEMA,
            SHOW_SOURCE_SCHEMA,
            RESOLVE_OPEN_LOOP_SCHEMA,
            CONTRADICTIONS_SCHEMA,
        ]

    def _working_prefetch_block(self, query: str, *, session_id: str = "") -> str:
        lowered = str(query or "").lower()
        wants_working = any(term in lowered for term in ("open loop", "open loops", "pending", "what next", "next action", "current", "working on", "left off"))
        if not wants_working:
            return ""
        current = self.store.read_current_working_memory()
        loops = self.store.list_open_loops(status="open")
        if not current and not loops:
            return ""
        payload: Dict[str, Any] = {
            "note": "Working-memory packet is mutable short-term state: use as current context, not durable fact.",
            "route": "current_task",
        }
        if current:
            payload["working_current"] = current.to_dict()
        if loops:
            payload["working_open_loops"] = loops[:10]
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)

    def _update_working_after_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        event_id: str,
        candidate: CandidateMemory | None,
    ) -> None:
        current = self.store.read_current_working_memory()
        focus = dict(current.focus if current else {})
        focus.update(
            {
                "last_user_message": user_text,
                "last_assistant_message": assistant_text,
                "last_event_id": event_id,
                "platform": self._platform,
            }
        )
        scratchpad = dict(current.scratchpad if current else {})
        retrieved_ids = list(scratchpad.get("retrieved_memory_ids") or [])
        retrieved_ids.append(candidate.id if candidate else event_id)
        scratchpad["retrieved_memory_ids"] = retrieved_ids[-20:]
        self.store.write_current_working_memory(
            WorkingMemory(session_id=self._session_id, focus=focus, scratchpad=scratchpad)
        )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._agent_context != "primary":
            return
        self.store.archive_working_session(session_id=self._session_id, messages=messages)

    @staticmethod
    def _redact_sensitive_text(text: str) -> str:
        return redact_text(text)

    def _candidate_from_turn(self, user_text: str, *, event_id: str) -> Optional[CandidateMemory]:
        decision = RuleBasedWriteGate().classify(user_text)
        if not decision.should_create_candidate:
            return None
        reason = decision.reason
        if not reason.lower().startswith(f"{decision.outcome.value}:"):
            reason = f"{decision.outcome.value}: {reason}"
        return CandidateMemory(
            id=f"cand_{uuid.uuid4().hex}",
            type=decision.memory_type,
            claim=decision.claim,
            proposed_destination=decision.proposed_destination,
            confidence=decision.confidence,
            importance=decision.importance,
            promotion_reason=reason,
            source_refs=[event_id],
        )

    @staticmethod
    def _candidate_dedupe_key(candidate: CandidateMemory) -> tuple[str, str, str]:
        normalized_claim = re.sub(r"[^a-z0-9\[\] ]+", " ", candidate.claim.lower())
        normalized_claim = re.sub(r"\s+", " ", normalized_claim).strip()
        candidate_type = getattr(candidate.type, "value", str(candidate.type))
        return (candidate_type, candidate.proposed_destination.strip().lower(), normalized_claim)

    def _find_duplicate_candidate(self, candidate: CandidateMemory) -> CandidateMemory | None:
        candidate_key = self._candidate_dedupe_key(candidate)
        for existing in self.store.list_candidates():
            if existing.gate_decision in {GateDecision.REJECTED, GateDecision.SUPERSEDED}:
                continue
            if self._candidate_dedupe_key(existing) == candidate_key:
                return existing
        return None

    @staticmethod
    def _candidate_is_obvious_redacted_secret(candidate: CandidateMemory) -> bool:
        claim = candidate.claim.lower()
        if "[redacted]" not in claim:
            return False
        secret_terms = (
            "password",
            "passwd",
            "token",
            "secret",
            "api key",
            "api_key",
            "api-key",
            "authorization",
            "bearer",
            "credential",
            "private key",
            "client secret",
        )
        return any(term in claim for term in secret_terms)

    @staticmethod
    def _looks_like_environment_claim(claim: str) -> bool:
        lowered = claim.lower()
        return any(term in lowered for term in ("hermes is running", "host", "wsl", "macos", "linux", "windows", "environment"))

    @staticmethod
    def _looks_like_contradiction_claim(user_text: str) -> bool:
        lowered = user_text.lower()
        return lowered.startswith("remember that") and " not " in lowered

    @staticmethod
    def _extract_explicit_memory_claim(user_text: str) -> str:
        patterns = [
            r"^\s*remember\s+that\s+(.+)$",
            r"^\s*please\s+remember\s+that\s+(.+)$",
            r"^\s*don't\s+forget\s+that\s+(.+)$",
            r"^\s*do\s+not\s+forget\s+that\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.match(pattern, user_text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if tool_name == "memory_v2_status":
            return json.dumps(self._status_payload())
        if tool_name == "memory_v2_search":
            query = str(args.get("query") or "")
            try:
                limit = max(1, min(int(args.get("limit") or 10), 50))
            except (TypeError, ValueError):
                return json.dumps({"success": False, "error": "limit must be an integer"})
            results = self.index.search(query, limit=limit)
            return json.dumps({"success": True, "count": len(results), "results": results})
        if tool_name == "memory_v2_consolidate":
            report = RuleBasedConsolidator().consolidate(self.store, self.index)
            return json.dumps({"success": True, **report.to_dict()})
        if tool_name == "memory_v2_daily_report":
            try:
                report = run_daily_consolidation_report(self.store, self.index, date=args.get("date"))
            except ValueError as exc:
                return json.dumps({"success": False, "error": str(exc)})
            return json.dumps(report)
        if tool_name == "memory_v2_candidates":
            return json.dumps(self._candidates_payload(args))
        if tool_name == "memory_v2_reject":
            return json.dumps(self._reject_candidate(args))
        if tool_name == "memory_v2_promote":
            return json.dumps(self._promote_candidate(args))
        if tool_name == "memory_v2_show_source":
            return json.dumps(self._show_source(args))
        if tool_name == "memory_v2_resolve_open_loop":
            return json.dumps(self._resolve_open_loop(args))
        if tool_name == "memory_v2_contradictions":
            return json.dumps(self._contradictions_payload(args))
        return json.dumps({"success": False, "error": f"Unknown Memory v2 tool: {tool_name}"})

    def _contradictions_payload(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            limit = max(1, min(int(args.get("limit") or 50), 200))
        except (TypeError, ValueError):
            return {"success": False, "error": "limit must be an integer"}
        create_candidates = bool(args.get("create_candidates") or False)
        auto_supersede = bool(args.get("auto_supersede") or False)
        try:
            min_confidence = float(args.get("min_confidence") or 0.9)
        except (TypeError, ValueError):
            return {"success": False, "error": "min_confidence must be a number"}
        min_confidence = max(0.0, min(min_confidence, 1.0))
        conflicts = self._detect_memory_conflicts(limit=limit)
        auto_superseded: List[Dict[str, Any]] = []
        if auto_supersede:
            auto_superseded = self._apply_auto_supersessions(conflicts, min_confidence=min_confidence)
        auto_superseded_ids = {item["superseded_id"] for item in auto_superseded}
        created_candidate_ids: List[str] = []
        if create_candidates:
            for conflict in conflicts:
                if conflict.get("proposed_superseded_id") in auto_superseded_ids:
                    continue
                if conflict.get("proposed_action") != "manual_review_supersession_candidate":
                    continue
                candidate = self._candidate_from_conflict(conflict)
                if candidate is None:
                    continue
                duplicate = self._find_duplicate_candidate(candidate)
                if duplicate is not None:
                    continue
                self.store.append_candidate(candidate)
                self.index.index_candidate(candidate)
                created_candidate_ids.append(candidate.id)
        return {
            "success": True,
            "mode": "dashboard_and_candidate_generator",
            "note": "Dashboard by default; automatic supersession only runs when auto_supersede=true and high-confidence source gates pass.",
            "mutated_memories": len(auto_superseded),
            "create_candidates": create_candidates,
            "auto_supersede": auto_supersede,
            "min_confidence": min_confidence,
            "count": len(conflicts),
            "created_candidate_ids": created_candidate_ids,
            "auto_superseded": auto_superseded,
            "conflicts": conflicts,
        }

    def _detect_memory_conflicts(self, *, limit: int) -> List[Dict[str, Any]]:
        active_items = self.store.list_memory_items(status="active")
        grouped: Dict[tuple[str, str, str], List[MemoryItem]] = {}
        for item in active_items:
            item_type = getattr(item.type, "value", str(item.type))
            if item_type not in {"preference", "fact", "environment", "constraint"}:
                continue
            predicate = str(item.predicate or "").strip()
            if not predicate:
                continue
            key = (
                item_type,
                self._normalize_conflict_text(item.subject),
                self._normalize_conflict_text(predicate),
            )
            grouped.setdefault(key, []).append(item)
        conflicts: List[Dict[str, Any]] = []
        for (item_type, subject_key, predicate_key), items in grouped.items():
            if len(items) < 2:
                continue
            sorted_items = sorted(items, key=lambda item: (str(item.updated_at or item.created_at or ""), item.id))
            for index, first in enumerate(sorted_items):
                for second in sorted_items[index + 1 :]:
                    first_value = self._memory_item_value(first)
                    second_value = self._memory_item_value(second)
                    if self._normalize_conflict_text(first_value) == self._normalize_conflict_text(second_value):
                        continue
                    conflict = self._conflict_payload(
                        first,
                        second,
                        item_type=item_type,
                        subject_key=subject_key,
                        predicate_key=predicate_key,
                    )
                    if conflict is not None:
                        conflicts.append(conflict)
                    if len(conflicts) >= limit:
                        return conflicts
        return conflicts

    def _conflict_payload(
        self,
        first: MemoryItem,
        second: MemoryItem,
        *,
        item_type: str,
        subject_key: str,
        predicate_key: str,
    ) -> Dict[str, Any] | None:
        classification = self._classify_conflict(first, second)
        older, newer = self._older_newer_memory(first, second)
        proposed_action = "manual_review_possible_conflict"
        proposed_superseded_id = ""
        proposed_superseded_by = ""
        if classification == "scope_difference":
            proposed_action = "keep_both_scoped"
        elif classification in {"true_contradiction", "preference_update"}:
            proposed_action = "manual_review_supersession_candidate"
            proposed_superseded_id = older.id
            proposed_superseded_by = newer.id
        source_refs = self._combined_source_refs(first, second)
        payload = {
            "id": f"conflict_{first.id}_{second.id}",
            "type": "contradiction_candidate",
            "classification": classification,
            "proposed_action": proposed_action,
            "subject": first.subject,
            "predicate": first.predicate or "",
            "group_key": {"type": item_type, "subject": subject_key, "predicate": predicate_key},
            "memory_a": self._compact_conflict_memory(first),
            "memory_b": self._compact_conflict_memory(second),
            "proposed_superseded_id": proposed_superseded_id,
            "proposed_superseded_by": proposed_superseded_by,
            "reason": self._conflict_reason(first, second, classification),
            "source_refs": source_refs,
            "sources": [source for source_id in source_refs if (source := self._source_payload(source_id)) is not None],
            "confidence": self._conflict_confidence(classification, bool(source_refs)),
        }
        eligible, blockers = self._auto_supersede_gate(payload)
        payload["auto_supersede_eligible"] = eligible
        payload["auto_supersede_blockers"] = blockers
        return payload

    def _apply_auto_supersessions(self, conflicts: List[Dict[str, Any]], *, min_confidence: float) -> List[Dict[str, Any]]:
        applied: List[Dict[str, Any]] = []
        already_superseded: set[str] = set()
        for conflict in conflicts:
            if float(conflict.get("confidence") or 0.0) < min_confidence:
                conflict.setdefault("auto_supersede_blockers", []).append(f"confidence below min_confidence {min_confidence:.2f}")
                conflict["auto_supersede_eligible"] = False
                continue
            eligible, blockers = self._auto_supersede_gate(conflict)
            conflict["auto_supersede_eligible"] = eligible
            conflict["auto_supersede_blockers"] = blockers
            if not eligible:
                continue
            old_id = str(conflict.get("proposed_superseded_id") or "")
            new_id = str(conflict.get("proposed_superseded_by") or "")
            if not old_id or not new_id or old_id in already_superseded:
                continue
            old_item = self.store.read_memory_item(old_id)
            new_item = self.store.read_memory_item(new_id)
            if old_item is None or new_item is None:
                continue
            now = utc_now_iso()
            reason = (
                "Automatic high-confidence supersession: newer explicit correction from source evidence; "
                f"conflict={conflict.get('id')}; classification={conflict.get('classification')}; "
                f"superseded_by={new_id}."
            )
            old_item.status = "superseded"
            old_item.superseded_by = new_id
            old_item.superseded_at = now
            old_item.supersession_reason = reason
            old_item.updated_at = now
            if "auto_superseded" not in old_item.tags:
                old_item.tags.append("auto_superseded")
            if old_id not in new_item.supersedes:
                new_item.supersedes.append(old_id)
            new_item.updated_at = now
            old_path = self.store.write_memory_item(old_item)
            new_path = self.store.write_memory_item(new_item)
            self.index.index_memory_item(old_item, file_path=old_path)
            self.index.index_memory_item(new_item, file_path=new_path)
            applied.append(
                {
                    "conflict_id": conflict.get("id"),
                    "superseded_id": old_id,
                    "superseded_by": new_id,
                    "reason": reason,
                    "superseded_at": now,
                }
            )
            already_superseded.add(old_id)
        return applied

    def _auto_supersede_gate(self, conflict: Dict[str, Any]) -> tuple[bool, List[str]]:
        blockers: List[str] = []
        if conflict.get("proposed_action") != "manual_review_supersession_candidate":
            blockers.append("not a supersession candidate")
        if conflict.get("classification") not in {"true_contradiction", "preference_update"}:
            blockers.append("classification is not auto-supersedable")
        old_id = str(conflict.get("proposed_superseded_id") or "")
        new_id = str(conflict.get("proposed_superseded_by") or "")
        if not old_id or not new_id:
            blockers.append("missing supersession target ids")
        memory_a = conflict.get("memory_a") or {}
        memory_b = conflict.get("memory_b") or {}
        old_payload = memory_a if memory_a.get("id") == old_id else memory_b if memory_b.get("id") == old_id else {}
        new_payload = memory_a if memory_a.get("id") == new_id else memory_b if memory_b.get("id") == new_id else {}
        old_sources = self._resolved_sources_for_conflict_memory(old_payload)
        new_sources = self._resolved_sources_for_conflict_memory(new_payload)
        if not old_payload.get("source_refs") or not new_payload.get("source_refs"):
            blockers.append("both memories need source refs")
        if not old_sources or not new_sources:
            blockers.append("both memories need resolvable source evidence")
        if self._looks_like_scoped_pair(
            self._normalize_conflict_text(old_payload.get("value") or ""),
            self._normalize_conflict_text(new_payload.get("value") or ""),
        ):
            blockers.append("scoped wording")
        if not self._has_explicit_newer_correction(new_sources):
            blockers.append("explicit newer correction")
        return not blockers, blockers

    def _resolved_sources_for_conflict_memory(self, memory_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for source_id in memory_payload.get("source_refs") or []:
            source = self._source_payload(str(source_id))
            if source is not None:
                sources.append(source)
        return sources

    @staticmethod
    def _has_explicit_newer_correction(sources: List[Dict[str, Any]]) -> bool:
        source_text_parts: List[str] = []
        for source in sources:
            source_text_parts.extend(str(source.get(field) or "") for field in ("title", "quote", "uri"))
            maybe_event = source.get("event")
            event = maybe_event if isinstance(maybe_event, dict) else {}
            source_text_parts.extend(str(event.get(field) or "") for field in ("user_content", "assistant_content"))
        combined = " ".join(source_text_parts).lower()
        correction_terms = (
            "corrected",
            "correction",
            "explicit correction",
            "instead",
            "not ",
            "no longer",
            "now",
            "new preference",
            "current preference",
            "replaces",
            "supersedes",
        )
        return any(term in combined for term in correction_terms)

    def _candidate_from_conflict(self, conflict: Dict[str, Any]) -> CandidateMemory | None:
        memory_a = conflict.get("memory_a") or {}
        memory_b = conflict.get("memory_b") or {}
        old_id = str(conflict.get("proposed_superseded_id") or "")
        new_id = str(conflict.get("proposed_superseded_by") or "")
        if not old_id or not new_id:
            return None
        claim = (
            f"Review possible Memory v2 supersession: supersede {old_id} with {new_id}. "
            f"Conflict between {memory_a.get('id')}={memory_a.get('value')!r} and "
            f"{memory_b.get('id')}={memory_b.get('value')!r}."
        )
        return CandidateMemory(
            id=f"cand_conflict_{uuid.uuid4().hex}",
            type="fact",
            claim=claim,
            proposed_destination="review/contradictions",
            confidence=float(conflict.get("confidence") or 0.7),
            importance=0.8,
            promotion_reason=(
                "dashboard_only: contradiction/supersession candidate generated for manual review; "
                f"classification={conflict.get('classification')}; reason={conflict.get('reason')}"
            ),
            source_refs=list(conflict.get("source_refs") or []),
        )

    @staticmethod
    def _normalize_conflict_text(value: Any) -> str:
        normalized = re.sub(r"[^a-z0-9:./+-]+", " ", str(value or "").lower())
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _memory_item_value(item: MemoryItem) -> str:
        return str(item.value or item.summary or item.body or "")

    @staticmethod
    def _older_newer_memory(first: MemoryItem, second: MemoryItem) -> tuple[MemoryItem, MemoryItem]:
        first_time = str(first.updated_at or first.created_at or "")
        second_time = str(second.updated_at or second.created_at or "")
        if (first_time, first.id) <= (second_time, second.id):
            return first, second
        return second, first

    @staticmethod
    def _combined_source_refs(first: MemoryItem, second: MemoryItem) -> List[str]:
        refs: List[str] = []
        for ref in list(first.source_refs) + list(second.source_refs):
            if ref not in refs:
                refs.append(ref)
        return refs

    def _compact_conflict_memory(self, item: MemoryItem) -> Dict[str, Any]:
        return {
            "id": item.id,
            "type": getattr(item.type, "value", str(item.type)),
            "subject": item.subject,
            "predicate": item.predicate,
            "value": self._memory_item_value(item),
            "status": getattr(item.status, "value", str(item.status)),
            "created_at": item.created_at,
            "updated_at": item.updated_at,
            "source_refs": list(item.source_refs),
        }

    def _classify_conflict(self, first: MemoryItem, second: MemoryItem) -> str:
        first_value = self._normalize_conflict_text(self._memory_item_value(first))
        second_value = self._normalize_conflict_text(self._memory_item_value(second))
        if self._looks_like_scoped_pair(first_value, second_value):
            return "scope_difference"
        item_type = getattr(first.type, "value", str(first.type))
        if item_type == "preference":
            return "preference_update" if self._looks_like_update_pair(first, second) else "possible_conflict"
        return "true_contradiction"

    @staticmethod
    def _looks_like_scoped_pair(first_value: str, second_value: str) -> bool:
        scoped_terms = {"default", "usually", "simple", "short", "concise", "complex", "detailed", "architecture", "research"}
        return any(term in first_value for term in scoped_terms) and any(term in second_value for term in scoped_terms)

    @staticmethod
    def _looks_like_update_pair(first: MemoryItem, second: MemoryItem) -> bool:
        combined = f"{first.value or ''} {second.value or ''}".lower()
        return any(term in combined for term in ("previously", "formerly", "old", "new", "current", "now", "instead"))

    def _conflict_reason(self, first: MemoryItem, second: MemoryItem, classification: str) -> str:
        first_value = self._memory_item_value(first)
        second_value = self._memory_item_value(second)
        if classification == "scope_difference":
            return "Same subject/predicate has different values, but wording suggests scoped preferences rather than a direct contradiction."
        older, newer = self._older_newer_memory(first, second)
        if classification in {"true_contradiction", "preference_update"}:
            return f"Same subject/predicate has mutually different active values; newer record {newer.id} may supersede older record {older.id}."
        return f"Same subject/predicate has different active values requiring manual review: {first_value!r} vs {second_value!r}."

    @staticmethod
    def _conflict_confidence(classification: str, has_sources: bool) -> float:
        base = {
            "true_contradiction": 0.82,
            "preference_update": 0.76,
            "possible_conflict": 0.62,
            "scope_difference": 0.45,
        }.get(classification, 0.5)
        return round(min(0.95, base + (0.08 if has_sources else 0.0)), 2)

    def _candidates_payload(self, args: Dict[str, Any]) -> Dict[str, Any]:
        type_filter = str(args.get("type") or "").strip()
        status_filter = str(args.get("status") or "").strip()
        try:
            limit = max(1, min(int(args.get("limit") or 100), 500))
        except (TypeError, ValueError):
            return {"success": False, "error": "limit must be an integer"}
        candidates = []
        for candidate in self.store.list_candidates():
            payload = candidate.to_dict()
            if type_filter and payload["type"] != type_filter:
                continue
            if status_filter and payload["gate_decision"] != status_filter:
                continue
            candidates.append(payload)
        return {"success": True, "count": len(candidates[:limit]), "candidates": candidates[:limit]}

    def _reject_candidate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        candidate_id = str(args.get("candidate_id") or "").strip()
        reason = str(args.get("reason") or "").strip()
        if not candidate_id:
            return {"success": False, "error": "candidate_id is required"}
        if not reason:
            return {"success": False, "error": "reason is required"}
        candidates = self.store.list_candidates()
        updated: List[CandidateMemory] = []
        rejected_candidate: CandidateMemory | None = None
        for candidate in candidates:
            if candidate.id == candidate_id:
                if candidate.gate_decision == GateDecision.REJECTED:
                    rejected_candidate = candidate
                    updated.append(candidate)
                else:
                    rejected_candidate = self._candidate_with_decision(candidate, GateDecision.REJECTED, reason)
                    updated.append(rejected_candidate)
            else:
                updated.append(candidate)
        if rejected_candidate is None:
            return {"success": False, "error": f"candidate not found: {candidate_id}"}
        already_rejected = rejected_candidate.gate_decision == GateDecision.REJECTED and any(
            candidate.id == candidate_id and candidate.gate_decision == GateDecision.REJECTED for candidate in candidates
        )
        if already_rejected:
            return {"success": True, "already_rejected": True, "candidate": rejected_candidate.to_dict()}
        self.store.rewrite_candidates(updated)
        self.store.append_rejected_candidate(rejected_candidate)
        self.index.index_candidate(rejected_candidate)
        return {"success": True, "candidate": rejected_candidate.to_dict()}

    def _promote_candidate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        candidate_id = str(args.get("candidate_id") or "").strip()
        force = bool(args.get("force") or False)
        if not candidate_id:
            return {"success": False, "error": "candidate_id is required"}
        candidates = self.store.list_candidates()
        target = next((candidate for candidate in candidates if candidate.id == candidate_id), None)
        if target is None:
            return {"success": False, "error": f"candidate not found: {candidate_id}"}
        if target.proposed_destination.strip().lower() == "skills" or str(getattr(target.type, "value", target.type)) == "procedure_ref":
            return {"success": False, "error": "procedure/skills candidates require skill authoring or manual rejection, not semantic promotion"}
        if target.gate_decision != GateDecision.PENDING:
            return {"success": False, "error": f"candidate is already {target.gate_decision.value}; only pending candidates can be promoted", "candidate": target.to_dict()}
        if force:
            force_reason = str(args.get("force_reason") or "").strip()
            if not force_reason:
                return {"success": False, "error": "force_reason is required when force=true"}
            valid_refs = [source_id for source_id in target.source_refs if self._source_payload(source_id) is not None]
            if valid_refs != list(target.source_refs):
                data = target.to_dict()
                data["source_refs"] = valid_refs
                data["confidence"] = min(float(data.get("confidence") or 0.0), 0.5)
                target = CandidateMemory.from_dict(data)
        if not force:
            if not target.source_refs:
                return {"success": False, "error": "candidate source_refs are required for manual promotion"}
            dangling = [source_id for source_id in target.source_refs if self._source_payload(source_id) is None]
            if dangling:
                return {"success": False, "error": f"candidate has dangling source_refs: {dangling}"}
        promoted_ids: List[str] = []
        consolidator = RuleBasedConsolidator()
        if consolidator._is_open_loop_candidate(target):
            existing_loop = next((loop for loop in self.store.list_open_loops() if loop.get("candidate_id") == target.id), None)
            loop = existing_loop or self.store.upsert_open_loop(
                {"text": target.claim, "source_refs": target.source_refs, "session_id": self._session_id, "candidate_id": target.id}
            )
            self.index.index_open_loop(loop, file_path=self.store.open_loops_path)
            updated_target = self._candidate_with_decision(
                target, GateDecision.ARCHIVED_ONLY, f"Manually routed to working/open_loops.yaml as {loop['id']}."
            )
            promoted_ids.append(loop["id"])
        elif consolidator._is_project_card_candidate(target):
            card = consolidator._merge_project_card(target, self.store)
            path = self.store.write_project_card(card)
            self.index.index_project_card(card, file_path=path)
            updated_target = self._candidate_with_decision(target, GateDecision.PROMOTED, f"Manually promoted to ProjectCard {card.id}.")
            promoted_ids.append(card.id)
        else:
            item = consolidator._memory_item_from_candidate(target)
            superseded = consolidator._superseded_items_for(target, item, self.store)
            if superseded:
                item.supersedes = [old.id for old in superseded]
                for old in superseded:
                    old.status = "superseded"
                    old.superseded_by = item.id
                    old.updated_at = utc_now_iso()
                    path = self.store.write_memory_item(old)
                    self.index.index_memory_item(old, file_path=path)
            path = self.store.write_memory_item(item)
            self.index.index_memory_item(item, file_path=path)
            updated_target = self._candidate_with_decision(target, GateDecision.PROMOTED, f"Manually promoted to MemoryItem {item.id}.")
            promoted_ids.append(item.id)
        updated_candidates = [updated_target if candidate.id == candidate_id else candidate for candidate in candidates]
        self.store.rewrite_candidates(updated_candidates)
        self.index.index_candidate(updated_target)
        return {"success": True, "promoted": 1, "promoted_ids": promoted_ids, "candidate": updated_target.to_dict()}

    def _show_source(self, args: Dict[str, Any]) -> Dict[str, Any]:
        record_id = str(args.get("id") or "").strip()
        if not record_id:
            return {"success": False, "error": "id is required"}
        record = self._record_payload(record_id)
        if record is None:
            source = self._source_payload(record_id)
            if source is None:
                return {"success": False, "error": f"record/source not found: {record_id}"}
            return {"success": True, "record": {"id": record_id, "type": "source_ref"}, "sources": [source]}
        source_refs = list(record.get("source_refs") or [])
        sources = [source for source_id in source_refs if (source := self._source_payload(source_id)) is not None]
        missing = [source_id for source_id in source_refs if self._source_payload(source_id) is None]
        return {"success": True, "record": record, "sources": sources, "missing_source_refs": missing}

    def _resolve_open_loop(self, args: Dict[str, Any]) -> Dict[str, Any]:
        loop_id = str(args.get("loop_id") or "").strip()
        status = str(args.get("status") or "").strip()
        resolution = str(args.get("resolution") or "").strip()
        allowed = {"open", "resolved", "abandoned", "blocked", "snoozed"}
        if not loop_id:
            return {"success": False, "error": "loop_id is required"}
        if status not in allowed:
            return {"success": False, "error": f"status must be one of: {sorted(allowed)}"}
        loops = self.store.list_open_loops()
        updated_loop: Dict[str, Any] | None = None
        for loop in loops:
            if loop.get("id") == loop_id:
                loop["status"] = status
                loop["updated_at"] = utc_now_iso()
                if resolution:
                    loop["resolution"] = resolution
                if status in {"resolved", "abandoned"}:
                    loop["resolved_at"] = utc_now_iso()
                updated_loop = loop
                break
        if updated_loop is None:
            return {"success": False, "error": f"open loop not found: {loop_id}"}
        self.store.write_open_loops(loops)
        return {"success": True, "loop": updated_loop}

    @staticmethod
    def _candidate_with_decision(candidate: CandidateMemory, decision: GateDecision, reason: str) -> CandidateMemory:
        data = candidate.to_dict()
        data["gate_decision"] = decision.value
        data["decision_reason"] = reason
        return CandidateMemory.from_dict(data)

    def _record_payload(self, record_id: str) -> Dict[str, Any] | None:
        if memory := self.store.read_memory_item(record_id):
            return memory.to_dict()
        if project := self.store.read_project_card(record_id):
            return project.to_dict()
        for candidate in self.store.list_candidates():
            if candidate.id == record_id:
                payload = candidate.to_dict()
                payload["type"] = "candidate"
                payload["candidate_memory_type"] = getattr(candidate.type, "value", str(candidate.type))
                return payload
        for loop in self.store.list_open_loops():
            if loop.get("id") == record_id:
                payload = dict(loop)
                payload.setdefault("type", "open_loop")
                return payload
        return None

    def _source_payload(self, source_id: str) -> Dict[str, Any] | None:
        source = self.store.read_source_ref(source_id)
        if source is not None:
            return source.to_dict()
        for event in self.store.read_raw_events():
            if str(event.get("id") or "") == str(source_id):
                safe_event = redact_data(event)
                quote = str(safe_event.get("user_content") or safe_event.get("content") or safe_event.get("assistant_content") or "")
                if len(quote) > 500:
                    quote = quote[:497].rstrip() + "..."
                return {
                    "id": str(source_id),
                    "type": "raw_event",
                    "uri": f"raw_event:{source_id}",
                    "quote": quote,
                    "observed_at": str(safe_event.get("created_at") or ""),
                }
        return None

    def _status_payload(self) -> Dict[str, Any]:
        base = self.base_dir
        return {
            "success": True,
            "provider": self.name,
            "initialized": self._initialized,
            "session_id": self._session_id,
            "platform": self._platform,
            "base_dir": base.name,
            "counts": {
                "raw_events": self.store.count_raw_events(),
                "pending_candidates": self.store.count_pending_candidates(),
                "rejected_candidates": self.store.count_rejected_candidates(),
                "core_records": len(self.store.list_core_memory_records()),
                "memory_items": len(self.store.list_memory_items()),
                "indexed_memories": self.index.count_memories(),
            },
        }


def register(ctx: Any) -> None:
    """Plugin registration hook used by Hermes memory provider discovery."""
    ctx.register_memory_provider(MemoryV2Provider())
