"""Memory v2 provider skeleton.

Local, profile-scoped memory provider intended to grow into the Memory v2
architecture described in ``docs/plans/memory-v2-spec.md``.  This initial
skeleton deliberately avoids network calls and only establishes the provider
identity, profile-scoped directory tree, a small stable system-prompt block,
empty recall behavior, and a status tool.
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
from .index import MemoryV2Index
from .retrieval import MemoryPacketComposer
from .schemas import CandidateMemory, GateDecision, WorkingMemory, utc_now_iso
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


class MemoryV2Provider(MemoryProvider):
    """Local profile-scoped Memory v2 provider skeleton."""

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
            if self._find_duplicate_candidate(candidate) is None:
                self.store.append_candidate(candidate)
                self.index.index_candidate(candidate)
            else:
                candidate = None
        self._update_working_after_turn(user_text, assistant_text, event_id=str(event["id"]), candidate=candidate)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            STATUS_SCHEMA,
            SEARCH_SCHEMA,
            CONSOLIDATE_SCHEMA,
            CANDIDATES_SCHEMA,
            PROMOTE_SCHEMA,
            REJECT_SCHEMA,
            SHOW_SOURCE_SCHEMA,
            RESOLVE_OPEN_LOOP_SCHEMA,
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
        redacted = text
        patterns = [
            r"(?i)(password\s*(?:is|=|:)\s*)\S+",
            r"(?i)(password\s+)\S+",
            r"(?i)(passwd\s*(?:is|=|:)\s*)\S+",
            r"(?i)(passwd\s+)\S+",
            r"(?i)(token\s*(?:is|=|:)\s*)\S+",
            r"(?i)(token\s+)\S+",
            r"(?i)(secret\s*(?:is|=|:)\s*)\S+",
            r"(?i)(secret\s+)\S+",
            r"(?i)(api[_ -]?key\s*(?:is|=|:)\s*)\S+",
            r"(?i)(api[_ -]?key\s+)\S+",
            r"(?i)(authorization\s*:\s*bearer\s+)\S+",
            r"(?i)(bearer\s+)\S+",
            r"(?i)([A-Z0-9_]*API[_-]?KEY\s*=\s*)\S+",
            r"(?i)([A-Z0-9_]*API[_-]?KEY\s+)\S+",
        ]
        for pattern in patterns:
            redacted = re.sub(pattern, lambda match: f"{match.group(1)}[REDACTED]", redacted)
        return redacted

    def _candidate_from_turn(self, user_text: str, *, event_id: str) -> Optional[CandidateMemory]:
        decision = RuleBasedWriteGate().classify(user_text)
        if not decision.should_create_candidate:
            return None
        return CandidateMemory(
            id=f"cand_{uuid.uuid4().hex}",
            type=decision.memory_type,
            claim=decision.claim,
            proposed_destination=decision.proposed_destination,
            confidence=decision.confidence,
            importance=decision.importance,
            promotion_reason=f"{decision.outcome.value}: {decision.reason}",
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
        return json.dumps({"success": False, "error": f"Unknown Memory v2 tool: {tool_name}"})

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
                rejected_candidate = self._candidate_with_decision(candidate, GateDecision.REJECTED, reason)
                updated.append(rejected_candidate)
            else:
                updated.append(candidate)
        if rejected_candidate is None:
            return {"success": False, "error": f"candidate not found: {candidate_id}"}
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
        if not force:
            if not target.source_refs:
                return {"success": False, "error": "candidate source_refs are required for manual promotion"}
            dangling = [source_id for source_id in target.source_refs if self._source_payload(source_id) is None]
            if dangling:
                return {"success": False, "error": f"candidate has dangling source_refs: {dangling}"}
        promoted_ids: List[str] = []
        consolidator = RuleBasedConsolidator()
        if consolidator._is_open_loop_candidate(target):
            loop = self.store.upsert_open_loop(
                {"text": target.claim, "source_refs": target.source_refs, "session_id": self._session_id, "candidate_id": target.id}
            )
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
                return {"id": str(source_id), "type": "raw_event", "uri": f"raw_event:{source_id}", "event": event}
        return None

    def _status_payload(self) -> Dict[str, Any]:
        base = self.base_dir
        return {
            "success": True,
            "provider": self.name,
            "initialized": self._initialized,
            "session_id": self._session_id,
            "platform": self._platform,
            "base_dir": str(base),
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
