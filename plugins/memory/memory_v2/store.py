"""Profile-scoped file store for Memory v2 records.

The store owns Memory v2's human-readable canonical files and append-only JSONL
inbox files. Indexes are intentionally out of scope here; this layer only
provides safe local persistence for raw events, candidates, and project cards.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .redaction import redact_data, redact_text
from .schemas import CandidateMemory, CoreMemoryRecord, GateDecision, MemoryItem, ProjectCard, SourceRef, ValidationError, WorkingMemory, normalize_project_id, utc_now_iso


MEMORY_V2_DIRS = [
    "working",
    "core",
    "sources",
    "inbox",
    "semantic",
    "semantic/items",
    "semantic/projects",
    "semantic/environment",
    "episodic",
    "episodic/daily",
    "episodic/sessions",
    "graph",
    "indexes",
    "indexes/vector",
    "evals",
    "reports",
    "reports/daily_consolidation",
    "reports/weekly_reflection",
]


class MemoryV2Store:
    """Small local file store rooted at ``{hermes_home}/memory_v2``."""

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()

    @property
    def inbox_dir(self) -> Path:
        return self.base_dir / "inbox"

    @property
    def core_dir(self) -> Path:
        return self.base_dir / "core"

    @property
    def projects_dir(self) -> Path:
        return self.base_dir / "semantic" / "projects"

    @property
    def memory_items_dir(self) -> Path:
        return self.base_dir / "semantic" / "items"

    @property
    def sources_dir(self) -> Path:
        return self.base_dir / "sources"

    @property
    def working_dir(self) -> Path:
        return self.base_dir / "working"

    @property
    def episodic_sessions_dir(self) -> Path:
        return self.base_dir / "episodic" / "sessions"

    @property
    def current_working_path(self) -> Path:
        return self.working_dir / "current.yaml"

    @property
    def open_loops_path(self) -> Path:
        return self.working_dir / "open_loops.yaml"

    @property
    def raw_events_path(self) -> Path:
        return self.inbox_dir / "raw_events.jsonl"

    @property
    def candidates_path(self) -> Path:
        return self.inbox_dir / "candidates.jsonl"

    @property
    def rejected_path(self) -> Path:
        return self.inbox_dir / "rejected.jsonl"

    def initialize(self) -> None:
        """Create the Memory v2 profile-scoped directory tree and seed files."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for rel in MEMORY_V2_DIRS:
            (self.base_dir / rel).mkdir(parents=True, exist_ok=True)

        readme = self.base_dir / "README.md"
        if not readme.exists():
            self._atomic_write_text(
                readme,
                "# Memory v2\n\n"
                "Profile-scoped local memory store for Hermes. Canonical records "
                "live in human-readable files; indexes are derived and rebuildable.\n",
            )

        config = self.base_dir / "config.yaml"
        if not config.exists():
            self._atomic_write_text(
                config,
                "version: 1\n"
                "online:\n"
                "  default_packet_budget_tokens: 1500\n"
                "embeddings:\n"
                "  enabled: false\n"
                "graph:\n"
                "  enabled: true\n",
            )

        for path in (self.raw_events_path, self.candidates_path, self.rejected_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)

    def append_raw_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Append a raw event dict to ``inbox/raw_events.jsonl``.

        Adds ``id`` and ``created_at`` if missing and returns the persisted event.
        """
        if not isinstance(event, dict):
            raise ValidationError("raw event must be a JSON object")
        payload = redact_data(dict(event))
        payload.setdefault("id", f"event_{uuid.uuid4().hex}")
        payload.setdefault("created_at", utc_now_iso())
        self._append_jsonl(self.raw_events_path, payload)
        self.write_source_ref(self._source_ref_from_raw_event(payload))
        return payload

    def read_raw_events(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        events = self._read_jsonl(self.raw_events_path)
        if limit is None:
            return events
        safe_limit = int(limit)
        if safe_limit < 0:
            raise ValidationError("limit must be non-negative")
        if safe_limit == 0:
            return []
        return events[-safe_limit:]

    def count_raw_events(self) -> int:
        return self._count_jsonl(self.raw_events_path)

    def append_candidate(self, candidate: CandidateMemory) -> None:
        self._append_jsonl(self.candidates_path, candidate.to_dict())

    def rewrite_candidates(self, candidates: List[CandidateMemory]) -> None:
        """Rewrite ``inbox/candidates.jsonl`` with updated candidate decisions."""
        lines = "".join(json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for candidate in candidates)
        self._atomic_write_text(self.candidates_path, lines)

    def list_candidates(self) -> List[CandidateMemory]:
        return [CandidateMemory.from_dict(item) for item in self._read_jsonl(self.candidates_path)]

    def count_pending_candidates(self) -> int:
        return sum(1 for candidate in self.list_candidates() if candidate.gate_decision == GateDecision.PENDING)

    def append_rejected_candidate(self, candidate: CandidateMemory) -> None:
        self._append_jsonl(self.rejected_path, candidate.to_dict())

    def list_rejected_candidates(self) -> List[CandidateMemory]:
        return [CandidateMemory.from_dict(item) for item in self._read_jsonl(self.rejected_path)]

    def count_rejected_candidates(self) -> int:
        return len(self.list_rejected_candidates())

    def write_core_memory_record(self, record: CoreMemoryRecord) -> Path:
        """Write a formal core-memory record into ``core/<category>.yaml``."""
        records = [existing for existing in self.list_core_memory_records(category=record.category.value) if existing.id != record.id]
        records.append(record)
        records.sort(key=lambda item: (-item.priority, item.id))
        path = self._core_category_path(record.category.value)
        payload = {"version": 1, "category": record.category.value, "records": [item.to_dict() for item in records]}
        self._atomic_write_yaml(path, payload)
        return path

    def read_core_memory_record(self, record_id: str) -> Optional[CoreMemoryRecord]:
        for record in self.list_core_memory_records():
            if record.id == record_id:
                return record
        return None

    def list_core_memory_records(self, *, category: str | None = None) -> List[CoreMemoryRecord]:
        records: List[CoreMemoryRecord] = []
        if not self.core_dir.exists():
            return records
        paths = [self._core_category_path(category)] if category else sorted(self.core_dir.glob("*.yaml"))
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            for item in data.get("records") or []:
                records.append(CoreMemoryRecord.from_dict(item))
        records.sort(key=lambda item: (-item.priority, item.id))
        return records

    def write_project_card(self, card: ProjectCard) -> Path:
        """Write a project card to ``semantic/projects/<slug>.yaml`` atomically."""
        path = self._project_card_path(card.id)
        self._atomic_write_yaml(path, card.to_dict())
        return path

    def read_project_card(self, project_id_or_name: str) -> Optional[ProjectCard]:
        path = self._project_card_path(project_id_or_name)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return ProjectCard.from_dict(data)

    def list_project_cards(self) -> List[ProjectCard]:
        cards: List[ProjectCard] = []
        if not self.projects_dir.exists():
            return cards
        for path in sorted(self.projects_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            cards.append(ProjectCard.from_dict(data))
        return cards

    def write_memory_item(self, item: MemoryItem) -> Path:
        """Write a semantic memory item to ``semantic/items/<memory-id>.yaml`` atomically."""
        path = self._memory_item_path(item.id)
        self._atomic_write_yaml(path, item.to_dict())
        return path

    def read_memory_item(self, memory_id: str) -> Optional[MemoryItem]:
        path = self._memory_item_path(memory_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return MemoryItem.from_dict(data)

    def list_memory_items(self, *, memory_type: str | None = None, status: str | None = None) -> List[MemoryItem]:
        items: List[MemoryItem] = []
        if not self.memory_items_dir.exists():
            return items
        for path in sorted(self.memory_items_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            item = MemoryItem.from_dict(data)
            item_type = getattr(item.type, "value", str(item.type))
            item_status = getattr(item.status, "value", str(item.status))
            if memory_type is not None and item_type != str(memory_type):
                continue
            if status is not None and item_status != str(status):
                continue
            items.append(item)
        return items

    def write_current_working_memory(self, working: WorkingMemory) -> Path:
        """Persist the mutable current working-memory snapshot."""
        self._atomic_write_yaml(self.current_working_path, working.to_dict())
        return self.current_working_path

    def read_current_working_memory(self) -> Optional[WorkingMemory]:
        if not self.current_working_path.exists():
            return None
        with self.current_working_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return WorkingMemory.from_dict(data)

    def clear_current_working_memory(self) -> None:
        if self.current_working_path.exists():
            self.current_working_path.unlink()

    def write_open_loops(self, loops: List[Dict[str, Any]]) -> Path:
        payload = {"version": 1, "updated_at": utc_now_iso(), "open_loops": loops}
        self._atomic_write_yaml(self.open_loops_path, payload)
        return self.open_loops_path

    def list_open_loops(self, *, status: str | None = None) -> List[Dict[str, Any]]:
        if not self.open_loops_path.exists():
            return []
        with self.open_loops_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        loops = list(data.get("open_loops") or [])
        if status is not None:
            loops = [loop for loop in loops if str(loop.get("status") or "") == str(status)]
        return loops

    def upsert_open_loop(self, loop: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(loop)
        payload.setdefault("id", f"loop_{uuid.uuid4().hex}")
        payload.setdefault("status", "open")
        payload.setdefault("created_at", utc_now_iso())
        payload["updated_at"] = utc_now_iso()
        payload["text"] = redact_text(str(payload.get("text") or "").strip())
        payload["source_refs"] = [str(ref) for ref in payload.get("source_refs") or []]
        payload["session_id"] = str(payload.get("session_id") or "")
        if not payload["text"]:
            raise ValidationError("open loop text is required")
        loops = [existing for existing in self.list_open_loops() if existing.get("id") != payload["id"]]
        loops.append(payload)
        self.write_open_loops(loops)
        return payload

    def archive_working_session(self, *, session_id: str, messages: List[Dict[str, Any]]) -> Path:
        working = self.read_current_working_memory()
        archive = {
            "session_id": str(session_id or ""),
            "archived_at": utc_now_iso(),
            "message_count": len(messages),
            "working_memory": working.to_dict() if working else None,
            "open_loops": self.list_open_loops(status="open"),
        }
        safe_session = self._safe_yaml_stem(str(session_id or "session"), "session id")
        path = self.episodic_sessions_dir / f"{safe_session}.yaml"
        self._atomic_write_yaml(path, archive)
        self.clear_current_working_memory()
        return path

    def list_session_archives(self) -> List[Dict[str, Any]]:
        archives: List[Dict[str, Any]] = []
        if not self.episodic_sessions_dir.exists():
            return archives
        for path in sorted(self.episodic_sessions_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as fh:
                archives.append(yaml.safe_load(fh) or {})
        return archives

    def write_source_ref(self, source: SourceRef) -> Path:
        """Write source metadata to ``sources/<source-id>.yaml`` atomically."""
        path = self._source_ref_path(source.id)
        self._atomic_write_yaml(path, source.to_dict())
        return path

    def read_source_ref(self, source_id: str) -> Optional[SourceRef]:
        path = self._source_ref_path(source_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return SourceRef.from_dict(data)

    def list_source_refs(self) -> List[SourceRef]:
        sources: List[SourceRef] = []
        if not self.sources_dir.exists():
            return sources
        for path in sorted(self.sources_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            sources.append(SourceRef.from_dict(data))
        return sources

    def _core_category_path(self, category: str) -> Path:
        safe = self._safe_yaml_stem(str(category), "core category")
        return self.core_dir / f"{safe}.yaml"

    def _project_card_path(self, project_id_or_name: str) -> Path:
        normalized = normalize_project_id(project_id_or_name)
        slug = normalized.split(":", 1)[1]
        return self.projects_dir / f"{slug}.yaml"

    def _memory_item_path(self, memory_id: str) -> Path:
        return self.memory_items_dir / f"{self._safe_yaml_stem(memory_id, 'memory id')}.yaml"

    def _source_ref_path(self, source_id: str) -> Path:
        return self.sources_dir / f"{self._safe_yaml_stem(source_id, 'source id')}.yaml"

    @staticmethod
    def _source_ref_from_raw_event(event: Dict[str, Any]) -> SourceRef:
        event_id = str(event.get("id") or "").strip()
        event_type = str(event.get("type") or "").strip().lower()
        session_id = str(event.get("session_id") or "").strip()
        created_at = str(event.get("created_at") or "")
        if event_type == "tool":
            source_type = "tool_result"
            tool_name = str(event.get("tool") or "tool").strip() or "tool"
            title = f"Raw tool evidence from session {session_id}" if session_id else "Raw tool evidence"
            quote = tool_name
        else:
            source_type = "message"
            title = f"Raw turn evidence from session {session_id}" if session_id else "Raw turn evidence"
            quote = str(event.get("user_content") or event.get("content") or event.get("assistant_content") or "").strip()
        quote = redact_text(quote)
        if len(quote) > 500:
            quote = quote[:497].rstrip() + "..."
        return SourceRef(
            id=event_id,
            type=source_type,
            uri=f"raw_event:{event_id}",
            title=title,
            observed_at=created_at,
            quote=quote or None,
        )

    @staticmethod
    def _safe_yaml_stem(value: str, field_name: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValidationError(f"{field_name} is required")
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in text).strip("-")
        if not safe:
            raise ValidationError(f"{field_name} must contain at least one safe filename character")
        if safe == text:
            return safe
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{safe}--{digest}"

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValidationError(f"JSONL record in {path} must be an object")
                records.append(payload)
        return records

    def _count_jsonl(self, path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def _atomic_write_yaml(self, path: Path, payload: Dict[str, Any]) -> None:
        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        self._atomic_write_text(path, text)

    def _atomic_write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
