"""Source-of-truth registry for active runtime/governance artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
from typing import Any, Dict, List

from hermes_constants import get_hermes_home
from .evidence import append_hash_chained_event, utc_now_iso


@dataclass
class RegistryEntry:
    component_type: str
    component_name: str
    active_status: str
    confidence: str
    path_or_ref: str
    owner_profile: str
    evidence_refs: List[str] = field(default_factory=list)
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.owner_profile}:{self.component_type}:{self.component_name}"


class SourceOfTruthRegistry:
    def __init__(self, *, path: str | Path | None = None):
        self.path = Path(path) if path is not None else get_hermes_home() / "governance" / "source_registry.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def _save(self, data: Dict[str, Dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def upsert(self, entry: RegistryEntry) -> Dict[str, Any]:
        if entry.active_status == "confirmed_active" and not entry.evidence_refs:
            return {"success": False, "error": "confirmed_active registry entries require at least one evidence ref"}
        data = self._load()
        row = asdict(entry)
        row["key"] = entry.key
        row["updated_at_utc"] = utc_now_iso()
        data[entry.key] = row
        self._save(data)
        append_hash_chained_event("source_registry_events", {"event_type": "source_registry_upsert", **row})
        return {"success": True, "entry": row}

    def list_entries(self) -> List[Dict[str, Any]]:
        return list(self._load().values())
