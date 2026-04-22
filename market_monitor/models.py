from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


@dataclass
class RawSnapshot:
    snapshot_id: str
    source_id: str
    dataset_id: Optional[str]
    fetch_time: str
    source_url: str
    period_hint: Optional[str]
    content_type: str
    content_hash: str
    local_path: str
    parse_status: str = "pending"
    parser_version: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class EntityRecord:
    entity_id: str
    entity_type: str
    name_norm: str
    name_raw: Optional[str] = None
    parent_entity_id: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ObservationRecord:
    obs_id: str
    dataset_id: str
    source_id: str
    period_label: str
    period_type: str
    metric_name: str
    metric_scope: str
    metric_type: str
    snapshot_id: Optional[str] = None
    observation_key: Optional[str] = None
    energy_type: Optional[str] = None
    value_numeric: Optional[float] = None
    value_text: Optional[str] = None
    unit: Optional[str] = None
    ranking: Optional[int] = None
    published_at: Optional[str] = None
    source_url: Optional[str] = None
    is_latest: int = 1
    revision_no: int = 1
    notes: Optional[str] = None


@dataclass
class ObservationEntityLink:
    obs_id: str
    entity_id: str
    entity_role: str


@dataclass
class ParseOutput:
    snapshot: RawSnapshot
    entities: list[EntityRecord] = field(default_factory=list)
    observations: list[ObservationRecord] = field(default_factory=list)
    observation_entities: list[ObservationEntityLink] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FetchResult:
    source_id: str
    dataset_id: Optional[str]
    fetch_time: str
    source_url: str
    content_type: str
    content_bytes: bytes
    text: Optional[str]
    local_path: str
    status_code: Optional[int] = None
    headers: dict[str, Any] = field(default_factory=dict)
    period_hint: Optional[str] = None
    notes: Optional[str] = None


class Parser(Protocol):
    source_id: str
    parser_version: str

    def parse(self, fetch_result: FetchResult, dataset_id: Optional[str] = None) -> ParseOutput:
        ...
