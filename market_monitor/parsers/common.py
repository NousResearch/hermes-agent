from __future__ import annotations

from dataclasses import replace
import json
import re
from typing import Iterable

from market_monitor.models import (
    EntityRecord,
    FetchResult,
    ObservationEntityLink,
    ObservationRecord,
    ParseOutput,
    RawSnapshot,
)


def make_snapshot(fetch_result: FetchResult, *, parser_version: str) -> RawSnapshot:
    dataset_id = fetch_result.dataset_id
    snapshot_id = f"{fetch_result.source_id}:{dataset_id or 'raw'}:{fetch_result.fetch_time}"
    return RawSnapshot(
        snapshot_id=snapshot_id,
        source_id=fetch_result.source_id,
        dataset_id=dataset_id,
        fetch_time=fetch_result.fetch_time,
        source_url=fetch_result.source_url,
        period_hint=fetch_result.period_hint,
        content_type=fetch_result.content_type,
        content_hash="",
        local_path=fetch_result.local_path,
        parse_status="parsed",
        parser_version=parser_version,
    )


def attach_snapshot_metadata(output: ParseOutput, *, content_hash: str) -> ParseOutput:
    snapshot = replace(output.snapshot, content_hash=content_hash)
    observations = [replace(obs, snapshot_id=snapshot.snapshot_id) for obs in output.observations]
    return ParseOutput(
        snapshot=snapshot,
        entities=output.entities,
        observations=observations,
        observation_entities=output.observation_entities,
        warnings=output.warnings,
    )


def html_text(text: str | None) -> str:
    raw = text or ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw)).strip()


def extract_period_label(text: str) -> str:
    match = re.search(r"(20\d{2})年\s*(1[0-2]|0?[1-9])月", text)
    if not match:
        raise ValueError("Unable to extract monthly period label")
    return f"{match.group(1)}-{int(match.group(2)):02d}"


def normalize_period_label(period_text: str) -> str:
    match = re.search(r"(20\d{2})[-年/.\s]*(1[0-2]|0?[1-9])", period_text)
    if not match:
        raise ValueError(f"Unable to normalize period label: {period_text}")
    return f"{match.group(1)}-{int(match.group(2)):02d}"


def chinese_wan_to_int(value: str, unit: str) -> int:
    number = float(value)
    if number < 0:
        raise ValueError(f"negative value not allowed for {unit}")
    multiplier = 10000 if unit == "wan" else 1
    return int(round(number * multiplier))


def extract_float(pattern: str, text: str, *, group: int = 1) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Pattern not found: {pattern}")
    value = float(match.group(group))
    return value


def make_obs_id(dataset_id: str, period_label: str, metric_name: str, metric_scope: str, suffix: str = "") -> str:
    return ":".join(part for part in [dataset_id, period_label, metric_name, metric_scope, suffix] if part)


def make_brand_entity_id(name_norm: str) -> str:
    return f"brand:{name_norm}"


def make_model_entity_id(brand_name_norm: str, model_name_norm: str) -> str:
    return f"model:{brand_name_norm}:{model_name_norm}"


def validate_non_negative(observations: Iterable[ObservationRecord]) -> None:
    for obs in observations:
        if obs.value_numeric is not None and obs.value_numeric < 0:
            raise ValueError(f"Negative value not allowed for {obs.metric_name}")


def validate_ranking_continuity(observations: Iterable[ObservationRecord]) -> None:
    groups: dict[tuple[str, str, str, str, str | None], list[int]] = {}
    for obs in observations:
        if obs.ranking is None:
            continue
        group_key = (obs.dataset_id, obs.period_label, obs.metric_name, obs.metric_scope, obs.energy_type)
        groups.setdefault(group_key, []).append(obs.ranking)
    for rankings in groups.values():
        ordered = sorted(rankings)
        expected = list(range(1, len(ordered) + 1))
        if ordered != expected:
            raise ValueError(f"Ranking sequence is not continuous: {ordered}")


def parse_embedded_json(text: str, element_id: str = "__DATA__") -> dict:
    pattern = rf'<script[^>]*id=["\']{re.escape(element_id)}["\'][^>]*>(.*?)</script>'
    match = re.search(pattern, text, re.S)
    if not match:
        raise ValueError(f"Embedded JSON script not found: {element_id}")
    return json.loads(match.group(1).strip())


def parse_script_assignment_json(text: str, variable_name: str) -> dict:
    pattern = rf"{re.escape(variable_name)}\s*=\s*(\{{.*?\}})\s*;"
    match = re.search(pattern, text, re.S)
    if not match:
        raise ValueError(f"Script assignment JSON not found: {variable_name}")
    return json.loads(match.group(1))


def build_entity_link(obs_id: str, entity_id: str, role: str) -> ObservationEntityLink:
    return ObservationEntityLink(obs_id=obs_id, entity_id=entity_id, entity_role=role)


def build_brand_and_model_entities(brand: str, model: str) -> tuple[EntityRecord, EntityRecord]:
    brand_id = make_brand_entity_id(brand)
    return (
        EntityRecord(entity_id=brand_id, entity_type="brand", name_norm=brand, name_raw=brand),
        EntityRecord(
            entity_id=make_model_entity_id(brand, model),
            entity_type="model",
            name_norm=model,
            name_raw=model,
            parent_entity_id=brand_id,
        ),
    )
