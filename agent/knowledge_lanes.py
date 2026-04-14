"""Typed knowledge lane storage with draft/promoted separation."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home

_SCHEMA_VERSION = 1
_ALLOWED_LANES = {"draft", "promoted"}
_ALLOWED_STATUS = {"draft", "promoted", "archived"}
_ALLOWED_CONFIDENCE = {"low", "medium", "high"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _knowledge_dir() -> Path:
    return get_hermes_home() / "knowledge"


def _knowledge_path() -> Path:
    return _knowledge_dir() / "knowledge_lanes.json"


def _base_payload() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "updated_at": _utc_now_iso(),
        "draft_items": [],
        "promoted_items": [],
    }


def _validate_record(record: Any, expected_lane: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return [f"{expected_lane} record must be an object"]

    for key in ("id", "title", "body", "source", "created_at"):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{expected_lane}.{key} must be a non-empty string")

    if record.get("lane") != expected_lane:
        errors.append(f"{expected_lane}.lane must be '{expected_lane}'")

    status = record.get("status")
    if status not in _ALLOWED_STATUS:
        errors.append(f"{expected_lane}.status must be one of {sorted(_ALLOWED_STATUS)}")

    confidence = record.get("confidence")
    if confidence not in _ALLOWED_CONFIDENCE:
        errors.append(f"{expected_lane}.confidence must be one of {sorted(_ALLOWED_CONFIDENCE)}")

    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        errors.append(f"{expected_lane}.provenance must be an object")

    tags = record.get("tags")
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        errors.append(f"{expected_lane}.tags must be a list of strings")

    promotion = record.get("promotion")
    if expected_lane == "promoted":
        if not isinstance(promotion, dict):
            errors.append("promoted.promotion must be an object")
        else:
            if not isinstance(promotion.get("reason"), str) or not promotion.get("reason", "").strip():
                errors.append("promoted.promotion.reason must be a non-empty string")
            evidence = promotion.get("evidence")
            if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
                errors.append("promoted.promotion.evidence must be a list of strings")
            promoted_at = promotion.get("promoted_at")
            if not isinstance(promoted_at, str) or not promoted_at.strip():
                errors.append("promoted.promotion.promoted_at must be a non-empty string")
    elif promotion is not None and not isinstance(promotion, dict):
        errors.append("draft.promotion must be null or an object")

    return errors


def validate_knowledge_payload(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["knowledge payload must be an object"]

    if payload.get("schema_version") != _SCHEMA_VERSION:
        errors.append(f"schema_version must be {_SCHEMA_VERSION}")

    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at.strip():
        errors.append("updated_at must be a non-empty string")

    for field, expected_lane in (("draft_items", "draft"), ("promoted_items", "promoted")):
        items = payload.get(field)
        if not isinstance(items, list):
            errors.append(f"{field} must be a list")
            continue
        for idx, item in enumerate(items):
            for err in _validate_record(item, expected_lane):
                errors.append(f"{field}[{idx}]: {err}")

    return errors


class KnowledgeLaneStore:
    def __init__(self):
        self.path = _knowledge_path()

    def read_state(self) -> dict[str, Any]:
        if not self.path.exists():
            return _base_payload()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _base_payload()
        if not isinstance(payload, dict):
            return _base_payload()
        merged = _base_payload()
        merged.update(payload)
        return merged

    def write_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        merged = _base_payload()
        merged.update(payload)
        merged["schema_version"] = _SCHEMA_VERSION
        merged["updated_at"] = _utc_now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
        return merged

    def _new_record(
        self,
        *,
        lane: str,
        title: str,
        body: str,
        source: str,
        provenance: dict[str, Any],
        tags: list[str] | None = None,
        confidence: str = "medium",
    ) -> dict[str, Any]:
        return {
            "id": uuid.uuid4().hex[:12],
            "lane": lane,
            "status": lane,
            "title": title.strip(),
            "body": body.strip(),
            "source": source.strip(),
            "provenance": provenance,
            "confidence": confidence,
            "created_at": _utc_now_iso(),
            "tags": list(tags or []),
            "promotion": None,
        }

    def add_draft(
        self,
        *,
        title: str,
        body: str,
        source: str,
        provenance: dict[str, Any],
        tags: list[str] | None = None,
        confidence: str = "medium",
    ) -> dict[str, Any]:
        state = self.read_state()
        record = self._new_record(
            lane="draft",
            title=title,
            body=body,
            source=source,
            provenance=provenance,
            tags=tags,
            confidence=confidence,
        )
        state["draft_items"].append(record)
        self.write_state(state)
        return record

    def promote_draft(self, draft_id: str, *, promotion_reason: str, evidence: list[str]) -> dict[str, Any]:
        state = self.read_state()
        draft_items = state.get("draft_items", [])
        promoted_items = state.get("promoted_items", [])
        match = None
        remaining = []
        for item in draft_items:
            if item.get("id") == draft_id:
                match = item
            else:
                remaining.append(item)
        if not match:
            raise KeyError(f"Unknown draft id: {draft_id}")
        match = dict(match)
        match["lane"] = "promoted"
        match["status"] = "promoted"
        match["promotion"] = {
            "reason": promotion_reason,
            "evidence": list(evidence),
            "promoted_at": _utc_now_iso(),
        }
        state["draft_items"] = remaining
        promoted_items.append(match)
        state["promoted_items"] = promoted_items
        self.write_state(state)
        return match

    def _items_for_lane(self, state: dict[str, Any], lane: str) -> Iterable[dict[str, Any]]:
        normalized = (lane or "promoted").strip().lower()
        if normalized == "draft":
            return list(state.get("draft_items", []))
        if normalized == "promoted":
            return list(state.get("promoted_items", []))
        if normalized == "all":
            return [*state.get("draft_items", []), *state.get("promoted_items", [])]
        raise ValueError("lane must be one of: draft, promoted, all")

    def find_relevant_items(
        self,
        query: str,
        *,
        lane: str = "promoted",
        tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        state = self.read_state()
        terms = [term for term in query.lower().split() if term]
        required_tags = {tag.strip().lower() for tag in (tags or []) if isinstance(tag, str) and tag.strip()}
        results: list[dict[str, Any]] = []

        for item in self._items_for_lane(state, lane):
            item_tags = {str(tag).strip().lower() for tag in item.get("tags", []) if str(tag).strip()}
            if required_tags and not required_tags.issubset(item_tags):
                continue
            haystack = " ".join(
                [
                    str(item.get("title", "")),
                    str(item.get("body", "")),
                    str(item.get("source", "")),
                    " ".join(sorted(item_tags)),
                ]
            ).lower()
            if terms and not all(term in haystack for term in terms):
                continue
            results.append(dict(item))
            if len(results) >= limit:
                break

        return results

    def validation_report(self) -> dict[str, Any]:
        payload = self.read_state()
        errors = validate_knowledge_payload(payload)
        return {
            "valid": not errors,
            "errors": errors,
            "counts": {
                "draft": len(payload.get("draft_items", [])),
                "promoted": len(payload.get("promoted_items", [])),
            },
            "path": str(self.path),
        }
