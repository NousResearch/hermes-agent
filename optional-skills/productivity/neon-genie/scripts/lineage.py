#!/usr/bin/env python3
"""Stable IDs and content hashes for Neon Genie artifacts (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(raw.encode("utf-8"))


def new_id(prefix: str) -> str:
    """Compact stable-looking id: prefix + 12 hex chars."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def run_id() -> str:
    return new_id("ng_run")


def artifact_id(kind: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", kind.lower()).strip("_") or "artifact"
    return new_id(f"ng_{safe}")


def request_id() -> str:
    return new_id("ng_req")


def skill_version(root: Path) -> str:
    for rel in ("VERSION", "references/VERSION"):
        p = root / rel
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()
    return "0.0.0"


# Heuristic: filename → artifact type for packaging outputs
PATH_TYPE_HINTS: list[tuple[str, str]] = [
    ("run-envelope", "run_envelope"),
    ("run-receipt", "run_receipt"),
    ("recipe-summary", "recipe_summary"),
    ("profile-route", "profile_route"),
    ("data-request", "data_request"),
    ("wayfinder", "wayfinder_execution_packet"),
    ("product-packet", "product_packet"),
    ("opportunity", "opportunity_packet"),
    ("zero-option", "zero_option_packet"),
    ("fragmentation", "fragmentation_packet"),
    ("commercial", "commercial_simulation"),
    ("agentic", "agentic_service_graph"),
    ("memetic", "memetic_pressure_packet"),
    ("audit", "audit_delivery_packet"),
    ("evidence", "evidence_intelligence_packet"),
]


def guess_artifact_type(path: Path) -> str:
    name = path.name.lower()
    for needle, kind in PATH_TYPE_HINTS:
        if needle in name:
            return kind
    if name.endswith(".json"):
        return "json_artifact"
    return "file"


PRIMARY_PREFERENCE = (
    "product_packet",
    "opportunity_packet",
    "zero_option_packet",
    "fragmentation_packet",
    "commercial_simulation",
    "agentic_service_graph",
    "audit_delivery_packet",
    "evidence_intelligence_packet",
    "memetic_pressure_packet",
    "wayfinder_execution_packet",
    "run_receipt",
)
