#!/usr/bin/env python3
"""Freshness-safe evidence collection for the weekly LLM benchmark report."""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

INSUFFICIENT_EVIDENCE = "Insufficient current evidence"
_ALLOWED_MEASUREMENT_FIELDS = (
    "provider", "model", "task", "latency_ms", "char_count",
    "content_preview", "status", "measured_at",
)


def _utc(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    return value if isinstance(value, dict) else {}


def read_hardware() -> dict[str, Any]:
    """Return stable, non-secret hardware facts from the current machine."""
    memory_bytes = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    return {
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "platform": platform.system(),
    }


def _route_projection(config: dict[str, Any]) -> dict[str, Any]:
    """Return only non-secret inference-routing fields from a profile config."""
    if not config:
        return {}
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        provider = model_cfg.get("provider")
        model = model_cfg.get("default") or model_cfg.get("model")
    else:
        provider = config.get("provider")
        model = model_cfg
    fallbacks = []
    for item in config.get("fallback_providers", []) or []:
        if isinstance(item, dict):
            fallbacks.append({key: item.get(key) for key in ("provider", "model", "base_url") if item.get(key) is not None})
    return {"provider": provider, "model": model, "fallback_providers": fallbacks}


def effective_configs(hermes_home: Path) -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {}
    profile_root = hermes_home / "profiles"
    if profile_root.is_dir():
        for config_path in sorted(profile_root.glob("*/config.yaml")):
            profiles[config_path.parent.name] = _route_projection(_read_yaml(config_path))
    return {"root": _route_projection(_read_yaml(hermes_home / "config.yaml")), "profiles": profiles}


def _active_routes(hermes_home: Path) -> list[dict[str, Any]]:
    raw = _read_json(hermes_home / "cron" / "jobs.json", {})
    jobs = raw.get("jobs", raw if isinstance(raw, list) else [])
    fields = ("id", "name", "enabled", "state", "profile", "provider", "model", "schedule", "script", "no_agent")
    return [
        {key: job.get(key) for key in fields if key in job}
        for job in jobs
        if isinstance(job, dict) and job.get("enabled", True) and job.get("state") != "paused"
    ]


def _previous_snapshot_summary(previous: Any) -> dict[str, Any] | None:
    """Keep only bounded comparison metadata; never recursively embed snapshots."""
    if not isinstance(previous, dict):
        return None
    sources = previous.get("source_retrievals")
    source_names = sorted({str(item.get("name")) for item in sources or [] if isinstance(item, dict) and item.get("name")})
    return {
        "schema_version": previous.get("schema_version"),
        "generated_at": previous.get("generated_at"),
        "source_names": source_names[:50],
    }


def build_preflight(
    hermes_home: Path,
    catalogue_path: Path,
    sources_path: Path,
    previous_path: Path,
    *,
    now: datetime | None = None,
    hardware_reader: Callable[[], dict[str, Any]] = read_hardware,
) -> dict[str, Any]:
    """Build deterministic report input from explicit filesystem sources."""
    catalogue = _read_json(catalogue_path, {})
    modified_at = _iso(datetime.fromtimestamp(catalogue_path.stat().st_mtime, timezone.utc)) if catalogue_path.is_file() else None
    source_retrievals = _read_json(sources_path, [])
    if not isinstance(source_retrievals, list):
        source_retrievals = []
    return {
        "schema_version": 1,
        "generated_at": _iso(_utc(now)),
        "effective_configs": effective_configs(hermes_home),
        "active_cron_routes": _active_routes(hermes_home),
        "catalogue": {"evidence_role": "availability_only", "modified_at": modified_at, "data": catalogue},
        "hardware": hardware_reader(),
        "source_retrievals": source_retrievals,
        "previous_snapshot": _previous_snapshot_summary(_read_json(previous_path, None)),
    }


def detect_config_drift(effective: dict[str, Any], declared: dict[str, Any]) -> list[dict[str, Any]]:
    """Report declared/effective model assignment drift without fleet constants."""
    drift: list[dict[str, Any]] = []

    def compare(path: str, actual: dict[str, Any], claim: dict[str, Any]) -> None:
        for key in ("model", "provider", "fallback_models"):
            if key in actual or key in claim:
                if actual.get(key) != claim.get(key):
                    drift.append({"path": f"{path}.{key}", "declared": claim.get(key), "effective": actual.get(key)})

    compare("root", effective.get("root", {}), declared.get("root", {}))
    names = sorted(set(effective.get("profiles", {})) | set(declared.get("profiles", {})))
    for name in names:
        compare(f"profiles.{name}", effective.get("profiles", {}).get(name, {}), declared.get("profiles", {}).get(name, {}))
    return drift


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def apply_evidence_gate(
    rows: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    now: datetime | None = None,
    max_age_days: int = 31,
) -> dict[str, Any]:
    """Fail closed unless two current independent sources and row provenance exist."""
    current_names: set[str] = set()
    reference = _utc(now)
    for source in sources:
        retrieved = _parse_time(source.get("retrieved_at"))
        source_date = _parse_time(source.get("source_date"))
        current = source_date or retrieved
        if source.get("status") == "success" and retrieved and current and 0 <= (reference - current).days <= max_age_days:
            current_names.add(str(source.get("name", "")))

    provenance_ok = bool(rows) and all(
        isinstance(row.get("provenance"), list)
        and row["provenance"]
        and all(item.get("source") in current_names and item.get("source_date") for item in row["provenance"])
        for row in rows
    )
    if len(current_names) < 2 or not provenance_ok:
        return {"status": "insufficient_current_evidence", "message": INSUFFICIENT_EVIDENCE, "rows": [], "current_sources": sorted(current_names)}
    return {"status": "current", "rows": rows, "current_sources": sorted(current_names)}


def sanitise_measurements(records: list[dict[str, Any]], *, max_records: int = 60, max_preview_chars: int = 200) -> list[dict[str, Any]]:
    """Return a bounded allow-list projection; credential material cannot pass through."""
    clean: list[dict[str, Any]] = []
    for record in records[:max_records]:
        projected = {field: record.get(field) for field in _ALLOWED_MEASUREMENT_FIELDS}
        projected["content_preview"] = str(projected.get("content_preview") or "")[:max_preview_chars]
        projected["status"] = projected.get("status") or "success"
        clean.append(projected)
    return clean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes-home", type=Path, required=True)
    parser.add_argument("--catalogue", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_preflight(args.hermes_home, args.catalogue, args.sources, args.previous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
