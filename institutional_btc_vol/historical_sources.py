from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

ALLOWED_LICENSE_LABELS = {
    "licensed_vendor_api",
    "licensed_vendor_api_databento",
    "public_screen_reference",
    "manual_fixture",
    "derived_internal",
}

SECRET_MARKERS = (
    "api_key=",
    "apikey=",
    "token=",
    "password=",
    "secret=",
    "access_key=",
    "connection_string=",
)


class HistoricalSourceError(ValueError):
    pass


@dataclass(frozen=True)
class HistoricalSource:
    source_id: str
    source_name: str
    venue: str
    instrument_scope: str
    provider: str
    license_label: str
    raw_path: str | Path
    coverage_start: str
    coverage_end: str
    event_time_field: str
    available_time_field: str
    redistribution: str
    notes: str = ""
    optional: bool = False

    def with_updates(self, **updates: Any) -> "HistoricalSource":
        return replace(self, **updates)


def source_file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _reject_secret_like_text(value: Any, *, field: str) -> None:
    text = str(value).lower()
    if any(marker in text for marker in SECRET_MARKERS):
        raise HistoricalSourceError(f"secret-like value found in {field}; refusing to write source manifest")


def _validate_source(source: HistoricalSource) -> None:
    if source.license_label not in ALLOWED_LICENSE_LABELS:
        raise HistoricalSourceError(f"invalid license_label: {source.license_label}")
    for field, value in asdict(source).items():
        _reject_secret_like_text(value, field=field)
    required_text_fields = [
        "source_id",
        "source_name",
        "venue",
        "instrument_scope",
        "provider",
        "coverage_start",
        "coverage_end",
        "event_time_field",
        "available_time_field",
        "redistribution",
    ]
    for field in required_text_fields:
        if not str(getattr(source, field)).strip():
            raise HistoricalSourceError(f"missing required source field: {field}")


def _source_to_manifest_row(source: HistoricalSource) -> dict[str, Any]:
    _validate_source(source)
    raw_path = Path(source.raw_path)
    exists = raw_path.exists() and raw_path.is_file()
    if not exists and not source.optional:
        raise HistoricalSourceError(f"missing raw source: {raw_path}")

    row = {
        "source_id": source.source_id,
        "source_name": source.source_name,
        "venue": source.venue,
        "instrument_scope": source.instrument_scope,
        "provider": source.provider,
        "license_label": source.license_label,
        "raw_path": str(raw_path),
        "sha256": source_file_sha256(raw_path) if exists else None,
        "bytes": raw_path.stat().st_size if exists else 0,
        "coverage_start": source.coverage_start,
        "coverage_end": source.coverage_end,
        "event_time_field": source.event_time_field,
        "available_time_field": source.available_time_field,
        "redistribution": source.redistribution,
        "notes": source.notes,
        "status": "available" if exists else "missing_optional",
        "execution_confidence": "screen_only_not_executable",
        "backtest_status": "backtest_only_not_executable",
    }
    return row


def write_source_manifest(sources: list[HistoricalSource], manifest_path: str | Path) -> dict[str, Any]:
    rows = [_source_to_manifest_row(source) for source in sources]
    manifest = {
        "ok": True,
        "manifest_type": "btc_vol_historical_source_manifest",
        "evidence_status": "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE",
        "source_count": len(rows),
        "sources": sorted(rows, key=lambda row: row["source_id"]),
    }
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "ok": True,
        "manifest_path": str(path),
        "manifest_sha256": source_file_sha256(path),
        "source_count": len(rows),
    }


def load_source_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"ok": True, "manifest_path": str(path), **payload}
