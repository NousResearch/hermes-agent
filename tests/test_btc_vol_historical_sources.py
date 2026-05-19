from __future__ import annotations

import json
from pathlib import Path

import pytest

from institutional_btc_vol.historical_sources import (
    HistoricalSource,
    HistoricalSourceError,
    load_source_manifest,
    source_file_sha256,
    write_source_manifest,
)


def test_write_source_manifest_hashes_files_and_preserves_provenance(tmp_path: Path):
    raw_file = tmp_path / "raw" / "ibit_opra.csv"
    raw_file.parent.mkdir()
    raw_file.write_text("ts,symbol,bid,ask\n2025-01-02T15:30:00Z,IBIT250117C00060000,1.20,1.35\n", encoding="utf-8")

    source = HistoricalSource(
        source_id="ibit_opra_2025_fixture",
        source_name="IBIT OPRA historical options fixture",
        venue="OPRA",
        instrument_scope="IBIT options",
        provider="fixture_vendor",
        license_label="licensed_vendor_api",
        raw_path=raw_file,
        coverage_start="2025-01-02T15:30:00Z",
        coverage_end="2025-01-02T15:31:00Z",
        event_time_field="ts",
        available_time_field="ts",
        redistribution="internal_only",
        notes="sanitized fixture",
    )

    manifest_path = tmp_path / "source_manifest.json"
    result = write_source_manifest([source], manifest_path)

    assert result["ok"] is True
    assert result["source_count"] == 1
    assert result["manifest_path"] == str(manifest_path)
    assert result["manifest_sha256"] == source_file_sha256(manifest_path)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = payload["sources"][0]
    assert row["source_id"] == "ibit_opra_2025_fixture"
    assert row["venue"] == "OPRA"
    assert row["license_label"] == "licensed_vendor_api"
    assert row["sha256"] == source_file_sha256(raw_file)
    assert row["bytes"] == raw_file.stat().st_size
    assert row["execution_confidence"] == "screen_only_not_executable"
    assert row["backtest_status"] == "backtest_only_not_executable"
    assert "fixture_vendor" in row["provider"]

    loaded = load_source_manifest(manifest_path)
    assert loaded["ok"] is True
    assert loaded["sources"][0]["source_id"] == source.source_id


def test_source_manifest_rejects_missing_required_file_unless_optional(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    source = HistoricalSource(
        source_id="missing_required",
        source_name="Missing required source",
        venue="Deribit",
        instrument_scope="BTC options",
        provider="fixture_vendor",
        license_label="manual_fixture",
        raw_path=missing,
        coverage_start="2025-01-01T00:00:00Z",
        coverage_end="2025-01-02T00:00:00Z",
        event_time_field="event_ts",
        available_time_field="available_ts",
        redistribution="internal_only",
    )

    with pytest.raises(HistoricalSourceError, match="missing raw source"):
        write_source_manifest([source], tmp_path / "manifest.json")

    optional = source.with_updates(optional=True)
    result = write_source_manifest([optional], tmp_path / "optional_manifest.json")
    assert result["ok"] is True
    payload = json.loads((tmp_path / "optional_manifest.json").read_text(encoding="utf-8"))
    assert payload["sources"][0]["status"] == "missing_optional"
    assert payload["sources"][0]["sha256"] is None


def test_source_manifest_rejects_secrets_and_invalid_labels(tmp_path: Path):
    raw_file = tmp_path / "raw.csv"
    raw_file.write_text("ok\n", encoding="utf-8")

    secret_source = HistoricalSource(
        source_id="bad_secret",
        source_name="Bad secret path",
        venue="CME",
        instrument_scope="BTC options",
        provider="databento",
        license_label="licensed_vendor_api_databento",
        raw_path=Path("/tmp/data?api_key=SECRET/raw.csv"),
        coverage_start="2025-01-01T00:00:00Z",
        coverage_end="2025-01-02T00:00:00Z",
        event_time_field="ts_event",
        available_time_field="ts_recv",
        redistribution="internal_only",
    )
    with pytest.raises(HistoricalSourceError, match="secret-like"):
        write_source_manifest([secret_source], tmp_path / "secret_manifest.json")

    bad_label = HistoricalSource(
        source_id="bad_label",
        source_name="Bad label",
        venue="OPRA",
        instrument_scope="IBIT options",
        provider="fixture_vendor",
        license_label="official_magic_feed",
        raw_path=raw_file,
        coverage_start="2025-01-01T00:00:00Z",
        coverage_end="2025-01-02T00:00:00Z",
        event_time_field="ts",
        available_time_field="ts",
        redistribution="internal_only",
    )
    with pytest.raises(HistoricalSourceError, match="invalid license_label"):
        write_source_manifest([bad_label], tmp_path / "bad_label_manifest.json")
