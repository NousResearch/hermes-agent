from __future__ import annotations

from datetime import datetime, timezone

from institutional_btc_vol.source_intake import validate_source_intake_manifest


def test_source_intake_manifest_rejects_fixture_and_missing_required_fields():
    manifest = {
        "sources": [
            {
                "source_group": "IBIT options history",
                "provenance": "manual_fixture",
                "license_label": "fixture",
                "format": "csv",
                "raw_sha256": "a" * 64,
                "fields": ["available_ts", "expiration", "strike", "option_type", "bid", "ask"],
                "row_count": 100,
                "available_start": "2026-05-01T14:30:00Z",
                "available_end": "2026-05-02T14:30:00Z",
            }
        ]
    }

    result = validate_source_intake_manifest(manifest, decision_ts=datetime(2026, 5, 3, tzinfo=timezone.utc))

    assert result["ready"] is False
    assert result["covered_source_groups"] == 0
    assert result["required_source_groups"] == 6
    assert "IBIT options history: fixture/manual_fixture sources cannot satisfy readiness" in result["blockers"]
    assert "IBIT options history: missing required fields volume, open_interest, source_ref" in result["blockers"]
    assert result["source_results"][0]["ready"] is False


def test_source_intake_manifest_passes_only_complete_licensed_replay_ready_sources():
    required = {
        "IBIT options history": ["available_ts", "expiration", "strike", "option_type", "bid", "ask", "volume", "open_interest", "source_ref"],
        "Deribit options history": ["available_ts", "instrument_name", "underlying_price", "bid_iv", "ask_iv", "mark_iv", "open_interest", "source_ref"],
        "CME Bitcoin options history": ["available_ts", "symbol", "expiration", "strike", "option_type", "bid", "ask", "settlement", "source_ref"],
        "BTC reference history": ["available_ts", "btc_usd", "venue_or_index", "source_ref"],
        "IBIT holdings history": ["available_ts", "btc_per_share", "shares_outstanding", "fund_assets", "source_ref"],
        "Rates and fee curves": ["available_ts", "tenor", "rate", "borrow_or_fee", "source_ref"],
    }
    manifest = {
        "sources": [
            {
                "source_group": source_group,
                "provenance": "licensed_vendor_api_databento" if source_group.startswith("CME") else "licensed_vendor_export",
                "license_label": "licensed",
                "format": "jsonl",
                "raw_sha256": "b" * 64,
                "fields": fields,
                "row_count": 25,
                "available_start": "2026-05-01T14:30:00Z",
                "available_end": "2026-05-02T14:30:00Z",
            }
            for source_group, fields in required.items()
        ]
    }

    result = validate_source_intake_manifest(manifest, decision_ts=datetime(2026, 5, 3, tzinfo=timezone.utc))

    assert result["ready"] is True
    assert result["covered_source_groups"] == 6
    assert result["blockers"] == []
    assert {row["source_group"] for row in result["source_results"] if row["ready"]} == set(required)


def test_source_intake_manifest_rejects_future_available_data():
    manifest = {
        "sources": [
            {
                "source_group": "BTC reference history",
                "provenance": "licensed_vendor_export",
                "license_label": "licensed",
                "format": "parquet",
                "raw_sha256": "c" * 64,
                "fields": ["available_ts", "btc_usd", "venue_or_index", "source_ref"],
                "row_count": 10,
                "available_start": "2026-05-01T14:30:00Z",
                "available_end": "2026-05-04T14:30:00Z",
            }
        ]
    }

    result = validate_source_intake_manifest(manifest, decision_ts=datetime(2026, 5, 3, tzinfo=timezone.utc))

    assert result["ready"] is False
    assert "BTC reference history: available_end is after decision_ts" in result["blockers"]
