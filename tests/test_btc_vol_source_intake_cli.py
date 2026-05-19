from __future__ import annotations

import json
from pathlib import Path

from institutional_btc_vol.cli import main
from institutional_btc_vol.source_intake import SOURCE_REQUIREMENTS, build_source_manifest_entry


def _complete_manifest() -> dict:
    return {
        "decision_ts": "2026-05-01T00:00:00+00:00",
        "sources": [
            {
                "source_group": source_group,
                "provenance": "licensed_vendor",
                "license_label": "internal_replay_license",
                "format": "csv",
                "raw_sha256": (hex(idx + 1)[2:] * 64)[:64],
                "fields": fields,
                "row_count": 100 + idx,
                "available_end": "2026-04-30T23:59:00+00:00",
                "source_ref": f"vendor://{idx}/{source_group}",
            }
            for idx, (source_group, fields) in enumerate(SOURCE_REQUIREMENTS.items())
        ],
    }


def test_cli_validate_source_intake_writes_report_and_json(capsys, tmp_path):
    manifest_path = tmp_path / "source_intake_manifest.json"
    report_path = tmp_path / "source_intake_validation.md"
    manifest_path.write_text(json.dumps(_complete_manifest()), encoding="utf-8")

    exit_code = main([
        "validate-source-intake",
        str(manifest_path),
        "--output",
        str(report_path),
    ])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["ready"] is True
    assert payload["covered_source_groups"] == 6
    assert payload["required_source_groups"] == 6
    assert payload["report_path"] == str(report_path)
    report = report_path.read_text(encoding="utf-8")
    assert "# Source Intake Validation Report" in report
    assert "SCREEN-ONLY · SOURCE INTAKE VALIDATION · NOT EXECUTABLE" in report
    assert "6/6 source groups structurally valid" in report
    assert "No executable quote, RFQ, advice, or investment readiness is implied" in report


def test_cli_validate_source_intake_fails_fixture_manifest(capsys, tmp_path):
    manifest = _complete_manifest()
    manifest["sources"][0]["provenance"] = "manual_fixture"
    manifest_path = tmp_path / "source_intake_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    exit_code = main(["validate-source-intake", str(manifest_path)])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["ready"] is False
    assert "IBIT options history: fixture/manual_fixture sources cannot satisfy readiness" in payload["blockers"]


def test_cli_write_source_intake_template_outputs_all_required_groups(capsys, tmp_path):
    output = tmp_path / "source_intake_template.json"

    exit_code = main(["write-source-intake-template", str(output)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["template_path"] == str(output)
    template = json.loads(output.read_text(encoding="utf-8"))
    assert template["evidence_status"] == "SCREEN-ONLY · SOURCE INTAKE TEMPLATE · NOT EXECUTABLE"
    assert len(template["sources"]) == 6
    assert [row["source_group"] for row in template["sources"]] == list(SOURCE_REQUIREMENTS)
    assert template["sources"][0]["fields"] == SOURCE_REQUIREMENTS["IBIT options history"]
    assert "replace-with-64-character-raw-file-sha256" in template["sources"][0]["raw_sha256"]


def test_build_source_manifest_entry_hashes_csv_and_derives_schema(tmp_path):
    source_file = tmp_path / "ibit_options.csv"
    source_file.write_text(
        "available_ts,expiration,strike,option_type,bid,ask,volume,open_interest,source_ref\n"
        "2026-04-29T15:00:00+00:00,2026-05-15,50,C,1.0,1.2,10,100,vendor-row-1\n"
        "2026-04-30T16:30:00+00:00,2026-05-15,55,P,0.8,1.0,12,110,vendor-row-2\n",
        encoding="utf-8",
    )

    entry = build_source_manifest_entry(
        source_file,
        source_group="IBIT options history",
        provenance="licensed_vendor",
        license_label="internal_replay_license",
        source_ref="vendor-export-123",
    )

    assert entry["source_group"] == "IBIT options history"
    assert entry["format"] == "csv"
    assert len(entry["raw_sha256"]) == 64
    assert entry["row_count"] == 2
    assert entry["fields"] == SOURCE_REQUIREMENTS["IBIT options history"]
    assert entry["available_end"] == "2026-04-30T16:30:00+00:00"
    assert entry["source_ref"] == "vendor-export-123"
    assert entry["raw_path"] == str(source_file)


def test_cli_build_source_intake_entry_writes_json(capsys, tmp_path):
    source_file = tmp_path / "deribit.jsonl"
    source_file.write_text(
        json.dumps({
            "available_ts": "2026-04-30T12:00:00+00:00",
            "instrument_name": "BTC-15MAY26-80000-C",
            "underlying_price": 80000,
            "bid_iv": 42.1,
            "ask_iv": 43.2,
            "mark_iv": 42.7,
            "open_interest": 25,
            "source_ref": "row-1",
        }) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "deribit_entry.json"

    exit_code = main([
        "build-source-intake-entry",
        str(source_file),
        "--source-group",
        "Deribit options history",
        "--provenance",
        "licensed_vendor",
        "--license-label",
        "internal_replay_license",
        "--source-ref",
        "vendor-export-deribit-1",
        "--output",
        str(output),
    ])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["entry_path"] == str(output)
    entry = json.loads(output.read_text(encoding="utf-8"))
    assert entry["source_group"] == "Deribit options history"
    assert entry["format"] == "jsonl"
    assert entry["row_count"] == 1
    assert entry["available_end"] == "2026-04-30T12:00:00+00:00"
    assert entry["license_label"] == "internal_replay_license"
    assert "SCREEN-ONLY · SOURCE INTAKE ENTRY · NOT EXECUTABLE" == payload["evidence_status"]
