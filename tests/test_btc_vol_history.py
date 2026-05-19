import json

from institutional_btc_vol.history import append_run_manifest, load_recent_runs


def test_append_run_manifest_creates_jsonl_and_limits_recent_runs(tmp_path):
    manifest = tmp_path / "run_manifest.jsonl"

    append_run_manifest(
        manifest,
        {
            "run_id": "btcvol-001",
            "as_of_cst": "2026-05-14 22:00:00 CDT",
            "btc_spot": 81000.0,
            "btc_per_share": 0.00056,
            "deribit_atm_rows": 12,
            "ibit_atm_rows": 7,
            "dislocations": 3,
            "quote_review_candidates": 1,
            "report_path": "reports/one.md",
            "dashboard_path": "dashboard/index.html",
        },
    )
    append_run_manifest(
        manifest,
        {
            "run_id": "btcvol-002",
            "as_of_cst": "2026-05-14 22:05:00 CDT",
            "btc_spot": 81200.0,
            "btc_per_share": 0.00056,
            "deribit_atm_rows": 12,
            "ibit_atm_rows": 7,
            "dislocations": 4,
            "quote_review_candidates": 2,
            "report_path": "reports/two.md",
            "dashboard_path": "dashboard/index.html",
        },
    )

    raw_lines = manifest.read_text().strip().splitlines()
    assert len(raw_lines) == 2
    assert json.loads(raw_lines[0])["run_id"] == "btcvol-001"

    recent = load_recent_runs(manifest, limit=1)
    assert len(recent) == 1
    assert recent[0]["run_id"] == "btcvol-002"
    assert recent[0]["quote_review_candidates"] == 2


def test_load_recent_runs_ignores_corrupt_lines_and_missing_file(tmp_path):
    missing = tmp_path / "missing.jsonl"
    assert load_recent_runs(missing, limit=5) == []

    manifest = tmp_path / "run_manifest.jsonl"
    manifest.write_text('{"run_id":"good-1"}\nnot-json\n{"run_id":"good-2"}\n')

    recent = load_recent_runs(manifest, limit=10)
    assert [row["run_id"] for row in recent] == ["good-2", "good-1"]
