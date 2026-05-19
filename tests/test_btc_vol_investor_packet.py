import hashlib
import json
from pathlib import Path

from institutional_btc_vol.investor_packet import build_investor_packet


def _seed_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    report = data_dir / "reports" / "latest.md"
    bundle = data_dir / "normalized" / "run" / "bundle.zip"
    manifest = data_dir / "normalized" / "run" / "evidence_manifest.json"
    ledger = data_dir / "normalized" / "run" / "candidate_triage.jsonl"
    quote_ledger = data_dir / "normalized" / "run" / "quote_evidence.jsonl"
    backtest = tmp_path / "backtests" / "vol-spread-backtest-7d-v0.md"
    legal_json = tmp_path / "legal" / "legal-wrapper-package-v1.json"
    legal_md = tmp_path / "legal" / "legal-wrapper-package-v1.md"
    files = {
        report: "# Latest Report\nSCREEN-ONLY · NOT EXECUTABLE\n",
        bundle: "zipbytes",
        manifest: json.dumps({
            "run_id": "run-packet",
            "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
            "artifacts": [{"label": "latest_report", "path": str(report), "sha256": "placeholder", "bytes": 1}],
        }),
        ledger: json.dumps({"rank": 1, "candidate": "IBIT 5D ATM vs Deribit 4D ATM", "gross_iv_diff_vol_pts": 3.47, "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE"}) + "\n",
        quote_ledger: json.dumps({
            "rfq_id": "rfq-packet-001",
            "candidate_id": "candidate-packet-001",
            "structure": "IBIT 5D ATM vs Deribit 4D ATM",
            "as_of_cst": "2026-05-15 13:35:00 CDT",
            "counterparty": "Dealer A",
            "venue": "manual-rfq",
            "instrument": "IBIT / Deribit comparison",
            "side": "two-way",
            "notional_btc": 25,
            "bid_iv": 0.41,
            "ask_iv": 0.45,
            "mid_iv": 0.43,
            "execution_confidence": "quote-verified",
            "source_confidence": "manual-indicative-rfq",
            "status": "indicative",
            "evidence_ref": "internal://rfq/packet-001",
            "notes": "Indicative only.",
        }) + "\n",
        backtest: "# BTC Vol Spread Backtest — 7d\nSCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE\nsynthetic fills; not executable economics\n",
        legal_json: json.dumps({
            "approved_by_counsel": False,
            "status": "draft-blocked",
            "evidence_status": "SCREEN-ONLY · LEGAL WRAPPER DRAFT · NOT EXECUTABLE",
        }),
        legal_md: "# Legal Wrapper\nSCREEN-ONLY · LEGAL WRAPPER DRAFT · NOT EXECUTABLE\nDRAFT — NOT APPROVED FOR EXTERNAL USE\n",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (data_dir / "run_manifest.jsonl").write_text(
        json.dumps(
            {
                "run_id": "run-packet",
                "as_of_cst": "2026-05-15 13:30:00 CDT",
                "btc_spot": 80000,
                "btc_per_share": 0.000567,
                "quality_score": 100,
                "quality_grade": "green",
                "freshness_grade": "green",
                "dislocations": 7,
                "quote_review_candidates": 3,
                "report_path": str(report),
                "evidence_bundle_path": str(bundle),
                "evidence_manifest_path": str(manifest),
                "candidate_ledger_path": str(ledger),
                "quote_evidence_ledger_path": str(quote_ledger),
                "evidence_bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return data_dir


def test_build_investor_packet_creates_handoff_folder_with_manifest_and_index(tmp_path):
    data_dir = _seed_data_dir(tmp_path)
    output_dir = tmp_path / "investor-packet"

    result = build_investor_packet(data_dir, output_dir)

    assert result["ok"] is True
    assert result["run_id"] == "run-packet"
    assert result["evidence_status"] == "SCREEN-ONLY · NOT EXECUTABLE"
    assert Path(result["packet_manifest_path"]).exists()
    assert Path(result["packet_index_path"]).exists()
    assert Path(result["memo_path"]).exists()
    assert Path(result["tearsheet_path"]).exists()
    assert Path(result["site_path"]).exists()
    assert (output_dir / "site" / "site-data.json").exists()
    assert (output_dir / "evidence" / "latest-report.md").exists()
    assert (output_dir / "evidence" / "latest-evidence-bundle.zip").exists()
    assert (output_dir / "evidence" / "evidence_manifest.json").exists()
    assert (output_dir / "evidence" / "candidate_triage.jsonl").exists()
    assert (output_dir / "evidence" / "quote_evidence.jsonl").exists()
    assert (output_dir / "evidence" / "backtest-report.md").exists()
    assert len(result["packet_sha256"]) == 64

    manifest = json.loads((output_dir / "packet_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-packet"
    assert manifest["evidence_status"] == "SCREEN-ONLY · NOT EXECUTABLE"
    assert manifest["publishability"] == "internal-diligence-only"
    assert manifest["packet_sha256"] == result["packet_sha256"]
    labels = {artifact["label"] for artifact in manifest["artifacts"]}
    assert labels == {
        "investor_memo",
        "investor_tearsheet",
        "site_index",
        "site_data",
        "latest_report",
        "latest_evidence_bundle",
        "evidence_manifest",
        "candidate_triage",
        "quote_evidence",
        "backtest_report",
        "legal_wrapper_json",
        "legal_wrapper_md",
        "packet_index",
    }
    assert all(len(artifact["sha256"]) == 64 for artifact in manifest["artifacts"])

    index = (output_dir / "packet_index.md").read_text(encoding="utf-8")
    assert "# BTC Vol Desk Investor Packet" in index
    assert "SCREEN-ONLY · NOT EXECUTABLE" in index
    assert "internal-diligence-only" in index
    assert "Not a client portal. Not an execution venue. Not a fund offering." in index
    assert "packet_manifest.json" in index
    assert "one-page-tear-sheet.md" in index
    assert "latest-evidence-bundle.zip" in index
    assert "backtest-report.md" in index

    copied_evidence_manifest = json.loads((output_dir / "evidence" / "evidence_manifest.json").read_text(encoding="utf-8"))
    for artifact in copied_evidence_manifest.get("artifacts", []):
        assert str(artifact["path"]).startswith("artifacts/")
        assert artifact["path_context"] == "inside latest-evidence-bundle.zip"


def test_build_investor_packet_reports_missing_evidence_artifacts(tmp_path):
    data_dir = _seed_data_dir(tmp_path)
    missing_bundle = data_dir / "normalized" / "run" / "bundle.zip"
    missing_bundle.unlink()

    result = build_investor_packet(data_dir, tmp_path / "packet")

    assert result["ok"] is False
    assert "latest_evidence_bundle" in result["missing_artifacts"]
