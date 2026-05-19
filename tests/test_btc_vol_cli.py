import hashlib
import json
import zipfile
from pathlib import Path

from institutional_btc_vol.cli import main


def _bundle(tmp_path, *, ok=True):
    content = b"SCREEN-ONLY report"
    sha = hashlib.sha256(content).hexdigest() if ok else "bad"
    manifest = {
        "run_id": "btcvol-cli-test",
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        "artifacts": [{"label": "report", "path": "report.md", "sha256": sha, "bytes": len(content)}],
    }
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("evidence_manifest.json", json.dumps(manifest))
        zf.writestr("evidence_index.md", "SCREEN-ONLY · NOT EXECUTABLE")
        zf.writestr("artifacts/report.md", content)
    return path


def test_cli_verify_bundle_prints_structured_success(tmp_path, capsys):
    bundle = _bundle(tmp_path)

    code = main(["verify-bundle", str(bundle)])

    captured = json.loads(capsys.readouterr().out)
    assert code == 0
    assert captured["ok"] is True
    assert captured["run_id"] == "btcvol-cli-test"
    assert captured["verified_artifacts"] == 1


def test_cli_verify_bundle_returns_nonzero_for_hash_mismatch(tmp_path, capsys):
    bundle = _bundle(tmp_path, ok=False)

    code = main(["verify-bundle", str(bundle)])

    captured = json.loads(capsys.readouterr().out)
    assert code == 1
    assert captured["ok"] is False
    assert captured["hash_mismatches"] == ["report"]


def test_cli_build_site_writes_investor_site_and_site_data(tmp_path, capsys):
    data_dir = tmp_path / "data"
    report = data_dir / "reports" / "latest.md"
    bundle = data_dir / "normalized" / "run" / "bundle.zip"
    ledger = data_dir / "normalized" / "run" / "candidate_triage.jsonl"
    for path, content in ((report, "# report"), (bundle, "zip"), (ledger, json.dumps({"rank": 1, "candidate": "candidate"}) + "\n")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (data_dir / "run_manifest.jsonl").write_text(
        json.dumps({
            "run_id": "run-site",
            "as_of_cst": "now",
            "report_path": str(report),
            "evidence_bundle_path": str(bundle),
            "candidate_ledger_path": str(ledger),
            "evidence_bundle_sha256": "hash",
        }) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "site" / "index.html"

    code = main(["build-site", str(data_dir), str(output)])

    captured = json.loads(capsys.readouterr().out)
    assert code == 0
    assert Path(captured["site_path"]).exists()
    assert Path(captured["site_data_path"]).exists()
    assert "BTC Treasury" in output.read_text(encoding="utf-8")


def _seed_packet_data(tmp_path, run_id="run-memo"):
    data_dir = tmp_path / "data"
    report = data_dir / "reports" / "latest.md"
    bundle = data_dir / "normalized" / "run" / "bundle.zip"
    manifest = data_dir / "normalized" / "run" / "evidence_manifest.json"
    ledger = data_dir / "normalized" / "run" / "candidate_triage.jsonl"
    quote_ledger = data_dir / "normalized" / "run" / "quote_evidence.jsonl"
    backtest = tmp_path / "backtests" / "vol-spread-backtest-7d-v0.md"
    legal_json = tmp_path / "legal" / "legal-wrapper-package-v1.json"
    legal_md = tmp_path / "legal" / "legal-wrapper-package-v1.md"
    for path, content in (
        (report, "# report"),
        (bundle, "zip"),
        (manifest, json.dumps({"run_id": run_id, "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE"})),
        (ledger, json.dumps({"rank": 1, "candidate": "IBIT 5D ATM vs Deribit 4D ATM", "gross_iv_diff_vol_pts": 3.47}) + "\n"),
        (quote_ledger, ""),
        (backtest, "# Backtest\nSCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE\nsynthetic fills; not executable economics\n"),
        (legal_json, json.dumps({
            "approved_by_counsel": False,
            "status": "draft-blocked",
            "evidence_status": "SCREEN-ONLY · LEGAL WRAPPER DRAFT · NOT EXECUTABLE",
        })),
        (legal_md, "# Legal Wrapper\nSCREEN-ONLY · LEGAL WRAPPER DRAFT · NOT EXECUTABLE\nDRAFT — NOT APPROVED FOR EXTERNAL USE\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (data_dir / "run_manifest.jsonl").write_text(
        json.dumps({
            "run_id": run_id,
            "as_of_cst": "now",
            "btc_spot": 80000,
            "report_path": str(report),
            "evidence_bundle_path": str(bundle),
            "evidence_manifest_path": str(manifest),
            "candidate_ledger_path": str(ledger),
            "quote_evidence_ledger_path": str(quote_ledger),
            "evidence_bundle_sha256": "hash",
        }) + "\n",
        encoding="utf-8",
    )
    return data_dir


def test_cli_build_memo_writes_investor_memo(tmp_path, capsys):
    data_dir = _seed_packet_data(tmp_path, "run-memo")
    output = tmp_path / "investor-packet" / "memo.md"

    code = main(["build-memo", str(data_dir), str(output)])

    captured = json.loads(capsys.readouterr().out)
    assert code == 0
    assert captured["ok"] is True
    assert captured["memo_path"] == str(output)
    assert captured["run_id"] == "run-memo"
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# BTC Treasury & Miner Hedging Desk Investor Memo")
    assert "SCREEN-ONLY · NOT EXECUTABLE" in text


def test_cli_build_packet_writes_complete_packet(tmp_path, capsys):
    data_dir = _seed_packet_data(tmp_path, "run-packet-cli")
    output = tmp_path / "investor-packet"

    code = main(["build-packet", str(data_dir), str(output)])

    captured = json.loads(capsys.readouterr().out)
    assert code == 0
    assert captured["ok"] is True
    assert captured["run_id"] == "run-packet-cli"
    assert Path(captured["packet_manifest_path"]).exists()
    assert Path(captured["packet_index_path"]).exists()
    assert (output / "btc-vol-desk-investor-memo.md").exists()
    assert (output / "site" / "index.html").exists()
    assert (output / "evidence" / "latest-evidence-bundle.zip").exists()
    assert len(captured["packet_sha256"]) == 64


def test_cli_verify_packet_prints_structured_success(tmp_path, capsys):
    data_dir = _seed_packet_data(tmp_path, "run-verify-packet-cli")
    output = tmp_path / "investor-packet"
    assert main(["build-packet", str(data_dir), str(output)]) == 0
    capsys.readouterr()

    code = main(["verify-packet", str(output)])

    captured = json.loads(capsys.readouterr().out)
    assert code == 0
    assert captured["ok"] is True
    assert captured["run_id"] == "run-verify-packet-cli"
    assert captured["packet_sha256_ok"] is True
    assert captured["errors"] == []


def test_cli_verify_packet_returns_nonzero_for_failed_packet(tmp_path, capsys):
    data_dir = _seed_packet_data(tmp_path, "run-bad-packet-cli")
    output = tmp_path / "investor-packet"
    assert main(["build-packet", str(data_dir), str(output)]) == 0
    capsys.readouterr()
    (output / "site" / "index.html").write_text("tampered", encoding="utf-8")

    code = main(["verify-packet", str(output)])

    captured = json.loads(capsys.readouterr().out)
    assert code == 1
    assert captured["ok"] is False
    assert any("site_index" in error for error in captured["errors"])



def test_cli_backtest_spread_writes_screen_only_report(tmp_path, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {"run_id": "r1", "as_of_cst": "2026-05-17 10:00:00 CDT", "spread_7d_vol_pts": 6.0},
        {"run_id": "r2", "as_of_cst": "2026-05-17 11:00:00 CDT", "spread_7d_vol_pts": 4.0},
        {"run_id": "r3", "as_of_cst": "2026-05-17 12:00:00 CDT", "spread_7d_vol_pts": -5.5},
        {"run_id": "r4", "as_of_cst": "2026-05-17 13:00:00 CDT", "spread_7d_vol_pts": -2.5},
    ]
    (data_dir / "run_manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "backtests" / "spread.md"
    summary = tmp_path / "backtests" / "spread.json"

    code = main(["backtest-spread", str(data_dir), str(output), "--threshold-vol-pts", "5", "--notional-vega", "10000", "--cost-per-trade", "250", "--summary-json", str(summary)])

    assert code == 0
    captured = json.loads(capsys.readouterr().out)
    assert captured["ok"] is True
    assert captured["trade_count"] == 2
    assert captured["summary_path"] == str(summary)
    assert captured["evidence_status"] == "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE"
    markdown = output.read_text(encoding="utf-8")
    assert "Gross PnL: $49,500.00" in markdown
    assert "not executable economics" in markdown
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["scenarios"][0]["tenor"] == "7d"
    assert payload["scenarios"][0]["sample_gate"] == "insufficient-history"
