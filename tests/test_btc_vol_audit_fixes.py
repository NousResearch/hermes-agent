import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

from institutional_btc_vol.evidence_bundle import build_evidence_bundle
from institutional_btc_vol.investor_packet import build_investor_packet
from institutional_btc_vol.packet_verify import verify_investor_packet
from institutional_btc_vol.run_monitor import parse_nasdaq_expiry_group, parse_nasdaq_ibit_rows, try_fetch_deribit_options


def _write_packet_manifest(packet_dir: Path, artifacts: list[dict], *, packet_hash: str = "placeholder") -> None:
    manifest = {
        "run_id": "run-path-test",
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        "publishability": "internal-diligence-only",
        "artifacts": artifacts,
        "missing_artifacts": [],
        "packet_sha256": packet_hash,
    }
    payload = {k: v for k, v in manifest.items() if k != "packet_sha256"}
    manifest["packet_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (packet_dir / "packet_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_packet_builder_hashes_packet_index_as_first_class_artifact(tmp_path):
    data_dir = tmp_path / "data"
    report = data_dir / "reports" / "latest.md"
    bundle = data_dir / "normalized" / "run" / "bundle.zip"
    evidence_manifest = data_dir / "normalized" / "run" / "evidence_manifest.json"
    candidate_ledger = data_dir / "normalized" / "run" / "candidate_triage.jsonl"
    quote_ledger = data_dir / "normalized" / "run" / "quote_evidence.jsonl"
    backtest = tmp_path / "backtests" / "vol-spread-backtest-7d-v0.md"
    legal_json = tmp_path / "legal" / "legal-wrapper-package-v1.json"
    legal_md = tmp_path / "legal" / "legal-wrapper-package-v1.md"
    for path, text in {
        report: "# Report\nSCREEN-ONLY · NOT EXECUTABLE\n",
        bundle: "bundle",
        evidence_manifest: json.dumps({"run_id": "run-index", "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE"}),
        candidate_ledger: json.dumps({"candidate": "demo", "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE"}) + "\n",
        quote_ledger: "",
        backtest: "# Backtest\nSCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE\nsynthetic fills; not executable economics\n",
        legal_json: json.dumps({
            "approved_by_counsel": False,
            "status": "draft-blocked",
            "evidence_status": "SCREEN-ONLY · LEGAL WRAPPER DRAFT · NOT EXECUTABLE",
        }),
        legal_md: "# Legal Wrapper\nSCREEN-ONLY · LEGAL WRAPPER DRAFT · NOT EXECUTABLE\nDRAFT — NOT APPROVED FOR EXTERNAL USE\n",
    }.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (data_dir / "run_manifest.jsonl").write_text(
        json.dumps({
            "run_id": "run-index",
            "as_of_cst": "2026-05-15 13:30:00 CDT",
            "btc_spot": 80000,
            "btc_per_share": 0.000567,
            "quality_score": 100,
            "quality_grade": "green",
            "freshness_grade": "green",
            "dislocations": 0,
            "quote_review_candidates": 0,
            "report_path": str(report),
            "evidence_bundle_path": str(bundle),
            "evidence_manifest_path": str(evidence_manifest),
            "candidate_ledger_path": str(candidate_ledger),
            "quote_evidence_ledger_path": str(quote_ledger),
            "evidence_bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        }) + "\n",
        encoding="utf-8",
    )

    result = build_investor_packet(data_dir, tmp_path / "packet")

    manifest = json.loads(Path(result["packet_manifest_path"]).read_text(encoding="utf-8"))
    labels = {artifact["label"] for artifact in manifest["artifacts"]}
    assert "packet_index" in labels
    assert verify_investor_packet(tmp_path / "packet")["ok"] is True


def test_packet_verifier_rejects_paths_that_escape_packet_directory(tmp_path):
    packet_dir = tmp_path / "packet"
    packet_dir.mkdir()
    external = tmp_path / "outside.md"
    external.write_text("SCREEN-ONLY · NOT EXECUTABLE\n", encoding="utf-8")
    _write_packet_manifest(packet_dir, [{
        "label": "investor_memo",
        "path": "../outside.md",
        "bytes": external.stat().st_size,
        "sha256": hashlib.sha256(external.read_bytes()).hexdigest(),
    }])

    result = verify_investor_packet(packet_dir)

    assert result["ok"] is False
    assert any("escapes packet_dir" in error for error in result["errors"])


def test_parse_nasdaq_expiry_group_accepts_no_year_abbrev_and_rollover():
    assert parse_nasdaq_expiry_group("May 17", datetime(2026, 5, 15)).isoformat() == "2026-05-17"
    assert parse_nasdaq_expiry_group("May 17, 2026", datetime(2026, 5, 15)).isoformat() == "2026-05-17"
    assert parse_nasdaq_expiry_group("Dec 31", datetime(2026, 12, 1)).isoformat() == "2026-12-31"
    assert parse_nasdaq_expiry_group("Jan 2", datetime(2026, 12, 30)).isoformat() == "2027-01-02"


def test_parse_nasdaq_ibit_rows_handles_realistic_expiry_group_formats():
    chain = {
        "lastTrade": "IBIT $45.00",
        "table": {"rows": [
            {"expirygroup": "May 17"},
            {"strike": "45", "c_Bid": "1.00", "c_Ask": "1.20", "p_Bid": "0.90", "p_Ask": "1.10"},
            {"expirygroup": "Jun 21, 2026"},
            {"strike": "46", "c_Bid": "1.10", "c_Ask": "1.30", "p_Bid": "1.00", "p_Ask": "1.20"},
        ]},
    }

    rows = parse_nasdaq_ibit_rows(chain, datetime(2026, 5, 15), 0.000567)

    expiries = {row["expiry"] for row in rows}
    assert "2026-05-17" in expiries
    assert "2026-06-21" in expiries


def test_try_fetch_deribit_options_degrades_to_empty_rows_on_provider_failure(tmp_path, monkeypatch):
    import institutional_btc_vol.run_monitor as runner

    def fail_fetch():
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(runner, "fetch_deribit_options", fail_fetch)

    options, diagnostics, warnings, captured_at = try_fetch_deribit_options(tmp_path)

    assert options == []
    assert diagnostics == []
    assert captured_at is None
    assert any("Deribit fetch or parse failed" in warning for warning in warnings)
    assert not (tmp_path / "deribit_book_summary_btc_options.json").exists()


def test_evidence_bundle_is_deterministic_for_identical_inputs(tmp_path):
    artifact = tmp_path / "report.md"
    artifact.write_text("SCREEN-ONLY report", encoding="utf-8")
    manifest = tmp_path / "evidence_manifest.json"
    manifest.write_text(json.dumps({
        "run_id": "btcvol-deterministic",
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        "artifacts": [{"label": "report", "path": str(artifact), "sha256": "abc", "bytes": artifact.stat().st_size}],
    }), encoding="utf-8")
    index = tmp_path / "evidence_index.md"
    index.write_text("# Evidence\nSCREEN-ONLY · NOT EXECUTABLE\n", encoding="utf-8")

    first = build_evidence_bundle(tmp_path / "first.zip", run_id="btcvol-deterministic", manifest_json=manifest, manifest_markdown=index)
    second = build_evidence_bundle(tmp_path / "second.zip", run_id="btcvol-deterministic", manifest_json=manifest, manifest_markdown=index)

    assert first["bundle_sha256"] == second["bundle_sha256"]
    with zipfile.ZipFile(tmp_path / "first.zip") as zf:
        assert all(info.date_time == (2026, 1, 1, 0, 0, 0) for info in zf.infolist())
