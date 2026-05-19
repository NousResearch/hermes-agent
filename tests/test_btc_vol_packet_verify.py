import json
from pathlib import Path

from institutional_btc_vol.packet_verify import FORBIDDEN_EXECUTION_CTAS, verify_investor_packet


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_packet(tmp_path: Path) -> Path:
    packet = tmp_path / "packet"
    files = {
        "btc-vol-desk-investor-memo.md": "# Memo\nSCREEN-ONLY · NOT EXECUTABLE\nNot a client portal. Not an execution venue. Not a fund offering.\n",
        "packet_index.md": "# Index\nSCREEN-ONLY · NOT EXECUTABLE\ninternal-diligence-only\n",
        "one-page-tear-sheet.md": "# Tear Sheet\nSCREEN-ONLY · NOT EXECUTABLE\nNot a client portal. Not an execution venue. Not a fund offering.\n",
        "site/index.html": "<html><body>SCREEN-ONLY · NOT EXECUTABLE</body></html>",
        "site/site-data.json": json.dumps({"evidence_status": "SCREEN-ONLY · NOT EXECUTABLE"}),
        "evidence/latest-report.md": "# Report\nSCREEN-ONLY · NOT EXECUTABLE\n",
        "evidence/latest-evidence-bundle.zip": "bundle-bytes",
        "evidence/evidence_manifest.json": json.dumps({"evidence_status": "SCREEN-ONLY · NOT EXECUTABLE"}),
        "evidence/candidate_triage.jsonl": json.dumps({"evidence_status": "SCREEN-ONLY · NOT EXECUTABLE"}) + "\n",
        "evidence/quote_evidence.jsonl": "",
        "evidence/legal-wrapper-package-v1.json": json.dumps({"approved_by_counsel": False, "status": "draft-blocked", "evidence_status": "SCREEN-ONLY · LEGAL WRAPPER DRAFT · NOT EXECUTABLE"}),
        "evidence/legal-wrapper-package-v1.md": "# Legal Wrapper\nSCREEN-ONLY · LEGAL WRAPPER DRAFT · NOT EXECUTABLE\nDRAFT — NOT APPROVED FOR EXTERNAL USE\n",
        "evidence/backtest-report.md": "# Backtest\nSCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE\nsynthetic fills; not executable economics\n",
    }
    artifacts = []
    for rel, content in files.items():
        path = packet / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        label = {
            "btc-vol-desk-investor-memo.md": "investor_memo",
            "packet_index.md": "packet_index",
            "one-page-tear-sheet.md": "investor_tearsheet",
            "site/index.html": "site_index",
            "site/site-data.json": "site_data",
            "evidence/latest-report.md": "latest_report",
            "evidence/latest-evidence-bundle.zip": "latest_evidence_bundle",
            "evidence/evidence_manifest.json": "evidence_manifest",
            "evidence/candidate_triage.jsonl": "candidate_triage",
            "evidence/quote_evidence.jsonl": "quote_evidence",
            "evidence/legal-wrapper-package-v1.json": "legal_wrapper_json",
            "evidence/legal-wrapper-package-v1.md": "legal_wrapper_md",
            "evidence/backtest-report.md": "backtest_report",
        }[rel]
        artifacts.append({"label": label, "path": rel, "bytes": path.stat().st_size, "sha256": _sha(path)})
    manifest = {
        "run_id": "run-verify",
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        "publishability": "internal-diligence-only",
        "artifacts": artifacts,
        "missing_artifacts": [],
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    manifest["packet_sha256"] = hashlib.sha256(payload).hexdigest()
    (packet / "packet_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return packet


def _refresh_manifest_hash(packet: Path) -> None:
    manifest_path = packet / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        artifact_path = packet / artifact["path"]
        artifact["bytes"] = artifact_path.stat().st_size
        artifact["sha256"] = _sha(artifact_path)
    import hashlib

    payload = json.dumps({k: v for k, v in manifest.items() if k != "packet_sha256"}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["packet_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _quote_record(counterparty: str, candidate_id: str = "RFQ-001") -> dict:
    return {
        "rfq_id": candidate_id,
        "candidate_id": candidate_id,
        "structure": "BTC put spread",
        "as_of_cst": "2026-05-18 16:00:00 CDT",
        "counterparty": counterparty,
        "venue": "external-dealer",
        "instrument": "BTC-OTC",
        "side": "buy",
        "notional_btc": 100,
        "execution_confidence": "quote-verified",
        "source_confidence": "external-indicative",
        "status": "indicative",
        "evidence_ref": f"legal-approved-redacted-email-{counterparty}",
        "promoted_by_operator": "operator-a",
        "promotion_timestamp": "2026-05-18T21:00:00Z",
        "promotion_basis": "redacted dealer quote record reviewed",
        "legal_review_ref": "counsel-memo-001",
        "bid_iv": 0.4,
        "ask_iv": 0.42,
    }


def _make_external_candidate_packet(packet: Path, *, quote_records: list[dict], source_ready: bool = True) -> None:
    manifest_path = packet / "packet_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["publishability"] = "external-use-approved"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    legal = packet / "evidence" / "legal-wrapper-package-v1.json"
    legal.write_text(json.dumps({
        "approved_by_counsel": True,
        "status": "approved",
        "evidence_status": "SCREEN-ONLY · LEGAL WRAPPER APPROVED · NOT EXECUTABLE",
    }), encoding="utf-8")
    site_data = packet / "site" / "site-data.json"
    site_data.write_text(json.dumps({
        "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        "source_intake_validation": {
            "ready": source_ready,
            "covered_source_groups": 6 if source_ready else 0,
            "required_source_groups": 6,
            "blockers": [] if source_ready else ["fixture/manual_fixture sources cannot satisfy readiness"],
        },
    }), encoding="utf-8")
    ledger = packet / "evidence" / "quote_evidence.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in quote_records), encoding="utf-8")
    _refresh_manifest_hash(packet)


def test_verify_investor_packet_accepts_clean_packet(tmp_path):
    packet = _write_packet(tmp_path)

    result = verify_investor_packet(packet)

    assert result["ok"] is True
    assert result["run_id"] == "run-verify"
    assert result["packet_sha256_ok"] is True
    assert result["artifact_count"] == 13
    assert result["external_use_gate"]["ok"] is False
    assert "counsel-approved legal wrapper missing" in result["external_use_gate"]["blockers"]
    assert result["legal_wrapper"]["present"] is True
    assert result["legal_wrapper"]["approved_by_counsel"] is False
    assert result["errors"] == []
    assert result["warnings"] == []
    assert result["secret_scan"]["matches"] == []
    assert result["execution_cta_scan"]["matches"] == []
    assert result["control_language"]["ok"] is True


def test_verify_investor_packet_detects_hash_size_and_missing_artifact(tmp_path):
    packet = _write_packet(tmp_path)
    (packet / "site" / "index.html").write_text("tampered SCREEN-ONLY · NOT EXECUTABLE", encoding="utf-8")
    (packet / "evidence" / "latest-report.md").unlink()

    result = verify_investor_packet(packet)

    assert result["ok"] is False
    assert any("site_index sha256 mismatch" in error for error in result["errors"])
    assert any("site_index byte size mismatch" in error for error in result["errors"])
    assert any("latest_report missing" in error for error in result["errors"])


def test_verify_investor_packet_detects_forbidden_text_and_missing_controls(tmp_path):
    packet = _write_packet(tmp_path)
    memo = packet / "btc-vol-desk-investor-memo.md"
    memo.write_text("# Memo\nsubmit rfq\nSTART TRADING\napi_key = abc123\n", encoding="utf-8")

    result = verify_investor_packet(packet)

    assert result["ok"] is False
    assert "Submit RFQ" in result["execution_cta_scan"]["matches"]
    assert "Start trading" in result["execution_cta_scan"]["matches"]
    assert "api_key" in result["secret_scan"]["matches"]
    assert any("investor_memo missing SCREEN-ONLY" in error for error in result["errors"])
    assert "Submit RFQ" in FORBIDDEN_EXECUTION_CTAS


def test_verify_investor_packet_scans_text_members_inside_zip_artifacts(tmp_path):
    packet = _write_packet(tmp_path)
    import zipfile

    bundle = packet / "evidence" / "latest-evidence-bundle.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("evidence_manifest.json", '{"evidence_status":"SCREEN-ONLY · NOT EXECUTABLE"}')
        zf.writestr("artifacts/bad.md", "open account now")
    manifest = json.loads((packet / "packet_manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact["label"] == "latest_evidence_bundle":
            artifact["bytes"] = bundle.stat().st_size
            artifact["sha256"] = _sha(bundle)
    payload = json.dumps({k: v for k, v in manifest.items() if k != "packet_sha256"}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    manifest["packet_sha256"] = hashlib.sha256(payload).hexdigest()
    (packet / "packet_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    result = verify_investor_packet(packet)

    assert result["ok"] is False
    assert any("Open account" in match for match in result["execution_cta_scan"]["matches"])


def test_verify_investor_packet_ignores_source_page_ctas_inside_raw_zip_captures(tmp_path):
    packet = _write_packet(tmp_path)
    import hashlib
    import zipfile

    bundle = packet / "evidence" / "latest-evidence-bundle.zip"
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("evidence_manifest.json", '{"evidence_status":"SCREEN-ONLY · NOT EXECUTABLE"}')
        zf.writestr("artifacts/raw_ibit_holdings.csv", "Issuer marketing footer,Buy now\n")
        zf.writestr("artifacts/report.md", "SCREEN-ONLY · NOT EXECUTABLE\n")
    manifest = json.loads((packet / "packet_manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact["label"] == "latest_evidence_bundle":
            artifact["bytes"] = bundle.stat().st_size
            artifact["sha256"] = _sha(bundle)
    payload = json.dumps({k: v for k, v in manifest.items() if k != "packet_sha256"}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["packet_sha256"] = hashlib.sha256(payload).hexdigest()
    (packet / "packet_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    result = verify_investor_packet(packet)

    assert result["ok"] is True
    assert result["execution_cta_scan"]["matches"] == []


def test_verify_investor_packet_requires_backtest_control_language(tmp_path):
    packet = _write_packet(tmp_path)
    backtest = packet / "evidence" / "backtest-report.md"
    backtest.write_text("# Backtest\nSCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE\n", encoding="utf-8")

    result = verify_investor_packet(packet)

    assert result["ok"] is False
    assert any("backtest_report missing synthetic fills" in error for error in result["errors"])


def test_verify_investor_packet_requires_backtest_non_executable_status(tmp_path):
    packet = _write_packet(tmp_path)
    backtest = packet / "evidence" / "backtest-report.md"
    backtest.write_text("# Backtest\nsynthetic fills\n", encoding="utf-8")

    result = verify_investor_packet(packet)

    assert result["ok"] is False
    assert any("backtest_report missing SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE" in error for error in result["errors"])


def test_external_use_gate_requires_two_counterparty_quote_verified_evidence(tmp_path):
    packet = _write_packet(tmp_path)
    _make_external_candidate_packet(packet, quote_records=[])

    result = verify_investor_packet(packet)

    assert result["external_use_gate"]["ok"] is False
    assert "two-counterparty quote-verified evidence missing" in result["external_use_gate"]["blockers"]


def test_external_use_gate_requires_licensed_replay_ready_source_intake(tmp_path):
    packet = _write_packet(tmp_path)
    _make_external_candidate_packet(
        packet,
        quote_records=[_quote_record("Dealer A"), _quote_record("Dealer B")],
        source_ready=False,
    )

    result = verify_investor_packet(packet)

    assert result["ok"] is True
    assert result["external_use_gate"]["ok"] is False
    assert "licensed/replay-ready source intake validation not ready" in result["external_use_gate"]["blockers"]
    assert result["external_use_gate"]["source_readiness"] == {
        "present": True,
        "ready": False,
        "summary": "0/6 licensed source groups covered",
        "errors": [],
    }


def test_external_use_gate_can_pass_only_with_legal_approval_source_readiness_and_two_distinct_quote_counterparties(tmp_path):
    packet = _write_packet(tmp_path)
    _make_external_candidate_packet(packet, quote_records=[_quote_record("Dealer A"), _quote_record("Dealer B")])

    result = verify_investor_packet(packet)

    assert result["ok"] is True
    assert result["external_use_gate"]["ok"] is True
    assert result["external_use_gate"]["quote_verified_candidates"] == 1
    assert result["external_use_gate"]["blockers"] == []
