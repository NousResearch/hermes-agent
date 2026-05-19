from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from institutional_btc_vol.investor_memo import write_investor_memo
from institutional_btc_vol.investor_site import write_investor_site
from institutional_btc_vol.investor_tearsheet import write_tearsheet
from institutional_btc_vol.site_data import build_site_data, write_site_data

EVIDENCE_STATUS = "SCREEN-ONLY · NOT EXECUTABLE"
PUBLISHABILITY = "internal-diligence-only"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_if_present(label: str, source: Any, target: Path, missing: list[str]) -> Path | None:
    if not source:
        missing.append(label)
        return None
    src = Path(str(source))
    if not src.exists():
        missing.append(label)
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    return target


def _copy_optional(label: str, source: Any, target: Path) -> Path | None:
    if not source:
        return None
    src = Path(str(source))
    if not src.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    return target


def _make_evidence_manifest_portable(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    for artifact in manifest.get("artifacts", []):
        label = str(artifact.get("label") or "artifact")
        suffix = Path(str(artifact.get("path") or "")).suffix
        artifact["path"] = f"artifacts/{label}{suffix}"
        artifact["path_context"] = "inside latest-evidence-bundle.zip"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _artifact(label: str, path: Path, packet_dir: Path) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path.relative_to(packet_dir)),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _packet_hash_without_self(manifest: dict[str, Any]) -> str:
    payload = {k: v for k, v in manifest.items() if k != "packet_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _render_index(manifest: dict[str, Any]) -> str:
    artifact_lines = "\n".join(
        f"- `{artifact['path']}` — {artifact['label']} — `{artifact['sha256']}`" for artifact in manifest.get("artifacts", [])
    )
    return f"""# BTC Vol Desk Investor Packet

**Run ID:** `{manifest.get('run_id', 'missing')}`
**Status:** {manifest.get('evidence_status', EVIDENCE_STATUS)}
**Publishability:** `{manifest.get('publishability', PUBLISHABILITY)}`
**Packet SHA-256:** see `packet_manifest.json`

> Not a client portal. Not an execution venue. Not a fund offering. This packet is for internal/investor diligence review only.

## Contents

{artifact_lines}

## Required Verification

1. Review `packet_manifest.json`.
2. Confirm artifact SHA-256 values match local files.
3. Confirm the latest evidence bundle remains independently verifiable.
4. Treat all economics as SCREEN-ONLY · NOT EXECUTABLE until quote/execution evidence and legal wrapper are approved.
"""


def build_investor_packet(data_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    data_base = Path(data_dir)
    packet_dir = Path(output_dir)
    packet_dir.mkdir(parents=True, exist_ok=True)

    data = build_site_data(data_base)
    latest = data.get("latest_run") or {}
    run_id = latest.get("run_id", "missing")

    memo_path = write_investor_memo(data, packet_dir / "btc-vol-desk-investor-memo.md")
    tearsheet_path = write_tearsheet(data, packet_dir / "one-page-tear-sheet.md")
    site_path = write_investor_site(data, packet_dir / "site" / "index.html")
    site_data_path = write_site_data(data_base, packet_dir / "site" / "site-data.json")

    missing: list[str] = []
    evidence_targets = {
        "latest_report": _copy_if_present("latest_report", latest.get("report_path"), packet_dir / "evidence" / "latest-report.md", missing),
        "latest_evidence_bundle": _copy_if_present("latest_evidence_bundle", latest.get("evidence_bundle_path"), packet_dir / "evidence" / "latest-evidence-bundle.zip", missing),
        "evidence_manifest": _copy_if_present("evidence_manifest", latest.get("evidence_manifest_path"), packet_dir / "evidence" / "evidence_manifest.json", missing),
        "candidate_triage": _copy_if_present("candidate_triage", latest.get("candidate_ledger_path"), packet_dir / "evidence" / "candidate_triage.jsonl", missing),
        "quote_evidence": _copy_optional("quote_evidence", latest.get("quote_evidence_ledger_path"), packet_dir / "evidence" / "quote_evidence.jsonl"),
        "backtest_report": _copy_optional("backtest_report", (data.get("backtest_research") or {}).get("path"), packet_dir / "evidence" / "backtest-report.md"),
        "legal_wrapper_json": _copy_if_present("legal_wrapper_json", (data.get("legal_wrapper_package") or {}).get("path"), packet_dir / "evidence" / "legal-wrapper-package-v1.json", missing),
        "legal_wrapper_md": _copy_if_present("legal_wrapper_md", str(Path((data.get("legal_wrapper_package") or {}).get("path", "")).with_suffix(".md")) if (data.get("legal_wrapper_package") or {}).get("path") else None, packet_dir / "evidence" / "legal-wrapper-package-v1.md", missing),
    }
    _make_evidence_manifest_portable(evidence_targets["evidence_manifest"])

    artifact_paths: list[tuple[str, Path]] = [
        ("investor_memo", memo_path),
        ("investor_tearsheet", tearsheet_path),
        ("site_index", site_path),
        ("site_data", site_data_path),
    ]
    artifact_paths.extend((label, path) for label, path in evidence_targets.items() if path is not None)

    preliminary_manifest: dict[str, Any] = {
        "run_id": run_id,
        "evidence_status": data.get("evidence_status", EVIDENCE_STATUS),
        "publishability": PUBLISHABILITY,
        "artifacts": [_artifact(label, path, packet_dir) for label, path in artifact_paths],
        "missing_artifacts": missing,
    }
    index_path = packet_dir / "packet_index.md"
    index_path.write_text(_render_index(preliminary_manifest), encoding="utf-8")
    artifact_paths.append(("packet_index", index_path))

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "evidence_status": data.get("evidence_status", EVIDENCE_STATUS),
        "publishability": PUBLISHABILITY,
        "artifacts": [_artifact(label, path, packet_dir) for label, path in artifact_paths],
        "missing_artifacts": missing,
    }
    manifest["packet_sha256"] = _packet_hash_without_self(manifest)
    manifest_path = packet_dir / "packet_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "ok": not missing,
        "run_id": run_id,
        "evidence_status": manifest["evidence_status"],
        "publishability": PUBLISHABILITY,
        "packet_dir": str(packet_dir),
        "packet_manifest_path": str(manifest_path),
        "packet_index_path": str(index_path),
        "memo_path": str(memo_path),
        "tearsheet_path": str(tearsheet_path),
        "site_path": str(site_path),
        "packet_sha256": manifest["packet_sha256"],
        "missing_artifacts": missing,
    }
