from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from .quote_evidence import load_quote_evidence_ledger

EVIDENCE_STATUS = "SCREEN-ONLY · NOT EXECUTABLE"
FORBIDDEN_EXECUTION_CTAS = [
    "Submit RFQ",
    "Execute Trade",
    "Request Quote",
    "Trade now",
    "Buy now",
    "Open account",
    "Start trading",
]
SECRET_PATTERNS = [
    r"api[_-]?key",
    r"password",
    r"secret",
    r"connection string",
    r"bearer\s+[A-Za-z0-9._-]+",
]
CONTROL_REQUIRED_LABELS = {"investor_memo", "investor_tearsheet", "packet_index", "site_index"}
EXPECTED_PACKET_LABELS = {
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def _scan_text_body(text: str, patterns: list[str], *, regex: bool = False) -> list[str]:
    matches: list[str] = []
    if not text:
        return matches
    folded = text.casefold()
    for pattern in patterns:
        found = re.search(pattern, text, re.IGNORECASE) if regex else pattern.casefold() in folded
        if found:
            display = pattern
            if regex and isinstance(found, re.Match):
                display = found.group(0)
            if display not in matches:
                matches.append(display)
    return matches


def _is_raw_capture_member(name: str) -> bool:
    return Path(name).name.startswith("raw_")


def _scan_zip_text_members(
    path: Path,
    patterns: list[str],
    *,
    regex: bool = False,
    skip_raw_captures: bool = False,
) -> list[str]:
    matches: list[str] = []
    try:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                if skip_raw_captures and _is_raw_capture_member(name):
                    continue
                try:
                    text = zf.read(name).decode("utf-8")
                except UnicodeDecodeError:
                    continue
                for match in _scan_text_body(text, patterns, regex=regex):
                    display = f"{path.name}:{name}:{match}"
                    if display not in matches:
                        matches.append(display)
    except zipfile.BadZipFile:
        pass
    return matches


def _scan_text(
    packet_dir: Path,
    patterns: list[str],
    *,
    regex: bool = False,
    skip_raw_zip_captures: bool = False,
) -> list[str]:
    matches: list[str] = []
    for path in packet_dir.rglob("*"):
        if not path.is_file():
            continue
        text = _read_text(path)
        for display in _scan_text_body(text, patterns, regex=regex):
            if display not in matches:
                matches.append(display)
        if path.suffix.lower() == ".zip":
            for display in _scan_zip_text_members(path, patterns, regex=regex, skip_raw_captures=skip_raw_zip_captures):
                if display not in matches:
                    matches.append(display)
    return matches


def _packet_hash_without_self(manifest: dict[str, Any]) -> str:
    payload = {k: v for k, v in manifest.items() if k != "packet_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def verify_investor_packet(packet_dir: str | Path) -> dict[str, Any]:
    base = Path(packet_dir)
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = base / "packet_manifest.json"
    if not manifest_path.exists():
        return {
            "ok": False,
            "run_id": None,
            "packet_dir": str(base),
            "errors": ["packet_manifest.json missing"],
            "warnings": [],
            "artifact_count": 0,
            "packet_sha256_ok": False,
            "secret_scan": {"matches": []},
            "execution_cta_scan": {"matches": []},
            "control_language": {"ok": False, "missing": []},
        }

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "run_id": None,
            "packet_dir": str(base),
            "errors": [f"packet_manifest.json invalid JSON: {exc}"],
            "warnings": [],
            "artifact_count": 0,
            "packet_sha256_ok": False,
            "secret_scan": {"matches": []},
            "execution_cta_scan": {"matches": []},
            "control_language": {"ok": False, "missing": []},
        }

    expected_packet_hash = manifest.get("packet_sha256")
    actual_packet_hash = _packet_hash_without_self(manifest)
    packet_sha256_ok = expected_packet_hash == actual_packet_hash
    if not packet_sha256_ok:
        errors.append("packet_sha256 mismatch")

    if manifest.get("evidence_status") != EVIDENCE_STATUS:
        errors.append("packet evidence_status missing SCREEN-ONLY · NOT EXECUTABLE")
    publishability = manifest.get("publishability")
    if publishability not in {"internal-diligence-only", "external-use-approved"}:
        errors.append("packet publishability must be internal-diligence-only or external-use-approved")

    artifacts = manifest.get("artifacts") or []
    checked = []
    control_missing: list[str] = []
    legal_wrapper_present = False
    legal_wrapper_approved = False
    legal_wrapper_status = "missing"
    legal_wrapper_errors: list[str] = []
    quote_evidence_summary: dict[str, Any] | None = None
    quote_evidence_errors: list[str] = []
    source_readiness_present = False
    source_readiness_ready = False
    source_readiness_summary = "missing"
    source_readiness_errors: list[str] = []
    for artifact in artifacts:
        label = str(artifact.get("label") or "unknown")
        rel = artifact.get("path")
        if not rel:
            errors.append(f"{label} path missing")
            continue
        rel_path = Path(str(rel))
        if rel_path.is_absolute():
            errors.append(f"{label} path escapes packet_dir: {rel}")
            continue
        path = (base / rel_path).resolve()
        try:
            path.relative_to(base.resolve())
        except ValueError:
            errors.append(f"{label} path escapes packet_dir: {rel}")
            continue
        if not path.exists():
            errors.append(f"{label} missing: {rel}")
            continue
        checked.append(label)
        actual_bytes = path.stat().st_size
        expected_bytes = artifact.get("bytes")
        if expected_bytes != actual_bytes:
            errors.append(f"{label} byte size mismatch")
        actual_sha = _sha256_file(path)
        if artifact.get("sha256") != actual_sha:
            errors.append(f"{label} sha256 mismatch")
        if label in CONTROL_REQUIRED_LABELS:
            text = _read_text(path)
            if "SCREEN-ONLY" not in text or "NOT EXECUTABLE" not in text:
                control_missing.append(label)
                errors.append(f"{label} missing SCREEN-ONLY · NOT EXECUTABLE control language")
        if label == "backtest_report":
            text = _read_text(path)
            if "SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE" not in text:
                errors.append("backtest_report missing SCREEN-ONLY · BACKTEST ONLY · NOT EXECUTABLE control language")
            if "synthetic fills" not in text.casefold():
                errors.append("backtest_report missing synthetic fills control language")
        if label == "legal_wrapper_json":
            legal_wrapper_present = True
            try:
                legal_wrapper = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                legal_wrapper_errors.append(f"legal_wrapper_json invalid JSON: {exc}")
                errors.append(f"legal_wrapper_json invalid JSON: {exc}")
            else:
                legal_wrapper_approved = bool(legal_wrapper.get("approved_by_counsel"))
                legal_wrapper_status = str(legal_wrapper.get("status") or ("approved" if legal_wrapper_approved else "draft-blocked"))
        if label == "quote_evidence":
            quote_ledger = load_quote_evidence_ledger(path)
            quote_evidence_summary = quote_ledger.get("summary") or {}
            quote_evidence_errors = [
                f"quote_evidence invalid row {row.get('line_number')}: {', '.join(map(str, row.get('errors') or []))}"
                for row in quote_ledger.get("invalid_rows", [])
            ]
            errors.extend(quote_evidence_errors)
        if label == "site_data":
            try:
                site_data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                source_readiness_errors.append(f"site_data invalid JSON: {exc}")
                errors.append(f"site_data invalid JSON: {exc}")
            else:
                validation = site_data.get("source_intake_validation") or {}
                source_readiness_present = bool(validation)
                source_readiness_ready = bool(validation.get("ready"))
                covered = validation.get("covered_source_groups")
                required = validation.get("required_source_groups")
                if covered is not None and required is not None:
                    source_readiness_summary = f"{covered}/{required} licensed source groups covered"
                else:
                    source_readiness_summary = str(validation.get("summary") or validation.get("status") or "source readiness not reported")

    missing_manifest_artifacts = manifest.get("missing_artifacts") or []
    represented_labels = set(checked) | {str(label) for label in missing_manifest_artifacts}
    absent_expected_labels = sorted(EXPECTED_PACKET_LABELS - represented_labels)
    for label in absent_expected_labels:
        errors.append(f"required packet artifact label not represented: {label}")

    secret_matches = _scan_text(base, SECRET_PATTERNS, regex=True)
    if secret_matches:
        errors.append("secret markers found")
    cta_matches = _scan_text(base, FORBIDDEN_EXECUTION_CTAS, skip_raw_zip_captures=True)
    if cta_matches:
        errors.append("forbidden execution CTA text found")

    if missing_manifest_artifacts:
        warnings.append(f"builder reported missing artifacts: {', '.join(map(str, missing_manifest_artifacts))}")
        errors.append("packet manifest contains missing_artifacts")

    external_use_blockers: list[str] = []
    quote_verified_candidates = int((quote_evidence_summary or {}).get("quote_verified_candidates") or 0)
    if publishability != "external-use-approved":
        external_use_blockers.append("packet publishability is not external-use-approved")
    if not legal_wrapper_present:
        external_use_blockers.append("legal wrapper JSON artifact missing")
    elif not legal_wrapper_approved:
        external_use_blockers.append("counsel-approved legal wrapper missing")
    if "quote_evidence" not in checked:
        external_use_blockers.append("quote evidence ledger artifact missing")
    elif quote_verified_candidates < 1:
        external_use_blockers.append("two-counterparty quote-verified evidence missing")
    if quote_evidence_errors:
        external_use_blockers.append("quote evidence ledger contains invalid rows")
    if "site_data" not in checked:
        external_use_blockers.append("site data artifact missing")
    elif not source_readiness_present:
        external_use_blockers.append("licensed/replay-ready source intake validation missing")
    elif not source_readiness_ready:
        external_use_blockers.append("licensed/replay-ready source intake validation not ready")
    if source_readiness_errors:
        external_use_blockers.append("source readiness validation contains invalid data")
    external_use_ok = not external_use_blockers

    return {
        "ok": not errors,
        "run_id": manifest.get("run_id"),
        "packet_dir": str(base),
        "evidence_status": manifest.get("evidence_status"),
        "publishability": manifest.get("publishability"),
        "legal_wrapper": {
            "present": legal_wrapper_present,
            "approved_by_counsel": legal_wrapper_approved,
            "status": legal_wrapper_status,
            "errors": legal_wrapper_errors,
        },
        "external_use_gate": {
            "ok": external_use_ok,
            "blockers": external_use_blockers,
            "quote_verified_candidates": quote_verified_candidates,
            "source_readiness": {
                "present": source_readiness_present,
                "ready": source_readiness_ready,
                "summary": source_readiness_summary,
                "errors": source_readiness_errors,
            },
        },
        "artifact_count": len(artifacts),
        "checked_artifacts": checked,
        "packet_sha256_expected": expected_packet_hash,
        "packet_sha256_actual": actual_packet_hash,
        "packet_sha256_ok": packet_sha256_ok,
        "secret_scan": {"matches": secret_matches},
        "execution_cta_scan": {"matches": cta_matches},
        "control_language": {"ok": not control_missing, "missing": control_missing},
        "errors": errors,
        "warnings": warnings,
    }
