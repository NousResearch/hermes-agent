from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .integrity import IntegrityVerifier
from .models import (
    AdmissionRecord,
    AdmissionStatus,
    CandidateKind,
    CandidateSource,
    InspectionReport,
    PromotionDecision,
)
from .report import render_report
from .store import AdmissionStore

_VERIFIER = IntegrityVerifier()


def admission_store(hermes_home: Path | None = None) -> AdmissionStore:
    home = Path(
        hermes_home or os.getenv("HERMES_HOME") or (Path.home() / ".hermes")
    ).expanduser()
    return AdmissionStore(home / "admission")


def admission_record_id(kind: CandidateKind, name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "candidate"
    return f"{kind.value}-{slug}"


def _next_revision(kind: CandidateKind, name: str, hermes_home: Path | None = None) -> int:
    records = find_records(kind, name=name, hermes_home=hermes_home)
    if not records:
        return 1
    return max(record.revision for record in records) + 1


def _record_id_for_revision(lineage_id: str, revision: int) -> str:
    return lineage_id if revision == 1 else f"{lineage_id}-rev-{revision}"


def find_records(
    kind: CandidateKind | None = None,
    *,
    name: str | None = None,
    status: AdmissionStatus | None = None,
    hermes_home: Path | None = None,
) -> list[AdmissionRecord]:
    store = admission_store(hermes_home)
    records = store.list_records()
    if kind is not None:
        records = [record for record in records if record.kind == kind]
    if name is not None:
        base_id = admission_record_id(kind or CandidateKind.SKILL, name) if kind else None
        lowered = name.lower()
        filtered: list[AdmissionRecord] = []
        for record in records:
            if record.source.display_name.lower() == lowered:
                filtered.append(record)
                continue
            if base_id and (record.record_id == base_id or record.record_id.startswith(f"{base_id}-")):
                filtered.append(record)
        records = filtered
    if status is not None:
        records = [record for record in records if record.status == status]
    return sorted(records, key=lambda record: (record.updated_at, record.created_at, record.record_id), reverse=True)


def load_latest_record(
    kind: CandidateKind,
    name: str,
    *,
    statuses: tuple[AdmissionStatus, ...] | None = None,
    hermes_home: Path | None = None,
) -> AdmissionRecord:
    records = find_records(kind, name=name, hermes_home=hermes_home)
    if statuses is not None:
        records = [record for record in records if record.status in statuses]
    if not records:
        raise FileNotFoundError(f"No admission record found for {kind.value} '{name}'")
    return records[0]


def read_report(record: AdmissionRecord) -> str:
    if not record.report_path:
        raise FileNotFoundError(f"No report recorded for {record.record_id}")
    return Path(record.report_path).read_text(encoding="utf-8")


def quarantine_mcp_server(
    name: str,
    server_config: dict[str, Any],
    tools: list[tuple[str, str]],
    hermes_home: Path | None = None,
) -> AdmissionRecord:
    store = admission_store(hermes_home)
    lineage_id = admission_record_id(CandidateKind.MCP_SERVER, name)
    revision = _next_revision(CandidateKind.MCP_SERVER, name, hermes_home)
    record_id = _record_id_for_revision(lineage_id, revision)
    source_uri = server_config.get("url") or server_config.get("command") or name
    record = AdmissionRecord(
        record_id=record_id,
        kind=CandidateKind.MCP_SERVER,
        source=CandidateSource(
            uri=str(source_uri),
            display_name=name,
            installer="hermes mcp add",
        ),
        lineage_id=lineage_id,
        revision=revision,
        source_fingerprint=_fingerprint_payload(server_config),
    )

    artifact_path = store.candidate_quarantine_path(record_id, "server-config.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "name": name,
        "server_config": server_config,
        "tools": [{"name": tool_name, "description": desc} for tool_name, desc in tools],
    }
    artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")

    warnings: list[str] = []
    if server_config.get("url"):
        warnings.append("Remote HTTP MCP servers can execute tool actions outside Hermes.")
    if server_config.get("command"):
        warnings.append("Local stdio MCP servers execute local binaries under the user account.")

    report = InspectionReport(
        summary=f"Quarantined MCP server '{name}' after discovery of {len(tools)} tool(s).",
        decision=PromotionDecision.HOLD,
        capabilities=[tool_name for tool_name, _ in tools],
        warnings=warnings,
        reasons=["Explicit approval is required before the server is activated."],
    )
    report_path = store.write_report(record_id, render_report(record, report))

    record.quarantine_path = str(artifact_path)
    record.report_path = str(report_path)
    record.integrity = _VERIFIER.capture(artifact_path)
    store.save_record(record)
    return record


def load_quarantined_mcp_server(
    name: str,
    hermes_home: Path | None = None,
) -> tuple[AdmissionRecord, dict[str, Any]]:
    record = load_latest_record(
        CandidateKind.MCP_SERVER,
        name,
        statuses=(AdmissionStatus.QUARANTINED,),
        hermes_home=hermes_home,
    )
    if not record.quarantine_path:
        raise FileNotFoundError(f"No quarantined artifact recorded for MCP server '{name}'")
    payload = json.loads(Path(record.quarantine_path).read_text(encoding="utf-8"))
    return record, payload


def quarantine_skill_install(
    bundle_name: str,
    identifier: str,
    source: str,
    trust_level: str,
    category: str,
    bundle_files: dict[str, Any],
    metadata: dict[str, Any],
    scan_result: Any,
    quarantine_path: Path,
    hermes_home: Path | None = None,
) -> AdmissionRecord:
    store = admission_store(hermes_home)
    lineage_id = admission_record_id(CandidateKind.SKILL, bundle_name)
    revision = _next_revision(CandidateKind.SKILL, bundle_name, hermes_home)
    record_id = _record_id_for_revision(lineage_id, revision)
    record = AdmissionRecord(
        record_id=record_id,
        kind=CandidateKind.SKILL,
        source=CandidateSource(
            uri=identifier,
            display_name=bundle_name,
            version=metadata.get("version"),
            installer="hermes skills install",
        ),
        lineage_id=lineage_id,
        revision=revision,
        source_fingerprint=_fingerprint_payload({
            "identifier": identifier,
            "files": bundle_files,
        }),
    )

    manifest_path = Path(quarantine_path) / ".hermes-admission.json"
    manifest = {
        "name": bundle_name,
        "identifier": identifier,
        "source": source,
        "trust_level": trust_level,
        "category": category,
        "files": bundle_files,
        "metadata": metadata,
        "scan_result": asdict(scan_result),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = InspectionReport(
        summary=getattr(scan_result, "summary", f"{bundle_name}: pending approval"),
        decision=PromotionDecision.HOLD,
        capabilities=sorted(bundle_files.keys()),
        warnings=[
            f"{finding.severity}/{finding.category}: {finding.description}"
            for finding in getattr(scan_result, "findings", [])
        ],
        reasons=["Third-party skills remain quarantined until explicitly approved."],
    )
    report_path = store.write_report(record_id, render_report(record, report))

    record.quarantine_path = str(quarantine_path)
    record.report_path = str(report_path)
    record.integrity = _VERIFIER.capture(Path(quarantine_path))
    store.save_record(record)
    return record


def load_quarantined_skill_install(
    name: str,
    hermes_home: Path | None = None,
) -> tuple[AdmissionRecord, dict[str, Any]]:
    record = load_latest_record(
        CandidateKind.SKILL,
        name,
        statuses=(AdmissionStatus.QUARANTINED,),
        hermes_home=hermes_home,
    )
    if not record.quarantine_path:
        raise FileNotFoundError(f"No quarantined artifact recorded for skill '{name}'")
    manifest_path = Path(record.quarantine_path) / ".hermes-admission.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return record, payload


def verify_record_integrity(record: AdmissionRecord) -> bool:
    if not record.quarantine_path or not record.integrity:
        return False
    target = Path(record.quarantine_path)
    if not target.exists():
        return False
    return _VERIFIER.verify(target, record.integrity)


def mark_record_approved(
    record: AdmissionRecord,
    approved_path: str | None = None,
    integrity_path: Path | None = None,
    hermes_home: Path | None = None,
) -> AdmissionRecord:
    from .models import AdmissionStatus

    store = admission_store(hermes_home)
    if approved_path:
        record.approved_path = approved_path
    if integrity_path is not None:
        record.integrity = _VERIFIER.capture(integrity_path)
    record.approved_at = record.updated_at if record.status is AdmissionStatus.APPROVED else None
    record.transition_to(AdmissionStatus.APPROVED)
    record.approved_at = record.updated_at
    store.save_record(record)
    return record


def revoke_record(
    record: AdmissionRecord,
    note: str | None = None,
    hermes_home: Path | None = None,
) -> AdmissionRecord:
    from .models import AdmissionStatus

    store = admission_store(hermes_home)
    if note:
        record.notes.append(note)
    if record.status is not AdmissionStatus.REVOKED:
        record.transition_to(AdmissionStatus.REVOKED)
    store.save_record(record)
    return record


def reject_record(
    record: AdmissionRecord,
    note: str | None = None,
    hermes_home: Path | None = None,
) -> AdmissionRecord:
    store = admission_store(hermes_home)
    if note:
        record.notes.append(note)
    if record.status is not AdmissionStatus.REJECTED:
        record.transition_to(AdmissionStatus.REJECTED)
    store.save_record(record)
    return record


def verify_approved_record_integrity(record: AdmissionRecord) -> bool:
    if not record.approved_path or not record.integrity:
        return False
    target = Path(record.approved_path)
    if not target.exists():
        return False
    return _VERIFIER.verify(target, record.integrity)


def write_approved_json_snapshot(
    record: AdmissionRecord,
    artifact_name: str,
    payload: dict[str, Any],
    hermes_home: Path | None = None,
) -> Path:
    store = admission_store(hermes_home)
    destination = store.candidate_approved_path(record.record_id, artifact_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def requarantine_mcp_server(
    name: str,
    payload: dict[str, Any],
    hermes_home: Path | None = None,
    note: str = "re-quarantined after integrity drift",
    previous_record: AdmissionRecord | None = None,
) -> AdmissionRecord:
    fingerprint = _fingerprint_payload(payload)
    store = admission_store(hermes_home)
    lineage_id = admission_record_id(CandidateKind.MCP_SERVER, name)
    revision = _next_revision(CandidateKind.MCP_SERVER, name, hermes_home)
    record_id = f"{_record_id_for_revision(lineage_id, revision)}-drift-{fingerprint}"
    record = AdmissionRecord(
        record_id=record_id,
        kind=CandidateKind.MCP_SERVER,
        source=CandidateSource(
            uri=str(payload.get("server_config", {}).get("url") or payload.get("server_config", {}).get("command") or name),
            display_name=name,
            installer="integrity-drift",
        ),
        lineage_id=lineage_id,
        parent_record_id=previous_record.record_id if previous_record else None,
        revision=revision,
        source_fingerprint=fingerprint,
        notes=[note],
    )
    artifact_path = store.candidate_quarantine_path(record_id, "server-config.json")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = InspectionReport(
        summary=f"Re-quarantined MCP server '{name}' due to integrity drift.",
        decision=PromotionDecision.HOLD,
        capabilities=[tool["name"] for tool in payload.get("tools", []) if isinstance(tool, dict) and "name" in tool],
        reasons=[note],
    )
    report_path = store.write_report(record_id, render_report(record, report))
    record.quarantine_path = str(artifact_path)
    record.report_path = str(report_path)
    record.integrity = _VERIFIER.capture(artifact_path)
    store.save_record(record)
    return record


def requarantine_skill_directory(
    name: str,
    installed_path: Path,
    identifier: str,
    source: str,
    trust_level: str,
    category: str,
    metadata: dict[str, Any],
    scan_result: Any,
    hermes_home: Path | None = None,
    note: str = "re-quarantined after integrity drift",
    previous_record: AdmissionRecord | None = None,
) -> AdmissionRecord:
    store = admission_store(hermes_home)
    fingerprint = _VERIFIER.hash_path(installed_path)[:8]
    lineage_id = admission_record_id(CandidateKind.SKILL, name)
    revision = _next_revision(CandidateKind.SKILL, name, hermes_home)
    record_id = f"{_record_id_for_revision(lineage_id, revision)}-drift-{fingerprint}"
    record = AdmissionRecord(
        record_id=record_id,
        kind=CandidateKind.SKILL,
        source=CandidateSource(
            uri=identifier,
            display_name=name,
            version=metadata.get("version"),
            installer="integrity-drift",
        ),
        lineage_id=lineage_id,
        parent_record_id=previous_record.record_id if previous_record else None,
        revision=revision,
        source_fingerprint=fingerprint,
        notes=[note],
    )
    quarantine_dir = store.candidate_quarantine_path(record_id, name)
    quarantine_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(installed_path), str(quarantine_dir))
    manifest_path = quarantine_dir / ".hermes-admission.json"
    manifest = {
        "name": name,
        "identifier": identifier,
        "source": source,
        "trust_level": trust_level,
        "category": category,
        "files": _read_directory_contents(quarantine_dir),
        "metadata": metadata,
        "scan_result": asdict(scan_result),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = InspectionReport(
        summary=getattr(scan_result, "summary", f"{name}: re-quarantined"),
        decision=PromotionDecision.HOLD,
        capabilities=sorted(manifest["files"].keys()),
        warnings=[note],
        reasons=["Approval is required again after integrity drift."],
    )
    report_path = store.write_report(record_id, render_report(record, report))
    record.quarantine_path = str(quarantine_dir)
    record.report_path = str(report_path)
    record.integrity = _VERIFIER.capture(quarantine_dir)
    store.save_record(record)
    return record


def _read_directory_contents(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for child in sorted(path.rglob("*")):
        if child.is_file():
            result[str(child.relative_to(path))] = child.read_text(encoding="utf-8")
    return result


def _fingerprint_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return _VERIFIER.algorithm + "-" + __import__("hashlib").new(_VERIFIER.algorithm, raw).hexdigest()[:8]
