from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

ACCEPTED_FORMATS = {"csv", "jsonl", "parquet"}
FIXTURE_PROVENANCE = {"fixture", "manual_fixture", "demo_fixture", "synthetic_fixture"}

SOURCE_REQUIREMENTS: dict[str, list[str]] = {
    "IBIT options history": ["available_ts", "expiration", "strike", "option_type", "bid", "ask", "volume", "open_interest", "source_ref"],
    "Deribit options history": ["available_ts", "instrument_name", "underlying_price", "bid_iv", "ask_iv", "mark_iv", "open_interest", "source_ref"],
    "CME Bitcoin options history": ["available_ts", "symbol", "expiration", "strike", "option_type", "bid", "ask", "settlement", "source_ref"],
    "BTC reference history": ["available_ts", "btc_usd", "venue_or_index", "source_ref"],
    "IBIT holdings history": ["available_ts", "btc_per_share", "shares_outstanding", "fund_assets", "source_ref"],
    "Rates and fee curves": ["available_ts", "tenor", "rate", "borrow_or_fee", "source_ref"],
}


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _is_hex_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def validate_source_intake_manifest(manifest: dict[str, Any], *, decision_ts: datetime) -> dict[str, Any]:
    """Validate a licensed/replay-ready historical source intake manifest.

    This is an evidence-readiness validator only. Passing it means the source package
    is structurally replay-ready; it does not make any economics executable.
    """
    sources = manifest.get("sources") or []
    source_results: list[dict[str, Any]] = []
    blockers: list[str] = []
    ready_groups: set[str] = set()

    for source in sources:
        source_group = str(source.get("source_group") or "missing source group")
        required_fields = SOURCE_REQUIREMENTS.get(source_group)
        source_blockers: list[str] = []
        if not required_fields:
            source_blockers.append("unknown source group")
        else:
            present_fields = {str(field) for field in (source.get("fields") or [])}
            missing = [field for field in required_fields if field not in present_fields]
            if missing:
                source_blockers.append(f"missing required fields {', '.join(missing)}")

        provenance = str(source.get("provenance") or "").strip().lower()
        license_label = str(source.get("license_label") or "").strip().lower()
        if provenance in FIXTURE_PROVENANCE or license_label in FIXTURE_PROVENANCE:
            source_blockers.append("fixture/manual_fixture sources cannot satisfy readiness")
        if not provenance:
            source_blockers.append("provider/license provenance missing")
        if not license_label:
            source_blockers.append("license_label missing")

        file_format = str(source.get("format") or "").strip().lower()
        if file_format not in ACCEPTED_FORMATS:
            source_blockers.append("unsupported source format")

        if not _is_hex_sha256(source.get("raw_sha256")):
            source_blockers.append("raw_sha256 missing or invalid")

        row_count = int(source.get("row_count") or 0)
        if row_count <= 0:
            source_blockers.append("row_count must be positive")

        available_end = _parse_dt(source.get("available_end"))
        if available_end is None:
            source_blockers.append("available_end missing or invalid")
        elif available_end > decision_ts:
            source_blockers.append("available_end is after decision_ts")

        source_ready = not source_blockers
        if source_ready and source_group in SOURCE_REQUIREMENTS:
            ready_groups.add(source_group)
        for blocker in source_blockers:
            blockers.append(f"{source_group}: {blocker}")
        source_results.append(
            {
                "source_group": source_group,
                "ready": source_ready,
                "blockers": source_blockers,
                "row_count": row_count,
                "format": file_format,
                "provenance": provenance,
                "license_label": license_label,
            }
        )

    for source_group in SOURCE_REQUIREMENTS:
        if source_group not in {row["source_group"] for row in source_results}:
            blockers.append(f"{source_group}: source group missing")

    return {
        "ready": len(ready_groups) == len(SOURCE_REQUIREMENTS) and not blockers,
        "ok": len(ready_groups) == len(SOURCE_REQUIREMENTS) and not blockers,
        "evidence_status": "SCREEN-ONLY · SOURCE INTAKE VALIDATION · NOT EXECUTABLE",
        "covered_source_groups": len(ready_groups),
        "required_source_groups": len(SOURCE_REQUIREMENTS),
        "blockers": blockers,
        "source_results": source_results,
    }


def source_intake_template() -> dict[str, Any]:
    return {
        "decision_ts": "YYYY-MM-DDTHH:MM:SS+00:00",
        "evidence_status": "SCREEN-ONLY · SOURCE INTAKE TEMPLATE · NOT EXECUTABLE",
        "control_note": "Replace placeholders with licensed/replay-ready sources only. Do not include API keys, tokens, passwords, or connection strings.",
        "sources": [
            {
                "source_group": source_group,
                "provenance": "replace-with-licensed-provider-or-broker-export",
                "license_label": "replace-with-license-or-contract-label",
                "format": "csv",
                "raw_sha256": "replace-with-64-character-raw-file-sha256",
                "fields": fields,
                "row_count": 0,
                "available_start": "YYYY-MM-DDTHH:MM:SS+00:00",
                "available_end": "YYYY-MM-DDTHH:MM:SS+00:00",
                "source_ref": "replace-with-vendor-export-id-or-internal-file-ref",
            }
            for source_group, fields in SOURCE_REQUIREMENTS.items()
        ],
    }


def write_source_intake_template(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(source_intake_template(), indent=2) + "\n", encoding="utf-8")
    return path


def load_source_intake_manifest(manifest_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def _iso_or_empty(value: datetime | None) -> str:
    return value.isoformat() if value else ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv_metadata(path: Path) -> tuple[list[str], int, str]:
    latest: datetime | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        row_count = 0
        for row in reader:
            row_count += 1
            parsed = _parse_dt(row.get("available_ts"))
            if parsed and (latest is None or parsed > latest):
                latest = parsed
    return fields, row_count, _iso_or_empty(latest)


def _read_jsonl_metadata(path: Path) -> tuple[list[str], int, str]:
    fields: list[str] = []
    seen: set[str] = set()
    latest: datetime | None = None
    row_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row_count += 1
        row = json.loads(line)
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
        parsed = _parse_dt(row.get("available_ts"))
        if parsed and (latest is None or parsed > latest):
            latest = parsed
    return fields, row_count, _iso_or_empty(latest)


def build_source_manifest_entry(
    raw_path: str | Path,
    *,
    source_group: str,
    provenance: str,
    license_label: str,
    source_ref: str,
) -> dict[str, Any]:
    path = Path(raw_path)
    file_format = path.suffix.lower().lstrip(".")
    if file_format == "csv":
        fields, row_count, available_end = _read_csv_metadata(path)
    elif file_format == "jsonl":
        fields, row_count, available_end = _read_jsonl_metadata(path)
    elif file_format == "parquet":
        fields, row_count, available_end = [], 0, ""
    else:
        raise ValueError(f"unsupported source format: {file_format}")

    return {
        "source_group": source_group,
        "provenance": provenance,
        "license_label": license_label,
        "format": file_format,
        "raw_sha256": _sha256_file(path),
        "fields": fields,
        "row_count": row_count,
        "available_end": available_end,
        "source_ref": source_ref,
        "raw_path": str(path),
        "evidence_status": "SCREEN-ONLY · SOURCE INTAKE ENTRY · NOT EXECUTABLE",
    }


def write_source_intake_entry(entry: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    return path


def write_source_intake_validation_report(result: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blocker_lines = "\n".join(f"- {blocker}" for blocker in result.get("blockers", [])) or "- none"
    source_lines = "\n".join(
        f"- {row.get('source_group')}: {'ready' if row.get('ready') else 'blocked'}; rows={row.get('row_count')}; format={row.get('format')}; provenance={row.get('provenance')}; license={row.get('license_label')}"
        for row in result.get("source_results", [])
    ) or "- none"
    text = (
        "# Source Intake Validation Report\n\n"
        "**Evidence status:** `SCREEN-ONLY · SOURCE INTAKE VALIDATION · NOT EXECUTABLE`\n\n"
        "No executable quote, RFQ, advice, or investment readiness is implied by this report. It only validates whether historical source packages are structurally replay-ready.\n\n"
        f"- Ready: `{bool(result.get('ready'))}`\n"
        f"- Coverage: `{result.get('covered_source_groups', 0)}/{result.get('required_source_groups', len(SOURCE_REQUIREMENTS))} source groups structurally valid`\n"
        f"- Blocker count: `{len(result.get('blockers', []))}`\n\n"
        "## Blockers\n\n"
        f"{blocker_lines}\n\n"
        "## Source Results\n\n"
        f"{source_lines}\n"
    )
    path.write_text(text, encoding="utf-8")
    return path
