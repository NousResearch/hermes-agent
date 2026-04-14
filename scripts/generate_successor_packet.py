#!/usr/bin/env python3
"""Generate compact successor packet artifacts for runtime continuity handoff (MEM-01)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

try:  # pragma: no cover - import surface validated in integration tests
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SCHEMA_PATH = Path("docs/ops/schemas/successor_packet.schema.json")
DEFAULT_CURRENT_PATH = Path("state/continuity/current.json")
DEFAULT_OUTPUT_PATH = Path("state/continuity/successor_packet/latest.json")
DEFAULT_ARCHIVE_DIR = Path("state/continuity/successor_packet/archive")
DEFAULT_LEDGER_PATH = Path("state/continuity/successor_packet/ledger.jsonl")
DEFAULT_FRESHNESS_SECONDS = 300
SCRIPT_VERSION = "mem01_successor_packet_runtime_v1_2026_04_03"

DEFAULT_ARTIFACT_REFS = {
    "execution_board": "reports/openclaw_full_maturity_roadmap_execution_board_2026-04-02.md",
    "continuity_current": "state/continuity/current.json",
    "continuity_read_pointer": "state/continuity/latest/continuity_read_pointer.json",
    "verify_last": "state/continuity/latest/verify_last.json",
}


class PacketGenerationError(RuntimeError):
    pass


def now_utc(now: str | None = None) -> dt.datetime:
    if isinstance(now, str) and now.strip():
        parsed = parse_iso(now)
        if parsed is None:
            raise PacketGenerationError(f"invalid_now:{now}")
        return parsed
    return dt.datetime.now(dt.timezone.utc)


def now_iso(now: str | None = None) -> str:
    return now_utc(now).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso(value: Any) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(text)
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PacketGenerationError(f"json_parse_failed:{path}:{exc}") from exc
    if not isinstance(payload, dict):
        raise PacketGenerationError(f"json_not_object:{path}")
    return payload


def resolve_path(root: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def ensure_readable_file(path: Path, *, key: str) -> None:
    if not path.exists() or not path.is_file():
        raise PacketGenerationError(f"artifact_missing:{key}:{path}")
    if not os.access(path, os.R_OK):
        raise PacketGenerationError(f"artifact_unreadable:{key}:{path}")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(stable_json(dict(row)) + "\n")


def parse_artifact_refs(raw_rows: Iterable[str]) -> Dict[str, str]:
    refs: Dict[str, str] = {}
    for row in raw_rows:
        token = str(row or "").strip()
        if not token:
            continue
        if "=" not in token:
            raise PacketGenerationError(f"artifact_ref_invalid:{token}")
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise PacketGenerationError(f"artifact_ref_invalid:{token}")
        refs[key] = value
    return refs


def validate_schema(payload: Mapping[str, Any], schema_path: Path) -> None:
    if Draft202012Validator is None or FormatChecker is None:
        raise PacketGenerationError("schema_validator_unavailable")
    if not schema_path.exists() or not schema_path.is_file():
        raise PacketGenerationError(f"schema_missing:{schema_path}")

    schema_doc = read_json(schema_path)
    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(dict(payload)),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return

    err = errors[0]
    data_ptr = "$" if not err.absolute_path else "$/" + "/".join(str(x) for x in err.absolute_path)
    schema_ptr = "$" if not err.absolute_schema_path else "$/" + "/".join(str(x) for x in err.absolute_schema_path)
    raise PacketGenerationError(
        f"schema_validation_failed:data_path={data_ptr}:schema_path={schema_ptr}:error={err.message}"
    )


def compute_validation_signature(packet_without_signature: Mapping[str, Any]) -> str:
    digest = sha256_bytes(stable_json(packet_without_signature).encode("utf-8"))
    return f"sha256:{digest}"


def _load_previous_chain(latest_path: Path) -> list[str]:
    if not latest_path.exists() or not latest_path.is_file():
        return []
    try:
        prev = read_json(latest_path)
    except PacketGenerationError:
        return []

    out: list[str] = []
    prev_id = str(prev.get("packet_id") or "").strip()
    if prev_id:
        out.append(prev_id)
    prev_chain = prev.get("provenance_chain")
    if isinstance(prev_chain, list):
        for item in prev_chain:
            token = str(item or "").strip()
            if token:
                out.append(token)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        if item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped[:10]


def build_packet(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    root = Path(args.root).resolve()
    schema_path = resolve_path(root, args.schema_path)
    current_path = resolve_path(root, args.current_path)
    latest_path = resolve_path(root, args.output_path)

    ensure_readable_file(current_path, key="continuity_current")
    current_obj = read_json(current_path)

    readiness = str(current_obj.get("readiness") or "").strip()
    mutation_gate = current_obj.get("mutation_gate") if isinstance(current_obj.get("mutation_gate"), dict) else {}
    mutation_gate_status = str(mutation_gate.get("status") or "").strip()

    allowed_readiness = {"READY", "RECONCILE_REQUIRED"} if args.allow_reconcile else {"READY"}
    if readiness not in allowed_readiness:
        raise PacketGenerationError(f"eligibility_gate_blocked:readiness={readiness or 'UNKNOWN'}")
    if mutation_gate_status != "allowed":
        raise PacketGenerationError(f"eligibility_gate_blocked:mutation_gate_status={mutation_gate_status or 'UNKNOWN'}")

    artifact_refs = parse_artifact_refs(args.artifact_ref or [])
    if not artifact_refs:
        artifact_refs = dict(DEFAULT_ARTIFACT_REFS)
    if "continuity_current" not in artifact_refs:
        raise PacketGenerationError("artifact_ref_required:continuity_current")

    resolved_refs: Dict[str, str] = {}
    artifact_hashes: Dict[str, str] = {}
    for key, raw_path in artifact_refs.items():
        resolved = resolve_path(root, raw_path)
        ensure_readable_file(resolved, key=key)
        if key == "continuity_current" and resolved != current_path:
            raise PacketGenerationError("artifact_ref_mismatch:continuity_current")
        resolved_refs[key] = str(resolved)
        artifact_hashes[key] = f"sha256:{sha256_file(resolved)}"

    continuity_hash = f"sha256:{sha256_file(current_path)}"
    generated_at = now_iso(args.now)
    packet_id = f"pkt_{uuid.uuid4()}"

    warnings = ((current_obj.get("coherence") or {}).get("connector_warning_reasons") or []) if isinstance(current_obj.get("coherence"), dict) else []
    warning_channels = [str(item).strip() for item in warnings if str(item).strip()]

    packet: Dict[str, Any] = {
        "packet_version": "1.0.0",
        "packet_id": packet_id,
        "generated_at": generated_at,
        "continuity_hash": continuity_hash,
        "readiness": readiness,
        "mutation_gate_status": mutation_gate_status,
        "freshness_seconds": int(args.freshness_seconds),
        "artifact_references": resolved_refs,
        "artifact_hashes": artifact_hashes,
        "provenance_chain": _load_previous_chain(latest_path),
        "generation_metadata": {
            "worker_lane": str(args.worker_lane),
            "model_selection": str(args.model_selection),
            "script_version": SCRIPT_VERSION,
            "generation_trigger": str(args.generation_trigger),
        },
    }

    if warning_channels:
        packet["warning_channels"] = warning_channels
    if args.next_slice_recommendation:
        packet["next_slice_recommendation"] = str(args.next_slice_recommendation)
    if args.handover_instructions:
        packet["handover_instructions"] = str(args.handover_instructions)

    packet["validation_signature"] = compute_validation_signature(packet)
    validate_schema(packet, schema_path)

    diagnostics = {
        "root": str(root),
        "schema_path": str(schema_path),
        "current_path": str(current_path),
        "latest_path": str(latest_path),
    }
    return packet, diagnostics


def write_and_validate_artifacts(args: argparse.Namespace, packet: Mapping[str, Any]) -> Dict[str, Any]:
    root = Path(args.root).resolve()
    schema_path = resolve_path(root, args.schema_path)
    current_path = resolve_path(root, args.current_path)
    latest_path = resolve_path(root, args.output_path)
    archive_dir = resolve_path(root, args.archive_dir)
    ledger_path = resolve_path(root, args.ledger_path)

    archive_path = (archive_dir / f"{packet['packet_id']}.json").resolve()
    atomic_write_json(latest_path, packet)
    atomic_write_json(archive_path, packet)

    latest_obj = read_json(latest_path)
    validate_schema(latest_obj, schema_path)

    expected_continuity_hash = f"sha256:{sha256_file(current_path)}"
    if str(latest_obj.get("continuity_hash") or "") != expected_continuity_hash:
        latest_path.unlink(missing_ok=True)
        archive_path.unlink(missing_ok=True)
        raise PacketGenerationError("post_generation_validation_failed:continuity_hash_mismatch")

    refs = latest_obj.get("artifact_references") if isinstance(latest_obj.get("artifact_references"), dict) else {}
    hashes = latest_obj.get("artifact_hashes") if isinstance(latest_obj.get("artifact_hashes"), dict) else {}
    for key, raw_path in refs.items():
        path = resolve_path(root, str(raw_path))
        ensure_readable_file(path, key=str(key))
        expected_hash = str(hashes.get(key) or "")
        actual_hash = f"sha256:{sha256_file(path)}"
        if expected_hash and expected_hash != actual_hash:
            latest_path.unlink(missing_ok=True)
            archive_path.unlink(missing_ok=True)
            raise PacketGenerationError(f"post_generation_validation_failed:artifact_hash_mismatch:{key}")

    ledger_row = {
        "event": "SUCCESSOR_PACKET_APPLIED",
        "generated_at": str(packet.get("generated_at")),
        "packet_id": str(packet.get("packet_id")),
        "continuity_hash": str(packet.get("continuity_hash")),
        "latest_path": str(latest_path),
        "archive_path": str(archive_path),
        "validation_gates_passed": [
            "eligibility_gate",
            "artifact_reference_validation_gate",
            "schema_validation_gate",
            "post_generation_validation_gate",
        ],
        "script_version": SCRIPT_VERSION,
    }
    append_jsonl(ledger_path, ledger_row)

    return {
        "latest_path": str(latest_path),
        "archive_path": str(archive_path),
        "ledger_path": str(ledger_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--current-path", default=str(DEFAULT_CURRENT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE_DIR))
    parser.add_argument("--ledger-path", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--freshness-seconds", type=int, default=DEFAULT_FRESHNESS_SECONDS)
    parser.add_argument("--worker-lane", default="core_execution")
    parser.add_argument("--model-selection", default="codex-worker-plus-4")
    parser.add_argument("--generation-trigger", default="continuity_checkpoint")
    parser.add_argument("--artifact-ref", action="append", default=[])
    parser.add_argument("--next-slice-recommendation", default="")
    parser.add_argument("--handover-instructions", default="")
    parser.add_argument("--allow-reconcile", action="store_true")
    parser.add_argument("--now", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if int(args.freshness_seconds) <= 0:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_code": "invalid_freshness_seconds",
                    "freshness_seconds": args.freshness_seconds,
                },
                ensure_ascii=False,
            )
        )
        return 1

    try:
        packet, diagnostics = build_packet(args)
        paths = write_and_validate_artifacts(args, packet)
    except PacketGenerationError as exc:
        failure = {
            "event": "SUCCESSOR_PACKET_FAILED",
            "failed_at": now_iso(args.now),
            "error_code": str(exc),
            "script_version": SCRIPT_VERSION,
        }
        try:
            append_jsonl(resolve_path(Path(args.root).resolve(), args.ledger_path), failure)
        except Exception:
            pass
        print(json.dumps({"ok": False, "error_code": str(exc)}, ensure_ascii=False))
        return 1

    result = {
        "ok": True,
        "packet_id": packet.get("packet_id"),
        "generated_at": packet.get("generated_at"),
        "continuity_hash": packet.get("continuity_hash"),
        "freshness_seconds": packet.get("freshness_seconds"),
        "paths": paths,
        "diagnostics": diagnostics,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
