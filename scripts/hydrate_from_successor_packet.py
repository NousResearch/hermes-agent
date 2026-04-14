#!/usr/bin/env python3
"""Hydrate runtime continuity handoff state from successor packet artifacts (MEM-01)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SCHEMA_PATH = Path("docs/ops/schemas/successor_packet.schema.json")
DEFAULT_PACKET_PATH = Path("state/continuity/successor_packet/latest.json")
DEFAULT_STATE_PATH = Path("state/continuity/successor_packet/hydrated_state.json")
DEFAULT_ACK_LEDGER = Path("state/continuity/successor_packet/handover_ack.jsonl")
SCRIPT_VERSION = "mem01_successor_packet_hydration_v1_2026_04_03"


class HydrationError(RuntimeError):
    pass


def parse_iso(raw: Any) -> dt.datetime | None:
    txt = str(raw or "").strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(txt)
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def now_utc(now: str | None = None) -> dt.datetime:
    if isinstance(now, str) and now.strip():
        parsed = parse_iso(now)
        if parsed is None:
            raise HydrationError(f"invalid_now:{now}")
        return parsed
    return dt.datetime.now(dt.timezone.utc)


def now_iso(now: str | None = None) -> str:
    return now_utc(now).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(root: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HydrationError(f"json_parse_failed:{path}:{exc}") from exc
    if not isinstance(payload, dict):
        raise HydrationError(f"json_not_object:{path}")
    return payload


def ensure_readable_file(path: Path, *, key: str) -> None:
    if not path.exists() or not path.is_file():
        raise HydrationError(f"artifact_missing:{key}:{path}")
    if not os.access(path, os.R_OK):
        raise HydrationError(f"artifact_unreadable:{key}:{path}")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(stable_json(dict(row)) + "\n")


def validate_schema(payload: Mapping[str, Any], schema_path: Path) -> None:
    if Draft202012Validator is None or FormatChecker is None:
        raise HydrationError("schema_validator_unavailable")
    if not schema_path.exists() or not schema_path.is_file():
        raise HydrationError(f"schema_missing:{schema_path}")

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
    raise HydrationError(
        f"schema_validation_failed:data_path={data_ptr}:schema_path={schema_ptr}:error={err.message}"
    )


def verify_signature(packet: Mapping[str, Any]) -> None:
    signature = str(packet.get("validation_signature") or "").strip()
    if not signature:
        raise HydrationError("validation_signature_missing")
    payload = dict(packet)
    payload.pop("validation_signature", None)
    digest = hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()
    expected = f"sha256:{digest}"
    if signature != expected:
        raise HydrationError("validation_signature_mismatch")


def validate_packet_runtime(packet: Mapping[str, Any], *, root: Path, now: str | None) -> Dict[str, Any]:
    generated_at = parse_iso(packet.get("generated_at"))
    if generated_at is None:
        raise HydrationError("generated_at_invalid")

    freshness_seconds = int(packet.get("freshness_seconds") or 0)
    if freshness_seconds <= 0:
        raise HydrationError("freshness_seconds_invalid")

    now_dt = now_utc(now)
    age_seconds = (now_dt - generated_at).total_seconds()
    if age_seconds > freshness_seconds:
        raise HydrationError(f"packet_stale:age_seconds={age_seconds:.3f}:freshness_seconds={freshness_seconds}")

    artifact_refs = packet.get("artifact_references") if isinstance(packet.get("artifact_references"), dict) else {}
    if not artifact_refs:
        raise HydrationError("artifact_references_missing")
    if "continuity_current" not in artifact_refs:
        raise HydrationError("continuity_reference_missing:continuity_current")

    artifact_hashes = packet.get("artifact_hashes") if isinstance(packet.get("artifact_hashes"), dict) else {}
    if not artifact_hashes:
        raise HydrationError("artifact_hashes_missing")
    for key, raw_path in artifact_refs.items():
        artifact_path = resolve_path(root, str(raw_path))
        ensure_readable_file(artifact_path, key=str(key))
        expected_hash = str(artifact_hashes.get(key) or "")
        if not expected_hash:
            raise HydrationError(f"artifact_hash_missing:{key}")
        actual_hash = f"sha256:{sha256_file(artifact_path)}"
        if expected_hash != actual_hash:
            raise HydrationError(f"artifact_hash_mismatch:{key}")

    continuity_ref = artifact_refs.get("continuity_current")
    continuity_path = resolve_path(root, str(continuity_ref))
    ensure_readable_file(continuity_path, key="continuity_current")
    actual_hash = f"sha256:{sha256_file(continuity_path)}"
    packet_hash = str(packet.get("continuity_hash") or "")
    if packet_hash != actual_hash:
        raise HydrationError("continuity_hash_mismatch")
    if str(artifact_hashes.get("continuity_current") or "") != packet_hash:
        raise HydrationError("continuity_hash_not_aligned_with_artifact_hash")

    verify_signature(packet)

    return {
        "age_seconds": age_seconds,
        "freshness_seconds": freshness_seconds,
        "verified_artifact_count": len(artifact_refs),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    parser.add_argument("--packet-path", default=str(DEFAULT_PACKET_PATH))
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--ack-ledger", default=str(DEFAULT_ACK_LEDGER))
    parser.add_argument("--successor-session-id", default="")
    parser.add_argument("--write-state", action="store_true")
    parser.add_argument("--now", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    schema_path = resolve_path(root, args.schema_path)
    packet_path = resolve_path(root, args.packet_path)
    state_path = resolve_path(root, args.state_path)
    ack_ledger = resolve_path(root, args.ack_ledger)

    hydration_result = "failure"
    packet_id = None

    try:
        ensure_readable_file(packet_path, key="packet")
        packet = read_json(packet_path)
        packet_id = str(packet.get("packet_id") or "").strip() or None

        validate_schema(packet, schema_path)
        diagnostics = validate_packet_runtime(packet, root=root, now=args.now)

        hydrated_state = {
            "schema": "clawd.successor_packet.hydrated_state.v1",
            "hydrated_at": now_iso(args.now),
            "packet_id": packet.get("packet_id"),
            "readiness": packet.get("readiness"),
            "mutation_gate_status": packet.get("mutation_gate_status"),
            "continuity_hash": packet.get("continuity_hash"),
            "provenance_chain": list(packet.get("provenance_chain") or []),
            "next_slice_recommendation": packet.get("next_slice_recommendation"),
            "handover_instructions": packet.get("handover_instructions"),
            "generation_metadata": dict(packet.get("generation_metadata") or {}),
            "diagnostics": diagnostics,
        }

        if args.write_state:
            atomic_write_json(state_path, hydrated_state)

        hydration_result = "success"
        ack_row = {
            "packet_id": packet.get("packet_id"),
            "handover_at": now_iso(args.now),
            "successor_session_id": str(args.successor_session_id or "") or None,
            "hydration_result": hydration_result,
            "script_version": SCRIPT_VERSION,
        }
        append_jsonl(ack_ledger, ack_row)

        print(
            json.dumps(
                {
                    "ok": True,
                    "packet_id": packet.get("packet_id"),
                    "hydration_result": hydration_result,
                    "state_path": str(state_path) if args.write_state else None,
                    "ack_ledger": str(ack_ledger),
                    "diagnostics": diagnostics,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    except HydrationError as exc:
        ack_row = {
            "packet_id": packet_id,
            "handover_at": now_iso(args.now),
            "successor_session_id": str(args.successor_session_id or "") or None,
            "hydration_result": hydration_result,
            "error_code": str(exc),
            "script_version": SCRIPT_VERSION,
        }
        try:
            append_jsonl(ack_ledger, ack_row)
        except Exception:
            pass

        print(json.dumps({"ok": False, "error_code": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
