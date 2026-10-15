#!/usr/bin/env python3
"""Build the canonical NeonGenieRunEnvelope for a run directory (stdlib only).

Every packaging recipe and sealed run should emit run-envelope.json so
downstream agents open one entry point.

Usage:
  python scripts/build_envelope.py --out-dir out/neon-genie/run1
  python scripts/build_envelope.py --out-dir out/neon-genie/run1 --recipe product-audit \\
      --brief examples/product-audit.brief.yaml --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import lineage  # noqa: E402
import paths as ng_paths  # noqa: E402
import privacy_runtime  # noqa: E402

ENVELOPE_SCHEMA_ID = "neon-genie/run-envelope"
ENVELOPE_SCHEMA_VERSION = "1.0.0"
SKIP_NAMES = {
    "run-envelope.json",
    "_data_requests.json",
    ".gitkeep",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        try:
            return path.relative_to(SKILL_ROOT).as_posix()
        except ValueError:
            return path.as_posix()


def collect_json_artifacts(out_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(out_dir.rglob("*.json")):
        if p.name in SKIP_NAMES:
            continue
        if p.name.startswith("."):
            continue
        files.append(p)
    return files


def build_envelope(
    out_dir: Path,
    *,
    recipe: str | None = None,
    brief: Path | None = None,
    route: dict[str, Any] | None = None,
    request_summary: str | None = None,
    mode_status: dict[str, str] | None = None,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    if not out_dir.is_dir():
        raise FileNotFoundError(f"NG-SCHEMA-001: out dir missing: {out_dir}")

    receipt_path = out_dir / "run-receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"NG-SCHEMA-002: run-receipt.json required before envelope: {receipt_path}"
        )
    receipt = _load(receipt_path)
    privacy = receipt.get("privacy") or privacy_runtime.default_privacy_context(str(out_dir))
    try:
        privacy_runtime.assert_envelope_privacy(privacy)
    except ValueError as exc:
        raise ValueError(privacy_runtime.sanitize_error_message(str(exc))) from exc

    route = route or {}
    route_path = out_dir / "profile-route.json"
    if not route and route_path.is_file():
        route = _load(route_path)

    profiles = (
        list(route.get("selected") or [])
        or list(receipt.get("profiles_loaded") or [])
        or ["core"]
    )
    if "core" not in profiles:
        profiles = ["core"] + profiles
    if "privacy" not in profiles:
        profiles.append("privacy")

    summary_path = out_dir / "recipe-summary.json"
    summary = _load(summary_path) if summary_path.is_file() else {}
    recipe = recipe or summary.get("recipe")
    if brief is None and summary.get("brief"):
        candidate = SKILL_ROOT / str(summary["brief"])
        if candidate.is_file():
            brief = candidate

    skill_ver = lineage.skill_version(SKILL_ROOT)
    rid = lineage.run_id()
    req_id = lineage.request_id()

    artifacts_meta: list[dict[str, Any]] = []
    parent_ids: list[str] = []
    for path in collect_json_artifacts(out_dir):
        kind = lineage.guess_artifact_type(path)
        aid = lineage.artifact_id(kind)
        entry = {
            "artifact_id": aid,
            "type": kind,
            "path": path.name if path.parent == out_dir else _rel(path, out_dir),
            "content_hash": lineage.sha256_file(path),
            "parent_artifact_ids": list(parent_ids[:1]),
            "schema_version": "1.0.0",
        }
        if kind == "run_receipt":
            entry["schema_id"] = "neon-genie/run-receipt"
        elif kind.endswith("_packet") or kind in {
            "commercial_simulation",
            "agentic_service_graph",
        }:
            entry["schema_id"] = f"neon-genie/{kind.replace('_', '-')}"
        artifacts_meta.append(entry)
        parent_ids.append(aid)

    # Prefer primary by type preference
    primary = None
    by_type = {a["type"]: a for a in artifacts_meta}
    for pref in lineage.PRIMARY_PREFERENCE:
        if pref in by_type and pref != "run_receipt":
            primary = by_type[pref]
            break
    if primary is None:
        # fallback: first non-receipt non-summary
        for a in artifacts_meta:
            if a["type"] not in {"run_receipt", "recipe_summary", "profile_route", "data_request"}:
                primary = a
                break
    if primary is None:
        primary = by_type.get("run_receipt") or {
            "type": "run_receipt",
            "path": "run-receipt.json",
            "artifact_id": lineage.artifact_id("run_receipt"),
            "content_hash": lineage.sha256_file(receipt_path),
        }

    data_requests = receipt.get("data_requests") or []
    open_blocking = receipt.get("open_blocking_requests") or []
    gates_failed = receipt.get("gates_failed") or []
    gate_results = []
    for g in gates_failed:
        gate_results.append({"gate": str(g), "status": "FAIL", "detail": "from receipt"})
    if receipt.get("gate"):
        gate_results.append(
            {
                "gate": str(receipt["gate"]),
                "status": "FAIL" if str(receipt.get("status")).upper() in {"GATE_FAIL", "NOT_COMPUTABLE"} else "INFO",
            }
        )

    modes = mode_status or {
        "OPEN": "complete",
        "ALIGN": "complete",
        "ASCEND": "complete",
        "CLEAR": "complete",
        "SEAL": "complete",
    }

    wayfinder_path = None
    for a in artifacts_meta:
        if a["type"] == "wayfinder_execution_packet":
            wayfinder_path = a["path"]
            break

    envelope: dict[str, Any] = {
        "schema_id": ENVELOPE_SCHEMA_ID,
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "run_id": rid,
        "request_id": req_id,
        "skill": "neon-genie",
        "skill_version": skill_ver,
        "created_at": lineage.utc_now(),
        "authority": "advisory_only",
        "grants_execution": False,
        "request": {
            "request_id": req_id,
            "summary": request_summary
            or (f"recipe:{recipe}" if recipe else "neon-genie packaging run"),
            "brief_path": _rel(brief, SKILL_ROOT) if brief else summary.get("brief"),
            "recipe": recipe,
            "requested_profiles": list(route.get("preferred") or []),
        },
        "resolved_profiles": profiles,
        "selected_profiles": profiles,  # v0 alias
        "mode_status": modes,
        "primary_artifact": {
            "type": primary["type"],
            "path": primary["path"],
            "artifact_id": primary["artifact_id"],
            "content_hash": primary.get("content_hash"),
        },
        "artifacts": artifacts_meta,
        "outputs": artifacts_meta,  # v0 alias
        "data_requests": data_requests,
        "gate_results": gate_results,
        "promotion": {
            "state": receipt.get("promotion_state") or receipt.get("status") or "RAW_SIGNAL",
            "blocking_gates": [str(g) for g in gates_failed],
            "open_blocking_requests": open_blocking,
        },
        "receipt": {
            "path": "run-receipt.json",
            "status": receipt.get("status"),
            "promotion_state": receipt.get("promotion_state"),
            "authority": receipt.get("authority", "advisory_only"),
            "grants_execution": bool(receipt.get("grants_execution", False)),
            "content_hash": lineage.sha256_file(receipt_path),
        },
        "receipt_path": "run-receipt.json",
        "generator": {
            "skill": "neon-genie",
            "version": skill_ver,
            "tool": "scripts/build_envelope.py",
        },
        "wayfinder": {
            "ingest": "run-envelope.json",
            "product_intent_changes_require_neon_genie_review": True,
            "handoff_path": wayfinder_path,
        },
        "canonical_sources": receipt.get("canonical_sources") or [],
        "assumptions": receipt.get("assumptions") or [],
        **privacy_runtime.privacy_receipt_fields(privacy),
        "external_actions": privacy_runtime.enumerate_external_actions(privacy),
    }

    # Fail closed: envelope must not embed raw secrets.
    privacy_runtime.assert_no_raw_secrets(
        json.dumps({k: v for k, v in envelope.items() if k != "content_hash"}, sort_keys=True),
        context="run-envelope",
    )

    # Optional schema fields must be omitted rather than emitted as null.
    envelope["request"] = {k: v for k, v in envelope["request"].items() if v is not None}
    envelope["wayfinder"] = {k: v for k, v in envelope["wayfinder"].items() if v is not None}

    # content_hash over body without content_hash
    body = {k: v for k, v in envelope.items() if k != "content_hash"}
    envelope["content_hash"] = lineage.sha256_json(body)
    return envelope


def write_envelope(out_dir: Path, envelope: dict[str, Any] | None = None, **kwargs: Any) -> Path:
    env = envelope or build_envelope(out_dir, **kwargs)
    path = out_dir / "run-envelope.json"
    path.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Neon Genie run-envelope.json")
    parser.add_argument("--out-dir", type=Path, required=True, help="Run output directory")
    parser.add_argument("--recipe", default=None)
    parser.add_argument("--brief", type=Path, default=None)
    parser.add_argument("--summary", default=None, help="Request summary text")
    parser.add_argument(
        "--write",
        action="store_true",
        default=True,
        help="Write run-envelope.json (default true)",
    )
    parser.add_argument("--stdout", action="store_true", help="Print envelope JSON")
    parser.add_argument("--validate", action="store_true", help="Validate against schema")
    args = parser.parse_args(argv)

    out_dir = args.out_dir if args.out_dir.is_absolute() else SKILL_ROOT / args.out_dir
    try:
        env = build_envelope(
            out_dir,
            recipe=args.recipe,
            brief=args.brief,
            request_summary=args.summary,
        )
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    if args.write:
        path = write_envelope(out_dir, env)
        print(f"PASS: wrote {path}")
        print(f"  run_id: {env['run_id']}")
        print(f"  primary: {env['primary_artifact']['type']} → {env['primary_artifact']['path']}")

    if args.validate:
        schema_path = ng_paths.schema_file("run-envelope.schema.json")
        # lightweight required-key check + optional full validate via validate_packet
        required = [
            "schema_id",
            "schema_version",
            "run_id",
            "skill_version",
            "authority",
            "resolved_profiles",
            "mode_status",
            "primary_artifact",
            "artifacts",
            "promotion",
            "receipt",
            "generator",
        ]
        missing = [k for k in required if k not in env]
        if missing:
            print(f"FAIL: NG-SCHEMA-003: envelope missing {missing}", file=sys.stderr)
            return 1
        if env.get("authority") != "advisory_only" or env.get("grants_execution") is True:
            print("FAIL: NG-SCHEMA-004: envelope must remain advisory_only", file=sys.stderr)
            return 1
        # full schema if available
        r = __import__("subprocess").run(
            [
                sys.executable,
                str(SCRIPT_DIR / "validate_packet.py"),
                "--packet",
                str(out_dir / "run-envelope.json"),
                "--type",
                "envelope",
            ],
            cwd=SKILL_ROOT,
        )
        if r.returncode != 0:
            return r.returncode
        print("PASS: envelope validates")

    if args.stdout:
        print(json.dumps(env, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
