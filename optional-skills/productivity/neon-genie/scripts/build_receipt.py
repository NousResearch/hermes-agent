#!/usr/bin/env python3
"""Build a deterministic NeonGenieRunReceipt skeleton from inputs (stdlib only).

Does not grant authority. Emits PROPOSED packaging metadata + content hashes.

Usage:
  python scripts/build_receipt.py --profiles core,product_architecture --out receipt.json
  python scripts/build_receipt.py --packet path.json --profiles core --status SEALED
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import paths as ng_paths  # noqa: E402
import privacy_runtime  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


DATA_REQUEST_REQUIRED = (
    "field",
    "why_decision_critical",
    "sensitivity",
    "suggested_source",
    "blocks_promotion",
    "status",
)


def load_data_requests(path: Path) -> list[dict]:
    """Load one DataRequest object or an array of DataRequests; validate required keys."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FAIL: cannot read data-requests {path}: {exc}") from exc

    if isinstance(raw, dict):
        requests_list = [raw]
    elif isinstance(raw, list):
        requests_list = raw
    else:
        raise SystemExit(f"FAIL: --data-requests must be a JSON object or array: {path}")

    for i, req in enumerate(requests_list):
        if not isinstance(req, dict):
            raise SystemExit(f"FAIL: data-requests[{i}] must be an object")
        missing = [k for k in DATA_REQUEST_REQUIRED if k not in req]
        if missing:
            raise SystemExit(
                f"FAIL: data-requests[{i}] missing required keys: {', '.join(missing)}"
            )
    return requests_list


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Neon Genie run receipt skeleton")
    parser.add_argument(
        "--profiles",
        default="core",
        help="Comma-separated profiles loaded (default: core)",
    )
    parser.add_argument("--packet", type=Path, action="append", default=[], help="Input packet path (repeatable)")
    parser.add_argument("--input-hash", help="Optional precomputed input hash")
    parser.add_argument(
        "--status",
        default="PROPOSED",
        help="Receipt status (default PROPOSED; never implies execution authority)",
    )
    parser.add_argument(
        "--promotion-state",
        default="RAW_SIGNAL",
        help="Promotion ladder state (default RAW_SIGNAL)",
    )
    parser.add_argument("--out", type=Path, help="Write JSON to path (default stdout)")
    parser.add_argument(
        "--not-computable",
        default="",
        help="Comma-separated NOT_COMPUTABLE field names",
    )
    parser.add_argument(
        "--data-requests",
        type=Path,
        help="JSON file: one DataRequest object or array of DataRequests",
    )
    parser.add_argument(
        "--privacy-mode",
        default=None,
        choices=["local_only", "external_research_allowed", "custom"],
        help="Privacy mode (default: from --brief privacy section or local_only)",
    )
    parser.add_argument(
        "--brief",
        type=Path,
        help="Optional request brief YAML — privacy/research config flows into receipt",
    )
    parser.add_argument(
        "--privacy-context",
        type=Path,
        help="Optional JSON privacy context (overrides --privacy-mode defaults)",
    )
    parser.add_argument(
        "--external-actions",
        type=Path,
        help="Optional JSON array of external action records to enumerate on the receipt",
    )
    args = parser.parse_args(argv)

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    if "core" not in profiles:
        profiles = ["core"] + profiles
    if "privacy" not in profiles:
        profiles.append("privacy")

    packet_hashes: dict[str, str] = {}
    for path in args.packet:
        if not path.is_file():
            print(f"FAIL: packet not found: {path}", file=sys.stderr)
            return 1
        packet_hashes[str(path)] = sha256_file(path)

    requests_list: list[dict] = []
    if args.data_requests is not None:
        if not args.data_requests.is_file():
            print(f"FAIL: data-requests not found: {args.data_requests}", file=sys.stderr)
            return 1
        requests_list = load_data_requests(args.data_requests)

    try:
        version = ng_paths.version_path().read_text(encoding="utf-8").strip()
    except (OSError, FileNotFoundError):
        version = "unknown"
    material = {
        "skill": "neon-genie",
        "version": version,
        "profiles": profiles,
        "packet_hashes": packet_hashes,
        "status": args.status,
        "promotion_state": args.promotion_state,
    }
    content_hash = sha256_text(json.dumps(material, sort_keys=True, separators=(",", ":")))
    input_hash = args.input_hash or (
        sha256_text(json.dumps(packet_hashes, sort_keys=True)) if packet_hashes else content_hash
    )

    nc = [x.strip() for x in args.not_computable.split(",") if x.strip()]

    open_blocking = [
        r
        for r in requests_list
        if r.get("status") == "open" and r.get("blocks_promotion") is True
    ]

    note = "Packaging receipt only. Does not authorize spend, publish, or execution."
    if open_blocking and args.status == "PROPOSED":
        note += (
            " Open blocking data requests present; do not promote until satisfied or waived."
        )

    artifact_path = str(args.out.parent) if args.out else "out/neon-genie/example"
    try:
        if args.privacy_context is not None:
            if not args.privacy_context.is_file():
                print(f"FAIL: privacy-context not found: {args.privacy_context}", file=sys.stderr)
                return 1
            privacy = json.loads(args.privacy_context.read_text(encoding="utf-8"))
            privacy_runtime.assert_contract_version(privacy.get("contract_version"))
        elif args.brief is not None:
            if not args.brief.is_file():
                print(f"FAIL: brief not found: {args.brief}", file=sys.stderr)
                return 1
            privacy = privacy_runtime.resolve_privacy_config_from_brief(
                args.brief, artifact_path=artifact_path
            )
            if args.privacy_mode is not None:
                privacy = privacy_runtime.default_privacy_context(
                    artifact_path,
                    args.privacy_mode,
                    approved_domains=privacy.get("egress", {}).get("approved_domains"),
                    approved_tool_classes=privacy.get("egress", {}).get("approved_tool_classes"),
                    consents=privacy.get("consents"),
                )
        else:
            privacy = privacy_runtime.default_privacy_context(
                artifact_path, args.privacy_mode or "local_only"
            )

        if args.external_actions is not None:
            if not args.external_actions.is_file():
                print(
                    f"FAIL: external-actions not found: {args.external_actions}",
                    file=sys.stderr,
                )
                return 1
            actions_raw = json.loads(args.external_actions.read_text(encoding="utf-8"))
            if not isinstance(actions_raw, list):
                print("FAIL: --external-actions must be a JSON array", file=sys.stderr)
                return 1
            privacy = privacy_runtime.merge_privacy_context(
                privacy, external_actions=actions_raw
            )
        else:
            privacy_runtime.assert_envelope_privacy(privacy)
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        msg = privacy_runtime.sanitize_error_message(str(exc))
        print(f"FAIL: {msg}", file=sys.stderr)
        return 1

    privacy_fields = privacy_runtime.privacy_receipt_fields(privacy)
    external_actions = privacy_runtime.enumerate_external_actions(privacy)

    receipt = {
        "status": args.status,
        "profiles_loaded": profiles,
        "claims_by_label": {
            "OBSERVED": [],
            "INFERRED": [],
            "SPECULATIVE": [],
            "NOT_COMPUTABLE": nc,
        },
        "not_computable_fields": nc,
        "promotion_state": args.promotion_state,
        "human_review_required": True,
        "authority": "advisory_only",
        "grants_execution": False,
        "skill_version": version,
        "input_hash": input_hash,
        "output_hash": content_hash,
        "packet_hashes": packet_hashes,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "scripts/build_receipt.py",
        "note": note,
        "data_requests": requests_list,
        "open_blocking_requests": open_blocking,
        "research_attempts": [],
        "evidence_protocol": "find_request_not_computable",
        **privacy_fields,
        "external_actions": external_actions,
    }

    # Fail closed: receipt body must not embed raw secrets.
    try:
        privacy_runtime.assert_no_raw_secrets(
            json.dumps(receipt, sort_keys=True), context="run-receipt"
        )
    except ValueError as exc:
        print(f"FAIL: {privacy_runtime.sanitize_error_message(str(exc))}", file=sys.stderr)
        return 1

    text = json.dumps(receipt, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"PASS: wrote receipt {args.out}")
        print(f"  output_hash: {content_hash}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
