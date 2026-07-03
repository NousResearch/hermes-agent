#!/usr/bin/env python
"""Generate a local proof bundle for Hermes OpenAI Agents SDK hardening.

The bundle is intentionally read-only. It does not push, fetch, create issues,
create Kanban cards, restart services, or call live models. It summarizes local
state and hashes the artifacts that prove the current SDK hardening posture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCOPED_PATHS = [
    "plugins/openai_agents",
    "tests/plugins/test_openai_agents_bridge.py",
    "docs/openai-agents-governed-runtime-spec.md",
    "docs/openai-agents-git-workflow.md",
    "docs/openai-agents-project-tracking.json",
    "docs/openai-agents-source-manifest.json",
    "schemas/openai-agents-receipt.schema.json",
    "evals/openai_agents/governance_cases.json",
    "scripts/check_openai_agents_quality.py",
    "scripts/generate_openai_agents_proof_bundle.py",
    "pyproject.toml",
    "uv.lock",
]


def _run(cmd: list[str], *, check: bool = False) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return {"cmd": cmd, "exit_code": proc.returncode, "output": proc.stdout.strip()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        return Path.home() / ".hermes"


def _latest_receipts(limit: int) -> list[dict[str, Any]]:
    receipt_dir = _hermes_home() / "receipts" / "openai_agents"
    if not receipt_dir.exists():
        return []
    receipts: list[dict[str, Any]] = []
    for path in sorted(receipt_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        item: dict[str, Any] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
        try:
            payload = _json_file(path)
            result = payload.get("result") if isinstance(payload, dict) else {}
            item.update({
                "success": payload.get("success") if isinstance(payload, dict) else None,
                "lane": payload.get("lane") if isinstance(payload, dict) else None,
                "workflow": payload.get("workflow") if isinstance(payload, dict) else None,
                "status": result.get("status") if isinstance(result, dict) else None,
                "model": payload.get("model") if isinstance(payload, dict) else None,
                "usage": payload.get("usage") if isinstance(payload, dict) else None,
            })
        except Exception as exc:
            item["parse_error"] = f"{type(exc).__name__}: {exc}"
        receipts.append(item)
    return receipts


def _tracked_artifact_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in SCOPED_PATHS:
        path = ROOT / rel
        if path.is_file():
            hashes[rel] = _sha256_file(path)
    return hashes


def build_bundle(*, receipt_limit: int) -> dict[str, Any]:
    tracking_path = ROOT / "docs/openai-agents-project-tracking.json"
    source_path = ROOT / "docs/openai-agents-source-manifest.json"
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_id": "OASDK-HARDENING",
        "scope": "local proof bundle only; no external side effects",
        "git": {
            "status_short_branch": _run(["git", "status", "--short", "--branch"]),
            "head": _run(["git", "rev-parse", "HEAD"]),
            "head_short": _run(["git", "rev-parse", "--short", "HEAD"]),
            "branch": _run(["git", "branch", "--show-current"]),
            "origin": _run(["git", "remote", "get-url", "origin"]),
            "recent_commits": _run(["git", "log", "--oneline", "--decorate", "-12"]),
            "diff_check": _run(["git", "diff", "--check", "--", *SCOPED_PATHS]),
        },
        "tracked_artifact_hashes": _tracked_artifact_hashes(),
        "tracking_manifest": {
            "path": str(tracking_path),
            "sha256": _sha256_file(tracking_path),
            "summary": {
                "roadmap_items": len(_json_file(tracking_path).get("roadmap_items", [])),
                "receipt_groups": len(_json_file(tracking_path).get("receipt_groups", [])),
                "next_actions": len(_json_file(tracking_path).get("next_actions", [])),
            },
        },
        "source_manifest": {
            "path": str(source_path),
            "sha256": _sha256_file(source_path),
            "source_count": len(_json_file(source_path).get("official_sources", [])),
        },
        "latest_openai_agent_receipts": _latest_receipts(receipt_limit),
        "next_required_actions": [
            "Run python scripts/check_openai_agents_quality.py after generating the bundle.",
            "Restart gateway and start /new before claiming Telegram runtime uses latest SDK retry code.",
            "Rebase/push/PR/GitHub/Kanban actions are external-scope work and should be recorded separately."
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="Output JSON path. Defaults under HERMES_HOME receipts/openai_agents/proof_bundles/.")
    parser.add_argument("--receipt-limit", type=int, default=10)
    args = parser.parse_args()

    bundle = build_bundle(receipt_limit=max(1, args.receipt_limit))
    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out_path = _hermes_home() / "receipts" / "openai_agents" / "proof_bundles" / f"{ts}-oasdk-proof-bundle.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    digest = _sha256_file(out_path)
    print(json.dumps({"proof_bundle_path": str(out_path), "proof_bundle_sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
