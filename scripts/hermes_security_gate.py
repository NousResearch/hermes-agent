#!/usr/bin/env python3
"""Validate Hermes security/dependency gate configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _check_file(root: Path, rel: str, label: str) -> dict[str, Any]:
    path = root / rel
    return {"id": label, "path": rel, "ok": path.exists(), "detail": "exists" if path.exists() else "missing"}


def _check_renovate(root: Path) -> dict[str, Any]:
    path = root / "renovate.json"
    if not path.exists():
        return {"id": "renovate", "path": "renovate.json", "ok": False, "detail": "missing"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"id": "renovate", "path": "renovate.json", "ok": False, "detail": str(exc)}
    automerge = data.get("automerge", False)
    ok = automerge is False
    return {
        "id": "renovate",
        "path": "renovate.json",
        "ok": ok,
        "detail": "automerge disabled" if ok else "automerge must be false",
    }


def check_security_gate(root: Path) -> dict[str, Any]:
    checks = [
        _check_file(root, ".github/workflows/osv-scanner.yml", "osv-scanner"),
        _check_file(root, ".github/workflows/supply-chain-audit.yml", "supply-chain-audit"),
        _check_file(root, ".semgrep/hermes-security.yml", "semgrep-rules"),
        _check_renovate(root),
    ]
    passed = sum(1 for check in checks if check["ok"])
    score = int(round((passed / len(checks)) * 100))
    return {
        "ok": passed == len(checks),
        "score": score,
        "remaining": 100 - score,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Hermes security gate files")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = check_security_gate(args.root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"SECURITY_GATE {report['score']} {report['remaining']}")
        for check in report["checks"]:
            print(f"{check['id']} {100 if check['ok'] else 0} {0 if check['ok'] else 100} {check['detail']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
