#!/usr/bin/env python3
"""Bootstrap run_0 baselines and validate Hermes AI-Scientist fork templates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAY_ROOT = REPO_ROOT / "scripts" / "merge_tools" / "overlays" / "ai-scientist"
VENDOR_ROOT = REPO_ROOT / "vendor" / "openclaw-mirror" / "AI-Scientist"
FORK_TEMPLATES = ("nc_kan", "nc_kan_proof", "hermes_self_evolve")
REQUIRED_FILES = ("experiment.py", "plot.py", "prompt.json", "seed_ideas.json")


def _template_roots(vendor: bool) -> list[Path]:
    base = VENDOR_ROOT if vendor else OVERLAY_ROOT
    return [base / "templates" / name for name in FORK_TEMPLATES]


def bootstrap_baselines(vendor: bool = False) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for template_dir in _template_roots(vendor):
        if not template_dir.is_dir():
            results.append({"template": template_dir.name, "status": "missing"})
            continue
        cmd = [sys.executable, "experiment.py", "--out_dir=run_0"]
        proc = subprocess.run(cmd, cwd=str(template_dir), capture_output=True, text=True)
        baseline = template_dir / "run_0" / "final_info.json"
        results.append(
            {
                "template": template_dir.name,
                "status": "ok" if proc.returncode == 0 and baseline.is_file() else "failed",
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-500:],
            }
        )
    return results


def verify_templates(vendor: bool = False) -> dict[str, object]:
    issues: list[str] = []
    checked: list[str] = []
    for template_dir in _template_roots(vendor):
        if not template_dir.is_dir():
            issues.append(f"missing template dir: {template_dir}")
            continue
        checked.append(template_dir.name)
        for fname in REQUIRED_FILES:
            if not (template_dir / fname).is_file():
                issues.append(f"{template_dir.name}: missing {fname}")
        baseline = template_dir / "run_0" / "final_info.json"
        if not baseline.is_file():
            issues.append(f"{template_dir.name}: missing run_0/final_info.json (run bootstrap)")
        else:
            try:
                json.loads(baseline.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                issues.append(f"{template_dir.name}: invalid run_0/final_info.json")
    return {"checked": checked, "issues": issues, "ok": not issues}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify/bootstrap Hermes AI-Scientist fork templates.")
    parser.add_argument("--bootstrap", action="store_true", help="Run experiment.py --out_dir=run_0 for fork templates.")
    parser.add_argument("--vendor", action="store_true", help="Target vendor tree instead of tracked overlay source.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.bootstrap:
        results = bootstrap_baselines(vendor=args.vendor)
        print(json.dumps(results, indent=2))
    report = verify_templates(vendor=args.vendor)
    print("Template verification:", "OK" if report["ok"] else "FAILED")
    for issue in report["issues"]:
        print(f"  - {issue}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
