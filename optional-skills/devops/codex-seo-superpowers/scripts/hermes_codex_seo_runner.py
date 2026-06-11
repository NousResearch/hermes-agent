#!/usr/bin/env python3
"""Artifact-first wrapper for Codex SEO + GEO Optimizer.

Examples:
  python hermes_codex_seo_runner.py /seo audit https://example.com --out ./audit
  python hermes_codex_seo_runner.py /seo geo https://example.com --out ./geo

Third-party engines are optional and installed outside the Hermes repo, by default under
`~/.hermes/vendor`. The wrapper labels missing engines instead of fabricating results.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

VENDOR_ROOT = Path(os.environ.get("HERMES_VENDOR_DIR", str(Path.home() / ".hermes" / "vendor"))).expanduser()
CODEX_ROOT = Path(os.environ.get("CODEX_SEO_ROOT", str(VENDOR_ROOT / "codex-seo"))).expanduser()
CODEX_PY = CODEX_ROOT / ".venv/bin/python"
CODEX_RUNNER = CODEX_ROOT / "scripts/run_skill_workflow.py"
GEO_BIN = Path(os.environ.get("GEO_OPTIMIZER_BIN", str(VENDOR_ROOT / "geo-optimizer-skill/.venv/bin/geo"))).expanduser()

COMMAND_TO_CODEX_SKILL = {
    "audit": "seo-audit",
    "page": "seo-page",
    "technical": "seo-technical",
    "content": "seo-content",
    "schema": "seo-schema",
    "images": "seo-images",
    "sitemap": "seo-sitemap",
    "geo": "seo-geo",
    "performance": "seo-performance",
    "visual": "seo-visual",
    "plan": "seo-plan",
    "programmatic": "seo-programmatic",
    "competitor-pages": "seo-competitor-pages",
    "hreflang": "seo-hreflang",
    "local": "seo-local",
    "maps": "seo-maps",
    "backlinks": "seo-backlinks",
    "cluster": "seo-cluster",
    "sxo": "seo-sxo",
    "drift": "seo-drift",
    "ecommerce": "seo-ecommerce",
}


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 240) -> dict:
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=timeout)
        return {
            "ok": p.returncode == 0,
            "exit_code": p.returncode,
            "cmd": cmd,
            "stdout": p.stdout[-20000:],
            "stderr": p.stderr[-12000:],
        }
    except Exception as exc:  # pragma: no cover - defensive wrapper
        return {"ok": False, "exit_code": None, "cmd": cmd, "error": repr(exc)}


def normalize_command(parts: list[str]) -> tuple[str, str, list[str]]:
    if parts and parts[0] == "/seo":
        parts = parts[1:]
    if len(parts) < 2:
        raise SystemExit("Expected: /seo <workflow> <url-or-seed>")
    workflow, target = parts[0], parts[1]
    return workflow, target, parts[2:]


def safe_domain(target: str) -> str:
    parsed = urlparse(target if re.match(r"^https?://", target) else "https://" + target)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", parsed.netloc or parsed.path)[:80] or "seo-target"


def write_plan(out: Path, workflow: str, target: str, summary: dict) -> None:
    lines = [
        f"# Hermes Codex SEO Superpower Plan — {target}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Workflow: `{workflow}`",
        "",
        "## Evidence artifacts",
        "",
    ]
    for name, path in summary.get("artifacts", {}).items():
        lines.append(f"- {name}: `{path}`")
    lines += [
        "",
        "## Next execution gates",
        "",
        "1. Review artifacts and classify findings as Confirmed / Likely / Needs credentials.",
        "2. For WordPress, check Yoast/RankMath/schema/robots/cache conflicts before implementation.",
        "3. Before deploying `llms.txt` or plugin changes: backup, purge cache, and verify public endpoints.",
        "4. Configure prompt/citation monitoring before claiming AI visibility improvement.",
    ]
    (out / "implementation-plan.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="+", help="Command-style prompt, e.g. /seo audit https://example.com")
    parser.add_argument("--out", help="Artifact directory")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args(argv)

    workflow, target, extra = normalize_command(args.command)
    skill = COMMAND_TO_CODEX_SKILL.get(workflow)
    if not skill:
        supported = ", ".join(sorted(COMMAND_TO_CODEX_SKILL))
        raise SystemExit(f"Unsupported /seo workflow: {workflow}. Supported: {supported}")

    out = Path(args.out or (Path.cwd() / f"codex-seo-{safe_domain(target)}-{workflow}"))
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workflow": workflow,
        "target": target,
        "codex_skill": skill,
        "extra": extra,
        "artifacts": {},
        "runs": {},
        "setup_required": [],
    }

    if CODEX_PY.exists() and CODEX_RUNNER.exists():
        codex_out = out / "codex-output"
        codex_out.mkdir(exist_ok=True)
        result = run(
            [str(CODEX_PY), str(CODEX_RUNNER), "--skill", skill, "--output-root", str(codex_out), "--json", target],
            cwd=CODEX_ROOT,
            timeout=args.timeout,
        )
        summary["runs"]["codex"] = result
        (out / "codex-run.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary["artifacts"]["codex_run"] = str(out / "codex-run.json")
        summary["artifacts"]["codex_output_dir"] = str(codex_out)
    else:
        summary["setup_required"].append(f"Codex SEO runner missing under {CODEX_ROOT}")

    if workflow in {"audit", "geo", "schema"}:
        if GEO_BIN.exists():
            geo_json = out / "geo-audit.json"
            geo_result = run([str(GEO_BIN), "audit", "--url", target, "--format", "json"], timeout=args.timeout)
            geo_json.write_text(geo_result.get("stdout") or json.dumps(geo_result, indent=2), encoding="utf-8")
            summary["runs"]["geo_audit"] = {k: v for k, v in geo_result.items() if k != "stdout"}
            summary["artifacts"]["geo_audit"] = str(geo_json)

            fix_txt = out / "geo-fix.txt"
            fix_result = run([str(GEO_BIN), "fix", "--url", target], timeout=args.timeout)
            fix_txt.write_text((fix_result.get("stdout") or "") + "\n\nSTDERR:\n" + (fix_result.get("stderr") or ""), encoding="utf-8")
            summary["runs"]["geo_fix"] = {k: v for k, v in fix_result.items() if k not in {"stdout", "stderr"}}
            summary["artifacts"]["geo_fix"] = str(fix_txt)
        else:
            summary["setup_required"].append(f"GEO Optimizer CLI missing at {GEO_BIN}")

    write_plan(out, workflow, target, summary)
    summary["artifacts"]["implementation_plan"] = str(out / "implementation-plan.md")
    (out / "run-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["artifacts"]["run_summary"] = str(out / "run-summary.json")
    print(json.dumps(summary, indent=2))
    return 0 if not summary["setup_required"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
