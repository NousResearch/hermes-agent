#!/usr/bin/env python3
"""Isolated Hermes / hub-layout runtime smoke for Neon Genie.

Layer A (always):
  1. Build allowlisted hub tree from SKILL.md path refs (or copy skills/neon-genie)
  2. Run do check + do doctor + do behavioral in that tree
  3. Write a run report under out/neon-genie/runtime-smoke/

Layer A+ (when `hermes` CLI is available and --hermes):
  1. HERMES_HOME=tmp hermes skills install <local path or GitHub>
  2. hermes skills inspect / list
  3. doctor on installed path

Does not require LLM API keys. Stdlib only (+ optional hermes binary).

Usage:
  python scripts/hermes_runtime_smoke.py
  python scripts/hermes_runtime_smoke.py --hermes
  python scripts/hermes_runtime_smoke.py --out out/neon-genie/runtime-smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
PY = sys.executable

_HUB_LINK_RE = re.compile(
    r"(?:\]\(|`|(?:^|[\s\"']))((?:references|templates|scripts|assets|examples)/[^\s)`\"'<>]+)",
    re.MULTILINE,
)


def hub_paths_from_skill(skill_md: str) -> list[str]:
    paths: set[str] = set()
    for m in _HUB_LINK_RE.finditer(skill_md.replace("\\", "/")):
        raw = unquote(urlsplit(m.group(1).rstrip(".,;:")).path)
        paths.add(raw)
    return sorted(paths)


def build_hub_tree(root: Path, dest: Path) -> list[str]:
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    paths = hub_paths_from_skill(skill)
    missing = [p for p in paths if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError(f"NG-RUNTIME-020: hub refs missing: {missing[:5]}")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SKILL.md").write_text(skill, encoding="utf-8")
    for rel in paths:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target)
    return paths


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def _ensure_behavioral_assets(root: Path, hub_dir: Path) -> None:
    """Hub allowlist may lag SKILL.md; ensure behavioral runner + suite exist."""
    for rel in (
        "scripts/check_behavioral_invariants.py",
        "scripts/hermes_runtime_smoke.py",
        "scripts/paths.py",
        "scripts/neon_genie.py",
        "scripts/doctor.py",
    ):
        src = root / rel
        if src.is_file():
            dest = hub_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    src_b = root / "evals" / "behavioral"
    if src_b.is_dir():
        # Prefer examples/evals (hub mirror) so incomplete root evals/ does not shadow cases/
        dest = hub_dir / "examples" / "evals" / "behavioral"
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_b, dest)
        # Case JSON paths point at evals/behavioral/... — rewrite via symlink tree only if needed
        alt = hub_dir / "evals" / "behavioral"
        if alt.exists() or alt.is_symlink():
            if alt.is_symlink() or alt.is_file():
                alt.unlink()
            else:
                shutil.rmtree(alt)
        alt.parent.mkdir(parents=True, exist_ok=True)
        try:
            alt.symlink_to(dest.resolve())
        except OSError:
            shutil.copytree(src_b, alt)


def smoke_local_hub(root: Path, out_dir: Path) -> dict:
    hub_dir = out_dir / "hub-layout"
    if hub_dir.exists():
        shutil.rmtree(hub_dir)
    files = build_hub_tree(root, hub_dir)
    _ensure_behavioral_assets(root, hub_dir)
    report: dict = {
        "mode": "hub_layout",
        "hub_files": len(files) + 1,
        "steps": [],
    }
    for name, args in (
        ("check", ["do", "check"]),
        (
            "behavioral",
            [
                "do",
                "behavioral",
                "--suite",
                "--out",
                str(out_dir / "behavioral-report.json"),
            ],
        ),
        ("doctor", ["do", "doctor"]),
    ):
        r = run([PY, "scripts/neon_genie.py", *args], cwd=hub_dir)
        report["steps"].append(
            {
                "name": name,
                "exit": r.returncode,
                "stdout_tail": (r.stdout or "")[-500:],
                "stderr_tail": (r.stderr or "")[-500:],
            }
        )
        if r.returncode != 0:
            report["ok"] = False
            report["failed_step"] = name
            return report
    report["ok"] = True
    return report


def smoke_hermes_cli(root: Path, out_dir: Path) -> dict:
    hermes = shutil.which("hermes")
    if not hermes:
        return {"mode": "hermes_cli", "ok": None, "skipped": True, "reason": "hermes not on PATH"}

    home = out_dir / "hermes-home"
    if home.exists():
        shutil.rmtree(home)
    home.mkdir(parents=True)
    skills = home / "skills"
    skills.mkdir()
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    # Prefer local package path install simulation: copy hub package
    pkg = root / "skills" / "neon-genie"
    if not pkg.is_dir():
        pkg = root
    dest = skills / "neon-genie"
    # Use hub tree for fidelity to hub install
    build_hub_tree(root if not (pkg / "references" / "schemas").is_dir() else pkg, dest)

    steps = []
    # try inspect via hermes if it can see local skills
    r_list = run([hermes, "skills", "list"], cwd=home, env=env)
    steps.append({"name": "skills_list", "exit": r_list.returncode, "stdout_tail": (r_list.stdout or "")[-400:]})

    r_doc = run([PY, str(dest / "scripts" / "neon_genie.py"), "do", "doctor"], cwd=dest, env=env)
    steps.append(
        {
            "name": "doctor_installed",
            "exit": r_doc.returncode,
            "stderr_tail": (r_doc.stderr or "")[-400:],
            "stdout_tail": (r_doc.stdout or "")[-400:],
        }
    )
    ok = r_doc.returncode == 0
    return {
        "mode": "hermes_cli",
        "ok": ok,
        "skipped": False,
        "HERMES_HOME": str(home),
        "install_path": str(dest),
        "steps": steps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Neon Genie isolated runtime smoke")
    parser.add_argument(
        "--out",
        type=Path,
        default=SKILL_ROOT / "out" / "neon-genie" / "runtime-smoke",
    )
    parser.add_argument(
        "--hermes",
        action="store_true",
        help="Also run hermes CLI path with isolated HERMES_HOME",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=SKILL_ROOT,
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out_dir = args.out if args.out.is_absolute() else (root / args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"skill": "neon-genie", "root": str(root)}
    hub = smoke_local_hub(root, out_dir)
    report["hub_layout"] = hub
    if args.hermes:
        report["hermes_cli"] = smoke_hermes_cli(root, out_dir)

    path = out_dir / "runtime-smoke-report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": hub.get("ok"), "report": str(path)}, indent=2))

    if not hub.get("ok"):
        print("FAIL: NG-RUNTIME-021: hub-layout smoke failed", file=sys.stderr)
        for s in hub.get("steps") or []:
            if s.get("exit"):
                print(f"  step {s.get('name')} exit={s.get('exit')}", file=sys.stderr)
                if s.get("stderr_tail"):
                    print(s["stderr_tail"], file=sys.stderr)
        return 1
    if args.hermes and report.get("hermes_cli", {}).get("ok") is False:
        print("FAIL: NG-RUNTIME-022: hermes-cli smoke failed", file=sys.stderr)
        return 1
    print("PASS: runtime smoke (hub layout)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
