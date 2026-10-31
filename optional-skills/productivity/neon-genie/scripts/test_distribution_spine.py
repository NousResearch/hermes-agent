#!/usr/bin/env python3
"""Negative and positive packaging tests for distribution spine (stdlib only)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
PY = sys.executable
SPINE = SCRIPT_DIR / "distribution_spine.py"


def run_spine(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PY, str(SPINE), *args, "--root", str(root)],
        cwd=root,
        capture_output=True,
        text=True,
    )


def copy_skill_tree() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="ng-dist-test-"))
    # Minimal tree needed for spine
    for name in (
        "distribution.yaml",
        "SKILL.md",
        "VERSION",
        "manifest.json",
        "PRIVACY.md",  # mirrored → references/PRIVACY.md (privacy spine)
        "schemas",
        "profiles",
        "evals",
        "references",
        "examples",
        "templates",
        "scripts",
        "skills.sh.json",
    ):
        src = ROOT / name
        dst = tmp / name
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "out"),
            )
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    # strip skills package so write recreates it
    if (tmp / "skills").exists():
        shutil.rmtree(tmp / "skills")
    return tmp


def test_write_and_verify() -> None:
    tmp = copy_skill_tree()
    try:
        r = run_spine(tmp, "write")
        assert r.returncode == 0, r.stderr + r.stdout
        r2 = run_spine(tmp, "verify")
        assert r2.returncode == 0, r2.stderr
        print("PASS: write+verify on clean tree")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mirror_drift_fails() -> None:
    tmp = copy_skill_tree()
    try:
        assert run_spine(tmp, "write").returncode == 0
        victim = tmp / "references" / "schemas" / "opportunity-packet.schema.json"
        victim.write_text(victim.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
        r = run_spine(tmp, "verify")
        assert r.returncode != 0
        assert "NG-PKG-004" in (r.stderr + r.stdout)
        print("PASS: mirror drift fails with NG-PKG-004")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_support_file_fails() -> None:
    tmp = copy_skill_tree()
    try:
        assert run_spine(tmp, "write").returncode == 0
        target = tmp / "scripts" / "paths.py"
        target.unlink()
        r = run_spine(tmp, "verify")
        assert r.returncode != 0
        combined = r.stderr + r.stdout
        assert "NG-PKG-010" in combined or "NG-PKG-008" in combined or "NG-PKG-003" in combined or "NG-PKG" in combined
        print("PASS: missing support file fails")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dir_style_hub_ref_fails() -> None:
    tmp = copy_skill_tree()
    try:
        assert run_spine(tmp, "write").returncode == 0
        skill = tmp / "SKILL.md"
        text = skill.read_text(encoding="utf-8")
        # inject a bare directory hub path that Hermes would reject
        text = text.replace(
            "## Mission",
            "See also `references/profiles` for contracts.\n\n## Mission",
            1,
        )
        skill.write_text(text, encoding="utf-8")
        r = run_spine(tmp, "verify")
        assert r.returncode != 0
        assert "NG-PKG-011" in (r.stderr + r.stdout)
        print("PASS: directory-style hub ref fails NG-PKG-011")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_package_parity_fails() -> None:
    tmp = copy_skill_tree()
    try:
        assert run_spine(tmp, "write").returncode == 0
        pkg_skill = tmp / "skills" / "neon-genie" / "VERSION"
        pkg_skill.write_text("0.0.0-broken\n", encoding="utf-8")
        r = run_spine(tmp, "verify")
        assert r.returncode != 0
        assert "NG-PKG-015" in (r.stderr + r.stdout)
        print("PASS: package parity fails NG-PKG-015")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    failures = 0
    for fn in (
        test_write_and_verify,
        test_mirror_drift_fails,
        test_missing_support_file_fails,
        test_dir_style_hub_ref_fails,
        test_package_parity_fails,
    ):
        try:
            fn()
        except AssertionError as exc:
            print(f"FAIL: {fn.__name__}: {exc}", file=sys.stderr)
            failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: {fn.__name__}: {exc}", file=sys.stderr)
            failures += 1
    if failures:
        print(f"FAIL: {failures} distribution spine test(s)", file=sys.stderr)
        return 1
    print("PASS: distribution spine tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
