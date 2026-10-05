"""``python -m scripts.code_health`` / ``scripts/check``: the code-health ratchet."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

from scripts.code_health import gitio
from scripts.code_health.compare import compare
from scripts.code_health.config import in_scope
from scripts.code_health.measure import Measurer
from scripts.code_health.report import (
    apply_allows,
    format_findings,
    summarize_tree,
    verdict,
)
from scripts.code_health.ruff_runner import pinned_version, resolve_ruff

_FOOTER = """
Each function and file has its own cap: its value on main if it is already over target,
otherwise the target. New code meets the target; existing code may only go down. Fix the code
(the `fix:` line is the repo's remedy); a genuine exception takes a reviewed comment on the line,
`# health: allow <RULE> -- <why>`. Rules and targets: scripts/code_health/config.py.
Reproduce locally: python scripts/check --only health (every lint check: python scripts/check;
on every push: python scripts/check --install-hook pre-push)."""

_SWITCH_FILE = "scripts/code_health/config.py"
_SWITCH = re.compile(r"^ENFORCEMENT\s*=\s*[\"'](blocking|advisory|off)[\"']", re.MULTILINE)


def enforcement(repo: Path, base: str) -> str:
    """The master switch as committed on the base revision; a base without one is blocking."""
    found = _SWITCH.search(gitio.read_file(repo, base, _SWITCH_FILE) or "")
    return found.group(1) if found else "blocking"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="code_health", description=__doc__)
    p.add_argument("--base", help="base revision (default: merge-base of HEAD with origin/main)")
    p.add_argument("--head", help="head commit or tree (default: the working tree, untracked included)")
    p.add_argument("--report", action="store_true", help="burn-down summary of the whole head tree")
    p.add_argument("--json", action="store_true", help="machine-readable findings")
    p.add_argument("--print-pins", action="store_true",
                   help="print the pinned ruff requirement (CI installs exactly this)")
    return p


def _report(repo: Path, head: str | None) -> int:
    paths = [p for p in gitio.tracked_files(repo, head) if in_scope(p)]
    measurer = Measurer(repo, resolve_ruff(repo), known_env=gitio.known_env_names(repo, head))
    print(summarize_tree(measurer.measure(head, paths)))
    return 0


def _skipped(why: str, as_json: bool) -> None:
    """A run that measured nothing: an empty findings list for --json, the reason for humans."""
    if as_json:
        print("[]")
    print(why, file=sys.stderr if as_json else sys.stdout)


def run(repo: Path, base: str, head: str | None, as_json: bool = False) -> int:
    started = time.monotonic()
    mode = enforcement(repo, base)
    if mode == "off":
        _skipped(f"code health: off (ENFORCEMENT in {_SWITCH_FILE} on {base[:12]})", as_json)
        return 0
    changes = gitio.changed_files(repo, base, head)
    head_paths = sorted({c.new for c in changes if c.new and in_scope(c.new)})
    base_paths = sorted({c.old for c in changes if c.old and in_scope(c.old)})
    if not head_paths:
        _skipped("code health: no measured files changed", as_json)
        return 0
    # Base files are measured too: a deleted .py still needs ruff on the base side.
    needs_ruff = any(p.endswith(".py") for p in (*head_paths, *base_paths))
    ruff = resolve_ruff(repo) if needs_ruff else []
    measurer = Measurer(repo, ruff, known_env=gitio.known_env_names(repo, base))
    base_m = measurer.measure(base, base_paths)
    head_m = measurer.measure(head, head_paths)
    findings = compare(base_m, head_m, changes)
    apply_allows(findings, head_m)
    blocking, advisory = verdict(findings)
    failed = bool(blocking) and mode == "blocking"
    if as_json:
        print(json.dumps([asdict(f) for f in findings], indent=2))
        return 1 if failed else 0
    body = format_findings(findings)
    if body:
        print(body)
    elapsed = time.monotonic() - started
    print(f"\ncode health: {len(head_paths)} files vs {base[:12]}: {blocking} blocking, "
          f"{advisory} advisory ({elapsed:.1f}s)")
    if blocking:
        print(_FOOTER)
    if blocking and not failed:
        print(f"\nadvisory mode (ENFORCEMENT in {_SWITCH_FILE}): not failing on the above.")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = gitio.repo_root(Path.cwd())
    if args.print_pins:
        print(f"ruff=={pinned_version(repo)}")
        return 0
    head = gitio.resolve_tree(repo, args.head) if args.head else None
    try:
        if args.report:
            return _report(repo, head)
        base = gitio.resolve_rev(repo, args.base) if args.base else gitio.default_base(repo)
        return run(repo, base, head, as_json=args.json)
    except RuntimeError as exc:
        print(f"code health: {exc}", file=sys.stderr)
        return 2
