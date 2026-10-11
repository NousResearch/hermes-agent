"""``python -m scripts.code_health`` / ``scripts/check``: the code-health ratchet."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
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
_MODES = ("blocking", "advisory", "off")
_TRUSTED_ALIGNMENT_PARENTS = frozenset((
    "0fc49025948aa41afa1e9b7683806042fcad7de6",
    "c57331677d7fc46298a7159cf3a650853f0966c5",
))


def parse_switch(text: str) -> str:
    """The module-level ``ENFORCEMENT`` value (plain or annotated assignment; the last one wins,
    as at runtime). Missing, unparseable or unknown is "blocking", so no edit relaxes it by
    accident."""
    try:
        body = ast.parse(text).body
    except SyntaxError:
        return "blocking"
    found = None
    for node in body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == "ENFORCEMENT" for t in targets):
            found = node.value.value if isinstance(node.value, ast.Constant) else None
    return found if found in _MODES else "blocking"


def enforcement(repo: Path, rev: str) -> str:
    """The master switch as committed on ``rev``; a revision without one is blocking."""
    return parse_switch(gitio.read_file(repo, rev, _SWITCH_FILE) or "")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="code_health", description=__doc__)
    p.add_argument("--base", help="base revision (default: merge-base of HEAD with origin/main)")
    p.add_argument("--inherited-base", action="append", default=[],
                   help="additional explicitly trusted comparison parent (repeatable)")
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


def run(repo: Path, base: str, head: str | None, as_json: bool = False,
        switch_rev: str | None = None, inherited_bases: tuple[str, ...] = ()) -> int:
    """Judge ``head`` against ``base``; the ENFORCEMENT switch is read from ``switch_rev``
    (default: the base), never from the head under test."""
    started = time.monotonic()
    switch_rev = switch_rev or base
    mode = enforcement(repo, switch_rev)
    if mode == "off":
        _skipped(f"code health: off (ENFORCEMENT in {_SWITCH_FILE} on {switch_rev[:12]})", as_json)
        return 0
    # A finding survives only when every trusted parent also reports it. Files
    # identical to any trusted parent cannot meet that test, so a full upstream
    # sync must not measure those thousands of inherited files.
    scoped = None
    if inherited_bases:
        scope = gitio.changed_names(repo, base, head)
        for parent in inherited_bases:
            scope &= gitio.changed_names(repo, parent, head)
        if not scope:
            _skipped("code health: no merge-only files outside the trusted parents", as_json)
            return 0
        scoped = sorted(scope)
    changes = gitio.changed_files(repo, base, head, scoped)
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
    if inherited_bases:
        # Each trusted parent independently evaluates the exact head. The finding
        # remains only when every parent also reports that same occurrence;
        # compare() preserves one-to-one credit within each parent, so two
        # copies cannot consume one occurrence.
        identities: set[tuple[str, str, str, int]] | None = None
        for parent in inherited_bases:
            parent_changes = gitio.changed_files(repo, parent, head, scoped)
            parent_head_paths = sorted({c.new for c in parent_changes if c.new and in_scope(c.new)})
            parent_base_paths = sorted({c.old for c in parent_changes if c.old and in_scope(c.old)})
            parent_measurer = Measurer(repo, ruff, known_env=gitio.known_env_names(repo, parent))
            parent_m = parent_measurer.measure(parent, parent_base_paths)
            parent_head_m = parent_measurer.measure(head, parent_head_paths)
            parent_findings = {(f.path, f.rule, f.scope, f.line)
                               for f in compare(parent_m, parent_head_m, parent_changes)}
            identities = parent_findings if identities is None else identities & parent_findings
        if identities is not None:
            findings = [f for f in findings if (f.path, f.rule, f.scope, f.line) in identities]
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
    try:
        head = gitio.resolve_tree(repo, args.head) if args.head else None
        if args.report:
            return _report(repo, head)
        if args.base:  # CI: the base is the target branch tip, which also holds the switch
            base = gitio.resolve_rev(repo, args.base)
            trusted = tuple(gitio.resolve_rev(repo, p) for p in args.inherited_base)
            for parent in trusted:
                if parent not in _TRUSTED_ALIGNMENT_PARENTS:
                    raise RuntimeError(f"unapproved inherited parent {parent}")
                proc = subprocess.run(["git", "merge-base", "--is-ancestor", parent,
                                       args.head or "HEAD"], cwd=repo, capture_output=True,
                                      stdin=subprocess.DEVNULL, timeout=60, check=False)
                if proc.returncode != 0:
                    raise RuntimeError(f"trusted parent {parent} is not an ancestor of the head")
            return run(repo, base, head, as_json=args.json, inherited_bases=trusted)
        # Locally the base is the merge-base, which lags main; the switch comes from the main tip
        # so a flip on main reaches a branch without a rebase. The head never supplies it.
        tip = gitio.commit_or_none(repo, args.head) if args.head else None
        base = gitio.default_base(repo, tip or "HEAD")
        switch_rev = gitio.commit_or_none(repo, "origin/main") or base
        trusted = tuple(gitio.resolve_rev(repo, p) for p in args.inherited_base)
        for parent in trusted:
            if parent not in _TRUSTED_ALIGNMENT_PARENTS:
                raise RuntimeError(f"unapproved inherited parent {parent}")
            proc = subprocess.run(["git", "merge-base", "--is-ancestor", parent,
                                   args.head or "HEAD"], cwd=repo, capture_output=True,
                                  stdin=subprocess.DEVNULL, timeout=60, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"trusted parent {parent} is not an ancestor of the head")
        return run(repo, base, head, as_json=args.json, switch_rev=switch_rev,
                   inherited_bases=trusted)
    except RuntimeError as exc:
        print(f"code health: {exc}", file=sys.stderr)
        return 2
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        # A measuring tool (npm install of the pinned TypeScript, ...) failed: that is a broken
        # run, not a finding, so it must not exit 1 behind a traceback.
        print(f"code health: measurement tooling failed: {exc}", file=sys.stderr)
        return 2
