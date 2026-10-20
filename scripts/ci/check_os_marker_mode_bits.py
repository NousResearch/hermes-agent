#!/usr/bin/env python3
"""Fail when an added test asserts POSIX file-mode bits without gating Windows away.

Windows does not enforce POSIX mode bits: ``os.chmod(path, 0o600)`` succeeds and
``os.umask`` exists, but the mode Windows *reports* is synthesized from the file's
read-only attribute, so a stat-based mode assertion never holds there:

    os.chmod(f, 0o600); oct(f.stat().st_mode & 0o777)  ->  '0o666'
    os.chmod(d, 0o700); oct(d.stat().st_mode & 0o777)  ->  '0o777'

A test that reads ``st_mode``/``stat.S_IMODE`` and compares it to an octal literal
therefore fails on every Windows checkout while passing on Linux and macOS. The OS
lanes cannot see that: ``.github/workflows/tests-os.yml`` imports only the files
carrying a matching ``platforms(...)`` marker, so an unmarked test that no Windows
host can pass is green in CI and red for every Windows developer locally.

The fix is the ordinary one — the same shape ``tests/AGENTS.md`` § "Don't fake the
host OS" prescribes: mark the test ``platforms("posix")`` (or ``platforms("linux")``/
``platforms("macos")``, whatever host the assertion is actually about), or make the
Windows arm assert the ACL/read-only-attribute form it really enforces
(``hermes_dir_acl``). Marking never *removes* coverage: both lanes still import the
file and run the test on the POSIX host it belongs to.

Judged per test FUNCTION, on the lines a change touches (nothing else):

* flagged: reads ``st_mode``/``S_IMODE`` and compares against any ``0o...`` literal,
  with no ``platforms(...)`` marker at all, or one whose specs admit Windows
  (``windows``, ``any``, ``not linux``, ...);
* not flagged: a marker that excludes Windows (``posix``, ``linux``, ``macos``,
  ``not windows``); a function that only mentions mode bits in a comment or
  docstring; a function with no octal literal (a plain ``st_mode`` read, e.g. a
  ``stat_result`` stub built as data).

Opt out of one function with ``# os-marker: ok — <why>`` on any of its lines or
decorators. Legitimate reasons: the assertion is about a fake/parametrized
``stat_result`` the test constructed itself, or the mode is part of a value the
test passes to production code without asserting on the host's real bits.
"It passes on my machine" is not one — it does not pass on Windows.

    python scripts/ci/check_os_marker_mode_bits.py                # working tree vs merge-base with origin/main
    python scripts/ci/check_os_marker_mode_bits.py --base HEAD^1  # CI: the merge commit vs the base branch
    python scripts/ci/check_os_marker_mode_bits.py --all          # every test file, ignored lines included

Exit codes: 0 = clean, 1 = at least one offender, 2 = git or usage error.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

OPT_OUT = "os-marker: ok"
TESTS_ROOT = "tests"
_OCTAL = re.compile(r"^0o([0-7]+)$")
_MARKER_CALL = re.compile(r"\bplatforms\(([^)]*)\)")
_QUOTED = re.compile(r"""["']([^"']*)["']""")

# Mirrors scripts/ci/list_os_marked_tests.py (and tests/_fixtures/platform_gating.py): which lanes a
# spec admits. Membership of "windows" is the whole question here.
_SPEC_HOSTS = {
    "linux": frozenset({"linux"}),
    "macos": frozenset({"macos"}),
    "windows": frozenset({"windows"}),
    "posix": frozenset({"linux", "macos"}),
    "any": frozenset({"linux", "macos", "windows"}),
}


class GitError(RuntimeError):
    pass


@dataclass(frozen=True)
class Hit:
    """One test function that asserts mode bits and can still run on Windows."""

    path: str
    func: str
    lineno: int
    reason: str  # "no marker" | "marker admits windows"


def _git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, encoding="utf-8",
                          errors="surrogateescape", stdin=subprocess.DEVNULL, timeout=300, check=False)
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout


def _spec_admits_windows(spec: str) -> bool:
    leaf = spec.removeprefix("not ").strip()
    hosts = _SPEC_HOSTS.get(leaf, frozenset())
    if spec.startswith("not "):
        return "windows" in _SPEC_HOSTS["any"] - hosts
    return "windows" in hosts


def _marker_specs(node: ast.AST, source: str) -> list[str]:
    """Every quoted spec inside ``platforms(...)`` calls in *node*'s decorators."""
    specs: list[str] = []
    for dec in getattr(node, "decorator_list", []):
        segment = ast.get_source_segment(source, dec) or ""
        for call in _MARKER_CALL.finditer(segment):
            specs += [s.strip().lower() for s in _QUOTED.findall(call.group(1))]
    return specs


def _module_marker_specs(tree: ast.Module, source: str) -> list[str]:
    """Specs of a module-level ``pytestmark = pytest.mark.platforms(...)``, if any."""
    specs: list[str] = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets):
            continue
        segment = ast.get_source_segment(source, stmt.value) or ""
        for call in _MARKER_CALL.finditer(segment):
            specs += [s.strip().lower() for s in _QUOTED.findall(call.group(1))]
    return specs


def _reads_mode_bits(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr in {"st_mode", "S_IMODE"}:
            return True
    return False


def _octal_literals(node: ast.AST, source: str) -> list[tuple[int, str]]:
    """``(lineno, literal)`` for every ``0o...`` constant in *node* (source spelling, not value)."""
    out: list[tuple[int, str]] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, int) and not isinstance(child.value, bool):
            spelling = (ast.get_source_segment(source, child) or "").strip()
            if _OCTAL.match(spelling):
                out.append((child.lineno, spelling))
    return out


def _test_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Module- and class-level functions whose name starts with ``test_``."""
    funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            funcs.append(node)
        elif isinstance(node, ast.ClassDef):
            funcs += [n for n in node.body
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")]
    return funcs


def _opted_out(source: str, func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when ``# os-marker: ok`` sits on the function, its decorators or its first line."""
    start = min([func.lineno, *(d.lineno for d in func.decorator_list)])
    body_start = min(getattr(b, "lineno", func.lineno) for b in func.body) if func.body else func.lineno
    lines = source.splitlines()
    # Decorator lines + the def line + the first body line: enough to place an opt-out comment,
    # and it cannot hide a hit on a later line by accident.
    for n in range(start, min(body_start, len(lines)) + 1):
        if OPT_OUT in lines[n - 1]:
            return True
    return False


def scan_file(path: str, source: str) -> list[Hit]:
    """Every test function in *source* that asserts mode bits and is not gated away from Windows."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    module_specs = _module_marker_specs(tree, source)
    hits: list[Hit] = []
    for func in _test_functions(tree):
        if not _reads_mode_bits(func):
            continue
        if not _octal_literals(func, source):
            continue
        specs = [*module_specs, *_marker_specs(func, source)]
        if specs and not any(_spec_admits_windows(s) for s in specs):
            continue
        if _opted_out(source, func):
            continue
        reason = "marker admits windows" if specs else "no marker"
        hits.append(Hit(path, func.name, func.lineno, reason))
    return hits


def added_lines(diff: str) -> dict[str, set[int]]:
    """Map ``path -> added line numbers`` from a ``--unified=0`` diff (post-image numbering)."""
    out: dict[str, set[int]] = {}
    path: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            path = None if target == "/dev/null" else target.removeprefix("b/")
        elif path and line.startswith("@@"):
            m = re.match(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)
            if m:
                start, count = int(m.group(1)), int(m.group(2) or 1)
                out.setdefault(path, set()).update(range(start, start + count))
    return out


def _untracked(repo: Path) -> list[str]:
    out = _git(repo, "ls-files", "--others", "--exclude-standard", "-z", "--", TESTS_ROOT)
    return [p for p in out.split("\0") if p.endswith(".py") and Path(p).name.startswith("test_")]


def changed_line_map(repo: Path, base: str | None, head: str | None) -> dict[str, set[int]] | None:
    """``path -> added lines`` for the judged change; None means "every file, whole"."""
    if base is None and head is None:
        return None
    args = ["diff", "--unified=0", "--no-color", "--no-ext-diff", "-M", base or "HEAD"]
    if head:
        args.append(head)
    args += ["--", "tests/**/test_*.py", "tests/*.py"]
    lines = added_lines(_git(repo, *args))
    if head is None:  # the working tree: untracked test files are new in full
        for rel in _untracked(repo):
            lines.setdefault(rel, set())
    return lines


def default_base(repo: Path, head: str | None) -> str:
    """Merge-base of the judged commit (HEAD for a working tree) with origin/main."""
    tip = "HEAD"
    if head and _git(repo, "rev-parse", "--verify", "--quiet", f"{head}^{{commit}}", check=False).strip():
        tip = head
    for ref in ("origin/main", "main"):
        out = _git(repo, "merge-base", tip, ref, check=False).strip()
        if out:
            return out
    raise GitError("no merge-base with origin/main or main; pass --base")


def _candidate_files(repo: Path, changed: dict[str, set[int]] | None) -> list[str]:
    """Repo-relative test paths to judge: every file (``--all``) or the changed ones."""
    if changed is None:
        return sorted(p.relative_to(repo).as_posix() for p in (repo / TESTS_ROOT).rglob("test_*.py"))
    return sorted(p for p in changed if Path(p).name.startswith("test_"))


def read_source(repo: Path, rel: str, head: str | None) -> str | None:
    """The judged side's bytes: the working tree (head None) or the ``head`` revision's blob.

    Content and diff must come from the SAME side, or a run that judges ``--head <rev>`` from a
    checkout of another revision mixes pre-fix bytes with post-fix line numbers.
    """
    if head is None:
        try:
            return (repo / rel).read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            return None
    if not _git(repo, "rev-parse", "--verify", "--quiet", f"{head}:{rel}", check=False).strip():
        return None  # not in that revision (deleted, or added later)
    return _git(repo, "cat-file", "blob", f"{head}:{rel}")


def find_hits(repo: Path, changed: dict[str, set[int]] | None, head: str | None = None) -> list[Hit]:
    """Offenders among the files *changed* selects (or every test file when it is None)."""
    hits: list[Hit] = []
    for rel in _candidate_files(repo, changed):
        source = read_source(repo, rel, head)
        if source is None:
            continue
        touched = None if changed is None else changed.get(rel, set())
        for hit in scan_file(rel, source):
            func = next(f for f in _test_functions(ast.parse(source)) if f.name == hit.func and f.lineno == hit.lineno)
            end = getattr(func, "end_lineno", func.lineno)
            if touched is None or (touched & set(range(func.lineno, end + 1))):
                hits.append(hit)
    return sorted(hits, key=lambda h: (h.path, h.lineno))


def render(hits: list[Hit]) -> list[str]:
    lines = [
        f"::error file={h.path},line={h.lineno}::{h.func} asserts POSIX mode bits with {h.reason}; "
        f"mark it platforms(\"posix\") (or the host it is about)"
        for h in hits
    ]
    lines += [
        "",
        f"{len(hits)} added test function(s) assert POSIX file-mode bits on a host where Windows also "
        "runs them. Windows does not enforce those bits (os.chmod(0o600) still reports 0o666), so the "
        "test fails on every Windows checkout while CI stays green - the OS lanes import only files "
        "carrying a matching platforms() marker (tests/AGENTS.md § Don't fake the host OS).",
        "",
        "Fix one of these ways:",
        "  - the assertion is about POSIX bits: @pytest.mark.platforms(\"posix\") (or \"linux\"/\"macos\")",
        "  - the assertion should also cover Windows: assert the form Windows enforces (ACL / read-only",
        "    attribute, see hermes_dir_acl) in a separate platforms(\"windows\") test",
        "  - the mode is data, not this host's bits (a fake stat_result, a value handed to production",
        f"    code): mark the line `# {OPT_OUT} — <why>`",
        "Reproduce locally: python scripts/ci/check_os_marker_mode_bits.py",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--base", help="base revision (default: merge-base with origin/main)")
    parser.add_argument("--head", help="head revision or tree (default: the working tree)")
    parser.add_argument("--all", action="store_true", help=f"judge every test file under {TESTS_ROOT}/")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    if args.all and (args.base or args.head):
        print("check_os_marker_mode_bits: --all takes no --base/--head", file=sys.stderr)
        return 2
    try:
        if args.all:
            changed = None
        else:
            base = args.base or default_base(repo, args.head)
            changed = changed_line_map(repo, base, args.head)
        hits = find_hits(repo, changed, args.head)
    except GitError as exc:
        print(f"check_os_marker_mode_bits: {exc}", file=sys.stderr)
        return 2
    if not hits:
        scope = "every test file" if args.all else "the lines this change adds"
        print(f"check_os_marker_mode_bits: no unmarked mode-bit assertion in {scope}")
        return 0
    print("\n".join(render(hits)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
