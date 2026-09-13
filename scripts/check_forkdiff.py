#!/usr/bin/env python3
"""Gate: ``fork.yaml`` must describe this fork as it is *now*.

The fork-diff page (rendered with protolambda/forkdiff and published on GitHub
Pages) is only useful while it is true. Two things make it go stale silently:

1. **A rebase onto newer upstream.** ``base.hash`` still points at the old
   upstream commit, so the page shows upstream's own changes as if the fork
   made them. This script requires ``base.hash`` to equal
   ``git merge-base HEAD <upstream>``; a rebase moves the merge-base, and the
   gate stays red until the hash is bumped.
2. **A new fork change nobody described.** Every path in
   ``git diff --name-only base.hash HEAD`` must match a glob in some section
   (or a global ``ignore``), and every glob must still match something — a
   section describing code the fork no longer carries is as misleading as a
   missing one.

Both are pure git + YAML, so the same check runs locally::

    python3 scripts/check_forkdiff.py                       # coverage only
    python3 scripts/check_forkdiff.py --upstream-ref upstream/main

Exit status is non-zero on any violation. ``--review-status-out`` writes the
JSON the CI comment synthesizer consumes (same shape as history-check).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml

GLOB_CLASS_RE = re.compile(r"\[([^\]]*)\]")


def glob_to_regex(glob: str) -> re.Pattern[str]:
    """Translate a forkdiff glob into a regex over the repo-relative path.

    Semantics follow the doublestar rules forkdiff uses: ``*`` and ``?`` never
    cross a ``/``; ``**`` matches any number of directories (``a/**/b`` also
    matches ``a/b``); ``[...]`` character classes pass through, with a leading
    ``!`` meaning negation.
    """
    out: list[str] = []
    i = 0
    while i < len(glob):
        c = glob[i]
        if c == "*":
            if glob.startswith("**", i):
                if glob.startswith("**/", i):
                    out.append("(?:.*/)?")
                    i += 3
                    continue
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            end = glob.find("]", i + 1)
            if end == -1:
                out.append(re.escape(c))
            else:
                cls = glob[i + 1 : end]
                if cls.startswith("!"):
                    cls = "^" + cls[1:]
                out.append("[" + cls + "]")
                i = end + 1
                continue
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("^" + "".join(out) + "$")


def collect_globs(node: dict, path: str = "def") -> list[tuple[str, str]]:
    """Every glob in the section tree as ``(section path, glob)``.

    A section's ``ignore`` list counts as coverage too: forkdiff still lists
    those files under the section (grayed out), so they are described.
    """
    found: list[tuple[str, str]] = []
    title = node.get("title") or "(untitled)"
    here = f"{path} › {title}" if path != "def" else title
    for g in node.get("globs") or []:
        found.append((here, str(g)))
    for g in node.get("ignore") or []:
        found.append((here, str(g)))
    for child in node.get("sub") or []:
        found.extend(collect_globs(child, here))
    return found


def git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def changed_paths(repo: Path, base: str, head: str) -> list[str]:
    out = git("diff", "--name-only", f"{base}..{head}", cwd=repo)
    return [line for line in out.splitlines() if line]


def check_coverage(
    paths: Iterable[str], globs: list[tuple[str, str]]
) -> tuple[list[str], list[tuple[str, str]], dict[str, int]]:
    """Return ``(uncovered paths, stale globs, matches per glob)``."""
    compiled = [(section, g, glob_to_regex(g)) for section, g in globs]
    hits: dict[str, int] = {g: 0 for _, g in globs}
    uncovered: list[str] = []
    for p in paths:
        matched = False
        for _, g, rx in compiled:
            if rx.match(p):
                hits[g] += 1
                matched = True
        if not matched:
            uncovered.append(p)
    stale = [(section, g) for section, g in globs if hits[g] == 0]
    return uncovered, stale, hits


def check_base(repo: Path, base: str, head: str, upstream_ref: str | None) -> list[str]:
    problems: list[str] = []
    if not re.fullmatch(r"[0-9a-f]{40}", base):
        problems.append(
            f"base.hash must be a full 40-hex commit id, got {base!r} — a short or "
            "symbolic ref would silently move under the page."
        )
        return problems
    try:
        git("cat-file", "-e", f"{base}^{{commit}}", cwd=repo)
    except RuntimeError:
        problems.append(
            f"base.hash {base[:12]} is not present in this repository. The fork must "
            "sit on top of it (fetch upstream if the clone is shallow)."
        )
        return problems
    if upstream_ref is None:
        return problems
    try:
        git("merge-base", "--is-ancestor", base, upstream_ref, cwd=repo)
    except RuntimeError:
        problems.append(
            f"base.hash {base[:12]} is not an ancestor of {upstream_ref} — it must name "
            "a commit on upstream main, not a fork commit."
        )
    merge_base = git("merge-base", head, upstream_ref, cwd=repo)
    if merge_base != base:
        problems.append(
            f"base.hash {base[:12]} != merge-base({head}, {upstream_ref}) = "
            f"{merge_base[:12]}. The fork was rebased onto newer upstream; update "
            "fork.yaml's base.hash and re-describe the sections (see docs/forkdiff.md)."
        )
    return problems


def review_status(problems: list[str], detail: str) -> list[dict]:
    if not problems:
        return []
    return [
        {
            "source": "fork diff analysis",
            "results": [
                {
                    "kind": "action_required",
                    "title": "fork.yaml no longer describes the fork",
                    "summary": problems[0]
                    if len(problems) == 1
                    else f"{len(problems)} issues: {problems[0]}",
                    "detail": detail,
                    "how_to_fix": (
                        "See docs/forkdiff.md. After a rebase: set base.hash to "
                        "`git merge-base HEAD upstream/main`. For new files: add them to "
                        "the section that explains them (or to a global `ignore` if they "
                        "are not code). Then run `python3 scripts/check_forkdiff.py "
                        "--upstream-ref upstream/main` locally."
                    ),
                }
            ],
        }
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=".", help="path to the fork checkout")
    ap.add_argument("--fork", default="fork.yaml", help="fork page definition")
    ap.add_argument("--head", default="HEAD", help="the fork revision to describe")
    ap.add_argument(
        "--upstream-ref",
        default=None,
        help="a ref holding upstream main (e.g. refs/remotes/upstream/main); "
        "enables the merge-base check",
    )
    ap.add_argument("--review-status-out", default=None, help="write review-status JSON here")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    fork_path = repo / args.fork
    spec = yaml.safe_load(fork_path.read_text(encoding="utf-8")) or {}
    base = str((spec.get("base") or {}).get("hash") or "").strip()
    problems: list[str] = []
    lines: list[str] = []

    problems += check_base(repo, base, args.head, args.upstream_ref)

    globs = collect_globs(spec.get("def") or {})
    globs += [("(global ignore)", str(g)) for g in spec.get("ignore") or []]

    paths: list[str] = []
    if re.fullmatch(r"[0-9a-f]{40}", base):
        try:
            paths = changed_paths(repo, base, args.head)
        except RuntimeError as exc:
            problems.append(str(exc))

    uncovered, stale, hits = check_coverage(paths, globs)
    if uncovered:
        problems.append(
            f"{len(uncovered)} changed file(s) are not described by any fork.yaml section"
        )
        lines.append("Uncovered files (add each to the section that explains it):")
        lines += [f"  - {p}" for p in uncovered]
    if stale:
        problems.append(f"{len(stale)} glob(s) match nothing the fork changes")
        lines.append("Stale globs (the fork no longer changes anything they name):")
        lines += [f"  - {g}    [{section}]" for section, g in stale]

    covered = len(paths) - len(uncovered)
    print(f"fork.yaml: base {base[:12]}  head {args.head}  changed files {len(paths)}  "
          f"described {covered}  sections+ignores {len(globs)}")
    if args.upstream_ref is None:
        print("  (no --upstream-ref: merge-base check skipped)")
    for line in lines:
        print(line)
    for p in problems:
        print(f"::error::{p}")

    if args.review_status_out:
        Path(args.review_status_out).write_text(
            json.dumps(review_status(problems, "\n".join(lines))), encoding="utf-8"
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
