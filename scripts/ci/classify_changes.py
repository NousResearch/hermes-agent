#!/usr/bin/env python3
"""Classify a PR's changed files into CI work lanes.

Reads newline-separated changed paths on stdin and writes ``key=value``
booleans (one per lane) to ``$GITHUB_OUTPUT`` and stdout. The
``detect-changes`` composite action consumes them so steps gate on
``if: steps.changes.outputs.<lane> == 'true'``.

Lanes: ``python`` (pytest / ruff / ty / footguns), ``frontend`` (TS typecheck
matrix + desktop build), ``site`` (Docusaurus + generated skill docs). Docker
is not a lane — it builds on push-to-main and release only, never per-PR.

Contract — *fail open, never closed*. We may run a lane we didn't need, but
must never skip one a change could break:

* An empty diff, or any ``.github/`` change, runs everything.
* ``python`` is a denylist: skipped only when *every* file is provably prose
  or a frontend-only package; an unrecognized path keeps it on.
* ``skills/`` (incl. ``SKILL.md``) is python-relevant — the skill-doc tests
  read that tree, so a doc-looking edit can still break Python.
"""

from __future__ import annotations

import os
import sys

_FRONTEND = ("ui-tui/", "web/", "apps/")  # TS typecheck-matrix packages
_ROOT_NPM = {"package.json", "package-lock.json"}  # shifts every package's tree
_SITE = ("website/", "skills/", "optional-skills/")  # docs site + skill pages
# Prose/frontend trees that can't touch Python. skills/ is excluded on purpose.
_PY_SKIP = ("docs/", "website/") + _FRONTEND


def _is_docs(p: str) -> bool:
    if p.startswith(("skills/", "optional-skills/")):
        return False
    return p.endswith((".md", ".mdx")) or p.startswith("docs/") or p.startswith("LICENSE")


def _py_irrelevant(p: str) -> bool:
    return _is_docs(p) or p in _ROOT_NPM or p.startswith(_PY_SKIP)


def classify(files: list[str]) -> dict[str, bool]:
    """Map changed paths to ``{lane: should_run}``."""
    files = [f.strip() for f in files if f.strip()]
    if not files or any(f.startswith(".github/") for f in files):
        return dict.fromkeys(("python", "frontend", "site"), True)
    return {
        "python": any(not _py_irrelevant(f) for f in files),
        "frontend": any(f.startswith(_FRONTEND) or f in _ROOT_NPM for f in files),
        "site": any(f.startswith(_SITE) for f in files),
    }


def main() -> int:
    lanes = classify(sys.stdin.read().splitlines())
    out = "\n".join(f"{k}={str(v).lower()}" for k, v in lanes.items())
    if dest := os.environ.get("GITHUB_OUTPUT"):
        with open(dest, "a", encoding="utf-8") as fh:
            fh.write(out + "\n")
    print(out)  # echo for local runs + CI step logs
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
