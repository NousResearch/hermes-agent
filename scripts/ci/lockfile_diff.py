#!/usr/bin/env python3
"""Semantic diff of npm ``package-lock.json`` files for PR comments.

``git diff`` on a lockfile is unreadable: npm reorders entries, rewrites
integrity hashes, and moves packages between nesting levels, so a one-line
``package.json`` bump can produce a thousand-line textual diff. This script
ignores the text entirely — it parses the ``packages`` map out of both
versions of each lockfile (lockfileVersion 2/3), reduces each to
``{install path: version}``, and set-diffs the two dicts. Reordering and
hash churn vanish; what's left is the actual dependency change.

Usage (from a checkout that still has the base ref available):

    python scripts/ci/lockfile_diff.py --base <ref> --head <ref> \
        --output diff.md [--repo-root .]

Reads every ``package-lock.json`` tracked at either ref (top-level and
nested — the repo has several), diffs each, and writes a Markdown fragment
to ``--output``. Exits 0 always; an empty output file means "no version
changes" (the caller uses that to decide whether to include the section).
The fragment is consumed by ``scripts/ci/assemble_review_comment.py``,
which wraps it in a section with a header and action note.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def parse_lockfile(text: str) -> dict[str, str]:
    """Reduce lockfile JSON to ``{install path: version}``.

    Keys are the ``packages`` map's keys (e.g. ``node_modules/react`` or
    ``node_modules/foo/node_modules/react``), so the same package deduped
    at two versions shows up as two distinct entries. The root entry
    (``""``, the workspace itself) is skipped, as are versionless link
    entries.
    """
    data = json.loads(text)
    out: dict[str, str] = {}
    for path, meta in data.get("packages", {}).items():
        if not path:
            continue  # root project entry, not a dependency
        version = meta.get("version")
        if version:
            out[path] = version
    return out


def diff_locks(base: dict[str, str], head: dict[str, str]) -> dict[str, list]:
    """Set-diff two ``{path: version}`` maps.

    Returns ``added`` / ``removed`` as ``[(path, version)]`` and
    ``updated`` as ``[(path, base_version, head_version)]``, each sorted
    by path.
    """
    added = sorted((p, v) for p, v in head.items() if p not in base)
    removed = sorted((p, v) for p, v in base.items() if p not in head)
    updated = sorted(
        (p, base[p], head[p]) for p in base.keys() & head.keys() if base[p] != head[p]
    )
    return {"added": added, "removed": removed, "updated": updated}


def _display_name(path: str) -> str:
    """``node_modules/foo/node_modules/@scope/bar`` → ``@scope/bar (nested under foo)``."""
    parts = path.split("node_modules/")
    name = parts[-1].rstrip("/")
    if len(parts) > 2:
        parents = " → ".join(p.rstrip("/") for p in parts[1:-1])
        return f"{name} *(nested under {parents})*"
    return name


def render_markdown(diffs: dict[str, dict[str, list]]) -> str:
    """Render per-lockfile diffs as a Markdown fragment.

    ``diffs`` maps lockfile repo-path → the output of :func:`diff_locks`.
    Lockfiles with no version changes are omitted. Returns ``""`` when
    nothing changed anywhere (caller skips the section entirely).

    The output is a fragment — per-lockfile ``####`` subsections with
    tables — not a standalone comment. The ``assemble_review_comment``
    script wraps this in a section with its own header and action note,
    so no top-level header or comment marker is emitted here.
    """
    sections = []
    for lockfile, d in sorted(diffs.items()):
        added, removed, updated = d["added"], d["removed"], d["updated"]
        n = len(added) + len(removed) + len(updated)
        if n == 0:
            continue
        lines = [f"#### `{lockfile}`", ""]
        lines.append("| Package | Before | After |")
        lines.append("| --- | --- | --- |")
        for path, old, new in updated:
            lines.append(f"| {_display_name(path)} | `{old}` | `{new}` |")
        for path, version in added:
            lines.append(f"| ➕ {_display_name(path)} | — | `{version}` |")
        for path, version in removed:
            lines.append(f"| ➖ {_display_name(path)} | `{version}` | — |")
        sections.append("\n".join(lines))

    if not sections:
        return ""

    return "\n\n".join(sections) + "\n"


def _git_show(ref: str, path: str, repo_root: str) -> str | None:
    """Contents of ``path`` at ``ref``, or None if it doesn't exist there."""
    proc = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        cwd=repo_root,
    )
    return proc.stdout if proc.returncode == 0 else None


def _tracked_lockfiles(ref: str, repo_root: str) -> set[str]:
    proc = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref],
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        cwd=repo_root,
        check=True,
    )
    return {
        line
        for line in proc.stdout.splitlines()
        if line.split("/")[-1] == "package-lock.json"
    }


def diff_refs(base: str, head: str, repo_root: str = ".") -> dict[str, dict[str, list]]:
    """Diff every package-lock.json tracked at either ref."""
    lockfiles = _tracked_lockfiles(base, repo_root) | _tracked_lockfiles(head, repo_root)
    diffs = {}
    for path in sorted(lockfiles):
        base_text = _git_show(base, path, repo_root)
        head_text = _git_show(head, path, repo_root)
        base_map = parse_lockfile(base_text) if base_text else {}
        head_map = parse_lockfile(head_text) if head_text else {}
        diffs[path] = diff_locks(base_map, head_map)
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="base git ref (merge base)")
    ap.add_argument("--head", required=True, help="head git ref")
    ap.add_argument("--output", required=True, help="markdown output path")
    ap.add_argument("--repo-root", default=".", help="repository root")
    args = ap.parse_args()

    diffs = diff_refs(args.base, args.head, args.repo_root)
    markdown = render_markdown(diffs)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(markdown)

    if markdown:
        changed = sum(len(v) for d in diffs.values() for v in d.values())
        print(f"{changed} package version change(s) — report written to {args.output}")
    else:
        print("No package version changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
