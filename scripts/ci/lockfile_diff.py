#!/usr/bin/env python3
"""Semantic diff of the repo's JS lockfiles for PR comments.

``git diff`` on a lockfile is unreadable: package managers reorder entries,
rewrite integrity hashes, and (npm) move packages between nesting levels, so a
one-line ``package.json`` bump can produce a thousand-line textual diff. This
script ignores the text entirely — it reduces both versions of each lockfile
to ``{package: version}`` and set-diffs the two dicts. Reordering and hash
churn vanish; what's left is the actual dependency change.

pnpm locks the workspace (``pnpm-lock.yaml``, lockfileVersion 9.0); the
standalone npm projects (website/, the sidecars) keep a ``package-lock.json``
(lockfileVersion 2/3). A pnpm lock has no install paths, so besides its
``{package: versions}`` rows the report lists a project's direct dependency
(``semver *(web)*``) or a package's dependency (``semver *(via node-gyp@10.3.1)*``)
that moved, appeared or went between versions locked on both sides, which the
package rows miss, and one ``project <path>`` row per workspace project that
appears or disappears.

Usage (from a checkout that still has the base ref available):

    python scripts/ci/lockfile_diff.py --base <ref> --head <ref> \
        --output diff.md [--repo-root .]

Reads every ``pnpm-lock.yaml`` and ``package-lock.json`` tracked at either
ref (top-level and nested — the repo has several), diffs each, and writes a
Markdown fragment to ``--output``. Exits 0 unless a lockfile is unreadable
(a pnpm-lock.yaml in a format this reader does not know fails loudly); an
empty output file means "no version changes" (the caller uses that to decide
whether to include the section).
The fragment is consumed by ``scripts/ci/assemble_review_comment.py``,
which wraps it in a section with a header and action note.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

# CI runs this file directly on the runner's Python, outside any package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci.pnpm_lock import section, section_keys


def parse_lockfile(text: str) -> dict[str, str]:
    """Reduce package-lock.json to ``{install path: version}``.

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


_DEP_FIELDS = ("dependencies", "devDependencies", "optionalDependencies")


def _split_key(key: str) -> tuple[str, str]:
    """``@scope/name@1.2.3(peer@4)`` → ``("@scope/name", "1.2.3(peer@4)")``."""
    at = key.index("@", 1)  # past a scope's leading @
    return key[:at], key[at + 1:]


def _bare(version: str) -> str:
    """Drop pnpm's peer suffix: ``0.33.0(react@19.3.0)`` → ``0.33.0``.

    A peer bump rewrites every suffix that names it; the peer's own row reports it.
    """
    return version.split("(", 1)[0]


def _joined(found: dict[str, set[str]]) -> dict[str, str]:
    return {key: ", ".join(sorted(values)) for key, values in found.items()}


def _pnpm_package_versions(text: str) -> dict[str, str]:
    versions: dict[str, set[str]] = {}
    for key in section_keys(text, "packages"):
        name, version = _split_key(key)
        versions.setdefault(name, set()).add(version)
    return _joined(versions)


def _map(value: object, where: str) -> dict:
    """``value`` as a block map; any other shape means a lock this reader does not know."""
    if not isinstance(value, dict):
        raise ValueError(f"pnpm-lock.yaml: {where} is not a block map: {value!r}")
    return value


def _reference(name: str, ref: object, where: str) -> str:
    """A locked dependency reference without its peer suffix.

    An ``npm:`` alias keeps the real package in front (``string-width@4.2.3``):
    pnpm writes a reference as an alias when an ``@`` comes before any ``(`` or ``:``.
    """
    if not isinstance(ref, str):
        raise ValueError(f"pnpm-lock.yaml: {where} {name} has no version: {ref!r}")
    return _bare(ref)


def _target(name: str, reference: str) -> tuple[str, str]:
    """The ``packages`` entry a reference points to: ``(real name, version)``."""
    if "@" in reference.split(":", 1)[0]:
        return _split_key(reference)
    return name, reference


def _pnpm_projects(text: str) -> dict[str, dict[str, str]]:
    """``{project: {dependency name: reference}}`` for each workspace project's direct dependencies."""
    projects: dict[str, dict[str, str]] = {}
    for project, fields in section(text, "importers").items():
        fields = _map(fields, f"importer {project}")
        found = projects[project] = {}
        for field in _DEP_FIELDS:
            for name, entry in _map(fields.get(field, {}), f"{project} {field}").items():
                ref = _reference(name, _map(entry, f"{project} {field} {name}").get("version"), project)
                if not ref.startswith("link:"):  # another workspace project, not a locked version
                    found[name] = ref
    return projects


def _pnpm_snapshot_edges(text: str) -> dict[str, dict[str, str]]:
    """``{"consumer@version": {dependency name: references}}`` from ``snapshots``.

    Peer variants of one consumer collapse into one entry.
    """
    found: dict[str, dict[str, set[str]]] = {}
    for key, entry in section(text, "snapshots").items():
        name, version = _split_key(key)
        entry = _map(entry, f"snapshot {key}")
        edges = found.setdefault(f"{name}@{_bare(version)}", {})
        for field in ("dependencies", "optionalDependencies"):
            for dep, target in _map(entry.get(field, {}), f"{key} {field}").items():
                edges.setdefault(dep, set()).add(_reference(dep, target, key))
    return {consumer: _joined(edges) for consumer, edges in found.items()}


def parse_pnpm_lockfile(text: str) -> dict[str, str]:
    """Reduce pnpm-lock.yaml to ``{package name: versions}``.

    pnpm has no install paths: ``packages`` holds one ``name@version`` entry
    per locked version. A package locked at several versions maps to all of
    them (``"17.0.2, 18.2.0"``), so moving any one of them updates that package.
    """
    return _pnpm_package_versions(text)


def diff_pnpm_locks(base_text: str, head_text: str) -> dict[str, list]:
    """Diff two pnpm locks: package versions, plus the dependency moves those hide.

    The ``packages`` keys cannot show a dependency moving between two versions that
    both stay locked (``semver`` 7.8.5 → 7.7.4 in one project). So the direct
    dependencies of each project present on both sides are diffed too (``semver
    *(web)*``), and so are the ``snapshots`` edges of each package present on both
    sides (``semver *(via node-gyp@10.3.1)*``), including edges that appear or go.
    Such a row is shown only when every package version it names is locked on both
    sides, i.e. when the package rows cannot explain it: a bump otherwise repeats
    once per dependent. An ``npm:`` alias is checked against the package it points
    to and shows it in its versions (``string-width-cjs`` ``string-width@4.2.3``).
    A project that appears or disappears is one ``project <path>`` row, not a row
    per dependency.
    """
    base = parse_pnpm_lockfile(base_text) if base_text else {}
    head = parse_pnpm_lockfile(head_text) if head_text else {}
    diff = diff_locks(base, head)

    def hidden(name: str, *references: str) -> bool:
        for reference in references:
            for one in reference.split(", "):
                real, version = _target(name, one)
                if version not in set(base.get(real, "").split(", ")) & set(head.get(real, "").split(", ")):
                    return False
        return True

    def moves(old: dict[str, str], new: dict[str, str], where: str) -> None:
        for kind, rows in diff_locks(old, new).items():
            diff[kind] += [(f"{row[0]} *({where})*", *row[1:]) for row in rows if hidden(*row)]

    base_projects = _pnpm_projects(base_text) if base_text else {}
    head_projects = _pnpm_projects(head_text) if head_text else {}
    for project in base_projects.keys() & head_projects.keys():
        moves(base_projects[project], head_projects[project], "root" if project == "." else project)
    for kind, projects in (("added", head_projects.keys() - base_projects.keys()),
                           ("removed", base_projects.keys() - head_projects.keys())):
        sides = head_projects if kind == "added" else base_projects
        for project in projects:
            count = len(sides[project])
            diff[kind].append((f"project {project}", f"{count} locked dependenc{'y' if count == 1 else 'ies'}"))

    base_edges = _pnpm_snapshot_edges(base_text) if base_text else {}
    head_edges = _pnpm_snapshot_edges(head_text) if head_text else {}
    for consumer in base_edges.keys() & head_edges.keys():
        moves(base_edges[consumer], head_edges[consumer], f"via {consumer}")

    return {kind: sorted(rows) for kind, rows in diff.items()}


def _diff_npm_locks(base_text: str, head_text: str) -> dict[str, list]:
    return diff_locks(parse_lockfile(base_text) if base_text else {},
                      parse_lockfile(head_text) if head_text else {})


_DIFFERS = {"package-lock.json": _diff_npm_locks, "pnpm-lock.yaml": diff_pnpm_locks}


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
        check=False,
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
        if line.split("/")[-1] in _DIFFERS
    }


def diff_refs(base: str, head: str, repo_root: str = ".") -> dict[str, dict[str, list]]:
    """Diff every JS lockfile tracked at either ref."""
    lockfiles = _tracked_lockfiles(base, repo_root) | _tracked_lockfiles(head, repo_root)
    diffs = {}
    for path in sorted(lockfiles):
        differ = _DIFFERS[path.split("/")[-1]]
        diffs[path] = differ(_git_show(base, path, repo_root) or "", _git_show(head, path, repo_root) or "")
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
        rows = sum(len(v) for d in diffs.values() for v in d.values())
        print(f"{rows} lockfile change row(s) — report written to {args.output}")
    else:
        print("No lockfile version changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
