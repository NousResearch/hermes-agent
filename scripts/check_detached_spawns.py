#!/usr/bin/env python3
"""Fail when production code starts a detached child outside the owned-child helper.

A child started in its own session/process group (``start_new_session=True``, ``os.setsid``,
``os.setpgid``/``os.setpgrp`` in a ``preexec_fn``) no longer dies with the Hermes process that
started it. When that process is SIGKILLed, OOM-killed or crashes, the child reparents to init
(or a ``systemd --user`` subreaper) and runs forever. Every long-lived child that should die with
its owner goes through ``hermes_cli.process_identity.spawn_owned`` (or hands its group to
``own_process_groups`` where something else spawned it), which registers the group with the
per-process owner-death supervisor.

Flags, in Python files of the production trees (tests, evals, scripts and nix checks excluded):

  start_new_session=True / start_new_session=<anything but False>, or a ``"start_new_session"``
  dict key / ``kwargs["start_new_session"] = ...`` item with a non-False value
  process_group=<anything but None> (``process_group=0`` puts the child in a new group), in the
  same three shapes
  any reference to os.setsid / os.setpgid / os.setpgrp (call or ``preexec_fn=os.setsid``)
  a call to ``windows_detach_popen_kwargs()`` (``start_new_session=True`` on POSIX)

Opt out of one line with ``detached-spawn: ok — <why>`` on that line or the line directly above
it. Legitimate reasons are the ones that stay valid after the owner dies: the child must OUTLIVE
its owner on purpose (an updater relaunching the gateway, a restart watcher, a detached service
launch), or the child is short-lived and awaited with a hard timeout that kills its group (the
group exists only so that kill reaches grandchildren).

``_BASELINE`` maps a file to a tolerated count of unmarked hits. It is empty: every detached
spawn site is on ``spawn_owned`` or marked with its reason. Keep it that way; an entry is a
temporary burn-down, and an overstated one is an advisory (``--strict-baseline`` fails on it).

Run: python scripts/check_detached_spawns.py [--all] [--print-baseline] [--strict-baseline] [paths...]
Exit 1 on any violation, 0 when clean.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = "detached-spawn: ok"
# The helper itself: the one place a detached spawn is the point.
HELPER_FILES = {"hermes_cli/process_identity.py"}
SCAN_TREES = ("agent", "cron", "gateway", "hermes_cli", "plugins", "tools", "tui_gateway", "acp_adapter", "pm")
SCAN_ROOT_FILES = ("cli.py", "run_agent.py", "hermes_state.py", "utils.py", "hermes_constants.py")
SKIP_PARTS = {"tests", "__pycache__", "evals", "node_modules"}
_DETACH_OS_ATTRS = {"setsid", "setpgid", "setpgrp"}
# Helpers that return detaching Popen kwargs (``start_new_session=True`` on POSIX).
_DETACH_HELPERS = {"windows_detach_popen_kwargs"}
# Cheap text prefilter: only files mentioning one of these are parsed.
_NEEDLES = ("start_new_session", "process_group", "setsid", "setpgid", "setpgrp", *_DETACH_HELPERS)


def _is_false(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in (False, None, 0)


def _detaches(name: object, value: ast.AST) -> bool:
    """Whether Popen keyword ``name`` set to ``value`` detaches the child into its own group."""
    if name == "start_new_session":
        return not _is_false(value)
    # process_group=0 is a new group; only the default (None) leaves the child in ours.
    return name == "process_group" and not (isinstance(value, ast.Constant) and value.value is None)


def _is_os_detach(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr in _DETACH_OS_ATTRS
            and isinstance(node.value, ast.Name) and node.value.id == "os")


def _is_detach_helper_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and (getattr(node.func, "id", None) in _DETACH_HELPERS
                                           or getattr(node.func, "attr", None) in _DETACH_HELPERS)


def detach_lines(text: str) -> list[int]:
    """Line numbers of detached-spawn shapes in ``text`` (a Python module)."""
    if not any(needle in text for needle in _NEEDLES):
        return []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and _detaches(node.arg, node.value):
            lines.add(node.value.lineno)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and _detaches(key.value, value):
                    lines.add(key.lineno)
        elif isinstance(node, ast.Assign):  # kwargs["start_new_session"] = True
            for target in node.targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                        and _detaches(target.slice.value, node.value)):
                    lines.add(target.lineno)
        elif _is_os_detach(node) or _is_detach_helper_call(node):
            lines.add(node.lineno)
    return sorted(lines)


def _marked(src_lines: list[str], lineno: int) -> bool:
    here = src_lines[lineno - 1] if lineno - 1 < len(src_lines) else ""
    above = src_lines[lineno - 2] if lineno >= 2 else ""
    return MARKER in here or MARKER in above


def _tracked_files(root: Path) -> list[Path]:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", *SCAN_TREES, *SCAN_ROOT_FILES],
                             capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [root / rel for rel in out.split("\0") if rel.endswith(".py")]


def _in_scope(rel: str) -> bool:
    parts = Path(rel).parts
    return not (SKIP_PARTS & set(parts)) and not Path(rel).name.startswith("test_") and rel not in HELPER_FILES


def scan(paths=None, root: Path | None = None) -> dict[str, list[tuple[int, str]]]:
    """Repo-relative path -> [(lineno, line)] of unmarked detached spawns."""
    root = (root or ROOT).resolve()
    hits: dict[str, list[tuple[int, str]]] = {}
    for path in (list(paths) if paths else _tracked_files(root)):
        path = Path(path).resolve()
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = str(path)
        if path.suffix != ".py" or not _in_scope(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
        except OSError:
            continue
        src = text.splitlines()
        found = [(n, src[n - 1]) for n in detach_lines(text) if not _marked(src, n)]
        if found:
            hits[rel] = found
    return hits


# Files that spawned unmarked detached children when this check landed. Burn-down list:
# migrate to spawn_owned / own_process_groups, or mark the line with a reason, then drop the entry.
_BASELINE: dict[str, int] = {
}


def _format_baseline(hits: dict[str, list]) -> str:
    body = "".join(f'    "{rel}": {len(found)},\n' for rel, found in sorted(hits.items()))
    return "_BASELINE: dict[str, int] = {\n" + body + "}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("paths", nargs="*", help="files to check (default: the production trees)")
    ap.add_argument("--all", action="store_true", help="ignore _BASELINE; report every hit")
    ap.add_argument("--print-baseline", action="store_true", help="print a _BASELINE for the current tree")
    ap.add_argument("--strict-baseline", action="store_true", help="also fail when _BASELINE overstates a file")
    args = ap.parse_args(argv)

    hits = scan(args.paths or None)
    if args.print_baseline:
        print(_format_baseline(hits))
        return 0
    baseline = {} if (args.all or args.paths) else _BASELINE
    problems: list[str] = []
    advisories: list[str] = []
    for rel in sorted(hits):
        allowed = baseline.get(rel)
        if allowed is not None and len(hits[rel]) <= allowed:
            if len(hits[rel]) < allowed:
                advisories.append(f"{rel}: {len(hits[rel])} detached spawn(s) left, _BASELINE says {allowed}")
            continue
        problems.extend(f"{rel}:{n}: {line.strip()[:160]}" for n, line in hits[rel])
    advisories.extend(f"{rel}: listed in _BASELINE but clean (or gone)" for rel in sorted(baseline) if rel not in hits)
    if advisories:
        print("advisory — _BASELINE in scripts/check_detached_spawns.py is stale; regenerate it with "
              "--print-baseline (fewer hits than listed is progress, not a failure):")
        print("\n".join("  " + a for a in advisories))
    if args.strict_baseline:
        problems.extend(advisories)
    if not problems:
        print("no detached spawns outside the owned-child helper and the baseline")
        return 0
    print("\n".join(problems))
    print(f"\n{len(problems)} detached spawn(s) flagged. A child in its own session/group outlives a "
          "SIGKILLed Hermes. Start long-lived children with hermes_cli.process_identity.spawn_owned "
          "(or hand the group to own_process_groups), or mark a child that must outlive its owner / "
          f"is awaited with a group-killing timeout with `{MARKER} — <why>` (same line or the line above).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
