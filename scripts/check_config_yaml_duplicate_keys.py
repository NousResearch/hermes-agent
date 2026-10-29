#!/usr/bin/env python3
"""Fail when a ``config.yaml`` contains a duplicate key or drops a top-level key.

One duplicate YAML key in ``profiles/ultron/config.yaml`` made that file
unparseable on 2026-10-28: every Ultron worker died at startup (134 crashed kanban
runs), the multiplexer skipped the profile (Friday went silent), and the crash loop
produced 85 identical error DMs to Bryan. The write path now refuses to persist an
unloadable config, but an OUT-OF-BAND edit (a regex/sed edit, an editor, a merge) never
passes through that seam — this check is the backstop for it.

What it catches:

* **duplicate key** — ruamel raises ``DuplicateKeyError``. PyYAML silently keeps the
  LAST value, so a PyYAML-based check cannot see this at all.
* **dropped top-level key** — the staged file lost a top-level key that HEAD still has.
  The incident's precursor was a regex edit that DELETED ``platform_toolsets:`` and then
  appended a second copy of the block; the key set was unchanged, but a plain
  "delete a section" edit is caught here.

Usage:

    python3 scripts/check_config_yaml_duplicate_keys.py <file>...   # check files
    python3 scripts/check_config_yaml_duplicate_keys.py --stdin      # check stdin
    python3 scripts/check_config_yaml_duplicate_keys.py --repo <root>  # check a repo

``--repo <root>`` walks the root for ``config.yaml`` and ``profiles/*/config.yaml``,
and (when the root is a git work tree) also reports top-level keys dropped relative to
HEAD for the tracked config files.

Exit codes: 0 = clean, 1 = a problem was found (with a diagnostic on stderr).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


def _load_yaml(text: str):
    """Parse *text* with a ruamel-backed reader that REJECTS duplicate keys.

    Returns ``(parsed, backend)``. ``backend`` is ``"ruamel"`` (duplicate keys are an
    error), ``"pyyaml"`` (duplicate keys are silently tolerated — the caller says so), or
    ``None`` when NO YAML parser is available, which the caller MUST treat as "could not
    check" rather than "failed": a missing parser must never block a commit.

    The PyYAML fallback exists only so the check can still report a file's top-level key
    set on a machine without ruamel. It CANNOT detect a duplicate key — PyYAML keeps the
    last occurrence silently — so every caller must treat a ``"pyyaml"`` verdict as
    "duplicate keys were not checked here", never as "this file is clean".
    """
    try:
        import hermes_yaml as yaml  # the shared ruamel-backed reader
        return yaml.safe_load(text), "ruamel"
    except ImportError:
        pass
    try:
        from ruamel.yaml import YAML
        from ruamel.yaml.error import YAMLError
    except ImportError:
        try:
            import yaml as pyyaml
        except ImportError:
            return None, None  # no parser at all: the caller must not fail the build
        return pyyaml.safe_load(text), "pyyaml"
    loader = YAML(typ="safe", pure=True)
    loader.version = (1, 1)
    try:
        return loader.load(text), "ruamel"
    except YAMLError:
        loader = YAML(typ="safe", pure=True)
        loader.version = (1, 1)
        return loader.load(text), "ruamel"


def _has_duplicate_key_scanner() -> bool:
    """True when this process can actually DETECT a duplicate key.

    The check's whole reason to exist is the duplicate key: PyYAML keeps the last
    occurrence silently, so a PyYAML-only environment cannot answer the question at all.
    Reporting "clean" from such an environment would be a false negative on exactly the
    defect the check exists to catch, so the caller uses this to refuse to certify.
    """
    try:
        import hermes_yaml  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import ruamel.yaml  # noqa: F401
        return True
    except ImportError:
        return False


def _describe(exc: Exception) -> str:
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    line = getattr(mark, "line", None)
    where = f" at line {line + 1}" if isinstance(line, int) else ""
    problem = getattr(exc, "problem", None)
    detail = " ".join(str(problem or exc).split())
    return f"{detail}{where}"


def _top_level_keys(text: str) -> Optional[set[str]]:
    try:
        parsed, backend = _load_yaml(text)
    except Exception:
        return None
    if backend is None:
        return None  # could not check
    if parsed is None:
        return set()
    if not isinstance(parsed, dict):
        return None
    return {str(k) for k in parsed.keys()}


def check_file(path: Path) -> list[str]:
    """Problems in one config file: a duplicate key (unparseable) or a non-mapping root.

    Returns an empty list when the file cannot be CHECKED (no YAML parser available) —
    a missing parser must not block a commit. PyYAML-only environments cannot see
    duplicate keys; that limitation is reported on stdout, not as a failure.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return []
    except OSError as exc:
        return [f"{path}: could not be read: {exc}"]
    if not text.strip():
        return []
    try:
        parsed, backend = _load_yaml(text)
    except Exception as exc:
        detail = _describe(exc)
        hint = ""
        if "duplicate" in detail.lower():
            hint = (
                "  A duplicate key makes the whole file unreadable by ruamel, so every "
                "worker for this profile dies at startup.\n"
                "  Fix: remove the repeated block (keep one copy of each key)."
            )
        return [f"{path}: duplicate key or invalid YAML: {detail}{hint}"]
    if backend is None:
        print(f"⚠ {path}: no YAML parser available — integrity check skipped for this file")
        return []
    if backend == "pyyaml":
        print(f"⚠ {path}: only PyYAML is available — duplicate keys cannot be detected here")
    if parsed is not None and not isinstance(parsed, dict):
        return [f"{path}: top level must be a mapping, got {type(parsed).__name__}"]
    return []


def _git_show(repo: Path, ref: str, path: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "show", f"{ref}:{path}"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def check_repo(repo: Path) -> list[str]:
    """Every config.yaml in *repo*, plus the top-level keys dropped relative to HEAD."""
    problems: list[str] = []
    configs = [repo / "config.yaml"]
    profiles = repo / "profiles"
    if profiles.is_dir():
        configs.extend(sorted(p for p in profiles.glob("*/config.yaml") if p.is_file()))

    for cfg in configs:
        problems.extend(check_file(cfg))

    # Dropped top-level keys: a staged config that lost a section relative to HEAD.
    try:
        tracked = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "--", "config.yaml", "profiles/*/config.yaml"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return problems
    if tracked.returncode != 0:
        return problems
    for rel in (line.strip() for line in tracked.stdout.splitlines()):
        if not rel:
            continue
        head_text = _git_show(repo, "HEAD", rel)
        if head_text is None:
            continue
        head_keys = _top_level_keys(head_text)
        if head_keys is None:
            continue  # HEAD itself does not parse; the duplicate check already spoke.
        try:
            work_text = (repo / rel).read_text(encoding="utf-8-sig")
        except OSError:
            continue
        work_keys = _top_level_keys(work_text)
        if work_keys is None:
            continue
        dropped = head_keys - work_keys
        if dropped:
            shown = ", ".join(sorted(dropped)[:12])
            problems.append(
                f"{rel}: drops {len(dropped)} top-level key(s) that HEAD still has: {shown}\n"
                f"  A silently dropped section breaks every reader that expects it.\n"
                f"  Fix: restore the key, or stage the deletion deliberately "
                f"(`git rm` the file / edit it by hand) if it is intentional."
            )
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="config.yaml paths to check")
    parser.add_argument("--repo", help="repository root to scan for config.yaml files")
    parser.add_argument("--stdin", action="store_true", help="check YAML read from stdin")
    args = parser.parse_args(argv)

    problems: list[str] = []
    if args.stdin:
        text = sys.stdin.read()
        try:
            parsed, backend = _load_yaml(text)
        except Exception as exc:
            problems.append(f"<stdin>: duplicate key or invalid YAML: {_describe(exc)}")
        else:
            if backend is None:
                print("⚠ <stdin>: no YAML parser available — integrity check skipped")
            elif backend == "pyyaml":
                print("⚠ <stdin>: only PyYAML is available — duplicate keys cannot be detected")
            elif parsed is not None and not isinstance(parsed, dict):
                problems.append(
                    f"<stdin>: top level must be a mapping, got {type(parsed).__name__}")
    if args.repo:
        problems.extend(check_repo(Path(args.repo)))
    for f in args.files:
        problems.extend(check_file(Path(f)))

    if problems:
        print("✖ config.yaml integrity check FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    # A "clean" verdict is only meaningful when a duplicate-key scanner was actually
    # available. Without ruamel the check could not see a duplicate key, so it must NOT
    # certify the file as clean — that would be a false negative on the one defect this
    # check exists to catch.
    if not _has_duplicate_key_scanner():
        print(
            "✖ config.yaml integrity check could NOT run: no duplicate-key scanner "
            "(ruamel) is available.\n"
            "  PyYAML keeps the last occurrence of a duplicate key silently, so it cannot "
            "answer the question this check exists to ask.\n"
            "  Install the Hermes venv (hermes-agent/venv) and re-run, or check manually "
            "with a ruamel-backed reader.",
            file=sys.stderr,
        )
        return 1

    checked = len(args.files) + (1 if args.stdin else 0) + (1 if args.repo else 0)
    print(f"✓ config.yaml integrity: clean ({checked} target(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
