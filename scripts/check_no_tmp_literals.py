#!/usr/bin/env python3
"""Fail when production code, skills, docs or prompts hard-code a literal ``/tmp`` path.

``/tmp`` is not portable: Termux has no ``/tmp`` at all, native Windows has no such directory,
macOS aliases it to ``/private/tmp`` (breaking naive path comparisons), and on most Linux
distributions it is a RAM-backed tmpfs that fills under Hermes load. Hermes resolves scratch
space through one helper (``hermes_constants.get_scratch_dir()`` → ``HERMES_HOME/cache/scratch``,
which every Hermes process also exports as ``TMPDIR``/``TMP``/``TEMP``), and prompts + skills
must steer the model the same way, because a literal ``/tmp`` in a SKILL.md or system prompt
becomes a literal ``/tmp`` in the model's shell commands on every platform.

Flags any line containing a ``/tmp`` path token (``/tmp``, ``/tmp/...``) in the scanned trees.
Automatically NOT flagged (no marker needed):

  ${TMPDIR:-/tmp}            shell fallback idiom: TMPDIR wins where it is set
  /var/tmp, /private/tmp     different directories, not the bare ``/tmp`` root
  tmpfs, tmp_path, ~/tmp     not a ``/tmp`` path at all
  code comments, docstrings  they describe code; nothing there reaches a shell or the model
                             (Markdown prose is NOT exempt: docs and skills are read by both)

Opt out of one line with ``no-tmp: ok — <why>`` on that line or on the line directly above it
(``# no-tmp: ok — ...`` in Python/shell, ``<!-- no-tmp: ok — ... -->`` in Markdown). Legitimate
reasons: the code *detects* ``/tmp`` (path-alias checks, security denylists, the scratch-dir
resolver's own POSIX fallback) or the text explains why ``/tmp`` is wrong. "It works on my
machine" is not one.

``_BASELINE`` maps files that already carried literals when this check landed to their hit
count. It is a burn-down list, not a policy: fix the file (or mark the lines) and drop the
entry. A baseline file gaining hits fails the check; an entry that overstates a file (partly or
fully burned down) is reported as an advisory so parallel clean-ups never turn CI red — refresh
it with ``--print-baseline`` (``--strict-baseline`` turns those advisories into failures).

Scope: every first-party ``.py .sh .ts .tsx .js .mjs .cjs .md .mdx .txt .yaml .yml .json .toml``
file except tests (``tests/``, ``tests-js/``, ``__tests__/``, ``e2e/``, ``test_*.py``,
``*.test.ts`` ...), ``evals/``, CI workflows (``.github/`` runs on Linux runners), container
build files (``Dockerfile*``, ``docker/``), generated lockfiles and the translated docs mirror
(``website/i18n/``, regenerated from the English source).

Run: python scripts/check_no_tmp_literals.py [--all] [--print-baseline] [paths...]
  --all              ignore ``_BASELINE`` and report every hit (burn-down view)
  --print-baseline   print a ``_BASELINE`` literal matching the current tree
  --strict-baseline  fail on stale/overstated ``_BASELINE`` entries too
Exit 1 on any violation, 0 when clean.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()

MARKER = "no-tmp: ok"

SCAN_SUFFIXES = {
    ".py", ".sh", ".bash", ".ts", ".tsx", ".js", ".mjs", ".cjs",
    ".md", ".mdx", ".txt", ".yaml", ".yml", ".json", ".toml",
}

# Pruned at every depth.
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", ".worktrees",
    "tests", "tests-js", "__tests__", "e2e", "evals", "docker", "MagicMock",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", "coverage", "target",
}
# Pruned only directly under the repo root.
ROOT_SKIP_DIRS = {".github"}
# Pruned as repo-relative paths.
SKIP_REL_DIRS = {Path("website/i18n"), Path("website/build"), Path("website/node_modules")}

SKIP_FILE_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "uv.lock", "poetry.lock"}
SKIP_FILE_PATTERNS = (
    re.compile(r"^test_.*\.py$"),
    re.compile(r"_test\.py$"),
    re.compile(r"^conftest\.py$"),
    re.compile(r"\.(test|spec)\.(ts|tsx|js|mjs|cjs)$"),
    re.compile(r"^Dockerfile(\..*)?$"),
)

# A `/tmp` path token: not glued to a preceding path/word char (`/var/tmp`, `~/tmp`, `a/tmp`),
# not the `${TMPDIR:-/tmp}` fallback idiom, and not followed by a word char (`/tmpfs`).
_TMP_TOKEN = re.compile(r"(?<![\w./~\\-])(?<!:-)/tmp(?![\w-])")


_LINE_COMMENT_PREFIX = {
    ".sh": ("#",), ".bash": ("#",), ".yaml": ("#",), ".yml": ("#",), ".toml": ("#",),
    ".ts": ("//", "/*", "*"), ".tsx": ("//", "/*", "*"), ".js": ("//", "/*", "*"),
    ".mjs": ("//", "/*", "*"), ".cjs": ("//", "/*", "*"),
}
_TRAILING_COMMENT = {".py": "#", ".sh": "#", ".bash": "#", ".yaml": "#", ".yml": "#", ".toml": "#",
                     ".ts": "//", ".tsx": "//", ".js": "//", ".mjs": "//", ".cjs": "//"}


def _python_comment_and_docstring_spans(text: str) -> tuple[dict[int, int], set[int]]:
    """(line -> column where a `#` comment starts, lines inside doc-strings) via tokenize/ast.

    Comments and docstrings *describe* code; a `/tmp` there cannot reach a shell or the model, so
    they stay out of the count (prompt strings, defaults and command templates are what matters).
    """
    import ast
    import io
    import tokenize

    comments: dict[int, int] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                comments[tok.start[0]] = tok.start[1]
    except (tokenize.TokenError, SyntaxError, IndentationError):
        pass
    doc_lines: set[int] = set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # scanned files' own SyntaxWarnings are not our business
            tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return comments, doc_lines
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_lines.update(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
    return comments, doc_lines


def _iter_lines_with_hits(text: str, suffix: str = ""):
    prev_marked = False
    comments: dict[int, int] = {}
    doc_lines: set[int] = set()
    if suffix == ".py":
        comments, doc_lines = _python_comment_and_docstring_spans(text)
    prefixes = _LINE_COMMENT_PREFIX.get(suffix, ())
    trailing = _TRAILING_COMMENT.get(suffix)
    for lineno, line in enumerate(text.splitlines(), start=1):
        marked = MARKER in line
        try:
            match = _TMP_TOKEN.search(line)
            if not match or marked or prev_marked or lineno in doc_lines:
                continue
            stripped = line.lstrip()
            if prefixes and stripped.startswith(prefixes):
                continue  # whole-line comment
            if suffix == ".py":
                if lineno in comments and match.start() >= comments[lineno]:
                    continue  # inside a trailing `#` comment (tokenize-exact: not a `#` in a string)
            elif trailing:
                cut = line.find(trailing)
                if 0 <= cut < match.start() and not re.search(r"""["'`]""", line[:cut]):
                    continue  # trailing comment on a line with no string literal before it
            yield lineno, line
        finally:
            prev_marked = marked


def _skip_file(path: Path) -> bool:
    if path == SELF or path.suffix not in SCAN_SUFFIXES or path.name in SKIP_FILE_NAMES:
        return True
    return any(p.search(path.name) for p in SKIP_FILE_PATTERNS)


def _git_ignored(root: Path) -> set[Path]:
    """Ignored/untracked-by-.gitignore paths (runner artifacts such as ``test_durations.json``)
    are build products, not sources; a scan that reads them fails on whatever the last test
    run wrote. Empty when *root* is not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
            capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {root / rel for rel in out.split("\0") if rel}


def iter_files(root: Path | None = None):
    root = (root or ROOT).resolve()
    ignored = _git_ignored(root)
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        rel_here = here.relative_to(root) if here != root else Path()
        excluded = SKIP_DIRS | (ROOT_SKIP_DIRS if here == root else set())
        dirnames[:] = sorted(
            d for d in dirnames if d not in excluded and (rel_here / d) not in SKIP_REL_DIRS
        )
        for filename in sorted(filenames):
            path = here / filename
            if path not in ignored and not _skip_file(path):
                yield path


def scan(paths=None, root: Path | None = None) -> dict[str, list[tuple[int, str]]]:
    """Repo-relative POSIX path -> [(lineno, line)] for every file with at least one hit."""
    root = (root or ROOT).resolve()
    hits: dict[str, list[tuple[int, str]]] = {}
    files = list(paths) if paths else list(iter_files(root))
    for path in files:
        path = Path(path).resolve()
        if paths and _skip_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
        except OSError:
            continue
        found = list(_iter_lines_with_hits(text, path.suffix))
        if found:
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                rel = str(path)
            hits[rel] = found
    return hits


# Files that carried literal /tmp paths when this check landed, with their hit counts.
# Burn-down list: fix or mark, then delete the entry. Regenerate with --print-baseline.
_BASELINE: dict[str, int] = {
    # a tree listing inside a fenced code block; an inline marker would render on the page
    "website/docs/getting-started/nix-setup.md": 1,
}


def _format_baseline(hits: dict[str, list]) -> str:
    body = "".join(f'    "{rel}": {len(found)},\n' for rel, found in sorted(hits.items()))
    return "_BASELINE: dict[str, int] = {\n" + body + "}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("paths", nargs="*", help="files to check (default: whole repo)")
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
    total = 0
    for rel in sorted(hits):
        found = hits[rel]
        allowed = baseline.get(rel)
        if allowed is not None and len(found) <= allowed:
            if len(found) < allowed:
                advisories.append(f"{rel}: {len(found)} literal /tmp path(s) left, _BASELINE says {allowed}")
            continue
        for lineno, line in found:
            total += 1
            problems.append(f"{rel}:{lineno}: {line.strip()[:160]}")
    for rel in sorted(baseline):
        if rel not in hits:
            advisories.append(f"{rel}: listed in _BASELINE but clean (or gone)")
    if advisories:
        print("advisory — _BASELINE in scripts/check_no_tmp_literals.py is stale; regenerate it with "
              "--print-baseline (fewer hits than listed is progress, not a failure):")
        print("\n".join("  " + a for a in advisories))
    if args.strict_baseline and advisories:
        problems.extend(advisories)

    if not problems:
        print("no literal /tmp paths outside the baseline")
        return 0
    print("\n".join(problems))
    print(
        f"\n{total} literal /tmp path(s) flagged. Resolve scratch space through "
        f"hermes_constants.get_scratch_dir() (or $TMPDIR / tempfile, which Hermes points there), tell the "
        f"model to do the same in skills and prompts, or mark a deliberate line with `{MARKER} — <why>` (same line or the line above). "
        f"See scripts/check_no_tmp_literals.py."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
