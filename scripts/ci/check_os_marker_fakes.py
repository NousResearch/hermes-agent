#!/usr/bin/env python3
"""Fail when a test file fakes macOS without carrying ``@pytest.mark.macos_only``.

The OS lanes are marker-driven: ``.github/workflows/tests-os.yml`` selects the
files the macOS job imports via ``scripts/ci/list_os_marked_tests.py macos_only``
and then runs ``-m macos_only``. A file whose tests only pass because they make
the interpreter believe it is on macOS (``is_macos`` patched to ``True``,
``sys.platform`` set to ``"darwin"``) but that carries no marker is invisible to
that lane: it is green on Linux over a faked branch and never imported on the
host it exists for (#111866). Root ``AGENTS.md`` § "Don't fake the host OS" is
the rule; this check makes a violation a red job instead of a review catch.

Flags, per ``tests/**/test_*.py`` without a whole-word ``macos_only``:

  monkeypatch.setattr(mod, "is_macos", lambda: True) / patch(..., return_value=True)
  monkeypatch.setattr(sys, "platform", "darwin")  /  patch("sys.platform", "darwin")
  platform.system patched to return "Darwin"

Opt out of one line with ``# os-marker: ok — <why>`` on that line (a pure
function taking the platform as data is host-independent and stays unmarked).
``_BASELINE`` lists the files that already faked macOS when this check landed;
they are a burn-down list, not a policy — split the macOS arm out, mark it,
and drop the entry (a stale entry fails the check).

Run: python scripts/ci/check_os_marker_fakes.py [tests_root]
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

MARKER = "macos_only"
OPT_OUT = "os-marker: ok"

_TRUE = r"(?:lambda[^:]*:\s*True|return_value\s*=\s*True|,\s*True\b)"
_FAKES = (
    re.compile(rf"\bis_macos\b.*{_TRUE}"),
    re.compile(r"\bis_macos\.return_value\s*=\s*True\b"),
    # setattr(sys, "platform", "darwin") / patch("sys.platform", "darwin") / x.platform = "darwin";
    # a host-honest READ (`if sys.platform == "darwin":`) is not a fake and does not match.
    re.compile(r"""\bplatform["']?\s*,\s*["']darwin["']"""),
    re.compile(r"""\.platform\s*=\s*["']darwin["']"""),
    re.compile(r"""\bplatform\.system\b.*(?:lambda[^:]*:|return_value\s*=)\s*["']Darwin["']"""),
)

# Files that faked macOS before this check existed (#111866). Burn down, never extend.
_BASELINE = frozenset(
    {
    }
)


def _code_lines(text: str) -> list[tuple[int, str, str]]:
    """Yield ``(lineno, code, raw)`` with the ``#`` comment stripped from *code*.

    A ``#`` inside a string literal is rare in these patterns and only ever
    hides a hit (never invents one), so a plain split is the honest trade
    against a full tokenizer.
    """
    out = []
    for i, raw in enumerate(text.splitlines(), 1):
        out.append((i, raw.split("#", 1)[0], raw))
    return out


def find_unmarked_fakes(root: Path, repo_root: Path) -> dict[str, list[tuple[int, str]]]:
    """Map repo-relative test path -> ``[(lineno, line)]`` of un-opted-out macOS fakes."""
    marker_pat = re.compile(rf"\b{MARKER}\b")
    hits: dict[str, list[tuple[int, str]]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if not (fname.startswith("test_") and fname.endswith(".py")):
                continue
            path = Path(dirpath) / fname
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if marker_pat.search(text):
                continue
            lines = [
                (n, raw.strip())
                for n, code, raw in _code_lines(text)
                if OPT_OUT not in raw and any(p.search(code) for p in _FAKES)
            ]
            if lines:
                resolved = path.resolve()
                base = repo_root if resolved.is_relative_to(repo_root) else root.resolve()
                hits[resolved.relative_to(base).as_posix()] = lines
    return hits


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    root = Path(argv[1]) if len(argv) > 1 else repo_root / "tests"
    if not root.is_dir():
        print(f"error: no such directory: {root}", file=sys.stderr)
        return 2
    hits = find_unmarked_fakes(root, repo_root)
    new = {rel: lines for rel, lines in hits.items() if rel not in _BASELINE}
    stale = sorted(_BASELINE - set(hits))
    for rel, lines in sorted(new.items()):
        for n, line in lines:
            print(f"{rel}:{n}: fakes macOS without @pytest.mark.{MARKER}: {line}")
    if new:
        print(
            f"\n{len(new)} test file(s) make the interpreter believe it is on macOS but carry no "
            f"`{MARKER}` marker, so the macOS lane never imports them (AGENTS.md § Don't fake the "
            f"host OS). Split the macOS arm into its own `@pytest.mark.{MARKER}` test that runs the "
            f"real branch, or mark `# {OPT_OUT} — <why>` on a host-independent line.",
            file=sys.stderr,
        )
    for rel in stale:
        print(f"{rel}: listed in _BASELINE but no longer fakes macOS — remove the entry")
    return 1 if new or stale else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
