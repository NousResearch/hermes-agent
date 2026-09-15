#!/usr/bin/env python3
"""Fail when a protected upstream identifier has been renamed, or a boundary crossed.

The NOVA platform layer lives alongside an upstream runtime that we merge from. Three
classes of mistake would be expensive and are each hard to spot in review:

1. **A renamed protected identifier.** Environment variables, on-disk paths, wire
   headers, entry-point groups and module names that existing installations, third-party
   plugins or the upstream project depend on.
2. **A renamed model identifier.** ``Hermes-4-405B`` and friends are *large language
   models*, not this runtime. Renaming one breaks model resolution and surfaces as a 404
   from an inference provider rather than as a failing test. This is the single most
   likely expensive mistake, so it is checked separately and absolutely.
3. **An inverted dependency.** Upstream code importing ``nova`` would make every future
   upstream merge a conflict, and would break a runtime that has no NOVA installed.

Run: ``python scripts/check_protected_identifiers.py``
Exit 1 with a file:line list on any violation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Paths owned by the platform layer. Everything else is upstream-owned.
PLATFORM_PATHS = (
    "nova/", "deploy/", "docs/platform/", "tests/platform/",
    # Browser tests drive the Control Centre and check the runtime state it produced,
    # so they are NOVA's own — the same side of the boundary as tests/platform/.
    "tests/browser/",
)

#: Directories never scanned.
SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build",
    "apps", "website", "web", "ui-tui", ".worktrees",
}

#: Identifiers that must keep their upstream spelling. A rename breaks live
#: installations, third-party plugins, or the wire protocol.
PROTECTED = (
    "HERMES_HOME",
    "hermes_cli",
    "hermes_constants",
    "hermes_state",
    "hermes_agent.plugins",
    "X-Hermes-Session-Id",
)

#: LLM model identifiers. A different product that shares a name with the runtime.
MODEL_ID_PATTERN = re.compile(r"[Hh]ermes-[0-9]")

#: Files that legitimately discuss the protections themselves.
DOC_EXEMPT = {
    "docs/platform/ARCHITECTURE_BOUNDARIES.md",
    "docs/platform/BRAND_SURFACE_AUDIT.md",
    "docs/platform/IMPLEMENTATION_PLAN.md",
    "docs/platform/HERMES_PLATFORM_AUDIT.md",
    "docs/platform/CORE_PATCHES.md",
    "nova/AGENTS.md",
    "nova/README.md",
    "scripts/check_protected_identifiers.py",
}


def iter_files(suffixes: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        out.append(path)
    return out


def is_platform(rel: Path) -> bool:
    text = rel.as_posix()
    return any(text.startswith(prefix) for prefix in PLATFORM_PATHS)


def check_import_direction() -> list[str]:
    """Upstream code must never import the platform layer."""
    problems: list[str] = []
    pattern = re.compile(r"^\s*(?:from\s+nova(?:\.\w+)*\s+import|import\s+nova\b)", re.MULTILINE)
    for path in iter_files((".py",)):
        rel = path.relative_to(ROOT)
        if is_platform(rel):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            line = text[: match.start()].count("\n") + 1
            problems.append(
                f"{rel}:{line}: upstream-owned file imports 'nova' — the dependency arrow "
                "points one way (see docs/platform/ARCHITECTURE_BOUNDARIES.md)"
            )
    return problems


def check_stdlib_shadowing() -> list[str]:
    """A top-level 'platform' package would shadow the stdlib module.

    40 upstream modules do ``import platform`` for OS detection. A package of that name
    at the repository root silently replaces it and breaks ``platform.system()``.
    """
    if (ROOT / "platform" / "__init__.py").is_file():
        return [
            "platform/__init__.py: a top-level 'platform' package shadows the stdlib "
            "'platform' module that upstream code imports for OS detection. "
            "The platform layer is named 'nova/' for this reason."
        ]
    return []


def check_model_identifiers() -> list[str]:
    """Model identifiers must never be rewritten by a branding sweep."""
    problems: list[str] = []
    for path in iter_files((".py", ".yaml", ".yml", ".json", ".toml")):
        rel = path.relative_to(ROOT)
        if not is_platform(rel):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if MODEL_ID_PATTERN.search(line) and "NOVA-" in line:
                problems.append(
                    f"{rel}:{number}: a model identifier appears next to a NOVA rename — "
                    "model names such as Hermes-4-405B are a third-party product and must "
                    "never be renamed"
                )
    return problems


def check_protected_present() -> list[str]:
    """Each protected identifier must still exist somewhere in the tree.

    A blanket rename would remove every occurrence at once; this catches that, which no
    per-file rule would.
    """
    problems: list[str] = []
    haystack: list[str] = []
    for path in iter_files((".py", ".toml", ".yaml", ".yml")):
        rel = path.relative_to(ROOT)
        if rel.as_posix() in DOC_EXEMPT:
            continue
        haystack.append(path.read_text(encoding="utf-8", errors="replace"))
    joined = "\n".join(haystack)
    for identifier in PROTECTED:
        if identifier not in joined:
            problems.append(
                f"protected identifier {identifier!r} no longer appears anywhere in the tree — "
                "it was probably renamed, which breaks existing installations or upstream merges"
            )
    return problems


def main() -> int:
    problems: list[str] = []
    problems += check_stdlib_shadowing()
    problems += check_import_direction()
    problems += check_model_identifiers()
    problems += check_protected_present()

    if problems:
        print("Protected-identifier check FAILED:\n", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "\nSee docs/platform/ARCHITECTURE_BOUNDARIES.md for why each of these is protected.",
            file=sys.stderr,
        )
        return 1

    print("Protected-identifier check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
