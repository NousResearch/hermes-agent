#!/usr/bin/env python3
"""Single source of truth for what a js-autofix patch may touch.

The autofix workflow splits work across a trust boundary: generate-patch runs
``npm run fix`` (which executes the repo's own eslint plugins and configs) and
uploads a patch artifact; apply-patch downloads and applies it. The produce
side can only pre-filter, since a hostile run controls both the artifact and
its own vetting. The authoritative check lives in apply-patch, which calls
``--staged`` here after ``git add -A``.

Two jobs needing the same policy is why this file exists: the last time the
deny list lived in two places (pathspec excludes on the produce side, a case
statement on the apply side) they diverged silently.

Usage:
    autofix_path_gate.py --deny-globs      git pathspec excludes, one per line
    autofix_path_gate.py --allowed PATH    exit 0 iff PATH may appear in a patch
    autofix_path_gate.py --staged          verify the staged diff in cwd
"""

from __future__ import annotations

import re
import subprocess
import sys

# Denied at any depth. Package-manager manifests and lockfiles (a hostile
# dependency lands postinstall), eslint configuration (controls what the
# unprivileged job executes), .github (workflows, actions, CODEOWNERS), and
# configs that tools execute or that steer compilation.
TOOL_CONFIGS = (
    "astro", "babel", "commitlint", "drizzle", "electron-builder", "jest",
    "knip", "lint-staged", "next", "nuxt", "oxlint", "playwright", "postcss",
    "prettier", "rollup", "svelte", "tailwind", "tsdown", "tsup", "typedoc",
    "vite", "vitest", "webpack",
)

DENY_GLOBS = [
    "**/.github",
    "**/.github/**",
    "**/package.json",
    "**/package-lock.json",
    "**/npm-shrinkwrap.json",
    "**/yarn.lock",
    "**/pnpm-lock.yaml",
    "**/.eslintrc",
    "**/.eslintrc.*",
    "**/eslint.config.*",
    "**/tsconfig*.json",
    *(f"**/{name}.config.*" for name in TOOL_CONFIGS),
]

ALLOWED_EXTENSIONS = (".js", ".cjs", ".mjs", ".ts", ".tsx", ".json")


def _glob_to_re(glob: str) -> re.Pattern[str]:
    """Translate a git pathspec glob to a regex. ``**/`` means zero or more
    leading components (glob magic), other ``**`` spans components, ``*``
    stays within one component."""
    out = re.escape(glob)
    out = out.replace(r"\*\*/", r"(?:.*/)?")
    out = out.replace(r"\*\*", r".*")
    out = out.replace(r"\*", r"[^/]*")
    return re.compile(f"^{out}$")


_DENY_RES = [_glob_to_re(g) for g in DENY_GLOBS]


def denied(path: str) -> bool:
    p = path.lower()
    return any(r.match(p) for r in _DENY_RES)


def allowed(path: str) -> bool:
    return not denied(path) and path.lower().endswith(ALLOWED_EXTENSIONS)


def _staged() -> int:
    """Verify the staged diff in cwd. Only plain content edits or new
    regular files under the allowed set may reach the bot PR; deletions,
    renames, mode changes, symlinks, and gitlinks are refused because
    ``npm run fix`` never produces any of them."""
    # -z so paths arrive raw: without it git C-quotes non-ASCII and
    # whitespace-bearing names and the real path never reaches the checks.
    name_status = subprocess.run(
        ["git", "diff", "--cached", "--name-status", "-z"],
        check=True, capture_output=True, text=True,
    ).stdout
    # --summary stays quoted on purpose: without -z a path containing a
    # newline cannot smuggle a fake " create mode 100644 " line.
    summary = subprocess.run(
        ["git", "diff", "--cached", "--summary"],
        check=True, capture_output=True, text=True,
    ).stdout

    bad: list[str] = []
    tokens = name_status.split("\0")
    i = 0
    while i < len(tokens):
        status, path = tokens[i], tokens[i + 1] if i + 1 < len(tokens) else ""
        i += 2
        if not status:
            continue
        if status.startswith(("R", "C")):
            i += 1  # renames and copies carry a second path token
        if not status.startswith(("M", "A")):
            bad.append(f"unsupported-change {status}: {path}")
            continue
        if denied(path):
            bad.append(f"denied-path: {path}")
        elif not allowed(path):
            bad.append(f"disallowed-extension: {path}")
    for line in summary.splitlines():
        if line and not line.startswith(" create mode 100644 "):
            bad.append(f"non-plain-change:{line}")

    if bad:
        print("\n".join(bad), file=sys.stderr)
        return 1
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--deny-globs"]:
        print("\n".join(DENY_GLOBS))
        return 0
    if len(argv) == 2 and argv[0] == "--allowed":
        return 0 if allowed(argv[1]) else 1
    if len(argv) == 2 and argv[0] == "--denied":
        return 0 if denied(argv[1]) else 1
    if argv == ["--staged"]:
        return _staged()
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
