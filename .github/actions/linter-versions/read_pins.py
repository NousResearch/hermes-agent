#!/usr/bin/env python3
"""
Print the pinned linter versions from pyproject.toml as ``name=version`` lines.

CI feeds these to ``uv tool install <name>==<version>`` so the blocking lint
job runs the same linters ``uv sync --extra dev`` gives contributors. A bare
``uv tool install ruff`` floats to whatever released this morning, which can
turn main red on code nobody touched — and lets a local ruff disagree with
CI. This repo is extra exposed to that: ``[tool.ruff] preview = true`` opts
into preview rules, whose behavior is explicitly allowed to change between
releases.

Usage:
    # Emit ``ruff=<version>`` / ``ty=<version>`` for $GITHUB_OUTPUT
    python3 .github/actions/linter-versions/read_pins.py

    # Read a pyproject.toml somewhere else
    python3 .github/actions/linter-versions/read_pins.py path/to/pyproject.toml

Exit status:
    0 — every tool had exactly one exact ``==`` pin
    1 — a pin is missing, duplicated, or not an exact version

Failing loud is the point: a silent fallback to a floating install would put
back the exact hole this closes.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

# Tools CI installs standalone (outside the venv) and therefore has to pin
# by hand. Both live in the ``dev`` extra.
TOOLS = ("ruff", "ty")

EXTRA = "dev"

# The version is interpolated into a shell install command and pyproject.toml
# is PR-editable, so require it to be inert as a shell word. PEP 440 versions
# only ever use this alphabet.
VERSION_RE = re.compile(r"\A[A-Za-z0-9._+!-]+\Z")

# Distribution name of a PEP 508 requirement — everything before the extras
# bracket, any version operator, or an environment marker. Matching on this
# rather than on ``name==`` means ``ruff>=0.15`` is recognised as the ruff
# requirement and rejected as inexact, instead of looking absent.
NAME_RE = re.compile(r"\A\s*([A-Za-z0-9._-]+)")


def pins(pyproject: Path) -> list[str]:
    with open(pyproject, "rb") as fh:
        data = tomllib.load(fh)

    try:
        specs = data["project"]["optional-dependencies"][EXTRA]
    except KeyError:
        sys.exit(
            f"{pyproject}: no [project.optional-dependencies] {EXTRA} extra — "
            f"cannot resolve pinned versions for {', '.join(TOOLS)}"
        )

    def name_of(spec: str) -> str:
        found = NAME_RE.match(spec)
        # Normalised per PEP 503 so ``Ruff`` / ``ruff`` compare equal.
        return re.sub(r"[-_.]+", "-", found.group(1)).lower() if found else ""

    lines = []
    for tool in TOOLS:
        matched = [s for s in specs if name_of(s) == tool]
        if len(matched) != 1:
            sys.exit(
                f"{pyproject}: expected exactly one '{tool}' requirement in the "
                f"{EXTRA} extra, found {len(matched)}: {matched}"
            )

        spec = matched[0]
        if "==" not in spec:
            sys.exit(
                f"{pyproject}: '{spec}' is not an exact pin. CI installs this "
                f"tool standalone and needs a '{tool}==<version>' pin to stay "
                f"reproducible."
            )

        version = spec.split("==", 1)[1].strip()
        if not VERSION_RE.match(version):
            sys.exit(f"{pyproject}: refusing unexpected version string {version!r} for {tool}")

        lines.append(f"{tool}={version}")

    return lines


def main() -> None:
    pyproject = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pyproject.toml")
    if not pyproject.is_file():
        sys.exit(f"{pyproject}: not found (run from the repository root)")

    for line in pins(pyproject):
        print(line)


if __name__ == "__main__":
    main()
