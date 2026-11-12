#!/usr/bin/env python3
"""Regression checks for the design specification compiler."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "design.md"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "kubrick.py"),
                "design-build",
                "--input",
                str(ROOT / "templates" / "design-specification.yaml"),
                "--output",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(result.stderr or result.stdout)
        text = output.read_text(encoding="utf-8")
        required = [
            "# Example Project — Design Specification",
            "## Design Pillars",
            "## Invariants",
            "## Design Languages",
            "## Continuity Contract",
            "## Drift Rule",
            "`OBSERVED` — document: README.md",
        ]
        missing = [heading for heading in required if heading not in text]
        if missing:
            raise SystemExit(f"missing rendered sections: {missing}")
        print("design specification regression: PASS")


if __name__ == "__main__":
    main()
