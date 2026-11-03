#!/usr/bin/env python3
"""Full-suite operator smoke for Neon Genie (stdlib only).

Runs check → fixtures → eval → transcripts → recipe list smoke.
Exit non-zero on first failure.

Usage:
  python scripts/doctor.py
  python scripts/neon_genie.py do doctor
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
PY = sys.executable
CLI = SCRIPT_DIR / "neon_genie.py"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import paths as ng_paths  # noqa: E402


def step(name: str, args: list[str]) -> int:
    print(f"==> {name}")
    r = subprocess.run([PY, str(CLI), *args], cwd=SKILL_ROOT)
    if r.returncode != 0:
        print(f"FAIL: {name} (exit {r.returncode})", file=sys.stderr)
        return r.returncode
    print(f"OK: {name}")
    return 0


def main() -> int:
    steps: list[tuple[str, list[str]]] = [
        ("skill integrity", ["do", "check"]),
        ("privacy diagnostics", ["do", "privacy", "--json"]),
    ]
    # Distribution spine only on full tree (has distribution.yaml + root schemas/)
    if (SKILL_ROOT / "distribution.yaml").is_file() and not ng_paths.is_hub_layout():
        steps.append(("distribution spine", ["do", "dist", "verify"]))
    else:
        print("==> distribution spine")
        print("SKIP: hub layout or no distribution.yaml (expected on Hub installs)")

    steps.extend(
        [
        ("golden gate evals", ["do", "eval"]),
        ("golden transcripts", ["do", "transcripts"]),
        ("behavioral invariants", ["do", "behavioral", "--suite"]),
        ("recipe list", ["do", "recipe", "--list"]),
        (
            "recipe smoke: product-audit",
            ["do", "recipe", "--name", "product-audit", "--out", "out/neon-genie/doctor-product-audit"],
        ),
        (
            "recipe smoke: commercial",
            ["do", "recipe", "--name", "commercial", "--out", "out/neon-genie/doctor-commercial"],
        ),
        (
            "recipe smoke: audit",
            ["do", "recipe", "--name", "audit", "--out", "out/neon-genie/doctor-audit"],
        ),
        (
            "recipe smoke: agentic",
            ["do", "recipe", "--name", "agentic", "--out", "out/neon-genie/doctor-agentic"],
        ),
        (
            "recipe smoke: memetic",
            ["do", "recipe", "--name", "memetic", "--out", "out/neon-genie/doctor-memetic"],
        ),
        (
            "recipe smoke: evidence",
            ["do", "recipe", "--name", "evidence", "--out", "out/neon-genie/doctor-evidence"],
        ),
        (
            "recipe smoke: opportunity",
            ["do", "recipe", "--name", "opportunity", "--out", "out/neon-genie/doctor-opportunity"],
        ),
        (
            "recipe smoke: capital-sprint",
            ["do", "recipe", "--name", "capital-sprint", "--out", "out/neon-genie/doctor-capital-sprint"],
        ),
        (
            "sample opportunity validate",
            [
                "do",
                "validate",
                "--packet",
                "examples/packets/sample-opportunity.packet.json",
                "--type",
                "opportunity",
            ],
        ),
        (
            "run-envelope present after product-audit",
            [
                "do",
                "validate",
                "--packet",
                "out/neon-genie/doctor-product-audit/run-envelope.json",
                "--type",
                "envelope",
                "--strict-authority",
            ],
        ),
        ]
    )

    for name, args in steps:
        code = step(name, args)
        if code != 0:
            return code

    # Founder-routing regression (packaging CLI) — keep cold-start triggers honest
    route_test = SCRIPT_DIR / "test_route_profiles.py"
    if route_test.is_file():
        print("==> founder profile routing")
        r = subprocess.run([PY, str(route_test)], cwd=SKILL_ROOT)
        if r.returncode != 0:
            print("FAIL: founder profile routing", file=sys.stderr)
            return r.returncode
        print("OK: founder profile routing")
    else:
        print("==> founder profile routing")
        print("SKIP: test_route_profiles.py not present")

    print("")
    print("PASS: neon-genie doctor (all smokes green)")
    print("  authority remains advisory_only; no execution granted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
