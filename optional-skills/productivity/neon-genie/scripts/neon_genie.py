#!/usr/bin/env python3
"""Neon Genie packaging CLI — one command shape for agents and operators.

  python scripts/neon_genie.py do <job> [options]
  python scripts/neon_genie.py help [job]
  python scripts/neon_genie.py aliases

Jobs are packaging-only (validate, recipe, tests). Product judgment stays
in Hermes + SKILL.md. Advisory only — never grants spend/execute.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
PY = sys.executable

# job -> script + short description (plain English for help)
INTENTS: dict[str, dict[str, str]] = {
    "doctor": {
        "script": "doctor.py",
        "description": "Full smoke suite (start here after install)",
    },
    "privacy": {
        "script": "privacy_diagnostics.py",
        "description": "Show resolved repository privacy boundary (--json)",
    },
    "check": {
        "script": "validate_hermes_skill.py",
        "description": "Validate skill install and required files",
    },
    "run": {
        "script": "run_job.py",
        "description": "Operator run: brief/recipe → workspace + envelope",
    },
    "recipe": {
        "script": "recipe_run.py",
        "description": "Run a named example end-to-end (--list / --name)",
    },
    "route": {
        "script": "route_profiles.py",
        "description": "Suggest profiles from text or a brief YAML",
    },
    "capabilities": {
        "script": "capabilities.py",
        "description": "Machine-readable skill surface (--json)",
    },
    "validate": {
        "script": "validate_packet.py",
        "description": "Check packet JSON against a schema",
    },
    "receipt": {
        "script": "build_receipt.py",
        "description": "Build an advisory run receipt",
    },
    "envelope": {
        "script": "build_envelope.py",
        "description": "Build canonical run-envelope.json for a run dir",
    },
    "eval": {
        "script": "run_hermes_evals.py",
        "description": "Run golden fail-closed gate tests",
    },
    "transcripts": {
        "script": "check_transcripts.py",
        "description": "Check golden prose transcript structure",
    },
    "learn": {
        "script": "record_learning.py",
        "description": "Append a PROPOSED outcome to a local ledger",
    },
    "reconcile": {
        "script": "reconcile_learning.py",
        "description": "Reconcile learning ledger entries to run_id envelopes",
    },
    "dist": {
        "script": "distribution_spine.py",
        "description": "Verify/write Hub mirrors + package (distribution.yaml)",
    },
    "behavioral": {
        "script": "check_behavioral_invariants.py",
        "description": "Semantic OPEN→SEAL behavioral suite",
    },
    "runtime": {
        "script": "hermes_runtime_smoke.py",
        "description": "Isolated hub-layout install smoke (+ optional --hermes)",
    },
    "release-check": {
        "script": "release_check.py",
        "description": "Pre-release version/changelog/dist gate",
    },
}

# Everyday first in help; aliases for older script-style names
EVERYDAY = ("doctor", "check", "run", "recipe", "route", "validate", "capabilities")
VERIFY = ("eval", "transcripts", "behavioral", "runtime", "dist", "release-check", "privacy")
OUTCOMES = ("receipt", "envelope", "learn", "reconcile")

ALIASES = {
    "validate-skill": ("check", []),
    "validate-packet": ("validate", []),
    "route-profiles": ("route", []),
    "build-receipt": ("receipt", []),
    "build-envelope": ("envelope", []),
    "run-evals": ("eval", []),
    "check-transcripts": ("transcripts", []),
    "record-learning": ("learn", []),
    "distribution": ("dist", []),
    "sync-dist": ("dist", ["write"]),
    "behavior": ("behavioral", ["--suite"]),
    "caps": ("capabilities", ["--json"]),
    "product-audit": ("recipe", ["--name", "product-audit"]),
    "zero-option": ("recipe", ["--name", "zero-option"]),
    "fragmentation": ("recipe", ["--name", "fragmentation"]),
    "commercial": ("recipe", ["--name", "commercial"]),
    "audit": ("recipe", ["--name", "audit"]),
    "agentic": ("recipe", ["--name", "agentic"]),
    "memetic": ("recipe", ["--name", "memetic"]),
    "evidence": ("recipe", ["--name", "evidence"]),
    "opportunity": ("recipe", ["--name", "opportunity"]),
}


def top_help() -> str:
    lines = [
        "Neon Genie CLI — packaging only · advisory only",
        "",
        "  python scripts/neon_genie.py do <job> [options]",
        "  python scripts/neon_genie.py help [job]",
        "",
        "Everyday jobs:",
    ]
    for name in EVERYDAY:
        lines.append(f"  {name:<12} {INTENTS[name]['description']}")
    lines.append("")
    lines.append("Verify / CI:")
    for name in VERIFY:
        lines.append(f"  {name:<12} {INTENTS[name]['description']}")
    lines.append("")
    lines.append("Outcomes:")
    for name in OUTCOMES:
        lines.append(f"  {name:<12} {INTENTS[name]['description']}")
    lines.extend(
        [
            "",
            "Examples:",
            "  python scripts/neon_genie.py do doctor",
            "  python scripts/neon_genie.py do run --recipe product-audit --out out/neon-genie/demo",
            "  python scripts/neon_genie.py do run --brief examples/product-audit.brief.yaml --out out/neon-genie/demo",
            "  python scripts/neon_genie.py do capabilities --json",
            "  python scripts/neon_genie.py do recipe --list",
            "  python scripts/neon_genie.py do route --text \"zero capital first cash\" --json",
            "  python scripts/neon_genie.py do validate --packet p.json --type opportunity",
            "  python scripts/neon_genie.py do eval",
            "",
            "Hermes chat: load the skill and describe the job in plain language (see README).",
            "This CLI does not invent opportunities or grant execution authority.",
            "",
        ]
    )
    return "\n".join(lines)


def intent_help(intent: str) -> str:
    if intent not in INTENTS:
        return f"Unknown job: {intent}\nKnown: {', '.join(INTENTS)}\n"
    meta = INTENTS[intent]
    script = SCRIPT_DIR / meta["script"]
    r = subprocess.run(
        [PY, str(script), "--help"],
        cwd=SKILL_ROOT,
        capture_output=True,
        text=True,
    )
    body = r.stdout or r.stderr or ""
    return (
        f"job: {intent}\n"
        f"{meta['description']}\n"
        f"script: scripts/{meta['script']}\n\n"
        f"{body}"
    )


def run_script(script_name: str, argv: list[str]) -> int:
    script = SCRIPT_DIR / script_name
    if not script.is_file():
        print(f"FAIL: missing script {script}", file=sys.stderr)
        return 1
    r = subprocess.run([PY, str(script), *argv], cwd=SKILL_ROOT)
    return int(r.returncode)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or (argv[0] in {"-h", "--help", "help"} and len(argv) == 1):
        sys.stdout.write(top_help())
        return 0

    if argv[0] == "help":
        if len(argv) < 2:
            sys.stdout.write(top_help())
            return 0
        sys.stdout.write(intent_help(argv[1]))
        return 0

    if argv[0] == "aliases":
        for alias, (intent, fixed) in sorted(ALIASES.items()):
            extra = f" {' '.join(fixed)}" if fixed else ""
            print(f"  {alias} → do {intent}{extra}")
        return 0

    if argv[0] in ALIASES:
        intent, fixed = ALIASES[argv[0]]
        return run_script(INTENTS[intent]["script"], list(fixed) + argv[1:])

    if argv[0] != "do":
        print(f"Unknown command: {argv[0]}", file=sys.stderr)
        print("Use: python scripts/neon_genie.py do <job> …", file=sys.stderr)
        print("     python scripts/neon_genie.py help", file=sys.stderr)
        return 2

    if len(argv) < 2:
        print("usage: neon_genie.py do <job> [options]", file=sys.stderr)
        return 2

    intent = argv[1]
    rest = argv[2:]

    if intent in {"-h", "--help"}:
        sys.stdout.write(top_help())
        return 0

    if intent not in INTENTS:
        print(f"Unknown job: {intent}", file=sys.stderr)
        print("Known:", ", ".join(INTENTS), file=sys.stderr)
        return 2

    # Default flags for suite-style jobs when no args given
    if intent == "behavioral" and not rest:
        rest = ["--suite"]

    if any(t in {"-h", "--help"} for t in rest):
        sys.stdout.write(intent_help(intent))
        return 0

    return run_script(INTENTS[intent]["script"], rest)


if __name__ == "__main__":
    raise SystemExit(main())
