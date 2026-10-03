#!/usr/bin/env python3
"""Unit tests for founder-language and profile routing (stdlib only)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_ROOT = SCRIPT_DIR.parent
PY = sys.executable
ROUTE = SCRIPT_DIR / "route_profiles.py"


def route(text: str) -> dict:
    r = subprocess.run(
        [PY, str(ROUTE), "--text", text, "--json"],
        cwd=SKILL_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr or r.stdout
    return json.loads(r.stdout)


def test_o1_founder_roadmap_selects_opportunity() -> None:
    out = route(
        "I'm between jobs with limited money and need a roadmap for my app idea"
    )
    selected = set(out["selected"])
    assert "core" in selected and "privacy" in selected
    assert "opportunity_mining" in selected, selected
    assert "zero_option" in selected, selected
    print("PASS: O1 founder roadmap + scarcity")


def test_product_audit_still_works() -> None:
    out = route("product audit for my SaaS")
    assert "product_architecture" in out["selected"]
    print("PASS: product audit")


def test_zero_capital_phrase_still_works() -> None:
    out = route("zero capital first cash")
    assert "zero_option" in out["selected"]
    print("PASS: zero capital phrase")


def test_venture_capital_does_not_force_zero_option() -> None:
    """'venture capital' alone is not scarcity — do not over-trigger zero_option."""
    out = route("research venture capital firm partners for a Series A thesis")
    assert "zero_option" not in out["selected"], out["selected"]
    print("PASS: venture capital not zero_option")


def test_capital_sprint_routes() -> None:
    out = route("design a capital sprint and impact object for our annual fund")
    assert "capital_sprint" in out["selected"], out["selected"]
    print("PASS: capital_sprint routes")


def test_brief_preferred_capital_sprint() -> None:
    brief = SKILL_ROOT / "examples" / "capital-sprint.brief.yaml"
    r = subprocess.run(
        [PY, str(ROUTE), "--request", str(brief), "--json"],
        cwd=SKILL_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr or r.stdout
    out = json.loads(r.stdout)
    assert "capital_sprint" in out["selected"]
    print("PASS: capital-sprint brief preferred_profiles")


if __name__ == "__main__":
    test_o1_founder_roadmap_selects_opportunity()
    test_product_audit_still_works()
    test_zero_capital_phrase_still_works()
    test_venture_capital_does_not_force_zero_option()
    test_capital_sprint_routes()
    test_brief_preferred_capital_sprint()
    print("ALL PASS: test_route_profiles")
