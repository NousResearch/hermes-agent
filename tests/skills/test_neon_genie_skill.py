"""Tests for the optional neon-genie skill (stdlib + pytest).

Covers SKILL.md authoring standards and packaging CLI helpers that ship
with the optional leaf: route_profiles founder language, and doctor surface.
No live network calls.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "productivity" / "neon-genie"
ROUTE = SKILL_DIR / "scripts" / "route_profiles.py"
CLI = SKILL_DIR / "scripts" / "neon_genie.py"
PY = sys.executable


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    # Prefer yaml if present; else minimal parse for required fields
    try:
        import yaml  # type: ignore

        return yaml.safe_load(m.group(1))
    except Exception:
        data: dict = {}
        for line in m.group(1).splitlines():
            if ":" in line and not line.startswith(" "):
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip().strip('"').strip("'")
        return data


@pytest.fixture(scope="module")
def route_mod():
    spec = importlib.util.spec_from_file_location("ng_route", ROUTE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # route_profiles imports paths from same dir
    sys.path.insert(0, str(SKILL_DIR / "scripts"))
    try:
        spec.loader.exec_module(module)
    finally:
        pass
    return module


def test_skill_files_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()
    assert CLI.is_file()
    assert ROUTE.is_file()
    assert (SKILL_DIR / "references" / "profiles" / "core.md").is_file()
    assert (SKILL_DIR / "references" / "profiles" / "privacy.md").is_file()
    assert (SKILL_DIR / "references" / "schemas" / "run-envelope.schema.json").is_file()
    assert (SKILL_DIR / "PRIVACY.md").is_file()


def test_description_within_limit(frontmatter: dict) -> None:
    desc = frontmatter["description"]
    if isinstance(desc, str) and "\n" in desc:
        # folded YAML becomes single string with spaces
        desc = " ".join(desc.split())
    assert isinstance(desc, str)
    assert len(desc) <= 60, f"description is {len(desc)} chars (limit 60): {desc!r}"
    assert desc.endswith(".")


def test_required_frontmatter_fields(frontmatter: dict) -> None:
    assert frontmatter["name"] == "neon-genie"
    for field in ("version", "author", "license", "platforms"):
        assert frontmatter.get(field), f"missing frontmatter field: {field}"
    author = str(frontmatter["author"])
    assert "Daniel Meyer" in author
    assert "@scrimshawlife-ctrl" in author
    meta = frontmatter.get("metadata") or {}
    hermes = meta.get("hermes") if isinstance(meta, dict) else {}
    assert hermes.get("category") == "productivity"


def test_skill_body_has_modern_sections() -> None:
    body = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert body.lstrip().startswith("---") or "# Neon Genie Skill" in body
    assert "# Neon Genie Skill" in body
    for heading in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in body, f"SKILL.md missing section: {heading}"


def test_founder_route_profiles(route_mod) -> None:
    text = (
        "I'm between jobs with limited money and need a roadmap for my app idea"
    )
    selected = set(route_mod.ensure_privacy(route_mod.match_profiles(text)))
    # ensure_privacy alone doesn't prepend core; match + ensure like CLI
    profiles = route_mod.match_profiles(text)
    selected = set(route_mod.ensure_privacy(["core"] + profiles))
    assert "opportunity_mining" in selected
    assert "zero_option" in selected
    assert "privacy" in selected


def test_venture_capital_not_zero_option(route_mod) -> None:
    profiles = set(route_mod.match_profiles("research venture capital firm partners"))
    assert "zero_option" not in profiles


def test_capital_sprint_routes(route_mod) -> None:
    profiles = set(
        route_mod.match_profiles(
            "design a capital sprint and impact object for our annual fund"
        )
    )
    assert "capital_sprint" in profiles


def test_cli_check_and_capabilities() -> None:
    r = subprocess.run(
        [PY, str(CLI), "do", "check"],
        cwd=SKILL_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    r2 = subprocess.run(
        [PY, str(CLI), "do", "capabilities", "--json"],
        cwd=SKILL_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r2.returncode == 0, r2.stderr + r2.stdout
    cap = json.loads(r2.stdout)
    assert cap["authority"] == "advisory_only"
    assert cap["grants_execution"] is False
