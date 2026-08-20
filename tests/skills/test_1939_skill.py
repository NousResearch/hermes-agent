"""
Smoke tests for the 1939 optional skill.

Verifies:
  - SKILL.md frontmatter conforms to the hardline format
  - shipped palette JSONs have the required schema (8 roles, 10 tints, hex prefix)
  - memes index is valid and has required fields
  - MCP server script parses as valid Python
  - the _keyword_search bug fix (uses query key, not stale name variable)
"""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "creative" / "1939"
FLAGSHIP_DIR = SKILL_DIR / "palettes" / "flagship"
MEMES_INDEX = SKILL_DIR / "palettes" / "memes" / "index.json"
SERVER_DIR = SKILL_DIR / "server"


# ---------------------------------------------------------------------------
# Frontmatter conformance (hardline standards)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline <=60): {desc!r}"


def test_description_ends_with_period(frontmatter) -> None:
    desc = frontmatter["description"]
    assert desc.endswith("."), f"description should end with a period: {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert str(frontmatter["name"]) == "1939"


def test_platforms_cross_platform(frontmatter) -> None:
    platforms = frontmatter["platforms"]
    assert set(platforms) == {"linux", "macos", "windows"}, f"expected cross-platform: {platforms!r}"


def test_author_credits_human_first(frontmatter) -> None:
    author = frontmatter["author"]
    assert "0xCuttlefish" in author, f"author should credit the human contributor: {author!r}"


def test_license_cc0(frontmatter) -> None:
    assert frontmatter["license"] == "CC0"


def test_category_is_creative(frontmatter) -> None:
    assert frontmatter["metadata"]["hermes"]["category"] == "creative"


# ---------------------------------------------------------------------------
# Palette JSON invariants
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def flagship_files() -> list[Path]:
    files = sorted(FLAGSHIP_DIR.glob("*.brand.json"))
    assert len(files) > 0, "no flagship palette JSONs found"
    return files


def test_flagship_count(flagship_files) -> None:
    assert len(flagship_files) == 29, f"expected 29 flagship palettes, got {len(flagship_files)}"


def test_flagship_schema(flagship_files) -> None:
    required_top = {"name", "slug", "collection", "roles"}
    for fpath in flagship_files:
        with open(fpath) as f:
            data = json.load(f)
        missing = required_top - set(data.keys())
        assert not missing, f"{fpath.name} missing top-level fields: {missing}"


def test_flagship_roles_count(flagship_files) -> None:
    for fpath in flagship_files:
        with open(fpath) as f:
            data = json.load(f)
        assert len(data["roles"]) == 8, f"{fpath.name} has {len(data['roles'])} roles, expected 8"


def test_flagship_role_fields(flagship_files) -> None:
    required_role = {"hex", "tints", "curve", "legend_text"}
    for fpath in flagship_files:
        with open(fpath) as f:
            data = json.load(f)
        for role_name, role_data in data["roles"].items():
            missing = required_role - set(role_data.keys())
            assert not missing, f"{fpath.name} role '{role_name}' missing: {missing}"


def test_flagship_tints_count(flagship_files) -> None:
    for fpath in flagship_files:
        with open(fpath) as f:
            data = json.load(f)
        for role_name, role_data in data["roles"].items():
            assert len(role_data["tints"]) == 10, (
                f"{fpath.name} role '{role_name}' has {len(role_data['tints'])} tints, expected 10"
            )


def test_hex_values_prefixed(flagship_files) -> None:
    for fpath in flagship_files:
        with open(fpath) as f:
            data = json.load(f)
        for role_name, role_data in data["roles"].items():
            assert role_data["hex"].startswith("#"), (
                f"{fpath.name} role '{role_name}' hex missing # prefix"
            )
            for tint in role_data["tints"]:
                assert tint.startswith("#"), (
                    f"{fpath.name} role '{role_name}' tint missing # prefix: {tint}"
                )


# ---------------------------------------------------------------------------
# Memes index invariants
# ---------------------------------------------------------------------------

def test_memes_index_loads() -> None:
    with open(MEMES_INDEX) as f:
        data = json.load(f)
    assert isinstance(data, list), "memes index should be a list"
    assert len(data) > 0, "memes index is empty"


def test_memes_entry_fields() -> None:
    with open(MEMES_INDEX) as f:
        data = json.load(f)
    for entry in data:
        assert "slug" in entry, f"memes entry missing slug"
        assert "name" in entry, f"memes entry missing name"


# ---------------------------------------------------------------------------
# Server script validity
# ---------------------------------------------------------------------------

def test_mcp_server_parses() -> None:
    src = (SERVER_DIR / "mcp_1939_server.py").read_text()
    ast.parse(src)


def test_keyword_search_uses_query_not_name() -> None:
    """Regression test for PR #51004 review: _keyword_search must receive the
    caller's query key, not a stale 'name' variable from a loop iteration."""
    src = (SERVER_DIR / "mcp_1939_server.py").read_text()
    assert "_keyword_search(key," in src, (
        "resolve_palette should pass 'key' to _keyword_search"
    )
    assert "_keyword_search(name," not in src, (
        "resolve_palette still passes stale 'name' to _keyword_search (bug)"
    )


def test_server_path_not_palettes() -> None:
    """Regression test for PR #51004 review: configuration example should
    point to server/ not palettes/."""
    src = (SERVER_DIR / "mcp_1939_server.py").read_text()
    assert "palettes/mcp_1939_server.py" not in src, (
        "server config example points to wrong path (palettes/ not server/)"
    )


def test_requirements_have_upper_bounds() -> None:
    """Regression test for PR #51004 review: all deps need upper bounds."""
    src = (SERVER_DIR / "requirements.txt").read_text()
    for line in src.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            assert "<" in line, f"dependency lacks upper bound: {line!r}"