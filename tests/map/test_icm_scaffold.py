"""Behavior-contract tests for the ICM system-map scaffold.

These tests validate the scaffold under `map/` created by this contribution:
- map/CLAUDE.md
- map/CONTEXT.md
- map/_meta/schema.md
- map/_templates/*.md
- new hand-authored cards in map/

They deliberately do NOT validate the auto-generated `map/objects/`,
`map/processes/`, or `map/objects-determinism-*` artifacts produced by
`scripts/generate_map_objects.py`, which have their own contract in
`tests/test_map_objects.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = REPO_ROOT / "map"

VALID_UNIVERSES = {"repo", "runtime", "config", "mcp"}
VALID_KINDS = {"object", "process"}
ID_PATTERN = re.compile(r"^[a-z0-9_.-]+$")

EXCLUDE_DIRS = {
    MAP_DIR / "objects",
    MAP_DIR / "objects-determinism-1",
    MAP_DIR / "objects-determinism-2",
    MAP_DIR / "processes",
}


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        raise ValueError(f"Missing YAML frontmatter in {path}")
    _, fm, _ = text.split("---", 2)
    return yaml.safe_load(fm) or {}


def _is_under_excluded(path: Path) -> bool:
    if path.parent in EXCLUDE_DIRS:
        return True
    for excluded in EXCLUDE_DIRS:
        if excluded in path.parents:
            return True
    return False


def _scaffold_cards() -> list[Path]:
    paths: list[Path] = []
    for path in MAP_DIR.glob("*.md"):
        if path.name in {"CLAUDE.md", "CONTEXT.md"}:
            continue
        paths.append(path)
    for path in (MAP_DIR / "_templates").glob("*.md"):
        paths.append(path)
    return sorted(paths)


def test_scaffold_files_exist():
    required = [
        MAP_DIR / "CLAUDE.md",
        MAP_DIR / "CONTEXT.md",
        MAP_DIR / "_meta" / "schema.md",
        MAP_DIR / "_templates" / "object.md",
        MAP_DIR / "_templates" / "process.md",
    ]
    for path in required:
        assert path.exists(), f"Missing required map file: {path}"


def test_claude_md_walkability():
    claude = (MAP_DIR / "CLAUDE.md").read_text(encoding="utf-8")
    assert "map/_meta/schema.md" in claude
    assert "map/_templates/" in claude
    assert "map/CONTEXT.md" in claude


def test_schema_defines_required_fields():
    schema_text = (MAP_DIR / "_meta" / "schema.md").read_text(encoding="utf-8")
    for field in ["id", "kind", "universe", "name", "summary"]:
        assert field in schema_text


def test_cards_follow_schema():
    card_paths = [p for p in _scaffold_cards() if p.parent.name != "_templates"]
    if not card_paths:
        raise AssertionError("No scaffold map cards found")

    ids_by_universe: dict[str, list[str]] = {}

    for path in card_paths:
        fm = _frontmatter(path)
        assert fm.get("kind") in VALID_KINDS, f"Bad kind in {path}: {fm.get('kind')}"
        universe = fm.get("universe")
        assert universe in VALID_UNIVERSES, f"Bad universe in {path}: {universe}"
        card_id = fm.get("id")
        assert isinstance(card_id, str) and ID_PATTERN.match(card_id), (
            f"Bad id in {path}: {card_id}"
        )
        ids_by_universe.setdefault(universe, []).append(card_id)

        if fm.get("kind") == "process":
            steps = fm.get("steps")
            assert isinstance(steps, list) and steps, f"process card {path} missing steps"
            for step in steps:
                assert isinstance(step, dict)
                assert "id" in step and "summary" in step

    for universe, ids in ids_by_universe.items():
        assert len(ids) == len(set(ids)), f"Duplicate ids in universe {universe}: {ids}"


def test_templates_have_valid_frontmatter():
    for template in (MAP_DIR / "_templates").glob("*.md"):
        fm = _frontmatter(template)
        assert fm.get("shape") in {"object", "process"}


def test_no_hermes_env_vars_in_scaffold():
    scaffold_text_paths = [MAP_DIR / "CLAUDE.md", MAP_DIR / "CONTEXT.md"]
    scaffold_text_paths.extend(
        p for p in _scaffold_cards() if not _is_under_excluded(p)
    )

    env_var_pattern = re.compile(r"\bHERMES_[A-Z0-9_]+\b")
    for path in scaffold_text_paths:
        text = path.read_text(encoding="utf-8")
        matches = env_var_pattern.findall(text)
        assert not matches, f"HERMES_ env var references in {path}: {matches}"


def test_walkability_from_root():
    assert (MAP_DIR / "CLAUDE.md").exists()
    assert (MAP_DIR / "_meta" / "schema.md").exists()
    assert (MAP_DIR / "_templates" / "object.md").exists()
    assert (MAP_DIR / "_templates" / "process.md").exists()


def test_card_dependencies_are_qualified_when_cross_universe():
    card_paths = [p for p in _scaffold_cards() if not _is_under_excluded(p)]
    for path in card_paths:
        fm = _frontmatter(path)
        universe = fm.get("universe")
        for field in ("depends_on", "produces", "consumes"):
            values = fm.get(field) or []
            for ref in values:
                if ":" not in ref:
                    same_universe_ids = [
                        _frontmatter(p).get("id")
                        for p in card_paths
                        if _frontmatter(p).get("universe") == universe
                    ]
                    assert ref in same_universe_ids, (
                        f"Unqualified ref {ref} in {path} not found in universe {universe}"
                    )
