"""Validation tests for map/objects/ module cards.

Covers:
- expected card files exist
- each card contains required YAML frontmatter and body sections
- module order matches deterministic manifest
- INDEX.md lists every generated card
- generation is deterministic across runs
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
OBJECTS_DIR = REPO_ROOT / "map" / "objects"
SCRIPT = REPO_ROOT / "scripts" / "generate_map_objects.py"


EXPECTED_MODULES = [
    {
        "name": "agent",
        "file": "agent.md",
        "id": "agent",
        "universe": "repo",
    },
    {
        "name": "gateway",
        "file": "gateway.md",
        "id": "gateway",
        "universe": "runtime",
    },
    {
        "name": "plugins",
        "file": "plugins.md",
        "id": "plugins",
        "universe": "repo",
    },
    {
        "name": "tools",
        "file": "tools.md",
        "id": "tools",
        "universe": "repo",
    },
    {
        "name": "hermes_cli",
        "file": "hermes_cli.md",
        "id": "hermes-cli",
        "universe": "repo",
    },
    {
        "name": "optional-mcps",
        "file": "optional-mcps.md",
        "id": "optional-mcps",
        "universe": "repo",
    },
    {
        "name": "src-graphify",
        "file": "src-graphify.md",
        "id": "src-graphify",
        "universe": "repo",
    },
]

REQUIRED_FRONTMATTER_FIELDS = [
    "id:",
    "kind: object",
    "universe:",
    "name:",
    "summary:",
    "shape: object",
    "path:",
    "interface:",
    "depends_on:",
]

REQUIRED_BODY_SECTIONS = ["## Purpose", "## Inputs", "## Outputs", "## Dependencies"]


def test_objects_dir_exists() -> None:
    assert OBJECTS_DIR.is_dir(), f"Missing map/objects directory at {OBJECTS_DIR}"


def test_script_exists() -> None:
    assert SCRIPT.is_file(), f"Missing generator script at {SCRIPT}"


def test_all_cards_present() -> None:
    for module in EXPECTED_MODULES:
        card = OBJECTS_DIR / module["file"]
        assert card.is_file(), f"Missing card for module: {module['name']}"


def test_index_present() -> None:
    index = OBJECTS_DIR / "INDEX.md"
    assert index.is_file(), "Missing INDEX.md manifest"


def test_index_lists_all_modules() -> None:
    index = OBJECTS_DIR / "INDEX.md"
    text = index.read_text(encoding="utf-8")
    for module in EXPECTED_MODULES:
        assert f"- [{module['name']}](./{module['file']})" in text, (
            f"INDEX.md missing link for {module['name']}"
        )


def test_cards_contain_required_frontmatter() -> None:
    for module in EXPECTED_MODULES:
        card = OBJECTS_DIR / module["file"]
        text = card.read_text(encoding="utf-8")
        assert text.startswith("---"), f"{module['file']} missing YAML frontmatter"
        for field in REQUIRED_FRONTMATTER_FIELDS:
            assert field in text, f"{module['file']} missing frontmatter field: {field}"


def test_cards_contain_required_body_sections() -> None:
    for module in EXPECTED_MODULES:
        card = OBJECTS_DIR / module["file"]
        text = card.read_text(encoding="utf-8")
        for section in REQUIRED_BODY_SECTIONS:
            assert section in text, f"{module['file']} missing section: {section}"


def test_index_module_order_is_deterministic() -> None:
    index = OBJECTS_DIR / "INDEX.md"
    text = index.read_text(encoding="utf-8")
    for first, second in zip(EXPECTED_MODULES, EXPECTED_MODULES[1:]):
        pos_first = text.index(f"- [{first['name']}]")
        pos_second = text.index(f"- [{second['name']}]")
        assert pos_first < pos_second, f"Module order mismatch: {first['name']} before {second['name']}"


def test_generation_is_deterministic_structure() -> None:
    """Running the generator should produce stable structure/content hashes."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("generate_map_objects", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out_first = REPO_ROOT / "map" / "objects-determinism-1"
    out_second = REPO_ROOT / "map" / "objects-determinism-2"
    for d in (out_first, out_second):
        if d.exists():
            for p in sorted(d.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()

    manifest_first = module.generate(REPO_ROOT, out_first)
    manifest_second = module.generate(REPO_ROOT, out_second)

    assert [m["name"] for m in manifest_first["modules"]] == [
        m["name"] for m in manifest_second["modules"]
    ]
    assert [m["id"] for m in manifest_first["modules"]] == [
        m["id"] for m in manifest_second["modules"]
    ]

    files_first = sorted(out_first.rglob("*.md"))
    files_second = sorted(out_second.rglob("*.md"))
    assert len(files_first) == len(files_second)

    for f1, f2 in zip(files_first, files_second):
        assert f1.name == f2.name
        if f1.name == "INDEX.md":
            continue
        h1 = hashlib.sha256(f1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(f2.read_bytes()).hexdigest()
        assert h1 == h2, f"Determinism mismatch: {f1.name}\n{h1}\n{h2}"

    for d in (out_first, out_second):
        for p in sorted(d.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
