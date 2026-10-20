"""Invariant tests for the optional petasos-finance-vault skill.

Covers optional-skills/finance/petasos-finance-vault — the Obsidian
household-finance ledger vault. Contracts only, no snapshots: frontmatter,
engine/data split, privacy tripwire presence, and the PyYAML dependency
contract (functional when present, LOUD exit-2 failure when absent —
INV-2 in the skill's own terms). No network access.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILL = REPO / "optional-skills" / "finance" / "petasos-finance-vault"


def _frontmatter() -> dict:
    content = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert content.startswith("---")
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "unclosed frontmatter"
    import yaml  # the skill's own single dependency

    return yaml.safe_load(content[3 : m.start() + 3])


def test_frontmatter_contract():
    fm = _frontmatter()
    assert fm["name"] == "petasos-finance-vault"
    desc = fm["description"]
    assert len(desc) <= 60 and desc.endswith(".")
    assert fm["author"].startswith("Nhut Duong")
    assert fm["license"] == "MIT"
    assert set(fm["platforms"]) <= {"linux", "macos", "windows"}
    hermes = fm.get("metadata", {}).get("hermes", {})
    assert hermes.get("tags")
    assert "schedule" in hermes.get("blueprint", {})
    env_vars = fm.get("required_environment_variables", [])
    names = {e["name"] for e in env_vars}
    assert {"AGENTMAIL_API_KEY", "PETASOS_INBOX"} <= names
    for entry in env_vars:
        assert {"name", "prompt", "help", "required_for"} <= set(entry)


def test_engine_data_split():
    # Generic engine package + thin wrappers; bank intelligence lives in
    # data files, never hardcoded in Python.
    assert (SKILL / "petasos" / "__init__.py").is_file()
    for wrapper in ("pipeline.py", "gen_views.py", "monthly_close.py", "bank_adapters.py"):
        assert (SKILL / "scripts" / wrapper).is_file()
    assert (SKILL / "banks.yaml").is_file()
    assert (SKILL / "templates" / "rules.md").is_file()
    engine = "".join(p.read_text(encoding="utf-8") for p in (SKILL / "petasos").glob("*.py"))
    assert "banks.yaml" not in engine or "yaml.safe_load" in engine
    # No real financial institution is hardcoded as logic in the engine.
    for bank_token in ("ACB", "BIDV", "HSBC", "Vietcombank"):
        assert f'"{bank_token}"' not in engine


def test_privacy_tripwire_shipped():
    # The sealed-corpus privacy guard is part of the distributed tree.
    assert (SKILL / "tests" / "test_privacy.py").is_file()
    assert (SKILL / "fixtures" / "expected.yaml").is_file()


def test_yaml_present_functional_else_loud(capsys):
    """INV-2: without PyYAML the engine fails LOUD (exit 2 + install hint),
    never silently; with PyYAML the fixture corpus stays green (offline)."""
    proc = subprocess.run(
        [sys.executable, "scripts/bank_adapters.py", "--banks"],
        cwd=SKILL, capture_output=True, text=True, timeout=120,
    )
    try:
        import yaml  # noqa: F401

        yaml_missing = False
    except ImportError:
        yaml_missing = True

    if yaml_missing:
        assert proc.returncode == 2
        assert "pyyaml" in (proc.stdout + proc.stderr).lower()
    else:
        assert proc.returncode == 0, proc.stderr
        assert "banks.yaml" in (proc.stdout + proc.stderr)


def test_fixture_corpus_regression():
    pytest.importorskip("yaml")
    proc = subprocess.run(
        [sys.executable, "scripts/bank_adapters.py", "--test"],
        cwd=SKILL, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
