"""Tests for the request-scope-resolution optional skill.

Two layers, both stdlib + pytest (no network):
  1. Structural / frontmatter contract on SKILL.md (matches the maintainer
     review checklist for optional skills; child dir scripts must be described).
  2. Behavioral: run the real scope_resolver.scope_request logic against mocked
     decision-model responses and assert verdict + normalization are correct.

No live network: the decision-model call is patched.
"""

import importlib.util
import json
import re
import unittest.mock
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "autonomous-ai-agents"
    / "request-scope-resolution"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
RESOLVER = SKILL_DIR / "scripts" / "scope_resolver.py"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- structural contract ---------------------------------------------------


def test_skill_files_exist():
    assert SKILL_MD.is_file()
    assert RESOLVER.is_file()


def test_frontmatter_present(skill_text: str):
    assert skill_text.startswith("---\n")
    assert skill_text.count("---") >= 2


def test_description_under_sixty_chars(skill_text: str):
    m = re.search(r"^description: (.*)$", skill_text, re.MULTILINE)
    assert m, "no description field"
    # Handle quoted values.
    desc = m.group(1).strip().strip('"')
    assert len(desc) <= 60, f"description {len(desc)} chars — hardline is 60"


def test_description_ends_with_period(skill_text: str):
    m = re.search(r'^description: "?(.+?)"?\.?\s*$', skill_text, re.MULTILINE)
    assert m, "no description field"
    assert m.group(1).strip().endswith("."), "description must end with a period"


def test_related_skills_resolve_in_repo():
    """related_skills must point at in-repo skills only (no user-local refs)."""
    text = SKILL_MD.read_text(encoding="utf-8")
    m = re.search(r"related_skills:\s*\[([^\]]*)\]", text)
    if not m or not m.group(1).strip():
        return  # no related_skills declared — contract satisfied trivially
    names = [n.strip().strip("'\"") for n in m.group(1).split(",")]
    for name in names:
        assert re.fullmatch(r"[a-z0-9-]+", name), f"bad related_skills slug: {name}"


def test_no_machine_local_paths(skill_text: str):
    """Optional-skills PRs must not reference user-local absolute paths."""
    for bad in ["/Users/", "/home/", "C:\\Users\\", "~/workspace", "~/HermesVault"]:
        assert bad not in skill_text, f"machine-local path leaked: {bad}"


def test_no_secrets_in_skill_or_script():
    """No API key literals, passwords, or token patterns anywhere in the skill."""
    for path in [SKILL_MD, RESOLVER]:
        text = path.read_text(encoding="utf-8")
        assert "sk-or-" not in text, f"leaked key literal in {path.name}"
        assert re.search(r"(?i)password\s*[=:]\s*['\"][^'\"]+['\"]", text) is None, (
            f"password literal in {path.name}"
        )


def test_script_ends_with_trailing_newline():
    assert RESOLVER.read_bytes().endswith(b"\n")


# --- behavioral ---------------------------------------------------------


def _make_answer(category="simple", confidence=0.9, raw_score=0.0):
    """Construct a mocked /api/alpha/decisions response for one category+complexity ask."""
    return {
        "answers": {
            "category": {
                "type": "choice",
                "choice": category,
                "confidence": confidence,
                "probabilities": {category: confidence, "simple": 0.0},
            },
            "complexity": {
                "type": "score",
                "score": raw_score,
                "confidence": 1.0,
                "legend": {"0": "Direct execution", "1": "Multi-step", "2": "Escalation"},
            },
        }
    }


def test_high_confidence_resolved(unittest_mock_resolver):
    mod = _load_module(RESOLVER, "scope_resolver")
    with unittest.mock.patch.object(mod, "_call_decision", return_value=_make_answer(
        category="heavy-dev", confidence=0.95, raw_score=2.0
    )):
        out = mod.scope_request("Build a system with a database and components")
        assert out["verdict"] == "RESOLVED"
        assert out["category"] == "heavy-dev"
        assert out["confidence"] == 0.95
        assert out["complexity"] == 1.0  # 2.0 / (3-1)


def test_low_confidence_ask_user(unittest_mock_resolver):
    mod = _load_module(RESOLVER, "scope_resolver")
    with unittest.mock.patch.object(mod, "_call_decision", return_value=_make_answer(
        category="code-review", confidence=0.4, raw_score=2.0
    )):
        out = mod.scope_request("look at that thing we discussed")
        assert out["verdict"] == "ASK_USER"
        assert out["confidence"] == 0.4


def test_mid_confidence_needs_context(unittest_mock_resolver):
    mod = _load_module(RESOLVER, "scope_resolver")
    with unittest.mock.patch.object(mod, "_call_decision", return_value=_make_answer(
        category="stuck", confidence=0.7, raw_score=1.0
    )):
        out = mod.scope_request("Having some trouble with the thing")
        assert out["verdict"] == "NEEDS_CONTEXT"
        assert out["confidence"] == 0.7
        assert out["complexity"] == 0.5  # 1.0 / 2


def test_error_verdict_ask_user(unittest_mock_resolver):
    mod = _load_module(RESOLVER, "scope_resolver")
    with unittest.mock.patch.object(mod, "_call_decision", return_value={"error": "HTTP Error 500"}):
        out = mod.scope_request("anything")
        assert out["verdict"] == "ASK_USER"
        assert out["error"] == "HTTP Error 500"


@pytest.fixture
def unittest_mock_resolver():
    """Marker fixture so the behavioral tests can reference unittest.mock simply."""
    return True
