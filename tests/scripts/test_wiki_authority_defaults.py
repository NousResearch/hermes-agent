"""Canonical wiki authority path-resolution tests.

The Git-backed wiki under ~/docs/wiki is the default authority. WIKI_DIR and
KENSEI_WIKI_ROOT remain supported for disposable tests and migrations.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"


def _load(script_name: str):
    module_name = f"wiki_authority_{script_name.replace('-', '_').replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / script_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def clean_wiki_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("WIKI_DIR", raising=False)
    monkeypatch.delenv("WIKI_PATH", raising=False)
    monkeypatch.delenv("KENSEI_WIKI_ROOT", raising=False)
    return tmp_path / "docs" / "wiki"


def test_wiki_on_complete_defaults_to_canonical_wiki(clean_wiki_env):
    module = _load("wiki_on_complete.py")
    assert module.WIKI_REPOS == clean_wiki_env / "repos"


def test_wiki_on_complete_wiki_dir_override_wins(clean_wiki_env, monkeypatch):
    override = clean_wiki_env.parent.parent / "override-wiki"
    monkeypatch.setenv("WIKI_DIR", str(override))
    module = _load("wiki_on_complete.py")
    assert module.WIKI_REPOS == override / "repos"


def test_approval_handler_defaults_to_canonical_wiki(clean_wiki_env):
    module = _load("approval_handler.py")
    assert module.WIKI_REPOS == clean_wiki_env / "repos"


def test_approval_handler_wiki_dir_override_wins(clean_wiki_env, monkeypatch):
    override = clean_wiki_env.parent.parent / "override-wiki"
    monkeypatch.setenv("WIKI_DIR", str(override))
    module = _load("approval_handler.py")
    assert module.WIKI_REPOS == override / "repos"


def test_brain_synthesis_defaults_to_canonical_wiki(clean_wiki_env):
    module = _load("brain-to-wiki-synthesis.py")
    assert module.WIKI_DIR == clean_wiki_env


def test_brain_synthesis_wiki_dir_override_wins(clean_wiki_env, monkeypatch):
    override = clean_wiki_env.parent.parent / "override-wiki"
    monkeypatch.setenv("WIKI_DIR", str(override))
    module = _load("brain-to-wiki-synthesis.py")
    assert module.WIKI_DIR == override


def test_daily_review_defaults_to_canonical_mashups(clean_wiki_env):
    module = _load("kensei_review_daily.py")
    assert module.MASHUPS_FILE == clean_wiki_env / "_meta" / "paper-mashups.md"


def test_daily_review_kensei_wiki_root_override_wins(clean_wiki_env, monkeypatch):
    override = clean_wiki_env.parent.parent / "override-wiki"
    monkeypatch.setenv("KENSEI_WIKI_ROOT", str(override))
    module = _load("kensei_review_daily.py")
    assert module.MASHUPS_FILE == override / "_meta" / "paper-mashups.md"


def test_research_preprocess_defaults_to_canonical_wiki(clean_wiki_env):
    module = _load("research_paper_preprocess.py")
    assert module.WIKI_PATH == clean_wiki_env


def test_research_preprocess_wiki_path_override_wins(clean_wiki_env, monkeypatch):
    override = clean_wiki_env.parent.parent / "override-wiki"
    monkeypatch.setenv("WIKI_PATH", str(override))
    module = _load("research_paper_preprocess.py")
    assert module.WIKI_PATH == override


def test_defaults_are_home_relative_not_hardcoded(clean_wiki_env, tmp_path, monkeypatch):
    """Defaults must derive from HOME so a second disposable HOME resolves
    independently — guards against a hard-coded /home/kensei path."""
    alt_home = tmp_path / "alt-home"
    alt_home.mkdir()
    monkeypatch.setenv("HOME", str(alt_home))
    for name, attr, tail in [
        ("wiki_on_complete.py", "WIKI_REPOS", "repos"),
        ("approval_handler.py", "WIKI_REPOS", "repos"),
    ]:
        module = _load(name)
        assert getattr(module, attr) == alt_home / "docs" / "wiki" / tail
    module = _load("brain-to-wiki-synthesis.py")
    assert module.WIKI_DIR == alt_home / "docs" / "wiki"
    module = _load("kensei_review_daily.py")
    assert module.MASHUPS_FILE == alt_home / "docs" / "wiki" / "_meta" / "paper-mashups.md"


@pytest.mark.parametrize(
    "relative_path",
    [
        "skills/research/llm-wiki/SKILL.md",
        "skills/research/research-paper-synthesis/SKILL.md",
    ],
)
def test_bundled_wiki_skills_name_only_the_canonical_default(relative_path):
    content = (REPO / relative_path).read_text(encoding="utf-8")
    assert "~/docs/wiki" in content or "$HOME/docs/wiki" in content
    assert "~/wiki" not in content
    assert "$HOME/wiki" not in content
