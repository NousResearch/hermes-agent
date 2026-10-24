"""skills.index_descriptions: the always-on skill index rendered with or without descriptions."""

import logging

import pytest

import agent.skill_utils as su
from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_skills_system_prompt_cache(clear_snapshot=True)
    su._warned_index_description_values.clear()
    yield
    clear_skills_system_prompt_cache(clear_snapshot=True)
    su._warned_index_description_values.clear()


def _skill(root, category, name, desc):
    d = root / category / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n", encoding="utf-8")


@pytest.mark.parametrize("setting", ["full", "names_only"])
def test_index_descriptions_knob(monkeypatch, tmp_path, setting):
    """names_only lists every category by name only, with a note not tied to the coding context."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"skills:\n  index_descriptions: {setting}\n", encoding="utf-8")
    _skill(tmp_path / "skills", "research", "arxiv", "Search arXiv papers")
    _skill(tmp_path / "skills", "devops", "deploy", "Ship the service")

    result = build_skills_system_prompt()

    assert "arxiv" in result and "deploy" in result
    assert "coding context" not in result
    if setting == "names_only":
        assert "Search arXiv papers" not in result and "Ship the service" not in result
        assert "research [names only]: arxiv" in result
        assert "devops [names only]: deploy" in result
        assert "skill_view(name) still loads the skill's full SKILL.md" in result
        assert "skills_list(category)" in result
    else:
        assert "Search arXiv papers" in result and "Ship the service" in result
        assert "[names only]" not in result


@pytest.mark.parametrize("config_text", ["skills: {}\n", ""])
def test_absent_setting_keeps_full_index_silently(monkeypatch, tmp_path, caplog, config_text):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")
    _skill(tmp_path / "skills", "research", "arxiv", "Search arXiv papers")

    with caplog.at_level(logging.WARNING, logger="agent.skill_utils"):
        result = build_skills_system_prompt()

    assert "Search arXiv papers" in result and "[names only]" not in result
    assert "index_descriptions" not in caplog.text


@pytest.mark.parametrize("bad", ["names-only", "names", "bogus"])
def test_invalid_value_warns_once_and_keeps_full_index(monkeypatch, tmp_path, caplog, bad):
    """A typo must not be silently ignored: name the bad value and the valid ones, once."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"skills:\n  index_descriptions: {bad}\n", encoding="utf-8")
    _skill(tmp_path / "skills", "research", "arxiv", "Search arXiv papers")

    with caplog.at_level(logging.WARNING, logger="agent.skill_utils"):
        result = build_skills_system_prompt()
        clear_skills_system_prompt_cache()
        build_skills_system_prompt()

    assert "Search arXiv papers" in result and "[names only]" not in result
    warnings = [r for r in caplog.records if "index_descriptions" in r.getMessage()]
    assert len(warnings) == 1
    assert bad in warnings[0].getMessage() and "'names_only'" in warnings[0].getMessage()


def test_index_descriptions_is_part_of_the_cache_key(monkeypatch, tmp_path):
    """Flipping the setting must not be served the other setting's cached index."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = tmp_path / "config.yaml"
    _skill(tmp_path / "skills", "research", "arxiv", "Search arXiv papers")

    config.write_text("skills:\n  index_descriptions: names_only\n", encoding="utf-8")
    assert "Search arXiv papers" not in build_skills_system_prompt()
    config.write_text("skills:\n  index_descriptions: full\n", encoding="utf-8")
    assert "Search arXiv papers" in build_skills_system_prompt()
    config.write_text("skills:\n  index_descriptions: names_only\n", encoding="utf-8")
    assert "Search arXiv papers" not in build_skills_system_prompt()


def test_trusted_project_skill_keeps_its_labelled_row(monkeypatch, tmp_path):
    """A project skill overrides same-named skills; names_only must not drop the [project] marker
    that tells the model which copy skill_view will load."""
    home = tmp_path / ".hermes"
    _skill(home / "skills", "devops", "deploy", "Ship the service")
    repo = tmp_path / "proj"
    (repo / ".git").mkdir(parents=True)
    _skill(repo / ".hermes" / "skills", "devops", "release", "Repo-specific release flow")
    (home / "config.yaml").write_text(
        f"skills:\n  index_descriptions: names_only\n  external_dirs: []\n  trusted_project_dirs: ['{repo}']\n",
        encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(repo)
    su._external_dirs_cache_clear()
    try:
        result = build_skills_system_prompt()
    finally:
        su._external_dirs_cache_clear()

    assert "devops [names only]: deploy" in result
    assert "    - release: [project] Repo-specific release flow" in result
    assert "Ship the service" not in result
