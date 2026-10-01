"""`hermes skills enable/disable` + `/skills enable|disable|disabled` — non-interactive toggles.

toggle_skills writes skills.disabled through save_disabled_skills (essential
skills stay enabled); the slash and CLI surfaces share it.
"""

import types

import pytest
import yaml


@pytest.fixture()
def skills_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("{}\n")
    from hermes_cli import skills_config
    monkeypatch.setattr(
        skills_config, "_list_all_skills",
        lambda: [{"name": "arxiv", "category": "research", "description": "x"},
                 {"name": "hermes-agent", "category": None, "description": "y"}])
    return home


def _disabled(home):
    cfg = yaml.safe_load((home / "config.yaml").read_text()) or {}
    return (cfg.get("skills") or {}).get("disabled")


def test_disable_enable_roundtrip_and_unknown(skills_home):
    from hermes_cli.skills_config import toggle_skills

    lines = toggle_skills("disable", ["arxiv", "nope"])
    assert _disabled(skills_home) == ["arxiv"]
    assert any("unknown skill: nope" in ln for ln in lines)
    assert any("Takes effect" in ln for ln in lines)

    toggle_skills("enable", ["arxiv"])
    assert _disabled(skills_home) == []


def test_essential_skill_cannot_be_disabled(skills_home):
    from hermes_cli.skills_config import toggle_skills

    lines = toggle_skills("disable", ["hermes-agent"])
    assert not _disabled(skills_home)
    assert any("essential" in ln for ln in lines)
