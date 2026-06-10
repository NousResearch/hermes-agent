"""Regression coverage for Hermes/OpenClaw skill home isolation."""

from pathlib import Path
from unittest.mock import patch


def _write_skill(root: Path, name: str, description: str = "Foreign skill") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return skill_dir


def test_openclaw_external_skills_dir_skipped_by_default(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "skills").mkdir()

    openclaw_skills = tmp_path / ".openclaw" / "skills"
    _write_skill(openclaw_skills, "openclaw-only")
    (hermes_home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {openclaw_skills}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_ALLOW_OPENCLAW_SKILLS", raising=False)

    from agent.skill_utils import get_external_skills_dirs

    assert get_external_skills_dirs() == []


def test_openclaw_external_skills_dir_requires_explicit_opt_in(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "skills").mkdir()

    openclaw_skills = tmp_path / ".openclaw" / "skills"
    _write_skill(openclaw_skills, "openclaw-only")
    (hermes_home / "config.yaml").write_text(
        "skills:\n"
        "  allow_openclaw_external_dirs: true\n"
        "  external_dirs:\n"
        f"    - {openclaw_skills}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    from agent.skill_utils import get_external_skills_dirs

    assert get_external_skills_dirs() == [openclaw_skills.resolve()]


def test_openclaw_local_skills_dir_opt_in_honors_config(tmp_path, monkeypatch):
    openclaw_home = tmp_path / ".openclaw"
    openclaw_skills = openclaw_home / "skills"
    _write_skill(openclaw_skills, "openclaw-only")
    (openclaw_home / "config.yaml").write_text(
        "skills:\n  allow_openclaw_external_dirs: true\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(openclaw_home))
    monkeypatch.delenv("HERMES_ALLOW_OPENCLAW_SKILLS", raising=False)

    from agent.skill_utils import get_all_skills_dirs

    assert get_all_skills_dirs() == [openclaw_skills]


def test_skills_list_does_not_scan_openclaw_home_when_hermes_home_is_mispointed(
    tmp_path,
    monkeypatch,
):
    openclaw_home = tmp_path / ".openclaw"
    openclaw_skills = openclaw_home / "skills"
    _write_skill(openclaw_skills, "openclaw-only")

    monkeypatch.setenv("HERMES_HOME", str(openclaw_home))
    monkeypatch.delenv("HERMES_ALLOW_OPENCLAW_SKILLS", raising=False)

    with patch("tools.skills_tool.SKILLS_DIR", openclaw_skills):
        from tools.skills_tool import _find_all_skills

        assert _find_all_skills() == []


def test_skill_view_rejects_absolute_openclaw_path_by_default(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "skills").mkdir()
    openclaw_skill = _write_skill(tmp_path / ".openclaw" / "skills", "openclaw-only")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_ALLOW_OPENCLAW_SKILLS", raising=False)

    with patch("tools.skills_tool.SKILLS_DIR", hermes_home / "skills"):
        from tools.skills_tool import skill_view
        import json

        result = json.loads(skill_view(str(openclaw_skill)))

    assert result["success"] is False
    assert "openclaw-only" not in result.get("available_skills", [])


def test_system_prompt_does_not_include_openclaw_skills_from_mispointed_home(
    tmp_path,
    monkeypatch,
):
    openclaw_home = tmp_path / ".openclaw"
    openclaw_skills = openclaw_home / "skills"
    _write_skill(openclaw_skills, "openclaw-only")

    monkeypatch.setenv("HERMES_HOME", str(openclaw_home))
    monkeypatch.delenv("HERMES_ALLOW_OPENCLAW_SKILLS", raising=False)

    from agent.prompt_builder import (
        build_skills_system_prompt,
        clear_skills_system_prompt_cache,
    )

    clear_skills_system_prompt_cache(clear_snapshot=True)
    try:
        assert "openclaw-only" not in build_skills_system_prompt()
    finally:
        clear_skills_system_prompt_cache(clear_snapshot=True)
