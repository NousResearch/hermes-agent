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


# ── Env var override ──────────────────────────────────────────────────────


def test_env_var_allow_openclaw_overrides_config(tmp_path, monkeypatch):
    """HERMES_ALLOW_OPENCLAW_SKILLS=1 allows OpenClaw skills via env var alone."""
    openclaw_home = tmp_path / ".openclaw"
    openclaw_skills = openclaw_home / "skills"
    _write_skill(openclaw_skills, "openclaw-only")

    monkeypatch.setenv("HERMES_HOME", str(openclaw_home))
    monkeypatch.setenv("HERMES_ALLOW_OPENCLAW_SKILLS", "1")

    from agent.skill_utils import get_local_skills_dir, _openclaw_allowed

    assert _openclaw_allowed() is True
    assert get_local_skills_dir() == openclaw_skills


# ── Mixed scenario: skipped local + valid external ───────────────────────


def test_external_skills_survive_when_local_dir_is_skipped(tmp_path, monkeypatch):
    """When HERMES_HOME points to OpenClaw, external non-OpenClaw dirs still work."""
    openclaw_home = tmp_path / ".openclaw"
    openclaw_skills = openclaw_home / "skills"
    _write_skill(openclaw_skills, "openclaw-only")

    # A valid external dir that is NOT OpenClaw-owned.
    external_dir = tmp_path / "external-skills"
    _write_skill(external_dir, "legit-skill", "A legitimate external skill")

    (openclaw_home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external_dir}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(openclaw_home))
    monkeypatch.delenv("HERMES_ALLOW_OPENCLAW_SKILLS", raising=False)

    from agent.skill_utils import get_external_skills_dirs

    dirs = get_external_skills_dirs()
    assert external_dir.resolve() in dirs
    # OpenClaw-owned external dirs should still be excluded.
    assert openclaw_skills not in dirs


def test_skills_list_includes_external_when_local_skipped(tmp_path, monkeypatch):
    """BLOCKER regression: skills_list must not drop external skills when
    the local skills directory is skipped (OpenClaw-owned)."""
    openclaw_home = tmp_path / ".openclaw"
    openclaw_skills = openclaw_home / "skills"
    _write_skill(openclaw_skills, "openclaw-only")

    external_dir = tmp_path / "external-skills"
    _write_skill(external_dir, "ext-skill", "An external skill")

    (openclaw_home / "config.yaml").write_text(
        f"skills:\n  external_dirs:\n    - {external_dir}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(openclaw_home))
    monkeypatch.delenv("HERMES_ALLOW_OPENCLAW_SKILLS", raising=False)

    with patch("tools.skills_tool.SKILLS_DIR", openclaw_skills):
        from tools.skills_tool import _find_all_skills

        skills = _find_all_skills()
        names = {s["name"] for s in skills}

        assert "openclaw-only" not in names, "OpenClaw skill leaked through"
        assert "ext-skill" in names, "External skill dropped when local was skipped"


# ── Marker depth behaviour ────────────────────────────────────────────────


def test_is_openclaw_owned_detects_direct_marker(tmp_path):
    """Directory with openclaw.json directly inside is detected."""
    target = tmp_path / "nested" / "dir"
    target.mkdir(parents=True)
    (target / "openclaw.json").write_text("{}")

    from agent.skill_utils import _is_openclaw_owned

    assert _is_openclaw_owned(target) is True


def test_is_openclaw_owned_detects_marker_in_parent(tmp_path, monkeypatch):
    """Marker in a parent directory (within home) is detected."""
    marker_dir = tmp_path / "project"
    marker_dir.mkdir()
    (marker_dir / "clawdbot.json").write_text("{}")

    target = marker_dir / "skills" / "nested"
    target.mkdir(parents=True)

    from agent.skill_utils import _is_openclaw_owned

    assert _is_openclaw_owned(target) is True


def test_is_openclaw_owned_ignores_marker_past_home(tmp_path, monkeypatch):
    """Marker in a parent past Path.home() is ignored (bounded walk).

    The marker sits *above* the home boundary; the target sits *below* it.
    Without the bound the walk would reach the marker and return True;
    with the bound it stops at home and returns False.
    """
    # home = fake_home/sub — the boundary where the walk must stop.
    # marker at fake_home/openclaw.json — one level ABOVE the boundary.
    # target at fake_home/sub/deep/skills — two levels BELOW the boundary.
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    (fake_home / "openclaw.json").write_text("{}")  # marker above boundary

    home_boundary = fake_home / "sub"
    home_boundary.mkdir()
    target = home_boundary / "deep" / "skills"
    target.mkdir(parents=True)

    monkeypatch.setattr("agent.skill_utils.Path.home", lambda: home_boundary)

    from agent.skill_utils import _is_openclaw_owned

    # Walk: target → fake-home/sub/deep → fake-home/sub (== home, break).
    # It never inspects fake-home/, so the marker is invisible.
    assert _is_openclaw_owned(target) is False


def test_is_openclaw_owned_clean_dir_not_flagged(tmp_path):
    """A directory without any OpenClaw markers is not flagged."""
    clean = tmp_path / "clean-dir"
    clean.mkdir()

    from agent.skill_utils import _is_openclaw_owned

    assert _is_openclaw_owned(clean) is False
