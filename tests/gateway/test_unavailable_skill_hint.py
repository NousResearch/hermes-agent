"""Tests for gateway.run._check_unavailable_skill.

Regression coverage for the dir-name-vs-frontmatter-name drift bug.
The hint function used to compare the skill's parent-directory name
against the typed command and the disabled list. That silently missed
every skill whose directory name differs from its declared frontmatter
name (~19 skills on a standard install), so users typing a real slug
like ``/stable-diffusion-image-generation`` got a generic "unknown
command" response instead of the intended "disabled — enable with …"
or "not installed — install with …" hint.

These tests pin the fixed behavior:

* Slug is derived from the frontmatter ``name:`` (exactly matching
  :func:`agent.skill_commands.scan_skill_commands`), so the slug differs
  from the directory name when the declared name is multi-word.
* ``disabled`` membership is checked by the declared name, because that
  is what :func:`hermes_cli.skills_config.save_disabled_skills` stores.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated skills dir + HERMES_HOME so the real user config is untouched."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home / "skills"


def _write_skill(skills_dir: Path, rel: str, frontmatter_name: str) -> Path:
    """Create a SKILL.md at ``<skills_dir>/<rel>/SKILL.md``."""
    skill_dir = skills_dir / rel
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {frontmatter_name}\ndescription: test skill\n---\nBody.\n",
        encoding="utf-8",
    )
    return skill_md


def test_frontmatter_slug_matched_even_when_dir_name_differs(
    tmp_skills: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directory ``stable-diffusion`` + frontmatter ``Stable Diffusion Image Generation``.

    Command typed: ``stable-diffusion-image-generation`` (the slug the
    agent actually registers). The old dir-name-based check would have
    compared ``stable-diffusion`` to the typed command and missed.
    """
    from gateway import run as gateway_run

    _write_skill(tmp_skills, "mlops/stable-diffusion", "Stable Diffusion Image Generation")

    # Config disables by declared name (matches what `hermes skills config` writes).
    monkeypatch.setattr(
        "gateway.run._get_disabled_skill_names",
        lambda: {"Stable Diffusion Image Generation"},
        raising=False,
    )
    with patch(
        "tools.skills_tool._get_disabled_skill_names",
        return_value={"Stable Diffusion Image Generation"},
    ), patch(
        "agent.skill_utils.get_all_skills_dirs",
        return_value=[tmp_skills],
    ):
        msg = gateway_run._check_unavailable_skill("stable-diffusion-image-generation")

    assert msg is not None, (
        "expected a 'disabled' hint for the frontmatter-derived slug; "
        "the old code compared the dir name 'stable-diffusion' and returned None"
    )
    assert "disabled" in msg.lower()
    assert "hermes skills config" in msg


def test_unknown_command_still_returns_none(
    tmp_skills: Path,
) -> None:
    """A command that matches no on-disk skill still returns None."""
    from gateway import run as gateway_run

    _write_skill(tmp_skills, "creative/ascii-art", "ascii-art")

    with patch(
        "tools.skills_tool._get_disabled_skill_names", return_value=set()
    ), patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[tmp_skills]
    ):
        assert gateway_run._check_unavailable_skill("no-such-skill") is None


def test_matched_but_not_disabled_returns_none(
    tmp_skills: Path,
) -> None:
    """A skill that exists and isn't disabled shouldn't produce a hint."""
    from gateway import run as gateway_run

    _write_skill(tmp_skills, "creative/ascii-art", "ascii-art")

    with patch(
        "tools.skills_tool._get_disabled_skill_names", return_value=set()
    ), patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[tmp_skills]
    ):
        assert gateway_run._check_unavailable_skill("ascii-art") is None


def test_slug_normalization_strips_non_alnum(
    tmp_skills: Path,
) -> None:
    """Frontmatter ``C++ Code Review`` → slug ``c-code-review`` (``+`` stripped)."""
    from gateway import run as gateway_run

    _write_skill(tmp_skills, "software-development/cpp-review", "C++ Code Review")

    with patch(
        "tools.skills_tool._get_disabled_skill_names",
        return_value={"C++ Code Review"},
    ), patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[tmp_skills]
    ):
        msg = gateway_run._check_unavailable_skill("c-code-review")

    assert msg is not None
    assert "disabled" in msg.lower()


def test_optional_skill_uses_frontmatter_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same drift bug applies to the optional-skills branch.

    Before: directory name was matched against the typed command, so an
    optional skill at ``optional-skills/mlops/stable-diffusion/SKILL.md``
    with frontmatter ``Stable Diffusion Image Generation`` returned None
    when the user typed the real slug.
    """
    from gateway import run as gateway_run

    # Build an isolated optional-skills dir
    optional = tmp_path / "optional-skills"
    skill_dir = optional / "mlops" / "stable-diffusion"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: Stable Diffusion Image Generation\ndescription: test\n---\n",
        encoding="utf-8",
    )

    # Point the optional lookup at our tmp dir. The source reads from
    # ``get_optional_skills_dir(repo_root / "optional-skills")`` — we
    # can't easily retarget ``repo_root``, so patch the resolver.
    monkeypatch.setattr(
        "hermes_constants.get_optional_skills_dir",
        lambda _default: optional,
        raising=False,
    )

    # Ensure the "disabled" branch doesn't match anything so we fall
    # through to the optional-skills branch.
    empty_skills = tmp_path / "empty-skills"
    empty_skills.mkdir()
    with patch(
        "tools.skills_tool._get_disabled_skill_names", return_value=set()
    ), patch(
        "agent.skill_utils.get_all_skills_dirs", return_value=[empty_skills]
    ):
        msg = gateway_run._check_unavailable_skill("stable-diffusion-image-generation")

    assert msg is not None, (
        "optional-skills branch should recognize the frontmatter-derived slug; "
        "the old dir-name-based check returned None here too"
    )
    assert "not installed" in msg.lower()
    assert "official/mlops/stable-diffusion" in msg
