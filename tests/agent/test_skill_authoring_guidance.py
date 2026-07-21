"""Tests for the shared raw bundled authoring-guidance loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import skill_authoring_guidance as guidance
from agent.skill_utils import _raw_config_cache_clear
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)


def _write_package(
    bundled_root: Path,
    *,
    name: str = "hermes-agent-skill-authoring",
    version: str = "2.0.0",
    skill_marker: str = "BUNDLED-V2-SKILL",
    contract_marker: str | None = "BUNDLED-V2-CONTRACT",
) -> Path:
    skill_dir = (
        bundled_root
        / "software-development"
        / "hermes-agent-skill-authoring"
    )
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            (
                "---",
                f"name: {name}",
                f"version: {version}",
                "description: Test authoring guidance.",
                "---",
                "",
                "# Hermes Agent Skill Authoring Skill",
                "",
                skill_marker,
                "",
            )
        ),
        encoding="utf-8",
    )
    if contract_marker is not None:
        contract = skill_dir / "references" / "authoring-contract.md"
        contract.parent.mkdir()
        contract.write_text(
            f"# Authoring Contract\n\n{contract_marker}\n",
            encoding="utf-8",
        )
    return skill_dir


def _configure_real_paths(
    monkeypatch,
    tmp_path: Path,
    bundled_root: Path,
) -> Path:
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(bundled_root))
    _raw_config_cache_clear()
    return hermes_home


def test_real_temp_hermes_home_loads_exact_raw_skill_and_contract(
    monkeypatch, tmp_path
):
    bundled_root = tmp_path / "bundled"
    raw_skill = "!`touch must-not-run` BUNDLED-V2-SKILL"
    _write_package(bundled_root, skill_marker=raw_skill)
    hermes_home = _configure_real_paths(
        monkeypatch,
        tmp_path,
        bundled_root,
    )
    # A user-local same-name skill must never shadow the exact bundled source.
    _write_package(
        hermes_home / "skills",
        skill_marker="USER-SHADOW-SKILL",
        contract_marker="USER-SHADOW-CONTRACT",
    )

    def _preprocess_must_not_run(*args, **kwargs):
        raise AssertionError("shared loader must return raw Markdown")

    monkeypatch.setattr(
        "agent.skill_preprocessing.preprocess_skill_content",
        _preprocess_must_not_run,
    )
    token = set_hermes_home_override(None)
    try:
        loaded = guidance.load_bundled_skill_authoring_guidance(
            platform="test"
        )
    finally:
        reset_hermes_home_override(token)

    assert loaded is not None
    assert raw_skill in loaded.skill_content
    assert "BUNDLED-V2-CONTRACT" in (loaded.contract_content or "")
    assert "USER-SHADOW" not in loaded.skill_content
    assert "USER-SHADOW" not in (loaded.contract_content or "")


@pytest.mark.parametrize(
    ("name", "version"),
    (
        ("wrong-name", "2.0.0"),
        ("hermes-agent-skill-authoring", "1.9.0"),
        ("hermes-agent-skill-authoring", ""),
    ),
)
def test_rejects_wrong_name_or_non_v2(
    monkeypatch, tmp_path, name, version
):
    bundled_root = tmp_path / f"bundled-{name}-{version or 'none'}"
    _write_package(bundled_root, name=name, version=version)
    _configure_real_paths(monkeypatch, tmp_path, bundled_root)

    assert (
        guidance.load_bundled_skill_authoring_guidance(platform="test")
        is None
    )


def test_disabled_or_opted_out_profile_returns_none(monkeypatch, tmp_path):
    bundled_root = tmp_path / "bundled"
    _write_package(bundled_root)
    hermes_home = _configure_real_paths(
        monkeypatch,
        tmp_path,
        bundled_root,
    )
    (hermes_home / "config.yaml").write_text(
        "skills:\n"
        "  disabled:\n"
        "    - hermes-agent-skill-authoring\n",
        encoding="utf-8",
    )
    _raw_config_cache_clear()
    assert (
        guidance.load_bundled_skill_authoring_guidance(platform="test")
        is None
    )

    (hermes_home / "config.yaml").write_text("skills: {}\n", encoding="utf-8")
    (hermes_home / ".no-bundled-skills").write_text("", encoding="utf-8")
    _raw_config_cache_clear()
    assert (
        guidance.load_bundled_skill_authoring_guidance(platform="test")
        is None
    )


def test_missing_contract_preserves_valid_skill(monkeypatch, tmp_path):
    bundled_root = tmp_path / "bundled"
    _write_package(bundled_root, contract_marker=None)
    _configure_real_paths(monkeypatch, tmp_path, bundled_root)

    loaded = guidance.load_bundled_skill_authoring_guidance(platform="test")

    assert loaded is not None
    assert "BUNDLED-V2-SKILL" in loaded.skill_content
    assert loaded.contract_content is None


def test_contract_path_escape_preserves_skill_with_missing_contract(
    monkeypatch, tmp_path
):
    bundled_root = tmp_path / "bundled"
    skill_dir = _write_package(bundled_root, contract_marker=None)
    outside = tmp_path / "outside-contract.md"
    outside.write_text("OUTSIDE-CONTRACT\n", encoding="utf-8")
    references = skill_dir / "references"
    references.mkdir()
    try:
        (references / "authoring-contract.md").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    _configure_real_paths(monkeypatch, tmp_path, bundled_root)

    loaded = guidance.load_bundled_skill_authoring_guidance(platform="test")

    assert loaded is not None
    assert loaded.contract_content is None
    assert "OUTSIDE-CONTRACT" not in loaded.skill_content


def test_skill_directory_path_escape_is_rejected(monkeypatch, tmp_path):
    bundled_root = tmp_path / "bundled"
    (bundled_root / "software-development").mkdir(parents=True)
    outside_root = tmp_path / "outside"
    outside_skill = _write_package(outside_root)
    expected = (
        bundled_root
        / "software-development"
        / "hermes-agent-skill-authoring"
    )
    try:
        expected.symlink_to(outside_skill, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    _configure_real_paths(monkeypatch, tmp_path, bundled_root)

    assert (
        guidance.load_bundled_skill_authoring_guidance(platform="test")
        is None
    )


def test_size_limits_reject_skill_but_only_drop_contract(
    monkeypatch, tmp_path
):
    bundled_root = tmp_path / "bundled"
    skill_dir = _write_package(
        bundled_root,
        contract_marker="C" * 2000,
    )
    _configure_real_paths(monkeypatch, tmp_path, bundled_root)
    skill_size = (skill_dir / "SKILL.md").stat().st_size
    monkeypatch.setattr(guidance, "_MAX_GUIDANCE_BYTES", skill_size + 10)

    loaded = guidance.load_bundled_skill_authoring_guidance(platform="test")
    assert loaded is not None
    assert loaded.contract_content is None

    monkeypatch.setattr(guidance, "_MAX_GUIDANCE_BYTES", skill_size - 1)
    assert (
        guidance.load_bundled_skill_authoring_guidance(platform="test")
        is None
    )

    skill_chars = len(
        (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    )
    monkeypatch.setattr(guidance, "_MAX_GUIDANCE_BYTES", 400_000)
    monkeypatch.setattr(guidance, "_MAX_GUIDANCE_CHARS", skill_chars - 1)
    assert (
        guidance.load_bundled_skill_authoring_guidance(platform="test")
        is None
    )
