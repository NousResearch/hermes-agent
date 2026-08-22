"""Behavior-contract tests for the bundled Obsidian skill."""

from pathlib import Path

import pytest


SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "note-taking"
    / "obsidian"
    / "SKILL.md"
)


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_explicit_target_precedes_configured_and_discovered_vaults(skill_text: str):
    explicit = skill_text.index("An explicit absolute path, vault name, or vault ID always wins")
    configured = skill_text.index("With no explicit target, use `OBSIDIAN_VAULT_PATH`")
    discovered = skill_text.index("Otherwise, use `obsidian vaults verbose`")

    assert explicit < configured < discovered


def test_single_known_vault_is_selected(skill_text: str):
    assert "Use the only known vault" in skill_text


def test_multiple_known_vaults_require_owner_selection(skill_text: str):
    assert "if multiple vaults are known, ask which one" in skill_text


def test_no_known_vault_uses_only_an_existing_fallback(skill_text: str):
    assert (
        "If none are known, use `~/Documents/Obsidian Vault` when it exists; "
        "otherwise ask for the absolute path."
    ) in skill_text


def test_ambiguous_name_or_id_requires_an_absolute_path(skill_text: str):
    assert "if the CLI is unavailable or the match is ambiguous, ask for the absolute path" in skill_text


def test_active_vault_is_not_an_implicit_target(skill_text: str):
    assert "Do not infer the target from Obsidian's currently active vault" in skill_text


def test_validation_failure_does_not_silently_fall_back(skill_text: str):
    assert "report the path and failed check instead of silently falling back" in skill_text


def test_file_tools_receive_a_concrete_absolute_path(skill_text: str):
    assert "Resolve the selected vault to a concrete absolute path before using" in skill_text
