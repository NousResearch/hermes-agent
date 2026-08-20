"""Tests for ``hermes_cli.vertical_agent_kit``.

These tests are hermetic: they use a temporary directory for generated
scaffolds and do not touch the operator's real ``~/.hermes``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.vertical_agent_kit import (
    blueprint_path,
    list_blueprints,
    render_blueprint,
    smoke_scaffold,
    verify_scaffold,
)


@pytest.fixture
def temp_out(tmp_path: Path) -> Path:
    return tmp_path / "out"


def test_list_blueprints_is_sorted():
    names = list_blueprints()
    assert "support" in names
    assert "research" in names
    assert names == sorted(names)


def test_blueprint_path_returns_existing():
    assert blueprint_path("support") is not None
    assert blueprint_path("support").name == "support"


def test_blueprint_path_unknown():
    assert blueprint_path("does-not-exist") is None


def test_render_blueprint_writes_expected_files(temp_out: Path):
    variables = {
        "PROFILE_NAME": "test-agent",
        "ROLE": "Test Specialist",
        "OBJECTIVE": "Test things",
        "USERS": "qa team",
        "TONE": "neutral",
        "SCOPE": "testing only",
        "REFUSALS": "redirect",
        "SOURCES": "docs",
        "SYSTEMS": "ci",
        "DECISION_STYLE": "careful",
    }
    written = render_blueprint("support", temp_out, variables)

    assert all(isinstance(p, Path) for p in written)
    assert (temp_out / "test-agent" / "SOUL.md").exists()
    assert (temp_out / "test-agent" / "USER.template.md").exists()
    assert (temp_out / "test-agent" / "OPERATIONS.md").exists()

    soul = (temp_out / "test-agent" / "SOUL.md").read_text()
    assert "product support specialist" in soul


def test_render_blueprint_overwrite_flag(temp_out: Path):
    variables = {
        "PROFILE_NAME": "dup-agent",
        "ROLE": "Role A",
        "OBJECTIVE": "Objective A",
        "USERS": "team",
        "TONE": "calm",
        "SCOPE": "scope",
        "REFUSALS": "refuse",
        "SOURCES": "docs",
        "SYSTEMS": "none",
        "DECISION_STYLE": "fast",
    }
    render_blueprint("support", temp_out, variables)

    # Without overwrite, second render must raise
    with pytest.raises(FileExistsError):
        render_blueprint("support", temp_out, variables)

    variables["ROLE"] = "Role B"
    render_blueprint("support", temp_out, variables, overwrite=True)


def test_render_blueprint_force_refuses_non_scaffold(temp_out: Path):
    variables = {
        "PROFILE_NAME": "existing-dir",
        "ROLE": "Role A",
        "OBJECTIVE": "Objective A",
        "USERS": "team",
        "TONE": "calm",
        "SCOPE": "scope",
        "REFUSALS": "refuse",
        "SOURCES": "docs",
        "SYSTEMS": "none",
        "DECISION_STYLE": "fast",
    }
    # Pre-create a directory that does not look like a scaffold.
    (temp_out / "existing-dir").mkdir(parents=True)
    (temp_out / "existing-dir" / "my-data.txt").write_text("important")

    with pytest.raises(FileExistsError) as exc_info:
        render_blueprint("support", temp_out, variables, overwrite=True)
    assert "does not look like a vertical-agent scaffold" in str(exc_info.value)

    # The pre-existing file must survive.
    assert (temp_out / "existing-dir" / "my-data.txt").exists()


def test_verify_scaffold_passes_for_rendered(temp_out: Path):
    variables = {
        "PROFILE_NAME": "verify-agent",
        "ROLE": "r",
        "OBJECTIVE": "o",
        "USERS": "u",
        "TONE": "t",
        "SCOPE": "s",
        "REFUSALS": "x",
        "SOURCES": "d",
        "SYSTEMS": "sys",
        "DECISION_STYLE": "cautious",
    }
    render_blueprint("research", temp_out, variables)
    errors = verify_scaffold(temp_out / "verify-agent")
    assert errors == []


def test_verify_scaffold_missing_files(temp_out: Path):
    errors = verify_scaffold(temp_out / "missing")
    assert any("does not exist" in e for e in errors)

    (temp_out / "empty-dir").mkdir(parents=True)
    errors = verify_scaffold(temp_out / "empty-dir")
    assert any("Missing SOUL.md" in e for e in errors)


def test_smoke_scaffold_passes_for_rendered(temp_out: Path, monkeypatch):
    variables = {
        "PROFILE_NAME": "smoke-agent",
        "ROLE": "r",
        "OBJECTIVE": "o",
        "USERS": "u",
        "TONE": "t",
        "SCOPE": "s",
        "REFUSALS": "x",
        "SOURCES": "d",
        "SYSTEMS": "sys",
        "DECISION_STYLE": "cautious",
    }
    render_blueprint("support", temp_out, variables)

    # Simulate a host where hermes is not on PATH.
    monkeypatch.setattr("shutil.which", lambda _cmd: None)

    errors, warnings = smoke_scaffold(temp_out / "smoke-agent")
    assert errors == []
    assert any("Hermes CLI not found on PATH" in w for w in warnings)

