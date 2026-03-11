"""Tests for hermes_cli.doctor."""

import os
import sys
import types
from argparse import Namespace

import pytest

from hermes_cli import doctor as doctor_mod
from hermes_cli.doctor import _has_provider_env_config


class TestHasProviderEnvConfig:
    def test_detects_openrouter_key(self):
        content = "OPENROUTER_API_KEY=sk-or-abc123\n"
        assert _has_provider_env_config(content)

    def test_detects_custom_endpoint_without_openrouter_key(self):
        content = "OPENAI_API_BASE=http://localhost:11434/v1\nOPENAI_API_KEY=ollama\n"
        assert _has_provider_env_config(content)

    def test_returns_false_when_no_provider_settings(self):
        content = "TERMINAL_ENV=local\n"
        assert not _has_provider_env_config(content)


def _make_fake_model_tools(seen):
    """Return a fake model_tools module that records HERMES_INTERACTIVE at call time."""
    def fake_check_tool_availability(*args, **kwargs):
        seen["interactive"] = os.getenv("HERMES_INTERACTIVE")
        raise SystemExit(0)

    return types.SimpleNamespace(
        check_tool_availability=fake_check_tool_availability,
        TOOLSET_REQUIREMENTS={},
    )


def test_run_doctor_sets_interactive_env_for_tool_checks(monkeypatch, tmp_path):
    """Doctor must set HERMES_INTERACTIVE so CLI-gated tools report correctly."""
    project_root = tmp_path / "project"
    hermes_home = tmp_path / ".hermes"
    project_root.mkdir()
    hermes_home.mkdir()

    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    seen = {}
    monkeypatch.setitem(sys.modules, "model_tools", _make_fake_model_tools(seen))

    with pytest.raises(SystemExit):
        doctor_mod.run_doctor(Namespace(fix=False))

    assert seen.get("interactive") == "1", (
        "HERMES_INTERACTIVE must be '1' when check_tool_availability() is called"
    )


def test_run_doctor_respects_existing_interactive_env(monkeypatch, tmp_path):
    """Doctor must not override HERMES_INTERACTIVE if already set by caller."""
    project_root = tmp_path / "project"
    hermes_home = tmp_path / ".hermes"
    project_root.mkdir()
    hermes_home.mkdir()

    monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(doctor_mod, "HERMES_HOME", hermes_home)
    monkeypatch.setenv("HERMES_INTERACTIVE", "already-set")

    seen = {}
    monkeypatch.setitem(sys.modules, "model_tools", _make_fake_model_tools(seen))

    with pytest.raises(SystemExit):
        doctor_mod.run_doctor(Namespace(fix=False))

    assert seen.get("interactive") == "already-set", (
        "setdefault must not override an existing HERMES_INTERACTIVE value"
    )
