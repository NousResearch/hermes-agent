"""Solstice maps reasoning through the published Gemini helpers, not core internals.

``agent.reasoning_effort.gemini_thinking_config`` is listed in ``hermes_cli/data/plugin_api.json`` for
providers on the native Gemini wire (its own behavior is covered in
tests/agent/test_reasoning_effort_plugin_api.py); the Solstice profile must reach the effort mapping
through it so the boundary lint holds without a baseline exception.
"""

from __future__ import annotations

from pathlib import Path

import providers
from agent.reasoning_effort import gemini_thinking_config
from hermes_cli.plugin_validate_private_imports import private_import_findings

_GOOGLE = Path(__file__).resolve().parents[4] / "plugins" / "model-providers" / "google"
_BASE = "https://generativelanguage.googleapis.com/v1alpha"


def test_google_plugin_imports_only_published_names():
    assert private_import_findings(_GOOGLE) == []


def test_solstice_profile_sends_what_the_published_helper_returns():
    profile = providers.get_provider_profile("solstice")
    config = {"enabled": True, "effort": "xhigh"}

    assert profile.build_extra_body(model="gemini-4-neon", reasoning_config=config, base_url=_BASE) == {
        "thinking_config": gemini_thinking_config("gemini-4-neon", config),
    }
