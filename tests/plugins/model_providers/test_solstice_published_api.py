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

_SOLSTICE = Path(__file__).resolve().parents[3] / "plugins" / "model-providers" / "solstice"
_BASE = "https://generativelanguage.googleapis.com/v1alpha"


def test_solstice_profile_imports_only_published_names():
    hits = [hit for hit in private_import_findings(_SOLSTICE) if hit.startswith("__init__.py:")]

    assert hits == []


def test_solstice_transport_reaches_the_gemini_stream_helpers_through_discovery():
    """The SSE iterator, the bounded error-body read and the model-id rule are not imported from core
    internals: the first two come from the discovered gemini plugin, the last is published. Only the base
    client and its error type remain (both needed before, and without, a gemini plugin)."""
    hits = [hit for hit in private_import_findings(_SOLSTICE) if hit.startswith("transport.py:")]

    assert sorted(hit.split(" ", 1)[-1] for hit in hits) == [
        "agent.gemini_native_adapter.GeminiAPIError", "agent.gemini_native_adapter.GeminiNativeClient",
    ], hits


def test_solstice_profile_sends_what_the_published_helper_returns():
    profile = providers.get_provider_profile("solstice")
    config = {"enabled": True, "effort": "xhigh"}

    assert profile.build_extra_body(model="gemini-4-neon", reasoning_config=config, base_url=_BASE) == {
        "thinking_config": gemini_thinking_config("gemini-4-neon", config),
    }
