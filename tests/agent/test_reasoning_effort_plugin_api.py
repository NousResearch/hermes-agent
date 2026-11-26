"""The Gemini reasoning helpers published for provider plugins behave as the plugin guide documents.

``gemini_thinking_config``, ``clamp_effort`` and ``bare_gemini_model_id`` are listed in
``hermes_cli/data/plugin_api.json``: a provider plugin on the native Gemini wire maps its reasoning config
with them, so its efforts must follow the same per-model rules as the built-in ``gemini`` provider.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import providers
from agent.reasoning_effort import GEMINI4_THINKING_LEVELS, bare_gemini_model_id, clamp_effort, gemini_thinking_config

_ROOT = Path(__file__).resolve().parents[2]
_NATIVE = "https://generativelanguage.googleapis.com/v1beta"
_MODELS = ("gemini-4-neon", "gemini-4-argon", "gemini-3.6-flash", "gemini-3-pro-preview", "gemini-2.5-pro", "gemma-4-27b-it")
_CONFIGS = (
    None,
    {"enabled": False},
    {"enabled": True},
    {"enabled": True, "effort": "minimal"},
    {"enabled": True, "effort": "low"},
    {"enabled": True, "effort": "medium"},
    {"enabled": True, "effort": "high"},
    {"enabled": True, "effort": "xhigh"},
    {"enabled": True, "effort": "max"},
)


def _gemini_profile_thinking_config(model: str, reasoning_config: dict | None) -> dict | None:
    profile = providers.get_provider_profile("gemini")
    if profile is None:
        pytest.skip("google model-provider plugin not installed")
    body = profile.build_extra_body(model=model, reasoning_config=reasoning_config, base_url=_NATIVE)
    return body.get("thinking_config")


@pytest.mark.parametrize("model", _MODELS)
@pytest.mark.parametrize("reasoning_config", _CONFIGS)
def test_helper_matches_the_gemini_provider(model, reasoning_config):
    assert gemini_thinking_config(model, reasoning_config) == _gemini_profile_thinking_config(model, reasoning_config)


@pytest.mark.parametrize("prefix", ["gemini/", "google/", "GEMINI/", " google/"])
@pytest.mark.parametrize("model", ["gemini-4-neon", "gemini-4-argon", "gemini-2.5-pro"])
@pytest.mark.parametrize("reasoning_config", _CONFIGS)
def test_prefixed_id_maps_like_the_id_the_native_client_sends(prefix, model, reasoning_config):
    assert gemini_thinking_config(prefix + model, reasoning_config) == _gemini_profile_thinking_config(model, reasoning_config)


@pytest.mark.parametrize(
    ("model", "reasoning_config", "expected"),
    [
        ("gemini-4-neon", {"enabled": False}, {"includeThoughts": False, "thinkingLevel": "low"}),
        ("gemini-4-neon", {"enabled": True, "effort": "medium"}, {"includeThoughts": True, "thinkingLevel": "medium"}),
        ("gemini-4-neon", {"enabled": True, "effort": "xhigh"}, {"includeThoughts": True, "thinkingLevel": "xhigh"}),
        ("gemini-4-neon", {"enabled": True, "effort": "max"}, {"includeThoughts": True, "thinkingLevel": "xhigh"}),
        # Argon has no xhigh level: the top effort stops at high.
        ("gemini-4-argon", {"enabled": True, "effort": "xhigh"}, {"includeThoughts": True, "thinkingLevel": "high"}),
        ("gemini-4-neon", None, None),
        (None, {"enabled": True, "effort": "high"}, None),
        ("gemma-4-27b-it", {"enabled": True, "effort": "high"}, None),
    ],
)
def test_published_gemini_thinking_config(model, reasoning_config, expected):
    assert gemini_thinking_config(model, reasoning_config) == expected


def test_published_clamp_effort_takes_a_per_model_level_tuple():
    assert clamp_effort("xhigh", GEMINI4_THINKING_LEVELS) == "high"
    assert clamp_effort("minimal", GEMINI4_THINKING_LEVELS) == "low"
    assert clamp_effort("medium", GEMINI4_THINKING_LEVELS) == "medium"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini/gemini-4-neon", "gemini-4-neon"),
        ("google/gemini-4-neon", "gemini-4-neon"),
        ("GOOGLE/gemini-4-neon", "gemini-4-neon"),
        ("  gemini/gemini-4-neon  ", "gemini-4-neon"),
        ("gemini-4-neon", "gemini-4-neon"),
        # Only Gemini's own prefixes: the native client keeps these, so the helper does too.
        ("models/gemini-4-neon", "models/gemini-4-neon"),
        ("openrouter/gemini-4-neon", "openrouter/gemini-4-neon"),
        # A bare prefix keeps the original id rather than sending an empty one.
        ("gemini/", "gemini/"),
        ("", ""),
        (None, ""),
    ],
)
def test_bare_gemini_model_id(model, expected):
    assert bare_gemini_model_id(model) == expected


def test_model_family_keeps_bare_gemini_model_id_identity():
    from agent import gemini_model_family

    assert gemini_model_family.bare_gemini_model_id is bare_gemini_model_id


def test_importing_the_helpers_loads_no_http_stack():
    code = "import sys; import agent.reasoning_effort; print(sorted(m for m in ('httpx', 'openai') if m in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "[]"


def test_helpers_are_on_the_published_plugin_api():
    data = json.loads((_ROOT / "hermes_cli" / "data" / "plugin_api.json").read_text(encoding="utf-8"))
    published = {entry for group in data["groups"] for entry in group.get("entries", [])}

    for name in ("gemini_thinking_config", "clamp_effort", "bare_gemini_model_id"):
        assert f"agent.reasoning_effort:{name}" in published
