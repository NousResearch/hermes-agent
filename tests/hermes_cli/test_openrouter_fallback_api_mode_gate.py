"""A persisted ``model.api_mode`` belongs to the provider it was recorded for.

``_resolve_openrouter_runtime`` (OpenRouter, and the bare/aliased ``custom`` fallback) used to honor
``model.api_mode`` whatever provider the config named. A profile whose default is direct Anthropic
(``provider: anthropic`` + ``api_mode: anthropic_messages``) then drove an OpenRouter pick over
OpenRouter's Anthropic-compatible Messages endpoint: turn 1 worked, every later turn 400'd on the
replayed thinking block, which carries no ``signature``. Every other resolver path already gates the
mode through ``_configured_api_mode`` ("a stale mode never leaks across a switch").
"""

import pytest

from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_constants import get_hermes_home

ANTHROPIC_PROFILE = """\
model:
  default: claude-sonnet-5
  provider: anthropic
  base_url: https://api.anthropic.com
  api_mode: anthropic_messages
"""

OPENROUTER_URL = "https://openrouter.ai/api/v1"
LOCAL_URL = "http://localhost:8080/v1"


def _write_config(text):
    (get_hermes_home() / "config.yaml").write_text(text, encoding="utf-8")


def _model_block(provider, base_url=""):
    lines = ["model:", "  default: some/model"]
    if provider:
        lines.append(f"  provider: {provider}")
    if base_url:
        lines.append(f"  base_url: {base_url}")
    lines.append("  api_mode: anthropic_messages")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize(
    ("kwargs", "env"),
    [
        pytest.param({"requested": "openrouter"}, {}, id="openrouter"),
        pytest.param(
            {"requested": "openrouter", "explicit_api_key": "sk-or-test", "explicit_base_url": OPENROUTER_URL},
            {},
            id="openrouter-after-model-switch",
        ),
        pytest.param({"requested": "custom"}, {"CUSTOM_BASE_URL": LOCAL_URL}, id="bare-custom"),
    ],
)
def test_api_mode_recorded_for_another_provider_does_not_leak(monkeypatch, kwargs, env):
    _write_config(ANTHROPIC_PROFILE)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    resolved = resolve_runtime_provider(target_model="deepseek/deepseek-v4.1-flash-1017", **kwargs)

    assert resolved["provider"] == ("custom" if kwargs["requested"] == "custom" else "openrouter")
    assert resolved["api_mode"] == "chat_completions"


@pytest.mark.parametrize(
    ("config_provider", "base_url", "requested", "env"),
    [
        pytest.param("openrouter", "", "openrouter", {}, id="openrouter-config"),
        pytest.param("", "", "openrouter", {}, id="no-provider-recorded"),
        pytest.param("custom", LOCAL_URL, "custom", {}, id="custom-config"),
        pytest.param("custom:lab", "", "custom", {"CUSTOM_BASE_URL": LOCAL_URL}, id="named-custom-config"),
        pytest.param("ollama", "http://192.168.1.50:11434/v1", "ollama", {}, id="local-server-alias"),
        pytest.param("auto", LOCAL_URL, "auto", {}, id="auto-with-local-endpoint"),
    ],
)
def test_api_mode_recorded_for_this_route_is_honored(monkeypatch, config_provider, base_url, requested, env):
    _write_config(_model_block(config_provider, base_url))
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    resolved = resolve_runtime_provider(requested=requested)

    assert resolved["api_mode"] == "anthropic_messages"
