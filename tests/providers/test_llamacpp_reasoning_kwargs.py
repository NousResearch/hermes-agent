"""llamacpp profile: chat_template_kwargs.reasoning_effort emission.

Configured reasoning effort reaches the server as
chat_template_kwargs.reasoning_effort (llama-server's Jinja variable
channel; the OpenAI SDK merges extra_body into the JSON body top level).
Per-model agent.reasoning_overrides win over the global effort -
resolution happens in hermes_constants.resolve_reasoning_config, so these
tests chain the real resolver into the hook.
"""

from __future__ import annotations

import shutil

import pytest

from tests.providers.test_llamacpp_profile import (
    _fresh_hermes_home,
    _installed_plugin_dir,
)
from tests.providers.test_plugin_discovery import _clear_provider_caches

pytestmark = pytest.mark.skipif(
    _installed_plugin_dir() is None,
    reason="real llamacpp user plugin not installed on this machine",
)


@pytest.fixture()
def llamacpp_profile(tmp_path, monkeypatch):
    hermes_home = _fresh_hermes_home(tmp_path, monkeypatch)
    plugin_dir = hermes_home / "plugins" / "model-providers" / "llamacpp"
    plugin_dir.parent.mkdir(parents=True)
    shutil.copytree(
        _installed_plugin_dir(),
        plugin_dir,
        ignore=shutil.ignore_patterns(".git", "__pycache__"),
    )
    _clear_provider_caches()
    from providers import get_provider_profile

    profile = get_provider_profile("llamacpp")
    assert profile is not None and profile.name == "llamacpp"
    yield profile
    _clear_provider_caches()


def test_effort_reaches_chat_template_kwargs(llamacpp_profile):
    extra, top = llamacpp_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "medium"}
    )
    assert extra == {"chat_template_kwargs": {"reasoning_effort": "medium"}}
    assert top == {}


@pytest.mark.parametrize("effort", ["low", "high", "xhigh", "minimal"])
def test_hook_passes_levels_verbatim(llamacpp_profile, effort):
    extra, _ = llamacpp_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": effort}
    )
    assert extra["chat_template_kwargs"]["reasoning_effort"] == effort


def test_no_reasoning_config_emits_nothing(llamacpp_profile):
    assert llamacpp_profile.build_api_kwargs_extras(reasoning_config=None) == (
        {},
        {},
    )


def test_empty_effort_emits_nothing(llamacpp_profile):
    extra, top = llamacpp_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": ""}
    )
    assert extra == {} and top == {}


def test_disabled_emits_no_reasoning_effort(llamacpp_profile):
    """Thinking-off wiring is tested separately; here we only guarantee no effort leaks."""
    extra, top = llamacpp_profile.build_api_kwargs_extras(
        reasoning_config={"enabled": False}
    )
    assert "reasoning_effort" not in str(extra)
    assert top == {}


def test_per_model_overrides_feed_the_hook(llamacpp_profile):
    from hermes_constants import resolve_reasoning_config

    cfg = {
        "agent": {
            "reasoning_effort": "medium",
            "reasoning_overrides": {"model-a": "low", "model-b": "xhigh"},
        }
    }
    for model, expected in (
        ("model-a", "low"),
        ("model-b", "xhigh"),
        ("model-c", "medium"),  # no override -> global
    ):
        rc = resolve_reasoning_config(cfg, model)
        extra, _ = llamacpp_profile.build_api_kwargs_extras(reasoning_config=rc)
        assert extra["chat_template_kwargs"]["reasoning_effort"] == expected, model


def test_transport_merges_into_extra_body(llamacpp_profile):
    from agent.transports.chat_completions import ChatCompletionsTransport

    kw = ChatCompletionsTransport().build_kwargs(
        model="qwen38-27b-mtp-q8",
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        provider_profile=llamacpp_profile,
        reasoning_config={"enabled": True, "effort": "medium"},
        base_url="http://rig:8080/v1",
    )
    assert kw["extra_body"]["chat_template_kwargs"] == {
        "reasoning_effort": "medium"
    }
    assert "reasoning_effort" not in kw  # never a top-level OpenAI param
