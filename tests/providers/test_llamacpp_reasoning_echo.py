"""llamacpp profile: reasoning_content echo-back opt-in.

Replayed assistant messages to a llamacpp endpoint keep their
original reasoning_content so llama-server's --reasoning-preserve can
reuse the prompt cache. The profile declares ``echo_reasoning_content``;
``AIAgent._needs_thinking_reasoning_pad`` consults it through the shared
requested-provider-first resolution, alongside the hardcoded
kimi/deepseek/mimo families - which stay exactly as before.
The strip-vs-keep policy itself (``apply_reasoning_content_policy``)
already keys off the pad flag, so no new row enters
``_REASONING_ECHO_RULES``; the interim local-llamacpp row is reverted.
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


def _fake_agent(provider, requested, model, base_url="http://rig:8080/v1"):
    """A real AIAgent shell (no __init__) with just the pad inputs set."""
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.requested_provider = requested
    agent.model = model
    agent.base_url = base_url
    return agent


def _pad(provider, requested, model, base_url="http://rig:8080/v1"):
    return _fake_agent(provider, requested, model, base_url)._needs_thinking_reasoning_pad()


def test_profile_declares_echo_flag(llamacpp_profile):
    assert llamacpp_profile.echo_reasoning_content is True


def test_base_default_is_off():
    from providers.base import ProviderProfile

    assert ProviderProfile(name="x").echo_reasoning_content is False


def test_pad_on_for_llamacpp_entry(llamacpp_profile):
    """Custom entry named llamacpp: provider canonicalizes to 'custom',
    the entry name rides requested_provider - the plugin's aliases claim
    both spellings."""
    assert _pad("custom", "llamacpp", "qwen38-27b-mtp-q8") is True
    assert _pad("custom", "llama-swap", "qwen38-27b-mtp-q8") is True


def test_pad_off_for_other_custom_entry(llamacpp_profile):
    """Same endpoint through a non-llamacpp entry resolves to the stock
    custom profile; a non-DeepSeek model name keeps it on the strict
    (stripped) side."""
    assert _pad("custom", "rigcustom", "qwen38-27b-mtp-q8") is False


def test_hardcoded_families_unchanged(llamacpp_profile):
    assert (
        _pad("deepseek", None, "deepseek-chat", "https://api.deepseek.com")
        is True
    )
    assert (
        _pad("openrouter", None, "qwen/qwen3-8b", "https://openrouter.ai/api/v1")
        is False
    )


def test_pad_cache_distinguishes_requested_provider(llamacpp_profile):
    """Two entries can share provider/model/base_url and differ only in
    the requested name - the pad cache key must tell them apart."""
    fake = _fake_agent("custom", "llamacpp", "m")
    assert fake._needs_thinking_reasoning_pad() is True
    fake.requested_provider = "rigcustom"
    assert fake._needs_thinking_reasoning_pad() is False


def test_policy_keeps_reasoning_verbatim_with_pad(llamacpp_profile):
    """The strip-vs-keep policy flows entirely from the pad flag - with
    it on, replays keep the original reasoning_content verbatim; with it
    off, the key is stripped (strict-provider side)."""
    from agent.message_sanitization import apply_reasoning_content_policy

    src = {"role": "assistant", "content": "x", "reasoning_content": "chain"}
    api = dict(src)
    apply_reasoning_content_policy(src, api, True)
    assert api["reasoning_content"] == "chain"

    api2 = dict(src)
    apply_reasoning_content_policy(src, api2, False)
    assert "reasoning_content" not in api2
