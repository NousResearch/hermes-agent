"""Regression: `hermes doctor` must not report a keyless Mem0 OSS install as broken.

Mem0 in ``oss`` mode builds its backend from the local ``oss`` config block
(an ollama-compatible LLM + a vector store) and resolves no platform
credential; the plugin's own ``_load_config()`` sets ``api_key=""`` for that
mode deliberately. The doctor check previously keyed only on ``api_key``, so
every self-hosted install was reported as "Mem0 API key not set" — a failure
row in an otherwise healthy setup.
"""

import importlib
import sys

import pytest

from hermes_cli import doctor_state


@pytest.fixture
def mem0_checker(monkeypatch):
    """Return _memory_provider_mem0 with the plugin's config loader stubbed."""

    def _run(cfg):
        pytest.importorskip("plugins.memory.mem0")
        plugin = importlib.import_module("plugins.memory.mem0")
        monkeypatch.setattr(plugin, "_load_config", lambda: cfg, raising=True)
        issues: list = []
        doctor_state._memory_provider_mem0(issues)
        return issues

    return _run


def test_oss_mode_without_api_key_is_ok(mem0_checker):
    """oss mode needs no platform key — it must not produce an issue row."""
    issues = mem0_checker(
        {"mode": "oss", "api_key": "", "user_id": "stone", "agent_id": "hermes", "oss": {}}
    )
    assert issues == []


def test_platform_mode_without_api_key_still_fails(mem0_checker):
    """The guard must not silence the genuine platform-mode misconfiguration."""
    issues = mem0_checker({"mode": "platform", "api_key": ""})
    assert issues, "platform mode without a key must still report an issue"


def test_platform_mode_with_api_key_is_ok(mem0_checker):
    """A configured platform key reports no issue (unchanged behaviour)."""
    issues = mem0_checker({"mode": "platform", "api_key": "mem0-key-placeholder"})
    assert issues == []


def test_missing_mode_key_defaults_to_platform(mem0_checker):
    """A config with no ``mode`` is platform mode — the historical default."""
    issues = mem0_checker({"api_key": ""})
    assert issues, "absent mode must behave as platform (needs a key)"
