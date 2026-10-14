"""Regression tests for the per-chat background-injection mute list loader.

The mute list (``_load_agent_inject_mute_chats``) decides which chats skip the
wake-up agent injection on *successful* background completion. Wrong parsing here
silently mutes (or fails to mute) the wrong chats, so the loader's three sources
(env / config list / empty default) and their precedence are pinned.
"""

import pytest

import gateway.platforms._shared as _shared
import gateway.run as _run
from gateway.run_config_loaders import GatewayConfigLoadersMixin


@pytest.fixture
def no_env(monkeypatch):
    """Isolate from the real env and the real gateway config."""
    monkeypatch.delenv("HERMES_AGENT_INJECT_MUTE_CHATS", raising=False)
    monkeypatch.setattr(_shared, "platform_gate_env", lambda name, default="": default)
    monkeypatch.setattr(_run, "_load_gateway_config", lambda: {})
    return monkeypatch


def test_mute_chats_empty_by_default(no_env):
    assert GatewayConfigLoadersMixin._load_agent_inject_mute_chats() == set()


def test_mute_chats_from_env_list(no_env):
    no_env.setattr(
        _shared,
        "platform_gate_env",
        lambda name, default="": " oc_a ,oc_b ,, oc_c " if name == "HERMES_AGENT_INJECT_MUTE_CHATS" else default,
    )
    assert GatewayConfigLoadersMixin._load_agent_inject_mute_chats() == {"oc_a", "oc_b", "oc_c"}


def test_mute_chats_from_config_list(no_env):
    no_env.setattr(
        _run,
        "_load_gateway_config",
        lambda: {"display": {"background_process_agent_inject_mute_chats": ["oc_x", 123]}},
    )
    assert GatewayConfigLoadersMixin._load_agent_inject_mute_chats() == {"oc_x", "123"}


def test_mute_chats_env_wins_over_config(no_env):
    no_env.setattr(
        _shared,
        "platform_gate_env",
        lambda name, default="": "oc_env" if name == "HERMES_AGENT_INJECT_MUTE_CHATS" else default,
    )
    no_env.setattr(
        _run,
        "_load_gateway_config",
        lambda: {"display": {"background_process_agent_inject_mute_chats": ["oc_cfg"]}},
    )
    assert GatewayConfigLoadersMixin._load_agent_inject_mute_chats() == {"oc_env"}


def test_mute_chats_non_list_config_returns_empty(no_env):
    no_env.setattr(
        _run,
        "_load_gateway_config",
        lambda: {"display": {"background_process_agent_inject_mute_chats": "oc_str"}},
    )
    assert GatewayConfigLoadersMixin._load_agent_inject_mute_chats() == set()
