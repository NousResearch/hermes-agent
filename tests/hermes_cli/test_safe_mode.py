"""Tests for `hermes chat --safe-mode` isolation."""

from __future__ import annotations

import os
import sys
import types

import pytest


_VARS = ("HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG", "HERMES_IGNORE_RULES")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)
    yield
    for var in _VARS:
        os.environ.pop(var, None)


def test_cmd_chat_safe_mode_sets_env_before_startup(monkeypatch):
    import hermes_cli.main as main_mod
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=main_mod.cmd_chat)
    args = parser.parse_args(["chat", "--safe-mode"])
    captured: dict[str, object] = {}
    fake_cli = types.ModuleType("cli")

    def fake_has_provider() -> bool:
        assert os.environ["HERMES_SAFE_MODE"] == "1"
        assert os.environ["HERMES_IGNORE_USER_CONFIG"] == "1"
        assert os.environ["HERMES_IGNORE_RULES"] == "1"
        return True

    def fake_main(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(main_mod, "_has_any_provider_configured", fake_has_provider)
    monkeypatch.setattr(main_mod, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(main_mod, "_sync_bundled_skills_for_startup", lambda: None)
    monkeypatch.setattr(main_mod, "_termux_should_prefetch_update_check", lambda: False)
    setattr(fake_cli, "main", fake_main)
    monkeypatch.setitem(sys.modules, "cli", fake_cli)

    main_mod.cmd_chat(args)

    assert captured["ignore_user_config"] is True
    assert captured["ignore_rules"] is True




def test_plugin_discovery_skipped(monkeypatch):
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    from hermes_cli.plugins import PluginManager

    mgr = PluginManager()
    called = []
    monkeypatch.setattr(mgr, "_discover_and_load_inner", lambda: called.append(True))

    mgr.discover_and_load()

    assert called == []
    assert mgr._discovered is True
    assert mgr._plugins == {}


def _write_honcho_home(home) -> None:
    """Minimal HERMES_HOME with memory.provider=honcho configured (#62406)."""
    import json

    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "memory:\n  provider: honcho\n", encoding="utf-8"
    )
    # Port 9 is unroutable in practice; is_available()/initialize() make no
    # network calls, so the backend is never contacted either way.
    (home / "honcho.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "base_url": "http://127.0.0.1:9",
                "api_key": "test-key",
                "workspace": "hermes",
                "aiPeer": "test",
            }
        ),
        encoding="utf-8",
    )


def _honcho_loaded(agent) -> bool:
    mm = getattr(agent, "_memory_manager", None)
    names = [type(p).__name__ for p in (mm.providers if mm else [])]
    return any("honcho" in n.lower() for n in names)


def test_memory_provider_skipped_in_safe_mode(monkeypatch, tmp_path):
    """#62406: --safe-mode (which sets HERMES_IGNORE_USER_CONFIG=1) must not
    load the external memory provider even when config.yaml sets
    memory.provider and honcho.json exists — safe mode is documented as
    disabling ALL customizations, and the Honcho plugin used to initialize
    and issue live HTTP calls to the configured backend in this mode."""
    _write_honcho_home(tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")

    import run_agent

    # Explicit inert LLM config so construction passes the provider check
    # without depending on ambient credentials (conftest blanks them).
    # Nothing contacts the network during __init__.
    agent = run_agent.AIAgent(
        provider="openai",
        model="test-model",
        base_url="http://127.0.0.1:9",
        api_key="test-key",
    )

    assert not _honcho_loaded(agent)


def test_memory_provider_loaded_without_safe_mode(monkeypatch, tmp_path):
    """Control for the safe-mode test: same home, no isolation flags — the
    Honcho provider loads, so the skip above is attributable to the flag
    and not to the fixture environment."""
    _write_honcho_home(tmp_path / "home")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

    import run_agent

    # Explicit inert LLM config so construction passes the provider check
    # without depending on ambient credentials (conftest blanks them).
    # Nothing contacts the network during __init__.
    agent = run_agent.AIAgent(
        provider="openai",
        model="test-model",
        base_url="http://127.0.0.1:9",
        api_key="test-key",
    )

    assert _honcho_loaded(agent)










