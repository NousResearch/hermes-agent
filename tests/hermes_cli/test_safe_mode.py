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










