import os
import sys
import time
import types
from types import SimpleNamespace

import pytest


class _Args(SimpleNamespace):
    def __getattr__(self, name):
        return None


class _StopChatStartup(Exception):
    pass


def _install_fake_cli_module(monkeypatch):
    fake_cli = types.ModuleType("cli")

    def _fake_main(*args, **kwargs):
        raise _StopChatStartup()

    fake_cli.main = _fake_main
    monkeypatch.setitem(sys.modules, "cli", fake_cli)


def test_cmd_chat_uses_directory_root_scope_and_does_not_inject_by_default(monkeypatch):
    import hermes_cli.main as main_mod
    import hermes_state
    import hermes_cli.config as config_mod

    call = {}

    class FakeSessionDB:
        def get_last_summarized_session(self, **kwargs):
            call.update(kwargs)
            return {
                "id": "sess-1",
                "title": "prior",
                "message_count": 12,
                "started_at": time.time() - 120,
                "exit_summary": "summary text",
            }

        def close(self):
            return None

    monkeypatch.setattr(hermes_state, "SessionDB", FakeSessionDB)
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "continuity": {"enabled": True, "recency_hours": 4, "min_messages": 5, "show_prompt": False},
            "session": {"inject_previous_summary": False, "lineage_scope": "directory_root"},
        },
    )
    monkeypatch.setattr(main_mod.os, "getcwd", lambda: "/tmp/repo/subdir")
    monkeypatch.setattr(main_mod, "_resolve_cli_lineage_root", lambda cwd: "/tmp/repo")

    _install_fake_cli_module(monkeypatch)

    args = _Args(resume=None, continue_last=False, continue_name=None)

    with pytest.raises(_StopChatStartup):
        monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
        main_mod.cmd_chat(args)

    assert call["cwd"] == "/tmp/repo"
    assert call["cwd_scope"] == "directory_root"
    assert getattr(args, "_last_session_summary", None) is None


def test_cmd_chat_exact_cwd_scope_and_injects_when_enabled(monkeypatch):
    import hermes_cli.main as main_mod
    import hermes_state
    import hermes_cli.config as config_mod

    call = {}

    class FakeSessionDB:
        def get_last_summarized_session(self, **kwargs):
            call.update(kwargs)
            return {
                "id": "sess-2",
                "title": "prior",
                "message_count": 9,
                "started_at": time.time() - 60,
                "exit_summary": "carry this",
            }

        def close(self):
            return None

    monkeypatch.setattr(hermes_state, "SessionDB", FakeSessionDB)
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda: {
            "continuity": {"enabled": True, "recency_hours": 4, "min_messages": 5, "show_prompt": False},
            "session": {"inject_previous_summary": True, "lineage_scope": "exact_cwd"},
        },
    )
    monkeypatch.setattr(main_mod.os, "getcwd", lambda: "/tmp/repo/subdir")

    _install_fake_cli_module(monkeypatch)

    args = _Args(resume=None, continue_last=False, continue_name=None)

    with pytest.raises(_StopChatStartup):
        monkeypatch.setattr(main_mod, "_has_any_provider_configured", lambda: True)
        main_mod.cmd_chat(args)

    assert call["cwd_scope"] == "exact"
    assert call["cwd"] == os.path.realpath("/tmp/repo/subdir")
    assert args._last_session_summary == "carry this"
    assert isinstance(args._last_session_time, (float, int))
