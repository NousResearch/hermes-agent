from argparse import Namespace
import sys
import types

import pytest


def _raise_exit(rc):
    raise SystemExit(rc)


@pytest.fixture
def main_mod(monkeypatch):
    import hermes_cli.main as mod

    monkeypatch.setattr(mod, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(mod, "_oneshot_cleanup_done", False)
    return mod


def test_run_and_exit_oneshot_forwards_resume_fields(monkeypatch, main_mod):
    calls = []
    exits = []

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.oneshot",
        types.SimpleNamespace(
            run_oneshot=lambda prompt, **kwargs: calls.append((prompt, kwargs)) or 0
        ),
    )
    monkeypatch.setattr(main_mod, "_cleanup_oneshot_runtime", lambda: None)
    monkeypatch.setattr(main_mod, "_exit_after_oneshot", lambda rc: exits.append(rc))

    main_mod._run_and_exit_oneshot(
        "continue",
        resume_session_id="session-1",
        continue_last=False,
        restore_resume_cwd=False,
    )

    assert calls == [
        (
            "continue",
            {
                "model": None,
                "provider": None,
                "toolsets": None,
                "usage_file": None,
                "resume_session_id": "session-1",
                "continue_last": False,
                "restore_resume_cwd": False,
            },
        )
    ]
    assert exits == [0]


def test_top_level_oneshot_forwards_default_resume_fields(monkeypatch, main_mod):
    captured = {}
    import hermes_cli.config as config_mod

    monkeypatch.setattr(sys, "argv", ["hermes", "-z", "hello", "--usage-file", "usage.json"])
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", types.SimpleNamespace(discover_plugins=lambda: None))
    monkeypatch.setitem(sys.modules, "tools.mcp_tool", types.SimpleNamespace(discover_mcp_tools=lambda: None))
    monkeypatch.setattr(config_mod, "load_config", lambda: {})
    monkeypatch.setattr(config_mod, "get_container_exec_info", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(register_from_config=lambda _cfg, accept_hooks=False: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.oneshot",
        types.SimpleNamespace(
            run_oneshot=lambda prompt, **kwargs: captured.update({"prompt": prompt, **kwargs}) or 0
        ),
    )
    monkeypatch.setattr(main_mod, "_exit_after_oneshot", _raise_exit)

    with pytest.raises(SystemExit) as exc:
        main_mod.main()

    assert exc.value.code == 0
    assert captured["prompt"] == "hello"
    assert captured["usage_file"] == "usage.json"
    assert captured["resume_session_id"] is None
    assert captured["continue_last"] is None
    assert captured["restore_resume_cwd"] is True


def test_termux_oneshot_forwards_default_resume_fields(monkeypatch, main_mod):
    captured = {}
    prepared = []

    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.delenv("HERMES_TUI", raising=False)
    monkeypatch.setattr(sys, "argv", ["hermes", "-z", "hello", "--usage-file", "usage.json"])
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda args: prepared.append(args.command))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.oneshot",
        types.SimpleNamespace(
            run_oneshot=lambda prompt, **kwargs: captured.update({"prompt": prompt, **kwargs}) or 0
        ),
    )
    monkeypatch.setattr(main_mod, "_exit_after_oneshot", _raise_exit)

    with pytest.raises(SystemExit) as exc:
        main_mod._try_termux_fast_cli_launch()

    assert exc.value.code == 0
    assert prepared == [None]
    assert captured["prompt"] == "hello"
    assert captured["resume_session_id"] is None
    assert captured["continue_last"] is None
    assert captured["restore_resume_cwd"] is True


def test_run_agent_reuses_tip_filters_meta_reopens_and_passes_history(monkeypatch):
    import hermes_cli.oneshot as oneshot_mod

    initialized = []
    run_calls = []
    reopened = []
    closed = []
    history = [
        {"role": "session_meta", "content": "internal"},
        {"role": "user", "content": "before"},
        {"role": "assistant", "content": "context"},
    ]

    class FakeAgent:
        def __init__(self, **kwargs):
            initialized.append(kwargs)
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt, **kwargs):
            run_calls.append((prompt, kwargs))
            return {"final_response": "continued"}

        def shutdown_memory_provider(self, messages=None):
            pass

        def close(self):
            pass

    class FakeSessionDB:
        def get_session(self, session_id):
            if session_id in {"root", "tip"}:
                return {"id": session_id, "cwd": "/missing"}
            return None

        def resolve_session_by_title(self, _title):
            return None

        def resolve_resume_session_id(self, session_id):
            assert session_id == "root"
            return "tip"

        def get_resume_conversations(self, session_id):
            assert session_id == "tip"
            return list(history), list(history)

        def reopen_session(self, session_id):
            reopened.append(session_id)

        def close(self):
            closed.append(True)

    monkeypatch.setitem(sys.modules, "run_agent", types.SimpleNamespace(AIAgent=FakeAgent))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "gpt-test", "provider": "openai"}},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "key",
            "base_url": "https://example.invalid",
            "provider": "openai",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: FakeSessionDB())

    assert oneshot_mod._run_agent(
        "continue",
        model="gpt-test",
        provider="openai",
        use_config_toolsets=False,
        resume_session_id="root",
        restore_resume_cwd=False,
    ) == ("continued", {"final_response": "continued"})
    assert initialized[0]["session_id"] == "tip"
    assert reopened == ["tip"]
    assert closed == [True]
    assert run_calls == [
        (
            "continue",
            {
                "conversation_history": [
                    {"role": "user", "content": "before"},
                    {"role": "assistant", "content": "context"},
                ]
            },
        )
    ]


def test_continue_prefers_current_workspace_session(monkeypatch, main_mod):
    calls = []

    class FakeDB:
        def search_sessions(self, **kwargs):
            calls.append(kwargs)
            return [{"id": "workspace-session"}] if kwargs.get("workspace_key") == "/repo" else []

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "hermes_state", types.SimpleNamespace(SessionDB=lambda: FakeDB()))
    monkeypatch.setattr(main_mod, "_resolve_workspace_key", lambda: "/repo")

    assert main_mod._resolve_last_session("cli") == "workspace-session"
    assert calls == [{"source": "cli", "limit": 1, "workspace_key": "/repo"}]


def test_continue_uses_global_mru_only_without_workspace_session(monkeypatch, main_mod):
    calls = []

    class FakeDB:
        def search_sessions(self, **kwargs):
            calls.append(kwargs)
            return [] if kwargs.get("workspace_key") else [{"id": "global-session"}]

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "hermes_state", types.SimpleNamespace(SessionDB=lambda: FakeDB()))
    monkeypatch.setattr(main_mod, "_resolve_workspace_key", lambda: "/repo")

    assert main_mod._resolve_last_session("cli") == "global-session"
    assert calls == [
        {"source": "cli", "limit": 1, "workspace_key": "/repo"},
        {"source": "cli", "limit": 1},
    ]
