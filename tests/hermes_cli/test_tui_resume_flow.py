from argparse import Namespace
from pathlib import Path
import sys
import types

import pytest


def _args(**overrides):
    base = {
        "continue_last": None,
        "model": None,
        "provider": None,
        "resume": None,
        "toolsets": None,
        "tui": True,
        "tui_dev": False,
    }
    base.update(overrides)
    return Namespace(**base)


@pytest.fixture
def main_mod(monkeypatch):
    import hermes_cli.main as mod

    monkeypatch.setattr(mod, "_has_any_provider_configured", lambda: True)
    return mod


def test_cmd_chat_tui_continue_uses_latest_tui_session(monkeypatch, main_mod):
    calls = []
    captured = {}

    def fake_resolve_last(source="cli"):
        calls.append(source)
        return "20260408_235959_a1b2c3" if source == "tui" else None

    def fake_launch(
        resume_session_id=None,
        tui_dev=False,
        model=None,
        provider=None,
        toolsets=None,
        **kwargs,
    ):
        captured["resume"] = resume_session_id
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_resolve_last_session", fake_resolve_last)
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda val: val)
    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(continue_last=True))

    assert calls == ["tui"]
    assert captured["resume"] == "20260408_235959_a1b2c3"


def test_cmd_chat_tui_continue_falls_back_to_latest_cli_session(monkeypatch, main_mod):
    calls = []
    captured = {}

    def fake_resolve_last(source="cli"):
        calls.append(source)
        if source == "tui":
            return None
        if source == "cli":
            return "20260408_235959_d4e5f6"
        return None

    def fake_launch(
        resume_session_id=None,
        tui_dev=False,
        model=None,
        provider=None,
        toolsets=None,
        **kwargs,
    ):
        captured["resume"] = resume_session_id
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_resolve_last_session", fake_resolve_last)
    monkeypatch.setattr(main_mod, "_resolve_session_by_name_or_id", lambda val: val)
    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(continue_last=True))

    assert calls == ["tui", "cli"]
    assert captured["resume"] == "20260408_235959_d4e5f6"


def test_cmd_chat_tui_resume_resolves_title_before_launch(monkeypatch, main_mod):
    captured = {}

    def fake_launch(
        resume_session_id=None,
        tui_dev=False,
        model=None,
        provider=None,
        toolsets=None,
        **kwargs,
    ):
        captured["resume"] = resume_session_id
        raise SystemExit(0)

    monkeypatch.setattr(
        main_mod, "_resolve_session_by_name_or_id", lambda val: "20260409_000000_aa11bb"
    )
    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(resume="my t0p session"))

    assert captured["resume"] == "20260409_000000_aa11bb"


def test_cmd_chat_tui_passes_model_and_provider(monkeypatch, main_mod):
    captured = {}

    def fake_launch(
        resume_session_id=None,
        tui_dev=False,
        model=None,
        provider=None,
        toolsets=None,
        **kwargs,
    ):
        captured.update(
            {
                "model": model,
                "provider": provider,
                "resume": resume_session_id,
                "toolsets": toolsets,
                "tui_dev": tui_dev,
            }
        )
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(
            _args(model="anthropic/claude-sonnet-4.6", provider="anthropic")
        )

    assert captured == {
        "model": "anthropic/claude-sonnet-4.6",
        "provider": "anthropic",
        "resume": None,
        "toolsets": None,
        "tui_dev": False,
    }


def test_cmd_chat_tui_passes_toolsets(monkeypatch, main_mod):
    captured = {}

    def fake_launch(
        resume_session_id=None,
        tui_dev=False,
        model=None,
        provider=None,
        toolsets=None,
        **kwargs,
    ):
        captured["toolsets"] = toolsets
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(_args(toolsets="web,terminal"))

    assert captured["toolsets"] == "web,terminal"


def test_cmd_chat_tui_forwards_chat_flags(monkeypatch, main_mod):
    captured = {}

    def fake_launch(resume_session_id=None, **kwargs):
        captured["resume_session_id"] = resume_session_id
        captured.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(main_mod, "_launch_tui", fake_launch)

    with pytest.raises(SystemExit):
        main_mod.cmd_chat(
            _args(
                skills=["foo,bar"],
                verbose=True,
                quiet=True,
                query="hello",
                image="/tmp/cat.png",
                worktree=True,
                checkpoints=True,
                pass_session_id=True,
                max_turns=7,
                accept_hooks=True,
            )
        )

    assert captured["skills"] == ["foo,bar"]
    assert captured["verbose"] is True
    assert captured["quiet"] is True
    assert captured["query"] == "hello"
    assert captured["image"] == "/tmp/cat.png"
    assert captured["worktree"] is True
    assert captured["checkpoints"] is True
    assert captured["pass_session_id"] is True
    assert captured["max_turns"] == 7
    assert captured["accept_hooks"] is True


def test_main_top_level_tui_accepts_toolsets(monkeypatch, main_mod):
    captured = {}

    import hermes_cli.config as config_mod

    monkeypatch.setattr(sys, "argv", ["hermes", "--tui", "--toolsets", "web,terminal"])
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=lambda: None),
    )
    monkeypatch.setattr(config_mod, "load_config", lambda: {})
    monkeypatch.setattr(config_mod, "get_container_exec_info", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(
            register_from_config=lambda _cfg, accept_hooks=False: None
        ),
    )
    monkeypatch.setattr(
        main_mod,
        "cmd_chat",
        lambda args: captured.update({"toolsets": args.toolsets, "tui": args.tui}),
    )

    main_mod.main()

    assert captured == {"toolsets": "web,terminal", "tui": True}


def test_main_top_level_oneshot_accepts_toolsets(monkeypatch, main_mod):
    captured = {}

    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        sys, "argv", ["hermes", "-z", "hello", "--toolsets", "web,terminal"]
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool",
        types.SimpleNamespace(discover_mcp_tools=lambda: None),
    )
    monkeypatch.setattr(config_mod, "load_config", lambda: {})
    monkeypatch.setattr(config_mod, "get_container_exec_info", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "agent.shell_hooks",
        types.SimpleNamespace(
            register_from_config=lambda _cfg, accept_hooks=False: None
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.oneshot",
        types.SimpleNamespace(
            run_oneshot=lambda prompt, **kwargs: captured.update(
                {"prompt": prompt, **kwargs}
            )
            or 0
        ),
    )

    with pytest.raises(SystemExit) as exc:
        main_mod.main()

    assert exc.value.code == 0
    assert captured == {
        "prompt": "hello",
        "model": None,
        "provider": None,
        "toolsets": "web,terminal",
    }


def _stub_plugin_discovery(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )


def test_oneshot_rejects_invalid_only_toolsets(monkeypatch, capsys):
    _stub_plugin_discovery(monkeypatch)
    from hermes_cli.oneshot import run_oneshot

    assert run_oneshot("hello", toolsets="nope") == 2
    err = capsys.readouterr().err
    assert "nope" in err
    assert "did not contain any valid toolsets" in err


def test_oneshot_filters_invalid_toolsets_before_redirect(monkeypatch, capsys):
    _stub_plugin_discovery(monkeypatch)
    from hermes_cli.oneshot import _validate_explicit_toolsets

    valid, error = _validate_explicit_toolsets("web,nope")

    assert valid == ["web"]
    assert error is None
    assert "nope" in capsys.readouterr().err


def test_oneshot_all_toolsets_means_all_not_configured_cli():
    from hermes_cli.oneshot import _validate_explicit_toolsets

    valid, error = _validate_explicit_toolsets("all")

    assert valid is None
    assert error is None


def test_oneshot_all_toolsets_warns_about_ignored_extra_entries(monkeypatch, capsys):
    _stub_plugin_discovery(monkeypatch)
    from hermes_cli.oneshot import _validate_explicit_toolsets

    valid, error = _validate_explicit_toolsets("all,nope")

    assert valid is None
    assert error is None
    assert "ignoring additional entries: nope" in capsys.readouterr().err


def test_oneshot_accepts_plugin_toolset_after_discovery(monkeypatch):
    import toolsets

    from hermes_cli.oneshot import _validate_explicit_toolsets

    discovered = {"ready": False}
    original_validate = toolsets.validate_toolset

    def fake_validate(name):
        return name == "plugin_demo" and discovered["ready"] or original_validate(name)

    monkeypatch.setattr(toolsets, "validate_toolset", fake_validate)
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(
            discover_plugins=lambda: discovered.update({"ready": True})
        ),
    )

    valid, error = _validate_explicit_toolsets("plugin_demo")

    assert valid == ["plugin_demo"]
    assert error is None


def test_oneshot_rejects_disabled_mcp_toolset(monkeypatch, capsys):
    _stub_plugin_discovery(monkeypatch)
    import hermes_cli.config as config_mod

    from hermes_cli.oneshot import _validate_explicit_toolsets

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: {"mcp_servers": {"mcp-off": {"enabled": False}}},
    )

    valid, error = _validate_explicit_toolsets("mcp-off")

    assert valid is None
    assert error == "hermes -z: --toolsets did not contain any valid toolsets.\n"
    err = capsys.readouterr().err
    assert "ignoring disabled MCP servers" in err
    assert "mcp-off" in err


def test_oneshot_distinguishes_disabled_mcp_from_unknown(monkeypatch, capsys):
    _stub_plugin_discovery(monkeypatch)
    import hermes_cli.config as config_mod

    from hermes_cli.oneshot import _validate_explicit_toolsets

    monkeypatch.setattr(
        config_mod,
        "read_raw_config",
        lambda: {"mcp_servers": {"mcp-off": {"enabled": False}}},
    )

    valid, error = _validate_explicit_toolsets("web,mcp-off,nope")

    assert valid == ["web"]
    assert error is None
    err = capsys.readouterr().err
    assert "ignoring unknown --toolsets entries: nope" in err
    assert "ignoring disabled MCP servers" in err
    assert "mcp-off" in err


def test_oneshot_wires_session_db_for_recall(monkeypatch):
    """hermes -z bypasses HermesCLI, but recall still needs SessionDB."""
    from hermes_cli.oneshot import _run_agent

    captured = {}
    sentinel_db = object()

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def chat(self, prompt):
            captured["prompt"] = prompt
            return "ok"

    class FakeSessionDB:
        def __new__(cls):
            return sentinel_db

    def mod(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(sys.modules, "hermes_state", mod("hermes_state", SessionDB=FakeSessionDB))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        mod("hermes_cli.models", detect_provider_for_model=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "k",
                "base_url": "u",
                "provider": "p",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        mod("hermes_cli.tools_config", _get_platform_tools=lambda *_args, **_kwargs: {"session_search"}),
    )

    assert _run_agent("recall this") == "ok"
    assert captured["session_db"] is sentinel_db
    assert captured["enabled_toolsets"] == ["session_search"]
    assert captured["prompt"] == "recall this"


def test_launch_tui_exports_model_provider_and_toolsets(monkeypatch, main_mod):
    captured = {}
    active_path_during_call = None

    monkeypatch.setattr(
        main_mod,
        "_make_tui_argv",
        lambda tui_dir, tui_dev: (["node", "dist/entry.js"], Path(".")),
    )

    def fake_call(argv, cwd=None, env=None):
        nonlocal active_path_during_call
        captured.update({"argv": argv, "cwd": cwd, "env": env})
        active_path_during_call = Path(env["HERMES_TUI_ACTIVE_SESSION_FILE"])
        assert active_path_during_call.exists()
        return 1

    monkeypatch.setattr(main_mod.subprocess, "call", fake_call)

    with pytest.raises(SystemExit):
        main_mod._launch_tui(
            model="nous/hermes-test", provider="nous", toolsets="web, terminal"
        )

    env = captured["env"]
    assert env["HERMES_MODEL"] == "nous/hermes-test"
    assert env["HERMES_INFERENCE_MODEL"] == "nous/hermes-test"
    assert env["HERMES_TUI_PROVIDER"] == "nous"
    assert env["HERMES_INFERENCE_PROVIDER"] == "nous"
    assert env["HERMES_TUI_TOOLSETS"] == "web,terminal"
    active_path = Path(env["HERMES_TUI_ACTIVE_SESSION_FILE"])
    assert active_path.name.startswith("hermes-tui-active-session-")
    assert active_path.suffix == ".json"
    assert active_path_during_call == active_path
    assert not active_path.exists()
    assert env["NODE_ENV"] == "production"


def test_print_tui_exit_summary_includes_resume_and_token_totals(monkeypatch, capsys):
    import hermes_cli.main as main_mod

    class _FakeDB:
        def get_session(self, session_id):
            assert session_id == "20260409_000001_abc123"
            return {
                "message_count": 2,
                "input_tokens": 10,
                "output_tokens": 6,
                "cache_read_tokens": 2,
                "cache_write_tokens": 2,
                "reasoning_tokens": 1,
            }

        def get_session_title(self, _session_id):
            return "demo title"

        def close(self):
            return None

    monkeypatch.setitem(
        sys.modules, "hermes_state", types.SimpleNamespace(SessionDB=lambda: _FakeDB())
    )

    main_mod._print_tui_exit_summary("20260409_000001_abc123")
    out = capsys.readouterr().out

    assert "Resume this session with:" in out
    assert "hermes --tui --resume 20260409_000001_abc123" in out
    assert 'hermes --tui -c "demo title"' in out
    assert "Tokens:         21 (in 10, out 6, cache 4, reasoning 1)" in out


def test_print_tui_exit_summary_prefers_actual_active_session_file(
    monkeypatch, capsys, tmp_path
):
    import hermes_cli.main as main_mod

    seen = []

    class _FakeDB:
        def get_session(self, session_id):
            seen.append(session_id)
            return {
                "message_count": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "reasoning_tokens": 0,
            }

        def get_session_title(self, _session_id):
            return "actual"

        def close(self):
            return None

    active = tmp_path / "active.json"
    active.write_text('{"session_id":"actual_session"}', encoding="utf-8")
    monkeypatch.setitem(
        sys.modules, "hermes_state", types.SimpleNamespace(SessionDB=lambda: _FakeDB())
    )

    main_mod._print_tui_exit_summary("startup_resume", str(active))
    out = capsys.readouterr().out

    assert seen == ["actual_session"]
    assert "hermes --tui --resume actual_session" in out
    assert "startup_resume" not in out
