from argparse import Namespace
import os
from pathlib import Path
import subprocess
import sys
import textwrap
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


def _raise_exit(rc):
    raise SystemExit(rc)


@pytest.fixture
def main_mod(monkeypatch):
    import hermes_cli.main as mod

    monkeypatch.setattr(mod, "_has_any_provider_configured", lambda: True)
    # Reset the idempotency guard so each test starts fresh.
    monkeypatch.setattr(mod, "_oneshot_cleanup_done", False)
    return mod
















def test_termux_skips_bundled_skill_sync_when_stamp_fresh(monkeypatch, tmp_path, main_mod):
    calls = []

    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.setattr(main_mod, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(main_mod, "_termux_bundled_skills_fingerprint", lambda: "fp1")
    main_mod._mark_termux_bundled_skills_synced()
    monkeypatch.setitem(
        sys.modules,
        "tools.skills_sync",
        types.SimpleNamespace(sync_skills=lambda quiet: calls.append(quiet)),
    )

    assert main_mod._sync_bundled_skills_for_startup() is False
    assert calls == []






def test_exit_after_oneshot_flushes_stdio_and_calls_os_exit(
    monkeypatch, main_mod
):
    flushed = []
    exits = []

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def flush(self):
            flushed.append(self.name)

    def fake_exit(rc):
        exits.append(rc)
        raise SystemExit(rc)

    monkeypatch.setattr(main_mod.sys, "stdout", FakeStream("stdout"))
    monkeypatch.setattr(main_mod.sys, "stderr", FakeStream("stderr"))
    monkeypatch.setattr(main_mod.os, "_exit", fake_exit)
    monkeypatch.setattr("logging.shutdown", lambda: None)

    with pytest.raises(SystemExit) as exc:
        main_mod._exit_after_oneshot(17)

    assert exc.value.code == 17
    assert exits == [17]
    assert flushed == ["stdout", "stderr"]






def test_oneshot_subprocess_exits_without_teardown_abort():
    program = textwrap.dedent(
        """
        import hermes_cli.oneshot as oneshot
        from hermes_cli.main import _exit_after_oneshot

        oneshot._run_agent = lambda *args, **kwargs: ("ok", {"final_response": "ok"})
        _exit_after_oneshot(oneshot.run_oneshot("hello"))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == b"ok\n"
    # Don't demand byte-empty stderr — an import-time warning from the heavy
    # CLI import chain shouldn't fail this. What matters is no crash traceback.
    assert b"Traceback" not in result.stderr








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


def test_oneshot_passes_runtime_default_headers_to_aiagent(monkeypatch):
    captured = {}

    class _FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_conversation(self, prompt):
            return {"final_response": "ok"}

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"model": {"default": "m", "provider": "custom"}},
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": "https://example.test/v1",
            "api_key": "test-key",
            "default_headers": {"X-Test": "value"},
        },
    )
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *a, **k: set())
    monkeypatch.setattr("run_agent.AIAgent", _FakeAgent)

    from hermes_cli.oneshot import _run_agent

    result = _run_agent("hello", model="m", provider="custom", use_config_toolsets=False)

    assert result[0] == "ok"
    assert captured["provider"] == "custom"
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["default_headers"] == {"X-Test": "value"}


def test_oneshot_fails_closed_on_empty_final_response(monkeypatch, capsys):
    _stub_plugin_discovery(monkeypatch)
    import hermes_cli.oneshot as oneshot_mod

    monkeypatch.setattr(oneshot_mod, "_run_agent", lambda *_args, **_kwargs: ("", {}))

    assert oneshot_mod.run_oneshot("hello") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no final response" in captured.err


def test_oneshot_prints_nonempty_final_response(monkeypatch, capsys):
    _stub_plugin_discovery(monkeypatch)
    import hermes_cli.oneshot as oneshot_mod

    monkeypatch.setattr(oneshot_mod, "_run_agent", lambda *_args, **_kwargs: ("done", {}))

    assert oneshot_mod.run_oneshot("hello") == 0
    captured = capsys.readouterr()
    assert captured.out == "done\n"
    assert captured.err == ""


def test_oneshot_fails_closed_on_agent_exception(monkeypatch, capsys):
    _stub_plugin_discovery(monkeypatch)
    import hermes_cli.oneshot as oneshot_mod

    def _boom(*_args, **_kwargs):
        raise OSError("not a TTY")

    monkeypatch.setattr(oneshot_mod, "_run_agent", _boom)

    assert oneshot_mod.run_oneshot("hello") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "agent failed" in captured.err
    assert "not a TTY" in captured.err


def test_oneshot_exit_code_when_failed_without_response(monkeypatch):
    from hermes_cli.oneshot import run_oneshot

    monkeypatch.setattr(
        "hermes_cli.oneshot._run_agent",
        lambda *_a, **_k: ("", {"failed": True, "partial": False}),
    )
    assert run_oneshot("hi") == 2


def test_oneshot_exit_code_zero_when_failed_with_error_text(monkeypatch, capsys):
    from hermes_cli.oneshot import run_oneshot

    monkeypatch.setattr(
        "hermes_cli.oneshot._run_agent",
        lambda *_a, **_k: (
            "API call failed after 3 retries: HTTP 404: model not found",
            {"failed": True, "partial": False},
        ),
    )
    assert run_oneshot("hi") == 0
    assert "HTTP 404" in capsys.readouterr().out


def test_oneshot_reraises_keyboard_interrupt(monkeypatch):
    _stub_plugin_discovery(monkeypatch)
    import hermes_cli.oneshot as oneshot_mod
    import pytest as _pytest

    def _interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(oneshot_mod, "_run_agent", _interrupt)

    with _pytest.raises(KeyboardInterrupt):
        oneshot_mod.run_oneshot("hello")


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

        def run_conversation(self, prompt, **_kwargs):
            captured["prompt"] = prompt
            return {"final_response": "ok", "failed": False, "partial": False}

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

    text, result = _run_agent("recall this")
    assert text == "ok"
    assert not result.get("failed")
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




def test_make_tui_argv_dev_prebuilds_hermes_ink(monkeypatch, main_mod, tmp_path):
    tui_dir = tmp_path / "ui-tui"
    tsx = tui_dir / "node_modules" / ".bin" / "tsx"
    ink_dir = tui_dir / "packages" / "hermes-ink"
    tsx.parent.mkdir(parents=True)
    ink_dir.mkdir(parents=True)
    tsx.write_text("#!/usr/bin/env node\n", encoding="utf-8")

    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main_mod, "_tui_need_npm_install", lambda _tui_dir: False)
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(main_mod.shutil, "which", lambda bin_name: f"/usr/bin/{bin_name}")

    calls = []

    def fake_run(cmd, cwd=None, **_kwargs):
        calls.append((cmd, cwd))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    argv, cwd = main_mod._make_tui_argv(tui_dir, tui_dev=True)

    assert argv == [str(tsx), "src/entry.tsx"]
    assert cwd == tui_dir
    assert calls == [(["/usr/bin/npm", "run", "build"], str(ink_dir))]
