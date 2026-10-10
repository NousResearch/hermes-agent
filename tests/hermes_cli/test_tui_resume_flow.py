from argparse import Namespace
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types

import pytest
from hermes_cli import main_tui_launch


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
    assert result.stdout in (b"ok\n", b"ok\r\n")
    # Don't demand byte-empty stderr — an import-time warning from the heavy
    # CLI import chain shouldn't fail this. What matters is no crash traceback.
    assert b"Traceback" not in result.stderr


def _stub_plugin_discovery(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.plugins",
        types.SimpleNamespace(discover_plugins=lambda: None),
    )


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

    def fake_acquire(db_path=None):
        return sentinel_db

    def mod(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=FakeAgent))
    # Oneshot borrows the process-shared registry handle (one writer per state.db path).
    monkeypatch.setitem(sys.modules, "hermes_state_registry", mod("hermes_state_registry", acquire=fake_acquire))
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
            resolve_runtime_with_fallback=lambda _cfg, **_kwargs: (
                {
                    "api_key": "k",
                    "base_url": "u",
                    "provider": "p",
                    "api_mode": "chat_completions",
                    "credential_pool": None,
                },
                None,
            ),
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


@pytest.mark.parametrize("run_fails", [False, True], ids=["success", "failure"])
def test_oneshot_closes_its_relay_root_before_agent_teardown(monkeypatch, run_fails):
    """hermes -z hard-exits past atexit, so _run_agent itself must finalize the Relay root it opened:
    keyed to the id at turn entry (compression may rotate it), on failure too, before agent.close()."""
    from hermes_cli import lifecycle
    from hermes_cli.oneshot import _run_agent

    events = []

    class FakeAgent:
        def __init__(self, **_kwargs):
            self.session_id = "entry-id"
            self.platform = "cli"
            self.suppress_status_output = False
            self.stream_delta_callback = self.tool_gen_callback = object()

        def run_conversation(self, _prompt, **_kwargs):
            self.session_id = "compressed-child-id"
            if run_fails:
                raise RuntimeError("agent failed")
            return {"final_response": "ok", "failed": False, "partial": False}

        def shutdown_memory_provider(self, *_args):
            events.append("memory")

        def close(self):
            events.append("agent_close")

    def mod(name, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(sys.modules, "hermes_state_registry", mod("hermes_state_registry", acquire=lambda db_path=None: None))
    monkeypatch.setitem(sys.modules, "hermes_cli.config", mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m"}}))
    monkeypatch.setitem(sys.modules, "hermes_cli.models",
                        mod("hermes_cli.models", detect_provider_for_model=lambda *_a, **_k: None))
    monkeypatch.setitem(sys.modules, "hermes_cli.runtime_provider", mod(
        "hermes_cli.runtime_provider",
        resolve_runtime_with_fallback=lambda _cfg, **_k: ({"api_key": "k", "base_url": "u", "provider": "p",
                                                           "api_mode": "chat_completions", "credential_pool": None}, None),
    ))
    monkeypatch.setitem(sys.modules, "hermes_cli.tools_config",
                        mod("hermes_cli.tools_config", _get_platform_tools=lambda *_a, **_k: set()))
    monkeypatch.setattr(lifecycle, "finalize_session", lambda **kw: events.append(("finalize", kw)))

    if run_fails:
        with pytest.raises(RuntimeError, match="agent failed"):
            _run_agent("finish this")
    else:
        assert _run_agent("finish this")[0] == "ok"
    assert events == [("finalize", {"session_id": "entry-id", "platform": "cli", "reason": "shutdown"}),
                      "memory", "agent_close"]


def test_launch_tui_exports_model_provider_and_toolsets(monkeypatch, main_mod):
    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    monkeypatch.delenv("NODE_ENV", raising=False)
    monkeypatch.delenv("HERMES_TUI_INJECTED_NODE_ENV", raising=False)
    captured = {}
    active_path_during_call = None

    monkeypatch.setattr(main_tui_launch, "_make_tui_argv",
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
    assert env["HERMES_TUI_INJECTED_NODE_ENV"] == "1"


def test_launch_tui_preserves_caller_owned_node_env(monkeypatch, main_mod):
    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    monkeypatch.setenv("NODE_ENV", "staging")
    monkeypatch.delenv("HERMES_TUI_INJECTED_NODE_ENV", raising=False)
    captured = {}
    monkeypatch.setattr(
        main_tui_launch,
        "_make_tui_argv",
        lambda tui_dir, tui_dev: (["node", "dist/entry.js"], Path(".")),
    )
    monkeypatch.setattr(
        main_mod.subprocess,
        "call",
        lambda argv, cwd=None, env=None: captured.update({"env": env}) or 1,
    )

    with pytest.raises(SystemExit):
        main_mod._launch_tui()

    assert captured["env"]["NODE_ENV"] == "staging"
    assert "HERMES_TUI_INJECTED_NODE_ENV" not in captured["env"]


def test_launch_tui_prefers_launch_cwd_over_inherited_hermes_cwd(monkeypatch, main_mod, tmp_path):
    """The directory `hermes --tui` was run from outranks an inherited HERMES_CWD.

    A shell export - or an outer `hermes --tui` - leaves HERMES_CWD naming a real but stale
    directory, and ui-tui/src/gatewayClient.ts:478 starts the gateway in whatever it names,
    so the session reads files and completions from the wrong project (#49637).
    """
    stale = tmp_path / "stale-project"
    launch = tmp_path / "launch-project"
    stale.mkdir()
    launch.mkdir()
    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    monkeypatch.setenv("HERMES_CWD", str(stale))
    monkeypatch.chdir(launch)

    captured = {}
    monkeypatch.setattr(main_tui_launch, "_make_tui_argv",
        lambda tui_dir, tui_dev: (["node", "dist/entry.js"], Path(".")),
    )
    monkeypatch.setattr(main_mod.subprocess, "call",
        lambda argv, cwd=None, env=None: captured.update({"env": env}) or 1,
    )

    with pytest.raises(SystemExit):
        main_mod._launch_tui()

    handed_to_tui = Path(captured["env"]["HERMES_CWD"]).resolve()
    assert handed_to_tui == launch.resolve(), "the TUI gateway must start where the user launched"
    assert handed_to_tui != stale.resolve()


def test_launch_tui_worktree_still_outranks_the_launch_cwd(monkeypatch, main_mod, tmp_path):
    """`--worktree` names an explicit destination, so it keeps precedence over the launch cwd."""
    worktree = tmp_path / "worktree"
    launch = tmp_path / "launch-project"
    worktree.mkdir()
    launch.mkdir()
    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    monkeypatch.setenv("HERMES_CWD", str(tmp_path))
    monkeypatch.chdir(launch)

    captured = {}
    monkeypatch.setattr(main_tui_launch, "_setup_tui_worktree", lambda: {"path": str(worktree)})
    monkeypatch.setattr(main_tui_launch, "_make_tui_argv",
        lambda tui_dir, tui_dev: (["node", "dist/entry.js"], Path(".")),
    )
    monkeypatch.setattr(main_mod.subprocess, "call",
        lambda argv, cwd=None, env=None: captured.update({"env": env}) or 1,
    )

    with pytest.raises(SystemExit):
        main_mod._launch_tui(worktree=True)

    assert captured["env"]["HERMES_CWD"] == str(worktree)
    assert captured["env"]["TERMINAL_CWD"] == str(worktree)
    assert Path(captured["env"]["HERMES_CWD"]).resolve() != launch.resolve()


@pytest.mark.parametrize("backend", ["local", "docker"])
def test_launch_tui_local_session_starts_in_launch_dir_not_terminal_cwd(monkeypatch, main_mod, tmp_path, backend):
    """A local TUI follows the classic CLI rule: the launch dir beats an absolute terminal.cwd (#84015).

    Remote backends keep terminal.cwd: the launch dir names nothing on the sandbox.
    """
    configured = tmp_path / "configured-home"
    launch = tmp_path / "launch-project"
    configured.mkdir()
    launch.mkdir()
    (Path(os.environ["HERMES_HOME"]) / "config.yaml").write_text(
        f"terminal:\n  backend: {backend}\n  cwd: {configured}\n", encoding="utf-8")
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    monkeypatch.setenv("HERMES_TUI_CWD", str(tmp_path))  # stale, from an outer launcher
    monkeypatch.chdir(launch)

    captured = {}
    monkeypatch.setattr(main_tui_launch, "_make_tui_argv",
        lambda tui_dir, tui_dev: (["node", "dist/entry.js"], Path(".")),
    )
    monkeypatch.setattr(main_mod.subprocess, "call",
        lambda argv, cwd=None, env=None: captured.update({"env": env}) or 1,
    )

    with pytest.raises(SystemExit):
        main_mod._launch_tui()

    env = captured["env"]
    if backend == "local":
        assert Path(env["HERMES_TUI_CWD"]).resolve() == launch.resolve()
        assert Path(env["TERMINAL_CWD"]).resolve() == launch.resolve()
    else:
        assert "HERMES_TUI_CWD" not in env
        assert Path(env["TERMINAL_CWD"]).resolve() == configured.resolve()
