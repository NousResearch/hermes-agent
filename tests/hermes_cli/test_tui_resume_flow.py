from argparse import Namespace
import os
from pathlib import Path
import subprocess
import sys
import tempfile

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


def test_launch_tui_exports_model_provider_and_toolsets(monkeypatch, main_mod):
    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
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


@pytest.mark.parametrize("flag", ["HERMES_SAFE_MODE", "HERMES_IGNORE_USER_CONFIG"])
def test_launch_tui_bypass_without_model_refuses_before_the_tui_starts(monkeypatch, main_mod, capsys, flag):
    """`hermes --tui --safe-mode` (or an exported HERMES_IGNORE_USER_CONFIG=1) reads no profile
    default model: the owner refuses session.create with bare invalid_params, so the launcher must
    say "pass --model" before spawning the TUI, as classic chat does."""
    monkeypatch.setenv(flag, "1")
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.setenv("HERMES_PYTHON", sys.executable)  # never resolve a python over the real PATH
    monkeypatch.setattr(main_tui_launch, "_make_tui_argv",
                        lambda tui_dir, tui_dev: (["node", "dist/entry.js"], Path(".")))
    spawned = []
    monkeypatch.setattr(main_mod.subprocess, "call", lambda argv, cwd=None, env=None: spawned.append(env) or 0)
    with pytest.raises(SystemExit) as exc:
        main_mod._launch_tui()
    assert exc.value.code == 1 and spawned == []
    assert "pass --model explicitly" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main_mod._launch_tui(model="stub-model")
    assert spawned and spawned[-1][flag] == "1" and spawned[-1]["HERMES_MODEL"] == "stub-model"


@pytest.fixture
def tui_scratch_tmp(monkeypatch, tmp_path):
    """mkstemp lands here, so a test sees an active-session file a launch leaves behind. TMPDIR alone is
    not enough: conftest caches tempfile.tempdir."""
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    return scratch


@pytest.mark.parametrize("worktree", [False, True])
def test_launch_tui_reports_pm_refusing_node_and_leaves_nothing_behind(
        monkeypatch, main_mod, tmp_path, tui_scratch_tmp, capsys, worktree):
    """Lazy installs are off (conftest) and PM's store is empty, so PM itself refuses node. The launch
    prints PM's reason and remedy instead of a traceback, before the active-session file or a
    --worktree exist."""
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "empty-store"))
    monkeypatch.delenv("HERMES_NODE", raising=False)
    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(main_tui_launch, "_find_bundled_tui", lambda: tmp_path / "entry.js")
    monkeypatch.setattr(main_tui_launch, "_setup_tui_worktree",
                        lambda: pytest.fail("the worktree must not exist before node is known"))
    monkeypatch.setattr(main_mod.subprocess, "call", lambda *args, **kwargs: pytest.fail("the TUI must not start"))
    with pytest.raises(SystemExit) as exc:
        main_mod._launch_tui(worktree=worktree)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Hermes could not prepare the TUI: node: not installed and lazy installs are disabled" in err
    assert "hermes pm install" in err and "hermes --cli" in err
    assert not list(tui_scratch_tmp.glob("hermes-tui-active-session-*"))


def test_launch_tui_reports_a_failed_build_step_without_traceback(
        monkeypatch, main_mod, tui_scratch_tmp, capsys):
    """A source checkout whose dependency step fails (with lazy installs off, node-deps.mjs refuses stale
    dependencies) ends in the same short report instead of a CalledProcessError traceback."""
    from hermes_cli import source_build

    def stale(project_root, *, env):
        raise subprocess.CalledProcessError(1, ["node", str(project_root / "scripts/build/node-deps.mjs")])

    monkeypatch.delenv("HERMES_TUI_DIR", raising=False)
    monkeypatch.setattr(main_tui_launch, "_find_bundled_tui", lambda: None)
    monkeypatch.setattr(main_tui_launch, "_ensure_tui_workspace", lambda tui_dir: None)
    monkeypatch.setattr(main_tui_launch, "_tui_need_rebuild", lambda tui_dir: True)
    monkeypatch.setattr(source_build, "source_build_env", lambda *args, **kwargs: {"PATH": os.environ.get("PATH", "")})
    monkeypatch.setattr(source_build, "prepare_launch_dependencies", stale)
    monkeypatch.setattr(main_mod.subprocess, "call", lambda *args, **kwargs: pytest.fail("the TUI must not start"))
    with pytest.raises(SystemExit) as exc:
        main_mod._launch_tui()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Hermes could not prepare the TUI: a build step failed with exit status 1" in err
    assert "hermes --cli" in err
    assert not list(tui_scratch_tmp.glob("hermes-tui-active-session-*"))
