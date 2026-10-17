"""Plugin TUI frontends: ``ctx.register_tui_frontend`` through real plugin discovery.

A plugin registers a Node bundle; ``hermes chat --tui --frontend <name>`` (and the ``hermes <name>``
shorthand) launches it through the normal TUI launcher, and an exit of 75 runs the built-in TUI once.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_cli import main_tui_launch


@pytest.fixture
def frontend_home(monkeypatch, tmp_path):
    """A HERMES_HOME holding one enabled plugin that registers the ``stub`` frontend."""
    from hermes_cli.plugins import _reset_plugin_managers_for_tests

    home = Path(os.environ["HERMES_HOME"])
    plugin = home / "plugins" / "stub-frontend"
    (plugin / "bundle" / "dist").mkdir(parents=True)
    (plugin / "bundle" / "dist" / "entry.js").write_text("process.exit(0)\n")
    (plugin / "plugin.yaml").write_text("name: stub-frontend\nversion: 0.0.1\n")
    (plugin / "__init__.py").write_text(textwrap.dedent("""
        from pathlib import Path

        def register(ctx):
            ctx.register_tui_frontend("stub", Path(__file__).parent / "bundle", help="Stub frontend")
    """))
    (home / "config.yaml").write_text("plugins:\n  enabled:\n    - stub-frontend\n")
    monkeypatch.setenv("HERMES_PYTHON", sys.executable)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_tui_launch, "_tui_node_bin", lambda _bin: "node")
    _reset_plugin_managers_for_tests()
    yield plugin / "bundle"
    _reset_plugin_managers_for_tests()


def _launch(monkeypatch, exit_codes, **kwargs):
    """Run ``_launch_tui`` with each child exiting the next code; returns (calls, exit code)."""
    calls = []
    codes = iter(exit_codes)
    monkeypatch.setattr(main_tui_launch, "_make_tui_argv",
                        lambda tui_dir, tui_dev: (["node", "builtin/dist/entry.js"], Path("builtin")))
    monkeypatch.setattr(main_tui_launch.subprocess, "call",
                        lambda argv, cwd=None, env=None: calls.append((argv, cwd)) or next(codes))
    with pytest.raises(SystemExit) as done:
        main_tui_launch._launch_tui(**kwargs)
    return calls, done.value.code


def test_registered_frontend_runs_its_bundle_instead_of_the_builtin_tui(monkeypatch, frontend_home):
    calls, code = _launch(monkeypatch, [0], frontend="stub")

    assert code == 0
    assert calls == [(["node", "--expose-gc", str(frontend_home.resolve() / "dist" / "entry.js")],
                      str(frontend_home.resolve()))]


def test_frontend_exit_75_runs_the_builtin_tui_once(monkeypatch, frontend_home, capsys):
    calls, code = _launch(monkeypatch, [75, 0], frontend="stub")

    assert code == 0
    assert [argv[-1] for argv, _cwd in calls] == [
        str(frontend_home.resolve() / "dist" / "entry.js"), "builtin/dist/entry.js"]
    assert "'stub' TUI frontend cannot run in this terminal" in capsys.readouterr().err


def test_unknown_frontend_fails_before_anything_launches(monkeypatch, frontend_home, capsys):
    calls, code = _launch(monkeypatch, [], frontend="nope")

    assert code == 2
    assert calls == []
    err = capsys.readouterr().err
    assert "no TUI frontend named 'nope'" in err and "Registered frontends: stub." in err


def test_shorthand_command_parses_as_a_chat_tui_launch(monkeypatch, frontend_home):
    """``hermes stub -m X`` is ``hermes chat --tui --frontend stub -m X``: same handler, same flags."""
    import hermes_cli.main as main_mod

    monkeypatch.setattr(sys, "argv", ["hermes", "stub", "-m", "test/model"])
    parser, subparsers = main_mod._build_cli_parser()
    args = main_mod._parse_cli_args(parser, subparsers, sys.argv[1:])

    assert (args.command, args.tui, args.tui_frontend, args.model) == ("chat", True, "stub", "test/model")
    assert args.func is main_mod.cmd_chat
