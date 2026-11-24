"""A profile a plugin runs elsewhere runs none of its quick commands on this host.

A ``quick_commands`` entry of ``type: exec`` is a shell command in the profile's ``config.yaml``,
a file the backend the plugin runs writes. Three places start one when its name is typed as a slash
command: the classic CLI, the gateway (messaging and local sessions) and the TUI / dashboard
backend. #156725 refuses to start those processes for a marked profile, but one that was already
running when the profile was marked still answers slash commands, and a backend that serves
several profiles takes the profile with the request. So each place asks before it starts the
command, and answers with the run-elsewhere message instead.

The stand-in command only appends a line to a file under ``tmp_path``.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import profiles_run_elsewhere as pre
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

PLUGIN = "sandbox-plugin"


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A Hermes root with the named profile ``ops``; this process's home is the default profile."""
    root = tmp_path / ".hermes"
    (root / "profiles" / "ops").mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    (root / "profiles" / "ops" / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    return root


@pytest.fixture
def ops(root, monkeypatch):
    """The same root, with ``ops`` as this process's home (not marked yet)."""
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))
    return root / "profiles" / "ops"


def _program(folder: Path) -> tuple[str, Path]:
    """A command the profile's backend could have written: it appends a line to ``probe.ran``."""
    ran = folder / "probe.ran"
    if os.name == "nt":
        program = folder / "probe.cmd"
        program.write_text(f'@echo off\r\necho ran>>"{ran}"\r\necho quick-ran\r\n', encoding="utf-8")
    else:
        program = folder / "probe"
        program.write_text(f'#!/bin/sh\necho ran >> "{ran}"\necho quick-ran\n', encoding="utf-8")
        program.chmod(0o755)
    return f'"{program}"', ran


def _deny_marker_folder(monkeypatch) -> None:
    """The view from inside the backend the plugin runs: the marker folder cannot be listed."""
    real_listdir = os.listdir

    def listdir(path):
        if Path(path).name == pre.RUN_ELSEWHERE_DIRNAME:
            raise PermissionError(13, "Permission denied", str(path))
        return real_listdir(path)

    monkeypatch.setattr(pre.os, "listdir", listdir)


# -- the TUI / dashboard backend (command.dispatch) -------------------------------------------------

def _tui_dispatch(monkeypatch, command: str, entry_type: str = "exec") -> dict:
    from tui_gateway import server

    entry = {"type": entry_type, "command": command, "target": "/help"}
    monkeypatch.setattr(server, "_load_cfg", lambda: {"quick_commands": {"probe": entry}})
    return server._methods["command.dispatch"](1, {"name": "probe"})


def test_tui_backend_runs_a_quick_command_of_a_profile_that_runs_here(ops, tmp_path, monkeypatch):
    command, ran = _program(tmp_path)

    reply = _tui_dispatch(monkeypatch, command)

    assert reply["result"] == {"type": "exec", "output": "quick-ran"}
    assert ran.exists()


def test_tui_backend_refuses_a_quick_command_of_a_marked_home(ops, tmp_path, monkeypatch):
    command, ran = _program(tmp_path)
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=ops.parent.parent)

    reply = _tui_dispatch(monkeypatch, command)

    assert not ran.exists()
    assert "runs elsewhere" in reply["error"]["message"] and PLUGIN in reply["error"]["message"]


def test_tui_backend_refuses_a_quick_command_of_a_marked_profile_bound_for_the_request(root, tmp_path, monkeypatch):
    """A backend on another profile's home that serves ``ops`` for one request (``profile`` param)."""
    command, ran = _program(tmp_path)
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)
    token = set_hermes_home_override(str(root / "profiles" / "ops"))
    try:
        reply = _tui_dispatch(monkeypatch, command)
    finally:
        reset_hermes_home_override(token)

    assert not ran.exists()
    assert "runs elsewhere" in reply["error"]["message"]


def test_tui_backend_still_answers_an_alias_of_a_marked_home(ops, tmp_path, monkeypatch):
    """An alias starts nothing: it names another slash command, which meets its own checks."""
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=ops.parent.parent)

    reply = _tui_dispatch(monkeypatch, "", entry_type="alias")

    assert reply["result"] == {"type": "alias", "target": "/help"}


def test_the_backend_the_plugin_runs_still_runs_its_quick_commands(ops, tmp_path, monkeypatch):
    command, ran = _program(tmp_path)
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=ops.parent.parent)
    _deny_marker_folder(monkeypatch)

    reply = _tui_dispatch(monkeypatch, command)

    assert reply["result"]["output"] == "quick-ran"
    assert ran.exists()


# -- the gateway (messaging and local sessions) -----------------------------------------------------

def _gateway_exec(command: str) -> str:
    from gateway.run_inbound_hooks import GatewayInboundHooksMixin

    return asyncio.run(GatewayInboundHooksMixin._hm_run_exec_quick_command(SimpleNamespace(), "probe", command))


def test_gateway_runs_a_quick_command_of_a_profile_that_runs_here(ops, tmp_path):
    command, ran = _program(tmp_path)

    assert _gateway_exec(command) == "quick-ran"
    assert ran.exists()


def test_gateway_refuses_a_quick_command_of_a_marked_home(ops, tmp_path):
    command, ran = _program(tmp_path)
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=ops.parent.parent)

    reply = _gateway_exec(command)

    assert not ran.exists()
    assert "runs elsewhere" in reply and PLUGIN in reply


# -- the classic CLI --------------------------------------------------------------------------------

def _cli_exec(command: str) -> list[str]:
    from cli import HermesCLI

    printed: list[str] = []
    cli = SimpleNamespace(_console_print=lambda text, *a, **k: printed.append(str(text)))
    assert HermesCLI._run_quick_command(cli, "/probe", {"type": "exec", "command": command}, "") is True
    return printed


def test_cli_runs_a_quick_command_of_a_profile_that_runs_here(ops, tmp_path):
    command, ran = _program(tmp_path)

    assert any("quick-ran" in line for line in _cli_exec(command))
    assert ran.exists()


def test_cli_refuses_a_quick_command_of_a_marked_home(ops, tmp_path):
    command, ran = _program(tmp_path)
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=ops.parent.parent)

    printed = _cli_exec(command)

    assert not ran.exists()
    assert any("runs elsewhere" in line for line in printed)


# -- the home asked is the one whose config.yaml names the command -----------------------------------

def _plant(home: Path, command: str) -> None:
    """``command`` as the quick command ``probe`` in *home*'s own ``config.yaml``."""
    entry = {"quick_commands": {"probe": {"type": "exec", "command": command}}}
    (home / "config.yaml").write_text(json.dumps(entry), encoding="utf-8")


def test_tui_backend_refuses_a_marked_profile_the_request_names(root, tmp_path, monkeypatch):
    """A dashboard or ``hermes serve`` backend running as the default profile, asked for ``ops`` with
    the request's ``profile``: the command is read from ``ops``' config, and ``ops`` is the home asked."""
    from tui_gateway import launch_profile_policy, server

    monkeypatch.setattr(launch_profile_policy, "activate_multi_profile_hosting", lambda: None)
    monkeypatch.setattr(server, "_hermes_home", root)
    monkeypatch.setattr(server, "_served_profile_homes", set())
    command, ran = _program(tmp_path)
    _plant(root / "profiles" / "ops", command)
    dispatch = server._methods["command.dispatch"]

    assert dispatch(1, {"name": "probe", "profile": "ops"})["result"]["output"] == "quick-ran"
    ran.unlink()
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)
    reply = dispatch(2, {"name": "probe", "profile": "ops"})

    assert not ran.exists()
    assert "runs elsewhere" in reply["error"]["message"] and PLUGIN in reply["error"]["message"]


def test_a_process_asks_again_at_every_quick_command(ops, tmp_path, monkeypatch):
    """One process, three commands: before the profile is marked, while it is, after it is cleared."""
    command, ran = _program(tmp_path)
    root = ops.parent.parent

    assert _tui_dispatch(monkeypatch, command)["result"]["output"] == "quick-ran"
    ran.unlink()
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)
    assert "runs elsewhere" in _tui_dispatch(monkeypatch, command)["error"]["message"]
    assert not ran.exists()
    pre.clear_profile_run_elsewhere("ops", PLUGIN, root=root)
    assert _tui_dispatch(monkeypatch, command)["result"]["output"] == "quick-ran"


def test_gateway_refuses_a_quick_command_in_a_marked_profiles_session(root, tmp_path):
    """A local session of ``ops`` on a gateway that runs as the default profile: the command was read
    in the session's profile scope and is started there, so ``ops`` is the home asked."""
    from gateway.run import _profile_runtime_scope

    command, ran = _program(tmp_path)
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)

    with _profile_runtime_scope(root / "profiles" / "ops"):
        reply = _gateway_exec(command)

    assert not ran.exists()
    assert "runs elsewhere" in reply and PLUGIN in reply


def _messaging_dispatch(command: str, *, multiplex: bool) -> str:
    """A messaging slash command through the runner's own dispatch, with ``probe`` in its config."""
    from gateway.run_inbound_hooks import GatewayInboundHooksMixin

    runner = GatewayInboundHooksMixin.__new__(GatewayInboundHooksMixin)
    runner.config = SimpleNamespace(
        multiplex_profiles=multiplex, quick_commands={"probe": {"type": "exec", "command": command}})
    runner._draining = False
    runner._check_slash_access = lambda source, name: None
    handled, reply, _ = asyncio.run(runner._hm_dispatch_quick_and_plugin_commands(None, None, "probe"))
    assert handled is True
    return reply


def test_multiplexing_gateway_refuses_the_quick_commands_of_its_marked_default_profile(ops, tmp_path):
    """A multiplexing gateway's own quick commands are the default profile's, whichever profile's
    process started it (``load_gateway_config_for_runner``): the default profile is the home asked."""
    command, ran = _program(tmp_path)
    root = ops.parent.parent

    assert _messaging_dispatch(command, multiplex=True) == "quick-ran"
    ran.unlink()
    pre.mark_profile_run_elsewhere("default", PLUGIN, root=root)
    reply = _messaging_dispatch(command, multiplex=True)

    assert not ran.exists()
    assert "runs elsewhere" in reply and "'default'" in reply


def test_single_profile_gateway_asks_its_own_home_for_its_quick_commands(ops, tmp_path):
    """Without multiplexing the runner's config is its own profile's: a marked ``default`` beside it
    changes nothing, its own marker refuses."""
    command, ran = _program(tmp_path)
    root = ops.parent.parent
    pre.mark_profile_run_elsewhere("default", PLUGIN, root=root)

    assert _messaging_dispatch(command, multiplex=False) == "quick-ran"
    ran.unlink()
    pre.mark_profile_run_elsewhere("ops", PLUGIN, root=root)
    reply = _messaging_dispatch(command, multiplex=False)

    assert not ran.exists()
    assert "runs elsewhere" in reply and "'ops'" in reply
