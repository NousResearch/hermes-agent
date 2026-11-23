"""A profile a plugin runs elsewhere starts none of its own commands on this host.

#156725's refusals cover the commands that run a turn. Some host commands start a command the
profile's own files name without building an agent: ``hermes hooks test`` / ``hooks doctor`` (a
shell hook) and ``hermes mcp test`` / ``mcp add`` (an MCP server). The backend the plugin runs
writes those files, so for a marked profile none of them starts here. The four host features that
start such a command (``tools.hosted_tool_policy.RUNS_PROFILE_COMMANDS``: shell hooks, MCP stdio
servers, cron scripts, inline skill shell) share one check, at the place each already asks whether
it may run in a hosted scope, and the MCP probe refuses before it opens any connection. The
backend itself (marker folder denied or absent) runs all of them as before.

The stand-in commands here only append a line to a file under ``tmp_path``.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import shell_hooks
from hermes_cli import hooks as hooks_cli
from hermes_cli import profiles_run_elsewhere as pre
from tools import hosted_tool_policy as policy

PLUGIN = "sandbox-plugin"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A Hermes root with the named profile ``ops`` as this process's home (not marked yet)."""
    root = tmp_path / ".hermes"
    ops = root / "profiles" / "ops"
    ops.mkdir(parents=True)
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    (ops / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(ops))
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)
    shell_hooks.reset_for_tests()
    yield root
    shell_hooks.reset_for_tests()


def _mark(root: Path, profile: str = "ops") -> None:
    pre.mark_profile_run_elsewhere(profile, PLUGIN, root=root)


def _program(folder: Path, name: str = "probe", says: str = "") -> tuple[Path, Path]:
    """A program the profile's backend could have written: it appends a line to ``<name>.ran``
    (and prints *says*)."""
    ran = folder / f"{name}.ran"
    if os.name == "nt":
        program = folder / f"{name}.cmd"
        program.write_text(f'@echo off\r\necho ran>>"{ran}"\r\n' + (f"echo {says}\r\n" if says else ""),
                           encoding="utf-8")
    else:
        program = folder / name
        program.write_text(f'#!/bin/sh\necho ran >> "{ran}"\n' + (f"echo {says}\n" if says else ""),
                           encoding="utf-8")
        program.chmod(0o755)
    return program, ran


def _deny_marker_folder(monkeypatch) -> None:
    """The view from inside the backend the plugin runs: the marker folder cannot be listed."""
    real_listdir = os.listdir

    def listdir(path):
        if Path(path).name == pre.RUN_ELSEWHERE_DIRNAME:
            raise PermissionError(13, "Permission denied", str(path))
        return real_listdir(path)

    monkeypatch.setattr(pre.os, "listdir", listdir)


# -- shell hooks -----------------------------------------------------------------------------------

def _hooks_test(program: Path) -> str:
    config = {"hooks": {"on_session_start": [{"command": str(program)}]}, "hooks_auto_accept": True}
    out = io.StringIO()
    with patch("hermes_cli.config.load_config", return_value=config), redirect_stdout(out):
        hooks_cli.hooks_command(SimpleNamespace(hooks_action="test", event="on_session_start",
                                                for_tool=None, payload_file=None))
    return out.getvalue()


def test_hooks_test_runs_the_hook_of_a_profile_that_runs_here(home, tmp_path):
    program, ran = _program(tmp_path)
    out = _hooks_test(program)
    assert ran.exists(), out
    assert "runs elsewhere" not in out


def test_hooks_test_does_not_start_a_marked_profiles_hook(home, tmp_path):
    program, ran = _program(tmp_path)
    _mark(home)
    out = _hooks_test(program)
    assert not ran.exists(), "a hook of a profile a plugin runs elsewhere ran on this host"
    assert "'ops'" in out and PLUGIN in out and "runs elsewhere" in out


def test_run_once_refuses_a_marked_profile_without_starting_a_process(home, tmp_path, monkeypatch):
    """``hooks doctor`` and every other caller go through the one subprocess site."""
    program, ran = _program(tmp_path)
    _mark(home)
    monkeypatch.setattr(shell_hooks.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("a process was started for a marked profile's hook"))
    result = shell_hooks.run_once(shell_hooks.ShellHookSpec(event="on_session_start", command=str(program)),
                                  {"session_id": "s"})
    assert "runs elsewhere" in str(result.get("error")) and not ran.exists()


def test_a_registered_hook_does_not_fire_once_its_profile_is_marked(home, tmp_path):
    """A hook registered before the profile was marked (a process already running): the callback
    asks at every event."""
    program, ran = _program(tmp_path)
    callback = shell_hooks._make_callback(shell_hooks.ShellHookSpec(event="on_session_start", command=str(program)))
    callback(session_id="s")
    assert ran.read_text(encoding="utf-8").count("ran") == 1
    _mark(home)
    assert callback(session_id="s") is None
    assert ran.read_text(encoding="utf-8").count("ran") == 1


def test_the_backend_the_plugin_runs_still_runs_its_hook(home, tmp_path, monkeypatch):
    program, ran = _program(tmp_path)
    _mark(home)
    _deny_marker_folder(monkeypatch)
    out = _hooks_test(program)
    assert ran.exists(), out


# -- the one check behind the four features --------------------------------------------------------

_FEATURES = ("cron_scripts", "mcp_stdio", "shell_hooks", "skill_inline_shell")


@pytest.mark.parametrize("feature", _FEATURES)
def test_a_feature_that_starts_a_profiles_command_is_off_for_a_marked_profile(home, feature):
    assert policy.host_feature_enabled(feature) and policy.host_feature_refusal(feature, "It") is None
    _mark(home)
    assert not policy.host_feature_enabled(feature)
    refusal = policy.host_feature_refusal(feature, "It")
    assert refusal == pre.run_elsewhere_for_home(home / "profiles" / "ops").message()
    with pytest.raises(policy.HostedToolRefused, match="runs elsewhere"):
        policy.require_host_feature(feature, "It")
    assert pre.clear_profile_run_elsewhere("ops", PLUGIN, root=home)
    assert policy.host_feature_enabled(feature)


def test_the_features_are_the_ones_that_start_a_command(home):
    assert policy.RUNS_PROFILE_COMMANDS == set(_FEATURES)
    assert policy.RUNS_PROFILE_COMMANDS <= policy.HOST_FEATURES


def test_the_other_host_features_stay_on_for_a_marked_profile(home):
    """They read or write the profile's state inside an agent's turn, and no agent is built for a
    marked profile; this check is for the commands that start a process without one."""
    _mark(home)
    for feature in sorted(policy.HOST_FEATURES - set(_FEATURES)):
        assert policy.host_feature_enabled(feature), feature
    assert policy.hosted_tool_refusal("terminal") is None


def test_an_unreadable_marker_folder_starts_no_profiles_command(home):
    """The rule's fail-closed case: the host cannot tell which profiles run elsewhere."""
    pre.run_elsewhere_dir(home).write_text("not a folder", encoding="utf-8")
    refusal = policy.host_feature_refusal("shell_hooks", "A shell hook")
    assert refusal is not None and "cannot read" in refusal
    pre.run_elsewhere_dir(home).unlink()
    assert policy.host_feature_enabled("shell_hooks")


def test_the_backend_the_plugin_runs_keeps_the_four_features(home, monkeypatch):
    _mark(home)
    _deny_marker_folder(monkeypatch)
    for feature in _FEATURES:
        assert policy.host_feature_enabled(feature), feature


def test_a_marked_profiles_cron_script_and_inline_skill_shell_do_not_start(home, tmp_path, monkeypatch):
    from agent import skill_preprocessing
    from cron import scheduler_script

    _mark(home)
    monkeypatch.setattr(skill_preprocessing.subprocess, "run",
                        lambda *a, **k: pytest.fail("inline skill shell started for a marked profile"))
    assert "runs elsewhere" in skill_preprocessing.run_inline_shell("echo hi", tmp_path, 5)
    ok, output = scheduler_script._run_job_script("job.py")
    assert ok is False and "runs elsewhere" in output


def test_a_marked_profiles_key_cmd_is_not_run(home, tmp_path):
    from agent import command_token_source

    program, ran = _program(tmp_path, "key_cmd", says="tok")
    source = command_token_source.build_command_token_provider(str(program), "myprov")
    _mark(home)
    with pytest.raises(command_token_source.CommandTokenError, match="runs elsewhere"):
        source()
    assert command_token_source.materialize_probe_api_key(source) == ""
    assert not ran.exists()
    assert pre.clear_profile_run_elsewhere("ops", PLUGIN, root=home)
    assert source() == "tok" and ran.exists()


def test_a_marked_profiles_speech_command_is_not_run(home, tmp_path):
    """``tts.providers.<name>.command`` and ``stt.providers.<name>.command`` are shell command lines in
    the profile's own ``config.yaml``, and one runner starts both. The dashboard's speak and transcribe
    requests reach it for a named profile without building an agent."""
    from tools import tts_command_provider

    program, ran = _program(tmp_path, "speech")
    _mark(home)
    with pytest.raises(RuntimeError, match="runs elsewhere"):
        tts_command_provider.run_command_provider(str(program), 30)
    assert not ran.exists()
    assert pre.clear_profile_run_elsewhere("ops", PLUGIN, root=home)
    assert tts_command_provider.run_command_provider(str(program), 30).returncode == 0 and ran.exists()


# -- `hermes secrets <source> <verb>` -----------------------------------------------------------------

def _secrets_verb(*argv: str) -> argparse.Namespace:
    from hermes_cli.subcommands.secrets import build_secrets_parser

    parser = argparse.ArgumentParser()
    build_secrets_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["secrets", *argv])


def test_secrets_verbs_start_no_program_for_a_marked_profile(home, tmp_path, capsys):
    """``secrets onepassword status`` runs the program ``secrets.onepassword.binary_path`` names
    (``--version``, ``whoami``), and ``sync`` reads every reference with it: config.yaml again."""
    program, ran = _program(tmp_path, "op", says="stand-in 1.0")
    (home / "profiles" / "ops" / "config.yaml").write_text(json.dumps({"secrets": {"onepassword": {
        "enabled": True, "binary_path": str(program), "env": {"MH2_OP_KEY": "op://vault/item/field"}}}}),
        encoding="utf-8")
    status = _secrets_verb("onepassword", "status")
    status.func(status)
    assert ran.exists(), "control: for a profile that runs here the verb runs the program"
    ran.unlink()
    _mark(home)
    capsys.readouterr()
    for verb in (("onepassword", "status"), ("onepassword", "sync"), ("bitwarden", "status"), ("bitwarden", "sync")):
        args = _secrets_verb(*verb)
        with pytest.raises(SystemExit) as refused:
            args.func(args)
        assert refused.value.code == 1, verb
        assert "runs elsewhere" in capsys.readouterr().err, verb
    assert not ran.exists(), "a secrets verb ran a marked profile's program on this host"


# -- `hermes send` -----------------------------------------------------------------------------------

def test_send_loads_and_sends_nothing_for_a_marked_profile(home, monkeypatch, capsys):
    """``hermes send`` messages a platform with the home's credentials, and has its own loader for
    the home's ``.env``: for a marked profile it is refused before that loader runs."""
    from hermes_cli import send_cmd

    monkeypatch.delenv("MH2_SEND_TOKEN", raising=False)
    (home / "profiles" / "ops" / ".env").write_text("MH2_SEND_TOKEN=from-the-home\n", encoding="utf-8")
    try:
        send_cmd._load_hermes_env()
        assert os.environ.pop("MH2_SEND_TOKEN", None) == "from-the-home", "control: a profile that runs here"
        _mark(home)
        with pytest.raises(SystemExit) as refused:
            send_cmd.cmd_send(argparse.Namespace(list_targets=True, message=None, json=False))
        assert refused.value.code == 1 and "runs elsewhere" in capsys.readouterr().err
        assert "MH2_SEND_TOKEN" not in os.environ
    finally:
        os.environ.pop("MH2_SEND_TOKEN", None)


# -- MCP servers -----------------------------------------------------------------------------------

def _probe(config: dict) -> BaseException | None:
    pytest.importorskip("mcp")
    from hermes_cli.mcp_config import _probe_single_server

    try:
        _probe_single_server("probe", config, connect_timeout=5)
    except BaseException as exc:  # the stand-in is no MCP server: every outcome is a failure to connect
        return exc
    return None


def test_mcp_probe_starts_the_stdio_command_of_a_profile_that_runs_here(home, tmp_path):
    program, ran = _program(tmp_path, "mcp-server")
    _probe({"command": str(program), "args": []})
    assert ran.exists()


def test_mcp_probe_does_not_start_a_marked_profiles_stdio_command(home, tmp_path):
    program, ran = _program(tmp_path, "mcp-server")
    _mark(home)
    exc = _probe({"command": str(program), "args": []})
    assert not ran.exists(), "an MCP stdio command of a profile a plugin runs elsewhere ran on this host"
    assert isinstance(exc, pre.ProfileRunsElsewhere) and "runs elsewhere" in str(exc)


def test_mcp_probe_opens_no_connection_for_a_marked_profile(home, monkeypatch):
    """An HTTP server's address comes from the same file: no request goes to it either."""
    from tools import mcp_tool_discovery

    _mark(home)
    monkeypatch.setattr(mcp_tool_discovery, "_connect_server",
                        lambda *a, **k: pytest.fail("a connection was started for a marked profile's MCP server"))
    exc = _probe({"url": "http://127.0.0.1:9/mcp"})
    assert isinstance(exc, pre.ProfileRunsElsewhere)


def test_mcp_login_clears_no_token_and_calls_no_server_for_a_marked_profile(home, monkeypatch, capsys):
    """``mcp login`` / ``reauth`` drop the stored tokens, and the device flow asks the entry's server
    for a code, before the probe would refuse."""
    pytest.importorskip("mcp")
    from hermes_cli import mcp_config
    from tools import mcp_oauth, mcp_oauth_device

    did: list = []

    async def login_device(*args, **kwargs):
        did.append("device login")

    monkeypatch.setattr(mcp_oauth_device, "login_device", login_device)
    monkeypatch.setattr(mcp_oauth.HermesTokenStorage, "remove", lambda self, **kwargs: did.append("tokens removed"))
    monkeypatch.setattr(mcp_config, "_probe_single_server", lambda *args, **kwargs: did.append("probe") or [])
    entry = {"url": "http://127.0.0.1:9/mcp", "auth": "oauth"}
    _mark(home)
    for flow in ("device", "browser"):
        assert mcp_config._reauth_oauth_server("probe", entry, flow=flow) is False
    assert did == [] and capsys.readouterr().out.count("runs elsewhere") == 2


@pytest.mark.parametrize("server", ["stdio", "http"])
def test_mcp_add_asks_and_saves_nothing_for_a_marked_profile(home, tmp_path, monkeypatch, capsys, server):
    """``mcp add`` is refused before its first question. Left to the probe, the refusal reads as a
    failed connection and ``mcp add`` offers to save the entry anyway; for an HTTP server it has by
    then asked for a token and written it to the home's ``.env``."""
    pytest.importorskip("mcp")
    from hermes_cli import cli_output, mcp_config

    program, ran = _program(tmp_path, "mcp-server")
    asked: list = []
    monkeypatch.setattr(mcp_config, "_confirm", lambda question, default=True: asked.append(question) or True)
    monkeypatch.setattr(cli_output, "prompt", lambda *args, **kwargs: asked.append(args[0]) or "a-token")
    ops = home / "profiles" / "ops"
    _mark(home)
    before = {path.name: path.read_bytes() for path in ops.iterdir() if path.is_file()}
    fields = {"mcp_command": str(program)} if server == "stdio" else {"url": "http://127.0.0.1:9/mcp", "auth": "header"}
    args = argparse.Namespace(**{"name": "probe", "url": None, "mcp_command": None, "args": [], "auth": None,
                                 "connect_timeout": None, "env": None, "preset": None, **fields})
    with pytest.raises(SystemExit) as refused:
        mcp_config.cmd_mcp_add(args)
    assert refused.value.code == 1 and "runs elsewhere" in capsys.readouterr().err
    assert asked == [] and not ran.exists()
    assert {path.name: path.read_bytes() for path in ops.iterdir() if path.is_file()} == before


def test_the_mcp_transport_itself_starts_no_stdio_command_for_a_marked_profile(home, tmp_path, monkeypatch):
    """Whoever asks for the connection: with the probe's own early check out of the way, the
    transport still starts nothing."""
    program, ran = _program(tmp_path, "mcp-server")
    _mark(home)
    monkeypatch.setattr(pre, "require_home_runs_here", lambda home=None: None)
    exc = _probe({"command": str(program), "args": []})
    assert exc is not None and not ran.exists(), "the transport started a marked profile's MCP stdio command"
