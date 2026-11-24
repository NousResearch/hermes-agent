"""No agent or backend runs on this host for a profile a plugin runs elsewhere.

``AIAgent`` refuses a marked profile before it reads any credential, whichever frontend asks
(``hermes serve`` and the dashboard, the PTY chat, ``/background``, ``hermes-agent``, cron,
curator). CLI commands that run a turn exit with the refusal before any banner, TUI or gateway
contact, and ``hermes serve``/``dashboard`` exit 78 before attaching or binding. The backend the
plugin runs is denied the marker folder, so it still builds its own agents.
"""
from __future__ import annotations

import errno
import os
import sys
import types
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import profiles_run_elsewhere as pre


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A Hermes root (the default profile) with the named profile ``ops`` marked by a plugin."""
    home = tmp_path / ".hermes"
    (home / "profiles" / "ops").mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (home / "profiles" / "ops" / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home / "profiles" / "ops"))
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin", root=home)
    return home


def _build_agent():
    from run_agent import AIAgent
    cfg = {"agent": {}}
    with patch("agent.process_bootstrap.OpenAI"), \
         patch("hermes_cli.config.load_config", return_value=cfg), \
         patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        return AIAgent(api_key="test-key", base_url="http://127.0.0.1:9/v1", model="test/model",
                       quiet_mode=True, skip_context_files=True, skip_memory=True)


def _fail_listing_marker_folder(monkeypatch, error: OSError):
    real_listdir = os.listdir

    def listdir(path):
        if Path(path).name == pre.RUN_ELSEWHERE_DIRNAME:
            raise error
        return real_listdir(path)

    monkeypatch.setattr(pre.os, "listdir", listdir)


def test_agent_refuses_a_marked_profile_before_init(root, monkeypatch):
    monkeypatch.setattr("agent.agent_init.init_agent",
                        lambda *a, **k: pytest.fail("agent init ran for a profile a plugin runs elsewhere"))
    with pytest.raises(pre.ProfileRunsElsewhere) as refused:
        _build_agent()
    message = str(refused.value)
    assert "'ops'" in message and "sandbox-plugin" in message and "desktop" in message


def test_agent_refuses_a_marked_sibling_built_by_a_multi_profile_host(root, monkeypatch):
    """The machine backend (default profile) builds a sibling's agent under a home override."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setenv("HERMES_HOME", str(root))
    built = []
    monkeypatch.setattr("agent.agent_init.init_agent", lambda agent, **k: built.append(agent))
    token = set_hermes_home_override(str(root / "profiles" / "ops"))
    try:
        with pytest.raises(pre.ProfileRunsElsewhere, match="sandbox-plugin"):
            _build_agent()
    finally:
        reset_hermes_home_override(token)
    _build_agent()  # the default profile itself is not marked
    assert len(built) == 1


def test_the_backend_denied_the_marker_folder_builds_its_agent(root, monkeypatch):
    _fail_listing_marker_folder(monkeypatch, PermissionError(errno.EACCES, "Access is denied"))
    agent = _build_agent()
    assert agent.model == "test/model"


def test_an_unreadable_marker_folder_refuses_every_agent(root, monkeypatch):
    _fail_listing_marker_folder(monkeypatch, OSError(errno.EIO, "I/O error"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    with pytest.raises(pre.ProfileRunsElsewhere, match="cannot tell"):
        pre.require_home_runs_here()


def test_cli_agent_commands_exit_with_the_refusal_before_startup(root, monkeypatch, capsys):
    from hermes_cli import main as main_mod

    monkeypatch.setattr(main_mod, "_bypass_chat_launch", lambda _a: False, raising=False)
    monkeypatch.setattr(main_mod, "_guard_noninteractive_user_config", lambda _a: None)
    monkeypatch.setitem(sys.modules, "hermes_cli.plugins", types.SimpleNamespace(
        discover_plugins=lambda: pytest.fail("plugin discovery before the refusal"),
        start_background_plugin_discovery=lambda: pytest.fail("plugin discovery before the refusal")))
    for command, extra in ((None, {}), ("chat", {}), ("acp", {}), ("cron", {"cron_command": "tick"})):
        args = Namespace(accept_hooks=False, command=command, cron_command=None, gateway_command=None,
                         mcp_action=None, tui=False, yolo=False, safe_mode=False, ignore_user_config=False)
        vars(args).update(extra)
        with pytest.raises(SystemExit) as exited:
            main_mod._prepare_agent_startup(args)
        assert exited.value.code == 1
        err = capsys.readouterr().err
        assert "'ops'" in err and "sandbox-plugin" in err and "desktop" in err


def test_cli_refusal_leaves_gateway_run_to_its_own_check_and_spares_unmarked_or_inside(root, monkeypatch):
    pre.exit_if_profile_runs_elsewhere("gateway")  # run_bootstrap refuses it with exit 78
    with pytest.raises(SystemExit):
        pre.exit_if_profile_runs_elsewhere("chat")
    monkeypatch.setenv("HERMES_HOME", str(root))
    pre.exit_if_profile_runs_elsewhere("chat")  # default is not marked
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))
    _fail_listing_marker_folder(monkeypatch, PermissionError(errno.EACCES, "Access is denied"))
    pre.exit_if_profile_runs_elsewhere("chat")  # inside the backend the plugin runs


@pytest.mark.parametrize("isolated", [False, True])
def test_serve_and_dashboard_exit_78_before_attaching_or_binding(root, monkeypatch, capsys, isolated):
    from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE
    from hermes_cli import main_dashboard

    monkeypatch.setattr(main_dashboard, "_host_backend_attachment",
                        lambda: pytest.fail("attach probed for a profile a plugin runs elsewhere"))
    with pytest.raises(SystemExit) as exited:
        main_dashboard._attach_to_host_backend(Namespace(isolated=isolated), True)
    assert exited.value.code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    err = capsys.readouterr().err
    assert "'ops'" in err and "sandbox-plugin" in err


def test_serve_inside_the_backend_the_plugin_runs_is_not_refused(root, monkeypatch):
    from hermes_cli import main_dashboard

    _fail_listing_marker_folder(monkeypatch, PermissionError(errno.EACCES, "Access is denied"))
    assert main_dashboard._attach_to_host_backend(Namespace(isolated=True), True) is None
