"""The registered console entry point handles Ctrl+C during startup."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _entrypoint():
    import tomllib

    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    module, name = config["project"]["scripts"]["hermes"].split(":")
    return importlib.import_module(module), name


def test_registered_entrypoint_handles_startup_interrupt(monkeypatch, capsys):
    module, name = _entrypoint()
    # Keep real main/startup dispatch, but avoid host boot and dependency work.
    for helper in (
        "_set_process_title", "_warn_if_unsupervised_pid1", "_advertise_agent_env",
        "_cleanup_quarantined_exes", "_sweep_stale_bytecode_if_checkout_changed",
    ):
        monkeypatch.setattr(module, helper, lambda: None)
    for helper in (
        "_try_termux_fast_tui_launch", "_try_termux_fast_cli_launch",
        "_try_fast_serve_launch", "_try_fast_chat_launch",
    ):
        monkeypatch.setattr(module, helper, lambda: False)
    monkeypatch.setattr(sys, "argv", ["hermes", "update"])
    monkeypatch.setattr("agent.ssl_verify.install_truststore", lambda: False)
    monkeypatch.setattr("hermes_cli.config.get_container_exec_info", lambda: None)
    monkeypatch.setattr("hermes_cli.venv_sync.check_runtime", lambda _root: None)
    monkeypatch.setattr(module, "_build_cli_parser", lambda: (None, None))
    monkeypatch.setattr(module, "_parse_cli_args", lambda *_args: SimpleNamespace(version=False))
    calls = []

    def interrupted(args):
        calls.append(args)
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "_prepare_agent_startup", interrupted)
    with pytest.raises(SystemExit) as exc:
        getattr(module, name)()
    assert exc.value.code == 130
    assert len(calls) == 1
    assert capsys.readouterr().err == "\nInterrupted.\n"


def test_entrypoint_preserves_normal_return(monkeypatch):
    module, name = _entrypoint()
    monkeypatch.setattr(module, "_main_body", lambda: 7)
    assert getattr(module, name)() == 7


def test_entrypoint_preserves_other_errors(monkeypatch):
    module, name = _entrypoint()

    def failed():
        raise ValueError("startup error")

    monkeypatch.setattr(module, "_main_body", failed)
    with pytest.raises(ValueError, match="startup error"):
        getattr(module, name)()
