"""Regression coverage for the consent-aware post-update lifecycle hook."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import shell_hooks
from hermes_cli import main as hermes_main
from hermes_cli import plugins


def _configure_post_update_hook(tmp_path: Path, monkeypatch) -> Path:
    """Create one auto-consented hook and return its append-only output path."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    output_path = tmp_path / "hook-events.jsonl"
    script = tmp_path / "post-update-hook.py"
    script.write_text(
        "import json, os, pathlib, sys\n"
        "payload = json.load(sys.stdin)\n"
        "with pathlib.Path(os.environ['HOOK_OUTPUT']).open('a', encoding='utf-8') as f:\n"
        "    f.write(json.dumps(payload) + '\\n')\n",
        encoding="utf-8",
    )
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
    (hermes_home / "config.yaml").write_text(
        json.dumps(
            {
                "hooks_auto_accept": True,
                "hooks": {
                    "post_update": [
                        {"command": command, "timeout": 10},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOOK_OUTPUT", str(output_path))
    shell_hooks.reset_for_tests()
    plugins._plugin_manager = plugins.PluginManager()
    # The update path must discover plugins, but this test isolates the shell
    # callback boundary from unrelated bundled plugin registration.
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    return output_path


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("route", ["git", "zip", "pip"])
def test_successful_routes_emit_one_configured_callback(tmp_path, monkeypatch, route):
    output_path = _configure_post_update_hook(tmp_path, monkeypatch)

    hermes_main._run_post_update_hooks(
        SimpleNamespace(accept_hooks=False),
        route=route,
    )

    events = _read_events(output_path)
    assert len(events) == 1
    assert events[0]["hook_event_name"] == "post_update"
    assert events[0]["extra"]["route"] == route


def test_pip_success_emits_post_update_once(tmp_path, monkeypatch):
    output_path = _configure_post_update_hook(tmp_path, monkeypatch)
    managed_uv = __import__("hermes_cli.managed_uv", fromlist=["ensure_uv"])
    config = __import__("hermes_cli.config", fromlist=["is_uv_tool_install"])
    monkeypatch.setattr(managed_uv, "update_managed_uv", lambda: None)
    monkeypatch.setattr(managed_uv, "ensure_uv", lambda: None)
    monkeypatch.setattr(config, "is_uv_tool_install", lambda: False)
    # Replace only main's subprocess module reference; shell_hooks keeps the
    # real subprocess module so the configured callback actually executes.
    monkeypatch.setattr(
        hermes_main,
        "subprocess",
        SimpleNamespace(
            run=lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0),
        ),
    )
    monkeypatch.setattr(hermes_main.sys, "prefix", "/usr")
    monkeypatch.setattr(hermes_main.sys, "base_prefix", "/usr")

    hermes_main._cmd_update_pip(SimpleNamespace(accept_hooks=False))

    events = _read_events(output_path)
    assert len(events) == 1
    assert events[0]["extra"]["route"] == "pip"


def test_pip_failure_never_emits_post_update(tmp_path, monkeypatch):
    output_path = _configure_post_update_hook(tmp_path, monkeypatch)
    managed_uv = __import__("hermes_cli.managed_uv", fromlist=["ensure_uv"])
    config = __import__("hermes_cli.config", fromlist=["is_uv_tool_install"])
    monkeypatch.setattr(managed_uv, "update_managed_uv", lambda: None)
    monkeypatch.setattr(managed_uv, "ensure_uv", lambda: None)
    monkeypatch.setattr(config, "is_uv_tool_install", lambda: False)
    monkeypatch.setattr(
        hermes_main,
        "subprocess",
        SimpleNamespace(
            run=lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1),
        ),
    )
    monkeypatch.setattr(hermes_main.sys, "prefix", "/usr")
    monkeypatch.setattr(hermes_main.sys, "base_prefix", "/usr")

    with pytest.raises(SystemExit) as exc_info:
        hermes_main._cmd_update_pip(SimpleNamespace(accept_hooks=False))

    assert exc_info.value.code == 1
    assert _read_events(output_path) == []


def test_zip_failure_never_emits_post_update(monkeypatch):
    dispatch = Mock()
    monkeypatch.setattr(hermes_main, "_run_post_update_hooks", dispatch)
    monkeypatch.setattr(hermes_main, "_resolve_update_branch", lambda _args: "main")

    import urllib.request

    def fail_download(*_args, **_kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(urllib.request, "urlretrieve", fail_download)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main._update_via_zip(SimpleNamespace())

    assert exc_info.value.code == 1
    dispatch.assert_not_called()


def test_update_parser_exposes_explicit_hook_consent():
    import argparse

    from hermes_cli.subcommands.update import build_update_parser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=lambda _args: None)
    args = parser.parse_args(["update", "--accept-hooks"])

    assert args.accept_hooks is True
