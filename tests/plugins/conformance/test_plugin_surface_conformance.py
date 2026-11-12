"""Native plugin surface conformance: a frozen plugin keeps working through the real discovery path.

``fixtures/conformance_surface`` is written the way the plugin guide tells an external author to
write a plugin. It is copied into a temp ``HERMES_HOME``, enabled in that home's ``config.yaml`` and
loaded by the process plugin manager, as ``hermes`` itself loads it. The tests assert what a user
of the plugin would see: the tool answers through ``handle_function_call``, hooks fire with the
documented keyword fields, middleware rewrites the call, ``/conformance-ping`` and
``hermes conformance-cli`` answer, each provider resolves by name, and unloading the plugin takes
every one of those back. See website/docs/developer-guide/plugins/index.md § Native plugin
compatibility contract.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_yaml as yaml

_FIXTURE = Path(__file__).parent / "fixtures" / "conformance_surface"
_ROOT = Path(__file__).resolve().parents[3]
PLUGIN = "conformance-surface"
NAME = "conformance"

# The documented kwargs (website/docs/user-guide/features/hooks.md, plugin hook table).
_PRE_TOOL_CALL = {"tool_name", "args", "task_id", "session_id", "conversation_id", "tool_call_id", "turn_id",
                  "api_request_id", "middleware_trace"}
_POST_TOOL_CALL = {"tool_name", "args", "result", "task_id", "session_id", "conversation_id", "tool_call_id",
                   "turn_id", "api_request_id", "duration_ms", "status", "error_type", "error_message",
                   "middleware_trace"}
_PRE_COMMAND = {"surface", "command", "alias_used", "args_raw", "session_key", "platform"}

# (ctx registrar, registry module, lookup function) for each register_<kind>_provider the fixture uses.
_PROVIDERS = [
    ("register_image_gen_provider", "hermes_agent.agent.image_gen_registry", "get_provider"),
    ("register_video_gen_provider", "hermes_agent.agent.video_gen_registry", "get_provider"),
    ("register_web_search_provider", "hermes_agent.agent.web_search_registry", "get_provider"),
    ("register_browser_provider", "hermes_agent.agent.browser_registry", "get_provider"),
    ("register_terminal_environment_provider", "hermes_agent.agent.terminal_env_registry", "get_provider"),
    ("register_tts_provider", "hermes_agent.agent.tts_registry", "get_provider"),
    ("register_transcription_provider", "hermes_agent.agent.transcription_registry", "get_provider"),
    ("register_delegation_provider", "hermes_agent.agent.delegation_registry", "get_provider"),
    ("register_secret_source", "hermes_agent.agent.secret_sources.registry", "get_source"),
]


def _install_fixture(home: Path) -> None:
    shutil.copytree(_FIXTURE, home / "plugins" / PLUGIN, ignore=shutil.ignore_patterns("__pycache__"))
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": [PLUGIN]}}), encoding="utf-8")


@pytest.fixture
def surface(tmp_path, monkeypatch):
    """The process plugin manager after real discovery of a temp home holding only the fixture."""
    from hermes_agent.hermes_cli import plugins

    home = tmp_path / "hermes-home"
    home.mkdir()
    _install_fixture(home)
    bundled = tmp_path / "no-bundled-plugins"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    plugins._reset_plugin_managers_for_tests()
    plugins.discover_plugins()
    manager = plugins.get_plugin_manager()
    yield manager, home
    plugins._reset_plugin_managers_for_tests()


def _events(home: Path, event: str) -> list[dict]:
    path = home / "conformance-events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if row["event"] == event]


def _provider(manager, registry_mod: str, lookup: str):
    import importlib

    return getattr(importlib.import_module(registry_mod), lookup)(NAME, scope=manager.scope_key)


def _call_echo(text: str = "hi") -> str:
    from hermes_agent import model_tools

    return model_tools.handle_function_call(
        "conformance_echo", {"text": text}, task_id="task-1", session_id="session-1",
        tool_call_id="call-1", turn_id="turn-1", api_request_id="req-1",
    )


def test_frozen_plugin_loads_without_error(surface):
    manager, _home = surface
    [row] = [p for p in manager.list_plugins() if p["name"] == PLUGIN]
    assert row["enabled"] is True and not row.get("error"), row


def test_tool_call_runs_middleware_hooks_and_handler_with_documented_kwargs(surface):
    """``register_tool`` + ``register_middleware("tool_request")`` + ``register_hook`` for the three
    tool hooks, all exercised by one real ``handle_function_call``."""
    _manager, home = surface

    out = _call_echo("hi")

    # tool_request middleware rewrote the args before the handler ran; transform_tool_result (a narrow
    # legacy signature) replaced the result.
    assert _events(home, "tool") == [{"event": "tool", "args": {"text": "hi+mw"}}]
    assert out == "transformed:" + json.dumps({"echo": {"text": "hi+mw"}})
    [pre] = _events(home, "pre_tool_call")
    assert _PRE_TOOL_CALL <= set(pre["keys"]), _PRE_TOOL_CALL - set(pre["keys"])
    assert pre["tool_name"] == "conformance_echo" and pre["args"] == {"text": "hi+mw"}
    [post] = _events(home, "post_tool_call")
    assert _POST_TOOL_CALL <= set(post["keys"]), _POST_TOOL_CALL - set(post["keys"])
    assert post["tool_name"] == "conformance_echo"
    assert post["result"] == json.dumps({"echo": {"text": "hi+mw"}})


def test_pre_command_hook_fires_with_documented_kwargs(surface):
    from hermes_agent.hermes_cli.plugins import fire_pre_command_hook

    _manager, home = surface
    fire_pre_command_hook(surface="cli", command="help", alias_used="help", args_raw="extra",
                          session_key="session-1", platform="cli")

    [row] = _events(home, "pre_command")
    assert _PRE_COMMAND <= set(row["keys"]), _PRE_COMMAND - set(row["keys"])
    assert (row["command"], row["args_raw"]) == ("help", "extra")


def test_slash_command_dispatches_to_the_plugin_handler(surface):
    from hermes_agent.hermes_cli.plugins import get_plugin_command_handler, get_plugin_commands

    assert get_plugin_commands()["conformance-ping"]["description"] == "Conformance ping"
    handler = get_plugin_command_handler("conformance-ping")
    assert handler is not None and handler("abc") == "pong:abc"


@pytest.mark.parametrize("registrar,registry_mod,lookup", _PROVIDERS, ids=[p[0] for p in _PROVIDERS])
def test_provider_registrar_makes_the_provider_resolvable_by_name(surface, registrar, registry_mod, lookup):
    manager, _home = surface
    provider = _provider(manager, registry_mod, lookup)
    assert provider is not None, f"ctx.{registrar} did not make {NAME!r} resolvable"
    assert provider.name == NAME


def test_unload_disposes_every_registration(surface):
    """Disabling/uninstalling a plugin unloads it: nothing it registered stays live."""
    from hermes_agent.hermes_cli import plugins
    from hermes_agent.tools.registry import registry

    manager, home = surface
    assert manager.unload(PLUGIN) is True

    assert _events(home, "on_unload") == [{"event": "on_unload"}]
    assert registry.get_entry("conformance_echo", scope=manager.scope_key) is None
    assert plugins.get_plugin_command_handler("conformance-ping") is None
    assert "conformance-cli" not in manager._cli_commands
    for registrar, registry_mod, lookup in _PROVIDERS:
        assert _provider(manager, registry_mod, lookup) is None, f"{registrar} survived unload"
    for hook in ("pre_tool_call", "post_tool_call", "transform_tool_result", "pre_command"):
        assert not manager.has_hook(hook), hook
    assert not manager.has_middleware("tool_request")
    plugins.fire_pre_command_hook(surface="cli", command="help", alias_used="help", args_raw="",
                                  session_key=None, platform="cli")
    assert _events(home, "pre_command") == []


def test_cli_command_runs_from_the_hermes_entry_point(tmp_path):
    """``register_cli_command`` adds ``hermes <name>``: run the real CLI in a child process."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    _install_fixture(home)
    bundled = tmp_path / "no-bundled-plugins"
    bundled.mkdir()
    env = {**os.environ, "HERMES_HOME": str(home), "HERMES_BUNDLED_PLUGINS": str(bundled),
           "HOME": str(tmp_path / "os-home"), "PYTHONPATH": str(_ROOT)}
    proc = subprocess.run([sys.executable, "-m", "hermes_cli.main", "conformance-cli", "--word", "hi"],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False)
    diagnostics = (proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
    assert "conformance-cli:hi" in proc.stdout, diagnostics
    assert proc.returncode == 0, diagnostics
