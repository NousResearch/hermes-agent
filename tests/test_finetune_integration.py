"""Integration tests for the finetune skill's wiring into core CLI/gateway.

Covers the review findings on the feat/finetune integration:
  - gateway reaction-hook registration honors finetune.feedback.gateway_reactions
    (the registration must actually run — it was previously dead code)
  - /finetune command registration is cli_only and includes the gc subcommand
  - streaming subprocess runner enforces its deadline even when the child
    hangs silently (no output)
  - the CLI feedback writer never raises out of a prompt_toolkit key handler,
    even with a read-only/unwritable HERMES_HOME
  - the finetune-routing plugin restores sys.path after importing route.py
"""

import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Gateway reaction-hook registration (config-gated builtin)
# ---------------------------------------------------------------------------

@pytest.fixture
def hooks_registry(monkeypatch, tmp_path):
    """Return a factory producing a fresh HookRegistry with a given config."""
    import gateway.hooks as hooks_mod
    import hermes_cli.config as config_mod

    # No on-disk hooks — only builtins should register.
    monkeypatch.setattr(hooks_mod, "HOOKS_DIR", tmp_path / "no-such-hooks")

    def _make(config):
        monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: config)
        registry = hooks_mod.HookRegistry()
        registry.discover_and_load()
        return registry

    return _make


def test_reaction_hook_registered_when_gateway_reactions_enabled(hooks_registry):
    import gateway.hooks as hooks_mod

    registry = hooks_registry(
        {"finetune": {"feedback": {"gateway_reactions": True}}}
    )
    handlers = registry._handlers.get("reaction:add", [])
    assert hooks_mod._finetune_reaction_handler in handlers
    names = [h["name"] for h in registry.loaded_hooks]
    assert "finetune-feedback" in names


@pytest.mark.parametrize(
    "config",
    [
        {"finetune": {"feedback": {"gateway_reactions": False}}},
        {"finetune": {}},
        {},
    ],
)
def test_reaction_hook_not_registered_when_disabled(hooks_registry, config):
    registry = hooks_registry(config)
    assert "reaction:add" not in registry._handlers
    names = [h["name"] for h in registry.loaded_hooks]
    assert "finetune-feedback" not in names


def test_reaction_handler_writes_feedback(hooks_registry, monkeypatch, tmp_path):
    """The registered handler records thumbs up/down into feedback.jsonl."""
    import gateway.hooks as hooks_mod

    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setattr(hooks_mod, "get_hermes_home", lambda: home)

    hooks_mod._finetune_reaction_handler(
        "reaction:add",
        {"emoji": "👍", "session_id": "sess-1", "platform": "telegram"},
    )
    feedback = home / "finetune" / "feedback.jsonl"
    assert feedback.exists()
    record = json.loads(feedback.read_text().strip())
    assert record["session_id"] == "sess-1"
    assert record["score"] == 1.0
    assert record["signal"] == "thumbs_up"


# ---------------------------------------------------------------------------
# 2. /finetune command registration
# ---------------------------------------------------------------------------

def test_finetune_command_is_cli_only_and_has_gc():
    from hermes_cli.commands import COMMAND_REGISTRY, GATEWAY_KNOWN_COMMANDS

    finetune = next(c for c in COMMAND_REGISTRY if c.name == "finetune")
    assert finetune.cli_only is True
    assert "gc" in finetune.subcommands
    # cli_only (with no gateway_config_gate) must keep it out of the gateway
    # dispatch set, so platforms neither advertise it nor forward
    # "/finetune train" to the LLM as a user turn.
    assert finetune.gateway_config_gate is None
    assert "finetune" not in GATEWAY_KNOWN_COMMANDS


def test_finetune_handler_lives_in_commands_mixin():
    """The /finetune handler sits with its 40 siblings in CLICommandsMixin."""
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    assert "_handle_finetune_command" in CLICommandsMixin.__dict__


# ---------------------------------------------------------------------------
# 3. Streaming subprocess runner — deadline on silent hangs
# ---------------------------------------------------------------------------

def test_streaming_runner_times_out_on_silent_hang(tmp_path):
    """A child that produces no output must still hit the deadline."""
    from hermes_cli.cli_commands_mixin import _stream_finetune_subprocess

    lines = []
    cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
    start = time.time()
    with pytest.raises(subprocess.TimeoutExpired):
        _stream_finetune_subprocess(
            cmd,
            cwd=str(tmp_path),
            env=os.environ.copy(),
            timeout_seconds=1.5,
            print_fn=lines.append,
        )
    elapsed = time.time() - start
    # Old implementation blocked forever on proc.stdout; the deadline plus
    # kill/cleanup must complete promptly.
    assert elapsed < 20
    assert lines == []


def test_streaming_runner_streams_output_and_returns_exit_code(tmp_path):
    from hermes_cli.cli_commands_mixin import _stream_finetune_subprocess

    lines = []
    rc = _stream_finetune_subprocess(
        [sys.executable, "-c", "print('alpha'); print('beta')"],
        cwd=str(tmp_path),
        env=os.environ.copy(),
        timeout_seconds=30,
        print_fn=lines.append,
    )
    assert rc == 0
    assert lines == ["alpha", "beta"]

    rc = _stream_finetune_subprocess(
        [sys.executable, "-c", "raise SystemExit(3)"],
        cwd=str(tmp_path),
        env=os.environ.copy(),
        timeout_seconds=30,
        print_fn=lines.append,
    )
    assert rc == 3


# ---------------------------------------------------------------------------
# 4. Feedback writer — must not raise out of a key handler
# ---------------------------------------------------------------------------

def test_record_finetune_feedback_swallows_unwritable_home(monkeypatch, tmp_path):
    import cli as cli_mod

    # Make mkdir fail deterministically (works even as root, unlike chmod):
    # a plain file where a parent directory is required.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setattr(cli_mod, "get_hermes_home", lambda: blocker / "home")

    dummy = types.SimpleNamespace(session_id="sess-ro")
    # Must not raise — this runs inside a prompt_toolkit key handler.
    cli_mod.HermesCLI._record_finetune_feedback(dummy, 1.0, "thumbs_up")
    cli_mod.HermesCLI._record_finetune_feedback(dummy, 0.0, "thumbs_down")


def test_record_finetune_feedback_writes_record(monkeypatch, tmp_path):
    import cli as cli_mod

    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setattr(cli_mod, "get_hermes_home", lambda: home)
    printed = []
    monkeypatch.setattr(cli_mod, "_cprint", printed.append)

    dummy = types.SimpleNamespace(session_id="sess-ok")
    cli_mod.HermesCLI._record_finetune_feedback(dummy, 1.0, "thumbs_up")

    record = json.loads((home / "finetune" / "feedback.jsonl").read_text().strip())
    assert record["session_id"] == "sess-ok"
    assert record["score"] == 1.0
    assert record["signal"] == "thumbs_up"
    assert printed  # confirmation line surfaced


# ---------------------------------------------------------------------------
# 5. finetune-routing plugin — sys.path hygiene
# ---------------------------------------------------------------------------

PLUGIN_INIT = (
    PROJECT_ROOT
    / "optional-skills" / "mlops" / "finetune"
    / "plugin" / "finetune-routing" / "__init__.py"
)


def _load_plugin_module(name):
    spec = importlib.util.spec_from_file_location(name, PLUGIN_INIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def scrub_route_modules():
    """Isolate sys.modules['route'/'common'] around the test."""
    saved = {k: sys.modules.pop(k, None) for k in ("route", "common")}
    yield
    for key, value in saved.items():
        if value is not None:
            sys.modules[key] = value
        else:
            sys.modules.pop(key, None)


def test_plugin_router_import_restores_sys_path(
    monkeypatch, tmp_path, scrub_route_modules
):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "common.py").write_text("MARKER = 'stub'\n")
    (scripts_dir / "route.py").write_text(
        "from common import MARKER\n"
        "class AdapterRouter:\n"
        "    enabled = True\n"
        "    def route(self, message):\n"
        "        return {}\n"
    )

    mod = _load_plugin_module("finetune_routing_plugin_it_ok")
    monkeypatch.setattr(mod, "_find_scripts_dir", lambda: scripts_dir)

    router = mod._get_router()
    assert router is not None
    assert mod._router_failed is False
    # sys.path must be restored after the import completes.
    assert str(scripts_dir) not in sys.path


def test_plugin_router_import_restores_sys_path_on_failure(
    monkeypatch, tmp_path, scrub_route_modules
):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "common.py").write_text("")
    (scripts_dir / "route.py").write_text("raise RuntimeError('boom')\n")

    mod = _load_plugin_module("finetune_routing_plugin_it_fail")
    monkeypatch.setattr(mod, "_find_scripts_dir", lambda: scripts_dir)

    router = mod._get_router()
    assert router is None
    assert mod._router_failed is True
    # Even on failure the temporary sys.path entry must be removed.
    assert str(scripts_dir) not in sys.path


def test_plugin_router_preserves_preexisting_sys_path_entry(
    monkeypatch, tmp_path, scrub_route_modules
):
    """If the scripts dir was already on sys.path, the plugin leaves it alone."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "common.py").write_text("MARKER = 'stub'\n")
    (scripts_dir / "route.py").write_text(
        "from common import MARKER\n"
        "class AdapterRouter:\n"
        "    enabled = True\n"
    )

    sys.path.insert(0, str(scripts_dir))
    try:
        mod = _load_plugin_module("finetune_routing_plugin_it_pre")
        monkeypatch.setattr(mod, "_find_scripts_dir", lambda: scripts_dir)
        assert mod._get_router() is not None
        # Entry the caller owns is not removed behind their back.
        assert str(scripts_dir) in sys.path
    finally:
        while str(scripts_dir) in sys.path:
            sys.path.remove(str(scripts_dir))
