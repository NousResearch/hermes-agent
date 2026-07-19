"""Integration tests for the finetune skill's wiring into core CLI/gateway.

Covers the review findings on the feat/finetune integration:
  - finetune.enabled is a real master gate: /finetune refuses every
    subcommand while disabled, and the Ctrl+Y/Ctrl+N feedback keybindings
    require it in addition to feedback.cli_keybindings
  - the gateway reaction-feedback builtin is gone from core (no platform
    adapter ever emitted reaction:add, and session-level emoji labels from
    arbitrary chat members are unsafe as training signal)
  - /finetune command registration is cli_only and includes the gc subcommand
  - streaming subprocess runner enforces its deadline even when the child
    hangs silently (no output), tracks the live child via on_proc so the TUI
    Ctrl+C handler can kill it, and kills the child honestly when it closes
    stdout but keeps running
  - the CLI feedback writer never raises out of a prompt_toolkit key handler,
    even with a read-only/unwritable HERMES_HOME
  - the finetune-routing plugin restores sys.path after importing route.py
"""

import importlib.util
import io
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
# 1. Gateway reactions — removed from core
# ---------------------------------------------------------------------------

def test_reaction_hook_removed_from_core(monkeypatch, tmp_path):
    """No reaction:add handler exists, even with a stale legacy config flag."""
    import gateway.hooks as hooks_mod
    import hermes_cli.config as config_mod

    assert not hasattr(hooks_mod, "_finetune_reaction_handler")

    # No on-disk hooks — only builtins could register.
    monkeypatch.setattr(hooks_mod, "HOOKS_DIR", tmp_path / "no-such-hooks")
    # A user config still carrying the removed flag must not resurrect it.
    monkeypatch.setattr(
        config_mod,
        "load_config",
        lambda *a, **k: {
            "finetune": {"enabled": True, "feedback": {"gateway_reactions": True}}
        },
    )
    registry = hooks_mod.HookRegistry()
    registry.discover_and_load()
    assert "reaction:add" not in registry._handlers
    assert all(h["name"] != "finetune-feedback" for h in registry.loaded_hooks)


# ---------------------------------------------------------------------------
# 1b. finetune.enabled master gate — /finetune command
# ---------------------------------------------------------------------------

@pytest.fixture
def finetune_cmd(monkeypatch):
    """Run _handle_finetune_command against a given config; return printed lines."""
    import cli as cli_mod
    import hermes_cli.config as config_mod
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    printed = []
    monkeypatch.setattr(cli_mod, "_cprint", printed.append)

    def _run(cmd, config):
        monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: config)
        CLICommandsMixin._handle_finetune_command(types.SimpleNamespace(), cmd)
        return printed

    return _run


@pytest.mark.parametrize(
    "config",
    [
        {"finetune": {"enabled": False}},
        {"finetune": {}},
        {},
    ],
)
def test_finetune_command_refused_when_disabled(finetune_cmd, config):
    """Every subcommand — even read-only status — is refused while disabled."""
    printed = finetune_cmd("/finetune status", config)
    out = "\n".join(printed)
    assert "disabled" in out.lower()
    assert "enabled: true" in out  # tells the user how to turn it on
    assert "Running:" not in out  # no script was launched


def test_finetune_command_allowed_when_enabled(finetune_cmd):
    """With the master gate on, /finetune proceeds (bare command → usage)."""
    printed = finetune_cmd("/finetune", {"finetune": {"enabled": True}})
    out = "\n".join(printed)
    assert "disabled" not in out.lower()
    assert "Usage: /finetune" in out


def test_finetune_command_unbalanced_quotes_friendly_error(finetune_cmd):
    """shlex ValueError (unbalanced quotes) surfaces as a message, not a crash."""
    printed = finetune_cmd(
        "/finetune retro good 'sess-1", {"finetune": {"enabled": True}}
    )
    out = "\n".join(printed)
    assert "Couldn't parse /finetune arguments" in out


# ---------------------------------------------------------------------------
# 1c. finetune.enabled master gate — feedback keybindings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "config, expected",
    [
        ({"finetune": {"enabled": True, "feedback": {"cli_keybindings": True}}}, True),
        ({"finetune": {"enabled": False, "feedback": {"cli_keybindings": True}}}, False),
        ({"finetune": {"enabled": True, "feedback": {"cli_keybindings": False}}}, False),
        ({"finetune": {"enabled": True}}, False),
        ({"finetune": {"feedback": {"cli_keybindings": True}}}, False),
        ({}, False),
    ],
)
def test_feedback_keybindings_require_enabled_and_flag(config, expected):
    """Ctrl+Y/Ctrl+N need BOTH finetune.enabled and feedback.cli_keybindings."""
    import cli as cli_mod

    assert cli_mod.HermesCLI._finetune_feedback_keys_enabled(config) is expected


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


def test_streaming_runner_tracks_active_proc(tmp_path):
    """on_proc gets the live Popen at launch and None when done — always."""
    from hermes_cli.cli_commands_mixin import _stream_finetune_subprocess

    seen = []
    rc = _stream_finetune_subprocess(
        [sys.executable, "-c", "print('ok')"],
        cwd=str(tmp_path),
        env=os.environ.copy(),
        timeout_seconds=30,
        print_fn=lambda line: None,
        on_proc=seen.append,
    )
    assert rc == 0
    assert len(seen) == 2
    assert seen[0] is not None and seen[0].pid > 0
    assert seen[-1] is None

    # Cleared even when the runner exits by raising (deadline hit).
    seen.clear()
    with pytest.raises(subprocess.TimeoutExpired):
        _stream_finetune_subprocess(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=str(tmp_path),
            env=os.environ.copy(),
            timeout_seconds=1.0,
            print_fn=lambda line: None,
            on_proc=seen.append,
        )
    assert seen[-1] is None


def test_streaming_runner_kills_child_that_outlives_stdout(monkeypatch, tmp_path):
    """A child that closes stdout but keeps running >10s is killed and
    reported honestly — the tail-wait TimeoutExpired must not escape and
    masquerade as the overall deadline in the caller."""
    import hermes_cli.cli_commands_mixin as mixin_mod
    from hermes_cli.cli_commands_mixin import _stream_finetune_subprocess

    class FakeProc:
        def __init__(self):
            self.pid = 424242
            self.stdout = io.StringIO("line1\n")
            self.returncode = None
            self.wait_calls = 0

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                # stdout hit EOF but the child is still alive.
                raise subprocess.TimeoutExpired("fake-cmd", timeout)
            self.returncode = -9
            return self.returncode

    fake = FakeProc()
    kills = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: fake)
    monkeypatch.setattr(mixin_mod.os, "killpg", lambda pid, sig: kills.append((pid, sig)))

    lines = []
    rc = _stream_finetune_subprocess(
        ["fake-cmd"],
        cwd=str(tmp_path),
        env={},
        timeout_seconds=30,
        print_fn=lines.append,
    )
    import signal as signal_mod

    assert rc == -9  # honest exit code, no TimeoutExpired raised
    assert (fake.pid, signal_mod.SIGKILL) in kills
    assert any("kept running" in line for line in lines)


def test_interrupt_finetune_subprocess_signals_group():
    """The UI-thread Ctrl+C path kills the tracked child's process group."""
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        dummy = types.SimpleNamespace(_finetune_active_proc=proc)
        assert CLICommandsMixin._interrupt_finetune_subprocess(dummy) is True
        assert proc.wait(timeout=10) != 0  # SIGTERM landed
        # A dead child is no longer interruptible.
        assert CLICommandsMixin._interrupt_finetune_subprocess(dummy) is False
    finally:
        if proc.poll() is None:  # pragma: no cover - cleanup on assert failure
            proc.kill()

    # Nothing tracked (attribute never set, or cleared) → nothing to do.
    assert CLICommandsMixin._interrupt_finetune_subprocess(
        types.SimpleNamespace()
    ) is False
    assert CLICommandsMixin._interrupt_finetune_subprocess(
        types.SimpleNamespace(_finetune_active_proc=None)
    ) is False


def test_set_finetune_active_proc_sets_and_clears():
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    dummy = types.SimpleNamespace()
    sentinel = object()
    CLICommandsMixin._set_finetune_active_proc(dummy, sentinel)
    assert dummy._finetune_active_proc is sentinel
    CLICommandsMixin._set_finetune_active_proc(dummy, None)
    assert dummy._finetune_active_proc is None


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
