"""Tests for the contribute_worker_env plugin hook.

Exercises the real path end to end: a plugin installed in a temp
HERMES_HOME is discovered and loaded for real, registers the hook through
``ctx.register_hook``, and the kanban dispatcher's worker spawn is
asserted against the environment it actually hands to ``subprocess.Popen``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from hermes_cli import kanban_db as kb
from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import VALID_HOOKS, PluginManager

PLUGIN_NAME = "worker_env_probe"

# The plugin echoes the binding it was handed back as env vars, so asserting
# on the spawned process's env proves both what the dispatcher passed in and
# what it merged out. It also tries to overwrite a dispatcher-pinned key and
# returns a non-dict from a second callback.
PLUGIN_SOURCE = '''
def contribute(**kwargs):
    return {
        "PROBE_TASK": kwargs["task_id"],
        "PROBE_BOARD": kwargs["board"],
        "PROBE_WORKSPACE": kwargs["workspace"],
        "PROBE_BRANCH": kwargs["branch"],
        "PROBE_RUN_ID": kwargs["run_id"],
        "PROBE_PROFILE": kwargs["profile"],
        "PROBE_UNSET": None,
        "HERMES_KANBAN_TASK": "hijacked",
        "HERMES_KANBAN_BOARD": "someone-elses-board",
    }


def contribute_garbage(**kwargs):
    return "not a dict"


def register(ctx):
    ctx.register_hook("contribute_worker_env", contribute)
    ctx.register_hook("contribute_worker_env", contribute_garbage)
'''


@pytest.fixture
def worker_env_plugin(tmp_path, monkeypatch, caplog):
    """Install and really load a plugin that registers contribute_worker_env.

    Yields ``(loaded_plugin, warning_messages)`` — the discovery record for
    the plugin and every WARNING the plugin system emitted while loading it.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)

    plugin_dir = home / "plugins" / PLUGIN_NAME
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": PLUGIN_NAME, "version": "0.1.0"})
    )
    (plugin_dir / "__init__.py").write_text(PLUGIN_SOURCE)
    # Plugins are opt-in — the allow-list lives in HERMES_HOME/config.yaml.
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [PLUGIN_NAME]}})
    )

    manager = PluginManager()
    with caplog.at_level(logging.WARNING, logger=plugins_mod.__name__):
        manager.discover_and_load()
    # invoke_hook() dispatches through the module-level singleton.
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)

    loaded = manager._plugins[PLUGIN_NAME]
    assert loaded.enabled and loaded.error is None
    yield loaded, [r.getMessage() for r in caplog.records]


def _task(tmp_path):
    return kb.Task(
        id="t_contrib_env",
        title="x",
        body=None,
        assignee="coder",
        status="ready",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=None,
        completed_at=None,
        workspace_kind="worktree",
        workspace_path=str(tmp_path / "ws"),
        claim_lock=None,
        claim_expires=None,
        tenant=None,
        branch_name="wt/t_contrib_env",
        current_run_id=7,
    )


def test_hook_is_a_public_hook_name(worker_env_plugin):
    """A plugin registering the hook is not reported as registering junk."""
    loaded, warnings = worker_env_plugin
    assert "contribute_worker_env" in VALID_HOOKS
    assert "contribute_worker_env" in loaded.hooks_registered
    assert [w for w in warnings if "unknown hook" in w] == []


def test_spawned_worker_env_carries_plugin_contributions(
    worker_env_plugin, tmp_path, monkeypatch
):
    """The dispatcher merges what the plugin returned into the worker env."""
    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["env"] = kwargs.get("env", {})
            self.pid = 4242

    monkeypatch.setattr("subprocess.Popen", _FakePopen)

    kb._default_spawn(_task(tmp_path), str(tmp_path / "ws"))
    env = captured["env"]

    # The plugin saw the real task binding the dispatcher was spawning for.
    assert env["PROBE_TASK"] == "t_contrib_env"
    assert env["PROBE_BRANCH"] == "wt/t_contrib_env"
    assert env["PROBE_WORKSPACE"] == str(tmp_path / "ws")
    assert env["PROBE_PROFILE"] == "coder"
    assert env["PROBE_RUN_ID"] == "7"
    assert env["PROBE_BOARD"] == env["HERMES_KANBAN_BOARD"]
    # None-valued contributions are dropped rather than stringified.
    assert "PROBE_UNSET" not in env


def test_contributions_never_override_dispatcher_pinned_env(
    worker_env_plugin, tmp_path, monkeypatch
):
    """Fill-only merge: a plugin cannot repoint a worker at another board."""
    captured = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["env"] = kwargs.get("env", {})
            self.pid = 4242

    monkeypatch.setattr("subprocess.Popen", _FakePopen)

    kb._default_spawn(_task(tmp_path), str(tmp_path / "ws"))
    env = captured["env"]

    # The same callback that tried to overwrite these did get merged in, so
    # the assertions below are about precedence, not about a hook that never
    # ran.
    assert env["PROBE_TASK"] == "t_contrib_env"
    assert env["HERMES_KANBAN_TASK"] == "t_contrib_env"
    assert env["HERMES_KANBAN_BOARD"] != "someone-elses-board"
    assert env["HERMES_KANBAN_DB"] == str(tmp_path / ".hermes" / "kanban.db")
