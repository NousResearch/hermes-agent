"""Tests for the ``pre_command`` / ``post_command`` plugin hooks.

The hooks fire inside ``HermesCLI.process_command()`` before any slash command
handler (``pre_command``) and before the CLI exits on ``/quit``
(``post_command``).  Driving the full CLI loop from a unit test would be
prohibitively heavy, so these tests exercise the ``PluginManager.invoke_hook``
dispatch semantics that the wiring in ``cli.py`` depends on.

Mirrors the pattern in ``test_transform_llm_output_hook.py`` and
``test_transform_tool_result_hook.py``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import PluginManager, VALID_HOOKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_enabled_plugin(hermes_home: Path, name: str, register_body: str) -> Path:
    """Create a plugin under <hermes_home>/plugins/<name> and opt it in."""
    plugin_dir = hermes_home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "0.1.0"}), encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        f"    {register_body}\n",
        encoding="utf-8",
    )
    cfg_path = hermes_home / "config.yaml"
    cfg = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg.setdefault("plugins", {}).setdefault("enabled", []).append(name)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return plugin_dir


# ---------------------------------------------------------------------------
# Registered in VALID_HOOKS
# ---------------------------------------------------------------------------


def test_pre_command_in_valid_hooks():
    assert "pre_command" in VALID_HOOKS


def test_post_command_in_valid_hooks():
    assert "post_command" in VALID_HOOKS


def test_on_quit_in_valid_hooks():
    assert "on_quit" in VALID_HOOKS


# ---------------------------------------------------------------------------
# Kwarg shape
# ---------------------------------------------------------------------------


def test_pre_command_receives_expected_kwargs(tmp_path, monkeypatch):
    """Hook callback should see command, raw, session_id, and a cli-like object."""
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)
    _make_enabled_plugin(
        hermes_home, "capture_hook",
        register_body=(
            'ctx.register_hook("pre_command", '
            'lambda **kw: f"{kw[\'command\']}|{kw[\'raw\']}|{kw[\'session_id\']}"'
            ")"
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    mgr = PluginManager()
    mgr.discover_and_load()

    results = mgr.invoke_hook(
        "pre_command",
        command="quit",
        raw="/quit",
        session_id="sess-001",
        cli=object(),  # placeholder — real call passes HermesCLI instance
    )
    assert results == ["quit|/quit|sess-001"]


def test_post_command_receives_expected_kwargs(tmp_path, monkeypatch):
    """post_command fires after every non-quit command handler."""
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)
    _make_enabled_plugin(
        hermes_home, "capture_hook",
        register_body=(
            'ctx.register_hook("post_command", '
            'lambda **kw: f"{kw[\'command\']}|{kw[\'session_id\']}"'
            ")"
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    mgr = PluginManager()
    mgr.discover_and_load()

    results = mgr.invoke_hook(
        "post_command",
        command="help",
        raw="/help",
        session_id="sess-002",
        cli=object(),
    )
    assert results == ["help|sess-002"]


# ---------------------------------------------------------------------------
# Exception safety — a raising callback must not break dispatch
# ---------------------------------------------------------------------------


def test_pre_command_hook_exception_does_not_break_dispatch(tmp_path, monkeypatch):
    """A plugin raising an exception must not stop invoke_hook from continuing."""
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)
    _make_enabled_plugin(
        hermes_home, "raising_hook",
        register_body=(
            "def _boom(**kw):\n"
            "        raise RuntimeError(\"boom\")\n"
            "    ctx.register_hook(\"pre_command\", _boom)"
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    mgr = PluginManager()
    mgr.discover_and_load()

    # Should not raise
    results = mgr.invoke_hook(
        "pre_command",
        command="help",
        raw="/help",
        session_id="s-1",
        cli=object(),
    )
    assert results == []  # raising callback contributes nothing


def test_on_quit_hook_exception_does_not_break_dispatch(tmp_path, monkeypatch):
    """Even on quit path, a raising hook must not prevent other hooks from running."""
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)

    # Two plugins: one raises, one produces a result
    _make_enabled_plugin(
        hermes_home, "raising_hook",
        register_body=(
            "def _boom(**kw):\n"
            "        raise RuntimeError(\"boom\")\n"
            "    ctx.register_hook(\"on_quit\", _boom)"
        ),
    )
    _make_enabled_plugin(
        hermes_home, "good_hook",
        register_body=(
            'ctx.register_hook("on_quit", '
            'lambda **kw: "title-ok"'
            ")"
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    mgr = PluginManager()
    mgr.discover_and_load()

    results = mgr.invoke_hook(
        "on_quit",
        command="quit",
        raw="/quit",
        session_id="s-2",
        cli=object(),
    )
    # good_hook's result must survive even though raising_hook threw
    assert "title-ok" in results


# ---------------------------------------------------------------------------
# No plugins loaded — invoke_hook returns empty list
# ---------------------------------------------------------------------------


def test_on_quit_receives_expected_kwargs(tmp_path, monkeypatch):
    """on_quit fires on /quit with command="quit"."""
    hermes_home = tmp_path / "hermes_test"
    hermes_home.mkdir(exist_ok=True)
    _make_enabled_plugin(
        hermes_home, "capture_hook",
        register_body=(
            'ctx.register_hook("on_quit", '
            'lambda **kw: f"{kw[\'command\']}|{kw[\'session_id\']}"'
            ")"
        ),
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    mgr = PluginManager()
    mgr.discover_and_load()

    results = mgr.invoke_hook(
        "on_quit",
        command="quit",
        raw="/quit",
        session_id="sess-003",
        cli=object(),
    )
    assert results == ["quit|sess-003"]


def test_no_plugins_returns_empty_results(tmp_path, monkeypatch):
    """With no plugins loaded, invoke_hook returns [] regardless of hook name."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_empty"))
    plugins_mod._plugin_manager = PluginManager()

    mgr = plugins_mod._plugin_manager
    for hook in ("pre_command", "post_command", "on_quit"):
        results = mgr.invoke_hook(
            hook, command="quit", raw="/quit", session_id="", cli=object(),
        )
        assert results == []
