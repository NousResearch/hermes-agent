"""Regression tests: startup must not false-flag plugin toolsets as unknown.

Covers the race where ``HermesCLI.__init__`` validates the configured
platform toolsets while background plugin discovery has not yet landed the
plugin-registered toolsets in the live tool registry. Mirrors the MCP-name
exclusion that already exists at the same call site.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch


def _make_cli_capturing_toolset_warnings(toolsets):
    """Construct HermesCLI (test_cli_init-style) and capture _console_print.

    Returns the instance plus every message printed via ``_console_print``
    during construction, so the "Unknown toolsets" warning can be asserted
    without depending on prompt_toolkit rendering.
    """
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    captured = []

    def _capture(self, message="", *args, **kwargs):
        captured.append(str(message))
        return None

    try:
        with patch.dict(sys.modules, prompt_toolkit_stubs), \
             patch.dict("os.environ", clean_env, clear=False):
            import cli as _cli_mod
            _cli_mod = importlib.reload(_cli_mod)
            with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), \
                 patch.dict(_cli_mod.__dict__, {"CLI_CONFIG": _clean_config}), \
                 patch.object(_cli_mod.HermesCLI, "_console_print", _capture):
                instance = _cli_mod.HermesCLI(toolsets=list(toolsets))
    finally:
        # Re-execute cli.py with real prompt_toolkit so module globals
        # rebind cleanly for later tests (same rationale as
        # tests/cli/test_cli_init.py).
        import cli as _cli_restore
        importlib.reload(_cli_restore)

    return instance, captured


def test_plugin_toolset_not_flagged_when_registry_cold(monkeypatch):
    """A toolset provided by a plugin must not warn when discovery hasn't run.

    Simulates the startup race: background plugin discovery in flight, live
    registry cold, persisted key set from the previous launch available.
    """
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_toolset_keys_nowait",
        lambda: {"evey_sandbox"},
    )
    _, captured = _make_cli_capturing_toolset_warnings(["evey_sandbox"])
    assert not [c for c in captured if "Unknown toolsets" in c]


def test_plugin_toolset_via_live_registry_not_flagged(monkeypatch):
    """A plugin toolset already landed in the registry validates normally."""
    from tools import registry as _registry

    _registry.registry.register(
        name="_test_probe_tool",
        toolset="evey_test_probe",
        schema={"name": "_test_probe_tool", "description": "probe", "parameters": {}},
        handler=lambda args, **kw: "{}",
    )
    try:
        _, captured = _make_cli_capturing_toolset_warnings(["evey_test_probe"])
        assert not [c for c in captured if "Unknown toolsets" in c]
    finally:
        _registry.registry.deregister("_test_probe_tool")


def test_genuinely_unknown_toolset_still_warns():
    """A typo must still surface — the warning must not be silenced globally."""
    _, captured = _make_cli_capturing_toolset_warnings(["definitely-not-a-toolset"])
    assert any(
        "Unknown toolsets" in c and "definitely-not-a-toolset" in c for c in captured
    )


def test_plugin_keys_helper_covers_cache_fallback(monkeypatch):
    """get_plugin_toolset_keys_nowait returns persisted keys during the race."""
    import json
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock

    from hermes_cli import plugins as plugins_mod

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "plugin_toolset_keys.json"
        cache.write_text(
            json.dumps({"toolset_keys": ["evey_sandbox"], "portable_mcp": []})
        )
        monkeypatch.setattr(
            plugins_mod, "_plugin_toolset_keys_cache_path", lambda: cache
        )
        fake_thread = Mock()
        fake_thread.is_alive.return_value = True
        monkeypatch.setattr(
            plugins_mod, "_background_discovery_thread", fake_thread, raising=False
        )
        manager = Mock()
        manager._discovered = False
        monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
        keys = plugins_mod.get_plugin_toolset_keys_nowait()
        assert "evey_sandbox" in keys


def test_subagent_lifecycle_accepts_plugin_toolset():
    """delegate_task allowed_toolsets must accept plugin-registered toolsets.

    The lifecycle check previously compared against the static TOOLSETS
    table only, hard-failing any request naming a plugin toolset even when
    the parent legitimately runs with it enabled.
    """
    from types import SimpleNamespace

    from agent.subagent_lifecycle import SubagentLaunchRequest
    from agent.subagent_lifecycle import SubagentLifecycleService  # noqa: F401
    from tools import registry as _registry

    _registry.registry.register(
        name="_test_probe_tool2",
        toolset="evey_test_probe2",
        schema={"name": "_test_probe_tool2", "description": "probe", "parameters": {}},
        handler=lambda args, **kw: "{}",
    )
    try:
        req = SubagentLaunchRequest(
            goal="probe",
            allowed_toolsets=("evey_test_probe2",),
        )
        parent = SimpleNamespace(enabled_toolsets=["evey_test_probe2", "web"])
        # The real validation the dispatcher runs — must not raise for a
        # plugin-registered toolset present in the parent's enabled set.
        SubagentLifecycleService._validate_request(req, parent)
    finally:
        _registry.registry.deregister("_test_probe_tool2")


def test_subagent_lifecycle_rejects_genuinely_unknown_toolset():
    """A typo in allowed_toolsets must still be rejected (guard stays real)."""
    from types import SimpleNamespace

    from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLifecycleService

    req = SubagentLaunchRequest(
        goal="probe",
        allowed_toolsets=("definitely-not-a-toolset",),
    )
    parent = SimpleNamespace(enabled_toolsets=["definitely-not-a-toolset"])
    try:
        SubagentLifecycleService._validate_request(req, parent)
    except Exception as exc:
        assert "Unknown toolsets" in str(exc)
    else:
        raise AssertionError("expected SubagentLifecycleError for unknown toolset")
