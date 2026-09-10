"""Regression: the background MCP discovery thread must carry the caller's secret scope.

Blocker #2 (PR #104607): ``start_background_mcp_discovery`` spawned a bare thread that
reinstalled only the HERMES_HOME override, dropping the profile's ``agent.secret_scope``
ContextVar. Under active multiplexing a named profile's credentialed ``mcp_servers`` config
(``${GITHUB_TOKEN}`` style placeholder) then hit ``get_secret`` with NO scope installed in the
thread -> ``UnscopedSecretError`` -> ``_load_mcp_config`` swallows it -> zero tools, silently.

These tests exercise the REAL path: start_background_mcp_discovery -> thread ->
_discover_mcp_tools_without_interactive_oauth -> discover_mcp_tools -> the real
_load_mcp_config -> real _interpolate_env_vars -> real agent.secret_scope.get_secret.
Process env is poisoned; two homes get different scoped tokens; each must resolve its OWN.
"""
from __future__ import annotations

from contextlib import nullcontext
import sys
import types

import pytest

from hermes_cli import mcp_startup
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from agent import secret_scope
from tools import mcp_tool_config


@pytest.fixture(autouse=True)
def _reset_state():
    saved_started = mcp_startup._mcp_discovery_started
    saved_thread = mcp_startup._mcp_discovery_thread
    saved_multiplex = secret_scope.is_multiplex_active()
    mcp_startup._mcp_discovery_started = False
    mcp_startup._mcp_discovery_thread = None
    try:
        yield
    finally:
        thread = mcp_startup._mcp_discovery_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        mcp_startup._mcp_discovery_started = saved_started
        mcp_startup._mcp_discovery_thread = saved_thread
        secret_scope.set_multiplex_active(saved_multiplex)


def _install_config_stubs(monkeypatch):
    """Route the real _load_mcp_config at a credentialed config; no OAuth/dotenv cost."""
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        types.SimpleNamespace(
            load_config=lambda: {
                "mcp_servers": {
                    "github": {
                        "command": "github-mcp-server",
                        "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"},
                    }
                }
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.env_loader",
        types.SimpleNamespace(load_hermes_dotenv=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_oauth",
        types.SimpleNamespace(suppress_interactive_oauth=lambda: nullcontext()),
    )


def _run_discovery_for_home(monkeypatch, home, token):
    """Install home + scope on THIS context, launch discovery, return the token the
    thread's real _load_mcp_config resolved for the ``github`` server (or a marker)."""
    captured = {}

    def _discover(allowed_mcp_names=None):
        # Real config load inside the discovery thread's copied context.
        cfg = mcp_tool_config._load_mcp_config()
        gh = cfg.get("github")
        captured["token"] = gh["env"]["GITHUB_TOKEN"] if gh else None
        captured["empty"] = not cfg
        return []

    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool_discovery",
        types.SimpleNamespace(discover_mcp_tools=_discover, get_mcp_status=lambda: []),
    )

    home_token = set_hermes_home_override(home)
    scope_token = secret_scope.set_secret_scope({"GITHUB_TOKEN": token})
    try:
        mcp_startup._mcp_discovery_started = False
        mcp_startup._mcp_discovery_thread = None
        mcp_startup.start_background_mcp_discovery(
            logger=types.SimpleNamespace(debug=lambda *a, **k: None,
                                         warning=lambda *a, **k: None),
            thread_name=f"test-{token}",
        )
        thread = mcp_startup._mcp_discovery_thread
        assert thread is not None, "discovery thread was not started"
        thread.join(timeout=3.0)
        assert not thread.is_alive(), "discovery thread did not finish"
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
    return captured


def test_named_profile_scope_reaches_discovery_thread(monkeypatch):
    """Two multiplexed homes share one MCP server name; process env is poisoned.
    Each home must resolve its OWN scoped token inside the background thread —
    never the poisoned os.environ value, never empty (the UnscopedSecretError bug)."""
    _install_config_stubs(monkeypatch)
    secret_scope.set_multiplex_active(True)
    # Poison the process env: a leaked/absent token would produce this, never the scope's.
    monkeypatch.setenv("GITHUB_TOKEN", "POISON-process-env-token")

    a = _run_discovery_for_home(monkeypatch, "/tmp/home-a", "ghp_AAA_home_a")
    b = _run_discovery_for_home(monkeypatch, "/tmp/home-b", "ghp_BBB_home_b")

    # Never silently empty when a scope IS present (the reported symptom).
    assert a["empty"] is False
    assert b["empty"] is False
    # Each home resolved its own credential — the scope crossed the thread boundary.
    assert a["token"] == "ghp_AAA_home_a"
    assert b["token"] == "ghp_BBB_home_b"
    # And never the poisoned process env value.
    assert "POISON" not in (a["token"] or "")
    assert "POISON" not in (b["token"] or "")


def test_active_multiplex_no_scope_still_fails_closed(monkeypatch):
    """Sanity floor: with multiplex active and NO scope, the credentialed config resolves
    empty (UnscopedSecretError swallowed) — proving the two-home test's non-empty result is
    the scope crossing the boundary, not interpolation being lax."""
    _install_config_stubs(monkeypatch)
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("GITHUB_TOKEN", "POISON-process-env-token")

    captured = {}

    def _discover(allowed_mcp_names=None):
        captured["cfg"] = mcp_tool_config._load_mcp_config()
        return []

    monkeypatch.setitem(
        sys.modules,
        "tools.mcp_tool_discovery",
        types.SimpleNamespace(discover_mcp_tools=_discover, get_mcp_status=lambda: []),
    )
    # No secret scope installed on this context.
    mcp_startup.start_background_mcp_discovery(
        logger=types.SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None),
        thread_name="test-noscope",
    )
    thread = mcp_startup._mcp_discovery_thread
    assert thread is not None
    thread.join(timeout=3.0)
    assert captured.get("cfg") == {}
