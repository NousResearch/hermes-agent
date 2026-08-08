"""Tests for one-shot Relay session finalization (#79471).

One-shot mode hard-exits via os._exit, so the Relay conversation was never
finalized: exported telemetry got session start + turn lifecycle but no
session end. The cleanup path must finalize the conversation before exit.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_cli.main import _cleanup_oneshot_runtime


def _run_cleanup(session_id):
    calls = []
    manager = SimpleNamespace(
        invoke_hook=lambda name, **kwargs: calls.append(("plugin", name, kwargs)) or []
    )
    with patch(
        "hermes_cli.observability.observe_lifecycle",
        lambda name, **kwargs: calls.append(("builtin", name, kwargs)),
    ), patch("hermes_cli.plugins.invoke_hook", manager.invoke_hook), patch(
        "hermes_cli.main._oneshot_cleanup_done", False
    ), patch(
        "tools.terminal_tool.cleanup_all_environments", MagicMock()
    ) as cleanup_term, patch(
        "tools.async_delegation.interrupt_all", MagicMock()
    ) as cleanup_deleg, patch(
        "tools.browser_tool._emergency_cleanup_all_sessions", MagicMock()
    ) as cleanup_browser, patch(
        "tools.mcp_tool.shutdown_mcp_servers", MagicMock()
    ) as cleanup_mcp, patch(
        "agent.auxiliary_client.shutdown_cached_clients", MagicMock()
    ) as cleanup_aux:
        _cleanup_oneshot_runtime(session_id)
    return calls, cleanup_term, cleanup_deleg, cleanup_browser, cleanup_mcp, cleanup_aux


def test_cleanup_oneshot_runtime_finalizes_relay_session():
    calls, ct, cd, cb, cm, ca = _run_cleanup("session-1")

    assert ct.call_count == 1
    assert cd.call_count == 1
    assert cb.call_count == 1
    assert cm.call_count == 1
    assert ca.call_count == 1
    builtin = [c for c in calls if c[0] == "builtin"]
    assert builtin
    assert builtin[0][1] == "on_session_finalize"
    assert builtin[0][2]["session_id"] == "session-1"
    assert builtin[0][2]["platform"] == "cli"
    assert builtin[0][2]["reason"] == "oneshot_complete"


def test_cleanup_oneshot_runtime_without_session_id_skips_finalize():
    calls, *_ = _run_cleanup(None)
    assert not [c for c in calls if c[0] == "builtin"]
