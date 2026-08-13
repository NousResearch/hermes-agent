"""Marks must parent to the live turn scope, not the session scope.

Scope events export when their OWNING scope closes. Turn scopes close every
turn; session scopes close only at session end. Parenting marks to the
session scope means a long-lived conversation (the normal enterprise case —
a Slack thread open all day) emits no approval or turn marks for hours, and
none at all if the process dies first. Audit dashboards then show an empty
approval table while approvals are demonstrably firing.

Contract: when a live turn exists for the mark's session, the mark is
attached to the turn handle so it exports at turn end. When no live turn
exists (session-level events such as session.end, or marks emitted outside
a turn), the mark falls back to the session handle — the historical
behavior, which is correct for those cases.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def runtime_and_state():
    """Build the plugin runtime with its relay + session state stubbed."""
    from plugins.observability import nemo_relay as plugin_mod

    runtime = plugin_mod._Runtime.__new__(plugin_mod._Runtime)
    runtime.nemo_relay = MagicMock()
    state = types.SimpleNamespace(
        session_id="sess-long-lived",
        handle="SESSION_HANDLE",
        relay_session=object(),
    )
    runtime.ensure_session = lambda kwargs: state
    runtime.run_in_session = MagicMock()
    return plugin_mod, runtime, state


def _mark_handle(runtime):
    """Return the handle kwarg the mark was dispatched with."""
    assert runtime.run_in_session.called, "mark must dispatch"
    return runtime.run_in_session.call_args.kwargs["handle"]


class TestMarkTurnParenting:
    def test_mark_uses_live_turn_handle(self, runtime_and_state):
        plugin_mod, runtime, state = runtime_and_state
        live_turn = types.SimpleNamespace(handle="TURN_HANDLE")

        with patch.object(
            plugin_mod.relay_runtime, "active_turn", return_value=live_turn
        ):
            runtime.mark("hermes.approval.response", {"choice": "once"})

        assert _mark_handle(runtime) == "TURN_HANDLE", (
            "an approval decided mid-conversation must export at turn end, "
            "not wait for the session to close"
        )

    def test_mark_falls_back_to_session_handle(self, runtime_and_state):
        plugin_mod, runtime, state = runtime_and_state

        with patch.object(
            plugin_mod.relay_runtime, "active_turn", return_value=None
        ):
            runtime.mark("hermes.session.end", {})

        assert _mark_handle(runtime) == "SESSION_HANDLE", (
            "session-level marks with no live turn keep session parentage"
        )

    def test_mark_falls_back_when_turn_has_no_handle(self, runtime_and_state):
        plugin_mod, runtime, state = runtime_and_state
        handleless_turn = types.SimpleNamespace(handle=None)

        with patch.object(
            plugin_mod.relay_runtime, "active_turn", return_value=handleless_turn
        ):
            runtime.mark("hermes.turn.start", {})

        assert _mark_handle(runtime) == "SESSION_HANDLE", (
            "a turn whose scope push failed must not strand the mark"
        )

    def test_active_turn_queried_for_this_session(self, runtime_and_state):
        """Turn lookup is session-scoped: never borrow another session's turn."""
        plugin_mod, runtime, state = runtime_and_state

        with patch.object(
            plugin_mod.relay_runtime, "active_turn", return_value=None
        ) as active_turn:
            runtime.mark("hermes.approval.request", {})

        active_turn.assert_called_once_with("sess-long-lived")
