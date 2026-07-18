"""Regression test for issue #64428.

When ``--yolo`` / ``HERMES_YOLO_MODE`` is active, the ACP edit-approval
policy for the default ("ask") mode must be promoted to ``"session"`` so
the agent can write files without blocking on approval prompts in editors
like Zed.  Without this fix the ACP adapter ignored ``--yolo`` and always
returned ``"ask"`` for the default mode.

The fix lives in ``acp_adapter.server.ACPHandler._edit_approval_policy_for_state``
and consults ``tools.approval._YOLO_MODE_FROZEN``.
"""

from __future__ import annotations

import pytest

import tools.approval as approval_mod
from acp_adapter.session import SessionState


def _make_state(mode: str = "default") -> SessionState:
    """Build a minimal SessionState with just the mode attribute set."""
    state = SessionState.__new__(SessionState)
    state.mode = mode
    state.cwd = "/tmp"
    return state


def _make_handler():
    """Instantiate HermesACPAgent without invoking heavy __init__ side-effects."""
    from acp_adapter.server import HermesACPAgent

    handler = HermesACPAgent.__new__(HermesACPAgent)
    handler.session_manager = None
    handler._conn = None
    return handler


@pytest.fixture
def reset_yolo_state(monkeypatch):
    """Snapshot _YOLO_MODE_FROZEN in both tools.approval (source) and
    acp_adapter.server (the cached import-time copy) and restore them.

    The fix in acp_adapter/server.py reads the module-level
    ``_YOLO_MODE_FROZEN`` symbol that was imported via
    ``from tools.approval import _YOLO_MODE_FROZEN``. Because bool is
    immutable, ``approval_mod._YOLO_MODE_FROZEN = True`` does NOT flip
    the copy already bound in ``acp_adapter.server``'s namespace — we
    must patch both module attributes for the fix to be exercised.
    """
    import acp_adapter.server as acp_server

    original_approval = approval_mod._YOLO_MODE_FROZEN
    original_acp = acp_server._YOLO_MODE_FROZEN
    try:
        yield acp_server
    finally:
        approval_mod._YOLO_MODE_FROZEN = original_approval
        acp_server._YOLO_MODE_FROZEN = original_acp


def _set_yolo(value: bool, acp_server) -> None:
    approval_mod._YOLO_MODE_FROZEN = value
    acp_server._YOLO_MODE_FROZEN = value


class TestYoloEditApprovalPolicy:
    """#64428: --yolo / HERMES_YOLO_MODE promotes ACP default ask → session."""

    def test_default_mode_returns_ask_when_yolo_off(self, reset_yolo_state):
        _set_yolo(False, reset_yolo_state)
        handler = _make_handler()
        policy, cwd = handler._edit_approval_policy_for_state(_make_state("default"))
        assert policy == "ask"
        assert cwd == "/tmp"

    def test_default_mode_promotes_to_session_when_yolo_on(self, reset_yolo_state):
        _set_yolo(True, reset_yolo_state)
        handler = _make_handler()
        policy, _cwd = handler._edit_approval_policy_for_state(_make_state("default"))
        assert policy == "session", (
            "yolo mode must promote default 'ask' policy to 'session' so the "
            "agent can write files without approval prompts (#64428)"
        )

    def test_accept_edits_mode_unaffected_by_yolo(self, reset_yolo_state):
        """accept_edits already maps to workspace_session — yolo must not
        promote it further to session, since the user explicitly chose the
        narrower workspace_session policy."""
        _set_yolo(True, reset_yolo_state)
        handler = _make_handler()
        policy, _cwd = handler._edit_approval_policy_for_state(_make_state("accept_edits"))
        assert policy == "workspace_session"

    def test_dont_ask_mode_unaffected_by_yolo(self, reset_yolo_state):
        """dont_ask already maps to session — yolo is a no-op for it."""
        _set_yolo(True, reset_yolo_state)
        handler = _make_handler()
        policy, _cwd = handler._edit_approval_policy_for_state(_make_state("dont_ask"))
        assert policy == "session"

    def test_unknown_mode_falls_back_to_default_ask_policy(self, reset_yolo_state):
        _set_yolo(False, reset_yolo_state)
        handler = _make_handler()
        policy, _cwd = handler._edit_approval_policy_for_state(_make_state("bogus"))
        assert policy == "ask"

    def test_unknown_mode_promoted_to_session_when_yolo_on(self, reset_yolo_state):
        """An unknown mode falls back to the default 'ask' policy, which
        yolo then promotes to 'session' — consistent with the default mode."""
        _set_yolo(True, reset_yolo_state)
        handler = _make_handler()
        policy, _cwd = handler._edit_approval_policy_for_state(_make_state("bogus"))
        assert policy == "session"

    def test_empty_mode_treated_as_default(self, reset_yolo_state):
        _set_yolo(False, reset_yolo_state)
        handler = _make_handler()
        policy, _cwd = handler._edit_approval_policy_for_state(_make_state(""))
        assert policy == "ask"

    def test_none_mode_treated_as_default(self, reset_yolo_state):
        _set_yolo(False, reset_yolo_state)
        handler = _make_handler()

        state = SessionState.__new__(SessionState)
        state.mode = None
        state.cwd = "/tmp"

        policy, _cwd = handler._edit_approval_policy_for_state(state)
        assert policy == "ask"

    def test_yolo_off_then_on_switches_policy(self, reset_yolo_state):
        """Toggle yolo and verify the policy flips accordingly within one
        handler instance — guards against caching the frozen state at
        construction time."""
        handler = _make_handler()

        _set_yolo(False, reset_yolo_state)
        policy_off, _ = handler._edit_approval_policy_for_state(_make_state("default"))
        assert policy_off == "ask"

        _set_yolo(True, reset_yolo_state)
        policy_on, _ = handler._edit_approval_policy_for_state(_make_state("default"))
        assert policy_on == "session"
