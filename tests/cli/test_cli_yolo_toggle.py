"""Regression tests for the CLI ``/yolo`` in-chat toggle.

Pre-fix bug (issue #33925): ``cli.HermesCLI._toggle_yolo`` mutated only
``os.environ["HERMES_YOLO_MODE"]``. That env var is captured once at
module-import time into ``tools.approval._YOLO_MODE_FROZEN`` (security
hardening: stops prompt-injected skills from flipping the bypass mid-run),
so the post-startup toggle was a silent no-op. ``/yolo`` advertised "YOLO ON"
in the status bar while every dangerous command still hit the approval
prompt. Only ``hermes --yolo`` (process-start env), ``HERMES_YOLO_MODE=1``,
and ``hermes config set approvals.mode off`` actually bypassed.

The fix routes the CLI toggle through ``enable_session_yolo`` /
``disable_session_yolo`` (matching the gateway and TUI ``/yolo`` paths) and
binds ``self.session_id`` as the active approval session key around each
``run_conversation`` call so ``is_current_session_yolo_enabled()`` resolves
against the same key the toggle writes under.

We test ``_toggle_yolo`` and ``_is_session_yolo_active`` as unbound methods
against a minimal stand-in object that exposes only the attribute they
read (``session_id``). This avoids the heavy ``HermesCLI`` construction
path used in ``test_cli_init.py``, which is incompatible with this test
file's path layout — ``HermesCLI.__init__`` imports a lot of optional
state we don't need here.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.approval as approval_module
from tools import approval_context
from cli import HermesCLI


SESSION_KEY = "test-cli-yolo-session"


@pytest.fixture(autouse=True)
def _clear_approval_state(monkeypatch):
    """Clear the YOLO bypass + env var around every test so cases are independent."""
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    # The value is intentionally frozen at tools.approval import time. Local
    # Hermes-driven test runs may inherit HERMES_YOLO_MODE=1 from the parent
    # agent process, so make the default test state hermetic; the one test that
    # covers startup-frozen YOLO explicitly patches it back to True.
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    approval_module.clear_session(SESSION_KEY)
    approval_module.clear_session("default")
    yield
    approval_module.clear_session(SESSION_KEY)
    approval_module.clear_session("default")


def _make_stand_in(session_id: str = SESSION_KEY) -> SimpleNamespace:
    """Minimal stand-in exposing only ``session_id``.

    ``_toggle_yolo`` and ``_is_session_yolo_active`` are both pure methods
    that only read ``self.session_id`` — no other CLI state is touched.
    Calling them as unbound functions against this stand-in is equivalent
    to invoking them on a fully-constructed ``HermesCLI`` for the
    behaviour under test, and avoids the brittle prompt_toolkit / config
    stubbing required to instantiate ``HermesCLI`` from this test file.
    """
    return SimpleNamespace(session_id=session_id)


class TestToggleYoloIsSessionScoped:
    """The CLI /yolo handler must mutate the session-yolo set, not the env var.

    The env var path is dead-on-arrival because ``_YOLO_MODE_FROZEN`` is
    captured once at module import, long before the CLI's ``/yolo`` command
    can run.
    """

    def test_toggle_yolo_enables_session_bypass(self):
        stand_in = _make_stand_in()

        assert approval_module.is_session_yolo_enabled(SESSION_KEY) is False

        with patch("cli._cprint"):
            HermesCLI._toggle_yolo(stand_in)

        assert approval_module.is_session_yolo_enabled(SESSION_KEY) is True

    def test_toggle_yolo_disables_session_bypass_on_second_call(self):
        stand_in = _make_stand_in()
        with patch("cli._cprint"):
            HermesCLI._toggle_yolo(stand_in)  # ON
            assert approval_module.is_session_yolo_enabled(SESSION_KEY) is True
            HermesCLI._toggle_yolo(stand_in)  # OFF
            assert approval_module.is_session_yolo_enabled(SESSION_KEY) is False



    def test_two_independent_sessions_are_isolated(self):
        """``/yolo`` toggled in one session must not bypass approvals in
        another session — mirrors the gateway-side invariant."""
        cli_a = _make_stand_in(session_id="session-yolo-a")
        cli_b = _make_stand_in(session_id="session-yolo-b")

        try:
            with patch("cli._cprint"):
                HermesCLI._toggle_yolo(cli_a)

            assert approval_module.is_session_yolo_enabled("session-yolo-a") is True
            assert approval_module.is_session_yolo_enabled("session-yolo-b") is False
        finally:
            approval_module.clear_session("session-yolo-a")
            approval_module.clear_session("session-yolo-b")


class TestIsSessionYoloActiveHelper:
    """The status-bar helper must read the live session-yolo state, not the
    env var (which is the bug class this PR fixes)."""

    def test_helper_reflects_toggle(self):
        stand_in = _make_stand_in()

        assert HermesCLI._is_session_yolo_active(stand_in) is False

        with patch("cli._cprint"):
            HermesCLI._toggle_yolo(stand_in)

        assert HermesCLI._is_session_yolo_active(stand_in) is True

        with patch("cli._cprint"):
            HermesCLI._toggle_yolo(stand_in)

        assert HermesCLI._is_session_yolo_active(stand_in) is False

    def test_helper_honors_frozen_yolo_mode(self):
        """``hermes --yolo`` sets ``HERMES_YOLO_MODE`` before tool imports, so
        ``_YOLO_MODE_FROZEN`` ends up True. The status bar should still
        reflect YOLO on in that case even when the session toggle is off."""
        stand_in = _make_stand_in()

        with patch.object(approval_module, "_YOLO_MODE_FROZEN", True):
            assert HermesCLI._is_session_yolo_active(stand_in) is True

    def test_toggle_under_frozen_yolo_reports_locked_and_stays_on(self):
        """With process-level YOLO frozen ON, /yolo must NOT claim approvals
        are back. Pre-fix, the second toggle printed "YOLO mode OFF —
        dangerous commands will require approval" while the frozen flag kept
        auto-approving everything — a false safety claim."""
        stand_in = _make_stand_in()

        printed = []
        with patch.object(approval_module, "_YOLO_MODE_FROZEN", True):
            with patch("cli._cprint", side_effect=lambda msg: printed.append(msg)):
                HermesCLI._toggle_yolo(stand_in)
                HermesCLI._toggle_yolo(stand_in)

            # Still effectively ON, and no session-level state was flipped.
            assert HermesCLI._is_session_yolo_active(stand_in) is True
            assert not approval_module.is_session_yolo_enabled(SESSION_KEY)

        joined = "\n".join(printed)
        assert "locked ON" in joined
        assert "will require approval" not in joined


class TestToggleYoloEndToEnd:
    """End-to-end: a dangerous command must auto-approve through the same
    ``check_all_command_guards`` path the terminal tool uses."""

    def test_toggle_yolo_bypasses_dangerous_command_check(self):
        stand_in = _make_stand_in()

        token = approval_context.set_current_session_key(SESSION_KEY)
        try:
            with patch("cli._cprint"):
                HermesCLI._toggle_yolo(stand_in)  # YOLO ON

            result = approval_module.check_all_command_guards(
                "rm -rf /tmp/scratch-xyzzy", "local",
            )
            assert result["approved"] is True, (
                f"YOLO toggle should auto-approve dangerous commands, got: {result}"
            )
        finally:
            approval_context.reset_current_session_key(token)




class TestSessionRotationTransfersYolo:
    """When the CLI's ``session_id`` rotates mid-run (``/branch``, auto
    compression continuation), YOLO state keyed under the old id must move
    to the new id. Otherwise the user's ``/yolo ON`` silently reverts on
    the next turn — the same UX failure mode this PR set out to fix.
    Mirrors ``tui_gateway/server.py`` ~line 1297-1305."""

    def test_transfer_moves_yolo_to_new_session(self):
        stand_in = _make_stand_in(session_id="old-id")
        try:
            approval_module.enable_session_yolo("old-id")
            assert approval_module.is_session_yolo_enabled("old-id") is True

            HermesCLI._transfer_session_yolo(stand_in, "old-id", "new-id")

            assert approval_module.is_session_yolo_enabled("new-id") is True
            assert approval_module.is_session_yolo_enabled("old-id") is False
        finally:
            approval_module.clear_session("old-id")
            approval_module.clear_session("new-id")



    def test_transfer_handles_empty_inputs_safely(self):
        stand_in = _make_stand_in(session_id="x")
        # Both directions of empty input should be safe no-ops; nothing
        # to transfer from "" / to "".
        HermesCLI._transfer_session_yolo(stand_in, "", "new")
        HermesCLI._transfer_session_yolo(stand_in, "old", "")
        # Neither key should have been touched.
        assert approval_module.is_session_yolo_enabled("new") is False
        assert approval_module.is_session_yolo_enabled("old") is False
