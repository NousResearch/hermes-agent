"""Regression tests for the CLI ``/auto`` in-chat toggle (Auto Mode).

Mirrors ``test_cli_yolo_toggle.py``'s structure and rationale: ``/auto``
routes through ``enable_session_auto`` / ``disable_session_auto`` (matching
``/yolo``'s session-scoped bypass pattern) and ``_transfer_session_auto``
carries the toggle across mid-run session-id rotations (``/branch``, auto
compression continuation) so it doesn't silently revert.

We test ``_toggle_auto_mode`` and ``_is_session_auto_active`` as unbound
methods against a minimal stand-in object that exposes only the attribute
they read (``session_id``), same as the yolo-toggle tests — this avoids the
heavy ``HermesCLI`` construction path.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tools.approval as approval_module
from cli import HermesCLI


SESSION_KEY = "test-cli-auto-mode-session"


@pytest.fixture(autouse=True)
def _clear_approval_state():
    """Clear Auto Mode state around every test so cases are independent."""
    approval_module.clear_session(SESSION_KEY)
    approval_module.clear_session("default")
    yield
    approval_module.clear_session(SESSION_KEY)
    approval_module.clear_session("default")


def _make_stand_in(session_id: str = SESSION_KEY) -> SimpleNamespace:
    """Minimal stand-in exposing only ``session_id`` — see the yolo-toggle
    tests' equivalent helper for the full rationale."""
    return SimpleNamespace(session_id=session_id)


class TestToggleAutoModeIsSessionScoped:
    def test_toggle_enables_session_auto_mode(self):
        stand_in = _make_stand_in()

        assert approval_module.is_session_auto_enabled(SESSION_KEY) is False

        with patch("cli._cprint"):
            HermesCLI._toggle_auto_mode(stand_in)

        assert approval_module.is_session_auto_enabled(SESSION_KEY) is True

    def test_toggle_disables_on_second_call(self):
        stand_in = _make_stand_in()
        with patch("cli._cprint"):
            HermesCLI._toggle_auto_mode(stand_in)  # ON
            assert approval_module.is_session_auto_enabled(SESSION_KEY) is True
            HermesCLI._toggle_auto_mode(stand_in)  # OFF
            assert approval_module.is_session_auto_enabled(SESSION_KEY) is False

    def test_toggle_falls_back_to_default_when_session_id_missing(self):
        stand_in = _make_stand_in(session_id="")
        with patch("cli._cprint"):
            HermesCLI._toggle_auto_mode(stand_in)

        assert approval_module.is_session_auto_enabled("default") is True

    def test_two_independent_sessions_are_isolated(self):
        cli_a = _make_stand_in(session_id="session-auto-a")
        cli_b = _make_stand_in(session_id="session-auto-b")

        try:
            with patch("cli._cprint"):
                HermesCLI._toggle_auto_mode(cli_a)

            assert approval_module.is_session_auto_enabled("session-auto-a") is True
            assert approval_module.is_session_auto_enabled("session-auto-b") is False
        finally:
            approval_module.clear_session("session-auto-a")
            approval_module.clear_session("session-auto-b")

    def test_toggle_on_and_off_are_mutually_exclusive_with_yolo_display(self):
        """Auto Mode and YOLO are independent toggles — enabling Auto Mode
        must not touch the YOLO session-state set."""
        stand_in = _make_stand_in()
        with patch("cli._cprint"):
            HermesCLI._toggle_auto_mode(stand_in)

        assert approval_module.is_session_auto_enabled(SESSION_KEY) is True
        assert approval_module.is_session_yolo_enabled(SESSION_KEY) is False


class TestIsSessionAutoActiveHelper:
    def test_helper_reflects_toggle(self):
        stand_in = _make_stand_in()

        assert HermesCLI._is_session_auto_active(stand_in) is False

        with patch("cli._cprint"):
            HermesCLI._toggle_auto_mode(stand_in)

        assert HermesCLI._is_session_auto_active(stand_in) is True

        with patch("cli._cprint"):
            HermesCLI._toggle_auto_mode(stand_in)

        assert HermesCLI._is_session_auto_active(stand_in) is False

    def test_helper_survives_missing_session_id_attr(self):
        """Mirrors the yolo equivalent — must not raise for __new__-built
        fixtures without a session_id attribute."""
        no_attr = SimpleNamespace()
        assert HermesCLI._is_session_auto_active(no_attr) is False


class TestToggleAutoModeEndToEnd:
    """End-to-end: a flagged command must resolve through the classifier
    (not a blocking prompt) through the same check_all_command_guards path
    the terminal tool uses."""

    def test_toggle_auto_mode_resolves_via_classifier_not_prompt(self):
        stand_in = _make_stand_in()

        session_token = approval_module.set_current_session_key(SESSION_KEY)
        interactive_token = approval_module.set_hermes_interactive_context(True)
        try:
            with patch("cli._cprint"):
                HermesCLI._toggle_auto_mode(stand_in)  # Auto Mode ON

            with patch("tools.approval._auto_mode_classify", return_value="approve") as mock_classify:
                result = approval_module.check_all_command_guards(
                    "rm -rf /tmp/scratch-xyzzy", "local",
                )
            assert result["approved"] is True
            assert result.get("auto_mode_approved") is True
            mock_classify.assert_called_once()
        finally:
            approval_module.reset_current_session_key(session_token)
            approval_module.reset_hermes_interactive_context(interactive_token)

    def test_toggle_auto_mode_can_deny(self):
        stand_in = _make_stand_in()

        session_token = approval_module.set_current_session_key(SESSION_KEY)
        interactive_token = approval_module.set_hermes_interactive_context(True)
        try:
            with patch("cli._cprint"):
                HermesCLI._toggle_auto_mode(stand_in)  # Auto Mode ON

            with patch("tools.approval._auto_mode_classify", return_value="deny"):
                result = approval_module.check_all_command_guards(
                    "rm -rf /tmp/scratch-xyzzy", "local",
                )
            assert result["approved"] is False
            assert result.get("auto_mode_denied") is True
        finally:
            approval_module.reset_current_session_key(session_token)
            approval_module.reset_hermes_interactive_context(interactive_token)


class TestSessionRotationTransfersAutoMode:
    """Mirrors TestSessionRotationTransfersYolo: when session_id rotates
    mid-run, Auto Mode state keyed under the old id must move to the new id."""

    def test_transfer_moves_auto_mode_to_new_session(self):
        stand_in = _make_stand_in(session_id="old-id")
        try:
            approval_module.enable_session_auto("old-id")
            assert approval_module.is_session_auto_enabled("old-id") is True

            HermesCLI._transfer_session_auto(stand_in, "old-id", "new-id")

            assert approval_module.is_session_auto_enabled("new-id") is True
            assert approval_module.is_session_auto_enabled("old-id") is False
        finally:
            approval_module.clear_session("old-id")
            approval_module.clear_session("new-id")

    def test_transfer_is_noop_when_auto_mode_was_off(self):
        stand_in = _make_stand_in(session_id="old-id")
        try:
            HermesCLI._transfer_session_auto(stand_in, "old-id", "new-id")
            assert approval_module.is_session_auto_enabled("new-id") is False
            assert approval_module.is_session_auto_enabled("old-id") is False
        finally:
            approval_module.clear_session("old-id")
            approval_module.clear_session("new-id")

    def test_transfer_is_noop_when_ids_match(self):
        stand_in = _make_stand_in(session_id="same-id")
        try:
            approval_module.enable_session_auto("same-id")
            HermesCLI._transfer_session_auto(stand_in, "same-id", "same-id")
            assert approval_module.is_session_auto_enabled("same-id") is True
        finally:
            approval_module.clear_session("same-id")

    def test_transfer_handles_empty_inputs_safely(self):
        stand_in = _make_stand_in(session_id="x")
        HermesCLI._transfer_session_auto(stand_in, "", "new")
        HermesCLI._transfer_session_auto(stand_in, "old", "")
        assert approval_module.is_session_auto_enabled("new") is False
        assert approval_module.is_session_auto_enabled("old") is False
