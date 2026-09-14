"""C004-R1 RED tests — CLI overlay predicate and interrupt-clear behaviour.

Small behavioural tests around two new private HermesCLI predicate methods
and the existing ``_clear_active_overlays_for_interrupt`` AUQ queue seam.

  * ``_is_normal_input_active()`` — returns False for every overlay including
    ``_auq_state``; True only when no overlay is active.
  * ``_has_interruptible_overlay()`` — returns True for approval/clarify/AUQ/
    sudo/secret overlays (the ones Ctrl+C / Ctrl+Q must clear).
  * ``_clear_active_overlays_for_interrupt()`` — when ``_auq_state`` is active
    must immediately push ``{index: '__skipped__'}`` onto the response queue
    and clear the state.

No source-grep, no change-detector — each test exercises a live seam.
"""
import queue
import threading
from unittest.mock import MagicMock

import pytest

from cli import HermesCLI


def _make_cli_stub() -> HermesCLI:
    """Minimal HermesCLI stub with all overlay state attrs initialised."""
    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_state = None
    cli._approval_deadline = 0
    cli._approval_lock = threading.Lock()
    cli._auq_state = None
    cli._auq_deadline = 0
    cli._sudo_state = None
    cli._sudo_deadline = 0
    cli._secret_state = None
    cli._secret_deadline = 0
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._clarify_deadline = 0
    cli._slash_confirm_state = None
    cli._model_picker_state = None
    cli._modal_input_snapshot = None
    cli._restore_modal_input_snapshot = MagicMock()
    return cli


# ── F1: _is_normal_input_active ─────────────────────────────────────


class TestIsNormalInputActive:
    """``_is_normal_input_active`` must be False when ANY overlay is active,
    including ``_auq_state``, and True only when all overlays are clear."""

    def test_true_when_no_overlay_active(self):
        cli = _make_cli_stub()
        assert cli._is_normal_input_active() is True

    @pytest.mark.parametrize(
        "attr,value",
        [
            ("_clarify_state", {"response_queue": queue.Queue()}),
            ("_approval_state", {"response_queue": queue.Queue()}),
            ("_slash_confirm_state", {"choices": []}),
            ("_sudo_state", {"response_queue": queue.Queue()}),
            ("_secret_state", {"response_queue": queue.Queue()}),
            ("_model_picker_state", {"models": []}),
            ("_auq_state", {"questions": [], "response_queue": queue.Queue()}),
        ],
    )
    def test_false_when_overlay_active(self, attr, value):
        cli = _make_cli_stub()
        setattr(cli, attr, value)
        assert cli._is_normal_input_active() is False, (
            f"_is_normal_input_active was True with {attr} set"
        )


# ── F2: _has_interruptible_overlay ──────────────────────────────────


class TestHasInterruptibleOverlay:
    """``_has_interruptible_overlay`` must be True for the five overlays that
    Ctrl+C / Ctrl+Q must clear (approval/clarify/AUQ/sudo/secret) and False
    for foreground-only UI (slash-confirm, model-picker) and for the idle
    state."""

    def test_false_when_idle(self):
        cli = _make_cli_stub()
        assert cli._has_interruptible_overlay() is False

    @pytest.mark.parametrize(
        "attr,value",
        [
            ("_approval_state", {"response_queue": queue.Queue()}),
            ("_clarify_state", {"response_queue": queue.Queue()}),
            ("_auq_state", {"questions": [], "response_queue": queue.Queue()}),
            ("_sudo_state", {"response_queue": queue.Queue()}),
            ("_secret_state", {"response_queue": queue.Queue()}),
        ],
    )
    def test_true_for_interruptible_overlay(self, attr, value):
        cli = _make_cli_stub()
        setattr(cli, attr, value)
        assert cli._has_interruptible_overlay() is True, (
            f"_has_interruptible_overlay was False with {attr} set"
        )

    @pytest.mark.parametrize(
        "attr,value",
        [
            ("_slash_confirm_state", {"choices": []}),
            ("_model_picker_state", {"models": []}),
        ],
    )
    def test_false_for_foreground_only_overlay(self, attr, value):
        cli = _make_cli_stub()
        setattr(cli, attr, value)
        assert cli._has_interruptible_overlay() is False, (
            f"_has_interruptible_overlay was True with foreground-only {attr} set"
        )


# ── F2b: _clear_active_overlays_for_interrupt AUQ queue ──────────────


class TestClearActiveOverlaysForInterruptAuq:
    """``_clear_active_overlays_for_interrupt`` must immediately push
    ``{index: '__skipped__'}`` onto the AUQ response queue and clear
    ``_auq_state`` so the blocked callback unblocks without waiting for the
    deadline."""

    def test_auq_queue_signalled_and_state_cleared(self):
        cli = _make_cli_stub()
        rq = queue.Queue()
        cli._auq_state = {
            "questions": [
                {"question": "Q1", "options": [
                    {"label": "A", "recommended": True},
                    {"label": "B"},
                ]},
                {"question": "Q2", "options": [
                    {"label": "C", "recommended": True},
                    {"label": "D"},
                ]},
            ],
            "activeIdx": 0,
            "selections": [0, 0],
            "response_queue": rq,
        }
        cli._auq_deadline = 1234567890

        cli._clear_active_overlays_for_interrupt()

        # State cleared immediately.
        assert cli._auq_state is None, "_auq_state not cleared"
        assert cli._auq_deadline == 0, "_auq_deadline not reset"
        # Queue received skipped answers for every question, immediately.
        answers = rq.get_nowait()
        assert answers == {0: "__skipped__", 1: "__skipped__"}, (
            f"expected skipped answers, got {answers}"
        )
        assert rq.empty(), "unexpected extra queue entries"

    def test_auq_clear_is_safe_when_no_state(self):
        """Calling with no AUQ state must be a no-op (safe)."""
        cli = _make_cli_stub()
        cli._clear_active_overlays_for_interrupt()
        assert cli._auq_state is None
