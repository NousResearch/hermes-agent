"""Behaviour-contract tests for the /model picker's entry and exit repaint.

Regression guard for the "/model freezes the console" bug.

The picker's *only* repaint on open (and on close) used to go through
``_invalidate()``, which has two early-outs:

  * ``_resize_recovery_pending`` -> returns without painting at all
  * a 250ms throttle window shared with unrelated background repaints

If that single repaint is dropped, ``_model_picker_state`` is already set and
the input area's key filters (up/down/escape) are already active, so the
console looks frozen with no visible prompt — the user has to guess to press
ESC.  ``_invalidate``'s own docstring says user-blocking modals must use
``_paint_now()`` instead (approval / clarify / sudo do).

These tests assert the *contract* ("a modal open/close always reaches
app.invalidate()"), not a snapshot of the current implementation, so they stay
valid if the surrounding picker logic is refactored.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cli as cli_mod


class _FakeApp:
    """Minimal stand-in that records invalidate() calls."""

    def __init__(self):
        self.invalidate_count = 0
        self.current_buffer = _FakeBuffer()

    def invalidate(self):
        self.invalidate_count += 1


class _FakeBuffer:
    text = ""
    cursor_position = 0

    def reset(self):
        self.text = ""


def _make_cli(app):
    """Build a bare CLI object with just the attributes the picker touches.

    We deliberately do NOT construct the full HermesCLI: that pulls in a live
    provider inventory and terminal state.  The methods under test only need
    ``_app``, ``_modal_input_snapshot`` and the picker state slot.
    """
    obj = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    obj._app = app
    obj._modal_input_snapshot = None
    obj._model_picker_state = None
    obj._last_invalidate = 0.0
    obj._resize_recovery_pending = False
    return obj


PROVIDERS = [
    {"name": "deepseek", "slug": "deepseek", "models": ["deepseek-chat"],
     "is_current": True, "total_models": 1},
]


@pytest.mark.parametrize("pending", [False, True],
                         ids=["normal", "resize-recovery-pending"])
def test_open_model_picker_always_paints(pending):
    """Opening the picker must repaint even mid resize-recovery..#41098 class."""
    app = _FakeApp()
    cli = _make_cli(app)
    cli._resize_recovery_pending = pending

    cli._open_model_picker(PROVIDERS, "deepseek-chat", "deepseek")

    assert cli._model_picker_state is not None, "picker state must be set"
    assert app.invalidate_count >= 1, (
        "the picker's only entry repaint was dropped — the panel would never "
        "render while its key filters are already active (console freeze)"
    )


@pytest.mark.parametrize("pending", [False, True],
                         ids=["normal", "resize-recovery-pending"])
def test_close_model_picker_always_paints(pending):
    """Closing must repaint, else the panel lingers over cleared state."""
    app = _FakeApp()
    cli = _make_cli(app)
    cli._model_picker_state = {"stage": "provider", "providers": PROVIDERS,
                               "selected": 0}
    cli._resize_recovery_pending = pending

    cli._close_model_picker()

    assert cli._model_picker_state is None, "picker state must be cleared"
    assert app.invalidate_count >= 1, (
        "the close repaint was dropped — the picker stays on screen after its "
        "state is cleared, which reads as a dead console"
    )


def test_open_then_close_roundtrip_paints_both_ways():
    """The full user gesture /model -> ESC must produce two paints."""
    app = _FakeApp()
    cli = _make_cli(app)
    cli._resize_recovery_pending = True  # worst case: both guards armed

    cli._open_model_picker(PROVIDERS, "deepseek-chat", "deepseek")
    first = app.invalidate_count
    cli._close_model_picker()

    assert first >= 1 and app.invalidate_count > first
