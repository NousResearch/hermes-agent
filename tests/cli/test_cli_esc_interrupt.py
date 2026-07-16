"""Tests for ESC-to-interrupt gating (issue #65303).

The ESC key binding that interrupts a running agent is a closure inside
``HermesCLI.run()`` and can't be imported, but its whole risk lives in the
filter predicate ``HermesCLI._esc_interrupt_active`` — a pure reader of
instance state. We bind that unbound method to a lightweight namespace so we
can assert the gating without constructing a full CLI or a prompt_toolkit
Application.

The predicate must be mutually exclusive with the other eager ``escape``
bindings (modal + /model picker) and must leave approval/clarify overlays to
Ctrl+C, so ESC there stays a no-op rather than aborting the turn.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import cli

# Every state attribute the predicate reads. Defaults describe the one case
# where ESC SHOULD interrupt: agent responding, nothing else on screen.
_BASE_STATE = dict(
    _agent_running=True,
    agent=object(),
    _model_picker_state=None,
    _secret_state=None,
    _sudo_state=None,
    _slash_confirm_state=None,
    _approval_state=None,
    _clarify_state=None,
)

# Attributes whose truthiness must veto the ESC interrupt (a modal/picker owns
# ESC, or an overlay is intentionally reserved for Ctrl+C).
_BLOCKING_STATES = [
    "_model_picker_state",
    "_secret_state",
    "_sudo_state",
    "_slash_confirm_state",
    "_approval_state",
    "_clarify_state",
]


def _predicate(**overrides) -> bool:
    state = {**_BASE_STATE, **overrides}
    return cli.HermesCLI._esc_interrupt_active(SimpleNamespace(**state))


def test_interrupts_when_agent_running_and_screen_clear():
    assert _predicate() is True


def test_no_interrupt_when_agent_not_running():
    assert _predicate(_agent_running=False) is False


def test_no_interrupt_when_agent_is_none():
    # Mirrors the `and self.agent` guard the Ctrl+C/Ctrl+Q paths use.
    assert _predicate(agent=None) is False


@pytest.mark.parametrize("blocking", _BLOCKING_STATES)
def test_no_interrupt_while_any_overlay_or_modal_active(blocking):
    # A truthy modal/picker/overlay state must veto ESC-interrupt so it never
    # competes with the modal/picker ESC bindings (mutual exclusivity) and so
    # approval/clarify keep their Ctrl+C-only behavior.
    assert _predicate(**{blocking: {"response_queue": object()}}) is False


def test_returns_a_plain_bool_not_truthy_object():
    # The prompt_toolkit Condition wraps this; keep it a real bool.
    result = _predicate()
    assert result is True and isinstance(result, bool)
