"""Delivery channel policy and the adviser wire format shared with the surfaces."""

from __future__ import annotations

import threading

from agent.adviser_delivery import deliver, format_batch, resolve_channel


class _Agent:
    def __init__(self):
        self._model_request_active = threading.Event()
        self._interrupt_requested = False
        self._executing_tools = False
        self.steered: list[tuple[str, str]] = []
        self.interrupts: list[str] = []

    def steer(self, text, *, kind="user"):
        self.steered.append((text, kind))
        return True

    def interrupt(self, *, tool_reason=""):
        self.interrupts.append(tool_reason)


class _Adviser:
    continue_severity = "concern"

    def __init__(self, agent, *, live=True, immune=False):
        self.agent = agent
        self._live = live
        self._immune = immune
        self.cards: list[str] = []
        self.continuations: list[str] = []

    def turn_live(self):
        return self._live

    def interrupt_immune_active(self):
        return self._immune

    def preserve_card(self, text, *, severity=None, turn=None):
        self.cards.append(text)

    def set_continuation(self, text, *, source=""):
        self.continuations.append(text)
        return True


def test_wire_format_parses_with_the_cli_contract():
    from hermes_cli.cli_agent_setup_mixin import _adviser_items

    text = format_batch([("re-read the failing test", "concern", "reviewer"),
                         ("the test never ran", "blocker", None)])

    assert _adviser_items(text) == [
        ("concern · reviewer", "re-read the failing test"),
        ("blocker", "the test never ran"),
    ]


def test_channel_table_follows_omp_precedence():
    assert resolve_channel(severity="nit", terminal_idle=True) == "preserve"
    assert resolve_channel(severity="blocker", terminal_idle=True) == "steer"
    assert resolve_channel(severity="nit", streaming=True) == "aside"
    assert resolve_channel(severity="concern", streaming=True) == "steer"
    assert resolve_channel(severity="concern", streaming=True, interrupt_immune=True) == "aside"
    assert resolve_channel(severity="blocker", streaming=True, interrupt_immune=True) == "steer"
    assert resolve_channel(severity="nit", preserve_only=True) == "preserve"
    assert resolve_channel(severity="concern", preserve_only=True, streaming=True) == "steer"


def test_live_concern_is_steered_as_an_adviser_row():
    agent = _Agent()
    adviser = _Adviser(agent)

    channel = deliver(adviser, [("read the diff", "concern", "reviewer")])

    assert channel == "steer"
    text, kind = agent.steered[0]
    assert kind == "adviser" and "read the diff" in text
    assert agent.interrupts == []


def test_blocker_interrupts_the_running_batch():
    agent = _Agent()
    adviser = _Adviser(agent)

    deliver(adviser, [("the write truncates", "blocker", None)])

    assert agent.interrupts == ["adviser"]


def test_terminal_concern_wakes_the_model_once():
    adviser = _Adviser(_Agent(), live=True)

    channel = deliver(adviser, [("fix before done", "concern", None)], terminal=True)

    assert channel == "continuation"
    assert adviser.continuations and "fix before done" in adviser.continuations[0]


def test_terminal_nit_becomes_a_card():
    adviser = _Adviser(_Agent(), live=True)

    channel = deliver(adviser, [("tiny cleanup", "nit", None)], terminal=True)

    assert channel == "preserve" and adviser.cards


def test_after_the_turn_advice_is_a_card_not_a_steer():
    agent = _Agent()
    adviser = _Adviser(agent, live=False)

    channel = deliver(adviser, [("late concern", "concern", None)], terminal=True)

    assert channel == "preserve"
    assert agent.steered == []
