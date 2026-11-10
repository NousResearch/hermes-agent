"""Steer requeue/drain semantics and the ``_is_openai_client_closed`` truth table."""
from __future__ import annotations

import threading
from unittest.mock import Mock

import pytest

from agent.agent_runtime_helpers import _requeue_pending_steer
from agent.client_lifecycle import ClientLifecycleMixin
from agent.interrupt_control import InterruptControlMixin


class _Bare:
    pass


# ---------------------------------------------------------------- steer requeue


def test_requeue_pending_steer_appends_to_existing_steer():
    agent = _Bare()
    agent._pending_steer_lock = threading.Lock()
    agent._pending_steer = "old"
    _requeue_pending_steer(agent, "new")
    assert agent._pending_steer == "old\nnew"


# ------------------------------------------------------- interrupt_control slots


def test_steer_and_drain_roundtrip_with_lock():
    agent = _Bare()
    agent._pending_steer_lock = threading.Lock()
    agent._pending_steer = None
    assert InterruptControlMixin.steer(agent, "a")
    assert InterruptControlMixin.steer(agent, "b")
    assert InterruptControlMixin._drain_pending_steer(agent) == "a\nb"
    assert agent._pending_steer is None


# ------------------------------------------------------ _is_openai_client_closed


class _Inner:
    def __init__(self, is_closed):
        self.is_closed = is_closed


def _duck(outer=None, inner=None, *, has_outer=True, has_inner=True):
    d = _Bare()
    if has_outer:
        d.is_closed = outer
    if has_inner:
        d._client = inner
    return d


@pytest.mark.parametrize(
    "client, expected",
    [
        # the reviewer's 4 combos: duck outer is_closed True/False, with/without _client
        (_duck(True, has_inner=False), True),
        (_duck(False, has_inner=False), False),
        (_duck(True, _Inner(False)), True),  # outer says closed => closed, inner not consulted
        (_duck(False, _Inner(True)), True),  # outer open, inner closed => inner is the answer
        (_duck(False, _Inner(False)), False),
        (_duck(lambda: True, _Inner(False)), True),  # openai.OpenAI.is_closed() method form
        (_duck(lambda: False, _Inner(True)), True),
        (_duck(lambda: False, _Inner(False)), False),
        (_duck(lambda: False, has_inner=False), False),
        (_duck(None, _Inner(True)), True),  # outer None => fall through to inner
        (_duck(None, None), False),
        (_duck(has_outer=False, inner=_Inner(True)), True),
        (_duck(has_outer=False, has_inner=False), False),
        (Mock(), False),
    ],
)
def test_is_openai_client_closed_truth_table(client, expected):
    assert ClientLifecycleMixin._is_openai_client_closed(client) is expected
