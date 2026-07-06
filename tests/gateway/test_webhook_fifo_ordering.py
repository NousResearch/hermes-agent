"""Regression tests for the two pre-merge blockers on the topic-ordering fix.

Blocker 1: a synthetic webhook turn (chat_type=='webhook' + arrival_seq) that
hits an ACTIVE session must FIFO-queue as the next turn — never steer, never
interrupt, never merge_text-coalesce — regardless of busy_input_mode.

Blocker 2: the active-session guard key must be derived via the session store
(profile + binding aware), so two same-binding deliveries under different
profiles do NOT collide on one guard key.
"""
import asyncio
import types

import pytest

from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, SessionBinding, build_session_key
from gateway.config import Platform


# --- Blocker 1: synthetic webhook FIFO branch -------------------------------

class _Runner:
    """Minimal stand-in exposing _handle_active_session_busy_message's deps."""
    def __init__(self, busy_input_mode):
        from gateway.run import GatewayRunner
        self._handle_active_session_busy_message = (
            GatewayRunner._handle_active_session_busy_message.__get__(self, _Runner)
        )
        self._queue_or_replace_pending_event = self._record_queue
        self._busy_input_mode = busy_input_mode
        self._busy_text_mode = "interrupt"
        self._draining = False
        self._running_agents = {}
        self.adapters = {Platform.WEBHOOK: object()}
        self.queued = []
        self.steered = []
        self.interrupted = []

    def _is_user_authorized(self, source):
        return True

    def _record_queue(self, session_key, event):
        self.queued.append((session_key, event))


def _webhook_event(seq=1):
    src = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="webhook:gh-review-ask:deliv-2",
        chat_type="webhook",
        user_id="webhook:gh-review-ask",
        user_name="gh-review-ask",
        session_binding=SessionBinding(namespace="gh-review", key="gh-pr-7735"),
    )
    evt = MessageEvent(text="this too", message_type=MessageType.TEXT, source=src,
                       message_id="deliv-2")
    evt.arrival_seq = seq
    return evt


@pytest.mark.parametrize("mode", ["steer", "interrupt", "queue"])
def test_webhook_turn_fifo_queued_never_steer_interrupt(mode):
    r = _Runner(busy_input_mode=mode)
    # a "running agent" with steer/interrupt that MUST NOT be called
    agent = types.SimpleNamespace(
        steer=lambda t: r.steered.append(t) or True,
        interrupt=lambda *a, **k: r.interrupted.append(True),
    )
    key = build_session_key(_webhook_event().source)
    r._running_agents[key] = agent

    handled = asyncio.run(r._handle_active_session_busy_message(_webhook_event(), key))
    assert handled is True
    assert len(r.queued) == 1            # queued as its own next turn
    assert r.steered == []               # never steered
    assert r.interrupted == []           # never interrupted


# --- Blocker 2: guard key is profile + binding aware ------------------------

def test_same_binding_different_profiles_distinct_keys():
    b = SessionBinding(namespace="gh-review", key="gh-pr-7735")
    src_main = SessionSource(platform=Platform.WEBHOOK, chat_id="c1",
                             chat_type="webhook", session_binding=b)
    src_coder = SessionSource(platform=Platform.WEBHOOK, chat_id="c2",
                              chat_type="webhook", session_binding=b, profile="coder")
    # The store passes profile; default profile -> agent:main, coder -> agent:coder.
    assert build_session_key(src_main) == "agent:main:gh-review:gh-pr-7735"
    assert build_session_key(src_coder, profile="coder") == "agent:coder:gh-review:gh-pr-7735"
    assert build_session_key(src_main) != build_session_key(src_coder, profile="coder")
