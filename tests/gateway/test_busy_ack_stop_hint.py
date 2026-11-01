"""#133817: the busy-path replies a running turn actually returns must name /stop.

``/stop`` routes to ``request_hard_interrupt`` and is the only text command that halts a
run mid-turn, yet the four acknowledgements users actually receive while deferred
(``queued_tail``, ``steered_tail``, ``steered_subagents_tail``, ``another_turn_running``)
omitted it, so the escape hatch was undiscoverable exactly when needed. Replies that do
not defer anything must stay unchanged (no ``/stop`` mention).
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so gateway code imports without heavy deps (same as test_busy_session_ack)
# ---------------------------------------------------------------------------
_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from agent.i18n import SUPPORTED_LANGUAGES, t
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType


DEFERRAL_KEYS = (
    "queued_tail",
    "steered_tail",
    "steered_subagents_tail",
    "another_turn_running",
)
NON_DEFERRAL_KEYS = ("redirected_tail", "interrupting_tail")


def _make_event() -> MessageEvent:
    source = SessionSource(
        platform=MagicMock(value="telegram"),
        chat_id="123",
        chat_type="private",
        user_id="user1",
    )
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg1",
    )


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._agent_has_active_subagents = lambda _running_agent: False
    return runner


@pytest.mark.parametrize("key", DEFERRAL_KEYS)
@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES)
def test_deferral_replies_name_the_stop_command(key, lang):
    """Each acknowledgement a deferred follow-up produces must name /stop in every shipped
    catalog (#133817) — ``t()`` falls back to ``en`` only when a key is *missing*, so the
    contract has to hold per language or translated users lose the escape hatch."""
    assert "/stop" in t(f"gateway.busy.{key}", lang=lang)


@pytest.mark.parametrize("key", NON_DEFERRAL_KEYS)
def test_non_deferral_replies_are_unchanged(key):
    """Redirect/interrupt tails defer nothing the user asked to cancel — no /stop noise."""
    assert "/stop" not in t(f"gateway.busy.{key}")


@pytest.mark.parametrize(
    "modes",
    [
        {"is_steer_mode": True, "is_queue_mode": False},
        {"is_steer_mode": False, "is_queue_mode": True},
    ],
)
def test_composed_busy_ack_names_stop(monkeypatch, modes):
    """The composed ack for a steered/queued follow-up carries the /stop hint end to end."""
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr("agent.onboarding.is_seen", lambda *_a, **_k: True)

    runner = _make_runner()
    message = runner._compose_busy_ack_message(
        _make_event(),
        now=0.0,
        _busy_state=None,
        running_agent=None,
        is_redirect_mode=False,
        demoted_for_subagents=False,
        demoted_for_compression=False,
        **modes,
    )
    assert "/stop" in message
