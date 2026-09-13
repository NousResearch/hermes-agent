"""End-to-end buffer→surface delivery for provider stall-watchdog notices.

``tests/gateway/test_telegram_noise_filter.py`` asserts the filter *predicate*.
This module drives the REAL path a notice travels on terminal failure:

    agent._buffer_status(...)          # agent/chat_completion_helpers.py
      -> agent._flush_status_buffer()  # run_agent.py, on retry exhaustion
        -> agent._emit_status(...)     # run_agent.py
          -> agent.status_callback("lifecycle", ...)
            -> _prepare_gateway_status_message(...)   # gateway/run.py (real)
              -> adapter.send(...)                    # stubbed sink

Only the adapter sink is stubbed, so a regression that keeps the regex green
while changing what actually reaches a chat bubble fails here.
"""

from __future__ import annotations

import pytest

from gateway.run import _prepare_gateway_status_message
from run_agent import AIAgent
from tests.gateway.test_telegram_noise_filter import (
    BUFFERED_STATUS_MUST_SURVIVE,
    CHAT_PLATFORMS,
    PROVIDER_WATCHDOG_KILL_NOTICES,
)


def _agent_wired_to_surface(platform: str):
    """A bare AIAgent whose status_callback is the real gateway filter.

    Mirrors ``gateway/run.py:_status_callback_sync``: prepare the message, drop
    it when the filter returns None, otherwise hand it to the adapter. Returns
    ``(agent, bubbles, stdout_lines)`` — ``bubbles`` is what a chat user would
    see, ``stdout_lines`` is the operator-facing CLI/log stream.
    """
    agent = object.__new__(AIAgent)
    agent.log_prefix = ""
    agent.suppress_status_output = False
    agent._mute_post_response = False
    agent._executing_tools = False

    bubbles: list[str] = []
    stdout_lines: list[str] = []
    agent._print_fn = lambda *args, **kwargs: stdout_lines.append(
        " ".join(str(a) for a in args)
    )

    def _status_callback(event_type: str, message: str) -> None:
        prepared = _prepare_gateway_status_message(platform, event_type, message)
        if prepared is None:
            return
        bubbles.append(prepared)

    agent.status_callback = _status_callback
    return agent, bubbles, stdout_lines


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
@pytest.mark.parametrize(
    "notice", PROVIDER_WATCHDOG_KILL_NOTICES, ids=lambda m: m.strip()[:48]
)
def test_terminal_failure_keeps_watchdog_notice_out_of_chat(platform, notice):
    """A flushed watchdog kill notice reaches stdout/logs but never a bubble."""
    agent, bubbles, stdout_lines = _agent_wired_to_surface(platform)

    agent._buffer_status(notice)
    # Buffered only — nothing emitted while the retry loop is still trying.
    assert bubbles == []
    assert stdout_lines == []

    # Terminal failure replays the buffer.
    agent._flush_status_buffer()

    assert bubbles == []
    # The operator surface still gets it — this is suppression, not deletion.
    assert stdout_lines == [notice]


@pytest.mark.parametrize("platform", CHAT_PLATFORMS)
def test_terminal_failure_delivers_actionable_notices_only(platform):
    """A realistic terminal ladder: noise dropped, actionable lines delivered.

    The buffer holds what a real exhausted turn accumulates — watchdog kills
    interleaved with the fallback switch and the final give-up line. The chat
    user must end up with exactly the actionable subset, in order.
    """
    agent, bubbles, stdout_lines = _agent_wired_to_surface(platform)

    fallback_notice, terminal_notice = (
        BUFFERED_STATUS_MUST_SURVIVE[1],
        BUFFERED_STATUS_MUST_SURVIVE[0],
    )
    for message in (
        PROVIDER_WATCHDOG_KILL_NOTICES[2],  # codex TTFB kill
        PROVIDER_WATCHDOG_KILL_NOTICES[4],  # codex post-TTFB idle kill
        fallback_notice,
        PROVIDER_WATCHDOG_KILL_NOTICES[1],  # streaming stale kill
        terminal_notice,
    ):
        agent._buffer_status(message)

    agent._flush_status_buffer()

    assert bubbles == [fallback_notice, terminal_notice]
    # Every line — noise included — is still on the operator surface.
    assert len(stdout_lines) == 5


@pytest.mark.parametrize(
    "notice", PROVIDER_WATCHDOG_KILL_NOTICES, ids=lambda m: m.strip()[:48]
)
def test_successful_recovery_emits_nothing_anywhere(notice):
    """When the retry loop recovers, the buffer is dropped, not flushed."""
    agent, bubbles, stdout_lines = _agent_wired_to_surface("telegram")

    agent._buffer_status(notice)
    agent._clear_status_buffer()
    agent._flush_status_buffer()

    assert bubbles == []
    assert stdout_lines == []


@pytest.mark.parametrize(
    "notice", PROVIDER_WATCHDOG_KILL_NOTICES, ids=lambda m: m.strip()[:48]
)
def test_programmatic_surface_receives_watchdog_notice_as_bubble(notice):
    """API/webhook consumers keep the raw diagnostic on the delivery path too."""
    agent, bubbles, _ = _agent_wired_to_surface("api_server")

    agent._buffer_status(notice)
    agent._flush_status_buffer()

    assert bubbles == [notice]
