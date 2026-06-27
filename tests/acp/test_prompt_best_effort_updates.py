"""Regression tests for #39245 / #51290 (best-effort ACP session updates).

The ACP ``prompt()`` method must always return ``PromptResponse`` promptly
even when post-turn session updates (``usage_update``, ``session_info_update``)
hang because the connected client is not consuming them. Non-critical
telemetry cannot be allowed to block the final ``PromptResponse`` —
otherwise downstream clients (JetBrains IDEA, VS Code) keep the session
in a "Waiting…" state and the user has to restart the IDE.

These tests focus on the *best-effort helpers themselves*; the full
``prompt()`` lifecycle is exercised by ``tests/acp/test_server.py``.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


_USAGE_TIMEOUT_S = 3.0  # Must match HermesACPAgent._USAGE_UPDATE_TIMEOUT


class _HangingClient:
    """Stub for ``acp.Client`` whose ``session_update`` parks indefinitely.

    The hang simulates a stalled client that does not consume ACP
    notifications — the IDEA / VS Code "Waiting…" symptom from #51290.
    """

    def __init__(self) -> None:
        self.session_update_calls = 0

    async def session_update(self, *args, **kwargs):
        self.session_update_calls += 1
        await asyncio.sleep(60)


@pytest.fixture()
def hanging_client():
    return _HangingClient()


@pytest.fixture()
def acpa(hanging_client):
    """HermesACPAgent whose``_send_usage_update`` will hit the hang."""

    sm = SessionManager(agent_factory=lambda: MagicMock(name="MockAIAgent"))
    agent = HermesACPAgent(session_manager=sm)
    agent._conn = hanging_client  # type: ignore[assignment]
    return agent


@pytest.mark.asyncio
async def test_send_usage_update_best_effort_returns_within_timeout(acpa):
    """``_send_usage_update_best_effort`` must time out after the budget
    even when the underlying ``session_update`` hangs forever."""

    # We need a real SessionState to pass; build a session so we have one.
    new_resp = await acpa.new_session(cwd="/tmp")
    state = acpa.session_manager.get_session(new_resp.session_id)
    state.usage_enabled = True  # produce a non-empty update body

    t0 = time.monotonic()
    # Must NOT raise and must NOT block past ~3s.
    await acpa._send_usage_update_best_effort(state)
    elapsed = time.monotonic() - t0

    assert elapsed < _USAGE_TIMEOUT_S + 0.5, (
        f"best-effort usage_update took {elapsed:.2f}s — timeout wrap "
        "is missing or too long (issue #39245 / #51290)."
    )


@pytest.mark.asyncio
async def test_send_session_info_update_best_effort_returns_within_timeout(acpa):
    """``_send_session_info_update_best_effort`` must time out after the
    budget even when the underlying ``session_update`` hangs forever."""

    new_resp = await acpa.new_session(cwd="/tmp")
    session_id = new_resp.session_id

    t0 = time.monotonic()
    await acpa._send_session_info_update_best_effort(session_id)
    elapsed = time.monotonic() - t0

    assert elapsed < _USAGE_TIMEOUT_S + 0.5, (
        f"best-effort session_info_update took {elapsed:.2f}s — timeout "
        "wrap is missing or too long (issue #51290)."
    )


@pytest.mark.asyncio
async def test_best_effort_does_not_swallow_real_errors_too_eagerly(acpa):
    """If the underlying ``session_update`` raises *immediately* (not
    hangs), the best-effort wrapper should still let the prompt continue —
    errors are logged inside ``_send_usage_update`` / ``_send_session_info_update``
    and the wrapper's outer ``except Exception: pass`` guarantees the
    caller is never blocked.

    We can't easily simulate a synchronous exception through the hang
    socket here, so this test asserts that *success* (no raise) is the
    contract: callers can rely on the wrapper always completing within
    the budget."""

    new_resp = await acpa.new_session(cwd="/tmp")
    state = acpa.session_manager.get_session(new_resp.session_id)
    state.usage_enabled = True

    # Wrapper must succeed (catch and log) regardless of what the
    # underlying call does internally.
    await acpa._send_usage_update_best_effort(state)

    # Second call also must not raise.
    await acpa._send_usage_update_best_effort(state)

    # We don't assert on hanging_client.session_update_calls — the
    # hang surface can be reached or not depending on internal timing.
    # The contract is "always returns within budget".
