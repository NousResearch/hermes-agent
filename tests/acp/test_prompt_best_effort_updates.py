"""Regression tests for #39245 / #51290 (best-effort ACP session updates).

The ACP ``prompt()`` method must always return ``PromptResponse`` promptly
even when post-turn metadata session updates (``usage_update``,
``session_info_update``) hang because the connected client is not
consuming them. Non-critical metadata notifications cannot be allowed
to block the final ``PromptResponse`` — otherwise downstream clients
(JetBrains IDEA, VS Code) keep the session in a "Waiting…" state and
the user has to restart the IDE.

These tests cover two contracts:

1. The *_best_effort helpers log + return within ``_USAGE_UPDATE_TIMEOUT``
   when the underlying ``session_update`` hangs OR raises immediately
   (helpers never raise to the caller).
2. The full ``prompt()`` lifecycle still returns ``PromptResponse`` within
   the budget when post-turn metadata notifications hang on the wire —
   the wrap on the final pre-return ``usage_update`` is what saves the
   contract. (Content-bearing awaits — agent-message delivery, queued
   message echo — are intentionally *not* wrapped; timing them out
   would silently drop the response. See the narrow contract in the
   PR description.)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


_USAGE_TIMEOUT_S = 3.0  # Must match HermesACPAgent._USAGE_UPDATE_TIMEOUT
# Slack for pytest-asyncio scheduling jitter + the helper's outer
# ``except Exception: pass`` after the timeout fires.
_PROMPT_BUDGET_S = _USAGE_TIMEOUT_S + 1.5


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


class _RaisingClient:
    """Stub for ``acp.Client`` whose ``session_update`` raises immediately.

    Mirrors the "client crashes mid-notification" failure mode: the RPC
    fails synchronously rather than hanging. The best-effort wrapper must
    log internally and still return without raising to the caller.
    """

    def __init__(self) -> None:
        self.session_update_calls = 0

    async def session_update(self, *args, **kwargs):
        self.session_update_calls += 1
        raise RuntimeError("simulated ACP client crash on session_update")


@pytest.fixture()
def hanging_client():
    return _HangingClient()


@pytest.fixture()
def raising_client():
    return _RaisingClient()


@pytest.fixture()
def acpa(request):
    """HermesACPAgent with a session_manager backed by a MagicMock AIAgent.

    The ``client`` is parameterized via the indirect fixture so individual
    tests can request hang / raise behavior on ``session_update``.
    """
    client = request.getfixturevalue(request.param)

    def _factory():
        m = MagicMock(name="MockAIAgent")
        m.session_id = "fake-internal-id"
        return m

    sm = SessionManager(agent_factory=_factory)
    agent = HermesACPAgent(session_manager=sm)
    agent._conn = client  # type: ignore[assignment]
    return agent


# ---------------------------------------------------------------------------
# Helper-level contracts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("acpa", ["hanging_client"], indirect=True)
async def test_send_usage_update_best_effort_returns_within_timeout(acpa):
    """``_send_usage_update_best_effort`` must time out after the budget
    even when the underlying ``session_update`` hangs forever."""

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
@pytest.mark.parametrize("acpa", ["hanging_client"], indirect=True)
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
@pytest.mark.parametrize("acpa", ["raising_client"], indirect=True)
async def test_best_effort_swallows_immediate_exception(acpa):
    """When the underlying ``session_update`` raises immediately (not hangs),
    the best-effort wrapper must let the helper return without raising to
    the caller. Errors are already logged inside ``_send_usage_update`` /
    ``_send_session_info_update``; the wrapper's outer
    ``except Exception: pass`` is the contract under test here.

    Separate test from the hang case so a regression in either branch
    points at the right root cause.
    """

    new_resp = await acpa.new_session(cwd="/tmp")
    state = acpa.session_manager.get_session(new_resp.session_id)
    state.usage_enabled = True

    t0 = time.monotonic()
    await acpa._send_usage_update_best_effort(state)
    elapsed = time.monotonic() - t0

    # Must complete immediately (not wait for the budget) because the
    # underlying call raises synchronously rather than hanging.
    assert elapsed < _USAGE_TIMEOUT_S, (
        f"best-effort wrapper blocked {elapsed:.2f}s after synchronous "
        "exception — outer ``except Exception: pass`` is missing or too late."
    )

    t0 = time.monotonic()
    await acpa._send_session_info_update_best_effort(new_resp.session_id)
    elapsed = time.monotonic() - t0
    assert elapsed < _USAGE_TIMEOUT_S, (
        f"session-info best-effort blocked {elapsed:.2f}s after synchronous "
        "exception — outer ``except Exception: pass`` is missing or too late."
    )


# ---------------------------------------------------------------------------
# Lifecycle regression — drive prompt() with a hanging metadata stub
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("acpa", ["hanging_client"], indirect=True)
async def test_prompt_returns_prompt_response_when_metadata_update_hangs(acpa):
    """End-to-end lifecycle regression for #39245 / #51290.

    Drive ``prompt()`` against a sim agent that returns an empty
    ``final_response`` (which skips the un-wrapped content-delivery
    ``session_update`` await), so the only post-turn ``session_update``
    on the path is the patched ``_send_usage_update_best_effort`` at
    line ~1736. With a hanging client stub, ``prompt()`` must still
    return ``PromptResponse`` inside ~budget seconds — proving the
    patch actually saves the prompt lifecycle, not just the helpers
    in isolation.

    NOTE — this test exercises the *narrow* contract covered by the
    patch: usage_update + session_info_update metadata notifications.
    Content-delivery awaits (final agent message, queued-prompt echo)
    remain un-wrapped by design; timing them out would silently drop
    the response.
    """

    # Mock agent factory: the production prompt() calls
    # ``agent.run_conversation(...)`` synchronously from inside an
    # executor thread (``loop.run_in_executor(_executor, ctx.run,
    # _run_agent)`` → ``agent.run_conversation(...)`` is a plain sync
    # attribute access). So ``run_conversation`` must be a regular
    # MagicMock returning the dict — NOT an AsyncMock (an AsyncMock
    # returns a coroutine object that the caller never awaits, and
    # the subsequent ``result.get(...)`` raises).
    inner = MagicMock(name="MockAIAgent")
    inner.run_conversation = MagicMock(
        return_value={
            # Empty final_response → skip `if final_response and conn and ...`
            # content-delivery block at line ~1691, leaving only the
            # patched usage_update_best_effort at ~1736.
            "final_response": "",
            "messages": [],
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }
    )
    inner.session_id = "fake-internal-id"

    sm = SessionManager(agent_factory=lambda: inner)
    # Rebuild acpa agent with this session_manager, retaining hanging conn.
    agent = HermesACPAgent(session_manager=sm)
    agent._conn = acpa._conn  # type: ignore[assignment]

    new_resp = await agent.new_session(cwd="/tmp")
    state = agent.session_manager.get_session(new_resp.session_id)
    # Force the usage-update body to be non-None so ``_build_usage_update``
    # emits a payload (otherwise the helper returns early before ever
    # touching the hung ``session_update``).
    state.usage_enabled = True

    from acp.schema import TextContentBlock

    t0 = time.monotonic()
    response = await agent.prompt(
        prompt=[TextContentBlock(type="text", text="/help")],
        session_id=new_resp.session_id,
    )
    elapsed = time.monotonic() - t0

    # The slash command ``/help`` doesn't match ``_handle_slash_command``,
    # but that's OK — the regular path still exercises the wrapped
    # ``_send_usage_update_best_effort`` at ~1736.
    assert response is not None, "prompt() must return a PromptResponse object"
    assert elapsed < _PROMPT_BUDGET_S, (
        f"prompt() took {elapsed:.2f}s with a hanging metadata stub — "
        "best-effort wrap is not saving the prompt lifecycle "
        "(regression of #39245 / #51290 narrow contract)."
    )
