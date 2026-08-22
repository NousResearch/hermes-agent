"""Relay /stop interrupt routing (relay Phase 1, Task 1.4).

Proves a connector-delivered mid-turn interrupt reaches the existing per-session
interrupt mechanism and cancels exactly the targeted session_key's turn — never
a sibling's. Mirrors the isolation discipline of test_stop_thread_sibling.py.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.config import PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

from tests.gateway.relay.stub_connector import StubConnector


def _desc() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="discord",
        label="Discord",
        max_message_length=2000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="discord",
        len_unit="chars",
    )


def _slack_desc() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="slack",
        label="Slack",
        max_message_length=4000,
        supports_draft_streaming=True,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="slack",
        len_unit="chars",
    )


@pytest.fixture
def adapter():
    return RelayAdapter(PlatformConfig(), _desc(), transport=StubConnector(_desc()))


@pytest.mark.asyncio
async def test_interrupt_sets_only_target_session_event(adapter):
    key_a = "agent:main:discord:group:chanA:userX"
    key_b = "agent:main:discord:group:chanB:userY"
    ev_a = asyncio.Event()
    ev_b = asyncio.Event()
    adapter._active_sessions[key_a] = ev_a
    adapter._active_sessions[key_b] = ev_b

    await adapter.on_interrupt(key_a, chat_id="chanA")

    assert ev_a.is_set() is True, "target session's interrupt Event must be set"
    assert ev_b.is_set() is False, "sibling session must be untouched"


class TestInterruptDoesNotBlockOnStopTypingSend:
    """RelayAdapter.on_interrupt runs ON the connector's read loop (an
    interrupt_inbound frame is dispatched to it directly from
    _handle_frame). stop_typing() is Slack-gated — for every other platform
    it no-ops — and for Slack it sends an outbound "typing" frame through
    the SAME transport this handler is invoked from. Awaiting that send
    self-deadlocks the read loop for the full outbound timeout, because the
    frame's own outbound_result can only be resolved by a later call to the
    very read loop this coroutine would be blocking. Same shape as the
    lifecycle-ack deadlock fixed for prompt-response sends.
    """

    def _slack_adapter(self) -> RelayAdapter:
        desc = _slack_desc()
        a = RelayAdapter(PlatformConfig(), desc, transport=StubConnector(desc))
        a._platform_by_chat["chanA"] = "slack"
        return a

    @pytest.mark.asyncio
    async def test_on_interrupt_returns_promptly_even_if_stop_typing_send_hangs(self):
        a = self._slack_adapter()
        key_a = "agent:main:slack:group:chanA:userX"
        ev_a = asyncio.Event()
        a._active_sessions[key_a] = ev_a

        gate = asyncio.Event()
        orig = a._transport.send_outbound

        async def gated_send(frame, platform=None):
            if frame.get("op") == "typing":
                await gate.wait()  # simulate outbound_result not readable yet
            return await orig(frame, platform=platform)

        a._transport.send_outbound = gated_send

        # Pre-fix this hangs until the gate opens (deadlock shape) and the
        # wait_for trips with TimeoutError. Post-fix it returns promptly,
        # and — critically — the interrupt Event is ALREADY set by the time
        # it does, because that half is synchronous and inline.
        await asyncio.wait_for(a.on_interrupt(key_a, chat_id="chanA"), timeout=1.0)
        assert ev_a.is_set() is True, (
            "the turn must be cancelled even though the typing-clear send is stuck"
        )

        # Release the gate; the background typing-clear must still go out.
        gate.set()
        await asyncio.sleep(0.05)
        typing_frames = [f for f in a._transport.sent if f.get("op") == "typing"]
        assert typing_frames, "background typing-clear was never sent"

    @pytest.mark.asyncio
    async def test_on_interrupt_still_clears_typing_when_the_send_resolves_promptly(self):
        """Regression contract: deferring the send must not silently drop it
        on the fast path (no hang at all)."""
        a = self._slack_adapter()
        key_a = "agent:main:slack:group:chanA:userX"
        a._active_sessions[key_a] = asyncio.Event()

        await a.on_interrupt(key_a, chat_id="chanA")
        await asyncio.sleep(0.05)  # yield once for the background task

        typing_frames = [f for f in a._transport.sent if f.get("op") == "typing"]
        assert typing_frames, "typing-clear frame should still be sent on the fast path"


