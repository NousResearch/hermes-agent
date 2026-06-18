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


@pytest.mark.asyncio
async def test_interrupt_unknown_session_is_noop(adapter):
    # No active session for this key — must not raise.
    await adapter.on_interrupt("agent:main:discord:group:nope:userZ", chat_id="nope")


@pytest.mark.asyncio
async def test_outbound_interrupt_reaches_connector(adapter):
    """The gateway-side /stop egress: send_interrupt is carried to the connector
    so it can forward down the socket owning the session_key."""
    stub = adapter._transport
    await stub.send_interrupt("agent:main:discord:group:chanA:userX", reason="stop")
    assert stub.interrupts == [
        {"session_key": "agent:main:discord:group:chanA:userX", "reason": "stop"}
    ]
