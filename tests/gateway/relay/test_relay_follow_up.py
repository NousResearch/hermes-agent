"""A2 outbound capability action: the token-less ``follow_up`` op.

Proves the gateway can act on a shared-identity capability (e.g. a Discord
interaction follow-up token) WITHOUT ever holding the credential: it names the
session it is in plus the capability ``kind``, and the connector resolves the
real value from its vault and egresses. See gateway/relay/transport.py
(send_follow_up) and docs/relay-connector-contract.md §4.

The gateway side is what's exercised here (against the stub connector); the
connector's resolve + tenant-match enforcement lives in the connector repo
(resolveOutboundCapability). The key gateway-side guarantees:
  - the wire action carries NO token (only session_key + kind + content),
  - success/failure surfaces from the connector's resolve result,
  - a failed resolve (absent/expired/tenant mismatch) returns success=False
    with nothing for the gateway to retry with.
"""

from __future__ import annotations

import pytest

from gateway.config import PlatformConfig
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

from tests.gateway.relay.stub_connector import StubConnector


def _discord_descriptor() -> CapabilityDescriptor:
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
def wired():
    stub = StubConnector(_discord_descriptor())
    adapter = RelayAdapter(PlatformConfig(), _discord_descriptor(), transport=stub)
    return adapter, stub


@pytest.mark.asyncio
async def test_follow_up_round_trips_without_a_token(wired):
    adapter, stub = wired
    await adapter.connect()
    stub.next_follow_up_result = {"success": True, "message_id": "fu-7"}

    result = await adapter.send_follow_up(
        session_key="agent:main:discord:group:chanA:userX",
        kind="discord.interaction_token",
        content="here is your follow-up",
    )

    assert result.success is True
    assert result.message_id == "fu-7"
    assert len(stub.follow_ups) == 1
    action = stub.follow_ups[0]
    assert action["op"] == "follow_up"
    assert action["session_key"] == "agent:main:discord:group:chanA:userX"
    assert action["kind"] == "discord.interaction_token"
    assert action["content"] == "here is your follow-up"


@pytest.mark.asyncio
async def test_follow_up_wire_action_carries_no_credential(wired):
    """The action dict must carry only session refs — no credential VALUE.

    Note the capability ``kind`` legitimately names the credential type
    (e.g. ``"discord.interaction_token"``) — that's a reference, not the secret.
    The guarantee is structural: the action has exactly the token-less semantic
    fields, and no field holds an actual credential value.
    """
    adapter, stub = wired
    await adapter.connect()
    await adapter.send_follow_up(
        session_key="sess-1", kind="discord.interaction_token", content="x", metadata={"a": 1}
    )
    action = stub.follow_ups[0]
    # Exactly the token-less semantic fields (+ metadata); no value/secret field.
    assert set(action.keys()) == {"op", "session_key", "kind", "content", "metadata"}
    # No field NAMES a credential carrier (the kind string is a type ref, allowed).
    assert "value" not in action
    assert "token" not in action
    assert "secret" not in action
    assert "credential" not in action


@pytest.mark.asyncio
async def test_follow_up_failure_surfaces_when_capability_unresolvable(wired):
    """Connector couldn't resolve (absent/expired/tenant mismatch) -> success=False."""
    adapter, stub = wired
    await adapter.connect()
    stub.next_follow_up_result = {"success": False, "error": "capability absent or tenant mismatch"}

    result = await adapter.send_follow_up(
        session_key="sess-1", kind="discord.interaction_token", content="x"
    )

    assert result.success is False
    assert result.message_id is None
    assert "tenant mismatch" in (result.error or "")


@pytest.mark.asyncio
async def test_follow_up_without_transport_fails_cleanly():
    adapter = RelayAdapter(PlatformConfig(), _discord_descriptor(), transport=None)
    result = await adapter.send_follow_up(session_key="s", kind="k", content="c")
    assert result.success is False
    assert result.error == "no transport"
