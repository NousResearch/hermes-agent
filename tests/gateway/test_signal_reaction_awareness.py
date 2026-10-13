"""Contract tests for the isolated Signal reaction-awareness lane."""
import asyncio
import json
import sqlite3

import pytest

from gateway.platforms.signal_reaction_awareness import ReactionAwarenessLane


class FakeProvider:
    model = "test/model"

    def __init__(self):
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"text": json.dumps({"note": "The reaction acknowledges the message."})}


def test_signal_reaction_identifiers_require_protocol_shaped_values():
    from gateway.platforms.signal import _safe_reaction_identifier

    synthetic_e164 = "+" + "0" * 7
    assert _safe_reaction_identifier(synthetic_e164) == synthetic_e164
    assert _safe_reaction_identifier("00000000-0000-0000-0000-000000000001") == "00000000-0000-0000-0000-000000000001"
    assert _safe_reaction_identifier("+123") is None
    assert _safe_reaction_identifier("owner name") is None


@pytest.mark.asyncio
async def test_authenticated_dm_reaction_completes_silently_and_is_profile_scoped():
    db = sqlite3.connect(":memory:")
    provider = FakeProvider()
    lane = ReactionAwarenessLane(db, model_name="test/model", model_client=provider)
    assert lane.register_target(profile="profile-a", scope="dm:chat-a", timestamp=101,
                                author="owner-a", body="A prior message")
    assert lane.enqueue({
        "owner_authorized": True, "chat_type": "dm", "profile": "profile-a", "scope": "dm:chat-a",
        "event_id": "reaction-1", "sender": "owner-a", "target_author": "owner-a",
        "target_timestamp": 101, "action": "added", "emoji": "👍", "event_timestamp": 102,
    }) == "queued"
    assert await lane.process_one("profile-a", "dm:chat-a", authorized_sender="owner-a") == "completed"
    assert lane.outbox_status("profile-a", "dm:chat-a", "reaction-1") == "completed"
    assert lane.notes("profile-a", "dm:chat-a")[0]["note"] == "The reaction acknowledges the message."
    assert not lane.notes("profile-b", "dm:chat-a")
    assert provider.calls[0]["tools"] == []
    db.close()


@pytest.mark.asyncio
async def test_signal_adapter_reaction_is_silent_and_completed_note_reaches_later_dm(monkeypatch, tmp_path):
    from gateway.config import PlatformConfig
    from gateway.platforms.signal import SignalAdapter
    import gateway.platforms.signal_reaction_awareness_runtime as runtime
    from hermes_constants import get_hermes_home

    owner = "00000000-0000-0000-0000-000000000002"
    bot = "00000000-0000-0000-0000-000000000001"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", owner)
    adapter = SignalAdapter(PlatformConfig(enabled=True, extra={"account": bot}))
    adapter.dm_allow_from = {owner}
    adapter.gateway_runner = object()
    adapter.set_authorization_check(lambda user_id, *_args: user_id == owner)
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"model": {"default": "test/model"}})
    provider = FakeProvider()
    monkeypatch.setattr(runtime.ConfiguredAIAgentModelClient, "from_gateway_runner",
                        classmethod(lambda *_args: provider))
    await runtime.start_awareness(adapter)
    sent, events = [], []
    async def quiet(*_args, **_kwargs):
        return None
    adapter._stop_typing_indicator = quiet
    async def successful_send(method, params, **_kwargs):
        sent.append((method, params))
        return {"timestamp": 200, "results": [{"type": "SUCCESS"}]}
    adapter._rpc = successful_send
    assert (await adapter.send(owner, "A confirmed outbound target.")).success
    sent.clear()
    async def capture(event):
        events.append(event)
    adapter.handle_message = capture

    await adapter._handle_envelope({"envelope": {"sourceUuid": owner, "timestamp": 201,
        "dataMessage": {"message": "Please remember this preference."}}})
    await adapter._handle_envelope({"envelope": {"sourceUuid": owner, "timestamp": 202,
        "dataMessage": {"reaction": {"emoji": "👍", "targetAuthorUuid": bot,
                                      "targetSentTimestamp": 200, "isRemove": False}}}})
    assert len(events) == 1
    service = runtime._services[id(adapter)]
    await asyncio.gather(*tuple(service.tasks.values()))
    scope = adapter._source_session_key(events[0].source)
    assert service.lane.outbox_status(str(get_hermes_home().resolve()), scope, f"{owner}:202") == "completed"
    assert len(provider.calls) == 1 and provider.calls[0]["tools"] == []
    assert sent == []

    await adapter._handle_envelope({"envelope": {"sourceUuid": owner, "timestamp": 203,
        "dataMessage": {"message": "What did I say?"}}})
    assert len(events) == 2
    assert events[1].metadata["signal_reaction_awareness"][0]["note"] == "The reaction acknowledges the message."
    await runtime.stop_awareness(adapter)
