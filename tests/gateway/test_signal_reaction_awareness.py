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
    from plugins.platforms.signal.adapter import _safe_reaction_identifier

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
    from plugins.platforms.signal.adapter import SignalAdapter
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
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {
        "model": {"default": "test/model"},
        "signal": {"reaction_awareness": True},
    })
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


@pytest.mark.asyncio
async def test_reaction_awareness_is_disabled_without_explicit_config(monkeypatch, tmp_path):
    from gateway.config import PlatformConfig
    from plugins.platforms.signal.adapter import SignalAdapter
    import gateway.platforms.signal_reaction_awareness_runtime as runtime

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    owner = "00000000-0000-0000-0000-000000000002"
    adapter = SignalAdapter(PlatformConfig(enabled=True, extra={"account": "00000000-0000-0000-0000-000000000001"}))
    adapter.dm_allow_from = {owner}
    adapter.gateway_runner = object()
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"model": {"default": "test/model"}})

    def unexpected_model_call(*_args, **_kwargs):
        pytest.fail("disabled reaction awareness must not resolve or construct a model")

    monkeypatch.setattr(runtime.ConfiguredAIAgentModelClient, "from_gateway_runner", unexpected_model_call)
    await runtime.start_awareness(adapter)
    assert adapter._reaction_awareness_callback is None
    assert adapter._reaction_awareness_note_getter is None
    assert not (tmp_path / "signal-reaction-awareness.sqlite3").exists()


@pytest.mark.asyncio
async def test_runtime_prunes_expired_rows_on_startup_and_during_idle_maintenance(monkeypatch, tmp_path):
    import time
    from gateway.config import PlatformConfig
    from plugins.platforms.signal.adapter import SignalAdapter
    from gateway.platforms.signal_reaction_awareness import ReactionAwarenessLane
    import gateway.platforms.signal_reaction_awareness_runtime as runtime

    owner = "00000000-0000-0000-0000-000000000002"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(runtime, "_MAINTENANCE_INTERVAL_SECONDS", 0.01)
    db_path = tmp_path / runtime._DB_NAME
    db = sqlite3.connect(db_path)
    lane = ReactionAwarenessLane(db, model_name="m", model_client=FakeProvider())
    old = time.time() - 31 * 86400
    lane.db.execute("INSERT INTO awareness_targets VALUES(?,?,?,?,?,?)", (str(tmp_path), "scope", 1, owner, "old", old))
    lane.db.execute("INSERT INTO awareness_outbox(profile,scope,event_id,target_timestamp,sender,target_author,action,emoji,event_timestamp,revision,status,lease_until,attempts,created,updated) VALUES(?,?,?,?,?,?,?,?,?,1,'completed',0,0,?,?)", (str(tmp_path), "scope", "old", 1, owner, owner, "added", "👍", 2, old, old))
    lane.db.execute("INSERT INTO awareness_latest VALUES(?,?,?,?,?,?,?,?)", (str(tmp_path), "scope", 1, owner, 2, "old", 1, old))
    lane.db.execute("INSERT INTO awareness_notes VALUES(?,?,?,?,?,?,?)", (str(tmp_path), "scope", "old", 1, 2, "old note", old))
    db.commit()
    db.close()

    adapter = SignalAdapter(PlatformConfig(enabled=True, extra={"account": "00000000-0000-0000-0000-000000000001"}))
    adapter.dm_allow_from = {owner}
    adapter.gateway_runner = object()
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {
        "model": {"default": "test/model"}, "signal": {"reaction_awareness": True},
    })
    provider = FakeProvider()
    monkeypatch.setattr(runtime.ConfiguredAIAgentModelClient, "from_gateway_runner",
                        classmethod(lambda *_args: provider))
    await runtime.start_awareness(adapter)
    service = runtime._services[id(adapter)]
    assert all(service.lane.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
               for table in ("awareness_targets", "awareness_outbox", "awareness_latest", "awareness_notes"))

    assert service.lane.register_target(profile=str(tmp_path), scope="idle", timestamp=3,
                                        author=owner, body="will expire without traffic")
    service.lane.db.execute("UPDATE awareness_targets SET created=? WHERE scope='idle'", (old,))
    service.lane.db.commit()
    from tests.waiting import async_wait_until
    await async_wait_until(
        lambda: service.lane.db.execute("SELECT COUNT(*) FROM awareness_targets WHERE scope='idle'").fetchone()[0] == 0,
        timeout=2,
    )
    assert provider.calls == []
    maintenance_task = service.maintenance_task
    assert maintenance_task is not None
    await runtime.stop_awareness(adapter)
    assert maintenance_task.cancelled()
