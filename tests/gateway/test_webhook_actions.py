"""Task discussion controls preserve existing session ownership and never approve execution."""
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import SendResult
from gateway.session import SessionStore, SessionSource
from gateway.platforms import webhook_actions as actions
from tools import clarify_gateway as clarify


def owner_for(home, profile=None):
    store = SessionStore(home / "sessions", GatewayConfig(multiplex_profiles=True))
    source = SessionSource(platform=Platform.WHATSAPP, chat_id="byron", user_id="byron", chat_type="dm", profile=profile)
    entry = store.get_or_create_session(source)
    target = SimpleNamespace(send_clarify=AsyncMock(return_value=SendResult(success=True, message_id="poll-1")))
    runner = SimpleNamespace(session_store=store, _restored_source=lambda entry: entry.origin,
                             _is_user_authorized=lambda source: source.user_id == "byron",
                             _delivery_adapter_for=lambda source: target)
    owner = SimpleNamespace(gateway_runner=runner, _background_tasks=set(), _profile_scope=lambda profile: nullcontext())
    return owner, target, entry


async def stop(owner, entry):
    for task in list(owner._background_tasks):
        task.cancel()
    await asyncio.gather(*owner._background_tasks, return_exceptions=True)
    clarify.clear_session(entry.session_key)


def test_action_owner_binding_delivery_dedup_and_admission(tmp_path, monkeypatch):
    async def scenario():
        owner, target, entry = owner_for(tmp_path)
        admit = AsyncMock()
        monkeypatch.setattr(actions, "admit_internal_event", admit)
        refs = {"eventId": "event-1", "taskId": "task-1", "cardId": "t_card", "sourceSessionId": "native-1"}
        delivery = {"payload": {"discussion_action": refs}, "profile": None}
        result = await actions.deliver(owner, target, Platform.WHATSAPP, "byron", None, "Recovery needs attention", delivery)
        # Same chat in another owning profile cannot borrow the default action or callback.
        other, other_target, other_entry = owner_for(tmp_path / "other", profile="other")
        other_delivery = {"payload": {"discussion_action": {**refs, "eventId": "event-other"}}, "profile": "other"}
        assert (await actions.deliver(other, other_target, Platform.WHATSAPP, "byron", None, "Other profile", other_delivery)).success
        assert other_entry.session_key != entry.session_key
        assert owner.gateway_runner.session_store.get_session_metadata(entry.session_key, actions.KEY)["binding"] == refs
        await stop(other, other_entry)
        assert result.success
        assert (await actions.deliver(owner, target, Platform.WHATSAPP, "byron", None, "Same alert", delivery)).success
        assert target.send_clarify.await_count == 1
        other_alert = {"payload": {"discussion_action": {**refs, "eventId": "another-alert"}}}
        assert await actions.deliver(owner, target, Platform.WHATSAPP, "byron", None, "Another task needs input", other_alert) is None
        assert target.send_clarify.await_count == 1
        record = owner.gateway_runner.session_store.get_session_metadata(entry.session_key, actions.KEY)
        assert not clarify.resolve_gateway_clarify(record["clarifyId"], actions.CHOICES[0], user_id="other")
        assert clarify.resolve_gateway_clarify(record["clarifyId"], actions.CHOICES[0], user_id="byron")
        await asyncio.gather(*owner._background_tasks)
        event = admit.await_args.args[1]
        assert event.metadata["gateway_session_id"] == entry.session_id
        assert "t_card" in event.text and "not approval" in event.text
        assert owner.gateway_runner.session_store.get_session_metadata(entry.session_key, actions.KEY)["state"] == "admitted"
        await stop(owner, entry)
    async def bounded():
        try:
            await asyncio.wait_for(scenario(), timeout=10)
        finally:
            for key in list(clarify._session_index):
                clarify.clear_session(key)
    asyncio.run(bounded())


def test_restored_action_keeps_exact_session_and_sends_nothing(tmp_path, monkeypatch):
    async def scenario():
        owner, target, entry = owner_for(tmp_path)
        delivery = {"payload": {"discussion_action": {"eventId": "event-2", "taskId": "task-1", "cardId": "t_card", "sourceSessionId": "native-1"}}}
        assert (await actions.deliver(owner, target, Platform.WHATSAPP, "byron", None, "Needs attention", delivery)).success
        await stop(owner, entry)
        # Reload through the real persisted session index, not an in-memory imitation.
        owner.gateway_runner.session_store = SessionStore(tmp_path / "sessions", GatewayConfig())
        await actions.restore(owner)
        assert target.send_clarify.await_count == 1
        restored = owner.gateway_runner.session_store.lookup_by_session_id(entry.session_id)
        assert restored.metadata[actions.KEY]["binding"]["cardId"] == "t_card"
        assert clarify.has_pending(entry.session_key)
        await stop(owner, entry)
    async def bounded():
        try:
            await asyncio.wait_for(scenario(), timeout=10)
        finally:
            for key in list(clarify._session_index):
                clarify.clear_session(key)
    asyncio.run(bounded())
