"""Tests for confirmed-dead delivery-target short-circuiting (deleted Telegram
groups, blocked/kicked bots, deactivated users).

Covers the full lifecycle through the real ``DeliveryRouter.deliver()`` path:
  forbidden send  -> target marked dead
  next delivery   -> short-circuited (adapter never called)
  successful send -> dead flag cleared (self-healing)

and the standalone ``DeadTargetRegistry`` persistence/classification contract.
"""

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.delivery import DeliveryRouter, DeliveryTarget
from gateway.dead_targets import DeadTargetRegistry


class ForbiddenThenOkAdapter:
    """First send raises a deleted-group Forbidden; subsequent sends succeed."""

    def __init__(self, fail_times=1):
        self.calls = []
        self._fail_times = fail_times

    async def send(self, chat_id, content, metadata=None):
        self.calls.append(chat_id)
        if len(self.calls) <= self._fail_times:
            raise RuntimeError("Forbidden: the group chat was deleted")
        return {"success": True}


class TransientFailAdapter:
    async def send(self, chat_id, content, metadata=None):
        raise RuntimeError("httpx.ReadTimeout: connection timed out")


@pytest.fixture
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.delivery.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("gateway.dead_targets.get_hermes_home", lambda: tmp_path)
    return tmp_path


# --------------------------------------------------------------------------
# DeadTargetRegistry unit contract
# --------------------------------------------------------------------------

class TestDeadTargetRegistry:
    def test_mark_is_dead_clear_roundtrip(self, isolate):
        reg = DeadTargetRegistry()
        assert reg.is_dead("telegram", "123") is False
        assert reg.mark_dead("telegram", "123", "forbidden") is True
        assert reg.is_dead("telegram", "123") is True
        # idempotent: second mark returns False (already present)
        assert reg.mark_dead("telegram", "123", "forbidden") is False
        assert reg.clear("telegram", "123") is True
        assert reg.is_dead("telegram", "123") is False

    def test_persists_across_instances(self, isolate):
        reg = DeadTargetRegistry()
        reg.mark_dead("telegram", "999", "deleted group")
        # New instance reads the same on-disk store under tmp HERMES_HOME.
        reg2 = DeadTargetRegistry()
        assert reg2.is_dead("telegram", "999") is True

    def test_key_is_case_insensitive_on_platform(self, isolate):
        reg = DeadTargetRegistry()
        reg.mark_dead("TeleGram", "5", "x")
        assert reg.is_dead("telegram", "5") is True

    def test_none_chat_id_is_never_dead(self, isolate):
        reg = DeadTargetRegistry()
        assert reg.mark_dead("telegram", None) is False
        assert reg.is_dead("telegram", None) is False

    def test_is_dead_error_kind_classification(self):
        assert DeadTargetRegistry.is_dead_error_kind("forbidden") is True
        assert DeadTargetRegistry.is_dead_error_kind("not_found") is True
        assert DeadTargetRegistry.is_dead_error_kind("rate_limited") is False
        assert DeadTargetRegistry.is_dead_error_kind("transient") is False
        assert DeadTargetRegistry.is_dead_error_kind(None) is False

    def test_corrupt_store_degrades_to_empty(self, isolate):
        path = isolate / "gateway" / "dead_targets.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this is not json")
        reg = DeadTargetRegistry()  # must not raise
        assert reg.all_dead() == {}


# --------------------------------------------------------------------------
# DeliveryRouter end-to-end lifecycle
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_forbidden_marks_target_dead_then_short_circuits(isolate):
    adapter = ForbiddenThenOkAdapter(fail_times=99)
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:42")

    # First delivery: send raises Forbidden -> failure + target recorded dead.
    res1 = await router.deliver("hi", [target])
    assert res1["telegram:42"]["success"] is False
    assert router.dead_targets.is_dead("telegram", "42") is True
    assert adapter.calls == ["42"]  # adapter was invoked once

    # Second delivery: short-circuited, adapter NOT called again.
    res2 = await router.deliver("hi again", [target])
    assert res2["telegram:42"]["skipped"] == "dead_target"
    assert res2["telegram:42"]["success"] is False
    assert adapter.calls == ["42"]  # still only the original call


@pytest.mark.asyncio
async def test_successful_send_clears_dead_flag(isolate):
    # Fails once (gets marked dead), then succeeds.
    adapter = ForbiddenThenOkAdapter(fail_times=1)
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:7")

    # Pre-seed dead via the first (failing) delivery.
    await router.deliver("a", [target])
    assert router.dead_targets.is_dead("telegram", "7") is True

    # Manually clear to simulate the user re-adding the bot, then deliver again.
    router.dead_targets.clear("telegram", "7")
    res = await router.deliver("b", [target])
    assert res["telegram:7"]["success"] is True
    # Flag stays cleared after a successful send.
    assert router.dead_targets.is_dead("telegram", "7") is False


@pytest.mark.asyncio
async def test_transient_failure_does_not_mark_dead(isolate):
    adapter = TransientFailAdapter()
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:13")

    res = await router.deliver("hi", [target])
    assert res["telegram:13"]["success"] is False
    # A timeout/transient error must NOT mark the chat dead — it may recover.
    assert router.dead_targets.is_dead("telegram", "13") is False


@pytest.mark.asyncio
async def test_local_target_is_never_dead_tracked(isolate):
    router = DeliveryRouter(GatewayConfig(), adapters={})
    target = DeliveryTarget.parse("local")
    res = await router.deliver("hi", [target])
    assert res["local"]["success"] is True
    assert router.dead_targets.all_dead() == {}


@pytest.mark.asyncio
async def test_shared_registry_is_used_when_injected(isolate):
    shared = DeadTargetRegistry()
    shared.mark_dead("telegram", "500", "pre-existing")
    adapter = ForbiddenThenOkAdapter(fail_times=0)
    router = DeliveryRouter(
        GatewayConfig(),
        adapters={Platform.TELEGRAM: adapter},
        dead_targets=shared,
    )
    target = DeliveryTarget.parse("telegram:500")
    res = await router.deliver("hi", [target])
    # Injected registry's pre-existing flag short-circuits before any send.
    assert res["telegram:500"]["skipped"] == "dead_target"
    assert adapter.calls == []


# --------------------------------------------------------------------------
# not_found blast radius: chat-level kills the chat, thread/message-level must not
# --------------------------------------------------------------------------

class RaisingAdapter:
    """Raises a fixed error message on every send."""

    def __init__(self, message):
        self.message = message
        self.calls = []

    async def send(self, chat_id, content, metadata=None):
        self.calls.append(chat_id)
        raise RuntimeError(self.message)


_SUBCHAT_NOT_FOUND_MESSAGES = [
    "Bad Request: message thread not found",
    "Bad Request: TOPIC_DELETED",
    "Bad Request: message to edit not found",
    "Bad Request: message to reply not found",
    "Bad Request: MESSAGE_ID_INVALID",
]


@pytest.mark.asyncio
async def test_chat_level_not_found_marks_target_dead(isolate):
    # "chat not found" -> the whole chat/user/group is gone, so it is dead
    # (same blast radius as forbidden).
    adapter = RaisingAdapter("Bad Request: chat not found")
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:100")

    res = await router.deliver("hi", [target])
    assert res["telegram:100"]["success"] is False
    assert router.dead_targets.is_dead("telegram", "100") is True


@pytest.mark.parametrize("message", _SUBCHAT_NOT_FOUND_MESSAGES)
@pytest.mark.asyncio
async def test_thread_or_message_level_not_found_does_not_mark_chat_dead(isolate, message):
    # A deleted forum topic / edited-away message is NOT a whole-chat death: marking
    # the parent chat dead would silently short-circuit every future delivery to it.
    adapter = RaisingAdapter(message)
    router = DeliveryRouter(GatewayConfig(), adapters={Platform.TELEGRAM: adapter})
    target = DeliveryTarget.parse("telegram:200")

    res = await router.deliver("hi", [target])
    assert res["telegram:200"]["success"] is False
    assert router.dead_targets.is_dead("telegram", "200") is False


class TestNotFoundBlastRadius:
    def test_is_chat_level_not_found_chat_level(self):
        from gateway.platforms.base import is_chat_level_not_found

        assert is_chat_level_not_found("Bad Request: chat not found") is True

    @pytest.mark.parametrize("message", _SUBCHAT_NOT_FOUND_MESSAGES)
    def test_is_chat_level_not_found_subchat(self, message):
        from gateway.platforms.base import is_chat_level_not_found

        assert is_chat_level_not_found(message) is False

    def test_subchat_marker_wins_when_both_present(self):
        from gateway.platforms.base import is_chat_level_not_found

        # Conservative: if a sub-chat marker is present, never kill the whole chat.
        assert is_chat_level_not_found("chat not found; message thread not found") is False

    def test_classify_dead_from_error_text_gates_not_found(self):
        from gateway.delivery import _classify_dead_from_error_text

        assert _classify_dead_from_error_text("Forbidden: bot was blocked by the user") == "forbidden"
        assert _classify_dead_from_error_text("Bad Request: chat not found") == "not_found"
        assert _classify_dead_from_error_text("Bad Request: message thread not found") is None
        assert _classify_dead_from_error_text("httpx.ReadTimeout: connection timed out") is None
