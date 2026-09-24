"""Extracted presentation + real-loader/admission contracts replacing the old UI owner tests.

The prior tests are preserved in the baseline archive. r2 deliberately replaces
edit-to-create and stale-receipt-discard assertions with no-recreate and exact settlement.
"""
import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
import pytest_asyncio
pytest.importorskip(
    "hermes_telegram_experience",
    reason="optional external-plugin integration; exercised by plugin repository CI",
)
from telegram.error import BadRequest, RetryAfter, TimedOut

from gateway.config import PlatformConfig
from gateway.live_todo import TodoBinding, DeliveryStatus
from hermes_cli.plugins import get_plugin_manager, PluginContext
from hermes_telegram_experience import register
from hermes_telegram_experience.render import render_todo_progress
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools.todo_tool import TodoStore, todo_tool


class BotBoundary:
    def __init__(self):
        self.calls = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.hold = False
        self.error = None
        self.receipt = True

    async def send_message(self, **kwargs):
        return await self._call("send", kwargs)

    async def edit_message_text(self, **kwargs):
        return await self._call("edit", kwargs)

    async def _call(self, kind, kwargs):
        self.calls.append((kind, kwargs))
        self.entered.set()
        if self.hold:
            await self.release.wait()
        if self.error:
            raise self.error
        return SimpleNamespace(message_id=kwargs.get("message_id", len(self.calls))) if self.receipt else True


def load(home, monkeypatch, *, enabled=True, setting=True, profile="a"):
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        f"plugins:\n  enabled: {'[hermes-telegram-experience]' if enabled else '[]'}\n"
        f"  entries:\n    hermes-telegram-experience:\n      settings:\n        enabled: {str(setting).lower()}\n"
        "        scope:\n"
        "          routes:\n"
        f"            - {{profile: {profile}, platform: telegram, chat_id: '-100', thread_id: '7'}}\n"
        "          task_resources: []\n"
    )
    manager = get_plugin_manager()
    manager.discover_and_load(force=True)
    return manager


@pytest_asyncio.fixture
async def lane(tmp_path, monkeypatch):
    manager = load(tmp_path / "a", monkeypatch)
    bot = BotBoundary()
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic", typing_indicator=False))
    adapter._bot = bot
    store = TodoStore()
    binding = TodoBinding("a", "a", str(manager.home_path), "session", 1, "-100", "7")
    reg = manager._live_todo_registration
    source = reg.open(adapter, binding, lambda: True, lambda: store)
    yield SimpleNamespace(manager=manager, reg=reg, bot=bot, adapter=adapter, store=store, source=source, binding=binding)
    await source.finish()
    manager.unload()
    await asyncio.sleep(0)


def publish(lane, content="first"):
    todo_tool([{"id": "a", "content": content, "status": "pending"}], store=lane.store)
    return lane.source.publish(lane.store.snapshot(), lane.store.incarnation)


async def until(predicate):
    async with asyncio.timeout(6):
        while not predicate():
            await asyncio.sleep(0.01)


def test_rendering_preserves_status_nesting_unicode_and_bounds():
    todos = [
        {"id": "p", "status": "pending"}, {"id": "i", "status": "in_progress"},
        {"id": "c", "status": "completed"}, {"id": "x", "status": "cancelled"},
        {"id": "child", "content": "子 _*<>", "parent": "p"},
        *({"id": str(i), "content": "🙂" * 500} for i in range(100)),
    ]
    text = render_todo_progress({"todos": todos, "revision": 9})
    for marker in ("Conversation steps", "○ (no description)", "◉ (no description)",
                   "✓ (no description)", "– (no description) (cancelled)",
                   "  ○ 子 _*<>", "omitted"):
        assert marker in text
    assert len(text.encode("utf-16-le")) // 2 <= 4096
    assert render_todo_progress({"todos": [], "revision": 10}) == "Conversation steps\nNo steps yet"
    deep = {"todos": [{"id": str(i), "parent": str(i-1)} for i in range(20)]}
    assert "deep nesting flattened" in render_todo_progress(deep)
    raw = render_todo_progress({"todos": [{"id": str(i), "content": "*" * 310} for i in range(60)]},
                               format_text=lambda value: value.replace("*", ""))
    assert len(raw.encode("utf-16-le")) // 2 <= 4096 and "omitted" in raw


@pytest.mark.asyncio
async def test_duplicate_registration_coalescing_and_same_message(lane):
    ctx = PluginContext(lane.manager._plugins["hermes-telegram-experience"].manifest, lane.manager)
    register(ctx)
    assert lane.manager._live_todo_registration is lane.reg
    assert lane.reg.open(lane.adapter, lane.binding, lambda: True, lambda: lane.store) is lane.source
    assert publish(lane, "old") and publish(lane, "latest")
    task = asyncio.create_task(lane.source.run())
    await until(lambda: lane.source.message_id is not None)
    assert len(lane.bot.calls) == 1 and "latest" in lane.bot.calls[0][1]["text"]
    assert publish(lane, "edited")
    await until(lambda: len(lane.bot.calls) == 2)
    assert lane.bot.calls[1][0] == "edit" and lane.bot.calls[1][1]["message_id"] == 1
    assert lane.bot.calls[0][1]["message_thread_id"] == 7
    await lane.source.finish()
    await task
    assert not publish(lane, "late")


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["close", "disable", "client", "generation", "store", "disconnect"])
async def test_admission_after_lock_wait_is_fenced(lane, stop):
    assert publish(lane)
    lock = lane.adapter._chat_send_lock("-100")
    await lock.__aenter__()
    operation = asyncio.create_task(lane.source.deliver("held"))
    await asyncio.sleep(0)  # operation is now awaiting the already-held real adapter lock
    if stop == "close":
        lane.source.close()
    elif stop == "disable":
        lane.reg.close()
    elif stop == "client":
        lane.adapter._bot = BotBoundary()
    elif stop == "generation":
        lane.source.is_current = lambda: False
    elif stop == "store":
        lane.source.store = lambda: TodoStore()
    else:
        await lane.adapter.disconnect()
    await lock.__aexit__(None, None, None)
    result = await operation
    assert result.status == DeliveryStatus.REJECTED
    assert lane.bot.calls == []


@pytest.mark.asyncio
async def test_after_dispatch_disable_settles_before_successor(lane):
    assert publish(lane)
    lane.bot.hold = True
    task = asyncio.create_task(lane.source.run())
    await lane.bot.entered.wait()
    lane.reg.close()
    assert lane.source.inflight
    # Effective stop cannot retract the remotely dispatched create.
    lane.bot.release.set()
    await task
    assert lane.source.last_outcome.status == DeliveryStatus.DELIVERED
    assert lane.source.message_id == "1" and not lane.source.active
    assert (await lane.source.deliver("late")).status == DeliveryStatus.REJECTED
    assert len(lane.bot.calls) == 1


@pytest.mark.asyncio
async def test_cancel_after_dispatch_quarantines_and_blocks_new_adapter(lane):
    assert publish(lane)
    lane.bot.hold = True
    task = asyncio.create_task(lane.source.run())
    await lane.bot.entered.wait()
    successor_adapter = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic", typing_indicator=False))
    successor_adapter._bot = BotBoundary()
    assert lane.reg.open(successor_adapter, replace(lane.binding, run_generation=2), lambda: True, lambda: lane.store) is None
    await lane.source.finish()
    assert task.done() and lane.source.unknown
    assert lane.source.last_outcome.status == DeliveryStatus.UNKNOWN
    assert lane.reg.open(successor_adapter, replace(lane.binding, run_generation=2), lambda: True, lambda: lane.store) is None
    assert successor_adapter._bot.calls == []


@pytest.mark.asyncio
async def test_new_client_has_fresh_binding_and_old_binding_never_forwards(lane):
    assert publish(lane)
    old = lane.source
    new_bot = BotBoundary()
    lane.adapter._bot = new_bot
    new = lane.reg.open(lane.adapter, replace(lane.binding, run_generation=2), lambda: True, lambda: lane.store)
    assert new is not old and new.client is new_bot
    assert new.epoch != old.epoch
    assert not hasattr(new.consumer.source, "adapter")
    assert not hasattr(new.consumer.source, "client")
    assert (await old.deliver("stale")).status == DeliveryStatus.REJECTED
    assert new.publish(lane.store.snapshot(), lane.store.incarnation)
    assert (await new.deliver("new")).status == DeliveryStatus.DELIVERED
    assert lane.bot.calls == [] and len(new_bot.calls) == 1
    assert lane.reg.open(lane.adapter, lane.binding, lambda: True, lambda: lane.store) is None
    await new.finish()


@pytest.mark.asyncio
@pytest.mark.parametrize("error,expected", [(TimedOut(), "unknown"), (BadRequest("message to edit not found"), "rejected")])
async def test_edit_failures_never_create_replacements(lane, error, expected):
    assert publish(lane)
    assert (await lane.source.deliver("first")).status == "delivered"
    lane.bot.error = error
    outcome = await lane.source.deliver("second")
    assert outcome.status == expected
    assert [kind for kind, _ in lane.bot.calls] == ["send", "edit"]
    if expected == "unknown":
        assert (await lane.source.deliver("third")).status == "rejected"


@pytest.mark.asyncio
async def test_unknown_create_and_missing_receipt_never_replay(lane):
    lane.bot.receipt = False
    assert publish(lane)
    task = asyncio.create_task(lane.source.run())
    await task
    assert lane.source.last_outcome.status == "unknown"
    assert not publish(lane, "newer")
    assert len(lane.bot.calls) == 1


@pytest.mark.asyncio
async def test_payload_rejected_without_split_and_cooldown_is_skipped(lane):
    assert publish(lane)
    assert (await lane.source.deliver("🙂" * 4096)).status == "rejected"
    # Actual adapter cooldown path, not fabricated SendResult(success=True, skipped=True).
    lane.adapter._send_flood_cooldown_remaining = lambda _: 4.0
    result = await lane.source.deliver("valid")
    assert result.status == "skipped" and result.retry_after == 4.0
    assert lane.bot.calls == []


@pytest.mark.asyncio
async def test_retry_after_is_bounded_and_new_demand_coalesces(lane):
    lane.bot.error = RetryAfter(0)
    assert publish(lane)
    task = asyncio.create_task(lane.source.run())
    await task
    assert len(lane.bot.calls) == 3
    assert not lane.source.unknown


@pytest.mark.asyncio
async def test_newer_snapshot_while_create_inflight_edits_exact_receipt(lane):
    lane.bot.hold = True
    assert publish(lane, "r1")
    task = asyncio.create_task(lane.source.run())
    await lane.bot.entered.wait()
    assert publish(lane, "r2")
    lane.bot.release.set()
    await until(lambda: len(lane.bot.calls) == 2)
    assert [kind for kind, _ in lane.bot.calls] == ["send", "edit"]
    assert lane.bot.calls[1][1]["message_id"] == 1 and "r2" in lane.bot.calls[1][1]["text"]
    await lane.source.finish()
    await task


@pytest.mark.asyncio
async def test_bad_or_stale_source_events_cannot_clear_projection(lane):
    assert publish(lane)
    for bad in ({"todos": []}, {"todos": ["bad"], "revision": 2}, {"todos": [], "revision": -1}, {"todos": [], "revision": 1}):
        assert not lane.source.publish(bad, lane.store.incarnation)
    assert not lane.source.publish({"todos": [], "revision": 2}, "wrong-store")
    assert lane.source.revision == 1


@pytest.mark.asyncio
async def test_profile_a_b_a_loader_isolation_and_disable_reenable(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import set_secret_scope, reset_secret_scope, set_multiplex_active
    # Managers load under explicit home scopes, never the launch profile's settings.
    a, b = tmp_path / "a", tmp_path / "b"
    ma = load(a, monkeypatch)
    mb = load(b, monkeypatch, setting=False)
    assert not hasattr(mb, "_live_todo_registration")
    set_multiplex_active(True)
    home_token = set_hermes_home_override(a)
    secret_token = set_secret_scope({})
    try:
        assert get_plugin_manager() is ma and ma._live_todo_registration.active
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)
    sentinel = a / "canonical-state.txt"
    sentinel.write_text("preserve me")
    ma = load(a, monkeypatch, enabled=False)
    assert not ma._live_todo_registration.active
    ma = load(a, monkeypatch)
    assert ma._live_todo_registration.active and sentinel.read_text() == "preserve me"
    ma.unload()
    mb.unload()


def test_missing_host_capability_fails_at_real_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(PluginContext, "live_todo_capability", None)
    manager = load(tmp_path / "unsupported", monkeypatch)
    loaded = manager._plugins["hermes-telegram-experience"]
    assert not loaded.enabled and "requires live_todo capability" in loaded.error
    assert not hasattr(manager, "_live_todo_registration")


@pytest.mark.parametrize("enabled,setting", [(False, True), (True, False)])
def test_opt_in_is_required_in_both_loader_and_settings(tmp_path, monkeypatch, enabled, setting):
    manager = load(tmp_path / "disabled", monkeypatch, enabled=enabled, setting=setting)
    assert not hasattr(manager, "_live_todo_registration")


@pytest.mark.asyncio
async def test_config_disable_is_requested_until_effective_loader_unload(lane):
    (lane.manager.home_path / "config.yaml").write_text("plugins:\n  enabled: []\n")
    assert lane.reg.active  # No false claim of hot disable on disk edit alone.
    assert publish(lane)
    lane.manager.unload()
    assert not lane.reg.active and not lane.source.admitted()
    assert (await lane.source.deliver("late")).status == "rejected"


@pytest.mark.asyncio
async def test_registration_rejects_wrong_home_or_missing_transport(lane):
    assert lane.reg.open(lane.adapter, replace(lane.binding, profile_home="another"), lambda: True, lambda: lane.store) is None
    lane.adapter.live_todo_transport = None
    assert lane.reg.open(lane.adapter, lane.binding, lambda: True, lambda: lane.store) is None


@pytest.mark.asyncio
async def test_after_dispatch_disconnect_fences_before_core_teardown(lane):
    assert publish(lane)
    lane.bot.hold = True
    task = asyncio.create_task(lane.source.run())
    await lane.bot.entered.wait()
    disconnect = asyncio.create_task(lane.adapter.disconnect())
    await until(lambda: not lane.source.active)
    assert (await lane.source.deliver("late")).status == "rejected"
    lane.bot.release.set()
    await disconnect
    await task
    assert lane.source.last_outcome.status == "delivered"
    assert len(lane.bot.calls) == 1


@pytest.mark.asyncio
async def test_old_adapter_never_forwards_to_runner_replacement(lane):
    assert publish(lane)
    replacement = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic", typing_indicator=False))
    replacement._bot = BotBoundary()
    lane.adapter._bot = None
    lane.adapter.gateway_runner = SimpleNamespace(adapters={lane.adapter.platform: replacement})
    assert (await lane.source.deliver("old")).status == "rejected"
    assert replacement._bot.calls == [] and lane.bot.calls == []


@pytest.mark.asyncio
async def test_disconnect_then_new_adapter_gets_new_epoch(lane):
    assert publish(lane)
    await lane.adapter.disconnect()
    replacement = TelegramAdapter(PlatformConfig(enabled=True, token="synthetic", typing_indicator=False))
    replacement._bot = BotBoundary()
    successor = lane.reg.open(replacement, replace(lane.binding, run_generation=2), lambda: True, lambda: lane.store)
    assert successor.epoch != lane.source.epoch
    assert successor.publish(lane.store.snapshot(), lane.store.incarnation)
    assert (await successor.deliver("fresh")).status == "delivered"
    assert lane.bot.calls == []
    await successor.finish()


@pytest.mark.asyncio
async def test_profile_a_b_a_deliveries_cannot_edit_each_others_message(lane, tmp_path, monkeypatch):
    assert publish(lane, "profile A")
    assert (await lane.source.deliver("profile A first")).message_id == "1"
    manager_b = load(tmp_path / "b", monkeypatch, profile="b")
    store_b = TodoStore()
    todo_tool([{"id": "b", "content": "profile B", "status": "pending"}], store=store_b)
    binding_b = replace(lane.binding, profile="b", profile_home=str(manager_b.home_path), session_id="session-b")
    source_b = manager_b._live_todo_registration.open(lane.adapter, binding_b, lambda: True, lambda: store_b)
    assert source_b.publish(store_b.snapshot(), store_b.incarnation)
    assert (await source_b.deliver("profile B first")).message_id == "2"
    assert not lane.source.publish(store_b.snapshot(), store_b.incarnation)
    assert (await lane.source.deliver("profile A again")).message_id == "1"
    assert [kind for kind, _ in lane.bot.calls] == ["send", "send", "edit"]
    assert lane.bot.calls[-1][1]["message_id"] == 1
    await source_b.finish()
    manager_b.unload()


@pytest.mark.asyncio
async def test_disable_during_shared_rate_slot_wait_never_dispatches(lane):
    assert publish(lane)
    lane.adapter._telegram_chat_outbound_slot_secs = 0.2
    lane.adapter._hold_chat_outbound_slot("-100")
    operation = asyncio.create_task(lane.source.deliver("rate-waiting"))
    await until(lambda: bool(lane.adapter.__dict__.get("_telegram_chat_send_lock_owners")))
    assert not lane.source.inflight
    lane.reg.close()
    assert (await operation).status == "rejected"
    assert lane.bot.calls == []
