"""CallStore persistence/restore and runtime singleton lifecycle."""

import asyncio
import json

import pytest

from plugins.platforms.voice_call import runtime as runtime_mod
from plugins.platforms.voice_call.events import CallRecord, CallState, new_call_id
from plugins.platforms.voice_call.manager import CallManager
from plugins.platforms.voice_call.providers.mock import MockProvider
from plugins.platforms.voice_call.store import CallStore



def _record(state=CallState.ACTIVE, **kwargs) -> CallRecord:
    record = CallRecord(
        call_id=kwargs.pop("call_id", new_call_id()),
        provider="mock",
        direction=kwargs.pop("direction", "outbound"),
        from_number="+15555550000",
        to_number="+15555550001",
        **kwargs,
    )
    record.state = state
    return record


# -- store ---------------------------------------------------------------


def test_store_appends_and_replays_latest(store):
    record = _record(state=CallState.INITIATED)
    store.append(record)
    record.state = CallState.ACTIVE
    store.append(record)
    other = _record(state=CallState.COMPLETED)
    store.append(other)

    latest = store.load_latest()
    assert latest[record.call_id].state == CallState.ACTIVE
    assert latest[other.call_id].state == CallState.COMPLETED

    active = store.load_active()
    assert record.call_id in active
    assert other.call_id not in active


def test_store_skips_corrupt_lines(store):
    record = _record()
    store.append(record)
    with open(store.calls_path, "a", encoding="utf-8") as f:
        f.write("{not json}\n")
        f.write('"a string, not an object"\n')
    latest = store.load_latest()
    assert list(latest) == [record.call_id]


def test_store_history_newest_first(store):
    first = _record(state=CallState.COMPLETED)
    first.started_at = 100.0
    second = _record(state=CallState.COMPLETED)
    second.started_at = 200.0
    store.append(first)
    store.append(second)
    history = store.history(limit=1)
    assert [r.call_id for r in history] == [second.call_id]


def test_store_compaction(tmp_path, monkeypatch):
    import plugins.platforms.voice_call.store as store_mod

    monkeypatch.setattr(store_mod, "_COMPACT_THRESHOLD", 5)
    store = CallStore(base_dir=tmp_path)
    record = _record()
    for _ in range(10):
        store.append(record)
    store.load_latest()  # triggers compaction
    with open(store.calls_path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["call_id"] == record.call_id


# -- manager restore -------------------------------------------------------


@pytest.mark.asyncio
async def test_restore_keeps_live_call_and_drops_terminal(store, provider, make_config):
    cfg = make_config()
    live = _record(state=CallState.ACTIVE)
    done = _record(state=CallState.ACTIVE, call_id=new_call_id())
    live.provider_call_id = "mock-live"
    done.provider_call_id = "mock-done"
    store.append(live)
    store.append(done)
    provider._terminal.add("mock-done")  # carrier says this one ended

    manager = CallManager(cfg, provider, store)
    await manager.initialize()
    assert manager.get_call(live.call_id) is not None
    assert manager.get_call(done.call_id) is None
    # The terminal one was finalized and persisted as completed.
    assert store.load_latest()[done.call_id].state == CallState.COMPLETED
    await manager.shutdown()


@pytest.mark.asyncio
async def test_restore_hangs_up_expired_calls(store, provider, make_config):
    cfg = make_config()
    cfg.timeouts.max_call_s = 10
    stale = _record(state=CallState.ACTIVE)
    stale.provider_call_id = "mock-stale"
    stale.started_at = 1.0  # eons ago
    store.append(stale)

    manager = CallManager(cfg, provider, store)
    await manager.initialize()
    assert manager.get_call(stale.call_id) is None
    assert provider.hangups == ["mock-stale"]
    assert store.load_latest()[stale.call_id].state == CallState.TIMEOUT


@pytest.mark.asyncio
async def test_restore_keeps_call_on_unknown_status(store, make_config):
    cfg = make_config()

    class FlakyProvider(MockProvider):
        async def get_call_status(self, call):
            raise ConnectionError("carrier 502")

    provider = FlakyProvider(cfg)
    live = _record(state=CallState.ACTIVE)
    live.provider_call_id = "mock-flaky"
    store.append(live)

    manager = CallManager(cfg, provider, store)
    await manager.initialize()
    # Unknown status → keep the call, rely on the max-duration timer.
    assert manager.get_call(live.call_id) is not None
    assert (live.call_id, "max") in manager._timers
    await manager.shutdown()


# -- runtime singleton -------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_runtime_is_singleton_and_concurrency_safe(tmp_path, make_config):
    cfg = make_config()
    results = await asyncio.gather(
        *[runtime_mod.ensure_runtime(cfg, store_dir=tmp_path) for _ in range(5)]
    )
    assert all(r is results[0] for r in results)
    assert runtime_mod.get_runtime() is results[0]
    await runtime_mod.stop_runtime()
    assert runtime_mod.get_runtime() is None


@pytest.mark.asyncio
async def test_stop_runtime_idempotent(tmp_path, make_config):
    cfg = make_config()
    await runtime_mod.ensure_runtime(cfg, store_dir=tmp_path)
    await runtime_mod.stop_runtime()
    await runtime_mod.stop_runtime()  # second stop is a no-op
    assert runtime_mod.get_runtime() is None


@pytest.mark.asyncio
async def test_failed_start_leaves_no_singleton(tmp_path, make_config):
    cfg = make_config()
    cfg.provider = "definitely-not-a-provider"
    with pytest.raises(ValueError):
        await runtime_mod.ensure_runtime(cfg, store_dir=tmp_path)
    assert runtime_mod.get_runtime() is None
    # And a subsequent good start works.
    cfg.provider = "mock"
    runtime = await runtime_mod.ensure_runtime(cfg, store_dir=tmp_path)
    assert runtime is not None
    await runtime_mod.stop_runtime()


@pytest.mark.asyncio
async def test_runtime_speak_for_chat_mock_e2e(tmp_path, make_config):
    cfg = make_config()
    runtime = await runtime_mod.ensure_runtime(cfg, store_dir=tmp_path)
    record = await runtime.manager.initiate_call("+15555550001", message="hi")
    deadline = asyncio.get_running_loop().time() + 1.0
    while record.state != record.state.LISTENING and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)

    ok, call_id = await runtime.speak_for_chat("+15555550001", "an update for you")
    assert ok and call_id == record.call_id
    assert runtime.provider.spoken[-1] == (record.provider_call_id, "an update for you")

    ok, error = await runtime.speak_for_chat("+19999999999", "nobody home")
    assert not ok and "no active voice call" in error
    await runtime_mod.stop_runtime()
