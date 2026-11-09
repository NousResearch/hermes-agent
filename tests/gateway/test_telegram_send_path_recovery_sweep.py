"""In-place Telegram polling recovery replays finals refused with ``send_path_degraded``."""

import asyncio

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import telegram_send_path_recovery as recovery
from plugins.platforms.telegram.adapter import TelegramAdapter


class _Runner:
    """The runner seam the adapter calls: the runtime-ledger sweep for one bot identity."""

    def __init__(self, gate=None):
        self.sweeps = []
        self.gate = gate

    async def _redeliver_failed_obligations_for_platform(self, platform, *, profile=None):
        self.sweeps.append((platform.value, profile))
        if self.gate is not None:
            await self.gate.wait()
        return 0


def _degraded_adapter(runner) -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter.gateway_runner = runner
    adapter._running = True
    adapter._owner_profile = "work"
    generation, _progress = adapter._begin_polling_generation()
    assert adapter.send_path_degraded
    return adapter, generation


@pytest.fixture(autouse=True)
def _no_settle(monkeypatch):
    monkeypatch.setattr(recovery, "_SWEEP_SETTLE_SECONDS", 0)


async def _drain(adapter):
    task = getattr(adapter, "_send_path_recovery_task", None)
    if task is not None:
        await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_in_place_recovery_sweeps_the_adapters_own_identity_once():
    runner = _Runner()
    adapter, generation = _degraded_adapter(runner)

    assert adapter._record_polling_progress(generation)
    await _drain(adapter)
    assert not adapter.send_path_degraded
    assert runner.sweeps == [("telegram", "work")]

    # Progress on an already-healthy send path is not a recovery edge: no further sweeps.
    assert adapter._record_polling_progress(generation)
    await asyncio.sleep(0)
    await _drain(adapter)
    assert runner.sweeps == [("telegram", "work")]


@pytest.mark.asyncio
async def test_an_edge_during_a_running_sweep_coalesces_into_one_rerun():
    gate = asyncio.Event()
    runner = _Runner(gate=gate)
    adapter, generation = _degraded_adapter(runner)

    assert adapter._record_polling_progress(generation)
    for _ in range(20):
        if runner.sweeps:
            break
        await asyncio.sleep(0)
    assert runner.sweeps == [("telegram", "work")]
    # Two more degraded -> healthy flaps while the first sweep is still sending.
    for _ in range(2):
        generation, _progress = adapter._begin_polling_generation()
        assert adapter._record_polling_progress(generation)
    gate.set()
    await _drain(adapter)
    assert runner.sweeps == [("telegram", "work"), ("telegram", "work")]


@pytest.mark.asyncio
async def test_a_refusal_whose_failed_row_lands_after_recovery_is_still_swept(monkeypatch):
    """The edge's sweep runs while ``_send_with_retry`` still holds the delivery-failure notice open;
    the ``failed`` row written after that sweep must re-arm it (healthy polling has no second edge)."""
    import gateway.delivery_ledger as ledger
    from gateway.config import Platform
    from gateway.platforms.base import SendResult
    from gateway.session import SessionSource
    from gateway.platforms.event import MessageEvent

    runner = _Runner()
    adapter, generation = _degraded_adapter(runner)
    marked = []
    monkeypatch.setattr(ledger, "mark_failed", lambda oid, error: marked.append((oid, error)))
    notice_open, release_notice = asyncio.Event(), asyncio.Event()

    async def _send_with_retry(**_kw):
        notice_open.set()
        await release_notice.wait()  # the notice send outlives the settle delay
        return SendResult(success=False, error="send_path_degraded", retryable=True)
    adapter._send_with_retry = _send_with_retry
    event = MessageEvent(text="q", source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
                         message_id="m1")
    adapter._record_delivery_obligation = lambda *a, **k: asyncio.sleep(0, result="oid-1")

    final = asyncio.ensure_future(adapter.send_final_ledgered(event, "sk", "answer", {}, reply_to=None))
    await notice_open.wait()
    assert adapter._record_polling_progress(generation)
    await _drain(adapter)
    assert runner.sweeps == [("telegram", "work")]  # ran before the row existed

    release_notice.set()
    await final
    assert marked == [("oid-1", "send_path_degraded")]
    await _drain(adapter)
    assert runner.sweeps == [("telegram", "work"), ("telegram", "work")]


@pytest.mark.asyncio
async def test_stopping_a_profile_mid_recovery_leaves_no_reply_stranded(tmp_path, monkeypatch):
    """Profile stop cancels the recovery task while its first resend is blocked; the batch the sweep
    already claimed must be replayed by the next sweep after the profile starts again, not wait for
    a gateway restart."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    import gateway.delivery_ledger as ledger
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "state.db")
    monkeypatch.setattr(ledger, "ledger_enabled", lambda config=None: True)
    for oid in ("ob-1", "ob-2", "ob-3"):
        ledger.record_obligation(obligation_id=oid, session_key=f"agent:work:telegram:dm:{oid}",
                                 platform="telegram", chat_id="1", thread_id=None,
                                 content=f"answer {oid}", adapter_profile="work")
        ledger.mark_failed(oid, "send_path_degraded")

    first_send_started, sent = asyncio.Event(), []

    async def _send(*, chat_id, content, metadata=None):
        if not sent:
            sent.append(None)
            first_send_started.set()
            await asyncio.Event().wait()  # the first resend never completes before the stop
        sent.append(content)
        return SimpleNamespace(success=True, error=None)

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._profile_adapters = {"work": {Platform.TELEGRAM: SimpleNamespace(send=_send)}}
    runner.session_store = None
    runner._async_session_store = MagicMock(_store=None, clear_resume_pending=AsyncMock())
    runner._arm_flood_timers_for_waiting_rows = AsyncMock()

    adapter, generation = _degraded_adapter(runner)
    assert adapter._record_polling_progress(generation)
    await asyncio.wait_for(first_send_started.wait(), timeout=5)
    await adapter.cancel_background_tasks()  # what stopping the profile does

    # The profile starts again and its next sweep runs.
    assert await runner._redeliver_failed_obligations_for_platform(Platform.TELEGRAM, profile="work") == 3
    states = {oid: ledger._connect().execute(
        "SELECT state FROM delivery_obligations WHERE obligation_id=?", (oid,)).fetchone()[0]
        for oid in ("ob-1", "ob-2", "ob-3")}
    assert states == {"ob-1": "delivered", "ob-2": "delivered", "ob-3": "delivered"}
