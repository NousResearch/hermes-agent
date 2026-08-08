"""Fault-injection contracts for gateway restart and delivery bookkeeping."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from gateway import restart_loop_guard, status
from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, Platform, SendResult


class _RetryAdapter(BasePlatformAdapter):
    def __init__(self) -> None:
        super().__init__(PlatformConfig(), Platform.TELEGRAM)
        self.results: list[SendResult] = []
        self.calls = 0

    async def send(self, chat_id, content, reply_to=None, metadata=None, **kwargs):
        self.calls += 1
        return self.results.pop(0)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id):
        return {"chat_id": chat_id}


def _record_start_in_process(home: str, gate, max_starts: int) -> None:
    os.environ["HERMES_HOME"] = home
    gate.wait()
    status.record_start_and_check_storm(max_starts=max_starts, window_s=120)


def test_respawn_start_ledger_keeps_every_concurrent_writer(tmp_path, monkeypatch):
    """Concurrent supervisors must not erase sibling start observations."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    workers = 32
    barrier = threading.Barrier(workers)

    def record() -> None:
        barrier.wait()
        status.record_start_and_check_storm(max_starts=workers + 1, window_s=120)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda _index: record(), range(workers)))

    lines = status._get_starts_log_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == workers


def test_respawn_start_ledger_keeps_every_process_writer(tmp_path):
    """The adjacent lock must protect independent supervisor processes too."""
    workers = 8
    context = multiprocessing.get_context("spawn")
    gate = context.Event()
    processes = [
        context.Process(
            target=_record_start_in_process,
            args=(str(tmp_path), gate, workers + 1),
        )
        for _ in range(workers)
    ]
    try:
        for process in processes:
            process.start()
        gate.set()
        for process in processes:
            process.join(timeout=20)
        assert [process.exitcode for process in processes] == [0] * workers
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    path = tmp_path / "gateway-starts.log"
    assert len(path.read_text(encoding="utf-8").splitlines()) == workers


@pytest.mark.parametrize("seed", [float("inf"), time.time() + 86_400])
def test_respawn_start_ledger_ignores_impossible_timestamps(
    tmp_path, monkeypatch, seed
):
    """Corrupt/future history must not manufacture a five-minute backoff."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = status._get_starts_log_path()
    path.write_text("\n".join([repr(seed)] * 6) + "\n", encoding="utf-8")

    result = status.record_start_and_check_storm(max_starts=5, window_s=120)

    assert result is None


def test_restart_loop_torn_write_preserves_last_good_history(tmp_path, monkeypatch):
    """An interrupted persistence attempt must not destroy the prior breaker."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = restart_loop_guard._state_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"boots": [1000.0, 1001.0]}), encoding="utf-8")
    original_write_text = Path.write_text

    def torn_write(self, data, *args, **kwargs):
        if self == path:
            original_write_text(self, '{"boots":[1000.0', encoding="utf-8")
            raise OSError("synthetic power loss")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", torn_write)
    restart_loop_guard.record_restart_interrupted_boot(60, now=1002.0)

    assert restart_loop_guard.is_restart_loop_tripped(3, 60, now=1003.0)


def test_restart_loop_malformed_root_fails_open_and_self_heals(tmp_path, monkeypatch):
    """A non-object state document must not escape the breaker's fail-open API."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = restart_loop_guard._state_path()
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")

    assert restart_loop_guard.check_and_record(3, 60, now=1000.0) is False
    assert json.loads(path.read_text(encoding="utf-8")) == {"boots": [1000.0]}


@pytest.mark.parametrize("seed", [float("inf"), 1_000_000_000_000.0])
def test_restart_loop_ignores_nonfinite_and_future_boots(tmp_path, monkeypatch, seed):
    """Impossible persisted boot times must never permanently trip recovery."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = restart_loop_guard._state_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"boots": [seed]}), encoding="utf-8")

    assert restart_loop_guard.is_restart_loop_tripped(1, 60, now=1000.0) is False


@pytest.mark.asyncio
async def test_retryable_flag_cannot_override_ambiguous_timeout():
    """Read/write timeouts stay non-retriable even if an adapter misflags one."""
    adapter = _RetryAdapter()
    adapter.results = [
        SendResult(success=False, error="ReadTimeout: request may have landed", retryable=True),
        SendResult(success=True, message_id="duplicate"),
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
        result = await adapter._send_with_retry("chat", "hello", base_delay=0)

    assert result.success is False
    assert adapter.calls == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_transition_to_timeout_never_sends_plaintext_duplicate():
    """A timeout after a safe connect retry must stop without fallback resend."""
    adapter = _RetryAdapter()
    adapter.results = [
        SendResult(success=False, error="ConnectError: refused", retryable=True),
        SendResult(success=False, error="WriteTimeout: request may have landed"),
        SendResult(success=True, message_id="duplicate-fallback"),
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await adapter._send_with_retry("chat", "hello", base_delay=0)

    assert result.success is False
    assert adapter.calls == 2


@pytest.mark.parametrize("retry_after", [-10.0, float("nan"), float("inf"), "bad"])
@pytest.mark.asyncio
async def test_invalid_retry_after_falls_back_to_bounded_local_backoff(retry_after):
    """Malformed server delay values cannot hot-loop, crash, or sleep forever."""
    adapter = _RetryAdapter()
    adapter.results = [
        SendResult(
            success=False,
            error="rate limited",
            retryable=True,
            retry_after=retry_after,
        ),
        SendResult(success=True, message_id="ok"),
    ]

    with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
        result = await adapter._send_with_retry(
            "chat", "hello", max_retries=1, base_delay=1.0
        )

    assert result.success is True
    delay = sleep.await_args.args[0]
    assert math.isfinite(delay)
    assert 1.0 <= delay < 3.0
