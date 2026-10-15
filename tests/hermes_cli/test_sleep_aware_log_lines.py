"""Two log lines that misled on a sleeping laptop.

On an RTX Spark, every lid close produced "event loop stalled 229s (GIL pressure suspected)": the
web server's heartbeat runs on the monotonic clock, which counts sleep on Windows. And the
statusbar's 5-second hardware poll logged the unified-memory pool reading on every poll, 12 lines a
minute of the same numbers.
"""

import asyncio
import logging

import pytest

from hermes_cli.local_runtime import hardware as hw
from hermes_cli.web_server_lifecycle import arm_loop_heartbeat, loop_stall_message

G = 1 << 30


def test_a_punctual_tick_logs_nothing():
    assert loop_stall_message(0.01, 0.0) is None
    assert loop_stall_message(4.9, 0.0) is None


def test_a_sleeping_computer_is_not_called_gil_pressure():
    level, line = loop_stall_message(229.0, 0.2)
    assert level == logging.INFO
    assert "probably asleep" in line and "GIL" not in line


def test_a_busy_process_stall_still_warns():
    level, line = loop_stall_message(8.0, 7.6)
    assert level == logging.WARNING
    assert line == "event loop stalled 8.0s (GIL pressure suspected)"


def test_heartbeat_reports_a_late_tick_through_the_loop(monkeypatch):
    """A fake loop clock jumps 64 s between ticks while the process uses no CPU."""
    from hermes_cli import web_server_lifecycle as lc

    clock = {"t": 100.0}
    scheduled = []
    loop = type("FakeLoop", (), {
        "time": lambda self: clock["t"],
        "call_later": lambda self, delay, fn, *args: scheduled.append((fn, args)),
    })()
    monkeypatch.setattr(lc.time, "process_time", lambda: 5.0)
    logged = []
    log = type("Log", (), {"log": lambda self, level, line: logged.append((level, line))})()

    arm_loop_heartbeat(loop, log)
    fn, args = scheduled.pop()
    clock["t"] = args[0] + 64.0
    fn(*args)
    assert logged == [(logging.INFO, "event loop paused 64s with the process idle (the computer was probably asleep)")]
    assert scheduled  # the next tick is armed


def test_heartbeat_arms_on_a_real_loop():
    async def arm():
        arm_loop_heartbeat(asyncio.get_running_loop(), logging.getLogger("test"))

    asyncio.run(arm())


@pytest.fixture
def spark(monkeypatch):
    monkeypatch.setattr(hw, "_logged_pool_reading", None)
    monkeypatch.setattr(hw, "_ram_bytes", lambda: (int(30.2 * G), int(15 * G)))
    monkeypatch.setattr(hw, "_nvidia_vram", lambda: (int(31.9 * G), 31 * G, "NVIDIA RTX Spark N1X", None))
    monkeypatch.setattr(hw, "_unified_pool_bytes", lambda smi_total, ram_total: int(46.3 * G))


def test_pool_reading_is_logged_once_across_polls(spark, caplog):
    with caplog.at_level(logging.INFO, logger=hw.logger.name):
        for _ in range(12):
            hw.probe_budget()
    lines = [r.getMessage() for r in caplog.records if "budgeting from the pool" in r.getMessage()]
    assert lines == ["unified-memory NVIDIA device: allocator pool 46.3 GiB (nvidia-smi carve-out: 31.9 GiB); "
                     "budgeting from the pool"]


def test_a_changed_pool_reading_is_logged_again(spark, monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger=hw.logger.name):
        hw.probe_budget()
        monkeypatch.setattr(hw, "_unified_pool_bytes", lambda smi_total, ram_total: int(54.3 * G))
        hw.probe_budget()
    assert sum("budgeting from the pool" in r.getMessage() for r in caplog.records) == 2
