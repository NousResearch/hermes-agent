"""Gateway event-loop freeze backstops for issue #69089."""

from __future__ import annotations

import asyncio
import json
import pathlib
import threading
import time
from unittest.mock import MagicMock, patch

from gateway.shutdown_watchdog import (
    loop_heartbeat_forever,
    start_loop_liveness_watchdog,
)

def test_loop_liveness_watchdog_stop_during_dump_disarms_hard_exit():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    handle_ready = threading.Event()
    handle_ref = {}
    exit_codes = []

    def stop_during_dump(*_args, **_kwargs) -> None:
        assert handle_ready.wait(timeout=2.0)
        handle_ref["handle"].stop()

    with (
        patch("gateway.shutdown_watchdog.logger.critical") as critical,
        patch(
            "gateway.shutdown_watchdog.faulthandler.dump_traceback",
            side_effect=stop_during_dump,
        ) as dump,
        patch("gateway.shutdown_watchdog.os._exit", side_effect=exit_codes.append),
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1
        )
        assert handle is not None
        handle_ref["handle"] = handle
        handle_ready.set()
        handle.join(timeout=2.0)

    assert not handle.is_alive()
    critical.assert_called_once()
    dump.assert_called()  # dump-file copy first, then stderr (#120356)
    assert exit_codes == []


def test_loop_liveness_death_leaves_its_stacks_in_the_dump_file(tmp_path, monkeypatch):
    # A console-less launcher discards stderr; the file is the only copy that survives (#120356).
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    exit_codes, ref, handle_ready = [], {}, threading.Event()

    def exit_and_stop(code: int) -> None:  # a patched exit returns, so stop the loop like a real one
        assert handle_ready.wait(timeout=2.0)
        exit_codes.append(code)
        ref["handle"].stop()

    with patch("gateway.shutdown_watchdog._hard_exit", side_effect=exit_and_stop), \
            patch("gateway.shutdown_watchdog._mark_exited_quietly"):
        ref["handle"] = handle = start_loop_liveness_watchdog(
            MagicMock(spec=asyncio.AbstractEventLoop), probe_interval=0.01, probe_timeout=0.01,
            max_strikes=1)
        handle_ready.set()
        handle.join(timeout=5.0)
    from gateway.shutdown_watchdog import get_shutdown_watchdog_dump_path
    dump = get_shutdown_watchdog_dump_path(tmp_path).read_text(encoding="utf-8-sig")
    assert '"event": "loop_liveness_watchdog_fired"' in dump
    assert "most recent call first" in dump
    assert exit_codes == [75]


def test_loop_liveness_watchdog_logs_a_nonterminal_miss_with_its_strike_and_wait():
    """Before #137838 only the terminal strike was logged, so how long the loop had been starved
    could not be measured from logs. Each miss now carries its strike count and measured wait."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)  # probes are scheduled but never answered

    with (
        patch("gateway.shutdown_watchdog.logger.warning") as warning,
        patch("gateway.shutdown_watchdog.logger.critical"),
        patch("gateway.shutdown_watchdog._write_watchdog_dump"),
        patch("gateway.shutdown_watchdog._mark_exited_quietly"),
        patch("gateway.shutdown_watchdog._hard_exit"),
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.2, max_strikes=3)
        assert handle is not None
        deadline = time.monotonic() + 2.0
        while warning.call_count < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert warning.call_count == 1, "the non-terminal miss was not logged exactly once"
        handle.stop()
        handle.join(timeout=2.0)

    assert not handle.is_alive()
    args = warning.call_args.args
    assert args[1] == 1 and args[2] == 3, "the miss log must name the strike out of max_strikes"
    assert args[3] >= 0.0, "the miss log must carry the measured probe wait"


def test_an_answered_probe_keeps_the_watchdog_quiet():
    """Ordinary path: a loop that answers its probe is never logged, dumped or exited."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    answered, two_probes = [], threading.Event()

    def answer_probe(callback) -> None:
        callback()  # the gateway loop answering the probe
        answered.append(1)
        if len(answered) >= 2:
            two_probes.set()

    loop.call_soon_threadsafe.side_effect = answer_probe

    with (
        patch("gateway.shutdown_watchdog.logger.warning") as warning,
        patch("gateway.shutdown_watchdog.logger.critical") as critical,
        patch("gateway.shutdown_watchdog._write_watchdog_dump") as dump,
        patch("gateway.shutdown_watchdog._hard_exit") as hard_exit,
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.05, max_strikes=2)
        assert handle is not None
        assert two_probes.wait(timeout=2.0), "the watchdog never ran two probes"
        handle.stop()
        handle.join(timeout=2.0)

    assert not handle.is_alive()
    warning.assert_not_called()
    critical.assert_not_called()
    dump.assert_not_called()
    hard_exit.assert_not_called()


def test_loop_liveness_death_dumps_the_loop_task_snapshot(tmp_path, monkeypatch):
    """Thread stacks cannot see coroutine state, so the terminal dump also carries the loop's task
    names, cancellation counts, frame locations and ready-queue depth (#137838)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ref, handle_ready, exit_codes = {}, threading.Event(), []
    task_snapshot = {
        "ready_queue_len": 3,
        "task_count": 1,
        "tasks": [{"name": "wedged-turn", "cancelling": 1, "done": False,
                   "frames": ["run_turn.py:42:_run_turn"]}],
    }

    def exit_and_stop(code: int) -> None:
        assert handle_ready.wait(timeout=2.0)
        exit_codes.append(code)
        ref["handle"].stop()

    with patch("gateway.shutdown_watchdog._hard_exit", side_effect=exit_and_stop), \
            patch("gateway.shutdown_watchdog._mark_exited_quietly"), \
            patch("gateway.shutdown_watchdog._describe_loop_tasks", return_value=task_snapshot):
        ref["handle"] = handle = start_loop_liveness_watchdog(
            MagicMock(spec=asyncio.AbstractEventLoop), probe_interval=0.01, probe_timeout=0.01,
            max_strikes=1)
        handle_ready.set()
        handle.join(timeout=5.0)

    from gateway.shutdown_watchdog import get_shutdown_watchdog_dump_path
    header = json.loads(
        get_shutdown_watchdog_dump_path(tmp_path).read_text(encoding="utf-8-sig").splitlines()[0])
    assert header["snapshot"]["loop_tasks"] == task_snapshot
    assert header["snapshot"]["missed_probe_wait_s"] >= 0.0
    assert exit_codes == [75]


def test_a_failed_task_snapshot_cannot_prevent_the_restart_exit(tmp_path, monkeypatch):
    """The snapshot reads a wedged loop from another thread: if it blows up, the restart contract
    still wins (#137838)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ref, handle_ready, exit_codes = {}, threading.Event(), []

    def exit_and_stop(code: int) -> None:
        assert handle_ready.wait(timeout=2.0)
        exit_codes.append(code)
        ref["handle"].stop()

    with patch("gateway.shutdown_watchdog._hard_exit", side_effect=exit_and_stop), \
            patch("gateway.shutdown_watchdog._mark_exited_quietly"), \
            patch("gateway.shutdown_watchdog._describe_loop_tasks",
                  side_effect=RuntimeError("snapshot blew up")):
        ref["handle"] = handle = start_loop_liveness_watchdog(
            MagicMock(spec=asyncio.AbstractEventLoop), probe_interval=0.01, probe_timeout=0.01,
            max_strikes=1)
        handle_ready.set()
        handle.join(timeout=5.0)

    assert exit_codes == [75]


def test_describe_loop_tasks_is_bounded_and_carries_no_locals():
    """The snapshot answers "which task refuses cancellation" with names and locations only; it is
    bounded, and it must never carry coroutine arguments or locals (secrets)."""
    from gateway.shutdown_watchdog import _describe_loop_tasks

    marker = "S3CRET-argument-should-never-reach-the-dump"

    async def scenario() -> dict:
        loop = asyncio.get_running_loop()
        release = asyncio.Event()

        async def worker(secret: str) -> None:
            assert secret  # a coroutine argument that must never reach the snapshot
            await release.wait()

        tasks = [asyncio.create_task(worker(marker), name=f"wedged-worker-{index}")
                 for index in range(3)]
        await asyncio.sleep(0)  # let the workers reach their await
        # The task set is unordered and includes the scenario task, so capture everything once and
        # assert the bound separately.
        snapshot = _describe_loop_tasks(loop, max_tasks=100, max_frames=3)
        bounded = _describe_loop_tasks(loop, max_tasks=2, max_frames=3)
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        return snapshot, bounded

    snapshot, bounded = asyncio.run(scenario())
    assert isinstance(snapshot["ready_queue_len"], int)
    assert snapshot["task_count"] >= 3
    assert len(bounded["tasks"]) == 2  # bounded by max_tasks
    workers = {described["name"]: described for described in snapshot["tasks"]
               if described["name"].startswith("wedged-worker-")}
    assert set(workers) == {f"wedged-worker-{index}" for index in range(3)}
    for described in workers.values():
        assert described["frames"], "frame locations are the point of the snapshot"
        assert len(described["frames"]) <= 3  # bounded by max_frames
    assert marker not in json.dumps(snapshot)


def test_loop_liveness_watchdog_stop_during_final_miss_disarms_hard_exit():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    probe_scheduled = threading.Event()
    release_probe = threading.Event()
    probe_event_ref = {}
    handle_ref = {}
    exit_codes = []

    class FinalStrikeLimit:
        def __gt__(self, _strikes: int) -> bool:
            # If strike evaluation is reached, keep recheck #2 from masking a
            # missing post-probe recheck #1 in this boundary test.
            handle_ref["handle"]._stop_event.clear()
            return False

    def hold_scheduled_probe(callback) -> None:
        probe_event_ref["event"] = callback.__self__
        probe_scheduled.set()
        assert release_probe.wait(timeout=2.0)

    loop.call_soon_threadsafe.side_effect = hold_scheduled_probe
    with (
        patch("gateway.shutdown_watchdog.logger.critical") as critical,
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback") as dump,
        patch("gateway.shutdown_watchdog.os._exit", side_effect=exit_codes.append),
    ):
        handle = start_loop_liveness_watchdog(
            loop,
            probe_interval=0.01,
            probe_timeout=0.01,
            max_strikes=FinalStrikeLimit(),
        )
        assert handle is not None
        handle_ref["handle"] = handle
        assert probe_scheduled.wait(timeout=2.0), "watchdog did not schedule a probe"

        def stop_during_miss() -> bool:
            handle.stop()
            return False

        probe_event_ref["event"].is_set = stop_during_miss
        release_probe.set()
        handle.join(timeout=1.0)

    assert not handle.is_alive()
    assert exit_codes == []
    critical.assert_not_called()
    dump.assert_not_called()

def test_loop_liveness_watchdog_stop_after_first_recheck_skips_final_actions():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    probe_scheduled = threading.Event()
    release_probe = threading.Event()

    def hold_scheduled_probe(callback) -> None:
        probe_scheduled.set()
        assert release_probe.wait(timeout=2.0)

    loop.call_soon_threadsafe.side_effect = hold_scheduled_probe
    with (
        patch("gateway.shutdown_watchdog.logger.critical") as critical,
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback") as dump,
        patch("gateway.shutdown_watchdog.os._exit") as hard_exit,
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1
        )
        assert handle is not None
        assert probe_scheduled.wait(timeout=2.0), "watchdog did not schedule a probe"

        original_is_set = handle._stop_event.is_set
        is_set_calls = 0

        def stop_on_final_recheck() -> bool:
            nonlocal is_set_calls
            is_set_calls += 1
            # With the forced immediate timeout: _wait_for_probe is call 1,
            # recheck #1 is call 2, and recheck #2 is call 3.
            if is_set_calls == 3:
                handle.stop()
            return original_is_set()

        handle._stop_event.is_set = stop_on_final_recheck
        with patch(
            "gateway.shutdown_watchdog.time.monotonic", side_effect=[0.0, 1.0]
        ):
            release_probe.set()
            handle.join(timeout=1.0)

    assert is_set_calls == 3
    assert not handle.is_alive()
    critical.assert_not_called()
    dump.assert_not_called()
    hard_exit.assert_not_called()

def test_gateway_config_loop_watchdog_round_trip():
    """loop_watchdog is a config.yaml knob: default on, nested-gateway form honored."""
    from gateway.config import GatewayConfig

    assert GatewayConfig.from_dict({}).loop_watchdog is True
    assert GatewayConfig.from_dict({"loop_watchdog": False}).loop_watchdog is False
    assert (
        GatewayConfig.from_dict(
            {"gateway": {"loop_watchdog": "off"}}
        ).loop_watchdog
        is False
    )
    config = GatewayConfig.from_dict({"loop_watchdog": False})
    assert config.to_dict()["loop_watchdog"] is False

def test_gateway_config_loop_watchdog_tuning_round_trip():
    """Watchdog tolerance knobs parse, serialize, and clamp malformed values."""
    from gateway.config import GatewayConfig

    # Defaults
    default = GatewayConfig.from_dict({})
    assert default.loop_watchdog is True
    assert default.loop_watchdog_probe_interval_s == 30.0
    assert default.loop_watchdog_probe_timeout_s == 10.0
    assert default.loop_watchdog_max_strikes == 3

    # Explicit values round-trip
    cfg = GatewayConfig.from_dict(
        {
            "loop_watchdog_probe_interval_s": 45,
            "loop_watchdog_probe_timeout_s": 15,
            "loop_watchdog_max_strikes": 12,
        }
    )
    assert cfg.loop_watchdog_probe_interval_s == 45.0
    assert cfg.loop_watchdog_probe_timeout_s == 15.0
    assert cfg.loop_watchdog_max_strikes == 12
    d = cfg.to_dict()
    assert d["loop_watchdog_probe_interval_s"] == 45.0
    assert d["loop_watchdog_probe_timeout_s"] == 15.0
    assert d["loop_watchdog_max_strikes"] == 12

    # Nested gateway.* form honored
    nested = GatewayConfig.from_dict(
        {
            "gateway": {
                "loop_watchdog_probe_interval_s": 60,
                "loop_watchdog_probe_timeout_s": 20,
                "loop_watchdog_max_strikes": 20,
            }
        }
    )
    assert nested.loop_watchdog_probe_interval_s == 60.0
    assert nested.loop_watchdog_probe_timeout_s == 20.0
    assert nested.loop_watchdog_max_strikes == 20

    # Malformed / degenerate values fall back to safe defaults
    clamped = GatewayConfig.from_dict(
        {
            "loop_watchdog_probe_interval_s": 0,
            "loop_watchdog_probe_timeout_s": -5,
            "loop_watchdog_max_strikes": 0,
        }
    )
    assert clamped.loop_watchdog_probe_interval_s == 30.0
    assert clamped.loop_watchdog_probe_timeout_s == 10.0
    assert clamped.loop_watchdog_max_strikes == 3

def test_gateway_config_loop_watchdog_nonfinite_values_degrade():
    """NaN/Inf tuning values fall back to defaults instead of reaching the
    watchdog's Event.wait loop (or aborting config load via int(inf))."""
    from gateway.config import GatewayConfig

    cfg = GatewayConfig.from_dict(
        {
            "loop_watchdog_probe_interval_s": float("inf"),
            "loop_watchdog_probe_timeout_s": float("nan"),
            "loop_watchdog_max_strikes": float("inf"),  # int() would raise
        }
    )
    assert cfg.loop_watchdog_probe_interval_s == 30.0
    assert cfg.loop_watchdog_probe_timeout_s == 10.0
    assert cfg.loop_watchdog_max_strikes == 3

    # Oversized-but-finite values also clamp to defaults.
    big = GatewayConfig.from_dict(
        {
            "loop_watchdog_probe_interval_s": 86400,
            "loop_watchdog_probe_timeout_s": 7200,
            "loop_watchdog_max_strikes": 10**9,
        }
    )
    assert big.loop_watchdog_probe_interval_s == 30.0
    assert big.loop_watchdog_probe_timeout_s == 10.0
    assert big.loop_watchdog_max_strikes == 3

def test_load_gateway_config_bridges_loop_watchdog_keys(tmp_path, monkeypatch):
    """The real startup loader must honor gateway.loop_watchdog* from
    config.yaml — from_dict's nested fallback never sees the yaml gateway
    section because load_gateway_config builds gw_data flat."""
    from gateway.config import load_gateway_config

    (tmp_path / "config.yaml").write_text(
        "gateway:\n"
        "  loop_watchdog: false\n"
        "  loop_watchdog_probe_interval_s: 45\n"
        "  loop_watchdog_probe_timeout_s: 15\n"
        "  loop_watchdog_max_strikes: 12\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)

    cfg = load_gateway_config()
    assert cfg.loop_watchdog is False
    assert cfg.loop_watchdog_probe_interval_s == 45.0
    assert cfg.loop_watchdog_probe_timeout_s == 15.0
    assert cfg.loop_watchdog_max_strikes == 12

def test_loop_liveness_watchdog_marks_runtime_degraded_before_restart():
    """The terminal watchdog observation must be visible before ``os._exit``."""
    from gateway.status import read_runtime_status, write_runtime_status

    write_runtime_status(gateway_state="running", exit_reason=None)
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    fired = threading.Event()
    exit_codes = []

    def fake_exit(code: int) -> None:
        exit_codes.append(code)
        fired.set()

    with (
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback"),
        patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit),
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=2
        )
        assert handle is not None
        assert fired.wait(timeout=2.0), "watchdog did not reach its restart exit"
        handle.stop()
        handle.join(timeout=1.0)

    record = read_runtime_status()
    assert exit_codes == [75]
    assert record["gateway_state"] == "degraded"
    assert record["exit_reason"] == "loop_liveness_watchdog"
    assert record["restart_requested"] is True

def test_heartbeat_write_does_not_block_the_loop_it_monitors():
    """The heartbeat write must not freeze the loop the watchdog is watching.

    ``write_loop_heartbeat`` ends in ``atomic_json_write`` -> ``os.fsync``, and on
    a stalling filesystem that fsync blocks whichever thread runs it. Run inline,
    that thread was the gateway loop — so the loop-liveness watchdog would time
    out its probe (10s, 3 strikes, a ~90-120s budget) and kill the loop for being
    unresponsive at the moment it was blocked inside the watchdog's own write. A
    WSL2 VHDX under io pressure was measured stalling a trivial stat-and-fsync
    probe at p99 31s, max 112s — longer than the whole budget.

    Bounded by a fixed sleep rather than an Event handshake on purpose: if the
    write ever goes back on-loop this fails on the tick count instead of hanging.
    """
    block_s = 0.30

    def slow_write(**_kwargs):
        time.sleep(block_s)
        return pathlib.Path("/dev/null")

    async def scenario() -> int:
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            deadline = time.monotonic() + block_s
            while time.monotonic() < deadline:
                await asyncio.sleep(0.02)
                ticks += 1

        with patch(
            "gateway.shutdown_watchdog.write_loop_heartbeat", slow_write
        ):
            # should_continue False -> exactly one write, then return.
            hb = asyncio.create_task(
                loop_heartbeat_forever(interval_s=60.0, should_continue=lambda: False)
            )
            await ticker()
            await asyncio.wait_for(hb, timeout=5.0)
        return ticks

    ticks = asyncio.run(scenario())
    # Off-loop, the ticker gets roughly block_s / 0.02 ticks. Inline it gets at
    # most one, because the loop cannot run anything while fsync blocks it.
    assert ticks >= 5, (
        f'the loop made only {ticks:d} tick(s) while the heartbeat was writing — the write is blocking the loop again'
    )
