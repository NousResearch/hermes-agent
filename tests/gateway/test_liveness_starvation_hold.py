"""A HOST-STARVED gateway must HOLD, not self-exit 75.

Observed on a 32-core host whose 1-minute load average reached 538 (runaway busy-loops in a
sibling process): the loop-liveness watchdog read the resulting probe misses as a wedge and
hard-exited 75 twice, and the supervisor relaunched into a 12-minute boot with every platform
adapter timing out at once.

Exit-75 exists for a WEDGED loop (deadlock / synchronous block) — a restart genuinely recovers
that. CPU starvation is not a wedge, and a restart makes it strictly worse, because the new
process contends for the same CPU.

So the missed-probe path must CLASSIFY before it exits:
  * load1 above max(factor * ncpu, floor) -> ``starved``: log one structured line and hold,
    leaving the strike counter one below the limit so the next healthy probe clears it and the
    next missed one re-evaluates.
  * anything else -> ``wedged``: exit 75, byte-for-byte as before.
A starved hold is bounded: after ``liveness_starvation_max_hold_s`` of continuous starvation with
no successful probe, exit 75 anyway.

Load is always MOCKED here — generating real load would be the very condition under test.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, load_gateway_config
from gateway.shutdown_watchdog import (
    DEFAULT_LIVENESS_STARVATION_LOAD_FACTOR,
    DEFAULT_LIVENESS_STARVATION_MAX_HOLD_S,
    LIVENESS_STARVATION_LOAD_FLOOR,
    evaluate_liveness_miss,
    start_loop_liveness_watchdog,
)
from hermes_cli.config_defaults import DEFAULT_CONFIG


def _dead_loop() -> MagicMock:
    """A loop that never runs the probe callback -> every probe misses."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.call_soon_threadsafe.side_effect = lambda callback: None
    return loop


def _critical_lines(critical_mock) -> list[str]:
    lines = []
    for call in critical_mock.call_args_list:
        args = call.args
        if not args:
            continue
        try:
            lines.append(args[0] % args[1:] if len(args) > 1 else args[0])
        except Exception:
            lines.append(str(args[0]))
    return lines


# --------------------------------------------------------------------------
# (a) missed probes under HIGH load -> hold, page, strike counter decremented
# --------------------------------------------------------------------------


def test_starved_miss_holds_instead_of_exiting():
    exit_codes: list[int] = []
    with (
        patch("gateway.shutdown_watchdog.logger.critical") as critical,
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback"),
        patch("gateway.shutdown_watchdog.os._exit", side_effect=exit_codes.append),
        patch("gateway.shutdown_watchdog.os.getloadavg", return_value=(538.0, 400.0, 200.0)),
        patch("gateway.shutdown_watchdog.os.cpu_count", return_value=32),
    ):
        handle = start_loop_liveness_watchdog(
            _dead_loop(),
            probe_interval=0.01,
            probe_timeout=0.01,
            max_strikes=2,
            starvation_max_hold_s=900.0,
        )
        assert handle is not None
        time.sleep(0.6)
        handle.stop()
        handle.join(timeout=2.0)

    assert exit_codes == [], "a starved gateway must not self-exit"
    lines = _critical_lines(critical)
    starved = [ln for ln in lines if "PHASE=liveness_starved" in ln]
    assert starved, f"expected a PHASE=liveness_starved page, got {lines!r}"
    assert not any("giveup" in ln for ln in lines)
    assert "load1=538.00" in starved[0]
    assert "ncpu=32" in starved[0]
    assert "strikes=2" in starved[0]
    # The counter is reset to max_strikes-1, so the very next missed probe
    # re-evaluates rather than the hold silently swallowing all future misses.
    assert len(starved) >= 2, "each subsequent missed probe must re-evaluate"


def test_starved_hold_decrements_the_strike_counter_to_one_below_the_limit():
    decision = evaluate_liveness_miss(
        strikes=3,
        strikes_limit=3,
        load1=538.0,
        ncpu=32,
        load_factor=2.0,
        starved_since=None,
        now=1000.0,
        max_hold_s=900.0,
    )
    assert decision.action == "hold"
    assert decision.strikes == 2
    assert decision.phase == "liveness_starved"
    assert decision.starved_since == 1000.0


# --------------------------------------------------------------------------
# (b) REGRESSION PIN: missed probes under LOW load -> exit 75, unchanged
# --------------------------------------------------------------------------


def test_wedged_miss_under_low_load_still_exits_75_unchanged():
    exit_codes: list[int] = []
    handle_ref: dict = {}

    def record_and_disarm(code: int) -> None:
        # os._exit is mocked, so the watchdog thread would otherwise loop on and
        # call the REAL os._exit once this patch lifts, killing the test process.
        exit_codes.append(code)
        handle_ref["handle"].stop()

    with (
        patch("gateway.shutdown_watchdog.logger.critical") as critical,
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback") as dump,
        patch("gateway.shutdown_watchdog.os._exit", side_effect=record_and_disarm),
        patch("gateway.shutdown_watchdog.os.getloadavg", return_value=(1.5, 1.2, 1.0)),
        patch("gateway.shutdown_watchdog.os.cpu_count", return_value=32),
    ):
        handle = start_loop_liveness_watchdog(
            _dead_loop(), probe_interval=0.01, probe_timeout=0.01, max_strikes=1
        )
        assert handle is not None
        handle_ref["handle"] = handle
        handle.join(timeout=2.0)
        handle.stop()
        handle.join(timeout=2.0)

    assert 75 in exit_codes, "a genuine wedge must still hard-exit 75"
    dump.assert_called_with(all_threads=True)
    lines = _critical_lines(critical)
    assert lines, "the wedge path must still log its CRITICAL"
    assert "missed" in lines[0] and "liveness probes" in lines[0]
    assert not any("PHASE=liveness_starved" in ln for ln in lines)


def test_wedged_decision_when_load_is_below_the_floor():
    # 4 cpus, factor 2 -> 2*ncpu is only 8; the absolute floor is what stops a
    # small host from being declared "starved" at a trivially low load.
    decision = evaluate_liveness_miss(
        strikes=3,
        strikes_limit=3,
        load1=LIVENESS_STARVATION_LOAD_FLOOR - 0.1,
        ncpu=4,
        load_factor=2.0,
        starved_since=None,
        now=1000.0,
        max_hold_s=900.0,
    )
    assert decision.action == "exit"
    assert decision.phase is None


def test_unavailable_load_average_is_treated_as_wedged():
    """Windows has no os.getloadavg; absence must not create a hold."""
    decision = evaluate_liveness_miss(
        strikes=3,
        strikes_limit=3,
        load1=None,
        ncpu=None,
        load_factor=2.0,
        starved_since=None,
        now=1000.0,
        max_hold_s=900.0,
    )
    assert decision.action == "exit"
    assert decision.phase is None


# --------------------------------------------------------------------------
# (c) starvation beyond max_hold -> exit 75 with the giveup line
# --------------------------------------------------------------------------


def test_starvation_beyond_max_hold_gives_up_and_exits():
    decision = evaluate_liveness_miss(
        strikes=3,
        strikes_limit=3,
        load1=538.0,
        ncpu=32,
        load_factor=2.0,
        starved_since=100.0,
        now=100.0 + 900.0,
        max_hold_s=900.0,
    )
    assert decision.action == "exit"
    assert decision.phase == "liveness_starved_giveup"


def test_starvation_within_max_hold_keeps_holding():
    decision = evaluate_liveness_miss(
        strikes=3,
        strikes_limit=3,
        load1=538.0,
        ncpu=32,
        load_factor=2.0,
        starved_since=100.0,
        now=100.0 + 899.0,
        max_hold_s=900.0,
    )
    assert decision.action == "hold"
    assert decision.phase == "liveness_starved"
    assert decision.starved_since == 100.0


def test_watchdog_gives_up_and_exits_75_after_the_hold_ceiling():
    exit_codes: list[int] = []
    handle_ref: dict = {}

    def record_and_disarm(code: int) -> None:
        exit_codes.append(code)
        handle_ref["handle"].stop()

    with (
        patch("gateway.shutdown_watchdog.logger.critical") as critical,
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback"),
        patch("gateway.shutdown_watchdog.os._exit", side_effect=record_and_disarm),
        patch("gateway.shutdown_watchdog.os.getloadavg", return_value=(538.0, 400.0, 200.0)),
        patch("gateway.shutdown_watchdog.os.cpu_count", return_value=32),
    ):
        handle = start_loop_liveness_watchdog(
            _dead_loop(),
            probe_interval=0.01,
            probe_timeout=0.01,
            max_strikes=1,
            starvation_max_hold_s=0.0,  # ceiling already exceeded on arrival
        )
        assert handle is not None
        handle_ref["handle"] = handle
        handle.join(timeout=2.0)
        handle.stop()
        handle.join(timeout=2.0)

    assert 75 in exit_codes
    lines = _critical_lines(critical)
    assert any("PHASE=liveness_starved_giveup" in ln for ln in lines), lines
    giveup = [ln for ln in lines if "PHASE=liveness_starved_giveup" in ln][0]
    assert "load1=538.00" in giveup and "ncpu=32" in giveup


def test_a_healthy_probe_clears_the_starvation_hold():
    """starved_since must not survive a successful probe."""
    decision = evaluate_liveness_miss(
        strikes=3,
        strikes_limit=3,
        load1=538.0,
        ncpu=32,
        load_factor=2.0,
        starved_since=None,  # cleared by the healthy probe
        now=100.0 + 5000.0,
        max_hold_s=900.0,
    )
    assert decision.action == "hold", "the hold clock restarts after a good probe"
    assert decision.starved_since == 100.0 + 5000.0


# --------------------------------------------------------------------------
# (d) knob declaration / default / config-bridge, house style
# --------------------------------------------------------------------------


def test_knobs_are_declared_in_default_config():
    """Absent here, `hermes config set gateway.<knob>` warns falsely."""
    assert "liveness_starvation_load_factor" in DEFAULT_CONFIG["gateway"]
    assert "liveness_starvation_max_hold_s" in DEFAULT_CONFIG["gateway"]


def test_default_config_values_match_the_code_defaults():
    assert (
        float(DEFAULT_CONFIG["gateway"]["liveness_starvation_load_factor"])
        == DEFAULT_LIVENESS_STARVATION_LOAD_FACTOR
        == 2.0
    )
    assert (
        float(DEFAULT_CONFIG["gateway"]["liveness_starvation_max_hold_s"])
        == DEFAULT_LIVENESS_STARVATION_MAX_HOLD_S
        == 900.0
    )


def test_gateway_config_round_trips_the_knobs():
    default = GatewayConfig.from_dict({})
    assert default.liveness_starvation_load_factor == 2.0
    assert default.liveness_starvation_max_hold_s == 900.0

    cfg = GatewayConfig.from_dict(
        {
            "liveness_starvation_load_factor": 4,
            "liveness_starvation_max_hold_s": 120,
        }
    )
    assert cfg.liveness_starvation_load_factor == 4.0
    assert cfg.liveness_starvation_max_hold_s == 120.0
    d = cfg.to_dict()
    assert d["liveness_starvation_load_factor"] == 4.0
    assert d["liveness_starvation_max_hold_s"] == 120.0

    nested = GatewayConfig.from_dict(
        {
            "gateway": {
                "liveness_starvation_load_factor": 3,
                "liveness_starvation_max_hold_s": 60,
            }
        }
    )
    assert nested.liveness_starvation_load_factor == 3.0
    assert nested.liveness_starvation_max_hold_s == 60.0


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan"), "nonsense", None])
def test_invalid_knob_values_degrade_to_defaults(bad):
    cfg = GatewayConfig.from_dict(
        {
            "liveness_starvation_load_factor": bad,
            "liveness_starvation_max_hold_s": bad,
        }
    )
    assert cfg.liveness_starvation_load_factor == 2.0
    assert cfg.liveness_starvation_max_hold_s == 900.0


def test_load_gateway_config_bridges_the_knobs(tmp_path, monkeypatch):
    """The real loader builds gw_data FLAT; without the bridge these are inert."""
    (tmp_path / "config.yaml").write_text(
        "gateway:\n"
        "  liveness_starvation_load_factor: 5\n"
        "  liveness_starvation_max_hold_s: 300\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = load_gateway_config()
    assert cfg.liveness_starvation_load_factor == 5.0
    assert cfg.liveness_starvation_max_hold_s == 300.0
