"""Regression for the 2026-10-24 outage: a non-retryable platform conflict at BOOT exited the
gateway unconditionally, ignoring ``gateway.on_all_adapters_down: stay_alive``.

One logged-out WhatsApp link produced 95 exit-78 boots in 7h19m (~12% availability) while every
run that woke during it died ``adapter_failed / Gateway draining: not_ready``. The policy already
governs "the last adapter left mid-run" and "booted with no platform"; a platform that refuses
fatally at boot is the same operator question — may one messaging transport take the process down
with it — so the conflict path must read the same answer. ``exit`` (the default) is unchanged.
"""
import asyncio
from unittest.mock import MagicMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.restart import GATEWAY_FATAL_CONFIG_EXIT_CODE
from gateway.run import GatewayRunner

WHATSAPP_LOGGED_OUT = (
    "whatsapp: WhatsApp is enabled but not paired — pair from the dashboard or run `hermes whatsapp`"
)


class _UnpairedWhatsAppAdapter(BasePlatformAdapter):
    """Boot preflight refuses non-retryably, exactly like the real ``whatsapp_not_paired`` path."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="token"), Platform.WHATSAPP)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        self._set_fatal_error("whatsapp_not_paired", WHATSAPP_LOGGED_OUT, retryable=False)
        return False

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _runner(tmp_path, policy: str | None) -> GatewayRunner:
    if policy is None:
        config = GatewayConfig(sessions_dir=tmp_path / "sessions")
    else:
        config = GatewayConfig(sessions_dir=tmp_path / "sessions", on_all_adapters_down=policy)
    return GatewayRunner(config)


def _status_capture(monkeypatch) -> list[dict]:
    import gateway.run as run_mod
    writes: list[dict] = []
    monkeypatch.setattr(
        run_mod, "_write_runtime_status_quiet", lambda **kw: writes.append(kw), raising=False
    )
    return writes


def test_sole_platform_non_retryable_boot_stays_up_under_stay_alive(tmp_path, monkeypatch):
    """The incident shape end to end through the real connect aggregation: one enabled platform,
    one non-retryable fatal, nothing connected — the process keeps serving under stay_alive."""
    _status_capture(monkeypatch)
    runner = _runner(tmp_path, "stay_alive")
    adapter = _UnpairedWhatsAppAdapter()
    retryable: list[str] = []
    nonretryable: list[str] = []
    assert asyncio.run(adapter.connect()) is False  # the boot preflight refusal, as on a real boot

    connected = asyncio.run(runner._start_aggregate_connect_results(
        [(Platform.WHATSAPP, adapter, PlatformConfig(enabled=True, token="token"), "failed", None)],
        retryable, nonretryable,
    ))
    must_exit = runner._start_handle_no_connections(connected, 1, retryable, nonretryable)

    assert (connected, retryable) == (0, [])
    assert nonretryable and "whatsapp" in nonretryable[0]
    assert must_exit is False
    assert runner._exit_cleanly is False
    assert runner._startup_parked_platforms is True
    assert runner._serving_state() == "degraded"


def test_stay_alive_parks_the_fatal_platform_instead_of_exiting(tmp_path, monkeypatch):
    """stay_alive: no exit-78 verdict, no shutdown request, and the run reports degraded."""
    status_writes = _status_capture(monkeypatch)
    runner = _runner(tmp_path, "stay_alive")
    runner.stop = MagicMock()

    must_exit = runner._start_handle_no_connections(0, 1, [], [WHATSAPP_LOGGED_OUT])

    assert must_exit is False
    assert runner._exit_cleanly is False  # the process keeps serving cron / control socket / ingress
    assert runner._exit_code != GATEWAY_FATAL_CONFIG_EXIT_CODE
    assert runner._startup_parked_platforms is True
    assert runner._serving_state() == "degraded"
    assert not [w for w in status_writes if w.get("gateway_state") == "startup_failed"]
    runner.stop.assert_not_called()


def test_exit_policy_keeps_the_historical_exit_78(tmp_path, monkeypatch):
    """Default (`exit`) is untouched: still a fatal-config exit so the service manager restarts."""
    status_writes = _status_capture(monkeypatch)
    runner = _runner(tmp_path, None)

    must_exit = runner._start_handle_no_connections(0, 1, [], [WHATSAPP_LOGGED_OUT])

    assert must_exit is True
    assert runner._exit_cleanly is True
    assert runner._exit_code == GATEWAY_FATAL_CONFIG_EXIT_CODE
    assert [w for w in status_writes if w.get("gateway_state") == "startup_failed"]
