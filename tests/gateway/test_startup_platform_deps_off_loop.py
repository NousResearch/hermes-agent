"""Regression test (#140619): the startup prefilter must not run a platform's ACTIVE dependency
installer on the event loop.

``GatewayRunner._start_prefilter_platforms`` creates one adapter per enabled platform *on the
loop thread*.  When a platform's optional extra is not installed yet, ``create_adapter()`` runs
``PlatformEntry.ensure_deps_fn`` there — ``pm.sync_venv`` on a slow disk, minutes of blocking
``subprocess``/``readline`` time.  The loop-liveness watchdog probes the loop every 10s and
hard-exits with code 75 after three consecutive misses, so the supervisor restarts the gateway
into the same install: an exit-75 restart loop that never reaches serving (observed on a
low-power NAS with a cold Feishu extra).

The prefilter therefore provisions the deps through the runner's executor hop and only then
builds the adapter.  Two independent proofs below:

* the installer runs on a non-loop thread (structural),
* the loop keeps ticking while the installer blocks (behavioural — this is the assertion that
  fails when the provisioning call is reverted to the inline path).
"""

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.run import GatewayRunner

_PLATFORM_NAME = "acme-slowdeps"
_INSTALL_S = 0.6
_TICK_S = 0.01


@pytest.fixture
def fake_platform():
    """Register a fake platform whose deps need a slow install; drop it afterwards."""
    state = {"installed": False, "install_thread": None, "check_thread": None}

    def _check() -> bool:
        state["check_thread"] = threading.get_ident()
        return state["installed"]

    def _install() -> bool:
        state["install_thread"] = threading.get_ident()
        time.sleep(_INSTALL_S)  # health: allow HX013 -- the slow installer IS the condition under test
        state["installed"] = True
        return True

    platform_registry.register(
        PlatformEntry(
            name=_PLATFORM_NAME,
            label="Acme Slowdeps",
            adapter_factory=lambda cfg: MagicMock(),
            check_fn=_check,
            ensure_deps_fn=_install,
            source="plugin",
        )
    )
    # ``Platform`` only mints a pseudo-member for an already-registered plugin name.
    platform = Platform(_PLATFORM_NAME)
    try:
        yield platform, state
    finally:
        with platform_registry._lock:
            platform_registry._entries.pop(_PLATFORM_NAME, None)


@pytest.mark.asyncio
async def test_prefilter_installs_platform_deps_off_the_event_loop(fake_platform, tmp_path):
    platform, state = fake_platform
    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True)},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    runner._wire_adapter_handlers = lambda adapter, **kw: None

    ticks: list[float] = []
    stop = asyncio.Event()
    started = asyncio.Event()

    async def _ticker() -> None:
        while not stop.is_set():
            ticks.append(time.monotonic())
            started.set()
            await asyncio.sleep(_TICK_S)

    ticker = asyncio.create_task(_ticker())
    await started.wait()  # one tick is on the loop before the prefilter can park it
    try:
        aborted, enabled, _skipped, pending = await runner._start_prefilter_platforms()
    finally:
        stop.set()
        await ticker

    assert aborted is False
    assert enabled == 1
    assert state["installed"] is True, "the platform's deps were never provisioned"
    assert state["install_thread"] != threading.get_ident(), (
        "the dependency installer ran on the event loop thread"
    )
    assert len(pending) == 1, "the adapter was not created after provisioning its deps"
    # A blocked loop yields ~0-1 ticks across the whole install; an off-loop install yields
    # roughly _INSTALL_S / _TICK_S.  The bar is deliberately loose (CI scheduling jitter).
    assert len(ticks) >= _INSTALL_S / _TICK_S / 4, (
        f"the event loop was parked for the whole install: {len(ticks)} ticks in {_INSTALL_S}s"
    )
