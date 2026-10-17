"""The standalone serve backend keeps the model picker's remote catalogs warm without a gateway.

Desktop's ``hermes serve`` backend never starts the gateway's ``_model_catalog_refresh_watcher``,
so before this loop existed its disk caches only refreshed via the picker's fire-and-forget SWR
thread — which dies with the process before its fetch lands (#140702)."""
import asyncio
from contextlib import suppress

import pytest


@pytest.mark.asyncio
async def test_refresh_loop_warms_then_repeats_on_the_configured_interval(monkeypatch):
    from hermes_cli import model_catalog
    from hermes_cli.web_server_app import _model_catalog_refresh_loop

    calls: list[int] = []
    monkeypatch.setattr(model_catalog, "refresh_catalogs", lambda: calls.append(1))
    monkeypatch.setattr(model_catalog, "refresh_interval_seconds", lambda: 0.05)

    task = asyncio.create_task(_model_catalog_refresh_loop(initial_delay_s=0.01))
    try:
        async with asyncio.timeout(5):
            while len(calls) < 3:
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_refresh_loop_survives_a_failed_refresh(monkeypatch):
    from hermes_cli import model_catalog
    from hermes_cli.web_server_app import _model_catalog_refresh_loop

    calls: list[int] = []

    def _flaky_refresh() -> None:
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("catalog fetch down")

    monkeypatch.setattr(model_catalog, "refresh_catalogs", _flaky_refresh)
    monkeypatch.setattr(model_catalog, "refresh_interval_seconds", lambda: 0.05)

    task = asyncio.create_task(_model_catalog_refresh_loop(initial_delay_s=0.01))
    try:
        async with asyncio.timeout(5):
            while len(calls) < 3:
                await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_refresh_loop_falls_back_to_default_interval_when_config_unreadable(monkeypatch):
    from hermes_cli import model_catalog
    from hermes_cli.web_server_app import _model_catalog_refresh_loop

    calls: list[int] = []
    monkeypatch.setattr(model_catalog, "refresh_catalogs", lambda: calls.append(1))

    def _no_config() -> float:
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(model_catalog, "refresh_interval_seconds", _no_config)

    task = asyncio.create_task(_model_catalog_refresh_loop(initial_delay_s=0.01))
    try:
        async with asyncio.timeout(5):
            while not calls:
                await asyncio.sleep(0.01)
        # The 1200s fallback sleep replaced the crash; the loop stays parked, not dead.
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    assert calls
