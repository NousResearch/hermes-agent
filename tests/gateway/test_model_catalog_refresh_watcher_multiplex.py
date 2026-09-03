"""Model-catalog refresh watcher must refresh EVERY multiplex profile's cache.

Regression guard: ``_model_catalog_refresh_watcher`` called
``model_catalog.refresh_catalogs()`` once per tick via ``asyncio.to_thread``
without ever entering a profile's ``_profile_runtime_scope``.
``refresh_catalogs()`` resolves its on-disk cache path via
``get_hermes_home()``, so the loop only ever refreshed the DEFAULT profile's
catalog cache. Secondary multiplex profiles never got the proactive refresh
the whole feature exists to provide -- they kept the pre-fix behavior of
refreshing only on a cold ``/model`` open.

Mirrors ``tests/gateway/test_multiplex_mcp_discovery.py`` for
``_discover_gateway_mcp_tools``, the reference template this fix follows.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from gateway.config import GatewayConfig
from hermes_constants import get_hermes_home


@pytest.mark.asyncio
async def test_single_profile_gateway_refreshes_once_unscoped(monkeypatch):
    """No multiplexing -> exactly the legacy single unscoped call."""
    import gateway.run as gateway_run

    seen: list[tuple[Path, str]] = []

    def fake_refresh() -> bool:
        seen.append((get_hermes_home(), threading.current_thread().name))
        return True

    monkeypatch.setattr("hermes_cli.model_catalog.refresh_catalogs", fake_refresh)

    await gateway_run._refresh_model_catalogs_for_all_profiles(
        GatewayConfig(multiplex_profiles=False)
    )

    assert len(seen) == 1
    # Off the event-loop thread, exactly like the pre-fix behavior.
    assert seen[0][1] != threading.current_thread().name


@pytest.mark.asyncio
async def test_multiplex_gateway_refreshes_every_profile_under_its_own_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gateway.run as gateway_run

    homes = [("default", tmp_path / "default"), ("worker", tmp_path / "worker")]
    for _name, home in homes:
        home.mkdir()
    seen: list[tuple[Path, str]] = []

    def fake_refresh() -> bool:
        seen.append((get_hermes_home(), threading.current_thread().name))
        return True

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex, profile_allowlist=None: homes,
    )
    monkeypatch.setattr("hermes_cli.model_catalog.refresh_catalogs", fake_refresh)

    await gateway_run._refresh_model_catalogs_for_all_profiles(
        GatewayConfig(multiplex_profiles=True)
    )

    # Ran once per profile, under that profile's own home, off the loop thread.
    assert [home for home, _ in seen] == [home for _, home in homes]
    assert all(thread != threading.current_thread().name for _, thread in seen)


@pytest.mark.asyncio
async def test_one_profile_refresh_failure_does_not_skip_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken/unreachable catalog fetch for one profile must not starve siblings."""
    import gateway.run as gateway_run

    homes = [("default", tmp_path / "default"), ("worker", tmp_path / "worker")]
    for _name, home in homes:
        home.mkdir()
    seen: list[Path] = []

    def flaky_refresh() -> bool:
        home = get_hermes_home()
        if home == tmp_path / "default":
            raise RuntimeError("network unreachable")
        seen.append(home)
        return True

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex, profile_allowlist=None: homes,
    )
    monkeypatch.setattr("hermes_cli.model_catalog.refresh_catalogs", flaky_refresh)

    await gateway_run._refresh_model_catalogs_for_all_profiles(
        GatewayConfig(multiplex_profiles=True)
    )

    assert seen == [tmp_path / "worker"]
