"""Gateway watcher uses the owning profile scope for automatic catalog refresh."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.run_watchers import _refresh_profile_provider_models


def test_refresh_reads_profile_config_and_secrets_in_own_scope(tmp_path, monkeypatch):
    from hermes_cli import config as config_mod
    from hermes_cli import provider_model_refresh
    from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

    homes = [tmp_path / "first", tmp_path / "second"]
    for home in homes:
        home.mkdir()

    @contextmanager
    def scoped(home):
        token = set_hermes_home_override(str(home))
        try:
            yield
        finally:
            reset_hermes_home_override(token)

    monkeypatch.setattr("gateway.run._profile_runtime_scope", scoped)
    configs = {
        str(homes[0]): {"provider_models": {"background_refresh": {
            "interval_seconds": 60, "providers": ["custom:a"]}}},
        str(homes[1]): {"provider_models": {"background_refresh": {
            "interval_seconds": 120, "providers": ["custom:b"]}}},
    }
    monkeypatch.setattr(config_mod, "load_config_readonly",
                        lambda: configs[str(get_hermes_home())])
    observed = []

    def refreshed(config):
        observed.append((str(get_hermes_home()), config))
        return (1, 0)

    monkeypatch.setattr(provider_model_refresh, "refresh_configured_provider_models",
                        refreshed)
    assert _refresh_profile_provider_models(homes[0]) == (60, 1, 0)
    assert _refresh_profile_provider_models(homes[1]) == (120, 1, 0)
    assert [name for name, _ in observed] == [str(home) for home in homes]
    assert observed[0][1] is configs[str(homes[0])]
    assert observed[1][1] is configs[str(homes[1])]


def test_disabled_profile_does_not_call_refresher(tmp_path, monkeypatch):
    from hermes_cli import config as config_mod
    from hermes_cli import provider_model_refresh

    def unexpected_scope(_home):
        raise AssertionError("disabled refresh must not hydrate profile secrets")

    monkeypatch.setattr("gateway.run._profile_runtime_scope", unexpected_scope)
    monkeypatch.setattr(config_mod, "load_config_readonly", dict)
    with patch.object(provider_model_refresh, "refresh_configured_provider_models") as refresh:
        assert _refresh_profile_provider_models(Path(tmp_path)) == (0, 0, 0)
    refresh.assert_not_called()

async def _run_watcher_ticks(monkeypatch, runner, *, stop_after=210):
    """Advance the timer without sleeping or changing asyncio's monotonic clock."""
    from types import SimpleNamespace
    from gateway import run_watchers

    clock = [0.0]
    calls = []
    actual_sleep = run_watchers.asyncio.sleep

    async def immediate_sleep(_seconds):
        await actual_sleep(0)

    async def step(_runner, seconds):
        assert seconds == 30
        clock[0] += seconds
        if clock[0] >= stop_after:
            runner._running = False

    def refresh(home, *, standalone_launch=False):
        calls.append((clock[0], str(home)))
        return (60, 1, 0)

    monkeypatch.setattr(run_watchers, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(run_watchers.asyncio, "sleep", immediate_sleep)
    monkeypatch.setattr(run_watchers, "_interruptible_sleep", step)
    monkeypatch.setattr(run_watchers, "_refresh_profile_provider_models", refresh)
    await run_watchers.GatewaySessionWatchersMixin._provider_models_refresh_watcher(runner)
    return calls


@pytest.mark.asyncio
async def test_watcher_ticks_each_selected_profile_on_configured_cadence(tmp_path, monkeypatch):
    from types import SimpleNamespace

    homes = {"one": tmp_path / "a", "two": tmp_path / "b"}
    runner = SimpleNamespace(_running=True, _served_profile_homes=homes)
    # Loop clock: 0,30,...; scheduled at 0,60,120,180; no wait for picker input.
    calls = await _run_watcher_ticks(monkeypatch, runner)
    assert calls == [
        (at, str(home))
        for at in (0, 60, 120, 180)
        for home in homes.values()
    ]


def test_real_profile_scope_uses_separate_keys_and_cache_files(tmp_path, monkeypatch):
    """End-to-end profile isolation: distinct .env values never share a cache row."""
    import json
    from hermes_cli import models

    observed = []

    def live(api_key, url, timeout=5.0, api_mode=None, headers=None):
        observed.append(api_key)
        return [f"model-{api_key}"]

    monkeypatch.setattr(models, "fetch_api_models", live)
    for label in ("one", "two"):
        home = tmp_path / label
        home.mkdir()
        (home / "config.yaml").write_text(
            "provider_models:\n"
            "  background_refresh:\n"
            "    interval_seconds: 60\n"
            "    providers: [custom:my-proxy]\n"
            "custom_providers:\n"
            "  - name: My Proxy\n"
            "    base_url: https://models.example/v1\n"
            "    key_env: PROFILE_TEST_KEY\n")
        (home / ".env").write_text(f"PROFILE_TEST_KEY={label}-key\n")
        assert _refresh_profile_provider_models(home) == (60, 1, 0)
        data = json.loads((home / "provider_models_cache.json").read_text())
        assert len(data) == 1
        assert next(iter(data.values()))["models"] == [f"model-{label}-key"]
    assert observed == ["one-key", "two-key"]


def test_standalone_host_uses_frozen_launch_scope(tmp_path, monkeypatch):
    """Do not reconstruct launch credentials from .env when hosted rooms enable multiplex."""
    from gateway.run_turn import GatewayTurnMixin
    from hermes_cli import config as config_mod
    from hermes_cli import provider_model_refresh

    scope_log = []

    @contextmanager
    def frozen_launch():
        scope_log.append("entered_launch")
        yield
        scope_log.append("exited_launch")

    def forbidden_scope(_home):
        raise AssertionError("launch credentials cannot be rebuilt as a secondary scope")

    monkeypatch.setattr(GatewayTurnMixin, "_standalone_launch_scope", staticmethod(frozen_launch))
    monkeypatch.setattr("gateway.run._profile_runtime_scope", forbidden_scope)
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: {
        "provider_models": {"background_refresh": {
            "interval_seconds": 60, "providers": ["custom:a"]}}})
    monkeypatch.setattr(provider_model_refresh, "refresh_configured_provider_models",
                        lambda config: (1, 0))
    assert _refresh_profile_provider_models(tmp_path, standalone_launch=True) == (60, 1, 0)
    assert scope_log == ["entered_launch", "exited_launch"]
