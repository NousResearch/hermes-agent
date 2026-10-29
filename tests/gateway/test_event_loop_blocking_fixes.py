"""Regression tests for synchronous work that used to block the gateway event loop.

Each case pins one of the four blocking sites found in the watchdog stack dumps of
2026-10-26 / 10-28 / 10-29: a frozen loop makes every asyncio recovery path fail with it,
which surfaced to users as "Resume failed: request timed out after 30s: session.resume" and,
past ~90 s of missed liveness probes, a forced exit (code 75) that reaped the session runtimes.
"""

import asyncio
import json
import threading

import pytest

from gateway import run as gateway_run
from gateway.config import Platform


@pytest.fixture
def isolated_bundled_cache():
    """Restore the module-level scan cache so a test cannot leak into the next one."""
    from gateway import config as config_mod

    saved = (
        config_mod._Platform__bundled_plugin_names,
        config_mod._Platform__bundled_plugin_aliases,
    )
    config_mod._Platform__bundled_plugin_names = None
    config_mod._Platform__bundled_plugin_aliases = None
    try:
        yield config_mod
    finally:
        config_mod._Platform__bundled_plugin_names, config_mod._Platform__bundled_plugin_aliases = saved


def test_cached_accessor_returns_one_stable_scan(isolated_bundled_cache):
    """``_cached_bundled_plugin_platforms`` must scan once and hand back the same objects.

    Identity, not equality: a rescan builds fresh sets, so a stable object proves the cache held.
    """
    config_mod = isolated_bundled_cache

    first_names, first_aliases = config_mod.Platform._cached_bundled_plugin_platforms()
    second_names, second_aliases = config_mod.Platform._cached_bundled_plugin_platforms()

    assert first_names, "the scan returned no bundled platforms"
    assert first_names is second_names, "accessor rescanned instead of returning the cached set"
    assert first_aliases is second_aliases, "accessor rescanned instead of returning the cached map"


def test_config_load_keeps_the_cached_scan(isolated_bundled_cache):
    """Repeated config loads must not rebuild the scan.

    ``_bundled_env_fallbacks``, ``core_ships_platform``, ``bundled_sibling`` and
    ``_static_gate_env_prefixes`` each called the raw scan, so every ``load_gateway_config()``
    re-read the whole ``plugins/platforms`` tree — including the load an adapter rebuild performs,
    which put that I/O on the gateway event loop.
    """
    config_mod = isolated_bundled_cache

    names_obj, aliases_obj = config_mod.Platform._cached_bundled_plugin_platforms()
    for _ in range(3):
        config_mod.load_gateway_config()

    assert config_mod._Platform__bundled_plugin_names is names_obj, (
        "load_gateway_config() rescanned the bundled platform manifests"
    )
    assert config_mod._Platform__bundled_plugin_aliases is aliases_obj, (
        "load_gateway_config() rescanned the bundled platform manifests"
    )


@pytest.mark.asyncio
async def test_hosted_service_skips_profile_scope_once_installed(monkeypatch):
    """The 1 s hosted-room watcher must not stat the profile home in the steady state.

    ``owner_scope()`` probes ``Path(profile_id).is_dir()`` on the loop and opens the profile's
    runtime scope; both are one-time setup, so a fully prepared authority must return first.
    """
    from gateway import session_hosted_service as hosted

    entered = []

    class FakeScope:
        def __enter__(self):
            entered.append(1)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(hosted, "owner_scope", lambda authority: FakeScope())

    class Ready:
        _transport_installed = True

    class Authority:
        hosted_room_service = Ready()

    await hosted._ensure_hosted_service(object(), Authority())

    assert entered == [], "steady-state watcher pass entered owner_scope (stat + scope open)"


@pytest.mark.asyncio
async def test_hosted_service_enters_profile_scope_while_unprepared(monkeypatch):
    """The early return must not skip the setup path: an unprepared authority still gets scoped."""
    from gateway import session_hosted_service as hosted

    entered = []

    class FakeScope:
        def __enter__(self):
            entered.append(1)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(hosted, "owner_scope", lambda authority: FakeScope())

    class Authority:
        hosted_room_service = None

    with pytest.raises(Exception):
        await hosted._ensure_hosted_service(object(), Authority())

    assert entered == [1], "unprepared authority was not scoped before setup"


@pytest.mark.asyncio
async def test_hygiene_settings_resolves_runtime_off_the_event_loop(monkeypatch):
    """Turn-prep runtime resolution can block on a credential refresh; it must not run on the loop."""
    from gateway import run_turn_hygiene as hygiene

    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda *a, **k: {})

    seen = {}

    class Bare:
        pass

    runner = Bare()

    def resolve(**kwargs):
        seen["thread"] = threading.current_thread()
        return "some/model", {"provider": "p", "base_url": "u", "api_key": "k"}

    class Settings:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    runner._resolve_session_agent_runtime = resolve
    runner._HygieneSettings = Settings
    runner._hmwa_hygiene_read_config = lambda hs, data: None

    await hygiene.GatewayTurnHygieneMixin._hmwa_hygiene_settings(runner, source=None, session_key="k")

    assert "thread" in seen, "runtime resolution was never reached"
    assert seen["thread"] is not threading.main_thread(), (
        "runtime resolution ran on the main thread — a credential refresh would freeze the loop"
    )


@pytest.mark.asyncio
async def test_reconnect_build_offloads_adapter_construction():
    """Adapter construction runs the dependency installer, so the reconnect path must dispatch it."""
    from gateway.run_adapters import GatewayAdapterLifecycleMixin

    seen = {}

    class Config:
        token = "t"
        extra = {}

    class Fake:
        def __init__(self):
            self._failed_platforms = {
                Platform.TELEGRAM: {
                    "paused": False,
                    "next_retry": 0,
                    "config": Config(),
                    "attempts": 0,
                    "reconnect_started_at": 0,
                    "last_error": None,
                },
            }
            self.config = type("C", (), {"group_sessions_per_user": True, "thread_sessions_per_user": False})()
            self.adapters = {}

        async def _run_in_executor_with_context(self, func, *args):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, func, *args)

        def _create_adapter(self, platform, config):
            seen["thread"] = threading.current_thread()

        def _flag_reconnect_needs_attention(self, *a, **k):
            return None

        def _drop_from_reconnect_queue(self, *a, **k):
            return None

        def _adapter_may_heal(self, *a, **k):
            return False

        def _update_platform_runtime_status(self, *a, **k):
            return None

        def _adapter_credential_claim(self, *a, **k):
            return None

        def _adapter_listener_claim(self, *a, **k):
            return None

    runner = Fake()
    error = None
    try:
        await GatewayAdapterLifecycleMixin._reconnect_failed_platform(
            runner, Platform.TELEGRAM, now=1_000_000.0)
    except Exception as exc:  # the assertion is about where construction ran, not how the pass ended
        error = exc

    assert "thread" in seen, f"adapter construction was never reached (pass raised {error!r})"
    assert seen["thread"] is not threading.main_thread(), (
        "adapter construction ran on the event loop"
    )
