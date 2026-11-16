"""A sibling whose secret store cannot serve it is parked by the reconcile, like a config outage.

A multiplexed gateway serves ``default`` and ``alpha``; ``beta`` is created afterwards and the
reconcile hot-serves it. When beta's secrets cannot be served (its one-time import refuses, or the
plane is unreachable), ``SecretStoreUnavailable`` (a ``SystemExit``, like
``ConfigBackendUnavailable``) must stop at the profile: beta is parked, the reconcile returns, the
other profiles keep their runtimes and the watcher keeps its loop (rv27 P2). The runner, its session
authorities, the reconcile and the store are real; only adapters and the API listener are stubbed.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_secret_routing as routing
from tests.plugins.secret_store.test_remote_import_refusal import REF, REFUSED_ENV

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]

_VALUES = ("key-default", "key-alpha", "key-beta", "first", "part|")


@pytest.fixture(autouse=True)
def _plane_token(monkeypatch):
    from plugins.secret_store.remote import client
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    monkeypatch.setattr(client, "bearer", lambda home: TOKEN)


@pytest.fixture(autouse=True)
def _release_reservations():
    from gateway.runtime_ownership import process_ownership
    yield
    for reserved in list(process_ownership.homes):
        process_ownership.release(reserved)


def _profile(root, name):
    home = root if name == "default" else root / "profiles" / name
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (home / ".env").write_text(f"OPENAI_API_KEY=key-{name}\n", encoding="utf-8")
    return home


async def _served_runner(tmp_path, monkeypatch):
    """A multiplexed runner serving ``default`` (launch) and ``alpha`` with real authorities."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_runtime import initialize_gateway_runtime, start_gateway_runtime_api
    from gateway.runtime_ownership import process_ownership
    import hermes_state

    root = tmp_path / ".hermes"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    homes = [("default", _profile(root, "default")), ("alpha", _profile(root, "alpha"))]
    monkeypatch.setattr(run, "_load_gateway_config", lambda *a: {"platform_toolsets": {"cli": []}})

    async def api(_runner):
        return SimpleNamespace(api_origin="http://127.0.0.1:1")
    monkeypatch.setattr("gateway.run_api.start_gateway_api", api)
    config = GatewayConfig(sessions_dir=tmp_path / "sessions")
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    runner = GatewayRunner(config)
    process_ownership.reserve([home for _, home in homes])
    await initialize_gateway_runtime(runner)
    await start_gateway_runtime_api(runner)
    runner._running = True
    runner._primary_profile_name = "default"
    runner._record_served_profiles("default", homes)

    async def no_adapters(profile_name, profile_home, claimed):
        return 0
    runner._start_one_profile_adapters = no_adapters
    return runner, root, dict(homes)


def _authority_homes(runner):
    return {Path(a.profile_id).resolve() for a in runner.session_authorities}


@pytest.mark.parametrize("failure", ["refused_import", "plane_unreachable"])
@pytest.mark.asyncio
async def test_a_sibling_the_secret_store_cannot_serve_is_parked_and_the_others_keep_serving(
        plane, monkeypatch, caplog, failure):
    from gateway.run_runtime import parked_profile_map
    runner, root, homes = await _served_runner(Path(plane.home).parents[2], monkeypatch)
    served_before = _authority_homes(runner)
    assert {homes["default"].resolve(), homes["alpha"].resolve()} <= served_before
    beta = _profile(root, "beta")
    if failure == "refused_import":
        (beta / ".env").write_text(REFUSED_ENV + "OPENAI_API_KEY=key-beta\n", encoding="utf-8", newline="")
    else:
        plane.fail_status = 503
    caplog.set_level(logging.ERROR)
    result = await runner.reconcile_served_profiles(reason="watcher")   # returns: the loop survives
    assert result["parked"] == ["beta"] and "beta" not in result["added"]
    assert "beta" in parked_profile_map(runner)
    assert runner.session_authorities.for_home(beta.resolve()) is None
    assert _authority_homes(runner) == served_before   # default and alpha keep their runtimes
    assert "beta" not in runner.served_profile_names()
    shown = caplog.text + str(parked_profile_map(runner)["beta"])
    assert "'beta'" in shown and not any(value in shown for value in _VALUES)
    if failure == "refused_import":   # stopped before anything was sent for beta
        assert REF in shown and [r for r in plane.requests if r["profile"] == "beta"] == []
    # The profile recovers on a later explicit rescan once its store can serve it.
    plane.fail_status = None
    (beta / ".env").write_text("OPENAI_API_KEY=key-beta\n", encoding="utf-8")
    result = await runner.reconcile_served_profiles(reason="control-socket")
    assert result["added"] == ["beta"] and runner.session_authorities.for_home(beta.resolve()) is not None


@pytest.mark.parametrize("exc_type", ["secret_store", "config_backend"])
@pytest.mark.asyncio
async def test_the_reconcile_watcher_survives_a_plane_outage(monkeypatch, exc_type):
    """The watcher logs a plane outage of either kind and runs its next cycle."""
    from gateway import run_profile_reconcile as rpr
    from hermes_cli.config_backend import ConfigBackendUnavailable

    class Runner(rpr.GatewayProfileReconcileMixin):
        _running = True

        def _multiplex_on(self):
            return True

        async def _sync_hooks_on_consent_change(self):
            pass

    runner = Runner()
    calls = []

    async def reconcile(*, reason="request"):
        calls.append(reason)
        if len(calls) == 1:
            raise (routing.SecretStoreUnavailable("Remote secrets: profile 'beta' unavailable")
                   if exc_type == "secret_store" else ConfigBackendUnavailable("config plane down"))
        runner._running = False
    runner.reconcile_served_profiles = reconcile
    await asyncio.wait_for(runner._profile_reconcile_watcher(interval=0), timeout=10)
    assert calls == ["watcher", "watcher"]


def test_a_platform_turn_that_hits_a_secrets_outage_ends_that_turn_only():
    """As for a config outage: the user is told once and the background task does not re-raise
    the ``SystemExit`` (which would stop the gateway's event loop for every profile)."""
    from gateway.session import build_session_key
    from tests.gateway.test_baseexception_turn_notify import _event, _make_adapter, _raising_handler
    outage = routing.SecretStoreUnavailable("Remote secrets: cannot load the secrets of profile 'beta' from https://plane")
    adapter = _make_adapter(_raising_handler(outage))
    event = _event()
    asyncio.run(adapter._process_message_background(event, build_session_key(event.source)))
    assert len(adapter.sent) == 1 and "https://plane" not in adapter.sent[0]
