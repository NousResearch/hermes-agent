"""Regression tests for #60955: gateway must not freeze fallback_providers.

Cron reloads ``fallback_providers`` from disk on every job. The gateway used to
freeze ``self._fallback_model`` at process start, so a chain configured (or
edited) after ``hermes gateway`` was already running never reached messaging
sessions — even though cron in the same process fell back correctly.

These tests pin the reload + cached-agent apply helpers without driving the
full Feishu session path.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def fallback_runner(tmp_path, monkeypatch):
    from gateway import run as gateway_run

    home = tmp_path / ".hermes"
    routed = home / "profiles" / "b"
    routed.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    return home, routed


def _write_fallback(home, model):
    (home / "config.yaml").write_text(
        f"fallback_providers:\n  - provider: openrouter\n    model: {model}\n",
        encoding="utf-8",
    )
    return [{"provider": "openrouter", "model": model}]


@pytest.mark.parametrize("multiplex", [False, True])
def test_refresh_failure_recovers_only_own_home(fallback_runner, multiplex):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner, _profile_runtime_scope

    home, routed = fallback_runner
    launch_chain = _write_fallback(home, "A/fallback")
    runner = GatewayRunner(GatewayConfig(
        multiplex_profiles=multiplex, sessions_dir=home / "sessions"))
    assert runner._fallback_model == launch_chain

    # Both files tear before the first refresh: the launch snapshot is still valid
    # for A, but B has never supplied a chain and must not inherit that snapshot.
    for target in (home, routed):
        (target / "config.yaml").write_text("fallback_providers: [", encoding="utf-8")
    with _profile_runtime_scope(routed):
        assert runner._refresh_fallback_model() is None
        assert runner._fallback_model is None
    assert runner._refresh_fallback_model() == launch_chain

    updated_launch_chain = _write_fallback(home, "A/updated")
    assert runner._refresh_fallback_model() == updated_launch_chain
    with _profile_runtime_scope(routed):
        assert runner._refresh_fallback_model() is None
    assert runner._refresh_fallback_model() == updated_launch_chain
    routed_chain = _write_fallback(routed, "B/fallback")
    with _profile_runtime_scope(routed):
        assert runner._refresh_fallback_model() == routed_chain
    assert runner._refresh_fallback_model() == [{"provider": "openrouter", "model": "A/updated"}]
    (routed / "config.yaml").write_text("fallback_providers: [", encoding="utf-8")
    with _profile_runtime_scope(routed):
        assert runner._refresh_fallback_model() == routed_chain
    assert runner._refresh_fallback_model() == [{"provider": "openrouter", "model": "A/updated"}]


@pytest.mark.parametrize("cleared_config", [None, "model: primary\n", "fallback_providers: []\n"])
def test_successful_clear_survives_later_profile_read_failure(fallback_runner, cleared_config):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner, _profile_runtime_scope

    home, routed = fallback_runner
    launch_chain = _write_fallback(home, "A/fallback")
    _write_fallback(routed, "B/fallback")
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True, sessions_dir=home / "sessions"))
    with _profile_runtime_scope(routed):
        assert runner._refresh_fallback_model() == [{"provider": "openrouter", "model": "B/fallback"}]
        cfg = routed / "config.yaml"
        if cleared_config is None:
            cfg.unlink()
        else:
            cfg.write_text(cleared_config, encoding="utf-8")
        assert runner._refresh_fallback_model() is None
    assert runner._refresh_fallback_model() == launch_chain
    cfg.write_text("fallback_providers: [", encoding="utf-8")
    with _profile_runtime_scope(routed):
        assert runner._refresh_fallback_model() is None
    assert runner._refresh_fallback_model() == launch_chain


@pytest.mark.parametrize("routed_state", ["valid", "torn", "missing"])
def test_concurrent_refresh_returns_own_profile_chain(fallback_runner, routed_state):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner, _profile_runtime_scope

    home, routed = fallback_runner
    launch_chain = _write_fallback(home, "A/fallback")
    routed_chain = _write_fallback(routed, "B/fallback")
    assigned = threading.Event()
    release = threading.Event()
    worker = threading.local()

    class PausingRunner(GatewayRunner):
        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if name == "_fallback_model" and getattr(worker, "pause", False):
                # Pause B after the compatibility slot is written, before refresh
                # returns; A then overwrites that slot on the very same runner.
                assigned.set()
                assert release.wait(5), "launch refresh did not release routed refresh"

    runner = PausingRunner(GatewayConfig(multiplex_profiles=True, sessions_dir=home / "sessions"))
    with _profile_runtime_scope(routed):
        assert runner._refresh_fallback_model() == routed_chain
    if routed_state == "torn":
        (routed / "config.yaml").write_text("fallback_providers: [", encoding="utf-8")
    elif routed_state == "missing":
        (routed / "config.yaml").unlink()
    expected = None if routed_state == "missing" else routed_chain
    # A must also retain its constructor snapshot while B is paused.
    (home / "config.yaml").write_text("fallback_providers: [", encoding="utf-8")

    def refresh_routed():
        with _profile_runtime_scope(routed):
            worker.pause = True
            return runner._refresh_fallback_model()

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(refresh_routed)
        try:
            assert assigned.wait(5), "routed refresh did not reach the assignment"
            assert runner._refresh_fallback_model() == launch_chain
        finally:
            release.set()
        assert pending.result(timeout=5) == expected

    # Recovery remains profile-scoped after both concurrent calls complete.
    (routed / "config.yaml").write_text("fallback_providers: [", encoding="utf-8")
    with _profile_runtime_scope(routed):
        assert runner._refresh_fallback_model() == expected
    assert runner._refresh_fallback_model() == launch_chain


def test_bootstrap_cache_retains_launch_snapshot(fallback_runner):
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner

    home, _ = fallback_runner
    launch_chain = _write_fallback(home, "A/fallback")
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    # A worker-mutated compatibility slot must not replace the launch snapshot.
    runner._fallback_model = None
    (home / "config.yaml").write_text("fallback_providers: [", encoding="utf-8")
    assert runner._refresh_fallback_model() == launch_chain


def test_partial_runner_does_not_seed_from_shared_slot(fallback_runner):
    from gateway.run import GatewayRunner, _profile_runtime_scope

    home, routed = fallback_runner
    stale_chain = _write_fallback(home, "A/stale")
    runner = object.__new__(GatewayRunner)
    runner._fallback_model = stale_chain
    (routed / "config.yaml").write_text("fallback_providers: [", encoding="utf-8")
    with _profile_runtime_scope(routed):
        assert runner._refresh_fallback_model() is None
    launch_chain = _write_fallback(home, "A/current")
    assert runner._refresh_fallback_model() == launch_chain
    cache = runner._fallback_model_by_home
    with _profile_runtime_scope(routed):
        routed_chain = _write_fallback(routed, "B/current")
        assert runner._refresh_fallback_model() == routed_chain
    assert runner._fallback_model_by_home is cache
    (home / "config.yaml").write_text("fallback_providers: [", encoding="utf-8")
    assert runner._refresh_fallback_model() == launch_chain


def test_refresh_fallback_model_rereads_config(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
    )

    runner = SimpleNamespace(
        _fallback_model=None,
    )
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    chain = bound()

    assert chain == [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    assert runner._fallback_model == chain

    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: openrouter\n"
        "    model: anthropic/claude-sonnet-4.6\n"
    )
    updated = bound()
    assert updated == [
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}
    ]
    assert runner._fallback_model == updated


def test_apply_fallback_chain_skips_while_cooldown_holds_fallback():
    """Do not clobber a live fallback activation during its cooldown window."""
    from gateway.run import GatewayRunner

    live = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    agent = SimpleNamespace(
        _fallback_chain=live,
        _fallback_model=live[0],
        _fallback_index=1,
        _fallback_activated=True,
        _rate_limited_until=time.monotonic() + 30,
    )
    GatewayRunner._apply_fallback_chain_to_agent(
        agent,
        [{"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}],
    )

    assert agent._fallback_chain == live
    assert agent._fallback_index == 1
    assert agent._fallback_activated is True




def test_load_fallback_model_static_unchanged_contract(tmp_path, monkeypatch):
    """_load_fallback_model remains a pure static reader used by refresh."""
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
        "fallback_model:\n"
        "  provider: nous\n"
        "  model: Hermes-4\n"
    )

    chain = GatewayRunner._load_fallback_model()
    assert chain == [
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "nous", "model": "Hermes-4"},
    ]
