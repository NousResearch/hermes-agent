"""The lazy env registry publishes a complete table once, and never retries a failed build."""

from __future__ import annotations

import threading

import hermes_cli.config as config_mod

_REGISTRY_NAMES = ("OPTIONAL_ENV_VARS", "CORE_DECLARED_ENV_NAMES", "BUNDLED_PLATFORM_SECRET_ENVS")


def _reset_registry(monkeypatch):
    """Make the registry look unbuilt; monkeypatch restores the real values at teardown."""
    for name in _REGISTRY_NAMES:
        monkeypatch.delitem(vars(config_mod), name, raising=False)
    # delitem(raising=False) only records a restore for names that existed; make sure of it.
    assert "BUNDLED_PLATFORM_SECRET_ENVS" not in vars(config_mod)


def test_registry_is_never_visible_half_built(monkeypatch):
    real = config_mod._inject_platform_plugin_env_vars
    seen_during_build: list[bool] = []

    def spy(table=None):
        # A reader on another thread, mid-build, must not see the registry names published yet.
        seen_during_build.append(any(n in vars(config_mod) for n in _REGISTRY_NAMES))
        return real(table)

    _reset_registry(monkeypatch)
    monkeypatch.setattr(config_mod, "_inject_platform_plugin_env_vars", spy)
    config_mod._complete_env_registry()
    assert seen_during_build == [False]
    assert all(n in vars(config_mod) for n in _REGISTRY_NAMES)


def test_failed_platform_injection_publishes_once_and_does_not_retry(monkeypatch):
    calls = []

    def boom(table=None):
        calls.append(1)
        raise RuntimeError("bad manifest")

    _reset_registry(monkeypatch)
    monkeypatch.setattr(config_mod, "_inject_platform_plugin_env_vars", boom)
    config_mod._complete_env_registry()
    config_mod._complete_env_registry()
    _ = config_mod.OPTIONAL_ENV_VARS
    assert calls == [1]
    assert config_mod.BUNDLED_PLATFORM_SECRET_ENVS is None  # policy re-reads bundled manifests strictly


def test_concurrent_first_reads_agree(monkeypatch):
    _reset_registry(monkeypatch)
    results: list[object] = []

    def read():
        results.append(config_mod.OPTIONAL_ENV_VARS)

    threads = [threading.Thread(target=read) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert len(results) == 4
    assert all(r is config_mod.OPTIONAL_ENV_VARS for r in results)  # one published table wins


def test_reentrant_read_on_the_building_thread_sees_the_base_table(monkeypatch):
    seen = {}
    real = config_mod._inject_platform_plugin_env_vars

    def reentrant(table=None):
        config_mod._complete_env_registry()  # a plugin touching config mid-discovery: must not recurse
        seen["table"] = config_mod._optional_env_vars()
        return real(table)

    _reset_registry(monkeypatch)
    monkeypatch.setattr(config_mod, "_inject_platform_plugin_env_vars", reentrant)
    config_mod._complete_env_registry()
    assert seen["table"] is config_mod._BASE_OPTIONAL_ENV_VARS
