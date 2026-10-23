"""Status polling reuses advisory work without retaining another profile's data."""
import asyncio
import contextvars
import os
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def status_route(tmp_path, monkeypatch):
    from hermes_cli.web_routers import status

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    getattr(status, '_STATUS_COUNT_CACHE', {}).clear()
    getattr(status, '_STATUS_CONFIG_CACHE', {}).clear()
    return status


def test_session_count_reuses_expires_and_separates_stores(status_route, tmp_path, monkeypatch):
    import hermes_state

    route = status_route
    clock = [100.0]
    monkeypatch.setattr(route, 'time', SimpleNamespace(time=lambda: 1000.0, monotonic=lambda: clock[0]))
    current = [tmp_path / 'a.db']
    current[0].touch()
    other = tmp_path / 'b.db'
    other.touch()
    opens = []

    class Store:
        def list_sessions_rich(self, **kwargs):
            return [{'ended_at': None, 'started_at': 999.0}] * (1 if current[0].name == 'a.db' else 2)

        def close(self):
            pass

    def open_store(*args, **kwargs):
        assert kwargs['read_only'] is True
        opens.append(current[0])
        return Store()

    monkeypatch.setattr(hermes_state, '_default_db_path', lambda: current[0])
    monkeypatch.setattr(route, '_open_session_db_for_profile', open_store)
    assert route._count_status_active_sessions() == 1
    assert route._count_status_active_sessions() == 1
    assert len(opens) == 1
    clock[0] += 6
    assert route._count_status_active_sessions() == 1
    assert len(opens) == 2
    current[0] = other
    assert route._count_status_active_sessions() == 2
    current[0] = tmp_path / 'a.db'
    assert route._count_status_active_sessions() == 1
    assert len(opens) == 3


def test_session_refresh_is_single_and_failure_releases_lock(status_route, tmp_path, monkeypatch):
    import hermes_state

    route = status_route
    key = str(tmp_path / 'state.db')
    monkeypatch.setattr(hermes_state, '_default_db_path', lambda: key)
    route._STATUS_COUNT_CACHE[key] = (-100.0, 4)
    started, release = threading.Event(), threading.Event()
    calls, errors = [], []

    def slow():
        calls.append(1)
        started.set()
        if not release.wait(5):
            raise RuntimeError('test synchronization timed out')
        return 6

    monkeypatch.setattr(route, '_count_status_active_sessions_uncached', slow)

    def worker():
        try:
            route._count_status_active_sessions()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert started.wait(5)
        assert route._count_status_active_sessions() == 4
        assert len(calls) == 1
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert errors == []
    assert route._count_status_active_sessions() == 6
    route._STATUS_COUNT_CACHE.clear()

    def fail():
        raise RuntimeError('synthetic failure')

    monkeypatch.setattr(route, '_count_status_active_sessions_uncached', fail)
    with pytest.raises(RuntimeError, match='synthetic failure'):
        route._count_status_active_sessions()
    assert not route._STATUS_COUNT_LOCK.locked()
    assert route._STATUS_COUNT_CACHE == {}


def test_config_cache_observes_replacement_deletion_and_strict_errors(status_route, tmp_path, monkeypatch):
    from hermes_cli import config

    route = status_route
    path = tmp_path / 'config.yaml'
    path.write_text('_config_version: 5\n', encoding='utf-8')
    monkeypatch.setattr(route, 'get_config_path', lambda: path)
    monkeypatch.setattr(config, 'get_config_path', lambda: path)
    calls = []

    def read_version():
        calls.append(1)
        return config.check_config_version()

    monkeypatch.setattr(route, 'check_config_version', read_version)
    assert route._cached_status_config_version()[0] == 5
    assert route._cached_status_config_version()[0] == 5
    assert len(calls) == 1
    before = path.stat()
    replacement = tmp_path / 'replacement.yaml'
    replacement.write_text('_config_version: 6\n', encoding='utf-8')
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(path)
    assert route._cached_status_config_version()[0] == 6
    path.write_text('broken: [\n', encoding='utf-8')
    route._cached_status_config_version()
    with pytest.raises(config.InvalidUserConfigError):
        config.check_config_version(raise_on_parse_error=True)
    path.unlink()
    assert route._cached_status_config_version() == config.check_config_version()


def test_config_cache_expires_and_does_not_store_a_racing_read(status_route, tmp_path, monkeypatch):
    route = status_route
    path = tmp_path / 'config.yaml'
    path.write_text('v: 1\n', encoding='utf-8')
    monkeypatch.setattr(route, 'get_config_path', lambda: path)
    clock, calls = [100.0], []
    monkeypatch.setattr(route, 'time', SimpleNamespace(monotonic=lambda: clock[0]))

    def read_version():
        calls.append(1)
        if len(calls) == 1:
            path.write_text('v: 1000\n', encoding='utf-8')
        return (len(calls), 99)

    monkeypatch.setattr(route, 'check_config_version', read_version)
    assert route._cached_status_config_version() == (1, 99)
    assert route._cached_status_config_version() == (2, 99)
    assert route._cached_status_config_version() == (2, 99)
    clock[0] += 6
    assert route._cached_status_config_version() == (3, 99)


def test_platform_config_cache_separates_profiles_and_retries_errors(status_route, tmp_path, monkeypatch):
    route = status_route
    paths = [tmp_path / 'a.yaml', tmp_path / 'b.yaml']
    for path in paths:
        path.write_text('gateway: {}\n', encoding='utf-8')
    current, calls = [paths[0]], []
    monkeypatch.setattr(route, 'get_config_path', lambda: current[0])

    def platforms():
        calls.append(current[0])
        return {current[0].stem}

    monkeypatch.setattr(route, '_load_configured_gateway_platforms', platforms)
    assert route._cached_status_configured_platforms() == {'a'}
    assert route._cached_status_configured_platforms() == {'a'}
    assert len(calls) == 1
    paths[0].write_text('gateway: {changed: true}\n', encoding='utf-8')
    assert route._cached_status_configured_platforms() == {'a'}
    assert len(calls) == 2
    current[0] = paths[1]
    assert route._cached_status_configured_platforms() == {'b'}
    assert len(calls) == 3
    route._STATUS_CONFIG_CACHE.clear()

    def fail():
        raise ValueError('synthetic invalid config')

    monkeypatch.setattr(route, '_load_configured_gateway_platforms', fail)
    with pytest.raises(ValueError, match='synthetic invalid config'):
        route._cached_status_configured_platforms()
    monkeypatch.setattr(route, '_load_configured_gateway_platforms', platforms)
    assert route._cached_status_configured_platforms() == {'b'}


def test_status_offloads_config_and_preserves_profile_context(status_route, tmp_path, monkeypatch):
    route = status_route
    current = contextvars.ContextVar('test_config_path')
    main_thread = threading.get_ident()
    observed = []

    class StopBeforeOtherWork(Exception):
        pass

    def reader():
        observed.append((threading.get_ident(), current.get()))
        raise StopBeforeOtherWork()

    monkeypatch.setattr(route, '_cached_status_config_version', reader)

    async def run():
        for name in ('a', 'b'):
            token = current.set(tmp_path / name / 'config.yaml')
            try:
                with pytest.raises(StopBeforeOtherWork):
                    await route.get_status()
            finally:
                current.reset(token)

    asyncio.run(run())
    assert len(observed) == 2
    assert all(thread != main_thread for thread, path in observed)
    assert [path.parent.name for thread, path in observed] == ['a', 'b']
