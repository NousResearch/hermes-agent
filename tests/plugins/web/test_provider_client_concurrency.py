from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import types

import pytest

import tools.web_tools as web_tools
from plugins.web.exa import provider as exa_provider
from plugins.web.parallel import provider as parallel_provider


@pytest.mark.parametrize(
    (
        "provider_module",
        "lock_name",
        "getter_name",
        "cache_name",
        "env_name",
        "sdk_module_name",
        "constructor_name",
        "needs_headers",
    ),
    [
        pytest.param(
            parallel_provider,
            "_sync_client_lock",
            "_get_sync_client",
            "_parallel_client",
            "PARALLEL_API_KEY",
            "parallel",
            "Parallel",
            False,
            id="parallel-sync",
        ),
        pytest.param(
            exa_provider,
            "_client_lock",
            "_get_exa_client",
            "_exa_client",
            "EXA_API_KEY",
            "exa_py",
            "Exa",
            True,
            id="exa",
        ),
    ],
)
def test_concurrent_first_use_constructs_one_client(
    monkeypatch,
    provider_module,
    lock_name,
    getter_name,
    cache_name,
    env_name,
    sdk_module_name,
    constructor_name,
    needs_headers,
):
    workers = 8
    callers_ready = threading.Barrier(workers, timeout=5)
    release_constructor = threading.Event()
    state_changed = threading.Condition()
    lock_attempts = [0]
    constructed = []

    class ObservedLock:
        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            with state_changed:
                lock_attempts[0] += 1
                state_changed.notify_all()
            self._lock.acquire()
            return self

        def __exit__(self, *_args):
            self._lock.release()

    def constructor(*_args, **_kwargs):
        client = types.SimpleNamespace(headers={}) if needs_headers else object()
        with state_changed:
            constructed.append(client)
            state_changed.notify_all()
        assert release_constructor.wait(timeout=5)
        return client

    sdk_module = types.ModuleType(sdk_module_name)
    setattr(sdk_module, constructor_name, constructor)
    monkeypatch.setitem(sys.modules, sdk_module_name, sdk_module)
    monkeypatch.setenv(env_name, "test-key")
    monkeypatch.setattr(web_tools, cache_name, None)
    monkeypatch.setattr(provider_module, lock_name, ObservedLock())
    if provider_module is parallel_provider:
        monkeypatch.setattr(
            parallel_provider, "_ensure_parallel_sdk_installed", lambda: None
        )
    else:
        from tools import lazy_deps

        monkeypatch.setattr(lazy_deps, "ensure", lambda *_args, **_kwargs: None)

    getter = getattr(provider_module, getter_name)

    def get_client(_index):
        callers_ready.wait()
        return getter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(get_client, index) for index in range(workers)]
        with state_changed:
            reached_boundary = state_changed.wait_for(
                lambda: lock_attempts[0] == workers or len(constructed) == workers,
                timeout=5,
            )
        release_constructor.set()
        returned = [future.result(timeout=5) for future in futures]

    assert reached_boundary
    assert len(constructed) == 1
    assert all(client is returned[0] for client in returned)
