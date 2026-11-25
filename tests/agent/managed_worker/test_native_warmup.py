"""Guards for the managed worker's native-import warm-up (kanban worker wedge, #58083).

A managed worker starts the memory provider's prefetch on a daemon thread during its first turn.
On Windows the first ``import numpy`` must not happen on that thread: a first-time native import
racing another thread's import chain deadlocks and then no thread in the process can be started
again, so the worker never opens the socket. ``execute()`` therefore warms the providers right after
the worker policy is bound — the point where runtime imports become legal and before the turn
machinery starts any thread.
"""

import threading

import pytest

from agent import managed_worker
import plugins.memory as memory_plugins


def test_warm_native_imports_prefers_the_shared_memory_helper(monkeypatch):
    calls = []
    monkeypatch.setattr(memory_plugins, "import_memory_provider_module", lambda: calls.append("warm") or True)

    managed_worker._warm_native_imports()

    assert calls == ["warm"]


@pytest.mark.parametrize("failing", [False, True])
def test_warm_native_imports_falls_back_to_the_native_modules(monkeypatch, failing):
    def _helper():
        if failing:
            raise RuntimeError("provider package is broken")
        return False  # no provider configured / provider isolated

    monkeypatch.setattr(memory_plugins, "import_memory_provider_module", _helper)
    imported = []
    monkeypatch.setattr(managed_worker.importlib, "import_module", lambda name: imported.append(name))

    managed_worker._warm_native_imports()

    assert imported == list(managed_worker._NATIVE_WARM_IMPORTS)


def test_execute_warms_native_imports_after_the_owner_epoch_check(monkeypatch, tmp_path):
    """The warm-up must sit after adopted process authority, and before the store/watch threads."""
    order = []
    seen = {}

    monkeypatch.setattr(managed_worker, "bind_worker_policy", lambda frame: order.append("policy"))

    def _warm():
        order.append("warm")
        seen["threads"] = [t.name for t in threading.enumerate()]

    monkeypatch.setattr(managed_worker, "_warm_native_imports", _warm)

    from agent import runtime_session_store

    class _Rpc:
        def __init__(self, home):
            order.append("rpc")

        def __call__(self, name, **kwargs):
            order.append(name)
            return {"owner_epoch": 1}

    class _Store:
        def __init__(self, *args, **kwargs):
            order.append("store")
            raise RuntimeError("stop after the warm-up")

    monkeypatch.setattr(runtime_session_store, "WorkerRPC", _Rpc)
    monkeypatch.setattr(runtime_session_store, "RuntimeSessionStore", _Store)

    frame = {
        "home": str(tmp_path),
        "scope": {
            "session_id": "s",
            "execution_id": "admission-worker:abc",
            "epoch": 1,
            "generation": 1,
        },
    }

    with pytest.raises(RuntimeError, match="stop after the warm-up"):
        managed_worker.execute(frame, channel=None)

    assert order == ["policy", "rpc", "worker.adopt", "warm", "store"]
    assert not [name for name in seen["threads"] if name.startswith("memory-prefetch-")]
