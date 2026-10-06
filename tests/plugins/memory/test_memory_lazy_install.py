"""Memory callers cross real PM admission before importing their optional SDK."""
import importlib.machinery
import importlib.abc
import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize("state", ["present", "absent", "failed"])
def test_provider_sdk_admission(monkeypatch, tmp_path, state):
    extra = "mem0"
    import pm.client
    from pm import paths
    from plugins.memory.mem0 import Mem0MemoryProvider

    sdk = ModuleType(extra)
    sdk.__spec__ = importlib.machinery.ModuleSpec(extra, loader=None)
    sdk.MemoryClient = lambda **kwargs: object()
    sdk.Memory = object
    monkeypatch.delitem(sys.modules, extra, raising=False)
    if state == "present":
        monkeypatch.setitem(sys.modules, extra, sdk)
    class MissingSDK(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == extra:
                raise ModuleNotFoundError("SDK unavailable")
    monkeypatch.setattr(sys, "meta_path", [MissingSDK(), *sys.meta_path])
    monkeypatch.setattr(paths, "runtime_facts_path", lambda: tmp_path / "unselected-facts")
    calls = []
    def sync(extras):
        calls.append(extras)
        if state == "failed":
            raise RuntimeError("SDK install refused")
        monkeypatch.setitem(sys.modules, extra, sdk)
    monkeypatch.setattr(pm.client, "sync_venv", sync)
    def construct():
        provider = Mem0MemoryProvider()
        provider._mode, provider._api_key = "platform", "k"
        return provider._create_backend()
    if state == "failed":
        assert construct() is None
    else:
        assert construct() is not None
    assert calls == ([] if state == "present" else [[extra]])
