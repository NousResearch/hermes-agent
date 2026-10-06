"""Memory callers cross real PM admission before importing their optional SDK."""
import importlib.machinery
import importlib.abc
import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize("extra", ["supermemory", "mem0"])
@pytest.mark.parametrize("state", ["present", "absent", "failed"])
def test_provider_sdk_admission(monkeypatch, tmp_path, extra, state):
    import pm.client
    from pm import paths
    from plugins.memory.supermemory import _SupermemoryClient
    from plugins.memory.mem0 import Mem0MemoryProvider

    sdk = ModuleType(extra)
    sdk.__spec__ = importlib.machinery.ModuleSpec(extra, loader=None)
    sdk.Supermemory = sdk.MemoryClient = lambda **kwargs: object()
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
        if extra == "supermemory":
            return _SupermemoryClient(api_key="k", timeout=5, container_tag="hermes")
        provider = Mem0MemoryProvider()
        provider._mode, provider._api_key = "platform", "k"
        return provider._create_backend()
    if state == "failed":
        if extra == "mem0":
            assert construct() is None
        else:
            with pytest.raises(ModuleNotFoundError, match="SDK unavailable"):
                construct()
    else:
        assert construct() is not None
    assert calls == ([] if state == "present" else [[extra]])


class TestSupermemoryIsAvailable:
    def test_available_with_key_even_when_sdk_absent(self, monkeypatch):
        """With the key set but the SDK not importable, is_available() must
        still return True — otherwise the provider never loads on a sealed
        venv and ensure_import() (which installs the SDK) never runs."""
        from plugins.memory.supermemory import SupermemoryMemoryProvider
        import builtins

        monkeypatch.setenv("SUPERMEMORY_API_KEY", "sk-test")

        # Make any attempt to import the SDK fail, simulating the
        # not-yet-installed sealed-venv state.
        real_import = builtins.__import__

        def _no_supermemory(name, *args, **kwargs):
            if name == "supermemory" or name.startswith("supermemory."):
                raise ImportError("No module named 'supermemory'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_supermemory)

        prov = SupermemoryMemoryProvider()
        assert prov.is_available() is True

    def test_unavailable_without_key(self, monkeypatch):
        from plugins.memory.supermemory import SupermemoryMemoryProvider

        monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
        prov = SupermemoryMemoryProvider()
        assert prov.is_available() is False
