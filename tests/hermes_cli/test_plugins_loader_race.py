"""Concurrent module-cache changes must not abort plugin loading."""

from types import SimpleNamespace

from hermes_cli import plugins_loader


def test_module_eviction_survives_unrelated_module_added_during_selection(monkeypatch):
    root = "hermes_plugins.concurrent_example"
    modules = {}

    class ConcurrentImportName(str):
        def __eq__(self, other):
            modules.setdefault("unrelated.imported_later", object())
            return super().__eq__(other)

        __hash__ = str.__hash__

    sibling = ConcurrentImportName(f"{root}.child")
    modules[sibling] = object()
    modules[root] = object()
    modules["unrelated.existing"] = object()
    monkeypatch.setattr(plugins_loader, "sys", SimpleNamespace(modules=modules))

    plugins_loader._evict_modules(root)

    assert root not in modules
    assert sibling not in modules
    assert "unrelated.existing" in modules
    assert "unrelated.imported_later" in modules


def test_module_eviction_survives_target_removed_after_snapshot(monkeypatch):
    root = "hermes_plugins.disappearing_example"
    modules = {}

    class RemovedByAnotherLoaderName(str):
        def startswith(self, prefix):
            modules.pop(self, None)
            return super().startswith(prefix)

    sibling = RemovedByAnotherLoaderName(f"{root}.child")
    modules[sibling] = object()
    monkeypatch.setattr(plugins_loader, "sys", SimpleNamespace(modules=modules))

    plugins_loader._evict_modules(root)

    assert sibling not in modules
