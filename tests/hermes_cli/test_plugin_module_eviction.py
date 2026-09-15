"""Plugin eviction tolerates unrelated imports and modules removed by peers."""

from types import SimpleNamespace

from hermes_cli import plugins_loader


def test_eviction_preserves_unrelated_module_added_during_filtering(monkeypatch):
    modules = {}

    class ImportingName(str):
        def startswith(self, prefix):
            # Deterministically interleave an unrelated import with prefix filtering.
            modules["unrelated.new_import"] = object()
            return super().startswith(prefix)

    modules.update({
        ImportingName("unrelated.existing"): object(),
        "hermes_plugins.example": object(),
        "hermes_plugins.example.adapter": object(),
        "hermes_plugins.example_other": object(),
    })
    monkeypatch.setattr(plugins_loader, "sys", SimpleNamespace(modules=modules))

    plugins_loader._evict_modules("hermes_plugins.example")

    assert set(modules) == {
        "unrelated.existing", "unrelated.new_import", "hermes_plugins.example_other",
    }


def test_eviction_tolerates_module_removed_after_enumeration(monkeypatch):
    class ModulesWithPeerRemoval(dict):
        def __iter__(self):
            yield from super().__iter__()
            # A peer unloads a selected module before our deletion starts.
            self.pop("hermes_plugins.example.adapter")

    modules = ModulesWithPeerRemoval({
        "hermes_plugins.example": object(),
        "hermes_plugins.example.adapter": object(),
        "unrelated": object(),
    })
    monkeypatch.setattr(plugins_loader, "sys", SimpleNamespace(modules=modules))

    plugins_loader._evict_modules("hermes_plugins.example")

    assert list(modules.keys()) == ["unrelated"]
