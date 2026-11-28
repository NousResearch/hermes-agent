"""URL-installed plugins outside the handoff map must not load the catalog."""

import json

from hermes_cli import plugin_catalog
from hermes_cli import plugins_cmd_catalog as catalog


def test_unrelated_install_annotations_need_no_catalog(tmp_path, monkeypatch):
    home = tmp_path / "home"
    plugin = home / "plugins" / "ordinary"
    plugin.mkdir(parents=True)
    metadata = plugin.parent / ".install-metadata.json"
    original = json.dumps({"ordinary": {
        "source": "https://example.invalid/ordinary.git",
        "revision": "a" * 40,
        "pinned": False,
    }})
    metadata.write_text(original, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    def unavailable_catalog():
        raise AssertionError("unrelated install provenance must not resolve the catalog")

    monkeypatch.setattr(plugin_catalog, "load_catalog_live", unavailable_catalog)

    assert catalog.catalog_install_record(plugin) is None
    assert catalog.catalog_annotation(plugin) is None
    assert catalog.catalog_row_fields(plugin, {}) == {}
    removed = [plugin_catalog.RemovedEntry(name="ordinary", reason="withdrawn")]
    assert catalog.removed_annotation("ordinary", plugin, removed) == "withdrawn"
    assert metadata.read_text(encoding="utf-8-sig") == original
