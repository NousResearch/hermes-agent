"""Transient Windows holds on a plugin manifest are retried, not fatal.

A freshly cloned plugin tree is intermittently held by antivirus/indexer scans
(WinError 5 -> PermissionError) for moments; ``pm.filesystem.retry_held`` exists for
exactly that class. The manifest inspector must ride out the hold instead of turning
it into a hard ``ValueError`` (Desktop install failure) or a silently dropped row.
"""
from __future__ import annotations

from errno import EACCES
from pathlib import Path

import pytest

from pm.plugin_declarations import native_manifest_file, read_native_manifest


@pytest.fixture
def held_manifest(tmp_path, monkeypatch):
    """A plugin dir whose manifest stat/read fail with PermissionError twice, then succeed —
    the shape of a Windows hold that clears in moments."""
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.yaml").write_text("name: held\nversion: 1.0.0\n", encoding="utf-8-sig")
    state = {"stats": 0, "reads": 0}
    real_stat, real_read = Path.stat, Path.read_text

    def stat(self, *args, **kwargs):
        if self.name in ("plugin.yaml", "plugin.yml") and state["stats"] < 2:
            state["stats"] += 1
            raise PermissionError(EACCES, "Access is denied")
        return real_stat(self, *args, **kwargs)

    def read_text(self, *args, **kwargs):
        if self.name in ("plugin.yaml", "plugin.yml") and state["reads"] < 2:
            state["reads"] += 1
            raise PermissionError(EACCES, "Access is denied")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "read_text", read_text)
    return plugin_dir, state


def test_manifest_stat_and_read_ride_out_a_transient_hold(held_manifest):
    """A PermissionError that clears after retries is retried: the manifest is found and read."""
    plugin_dir, state = held_manifest
    assert native_manifest_file(plugin_dir) == plugin_dir / "plugin.yaml"
    assert read_native_manifest(plugin_dir / "plugin.yaml")["name"] == "held"
    assert state["stats"] == 2 and state["reads"] == 2  # both denials were retried, not fatal
