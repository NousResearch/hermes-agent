"""Regression for #157829: retry transient Windows permission errors when publishing provider snapshots."""

from __future__ import annotations

from unittest.mock import Mock

import pytest


@pytest.mark.platforms("windows")
def test_snapshot_retries_transient_publish_permission_error(tmp_path, monkeypatch):
    """Verify a transient Windows rename denial is retried and the frozen tree is published.

    @param tmp_path: Isolated filesystem root for the plugin and snapshot.
    @param monkeypatch: Temporary replacements for rename and retry delay.
    @returns: None.
    """
    from providers import plugin_reload

    plugin_dir = tmp_path / "plugins" / "snapshot-demo"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.setattr(plugin_reload, "_snapshot_root", tmp_path / "snapshots")
    monkeypatch.setattr(plugin_reload.time, "sleep", lambda _seconds: None)

    original_rename = plugin_reload.os.rename
    attempts = 0

    def rename_with_transient_failure(source, destination):
        """Fail once, then publish with the real filesystem rename.

        @param source: Private staged plugin directory.
        @param destination: Final frozen snapshot directory.
        @returns: Result from the real rename call.
        """
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(13, "sharing violation")
        return original_rename(source, destination)

    rename = Mock(side_effect=rename_with_transient_failure)
    monkeypatch.setattr(plugin_reload.os, "rename", rename)

    snapshot = plugin_reload._snapshot(plugin_dir, "snapshot_demo_v1")

    assert rename.call_count == 2
    assert (snapshot / "__init__.py").read_text(encoding="utf-8") == "value = 1\n"
    assert list(snapshot.parent.glob(".copy-*")) == []
