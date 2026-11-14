"""Regression for #140080: updater helpers must survive a Python without ``os.link``.

The failure mode on Termux is ``AttributeError`` (the attribute does not exist), which the
``except OSError`` fallbacks never saw. Each helper must keep its documented behaviour —
``False`` for the "no hard links" probe, a real copy for the artifact helper.
"""

from __future__ import annotations

import os

import pytest

from hermes_cli.update_cmd_zip import _hardlink_backup, _link_or_copy_artifact
from hermes_cli.update_lock import _publish_exclusive


@pytest.fixture
def without_os_link(monkeypatch):
    monkeypatch.delattr(os, "link", raising=False)
    assert not hasattr(os, "link")


def test_publish_exclusive_writes_the_marker(tmp_path, without_os_link):
    marker = tmp_path / "update.marker"

    assert _publish_exclusive(marker, b"pid 1234") is True
    assert marker.read_bytes() == b"pid 1234"


def test_publish_exclusive_reports_a_taken_marker(tmp_path, without_os_link):
    marker = tmp_path / "update.marker"
    marker.write_bytes(b"first")

    assert _publish_exclusive(marker, b"second") is False
    assert marker.read_bytes() == b"first"


def test_hardlink_backup_reports_no_hardlink_support(tmp_path, without_os_link):
    source = tmp_path / "state.db"
    source.write_bytes(b"payload")

    assert _hardlink_backup(str(source), str(tmp_path / "state.db.bak")) is False


def test_link_or_copy_artifact_copies_the_file(tmp_path, without_os_link):
    source = tmp_path / "release.bin"
    source.write_bytes(b"artifact-bytes")
    destination = tmp_path / "staged.bin"

    _link_or_copy_artifact(str(source), str(destination))

    assert destination.read_bytes() == b"artifact-bytes"
