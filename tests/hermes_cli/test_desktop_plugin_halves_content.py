"""An installed (non-linked) desktop half reconciles by content, not by timestamp.

A source ``plugin.js`` whose bytes changed but whose mtime did not (``cp -p``, a restore, a
checkout that keeps timestamps) or went backwards (``git stash pop``) used to leave the published
copy stale: the pass kept it whenever the recorded ``sourceMtimeMs`` was greater than or equal to
the source's. The pass now compares the entry point byte for byte, as it already did for linked
dev checkouts: changed bytes re-copy whatever the clock says, matching bytes never re-copy, and a
missing installed entry point is repaired. ``sourceMtimeMs`` stays in the marker for older writers.
"""

import json
import os
import stat
from pathlib import Path

import pytest

from hermes_cli.desktop_plugin_halves import (
    PACKAGE_MARKER,
    materialize_desktop_half,
    mtime_ms,
    reconcile_desktop_halves,
)


def _package(folder: Path, body: str = "v1") -> Path:
    (folder / "desktop").mkdir(parents=True)
    (folder / "plugin.yaml").write_text("name: media\n", encoding="utf-8")
    entry = folder / "desktop" / "plugin.js"
    entry.write_text(body, encoding="utf-8")
    return entry


def _read_marker(app_root: Path, name: str = "media") -> dict:
    return json.loads((app_root / name / PACKAGE_MARKER).read_text(encoding="utf-8"))


def _retime(entry: Path, mtime_ns: int) -> None:
    os.utime(entry, ns=(mtime_ns, mtime_ns))


@pytest.mark.parametrize("shift_ns", [0, -3_600_000_000_000], ids=["same-mtime", "older-mtime"])
def test_changed_content_is_installed_whatever_the_clock_says(tmp_path, shift_ns):
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    entry = _package(root / "plugins" / "media")
    assert reconcile_desktop_halves(root, app_root) == [str(app_root / "media")]
    pinned_ns = entry.stat().st_mtime_ns + shift_ns

    entry.write_text("v2-CHANGED", encoding="utf-8")
    _retime(entry, pinned_ns)

    assert reconcile_desktop_halves(root, app_root) == [str(app_root / "media")]
    assert (app_root / "media" / "plugin.js").read_text(encoding="utf-8") == "v2-CHANGED"


def test_unchanged_content_with_a_newer_mtime_is_not_recopied(tmp_path):
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    entry = _package(root / "plugins" / "media")
    assert reconcile_desktop_halves(root, app_root) == [str(app_root / "media")]
    marker_before = (app_root / "media" / PACKAGE_MARKER).read_bytes()

    _retime(entry, entry.stat().st_mtime_ns + 3_600_000_000_000)

    assert reconcile_desktop_halves(root, app_root) == []
    assert (app_root / "media" / PACKAGE_MARKER).read_bytes() == marker_before


def test_a_legacy_marker_is_judged_by_content_not_by_its_timestamp(tmp_path):
    """A marker written before any content fix (package/source/mtime only) needs no migration:
    changed bytes pinned to the recorded mtime still re-copy."""
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    entry = _package(root / "plugins" / "media")
    assert reconcile_desktop_halves(root, app_root) == [str(app_root / "media")]
    recorded_ns = entry.stat().st_mtime_ns
    legacy = {"package": "media", "source": str(entry.parent),
              "sourceMtimeMs": mtime_ms(entry.stat())}
    (app_root / "media" / PACKAGE_MARKER).write_text(json.dumps(legacy), encoding="utf-8")

    entry.write_text("v2-CHANGED", encoding="utf-8")
    _retime(entry, recorded_ns)

    assert reconcile_desktop_halves(root, app_root) == [str(app_root / "media")]
    assert (app_root / "media" / "plugin.js").read_text(encoding="utf-8") == "v2-CHANGED"


def test_a_missing_installed_entry_point_is_repaired(tmp_path):
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    _package(root / "plugins" / "media")
    assert reconcile_desktop_halves(root, app_root) == [str(app_root / "media")]
    (app_root / "media" / "plugin.js").unlink()

    assert reconcile_desktop_halves(root, app_root) == [str(app_root / "media")]
    assert (app_root / "media" / "plugin.js").read_text(encoding="utf-8") == "v1"


def test_a_corrupt_marker_timestamp_is_repaired_without_a_recopy(tmp_path):
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    _package(root / "plugins" / "media")
    assert reconcile_desktop_halves(root, app_root) == [str(app_root / "media")]
    marker_path = app_root / "media" / PACKAGE_MARKER
    marker_path.write_text(json.dumps({**_read_marker(app_root), "sourceMtimeMs": None}),
                           encoding="utf-8")

    assert reconcile_desktop_halves(root, app_root) == []
    assert isinstance(_read_marker(app_root)["sourceMtimeMs"], (int, float))
    assert (app_root / "media" / "plugin.js").read_text(encoding="utf-8") == "v1"


def test_a_fresh_install_stamps_the_marker_older_writers_read(tmp_path):
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    _package(root / "plugins" / "media")

    assert reconcile_desktop_halves(root, app_root) == [str(app_root / "media")]
    marker = _read_marker(app_root)
    assert (marker["package"], marker["source"]) == ("media", str(root / "plugins" / "media" / "desktop"))
    assert marker["sourceMtimeMs"] > 0


@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes into read-only folders")
def test_an_unwritable_marker_repair_leaves_current_content_alone(tmp_path):
    """The timestamp repair is best-effort: content already current stays current (no raise, no
    recopy) when the marker cannot be rewritten; the next pass retries the repair."""
    root, app_root = tmp_path / "hermes", tmp_path / "hermes" / "desktop-plugins"
    _package(root / "plugins" / "media")
    assert reconcile_desktop_halves(root, app_root) == [str(app_root / "media")]
    target = app_root / "media"
    (target / PACKAGE_MARKER).write_text(
        json.dumps({**_read_marker(app_root), "sourceMtimeMs": None}), encoding="utf-8")
    os.chmod(target, stat.S_IREAD | stat.S_IEXEC)
    try:
        assert materialize_desktop_half(root / "plugins" / "media", app_root, "media") is None
    finally:
        os.chmod(target, 0o700)
    assert (target / "plugin.js").read_text(encoding="utf-8") == "v1"
