"""Manifest policy and ordering for the embedded kanban dispatcher."""

from __future__ import annotations

from pathlib import Path

import gateway.kanban_watchers as watchers


class FakeKanban:
    DEFAULT_BOARD = "default"

    def __init__(self, slugs):
        self._slugs = slugs

    def list_boards(self, include_archived=False):
        # Match the native API's name-sorted result, which is the regression
        # being guarded against.
        return [{"slug": slug} for slug in sorted(self._slugs)]


def _reader(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def test_manifest_filters_boards_and_preserves_manifest_order(tmp_path, monkeypatch):
    reader = _reader(
        tmp_path / "fleet_boards.py",
        "def boards_for(flag, strict=False):\n"
        "    assert flag == 'dispatch' and strict\n"
        "    return ['yorkstone-supplies', 'jarvis-os']\n",
    )
    monkeypatch.setattr(watchers, "_fleet_boards_reader_path", lambda: reader)

    assert watchers._board_slugs(
        FakeKanban(["jarvis-os", "legacy-yss", "yorkstone-supplies"])
    ) == ["yorkstone-supplies", "jarvis-os"]


def test_manifest_reader_failure_fails_closed(tmp_path, monkeypatch):
    reader = _reader(
        tmp_path / "fleet_boards.py",
        "def boards_for(flag, strict=False):\n"
        "    raise RuntimeError('manifest unavailable')\n",
    )
    monkeypatch.setattr(watchers, "_fleet_boards_reader_path", lambda: reader)

    assert watchers._board_slugs(FakeKanban(["jarvis-os"])) == []


def test_missing_manifest_keeps_upstream_board_enumeration(tmp_path, monkeypatch):
    monkeypatch.setattr(
        watchers,
        "_fleet_boards_reader_path",
        lambda: tmp_path / "not-installed" / "fleet_boards.py",
    )

    assert watchers._board_slugs(
        FakeKanban(["jarvis-os", "yorkstone-supplies"])
    ) == ["jarvis-os", "yorkstone-supplies"]
