"""Behavior contracts for packaged Desktop build provenance."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from hermes_cli import main as cli_main


def _packaged_desktop(desktop_dir: Path, payload: object) -> Path:
    app_dir = desktop_dir / "release" / "test-unpacked"
    app_dir.mkdir(parents=True)
    executable = app_dir / "hermes"
    executable.write_bytes(b"executable")
    resources = app_dir / "resources"
    resources.mkdir()
    (resources / "install-stamp.json").write_text(json.dumps(payload), encoding="utf-8")
    return executable


def test_packaged_desktop_matches_its_embedded_source_hash(tmp_path: Path) -> None:
    expected = "a" * 64
    executable = _packaged_desktop(tmp_path, {"desktopContentHash": expected.upper()})

    with patch.object(cli_main, "_desktop_packaged_executable", return_value=executable):
        assert cli_main._packaged_desktop_matches_source(tmp_path, expected)
        assert not cli_main._packaged_desktop_matches_source(tmp_path, "b" * 64)


def test_packaged_desktop_without_valid_provenance_fails_closed(tmp_path: Path) -> None:
    executable = _packaged_desktop(tmp_path, {"commit": "a" * 40})

    with patch.object(cli_main, "_desktop_packaged_executable", return_value=executable):
        assert not cli_main._packaged_desktop_matches_source(tmp_path, "a" * 64)


def test_matching_external_stamp_cannot_hide_a_stale_package(tmp_path: Path) -> None:
    expected = "a" * 64
    executable = _packaged_desktop(tmp_path, {"desktopContentHash": "b" * 64})
    external_stamp = tmp_path / "desktop-build-stamp.json"
    external_stamp.write_text(
        json.dumps({"contentHash": expected, "sourceMode": False}), encoding="utf-8"
    )

    with (
        patch.object(cli_main, "_desktop_packaged_executable", return_value=executable),
        patch.object(cli_main, "_renderer_bundle_dir", return_value=None),
        patch.object(cli_main, "_desktop_stamp_path", return_value=external_stamp),
        patch.object(cli_main, "_compute_desktop_content_hash", return_value=expected),
    ):
        assert cli_main._desktop_build_needed(tmp_path, tmp_path, source_mode=False)
