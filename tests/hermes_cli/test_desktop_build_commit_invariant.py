"""Packaged desktop builds must represent the refreshed checkout commit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

from hermes_cli import main as cli_main


OLD_COMMIT = "644efbd500000000000000000000000000000000"
NEW_COMMIT = "bab1651082c77af61999b0827f871b013252667f"


def _make_packaged_executable(project_root: Path) -> Path:
    release = project_root / "apps" / "desktop" / "release"
    if sys.platform == "darwin":
        executable = release / "mac-arm64" / "Hermes.app" / "Contents" / "MacOS" / "Hermes"
    elif sys.platform == "win32":
        executable = release / "win-unpacked" / "Hermes.exe"
    else:
        executable = release / "linux-unpacked" / "hermes"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"stub")
    return executable


def test_packaged_build_is_stale_when_checkout_commit_changed_but_desktop_inputs_did_not(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "hermes-agent"
    desktop_dir = project_root / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}\n", encoding="utf-8")
    (project_root / ".gitignore").write_text(
        "apps/desktop/release/\n", encoding="utf-8"
    )
    executable = _make_packaged_executable(project_root)

    package_stamp = cli_main._desktop_packaged_install_stamp_path(executable)
    package_stamp.parent.mkdir(parents=True, exist_ok=True)
    package_stamp.write_text(json.dumps({"commit": OLD_COMMIT}), encoding="utf-8")

    build_stamp = tmp_path / "desktop-build-stamp.json"
    build_stamp.write_text(
        json.dumps(
            {
                "contentHash": cli_main._compute_desktop_content_hash(project_root),
                "sourceMode": False,
            }
        ),
        encoding="utf-8",
    )

    with patch("hermes_cli.main._desktop_stamp_path", return_value=build_stamp), patch(
        "hermes_cli.main._read_git_revision_fingerprint",
        return_value=f"git:refs/heads/main:{NEW_COMMIT}",
    ):
        assert cli_main._desktop_build_needed(
            desktop_dir, project_root, source_mode=False
        )

        package_stamp.write_text(json.dumps({"commit": NEW_COMMIT}), encoding="utf-8")
        assert not cli_main._desktop_build_needed(
            desktop_dir, project_root, source_mode=False
        )
