"""_tui_need_npm_install: auto npm when node_modules is behind the lockfile."""

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def main_mod():
    import hermes_cli.main as m

    return m


def _touch_ink(root: Path) -> None:
    ink = root / "node_modules" / "@hermes" / "ink" / "package.json"
    ink.parent.mkdir(parents=True, exist_ok=True)
    ink.write_text("{}")


def _touch_tui_entry(root: Path) -> None:
    entry = root / "dist" / "entry.js"
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text("console.log('tui')")


def _touch_ink_bundle(root: Path) -> None:
    bundle = root / "packages" / "hermes-ink" / "dist" / "ink-bundle.js"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text("export {}")


def test_need_install_when_ink_missing(tmp_path: Path, main_mod) -> None:
    (tmp_path / "package-lock.json").write_text("{}")
    assert main_mod._tui_need_npm_install(tmp_path) is True


def test_no_install_when_lock_newer_but_hidden_lock_matches(tmp_path: Path, main_mod) -> None:
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text('{"packages":{"node_modules/foo":{"version":"1.0.0"}}}')
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0","ideallyInert":true}}}'
    )
    os.utime(tmp_path / "package-lock.json", (200, 200))
    os.utime(tmp_path / "node_modules" / ".package-lock.json", (100, 100))
    assert main_mod._tui_need_npm_install(tmp_path) is False


def test_need_install_when_required_package_missing_from_hidden_lock(tmp_path: Path, main_mod) -> None:
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0"},"node_modules/bar":{"version":"1.0.0"}}}'
    )
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0"}}}'
    )
    assert main_mod._tui_need_npm_install(tmp_path) is True


def test_no_install_when_only_optional_peer_package_missing_from_hidden_lock(tmp_path: Path, main_mod) -> None:
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0"},"node_modules/optional":{"version":"1.0.0","optional":true,"peer":true}}}'
    )
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0"}}}'
    )
    assert main_mod._tui_need_npm_install(tmp_path) is False


def test_no_install_when_only_peer_annotation_differs(tmp_path: Path, main_mod) -> None:
    """npm 9 drops the ``peer`` flag from the hidden lock on dev-deps that are
    *also* declared as peers.  That's a cosmetic difference — the package is
    installed at the requested version — so it must not trigger a reinstall.
    Regression for the TUI-in-Docker failure where 16 such mismatches caused
    `Installing TUI dependencies…` → EACCES on every launch.
    """
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text(
        '{"packages":{'
        '"node_modules/foo":{"version":"1.0.0","dev":true,"peer":true,"resolved":"https://x/foo.tgz"}'
        '}}'
    )
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        '{"packages":{'
        '"node_modules/foo":{"version":"1.0.0","dev":true,"resolved":"https://x/foo.tgz"}'
        '}}'
    )
    assert main_mod._tui_need_npm_install(tmp_path) is False


def test_install_when_version_differs_even_with_peer_drop(tmp_path: Path, main_mod) -> None:
    """The peer-drop tolerance must not mask a real version skew."""
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"2.0.0","dev":true,"peer":true}}}'
    )
    (tmp_path / "node_modules" / ".package-lock.json").write_text(
        '{"packages":{"node_modules/foo":{"version":"1.0.0","dev":true}}}'
    )
    assert main_mod._tui_need_npm_install(tmp_path) is True


def test_no_install_when_lock_older_than_marker(tmp_path: Path, main_mod) -> None:
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text("{}")
    (tmp_path / "node_modules" / ".package-lock.json").write_text("{}")
    os.utime(tmp_path / "package-lock.json", (100, 100))
    os.utime(tmp_path / "node_modules" / ".package-lock.json", (200, 200))
    assert main_mod._tui_need_npm_install(tmp_path) is False


def test_need_install_when_marker_missing(tmp_path: Path, main_mod) -> None:
    _touch_ink(tmp_path)
    (tmp_path / "package-lock.json").write_text("{}")
    assert main_mod._tui_need_npm_install(tmp_path) is True


def test_no_install_without_lockfile_when_ink_present(tmp_path: Path, main_mod) -> None:
    _touch_ink(tmp_path)
    assert main_mod._tui_need_npm_install(tmp_path) is False


def test_build_needed_when_local_ink_bundle_missing(tmp_path: Path, main_mod) -> None:
    _touch_tui_entry(tmp_path)
    _touch_ink(tmp_path)

    assert main_mod._tui_need_npm_install(tmp_path) is False
    assert main_mod._tui_build_needed(tmp_path) is True


def test_build_not_needed_when_entry_and_ink_bundle_present(tmp_path: Path, main_mod) -> None:
    _touch_tui_entry(tmp_path)
    _touch_ink(tmp_path)
    _touch_ink_bundle(tmp_path)

    assert main_mod._tui_build_needed(tmp_path) is False


def test_tui_build_script_uses_cross_platform_executable_helper() -> None:
    root = Path(__file__).resolve().parents[2] / "ui-tui"
    package = json.loads((root / "package.json").read_text(encoding="utf-8"))

    build = package["scripts"]["build"]

    assert "chmod +x" not in build
    assert "node scripts/mark-executable.mjs dist/entry.js" in build
    assert (root / "scripts" / "mark-executable.mjs").is_file()


def test_make_tui_argv_uses_cmd_shim_for_tsx_on_windows(tmp_path: Path, main_mod, monkeypatch) -> None:
    _touch_ink(tmp_path)
    _touch_ink_bundle(tmp_path)
    bin_dir = tmp_path / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "tsx").write_text("#!/bin/sh\n")
    (bin_dir / "tsx.cmd").write_text("@echo off\r\n")

    monkeypatch.setattr(main_mod, "_ensure_tui_node", lambda: None)
    monkeypatch.setattr(main_mod.shutil, "which", lambda name: f"C:\\nodejs\\{name}.cmd")
    monkeypatch.setattr(main_mod.os, "name", "nt")

    argv, cwd = main_mod._make_tui_argv(tmp_path, tui_dev=True)

    assert cwd == tmp_path
    assert argv == [str(bin_dir / "tsx.cmd"), "src/entry.tsx"]
