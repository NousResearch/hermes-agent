"""_tui_need_npm_install: auto npm when node_modules is behind the lockfile."""

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


def test_no_install_prebuilt_bundle_mode(tmp_path: Path, main_mod) -> None:
    """dist/entry.js present and no package-lock.json → prebuilt bundle, skip npm install."""
    _touch_tui_entry(tmp_path)
    assert main_mod._tui_need_npm_install(tmp_path) is False


def test_build_needed_when_source_newer_than_entry(tmp_path: Path, main_mod) -> None:
    _touch_tui_entry(tmp_path)
    _touch_ink(tmp_path)
    src = tmp_path / "src" / "entry.tsx"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("console.log('newer')")
    os.utime(src, (200, 200))
    os.utime(tmp_path / "dist" / "entry.js", (100, 100))

    assert main_mod._tui_need_npm_install(tmp_path) is False
    assert main_mod._tui_build_needed(tmp_path) is True


def test_build_not_needed_when_entry_exists_and_sources_unchanged(tmp_path: Path, main_mod) -> None:
    _touch_tui_entry(tmp_path)
    _touch_ink(tmp_path)

    assert main_mod._tui_need_npm_install(tmp_path) is False
    assert main_mod._tui_build_needed(tmp_path) is False


def test_build_not_needed_when_entry_and_ink_bundle_present(tmp_path: Path, main_mod) -> None:
    _touch_tui_entry(tmp_path)
    _touch_ink(tmp_path)
    _touch_ink_bundle(tmp_path)

    assert main_mod._tui_build_needed(tmp_path) is False
