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


def test_no_install_prebuilt_bundle_mode(tmp_path: Path, main_mod) -> None:
    """dist/entry.js present and no package-lock.json → prebuilt bundle, skip npm install."""
    _touch_tui_entry(tmp_path)
    assert main_mod._tui_need_npm_install(tmp_path) is False
