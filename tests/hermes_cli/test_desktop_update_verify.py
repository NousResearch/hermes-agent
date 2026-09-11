"""Receipt validation uses packaged output, not just a source stamp."""
import json
import struct

import pytest

from hermes_cli import desktop_update_verify as verify
from hermes_cli.main_desktop import _write_desktop_build_stamp


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    desktop = tmp_path / 'apps/desktop'
    resources = desktop / 'release/fixture/resources'
    dist = resources / 'app.asar.unpacked/dist'
    (dist / 'assets').mkdir(parents=True)
    (dist / 'index.html').write_text('<script type="module" src="./assets/index.js"></script>', encoding='utf-8')
    (dist / 'assets/index.js').write_text('export {};', encoding='utf-8')
    entry = b'import "electron";'
    (dist / 'electron-main.mjs').write_bytes(entry)
    package = json.dumps({'main': 'dist/electron-main.mjs'}).encode()
    header = json.dumps({'files': {'package.json': {'size': len(package), 'offset': '0'}, 'dist': {'files': {'electron-main.mjs': {'size': len(entry), 'unpacked': True}}}}}).encode()
    padded = header + b'\0' * (-len(header) % 4)
    archive = resources / 'app.asar'
    archive.write_bytes(struct.pack('<4I', 4, 8 + len(padded), 4 + len(padded), len(header)) + padded + package)
    (tmp_path / '.gitignore').write_text('apps/desktop/release/\n', encoding='utf-8')
    monkeypatch.setattr(verify, '_desktop_packaged_executable', lambda _: resources.parent / 'Hermes.exe')
    monkeypatch.setattr(verify, '_desktop_exe_integrity_error', lambda _: None)
    # Host-independent artifact contract; executable lookup itself is covered natively.
    from hermes_cli import main_desktop
    monkeypatch.setattr(main_desktop, '_desktop_packaged_executable', lambda _: resources.parent / 'Hermes.exe')
    _write_desktop_build_stamp(tmp_path, source_mode=False)
    return tmp_path, archive, dist


def test_readable_packaged_entry_passes(bundle):
    root, _, _ = bundle
    verify.verify_windows_desktop_update(root)


@pytest.mark.parametrize('damage', ['archive', 'truncated', 'entry', 'empty-index', 'unreadable-index', 'no-module'])
def test_current_stamp_does_not_hide_damaged_output(bundle, damage):
    root, archive, dist = bundle
    if damage == 'archive':
        archive.write_bytes(b'not an asar')
    elif damage == 'truncated':
        archive.write_bytes(archive.read_bytes()[:-4])
    elif damage == 'entry':
        (dist / 'electron-main.mjs').write_bytes(b'')
    else:
        (dist / 'index.html').write_bytes({'empty-index': b'', 'unreadable-index': b'\xff', 'no-module': b'<html></html>'}[damage])
    with pytest.raises((RuntimeError, OSError, ValueError)):
        verify.verify_windows_desktop_update(root)


def test_default_root_is_the_imported_checkout_not_cwd(tmp_path, monkeypatch):
    # The Windows hand-off is spawned from HERMES_HOME; the receipt must describe the checkout anyway.
    monkeypatch.chdir(tmp_path)
    seen = {}
    monkeypatch.setattr(verify, '_desktop_packaged_executable', lambda desktop: seen.setdefault('desktop', desktop) and None)
    with pytest.raises(RuntimeError, match='executable is missing'):
        verify.verify_windows_desktop_update()
    assert seen['desktop'] == verify.checkout_root() / 'apps' / 'desktop'
    assert (verify.checkout_root() / 'hermes_cli' / 'desktop_update_verify.py').is_file()
    assert verify.checkout_root() != tmp_path


def _app_only_under(root):
    """A packaged-app lookup that finds an app under *root* and nowhere else."""
    from pathlib import Path

    desktop = (root / 'apps' / 'desktop').resolve()

    def lookup(candidate):
        return desktop / 'release/fixture/Hermes.exe' if Path(candidate).resolve() == desktop else None

    return lookup


def test_caller_root_without_a_packaged_app_falls_back_to_the_checkout(bundle, tmp_path, monkeypatch):
    # A hand-off script read before the pull still passes HERMES_HOME, which never
    # holds a packaged app. The healthy checkout it just wrote must be verified instead.
    checkout, _, _ = bundle
    monkeypatch.setattr(verify, '_desktop_packaged_executable', _app_only_under(checkout))
    monkeypatch.setattr(verify, 'checkout_root', lambda: checkout)
    verify.verify_windows_desktop_update(tmp_path / 'hermes_home')


def test_no_packaged_app_under_either_root_still_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, '_desktop_packaged_executable', lambda _: None)
    monkeypatch.setattr(verify, 'checkout_root', lambda: tmp_path / 'checkout')
    with pytest.raises(RuntimeError, match='executable is missing'):
        verify.verify_windows_desktop_update(tmp_path / 'hermes_home')


def test_caller_root_that_has_a_packaged_app_is_verified_as_given(bundle, tmp_path, monkeypatch):
    # Fail-closed: the fallback is for a root with nothing to read, never a way to
    # launder a damaged build past the receipt by switching to a healthy tree.
    checkout, _, _ = bundle
    supplied = tmp_path / 'supplied'
    resources = supplied / 'apps/desktop/release/fixture/resources'
    resources.mkdir(parents=True)
    (resources / 'app.asar').write_bytes(b'not an asar')
    monkeypatch.setattr(verify, '_desktop_packaged_executable', _app_only_under(supplied))
    monkeypatch.setattr(verify, 'checkout_root', lambda: checkout)
    with pytest.raises(RuntimeError, match='archive or main entry is invalid'):
        verify.verify_windows_desktop_update(supplied)
