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


def _exe_only_under(root):
    desktop = (root / 'apps' / 'desktop').resolve()
    exe = desktop / 'release' / 'fixture' / 'Hermes.exe'

    def packaged_exe(desktop_path):
        if desktop_path.resolve() == desktop:
            return exe
        return None

    return packaged_exe


def test_cwd_derived_root_falls_back_to_imported_checkout(bundle, tmp_path, monkeypatch):
    # Stale windows.ps1 still passes Path.cwd()/HERMES_HOME; that root has no
    # packaged exe. The imported checkout is healthy and must be consulted.
    healthy, _, _ = bundle
    hermes_home = tmp_path / 'hermes_home'
    hermes_home.mkdir()
    monkeypatch.setattr(verify, '_desktop_packaged_executable', _exe_only_under(healthy))
    monkeypatch.setattr(verify, 'checkout_root', lambda: healthy)
    verify.verify_windows_desktop_update(hermes_home)


def test_missing_exe_on_both_roots_still_fails(tmp_path, monkeypatch):
    empty = tmp_path / 'empty_checkout'
    empty.mkdir()
    wrong = tmp_path / 'hermes_home'
    wrong.mkdir()
    monkeypatch.setattr(verify, 'checkout_root', lambda: empty)
    with pytest.raises(RuntimeError, match='executable is missing'):
        verify.verify_windows_desktop_update(wrong)


def test_damaged_supplied_root_with_exe_does_not_fall_back(bundle, tmp_path, monkeypatch):
    # Fail-closed: a caller root that HAS an exe must be verified as-is.
    # Integrity/ASAR/stamp errors must not silently switch to checkout_root().
    healthy, _, _ = bundle
    supplied = tmp_path / 'supplied_damaged'
    resources = supplied / 'apps' / 'desktop' / 'release' / 'fixture' / 'resources'
    resources.mkdir(parents=True)
    (resources / 'app.asar').write_bytes(b'not an asar')
    monkeypatch.setattr(verify, '_desktop_packaged_executable', _exe_only_under(supplied))
    monkeypatch.setattr(verify, 'checkout_root', lambda: healthy)
    with pytest.raises(RuntimeError, match='archive or main entry is invalid'):
        verify.verify_windows_desktop_update(supplied)


def test_caller_missing_exe_surfaces_checkout_verification_error(bundle, tmp_path, monkeypatch):
    damaged, archive, _ = bundle
    archive.write_bytes(b'not an asar')
    hermes_home = tmp_path / 'hermes_home'
    hermes_home.mkdir()
    monkeypatch.setattr(verify, '_desktop_packaged_executable', _exe_only_under(damaged))
    monkeypatch.setattr(verify, 'checkout_root', lambda: damaged)
    with pytest.raises(RuntimeError, match='archive or main entry is invalid'):
        verify.verify_windows_desktop_update(hermes_home)
