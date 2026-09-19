"""Receipt validation uses packaged output, not just a source stamp."""
import json
import subprocess
import sys
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


def test_failed_verification_rebuilds_once_then_reverifies(tmp_path, monkeypatch):
    checks = []
    rebuilds = []

    def check(project_root=None):
        checks.append(project_root)
        if len(checks) == 1:
            raise RuntimeError('The updated Desktop executable is missing')

    def run(command, **kwargs):
        rebuilds.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(verify, 'verify_windows_desktop_update', check)

    verify.verify_or_rebuild_windows_desktop_update(tmp_path, run=run)

    assert checks == [tmp_path, tmp_path]
    assert len(rebuilds) == 1
    command, kwargs = rebuilds[0]
    assert command[0] == sys.executable
    assert command[1:] == ['-m', 'hermes_cli.main', 'desktop', '--force-build', '--build-only']
    assert kwargs == {'cwd': tmp_path, 'check': False}


def test_failed_rebuild_preserves_the_verification_failure(tmp_path, monkeypatch):
    def fails_to_start(command, **kwargs):
        raise PermissionError('blocked')

    def exits_nonzero(command, **kwargs):
        return subprocess.CompletedProcess(command, 9)

    for run, match in [(fails_to_start, 'could not start'), (exits_nonzero, 'exited 9')]:
        initial_error = RuntimeError('The updated Desktop executable is missing')
        checks = []

        def check(project_root=None):
            checks.append(project_root)
            raise initial_error

        monkeypatch.setattr(verify, 'verify_windows_desktop_update', check)

        with pytest.raises(RuntimeError, match=match) as exc_info:
            verify.verify_or_rebuild_windows_desktop_update(tmp_path, run=run)

        assert checks == [tmp_path]
        assert exc_info.value.__cause__ is initial_error
