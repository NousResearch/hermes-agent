"""Cutover identity representation only; no name-lifetime or closing scenarios."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_input_custody as custody
from gateway import session_ingress_media as media
from hermes_state_runtime import RuntimeStoreError, admit_session_input
from tests.gateway.test_input_custody_migration import candidate
from tests.gateway.test_native_media_budget import _authority


def seed(home, owner):
    _, path = candidate(home, owner, 'old.txt', b'ordinary file')
    admit_session_input(owner.db, epoch=owner.epoch, principal_id='human', session_id='s',
        request_id='hosted:old', payload={'text': 'opaque'})
    return path


@pytest.mark.parametrize('wide', [False, True], ids=['real-stat', 'unsigned-wide-representation'])
def test_inventory_persists_observed_identity_as_text_and_compares_it(tmp_path, monkeypatch, wide):
    db, owner = _authority(tmp_path, monkeypatch)
    with db:
        path = seed(tmp_path, owner)
        observed = path.stat(follow_symlinks=False)
        expected = ((1 << 64) - 1, (1 << 96) + 17) if wide else (observed.st_dev, observed.st_ino)
        with monkeypatch.context() as capture:
            if wide:
                original = Path.stat
                def metadata(self, *args, **kwargs):
                    if self == path and kwargs.get('follow_symlinks') is False:
                        return SimpleNamespace(st_mode=observed.st_mode, st_dev=expected[0], st_ino=expected[1])
                    return original(self, *args, **kwargs)
                capture.setattr(Path, 'stat', metadata)
            custody.initialize_input_custody(db)
        with db._read_ctx() as conn:
            row = conn.execute('SELECT path,digest,device,inode,typeof(device),typeof(inode) FROM gateway_legacy_input_paths').fetchone()
            assert row[:2] == (str(path.relative_to(media._media_root())), path.parent.name)
            assert row[2:4] == tuple(str(value) for value in expected)
            assert row[4:] == ('text', 'text')
            assert custody._stored_identity(*row[2:4]) == expected
            assert custody._ready(conn, media._media_root())
        custody.initialize_input_custody(db)
        if not wide:
            alias = path.with_name('ordinary-hardlink.txt')
            os.link(path, alias)
            calls = []
            identity = media._file_identity
            def observed_identity(value):
                calls.append(value)
                return identity(value)
            monkeypatch.setattr(media, '_file_identity', observed_identity)
            ref = {'path': str(alias), 'sha256': path.parent.name, 'size': alias.stat().st_size}
            assert db._execute_write(lambda conn: custody.custody_holds(conn, db.db_path, ref))
            assert calls == [alias]  # Saved identity matched; no old-name re-stat was needed.


@pytest.mark.parametrize('invalid', ['malformed', 'zero-inode', 'old-marker', 'old-schema', 'integer-columns'])
def test_incompatible_or_malformed_inventory_is_not_ready(tmp_path, monkeypatch, invalid):
    db, owner = _authority(tmp_path, monkeypatch)
    with db:
        seed(tmp_path, owner)
        if invalid in {'old-schema', 'integer-columns'}:
            columns = ', device INTEGER NOT NULL, inode INTEGER NOT NULL' if invalid == 'integer-columns' else ''
            db._execute_write(lambda conn: conn.execute(
                'CREATE TABLE gateway_legacy_input_paths(path TEXT PRIMARY KEY, digest TEXT NOT NULL' + columns + ')'))
        else:
            custody.initialize_input_custody(db)
            if invalid == 'old-marker':
                db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                    (json.dumps({'version': 2, 'root': str(media._media_root())}), custody._READY)))
            else:
                value = 'not-decimal' if invalid == 'malformed' else '0'
                db._execute_write(lambda conn: conn.execute('UPDATE gateway_legacy_input_paths SET inode=?', (value,)))
        with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
            custody.initialize_input_custody(db)
        with db._read_ctx() as conn:
            assert not custody._ready(conn, media._media_root())
            if invalid in {'old-schema', 'integer-columns'}:
                assert conn.execute('SELECT 1 FROM state_meta WHERE key=?', (custody._READY,)).fetchone() is None
