"""Captured content identity must survive retention and precede restore writes."""

import json
import sqlite3

from hermes_cli import backup


def test_same_size_changed_snapshot_cannot_displace_recovery_or_restore(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text('model: original\n')
    with sqlite3.connect(home / 'state.db') as conn:
        conn.execute('CREATE TABLE records(value TEXT)')
        conn.execute("INSERT INTO records VALUES ('original')")
    old = backup.create_quick_snapshot(hermes_home=home, keep=3)
    new = backup.create_quick_snapshot(hermes_home=home, keep=3)
    root = home / 'state-snapshots'
    db = root / new / 'state.db'
    size = db.stat().st_size
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE records SET value='replaced'")
    assert db.stat().st_size == size
    assert backup.verify_sqlite_integrity(db)['valid']
    backup.prune_quick_snapshots(keep=1, hermes_home=home)
    assert (root / old).exists(), 'altered payload must not replace genuine recovery copy'
    (home / 'config.yaml').write_text('model: current\n')
    assert not backup.restore_quick_snapshot(new, hermes_home=home)
    assert (home / 'config.yaml').read_text() == 'model: current\n'
    assert backup.restore_quick_snapshot(old, hermes_home=home)
    assert (home / 'config.yaml').read_text() == 'model: original\n'
    with sqlite3.connect(home / 'state.db') as conn:
        assert conn.execute('SELECT value FROM records').fetchone() == ('original',)


def test_digest_manifest_requires_complete_supported_integrity_metadata(tmp_path):
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text('model: captured\n')
    snap = backup.create_quick_snapshot(hermes_home=home)
    path = home / 'state-snapshots' / snap / 'manifest.json'
    meta = json.loads(path.read_text())
    assert meta['version'] == 2
    (home / 'config.yaml').write_text('model: current\n')
    meta['sha256'] = {}
    path.write_text(json.dumps(meta))
    assert not backup.restore_quick_snapshot(snap, hermes_home=home)
    assert (home / 'config.yaml').read_text() == 'model: current\n'
    meta.pop('version')
    meta.pop('sha256')
    path.write_text(json.dumps(meta))
    assert backup.restore_quick_snapshot(snap, hermes_home=home), 'legacy snapshots remain restorable'
