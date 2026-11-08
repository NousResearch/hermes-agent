"""``scripts/repair_false_legacy_claims.py`` replaces a false claim with the home it was minted under."""
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

from hermes_state import SessionDB
from hermes_state_keys import local_identity
from hermes_state_local import POLICY_PREFIX
from hermes_state_local_lineage import validate_local_lineage
from hermes_state_local_migration import LEGACY_PREFIX

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'repair_false_legacy_claims.py'
PRINCIPAL = 'sid:owner'
REQUEST = 'uuid-create'


def _tool():
    spec = importlib.util.spec_from_file_location('repair_false_legacy_claims', SCRIPT)
    assert spec is not None and spec.loader is not None, SCRIPT
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _false_claim_store(tmp_path, *, binding='self'):
    """A store moved from *old* to *new*: its receipt's id derives from *old*, and the move was
    papered over with a self-referencing legacy claim plus the binding row that claim implies.

    ``binding`` selects what sits at ``gateway.local_legacy.v1:<sid>``: ``'self'`` (the row the
    claim implies), ``'absent'`` (the claim survived the repair but the row did not), or
    ``'foreign'`` (a row that names some other session).
    """
    old, new = str(tmp_path / 'profiles' / 'chief-oizy'), str(tmp_path / 'profiles' / 'nico')
    path = tmp_path / 'profiles' / 'nico' / 'state.db'
    path.parent.mkdir(parents=True)
    db = SessionDB(db_path=path)
    sid = local_identity(old, PRINCIPAL, REQUEST)
    receipt = {
        'profile_id': new, 'principal_id': PRINCIPAL, 'request_id': REQUEST, 'session_id': sid,
        'route': 'agent:nico:local:dm:' + sid, 'legacy_session_id': sid,
        'entry': {'session_id': sid, 'created_at': '2026-10-27T08:00:25.474769'},
        'policy': {'source': 'gui', 'model': 'frozen',
                   'config_secret_ref': 'profile-config-v1:' + json.dumps({'home': old})},
    }
    bound_to = sid if binding == 'self' else 'someone-else'
    row = {'session_id': sid, 'profile_id': new, 'principal_id': PRINCIPAL, 'legacy_session_id': bound_to}

    def write(conn):
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (POLICY_PREFIX + sid, json.dumps(receipt)))
        if binding != 'absent':
            conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (LEGACY_PREFIX + sid, json.dumps(row)))
    db._execute_write(write)
    db.close()
    return path, sid, old, new


def _receipt(path, sid):
    db = SessionDB(db_path=path)
    try:
        with db._read_ctx() as conn:
            return json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?',
                                           (POLICY_PREFIX + sid,)).fetchone()[0])
    finally:
        db.close()


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_false_claim_becomes_the_home_its_id_was_minted_under(tmp_path, capsys):
    tool = _tool()
    path, sid, old, new = _false_claim_store(tmp_path)

    assert tool.main(['--db', str(path)]) == 0
    assert 'false legacy claims: 1 (1 repairable, 0 not)' in capsys.readouterr().out

    assert tool.main(['--db', str(path), '--apply', '--backup-dir', str(tmp_path), '--gateway-stopped']) == 0
    db = SessionDB(db_path=path)
    with db._read_ctx() as conn:
        receipt = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,)).fetchone()[0])
        binding = conn.execute('SELECT 1 FROM state_meta WHERE key=?', (LEGACY_PREFIX + sid,)).fetchone()
        assert receipt['adopted_profile_id'] == old
        assert 'legacy_session_id' not in receipt
        assert receipt['profile_id'] == new, 'the serving profile is unchanged'
        assert binding is None, 'the binding row the false claim implied is gone'
        assert validate_local_lineage(conn, receipt) == sid
    db.close()


def test_apply_refuses_a_backup_dir_that_does_not_exist(tmp_path, capsys):
    """R1: a missing --backup-dir is a usage error, and the store is not touched."""
    tool = _tool()
    path, sid, _old, _new = _false_claim_store(tmp_path)
    before = _sha256(path)

    assert tool.main(['--db', str(path), '--apply', '--gateway-stopped',
                      '--backup-dir', str(tmp_path / 'does-not-exist')]) == 2
    assert 'existing directory' in capsys.readouterr().err
    assert _sha256(path) == before, 'the store is byte-identical'
    assert _receipt(path, sid)['legacy_session_id'] == sid


def test_apply_refuses_a_backup_dir_that_is_a_file(tmp_path, capsys):
    tool = _tool()
    path, sid, _old, _new = _false_claim_store(tmp_path)
    blocker = tmp_path / 'not-a-directory'
    blocker.write_text('file, not dir')
    before = _sha256(path)

    assert tool.main(['--db', str(path), '--apply', '--gateway-stopped',
                      '--backup-dir', str(blocker)]) == 2
    assert 'existing directory' in capsys.readouterr().err
    assert _sha256(path) == before
    assert _receipt(path, sid)['legacy_session_id'] == sid


def test_apply_writes_and_names_a_verified_pre_repair_backup(tmp_path, capsys):
    """R1: the backup the tool reports is a file it wrote and checked, not the path it was handed."""
    tool = _tool()
    path, sid, _old, _new = _false_claim_store(tmp_path)
    backups = tmp_path / 'backups'
    backups.mkdir()

    assert tool.main(['--db', str(path), '--apply', '--backup-dir', str(backups),
                      '--gateway-stopped']) == 0
    out = capsys.readouterr().out
    produced = list(backups.glob('*.db'))
    assert len(produced) == 1, produced
    assert str(produced[0]) in out and 'quick_check ok' in out
    copy = sqlite3.connect(produced[0])
    try:
        assert copy.execute('PRAGMA quick_check').fetchall() == [('ok',)]
        raw = copy.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,)).fetchone()[0]
        assert json.loads(raw)['legacy_session_id'] == sid, 'the copy is the pre-image'
    finally:
        copy.close()


def test_every_rewrite_is_counted_when_the_binding_row_is_absent(tmp_path, capsys):
    """R2: a claim with no binding row is still a rewrite, and the printed total proves it."""
    tool = _tool()
    path, sid, old, _new = _false_claim_store(tmp_path, binding='absent')
    backups = tmp_path / 'backups'
    backups.mkdir()

    assert tool.main(['--db', str(path)]) == 0
    assert 'false legacy claims: 1 (1 repairable, 0 not)' in capsys.readouterr().out
    assert tool.main(['--db', str(path), '--apply', '--backup-dir', str(backups),
                      '--gateway-stopped']) == 0
    out = capsys.readouterr().out
    assert 'repaired 1 receipt(s) of 1 surveyed' in out
    assert 'bindings: 0 removed, 1 absent' in out
    receipt = _receipt(path, sid)
    assert receipt['adopted_profile_id'] == old
    assert 'legacy_session_id' not in receipt


def test_a_binding_row_that_names_another_session_is_reported_not_rewritten(tmp_path, capsys):
    """R2: a binding row the tool cannot tie to the claim is reported, and nothing is written."""
    tool = _tool()
    path, sid, _old, _new = _false_claim_store(tmp_path, binding='foreign')
    before = _sha256(path)

    assert tool.main(['--db', str(path)]) == 1
    out = capsys.readouterr().out
    assert 'false legacy claims: 1 (0 repairable, 1 not)' in out
    assert 'UNREPAIRABLE' in out and 'does not name this session' in out

    backups = tmp_path / 'backups'
    backups.mkdir()
    assert tool.main(['--db', str(path), '--apply', '--backup-dir', str(backups),
                      '--gateway-stopped']) == 1
    assert 'partial repair' in capsys.readouterr().err
    assert _sha256(path) == before, 'the store is byte-identical'
    assert list(backups.iterdir()) == [], 'no backup for a run that repaired nothing'
    assert _receipt(path, sid)['legacy_session_id'] == sid


def test_apply_refuses_a_store_it_cannot_prove(tmp_path, capsys):
    tool = _tool()
    path, sid, _old, _new = _false_claim_store(tmp_path)
    db = SessionDB(db_path=path)
    with db._read_ctx() as conn:
        receipt = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?', (POLICY_PREFIX + sid,)).fetchone()[0])
    receipt['request_id'] = 'rewritten-again'
    db._write_sql('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(receipt), POLICY_PREFIX + sid))
    db.close()
    assert tool.main(['--db', str(path)]) == 1
    assert 'UNREPAIRABLE' in capsys.readouterr().out


def test_apply_requires_an_operator_confirmation(tmp_path):
    tool = _tool()
    path, _, _, _ = _false_claim_store(tmp_path)
    assert tool.main(['--db', str(path), '--apply']) == 2
