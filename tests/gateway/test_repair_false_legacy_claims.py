"""``scripts/repair_false_legacy_claims.py`` replaces a false claim with the home it was minted under."""
import importlib.util
import json
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


def _false_claim_store(tmp_path):
    """A store moved from *old* to *new*: its receipt's id derives from *old*, and the move was
    papered over with a self-referencing legacy claim plus the binding row that claim implies."""
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
    binding = {'session_id': sid, 'profile_id': new, 'principal_id': PRINCIPAL, 'legacy_session_id': sid}

    def write(conn):
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (POLICY_PREFIX + sid, json.dumps(receipt)))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (LEGACY_PREFIX + sid, json.dumps(binding)))
    db._execute_write(write)
    db.close()
    return path, sid, old, new


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
