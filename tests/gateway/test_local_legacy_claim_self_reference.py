"""A self-referencing legacy claim over a gateway-minted id is malformed, not lineage.

The only supported writer of ``legacy_session_id`` is ``adopt_legacy_session``: it adopts an
unrouted pre-gateway row whose id it keeps, and ``commit_local_session`` binds that row through
``bind_legacy_target``, which refuses any row carrying a route. A ``local-`` id comes from
``local_identity`` and its row is always routed, so a receipt pairing a ``local-`` id with a
self-referencing legacy claim cannot have been written by the store. It is what a hand rewrite of
``state_meta`` produces when it moves a store between profiles and rewrites ``profile_id``: the
ids no longer derive from the new home, so the rewrite papers over the identity check with a claim
whose binding row it never writes (t_988ef254). That claim must not excuse the check.

A store genuinely moved between profiles records the home it was minted under instead
(``adopted_profile_id``), so the identity check keeps working without a false claim.
"""
import json

import pytest

from hermes_state import SessionDB
from hermes_state_keys import local_identity
from hermes_state_local import POLICY_PREFIX
from hermes_state_local_lineage import validate_local_lineage
from hermes_state_local_migration import LEGACY_PREFIX
from hermes_state_runtime import RuntimeStoreError

CHIEF = 'C:/Users/owner/AppData/Local/hermes/profiles/chief-oizy'
NICO = 'C:/Users/owner/AppData/Local/hermes/profiles/nico'
PRINCIPAL = 'sid:owner'
REQUEST = 'create-1'


def _db(tmp_path):
    return SessionDB(db_path=tmp_path / 'state.db')


def _receipt(session_id, profile_id):
    """A receipt in the shape ``commit_local_session`` stores for a plain create."""
    return {
        'profile_id': profile_id,
        'principal_id': PRINCIPAL,
        'request_id': REQUEST,
        'session_id': session_id,
        'route': 'agent:nico:local:dm:' + session_id,
        'entry': {'session_id': session_id, 'created_at': '2026-10-27T08:00:25.474769'},
        'policy': {'source': 'gui', 'model': 'frozen'},
    }


def _bind(db, session_id, profile_id):
    """The binding row ``bind_legacy_target`` writes for an adopted legacy row."""
    value = {'session_id': session_id, 'profile_id': profile_id,
             'principal_id': PRINCIPAL, 'legacy_session_id': session_id}

    def write(conn):
        conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)',
                     (LEGACY_PREFIX + session_id, json.dumps(value)))
    db._execute_write(write)


def test_a_legacy_claim_over_a_gateway_id_is_refused_even_when_bound(tmp_path):
    """The hole: the claim alone excused a ``local-`` id whose identity does not derive."""
    db = _db(tmp_path)
    sid = local_identity(CHIEF, PRINCIPAL, REQUEST)
    assert sid.startswith('local-')
    receipt = _receipt(sid, profile_id=NICO)
    receipt['legacy_session_id'] = sid
    _bind(db, sid, NICO)
    with db._read_ctx() as conn:
        with pytest.raises(RuntimeStoreError):
            validate_local_lineage(conn, receipt)


def test_a_store_moved_between_profiles_validates_under_the_home_it_was_minted_in(tmp_path):
    """The repair: record the minting home instead of claiming legacy lineage."""
    db = _db(tmp_path)
    sid = local_identity(CHIEF, PRINCIPAL, REQUEST)
    receipt = _receipt(sid, profile_id=NICO)
    with db._read_ctx() as conn:
        with pytest.raises(RuntimeStoreError):
            validate_local_lineage(conn, receipt)
        receipt['adopted_profile_id'] = CHIEF
        assert validate_local_lineage(conn, receipt) == sid
        del receipt['adopted_profile_id']
        receipt['adopted_profile_id'] = NICO
        with pytest.raises(RuntimeStoreError):
            validate_local_lineage(conn, receipt)


def test_an_adopted_home_does_not_excuse_a_second_rewrite(tmp_path):
    """``adopted_profile_id`` names where the id derives from; it is checked, not trusted."""
    db = _db(tmp_path)
    sid = local_identity(CHIEF, PRINCIPAL, REQUEST)
    receipt = _receipt(sid, profile_id=NICO)
    receipt['adopted_profile_id'] = CHIEF
    receipt['request_id'] = 'rewritten'
    with db._read_ctx() as conn:
        with pytest.raises(RuntimeStoreError):
            validate_local_lineage(conn, receipt)


def test_a_real_legacy_import_still_validates(tmp_path):
    """An adopted pre-gateway row keeps its timestamp id and its binding, and is accepted."""
    db = _db(tmp_path)
    sid = '20260923_223346_4134ec'
    receipt = _receipt(sid, profile_id=NICO)
    receipt['legacy_session_id'] = sid
    receipt['lineage'] = [sid]
    _bind(db, sid, NICO)
    with db._read_ctx() as conn:
        assert validate_local_lineage(conn, receipt) == sid


def test_a_plain_create_still_validates(tmp_path):
    db = _db(tmp_path)
    sid = local_identity(NICO, PRINCIPAL, REQUEST)
    with db._read_ctx() as conn:
        assert validate_local_lineage(conn, _receipt(sid, profile_id=NICO)) == sid
