"""Repair diagnostics retain unsafe inventory and name held metadata rows (#149132)."""
import re
import sqlite3
from pathlib import Path

import pytest

from hermes_cli.sessions_repair_profiles import RepairPlan, Store, _Session
from hermes_constants import mark_named_profile_deleted
from hermes_state import SessionDB
from hermes_state_lineage import SessionLineageInventoryError


@pytest.mark.parametrize('apply_mode', [False, True])
@pytest.mark.parametrize('damaged', [b'', b'not a sqlite database'])
def test_unreadable_inventory_names_its_store_and_stays_fail_closed(tmp_path, monkeypatch, apply_mode, damaged):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    ghost = tmp_path / 'profiles/ghost'
    ghost.mkdir(parents=True)
    local = SessionDB(tmp_path / 'state.db')
    try:
        local.create_session('kept', 'cli')
        local.append_message('kept', 'user', 'Preserved history')
        broken = ghost / 'state.db'
        broken.write_bytes(damaged)
        mark_named_profile_deleted(ghost)
        before = {table: [tuple(row) for row in local._conn.execute(f'SELECT * FROM {table} ORDER BY 1')]
                  for table in ('sessions', 'messages')}
        with pytest.raises(SessionLineageInventoryError, match=re.escape(str(broken))):
            local.append_message('kept', 'user', 'Must not write')
        session = _Session(read_only=not apply_mode)
        try:
            with pytest.raises(sqlite3.DatabaseError, match=re.escape(str(broken))):
                RepairPlan([Store('default', tmp_path, routing=True)], root=tmp_path).scan(session)
        finally:
            session.close()
        assert broken.read_bytes() == damaged
        assert {table: [tuple(row) for row in local._conn.execute(f'SELECT * FROM {table} ORDER BY 1')]
                for table in ('sessions', 'messages')} == before
    finally:
        local.close()


@pytest.mark.parametrize('apply_mode', [False, True])
@pytest.mark.parametrize('shape', ['mislabelled', 'crossed_parent', 'both'])
def test_every_held_metadata_defect_is_reported_without_becoming_repairable(tmp_path, monkeypatch, apply_mode, shape):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    profile = tmp_path / 'profiles/science'
    profile.mkdir(parents=True)
    local, named = SessionDB(tmp_path / 'state.db'), SessionDB(profile / 'state.db')
    try:
        local._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name,session_key) VALUES ('dup','cli',1,'default','agent:main:local:dm:dup')")
        parent_profile = 'science' if shape == 'mislabelled' else 'main'
        child_profile = 'science' if shape == 'crossed_parent' else 'wrong'
        named._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name,session_key) VALUES ('dup','cli',1,?,?)",
                             (parent_profile, f'agent:{parent_profile}:local:dm:dup'))
        named._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name,session_key,parent_session_id) "
                             "VALUES ('child','cli',2,?,'agent:science:local:dm:child','dup')", (child_profile,))
        named._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name,session_key) "
                             "VALUES ('unrelated','cli',3,'wrong','agent:science:local:dm:unrelated')")
        local._conn.commit()
        named._conn.commit()
        before_local = [tuple(row) for row in local._conn.execute('SELECT * FROM sessions ORDER BY id')]
        before_held = [tuple(row) for row in named._conn.execute("SELECT * FROM sessions WHERE id!='unrelated' ORDER BY id")]
        session = _Session(read_only=not apply_mode)
        try:
            plan = RepairPlan([Store('default', tmp_path, routing=True), Store('science', profile)], root=tmp_path).scan(session)
            expected = {'mislabelled', 'crossed_parent'} if shape == 'both' else {shape}
            held = [finding for finding in plan.findings if finding.subject == 'child']
            assert {finding.kind for finding in held} == expected
            assert all(not finding.repairable and finding.action is None and 'dual_home' in finding.reason for finding in held)
            assert {finding.subject for finding in plan.repairable} == {'unrelated'}
            if apply_mode:
                result = plan.apply(session, snapshot=lambda store: None)
                assert not result['failures']
                assert named.get_session('unrelated')['profile_name'] == 'science'
            else:
                assert named.get_session('unrelated')['profile_name'] == 'wrong'
        finally:
            session.close()
        assert [tuple(row) for row in local._conn.execute('SELECT * FROM sessions ORDER BY id')] == before_local
        assert [tuple(row) for row in named._conn.execute("SELECT * FROM sessions WHERE id!='unrelated' ORDER BY id")] == before_held
    finally:
        local.close()
        named.close()
