"""session_cron.py scopes correctly when state.db is a symlink outside profile home.

Regression for #145075: bind_owner / unbind_owner / execute derived the profile home via
Path(authority.db.db_path).resolve().parent, which follows symlinks and returns the
physical store location instead of the profile home. This caused cron runs to scope into
the wrong directory (empty skills/, no .env), producing ProviderNotConfiguredError.
"""
import asyncio
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def symlinked_store_authority(tmp_path, monkeypatch):
    """An authority whose state.db is a symlink to a store outside the profile home."""
    from gateway.session_authority import SessionAuthority
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch
    
    # Profile home
    profile_home = tmp_path / "profile_home"
    profile_home.mkdir()
    
    # Store lives elsewhere
    store_dir = tmp_path / "external_store"
    store_dir.mkdir()
    physical_db = store_dir / "state.db"
    
    # Symlink profile_home/state.db -> external_store/state.db
    symlink_db = profile_home / "state.db"
    symlink_db.symlink_to(physical_db)
    
    monkeypatch.setenv('HERMES_HOME', str(profile_home))
    
    # Open the database through the symlink
    db = SessionDB(symlink_db)
    
    runner = SimpleNamespace(_draining=False)
    authority = SessionAuthority(
        runner, 
        profile_id=str(profile_home),  # The profile_id IS the home, as set in run_runtime.py
        instance_id='test',
        db=db,
        epoch=begin_runtime_epoch(db, instance_id='test')
    )
    
    return authority, profile_home, store_dir


def test_bind_owner_uses_profile_id_not_resolved_db_path(symlinked_store_authority):
    """bind_owner must key the owner by authority.profile_id, not db.db_path.resolve().parent.
    
    RED on unfixed code: the key is store_dir (symlink target), not profile_home.
    GREEN after fix: the key is profile_home.
    """
    from gateway import session_cron
    
    authority, profile_home, store_dir = symlinked_store_authority
    
    # The db path resolves through the symlink
    assert Path(authority.db.db_path).resolve().parent == store_dir
    
    # But the authority's profile_id is the profile home
    assert Path(authority.profile_id) == profile_home
    
    # Bind the owner (needs an event loop)
    async def bind_test():
        session_cron.bind_owner(authority)
        
        try:
            # The owner must be keyed by profile_home (authority.profile_id), not store_dir
            bound = session_cron.owner_for_home(profile_home)
            assert bound is not None, "Owner not found by profile_home — bind_owner used db.db_path.resolve().parent instead of profile_id"
            assert bound[0] is authority
            
            # The WRONG key (store_dir) must NOT find the owner
            wrong_bound = session_cron.owner_for_home(store_dir)
            assert wrong_bound is None, "Owner found by symlink target directory — bind_owner incorrectly used db.db_path.resolve().parent"
        finally:
            session_cron.unbind_owner(authority)
    
    asyncio.run(bind_test())


def test_unbind_owner_uses_profile_id_not_resolved_db_path(symlinked_store_authority):
    """unbind_owner must use the same key as bind_owner: authority.profile_id.
    
    RED on unfixed code: unbind tries store_dir, leaves profile_home dangling.
    GREEN after fix: unbind removes the profile_home entry.
    """
    from gateway import session_cron
    
    authority, profile_home, _store_dir = symlinked_store_authority

    async def unbind_test():
        session_cron.bind_owner(authority)
        
        # Verify it's bound
        assert session_cron.owner_for_home(profile_home) is not None
        
        # Unbind
        session_cron.unbind_owner(authority)
        
        # Must be gone
        assert session_cron.owner_for_home(profile_home) is None, "Owner still bound after unbind — unbind_owner used the wrong home key"
    
    asyncio.run(unbind_test())


@pytest.mark.asyncio
async def test_execute_scopes_to_profile_id_not_resolved_db_parent(symlinked_store_authority, monkeypatch):
    """execute() must scope _profile_runtime_scope(authority.profile_id), not db.db_path.resolve().parent.
    
    RED on unfixed code: scopes to store_dir (symlink target), which is empty.
    GREEN after fix: scopes to profile_home, where HERMES_HOME actually points.
    """
    from cron import scheduler
    from gateway import run, session_cron
    from gateway.session_authority import LiveSession
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import admit_session_input, claim_session_input
    
    authority, profile_home, store_dir = symlinked_store_authority
    
    # The db resolves to the symlink target
    assert Path(authority.db.db_path).resolve().parent == store_dir
    
    # Track which home was passed to _profile_runtime_scope
    scoped_home = {}
    original_scope = run._profile_runtime_scope
    
    def capturing_scope(home, **kwargs):
        scoped_home['path'] = Path(home)
        return original_scope(home, **kwargs)
    
    monkeypatch.setattr(run, '_profile_runtime_scope', capturing_scope)
    
    # Stub run_job
    def stub_run_job(job, *, extra_prompt, execution_id, cancel_event):
        return (True, 'ok', 'done', None)
    monkeypatch.setattr(scheduler, 'run_job', stub_run_job)
    
    # Set up a cron session
    db = authority.db
    with db:
        db.create_session('s', source='cron')
        authority.sessions['s'] = LiveSession(
            SimpleNamespace(platform=None, user_id='cron-owner'), 
            'route'
        )
        admit_session_input(db, epoch=authority.epoch, principal_id='cron-owner',
                          session_id='s', request_id='cron:job:fire', payload={'text': ''})
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        
        policy = SimpleNamespace(request_json=json.dumps({
            'extra_prompt': None, 
            'request_id': 'cron:job:fire',
            'cron_job': {'id': 'job'}
        }))
        
        ref = SessionRef(authority.profile_id, 's')
        await session_cron.execute(authority, ref, row, policy)
    
    # Verify the scope was the profile_home (authority.profile_id), not the symlink target
    assert 'path' in scoped_home, "execute() never called _profile_runtime_scope"
    assert scoped_home['path'].resolve() == profile_home.resolve(), \
        f"execute() scoped to {scoped_home['path']} instead of profile_home {profile_home} — used db.db_path.resolve().parent"
