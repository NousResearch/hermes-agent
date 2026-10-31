"""The local Desktop opens its own login-stamped conversation: adoption's not_found falls through.

``resolve_local_target`` adopted legacy rows as writable conversations, but a login-stamped Desktop
row (the served dashboard's backend stamp on ``user_id``) is a foreign owner for every other
caller — the host's own native peer included — so adoption refused exactly ``not_found`` and the
machine's own Desktop could not open its chat. The refusal now falls through to the desktop
recovery floor, mapping a compression tip to its lineage root first (a child mount would fork the
FIFO). A login-stamped NON-desktop lineage keeps refusing identically.
"""
import threading
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch

STAMP = 'basic:admin'


def _desktop_lineage(db, root, child_ids):
    """A desktop root closed by compression with continuation children; the tip resumes it."""
    db.create_session(root, source='desktop', user_id=STAMP, profile_name='default')
    prev = root
    for child in child_ids:
        db.create_session(child, source='desktop', user_id=STAMP, profile_name='default')
        db._write_sql("UPDATE sessions SET end_reason='compression' WHERE id=?", (prev,))
        db._write_sql("UPDATE sessions SET parent_session_id=? WHERE id=?", (prev, child))
        prev = child
    tip = child_ids[-1]
    lineage = db.get_compression_lineage(tip)
    assert lineage[0] == root and lineage[-1] == tip
    return tip


@pytest.fixture
def authority(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from gateway.session_authority import SessionAuthority
    from hermes_state_runtime import begin_runtime_epoch

    db = SessionDB(tmp_path / 'state.db')
    epoch = begin_runtime_epoch(db, instance_id='owner')

    class _Store:
        def __init__(self):
            self._lock = threading.RLock()
            self._entries = {}

        def _ensure_loaded_locked(self):
            pass

        def _generate_session_key(self, source):
            from gateway.session import build_session_key
            return build_session_key(source)

    runner = SimpleNamespace(_draining=False, session_store=_Store())
    authority = SessionAuthority(runner, profile_id='p', instance_id='owner', db=db, epoch=epoch)
    adapter = SimpleNamespace(gateway_runner=runner)
    runner._adapter_for_source = lambda source: adapter
    # the post-restart local native peer: authenticated subject is NOT the row's login stamp
    actor = SimpleNamespace(profile_id=authority.profile_id, subject='local:user',
                            transport_id='native-1',
                            capabilities=frozenset({'session:create', 'session:read',
                                                    'session:control', 'session:operator'}))
    from gateway.session_local_migration import bind_native_transport
    bind_native_transport(authority, actor,
                          {'provider': 'local', 'profile_id': authority.profile_id,
                           'instance_id': authority.instance_id, 'actor': actor})
    authority._native_legacy_transports[actor.transport_id] = actor
    yield authority
    db.close()


@pytest.mark.asyncio
async def test_a_login_stamped_desktop_tip_resolves_to_its_lineage_root(authority):
    """resolve_local_target on the tip mounts the ROOT as an API-family source, keeping the row's
    ownership stamps untouched; the routing entry still executes on the physical tip."""
    from gateway.config import Platform
    from gateway.session_authority import LiveSession
    from gateway.session_contract import SessionRef
    from gateway.session_local_migration import resolve_local_target

    root = '20261030_080000_droot00'
    tip = _desktop_lineage(authority.db, root, ['20261030_090000_dchild1',
                                               '20261030_100000_dchild2'])

    ref = resolve_local_target(authority, actor_of(authority), tip)
    assert isinstance(ref, SessionRef) and ref.session_id == root

    live = authority.sessions[root]
    assert isinstance(live, LiveSession)
    assert live.source.platform == Platform.API_SERVER
    assert live.source.chat_id == root
    assert live.source.user_id == STAMP  # stamps untouched: no receipt, no policy commit
    entry = authority.runner.session_store._entries['agent:main:api_server:dm:' + root]
    assert entry.session_id == tip  # the root mount executes on the physical tip


@pytest.mark.asyncio
async def test_a_login_stamped_non_desktop_lineage_still_refuses(authority):
    """The fall-through is desktop-only: the same stamped shape on a gui lineage answers bare
    not_found and mounts nothing."""
    from gateway.session_local_migration import resolve_local_target

    root = '20261030_110000_groot00'
    db = authority.db
    db.create_session(root, source='gui', user_id=STAMP, profile_name='default')
    db._write_sql("UPDATE sessions SET end_reason='compression' WHERE id=?", (root,))
    db.create_session('20261030_120000_gchild1', source='gui', user_id=STAMP, profile_name='default')
    db._write_sql("UPDATE sessions SET parent_session_id=? WHERE id=?", (root, '20261030_120000_gchild1'))
    tip = '20261030_120000_gchild1'

    with pytest.raises(RuntimeStoreError, match='not_found'):
        resolve_local_target(authority, actor_of(authority), tip)
    assert root not in authority.sessions and tip not in authority.sessions
    assert not list(authority.runner.session_store._entries)


def actor_of(authority):
    return authority._native_legacy_transports['native-1']
