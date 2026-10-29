"""Regression for #146344: a live-mounted local session whose store entry was never bound
(a pre-scope-era listing/restore mounts it as a history view without binding) must self-heal
from its validated receipt instead of refusing ``admission_conflict`` on every resume."""
import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


@pytest.mark.asyncio
async def test_live_mount_without_store_entry_self_heals(tmp_path, monkeypatch):
    from gateway import run
    from gateway.run import _profile_runtime_scope
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway.session_local_recovery import restore_local_session

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    process_ownership.reserve([home for _, home in homes])
    runner = _runner(root, homes)
    try:
        await initialize_gateway_runtime(runner)
        home = homes[1][1]
        authority = runner.session_authorities.for_home(home)
        actor = Principal('owner', authority.profile_id,
                          frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
        with _profile_runtime_scope(home, {}):
            ref = create_local_session(authority, actor, {'request_id': 'old-desktop', 'source': 'gui',
                'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
        assert ref.session_id in authority.sessions  # live-mounted by the create
        store = runner.session_store
        live = authority.sessions[ref.session_id]
        with store._lock:
            assert store._entries.get(live.route) is not None  # create bound the entry
            # The failing state: the session stays live-mounted while its store entry
            # is gone (how a pre-scope-era listing mount left these sessions).
            del store._entries[live.route]
        # The resume must heal from the receipt instead of refusing forever (#146344).
        with _profile_runtime_scope(home, {}):
            assert restore_local_session(authority, ref.session_id) == ref
        with store._lock:
            current = store._entries.get(live.route)
        assert current is not None and current.session_id == ref.session_id
        # And the healed state is stable: a second resume keeps succeeding.
        with _profile_runtime_scope(home, {}):
            assert restore_local_session(authority, ref.session_id) == ref
    finally:
        for owner in runner.session_authorities:
            owner.db.close()
