"""A ``/p/<profile>/`` API session runs under the profile whose authority admitted it.

Under multiplex the launch profile's ``api_server`` listener receives every ``/p/<name>/`` request
and the routed authority admits the turn — but the turn's runtime home comes from the identity
pinned on the session's source. A source bound with no ``profile`` resolved as the launch profile,
so a secondary's API turn executed with the default profile's config, model and credentials.
"""
from pathlib import Path

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


@pytest.mark.asyncio
async def test_api_binding_pins_runtime_to_the_admitting_profile(tmp_path, monkeypatch):
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_api import bind_api_session, restore_api_session
    from gateway.session_identity import identity_of
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    runner = None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        registry = runner.session_authorities
        alpha_home = homes[1][1]
        alpha = registry.for_home(alpha_home)
        launch = registry.launch
        alpha.db.create_session('api-alpha', source='api_server')
        launch.db.create_session('api-launch', source='api_server')

        ref = bind_api_session(alpha, 'api-alpha')
        live = alpha.sessions[ref.session_id]
        identity = identity_of(live.source)
        assert live.source.profile == 'alpha'
        assert identity is not None and identity.multiplexed
        # Runs under alpha; the launch listener is the transport that received and answers it.
        assert identity.runtime_home == alpha_home.resolve()
        assert identity.transport_profile == 'default'
        assert runner._resolve_profile_home_for_source(live.source) == alpha_home.resolve()
        assert runner._profile_scope_key_for_source(live.source) == alpha_home.resolve()
        assert live.route.startswith('agent:alpha:')
        # The launch authority's own session stays in the launch profile (no cross-talk).
        launch_ref = bind_api_session(launch, 'api-launch')
        launch_source = launch.sessions[launch_ref.session_id].source
        assert launch_source.profile is None
        assert identity_of(launch_source).runtime_home == root.resolve()
        assert launch.sessions[launch_ref.session_id].route.startswith('agent:main:')

        # A restart restores the same placement from the durable receipt.
        alpha.sessions.clear()
        restored = restore_api_session(alpha, 'api-alpha')
        assert restored == ref
        restored_source = alpha.sessions['api-alpha'].source
        assert restored_source.profile == 'alpha'
        restored_identity = identity_of(restored_source)
        assert restored_identity.runtime_home == alpha_home.resolve()
        assert restored_identity.transport_profile == 'default'
        assert Path(runner._resolve_profile_home_for_source(restored_source)) == alpha_home.resolve()
    finally:
        for authority in list(getattr(runner, 'session_authorities', None) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.asyncio
@pytest.mark.parametrize("profile_index", [0, 1])
async def test_api_binding_refuses_legacy_duplicate_without_routing_writes(tmp_path, monkeypatch, profile_index):
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_api import bind_api_session
    from hermes_state_runtime import RuntimeStoreError

    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    runner = None
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        registry = runner.session_authorities
        owners = (registry.launch, registry.for_home(homes[1][1]))
        # Model an existing poisoned install; guarded public creation correctly refuses it.
        for authority in owners:
            authority.db._conn.execute("INSERT INTO sessions(id,source,started_at) "
                                       "VALUES ('shared-legacy','api_server',1)")
            authority.db._conn.commit()

        def snapshot():
            return [{table: [tuple(row) for row in authority.db._conn.execute(
                f"SELECT * FROM {table} ORDER BY 1")]
                for table in ("sessions", "gateway_routing", "state_meta")} for authority in owners]

        before = snapshot()
        with pytest.raises(RuntimeStoreError) as refused:
            bind_api_session(owners[profile_index], 'shared-legacy')
        assert refused.value.reason == 'admission_conflict'
        assert snapshot() == before
    finally:
        for authority in list(getattr(runner, 'session_authorities', None) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)
