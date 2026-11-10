"""Real YAML admission policy, profile isolation and conservative physical recovery."""
import asyncio
from pathlib import Path

import pytest

from tests.gateway.test_parallel_native_ingress import admissions, event, setup_owner
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', ['0', '9', 'true', '1.5', '"2"', 'null'])
async def test_invalid_yaml_parallel_limit_refuses_without_custody(monkeypatch, raw):
    from hermes_constants import get_hermes_home
    (get_hermes_home() / 'config.yaml').write_text(
        f'display:\n  busy_input_mode: parallel\n  parallel_max_runs: {raw}\n')
    calls = []

    async def produce(*args):
        calls.append(args)
        return 'must not execute'

    _runner, authority, adapter = await setup_owner(monkeypatch, produce)
    try:
        await adapter.handle_message(event(adapter, 'invalid', 'work'))
        await async_wait_until(lambda: not adapter._background_tasks, what='invalid config refusal')
        assert admissions(authority) == []
        assert calls == []
        assert adapter.sent
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_reboot_reconciliation_keeps_unknown_and_never_releases_current_or_foreign_boot(monkeypatch):
    from gateway.session_authority import SessionAuthority
    from gateway.session_parallel import physical_count, route_input
    from hermes_state_parallel import reconcile_rebooted_physical
    from hermes_state_runtime import claim_session_input, recover_session_inputs
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda *args: None)

    async def produce(*args):
        raise AssertionError('unknown work must never be replayed')

    runner, authority, adapter = await setup_owner(monkeypatch, produce)
    try:
        incoming = event(adapter, 'lost', 'unknown input')
        assert await route_input(runner, incoming) is None
        receipt = await authority.admit_native(incoming)
        sid = receipt.ref.session_id
        claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
        # Model reboot identities as ledger data, not a fake host OS. Real host
        # facts are checked separately; this test never reboots the machine.
        identities = [('old-local', 'machine', 'previous-boot'),
                      ('current', 'machine', 'current-boot'),
                      ('foreign', 'other-machine', 'previous-boot'),
                      ('unprovable', '', '')]
        for token, machine, boot in identities:
            authority.db._execute_write(lambda conn: conn.execute(
                'INSERT INTO parallel_work_physical(token,work_id,machine_id,boot_id) VALUES(?,?,?,?)',
                (token, incoming.source.parallel_work.work_id, machine, boot)))
        recover_session_inputs(authority.db, epoch=authority.epoch)
        assert reconcile_rebooted_physical(authority.db, '', '') == 0
        assert reconcile_rebooted_physical(authority.db, 'machine', 'current-boot') == 1
        assert physical_count(authority) == 3
        assert reconcile_rebooted_physical(authority.db, 'machine', 'current-boot') == 0
        # Make the claimed input unknown using the native epoch recovery path.
        from hermes_state_runtime import begin_runtime_epoch
        successor = begin_runtime_epoch(authority.db, instance_id='successor')
        recover_session_inputs(authority.db, epoch=successor)
        row = admissions(authority)[0]
        assert row['status'] == 'unknown'
        # The same native operator acknowledgement used by /discard never
        # clears live, foreign or unprovable physical occupancy.
        from hermes_state_runtime import resolve_unknown_session_input
        resolve_unknown_session_input(authority.db, epoch=successor,
                                      admission_id=row['admission_id'], generation=row['generation'])
        assert physical_count(authority) == 3
        assert admissions(authority)[0]['status'] == 'terminal'
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_multiplex_yaml_limits_and_work_anchors_do_not_cross_profiles(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.profile_routing import parse_profile_routes
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from tests.gateway.test_parallel_native_ingress import DiscordFixture
    from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner
    import gateway.session_finite as finite
    gate = asyncio.Event()
    entered = []

    async def produce(authority, ref, row):
        entered.append((authority.profile_id, row['request_id']))
        await gate.wait()
        return row['payload']['text'].upper()

    monkeypatch.setattr(finite, 'execute_finite_admission', produce)
    root, homes = _reserve_homes(tmp_path, monkeypatch, names=('alpha',))
    for name, home in homes:
        limit = 1 if name == 'default' else 2
        (home / 'config.yaml').write_text(f'display:\n  busy_input_mode: parallel\n  parallel_max_runs: {limit}\n')
        (home / '.env').write_text('DISCORD_ALLOWED_USERS=alice,bob\n')
    process_ownership.reserve([home for _, home in homes])
    runner = _runner(root, homes)
    runner.config.profile_routes = parse_profile_routes([
        {'name': 'alpha-room', 'platform': 'discord', 'profile': 'alpha', 'chat_id': 'alpha-room'}])
    runner._snapshot_profile_busy_modes('alpha', {'display': {'busy_input_mode': 'parallel'}})
    runner._busy_input_mode = 'parallel'
    adapter = DiscordFixture()
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    try:
        await initialize_gateway_runtime(runner)
        runner._wire_adapter_handlers(adapter)
        for mid, chat in [('same', 'room'), ('default-overflow', 'room'),
                          ('same', 'alpha-room'), ('alpha-second', 'alpha-room')]:
            await adapter.handle_message(event(adapter, mid, mid, chat=chat))
        await async_wait_until(lambda: len(entered) == 3, what='independent profile limits')
        launch = runner.session_authority
        alpha = runner.session_authorities.for_home(homes[1][1])
        assert len(admissions(launch)) == 1
        assert len(admissions(alpha)) == 2
        assert {r['request_id'] for r in admissions(alpha)} == {'same', 'alpha-second'}
        assert admissions(launch)[0]['target_session_id'] not in alpha.sessions
        await adapter.handle_message(event(adapter, 'foreign-stop', '/stop', reply='same', chat='elsewhere'))
        assert len(entered) == 3
        assert all(r['status'] == 'started' for a in (launch, alpha) for r in admissions(a))
    finally:
        gate.set()
        await async_wait_until(lambda: not adapter._background_tasks, what='multiplex cleanup')
        for authority in list(getattr(runner, 'session_authorities', ())):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)


@pytest.mark.platforms('linux')
def test_kernel_boot_identity_reads_the_real_host():
    from hermes_platform.host.facts import kernel_boot_identity
    import uuid
    machine, boot = kernel_boot_identity()
    # Restricted procfs hosts may conservatively return no proof.
    if machine or boot:
        assert uuid.UUID(machine).hex == Path('/etc/machine-id').read_text().strip()
        assert str(uuid.UUID(boot)) == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
