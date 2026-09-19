"""Real PostgreSQL contracts for projects, API stores, and metrics."""
import concurrent.futures

import pytest


@pytest.fixture
def stores_home(postgres_home):
    from hermes_cli import postgres_util
    from hermes_state_backend import resolve_database_settings
    import psycopg
    from psycopg import sql
    settings = resolve_database_settings()
    yield postgres_home
    postgres_util.close_pools()
    with psycopg.connect(settings.database_url, autocommit=True) as connection:
        for name in (
            'projects', 'api_responses', 'api_runs', 'shared_metrics', 'hosted_rooms',
            'verification_evidence', 'discord_recovery', 'retaindb_queue',
            'holographic_memory', 'plugin_sample_data.db', 'matrix_crypto',
        ):
            connection.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(
                sql.Identifier(postgres_util.store_schema(settings.schema, name))))


def test_project_lifecycle_and_rollback(stores_home):
    from hermes_cli import projects_db as projects
    from hermes_cli.sqlite_util import write_txn
    with projects.connect_closing() as connection:
        project_id = projects.create_project(connection, name='Work', folders=[str(stores_home / 'repo')])
        projects.set_active(connection, project_id)
        projects.add_folder(connection, project_id, str(stores_home / 'second'), is_primary=True)
        projects.record_discovered_repos(connection, [(str(stores_home / 'repo'), 'Repo')])
        with pytest.raises(RuntimeError):
            with write_txn(connection):
                connection.execute('DELETE FROM projects WHERE id=?', (project_id,))
                raise RuntimeError('rollback')
    with projects.connect_closing() as connection:
        assert projects.get_active_id(connection) == project_id
        assert projects.get_project(connection, project_id).primary_path == str(stores_home / 'second')
        assert len(projects.list_discovered_repos(connection)) == 1
        projects.archive_project(connection, project_id)
        assert projects.list_projects(connection) == []
        projects.restore_project(connection, project_id)
        assert projects.delete_project(connection, project_id)
        assert connection.execute('SELECT count(*) FROM project_folders').fetchone()[0] == 0
    assert not (stores_home / 'projects.db').exists()


def test_store_schema_is_readable_safe_and_stable():
    from hermes_cli.postgres_util import store_schema
    name = store_schema('profile_' + 'x' * 80, 'Plugin/My Data.db')
    assert len(name) <= 63
    assert set(name) <= set('abcdefghijklmnopqrstuvwxyz0123456789_')
    assert name == store_schema('profile_' + 'x' * 80, 'Plugin/My Data.db')
    assert name != store_schema('profile_' + 'x' * 80, 'plugin-my-data.db')


def test_response_persistence_chaining_and_eviction(stores_home):
    from gateway.platforms.api_server import ResponseStore
    store = ResponseStore(max_size=2)
    try:
        store.put('one', {'messages': ['hello']})
        store.set_conversation('chat', 'one')
        store.put('two', {'messages': ['world']})
    finally:
        store.close()
    store = ResponseStore(max_size=2)
    try:
        assert store.get_conversation('chat') == 'one'
        assert store.get('one')['messages'] == ['hello']
        store.put('three', {'messages': ['again']})
        assert store.get('two') is None
        assert store.delete('one')
        assert store.get_conversation('chat') is None
        assert len(store) == 1
    finally:
        store.close()
    assert not (stores_home / 'response_store.db').exists()


def test_run_reservation_has_one_winner_and_lease(stores_home):
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
    stores = [RunIdempotencyStore() for _ in range(4)]
    try:
        def reserve(index):
            return stores[index].reserve('tenant', 'key', 'same', f'run-{index}', {'status': 'running'})
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(reserve, range(4)))
        assert sum(outcome == 'created' for outcome, _ in outcomes) == 1
        ids = {record['run_id'] for _, record in outcomes}
        assert len(ids) == 1
        run_id = ids.pop()
        assert stores[0].status_for_run('tenant', run_id)['owner_alive']
        assert stores[0].lookup('tenant', 'key', 'different')[0] == 'conflict'
        assert stores[0].lookup('another-tenant', 'key', 'same')[0] == 'missing'
        stores[0]._conn.execute('UPDATE run_idempotency SET owner_expires_at=0')
        assert not stores[1].status_for_run('tenant', run_id)['owner_alive']
        stores[0].update_status(run_id, {'status': 'completed'})
        assert stores[1].status_for_run('tenant', run_id)['status']['status'] == 'completed'
    finally:
        for store in stores:
            store.close()
    assert not (stores_home / 'runs_idempotency.db').exists()


def test_metrics_concurrent_counters_and_export(stores_home):
    from hermes_cli.observability.shared_metrics import SharedMetricsStore
    from hermes_cli.observability.shared_metrics_contract import CLIENT_ACTIVE_METRIC
    from hermes_cli.observability.shared_metrics_sender import reconcile_send_consent
    from datetime import datetime, timedelta, timezone
    store = SharedMetricsStore()
    resource = {'hermes_version': '1.0.0', 'os_family': 'linux', 'architecture': 'arm64', 'install_method': 'docker'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: store.record_counter(CLIENT_ACTIVE_METRIC, {}, resource), range(12)))
    assert store.counter_snapshot()[0]['value'] == 12
    with store._write() as connection:
        now = datetime.now(timezone.utc)
        reconcile_send_consent(connection, True, now=now)
        reconcile_send_consent(connection, True, now=now + timedelta(seconds=1))
        reconcile_send_consent(connection, False, now=now + timedelta(seconds=2))
    exported = store.create_and_export_package()
    assert exported and all(path.is_file() for path in exported)
    assert store.counter_snapshot()[0]['packaged_value'] == 12
    assert not store.database_path.exists()


def test_hosted_room_events_policy_and_owner_lease(stores_home):
    from gateway import hosted_rooms as rooms, hosted_room_driver as driver
    from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
    path = stores_home / 'shared-state.db'
    rooms.create_room(path, room_id='room', name='Work', members=[{'profile': 'ops', 'handle': 'ops'}], authority_gateway_id='gateway', now=10)
    event = rooms.append_event(path, room_id='room', event_id='hello', kind='message.user',
        actor={'kind': 'user', 'id': 'user', 'display_name': 'User'}, payload={'text': 'hello 世界'},
        authority_gateway_id='gateway', authority_epoch=1, now=11)
    assert rooms.probe_hosted_room(path, room_id='room')
    page = rooms.read_events(path, room_id='room', since_seq=0)
    assert page['events'][-1]['event_id'] == event['event_id']
    checkpoint = HostedRoomPolicyCheckpoint(path)
    checkpoint.sync(room_id='room', latest_seq=page['latest_seq'])
    lease = driver.acquire_lease(path, room_id='room', gateway_id='gateway', authority_epoch=1,
        process_generation='first', ttl_seconds=30, clock=lambda: 12)
    with pytest.raises(driver.LeaseHeldError):
        driver.acquire_lease(path, room_id='room', gateway_id='gateway', authority_epoch=1,
            process_generation='second', ttl_seconds=30, clock=lambda: 13)
    replacement = driver.acquire_lease(path, room_id='room', gateway_id='gateway', authority_epoch=1,
        process_generation='second', ttl_seconds=30, clock=lambda: 43)
    assert replacement.lease_generation > lease.lease_generation
    assert not path.exists()


def test_auxiliary_ledgers_and_plugin_storage(stores_home, monkeypatch):
    from agent import verification_evidence as evidence
    from plugins.plugin_storage import plugin_db
    from plugins.platforms.discord.recovery import DiscordRecoveryStore

    monkeypatch.setattr(evidence, '_ledger_enabled', lambda: True)
    row = evidence._insert_evidence(evidence.VerificationEvidence(
        command='pytest', canonical_command='pytest', kind='test', scope='targeted',
        status='passed', exit_code=0, cwd=str(stores_home), root=str(stores_home),
        session_id='session', output_summary='passed'))
    assert row['id'] > 0
    assert not (stores_home / 'verification_evidence.db').exists()

    discord = DiscordRecoveryStore(stores_home)
    discord.call(lambda connection: connection.execute(
        "INSERT INTO discord_messages(message_id, status, updated_at) VALUES (?, ?, ?)",
        ('message', 'responded', '2026-01-01T00:00:00Z')))
    assert discord.call(lambda connection: connection.execute(
        'SELECT status FROM discord_messages WHERE message_id=?', ('message',)).fetchone()[0]
    ) == 'responded'
    assert not discord.path().exists()

    connection = plugin_db('sample')
    try:
        connection.execute('CREATE TABLE IF NOT EXISTS values_store (key TEXT PRIMARY KEY, value TEXT)')
        connection.execute('INSERT INTO values_store VALUES (?, ?)', ('key', 'value'))
        assert connection.execute('SELECT value FROM values_store WHERE key=?', ('key',)).fetchone()[0] == 'value'
    finally:
        connection.close()


def test_optional_memory_stores_use_postgres(stores_home):
    import threading
    from plugins.memory.holographic.retrieval import FactRetriever
    from plugins.memory.holographic.store import MemoryStore
    from plugins.memory.retaindb import _WriteQueue

    with MemoryStore(hrr_dim=64) as memory:
        fact_id = memory.add_fact('PostgreSQL remembers project notes', category='ops')
        assert memory.list_facts(category='ops')[0]['fact_id'] == fact_id
        assert FactRetriever(memory, hrr_dim=64).search('project notes', limit=2)[0]['fact_id'] == fact_id
    assert not (stores_home / 'memory_store.db').exists()

    release = threading.Event()
    started = threading.Event()

    class Client:
        def ingest_session(self, *args):
            started.set()
            release.wait(timeout=10)

    queue_path = stores_home / 'retaindb_queue.db'
    queue = _WriteQueue(Client(), queue_path)
    try:
        queue.enqueue('user', 'session', [{'role': 'user', 'content': 'hello'}])
        assert started.wait(timeout=5)
        assert queue._get_conn().execute('SELECT count(*) FROM pending').fetchone()[0] == 1
        release.set()
    finally:
        release.set()
        queue.shutdown()
    assert not queue_path.exists()
