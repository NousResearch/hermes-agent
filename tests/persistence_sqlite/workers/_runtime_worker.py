"""The registered compute workers the runtime worker tests drive: executions on the backend's profile store, written
through the owner's receipt transaction (``mutate_worker_execution`` on the store's holder)."""
import pytest


def worker_store(holder, scope, outbox):
    """A ``RuntimeSessionStore`` whose RPC is the owner's receipt transaction on *holder*."""
    from agent.runtime_session_store import RuntimeSessionStore
    from persistence.sqlite.workers._receipt import mutate_worker_execution

    return RuntimeSessionStore(lambda method, **p: mutate_worker_execution(holder, **p), scope, outbox)


def setup_store(persistence, tmp_path):
    """``(holder, RuntimeSessionStore, scope)`` for a compute worker on session ``owned`` (cwd ``/reserved``,
    profile ``fixture``) beside a ``foreign`` session."""
    from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
    from persistence.sqlite.conversation._store.sessions import create_session
    from persistence.sqlite.workers._executions import register_worker_execution

    holder = persistence.holder()
    create_session(holder, 'owned', 'cli', cwd='/reserved', profile_name='fixture')
    create_session(holder, 'foreign', 'cli', system_prompt='private')
    epoch = begin_runtime_epoch(holder, instance_id='fixture')
    scope = dict(epoch=epoch, execution_id='worker', session_id='owned', generation=0)
    register_worker_execution(holder, **scope, kind='compute', adoption_secret='private')
    return holder, worker_store(holder, scope, tmp_path / 'outbox'), scope


@pytest.fixture
def worker(backend, scope_a, tmp_path):
    """``(backend store, RuntimeSessionStore)`` for a registered compute worker execution on session ``owned``."""
    from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
    from persistence.sqlite.conversation._store.sessions import create_session
    from persistence.sqlite.workers._executions import register_worker_execution

    persistence = backend.open(scope_a)
    holder = persistence.holder()
    create_session(holder, 'owned', 'cli', system_prompt='prefix')
    create_session(holder, 'foreign', 'cli', system_prompt='secret')
    epoch = begin_runtime_epoch(holder, instance_id='fixture')
    scope = dict(epoch=epoch, execution_id='worker', session_id='owned', generation=0)
    register_worker_execution(holder, **scope, kind='compute', adoption_secret='secret')
    store = worker_store(holder, scope, tmp_path / 'outbox')
    yield persistence, store
    store.failure = None
    store.journal['pending'] = []
    store.close()
