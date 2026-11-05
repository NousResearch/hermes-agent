"""The canonical store must not import gateway/ to compute its own identities (latent cycle)."""
import subprocess
import sys


def test_runtime_storage_modules_import_without_loading_gateway():
    probe = ("import sys, persistence.sqlite, persistence.sqlite._holder, persistence.sqlite.conversation._store.core, "
             "persistence.sqlite.authority._admission_store, persistence.sqlite.authority._mutation, "
             "persistence.sqlite.authority._runtime, persistence.sqlite.authority._runtime_ends, "
             "persistence.sqlite.workers._executions, persistence.sqlite.workers._receipt, "
             "persistence.sqlite.authority._local, persistence.sqlite.conversation._store.local, "
             "persistence.sqlite.conversation._store.local_lineage, persistence.sqlite.gateway._local_lineage, "
             "persistence.sqlite.authority._mutation_publish, persistence.sqlite.workers._compression; "
             "print(sorted(m for m in sys.modules if m == 'gateway' or m.startswith('gateway.')))")
    out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '[]', out.stdout


def test_gateway_names_are_the_storage_definitions():
    from persistence.sqlite._state import keys
    import session_keys
    from gateway import session_worker
    from gateway.session import SessionStore, profile_from_session_key_namespace
    from gateway import session_local_recovery
    from persistence.authority import bindings
    from persistence.authority.vocabulary import admission_fingerprint
    assert session_worker.admission_fingerprint is admission_fingerprint
    for target, payload in (('s', {'a': 1, 'b': [2, 'x']}), ('t', {'text': 'caf\u00e9', 'n': None})):
        assert admission_fingerprint(canonical_target=target, payload=payload) == keys.admission_fingerprint(
            canonical_target=target, payload=payload)
    assert session_local_recovery.local_session_id is bindings.local_session_id
    assert profile_from_session_key_namespace is session_keys.profile_from_session_key_namespace
    for key, profile in (('agent:main:local:dm:x', 'default'), ('agent:main~:local:dm:x', 'main'),
                         ('agent:work:local:dm:x', 'work'), ('agent::x', 'default'), ('cli:x', None), (None, None)):
        assert SessionStore._profile_from_session_key(key) == keys.profile_from_session_key(key) == profile
        assert session_keys.profile_from_session_key(key) == profile
