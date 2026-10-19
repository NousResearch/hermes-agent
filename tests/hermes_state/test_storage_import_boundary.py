"""The canonical store must not import gateway/ to compute its own identities (latent cycle)."""
import subprocess
import sys


def test_runtime_storage_modules_import_without_loading_gateway():
    probe = ("import sys, hermes_state, hermes_state_runtime, hermes_state_local, hermes_state_local_lineage, "
             "hermes_state_mutation_branch, hermes_state_mutations, hermes_state_worker_compression; "
             "print(sorted(m for m in sys.modules if m == 'gateway' or m.startswith('gateway.')))")
    out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '[]', out.stdout


def test_gateway_names_are_the_storage_definitions():
    import hermes_state_keys as keys
    from gateway.session import SessionStore, profile_from_session_key_namespace
    from gateway.session_admission import admission_fingerprint
    from gateway.session_local_recovery import local_identity
    assert admission_fingerprint is keys.admission_fingerprint
    assert local_identity is keys.local_identity
    assert profile_from_session_key_namespace is keys.profile_from_session_key_namespace
    for key, profile in (('agent:main:local:dm:x', 'default'), ('agent:main~:local:dm:x', 'main'),
                         ('agent:work:local:dm:x', 'work'), ('agent::x', 'default'), ('cli:x', None), (None, None)):
        assert SessionStore._profile_from_session_key(key) == keys.profile_from_session_key(key) == profile
