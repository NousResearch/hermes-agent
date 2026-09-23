"""Absent optional editor state cannot revoke an older credential reference."""
from dataclasses import asdict
import hashlib
import json


def test_policy_identity_preserves_pre_editor_credential_binding(tmp_path):
    from gateway.session_policy import build_policy
    from gateway.session_policy_credentials import policy_identity
    policy = build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'local'}, {})
    legacy = asdict(policy)
    legacy.pop('credential_ref')
    legacy.pop('config_secret_ref')
    legacy.pop('editor_mcp_json')
    expected = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
    assert policy_identity(policy) == expected
