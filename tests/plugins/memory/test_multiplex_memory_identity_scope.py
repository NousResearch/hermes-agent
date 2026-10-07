"""Multiplex invariant: memory-provider identity/tenant/endpoint never comes from the default profile.

Under ``gateway.multiplex_profiles`` ``os.environ`` is the DEFAULT profile's ``.env``. When a secondary
profile's scope does not define RETAINDB_PROJECT /
HERMES_HONCHO_HOST, the provider must fall back to its own default
(per-profile partition), NOT write the secondary's memories into the default profile's account.
"""
from __future__ import annotations

import pytest

from agent import secret_scope

_DEFAULT_ENV = {
    "RETAINDB_PROJECT": "proj-default", "RETAINDB_BASE_URL": "https://rdb.default",
    "HERMES_HONCHO_HOST": "host-default", "HONCHO_BASE_URL": "https://honcho.default",
}


@pytest.fixture
def secondary_profile(monkeypatch, tmp_path):
    """Multiplex ON; default profile's values in environ; secondary profile `b` scope with only its
    own API keys (no identity/tenant/endpoint vars)."""
    for k, v in _DEFAULT_ENV.items():
        monkeypatch.setenv(k, v)
    home = tmp_path / ".hermes"
    prof_b = home / "profiles" / "b"
    prof_b.mkdir(parents=True)
    (prof_b / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(prof_b))
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"RETAINDB_API_KEY": "rdb-b", "HONCHO_API_KEY": "honcho-b"})
    try:
        yield prof_b
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_secondary_profile_memory_identity_never_inherits_default_environ(secondary_profile):
    import plugins.memory.retaindb as retaindb

    provider = retaindb.RetainDBMemoryProvider()
    provider.initialize("s1", hermes_home=str(secondary_profile))
    assert provider._client.project == "hermes-b"
    assert "default" not in provider._client.base_url
