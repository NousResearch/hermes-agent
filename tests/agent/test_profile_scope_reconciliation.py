"""Behavior contracts where immutable profile authority meets current main."""
from __future__ import annotations

import pytest


def test_identity_bound_scope_cannot_be_relabelled(tmp_path):
    from agent.secret_scope import (
        build_profile_secret_scope, current_secret_scope, get_secret,
        reset_secret_scope, set_secret_scope,
    )
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / ".env").write_text("ACME_LOGIN=source-private\n", encoding="utf-8")
    scope = build_profile_secret_scope(source)
    token = set_secret_scope(scope, profile_home=str(source))
    try:
        assert get_secret("ACME_LOGIN") == "source-private"
        with pytest.raises(RuntimeError, match="profile.*home"):
            set_secret_scope(scope, profile_home=str(target))
        assert current_secret_scope() is scope
    finally:
        reset_secret_scope(token)


def test_legacy_decode_cache_cannot_admit_private_snapshot(tmp_path):
    from agent.secret_scope import load_env_file, load_env_file_snapshot
    env = tmp_path / ".env"
    env.write_bytes(b"ACME_LOGIN=caf\xe9\n")
    # Legacy UI/config readers retain main's latin-1 fallback. Child authority
    # must not gain a healthy UTF-8 snapshot from that reader's cached result.
    assert load_env_file(env) == {"ACME_LOGIN": "caf\u00e9"}
    snapshot = load_env_file_snapshot(env)
    assert snapshot.status == "failed"
    assert snapshot.error_kind == "decode"
    assert not snapshot.data
    env.write_text("ACME_LOGIN=valid\n", encoding="utf-8")
    assert load_env_file_snapshot(env).data == {"ACME_LOGIN": "valid"}
