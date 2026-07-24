"""Shared Anthropic store schema, staging, and selection tests."""

from __future__ import annotations

import json
import time

import pytest

from tests.agent.anthropic_shared_test_helpers import (
    FIXTURE_ACCESS,
    enable_marker,
    make_row,
    shared_root,
    stage_three,
    write_root_auth,
)


def test_profile_mode_ignores_dormant_shared_rows(shared_root):
    stage_three(shared_root, attest=False)
    from agent.anthropic_adapter import resolve_anthropic_token
    from agent.credential_pool import load_pool

    # No marker → legacy; dormant rows not used; no env → None
    assert resolve_anthropic_token() is None
    pool = load_pool("anthropic")
    # May be empty or env-seeded only — must not contain shared ids
    ids = {e.id for e in pool.entries()}
    stored = json.loads((shared_root / "auth.json").read_text())
    shared_ids = {
        e["id"] for e in stored["shared_credential_pools"]["anthropic"]["entries"]
    }
    assert ids.isdisjoint(shared_ids)


def test_shared_mode_profiles_see_same_three_ids(shared_root, monkeypatch):
    pool = stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    from agent import anthropic_shared_pool as sp
    from agent.credential_pool import load_pool

    sp.reset_startup_epoch_for_tests()
    p1 = load_pool("anthropic")
    ids1 = [e.id for e in p1.entries()]
    assert ids1 == [e["id"] for e in pool["entries"]]

    # Switch to a named profile HERMES_HOME
    prof = shared_root / "profiles" / "coder"
    prof.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(prof))
    sp.reset_startup_epoch_for_tests()
    p2 = load_pool("anthropic")
    assert [e.id for e in p2.entries()] == ids1
    # Profile auth.json must not be created with anthropic rows
    prof_auth = prof / "auth.json"
    if prof_auth.exists():
        data = json.loads(prof_auth.read_text())
        cp = data.get("credential_pool") or {}
        assert not cp.get("anthropic")


def test_shared_persist_refuses_profile_write(shared_root):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    from agent import anthropic_shared_pool as sp
    from agent.credential_pool import load_pool
    from hermes_cli.auth import AuthError

    sp.reset_startup_epoch_for_tests()
    pool = load_pool("anthropic")
    with pytest.raises(AuthError):
        pool._persist()


def test_api_key_row_rejected(shared_root):
    from agent.anthropic_shared_pool import AuthError, validate_shared_row
    from hermes_cli.auth import AuthError as AE

    row = make_row(priority=0)
    row["auth_type"] = "api_key"
    with pytest.raises((AuthError, AE)):
        validate_shared_row(row)


def test_duplicate_fingerprint_rejected(shared_root):
    from agent.anthropic_shared_pool import (
        append_row,
        get_shared_mutation_capability,
        AuthError,
    )
    from hermes_cli.auth import AuthError as AE

    cap = get_shared_mutation_capability()
    r1 = make_row(priority=0, refresh="fixture-oauth-refresh-same")
    r2 = make_row(priority=1, refresh="fixture-oauth-refresh-same")
    append_row(r1, capability=cap)
    # Same fingerprint → idempotent return, not fourth
    existing = append_row(r2, capability=cap)
    assert existing["id"] == r1["id"]


def test_fourth_grant_rejected(shared_root):
    from agent.anthropic_shared_pool import (
        append_row,
        get_shared_mutation_capability,
    )
    from hermes_cli.auth import AuthError

    cap = get_shared_mutation_capability()
    for i in range(3):
        append_row(make_row(priority=i, refresh=f"fixture-oauth-refresh-uniq-{i}"), capability=cap)
    with pytest.raises(AuthError):
        append_row(make_row(priority=3, refresh="fixture-oauth-refresh-uniq-3"), capability=cap)


def test_fill_first_order_and_exhaustion_rotate(shared_root):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    from agent import anthropic_shared_pool as sp

    sp.reset_startup_epoch_for_tests()
    ctx = sp.resolve_shared_anthropic_credential()
    pool = sp.load_shared_pool_for_management(require_active_three=True)
    assert ctx.row_id == pool["entries"][0]["id"]

    # Exhaust first
    sp.patch_status(
        ctx.row_id,
        expected_generation=ctx.token_generation,
        last_status="exhausted",
        last_error_reason="rate_limit",
        last_error_reset_at=time.time() + 3600,
        last_error_code=429,
        last_error_message="rate limited",
    )
    ctx2 = sp.resolve_shared_anthropic_credential()
    assert ctx2.row_id == pool["entries"][1]["id"]


def test_elapsed_exhaustion_normalizes_under_lock(shared_root):
    pool = stage_three(shared_root, attest=True)
    pool["entries"][0]["last_status"] = "exhausted"
    pool["entries"][0]["last_error_reason"] = "rate_limit"
    pool["entries"][0]["last_error_reset_at"] = time.time() - 10
    pool["entries"][0]["last_status_at"] = time.time() - 100
    pool["entries"][0]["last_error_code"] = 429
    pool["entries"][0]["last_error_message"] = "old"
    write_root_auth(
        shared_root,
        {
            "version": 1,
            "providers": {},
            "shared_credential_pools": {"anthropic": pool},
        },
    )
    enable_marker(shared_root)
    from agent import anthropic_shared_pool as sp

    sp.reset_startup_epoch_for_tests()
    ctx = sp.resolve_shared_anthropic_credential()
    assert ctx.row_id == pool["entries"][0]["id"]


def test_malformed_marker_fail_closed(shared_root):
    stage_three(shared_root, attest=True)
    path = shared_root / "shared" / "anthropic_pool_scope.json"
    path.write_text("{not-json")
    path.chmod(0o600)
    from agent.anthropic_shared_pool import read_scope_state
    from hermes_cli.auth import AuthError

    with pytest.raises(AuthError):
        read_scope_state()


def test_symlink_marker_fail_closed(shared_root, tmp_path):
    stage_three(shared_root, attest=True)
    real = tmp_path / "marker.json"
    real.write_text(json.dumps({"version": 1, "scope": "shared", "epoch": "x"}))
    marker = shared_root / "shared" / "anthropic_pool_scope.json"
    marker.symlink_to(real)
    from agent.anthropic_shared_pool import read_scope_state
    from hermes_cli.auth import AuthError

    with pytest.raises(AuthError):
        read_scope_state()


def test_stale_generation_status_ignored(shared_root):
    stage_three(shared_root, attest=True)
    enable_marker(shared_root)
    from agent import anthropic_shared_pool as sp

    sp.reset_startup_epoch_for_tests()
    ctx = sp.resolve_shared_anthropic_credential()
    # Bump generation artificially
    with sp.root_auth_lock():
        store = sp.load_root_auth_strict()
        pool = sp.get_shared_namespace(store)
        row = next(e for e in pool["entries"] if e["id"] == ctx.row_id)
        row["token_generation"] = ctx.token_generation + 1
        pool["revision"] += 1
        sp.set_shared_namespace(store, pool)
        sp.save_root_auth_strict(store)
    # Stale gen-N callback must not kill gen N+1
    sp.patch_status(
        ctx.row_id,
        expected_generation=ctx.token_generation,
        last_status="dead",
        last_error_reason="auth_terminal",
        last_error_message="stale 401",
    )
    with sp.root_auth_lock():
        store = sp.load_root_auth_strict()
        pool = sp.get_shared_namespace(store)
        row = next(e for e in pool["entries"] if e["id"] == ctx.row_id)
        assert row["last_status"] is None
        assert row["token_generation"] == ctx.token_generation + 1
