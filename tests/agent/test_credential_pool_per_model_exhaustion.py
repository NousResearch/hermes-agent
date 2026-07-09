"""
Regression tests for issue #61451 - credential_pool: a model-scoped 429
exhausts the whole credential, blocking other models with free quota.

The bug: PooledCredential tracks exhaustion (last_status, last_error_code,
last_error_reset_at) at the credential level, not at the (credential,
model) level. When a 429 hits for model X (e.g. claude-fable-5 which
has its own rate-limit bucket), the entire credential is marked
exhausted. Subsequent requests for model Y (e.g. claude-sonnet-4.6) on
the SAME credential are rejected because the credential-wide state
says "exhausted" even though model Y has plenty of quota.

The fix: track exhaustion per (credential, model) pair. When
_mark_exhausted is called with a model argument, only mark THAT
model as exhausted on the credential — not the whole credential.
Selection then skips the entry only when the requested model is
marked exhausted.

Test approach: write failing tests that exercise the per-model state
shape, then refactor the dataclass + selection logic. The tests
use the same auth.json fixture pattern as existing
test_credential_pool.py tests.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_auth_store(tmp_path: Path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _anthropic_pool_entry(entry_id: str, label: str, priority: int) -> dict:
    return {
        "id": entry_id,
        "label": label,
        "auth_type": "api_key",
        "priority": priority,
        "source": "manual",
        "access_token": "***",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mark_exhausted_with_model_only_blocks_that_model(
    tmp_path, monkeypatch
):
    """The bug: a single model-specific 429 marks the whole credential
    as exhausted. After the fix, marking model X as exhausted on
    credential C must NOT block selection of C for model Y.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    _anthropic_pool_entry("cred-1", "primary", 0),
                ],
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    entry = pool.select()

    # Mark claude-fable-5 as exhausted on this credential (the bug case:
    # Fable has its own rate-limit bucket; its 429 carries Fable's reset
    # window which is much longer than sonnet's).
    pool._mark_exhausted(
        entry,
        status_code=429,
        error_context={
            "reason": "rate_limit",
            "message": "Fable rate limit exceeded",
            "reset_at": time.time() + 3600,  # 1 hour ahead (Fable's bucket)
        },
        model="claude-fable-5",
    )

    # CRITICAL ASSERTION: querying for claude-sonnet-4.6 (which has
    # its own, separate, larger bucket) on the SAME credential must
    # still succeed. The fix makes exhaustion model-scoped, so the
    # credential is NOT exhausted for sonnet.
    sonnet_entry = pool.select(model="claude-sonnet-4.6")
    assert sonnet_entry is not None, (
        "credential_pool marked the whole credential exhausted for "
        "Fable's 429; claude-sonnet-4.6 on the same credential should "
        "still be available. Issue #61451."
    )
    assert sonnet_entry.id == "cred-1", (
        f"the only available credential was not returned for sonnet; "
        f"got entry {sonnet_entry!r}"
    )


def test_mark_exhausted_with_model_still_blocks_same_model(
    tmp_path, monkeypatch
):
    """Regression guard: the per-model fix MUST still block the
    exhausted model — otherwise we lose the rate-limit protection
    entirely.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    _anthropic_pool_entry("cred-1", "primary", 0),
                    _anthropic_pool_entry("cred-2", "secondary", 1),
                ],
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    entry = pool.select()

    pool._mark_exhausted(
        entry,
        status_code=429,
        error_context={
            "reason": "rate_limit",
            "message": "Fable rate limit exceeded",
            "reset_at": time.time() + 3600,
        },
        model="claude-fable-5",
    )

    # Fable query on the same credential SHOULD be blocked — the rate
    # limit is real. Selection should fall through to cred-2 (no Fable
    # exhaustion on it).
    fable_entry = pool.select(model="claude-fable-5")
    if fable_entry is not None:
        # If something IS returned, it must not be cred-1 (which has
        # the Fable exhaustion recorded).
        assert fable_entry.id != "cred-1", (
            "cred-1 has Fable exhaustion recorded but was still returned "
            "for a Fable query; the per-model tracking isn't enforcing "
            "the rate-limit semantics"
        )


def test_mark_exhausted_without_model_blocks_credential_legacy(
    tmp_path, monkeypatch
):
    """Regression guard for backwards compatibility: callers that
    don't pass a model argument (legacy call sites) should still get
    credential-wide exhaustion — the previous behavior. This protects
    code paths that don't yet know about per-model tracking (health
    checks, OAuth writethrough, etc.).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    _anthropic_pool_entry("cred-1", "primary", 0),
                    _anthropic_pool_entry("cred-2", "secondary", 1),
                ],
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    entry = pool.select()

    # No model argument — credential-wide exhaustion (legacy behavior).
    pool._mark_exhausted(
        entry,
        status_code=429,
        error_context={"reason": "rate_limit"},
    )

    # Both sonnet and fable queries should fall through to cred-2.
    sonnet_entry = pool.select(model="claude-sonnet-4.6")
    assert sonnet_entry is None or sonnet_entry.id != "cred-1", (
        "no-model-arg mark_exhausted should still credential-wide block; "
        "sonnet query got cred-1 back"
    )


def test_exhaustion_state_serialization_round_trip(
    tmp_path, monkeypatch
):
    """The per-model exhaustion state must persist across pool reload.
    Mark a credential's claude-fable-5 as exhausted, persist+reload,
    and confirm the per-model state survives the round trip.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    _anthropic_pool_entry("cred-1", "primary", 0),
                    _anthropic_pool_entry("cred-2", "secondary", 1),
                ],
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    entry = pool.select()
    pool._mark_exhausted(
        entry,
        status_code=429,
        error_context={
            "reason": "rate_limit",
            "reset_at": time.time() + 3600,
        },
        model="claude-fable-5",
    )

    # Force a fresh load — reads from disk. (load_pool() re-reads every
    # call; no force_reload needed.)
    pool2 = load_pool("anthropic")
    entry2 = pool2.select(model="claude-fable-5")

    # The same credential (cred-1) should still be exhausted for Fable
    # after reload. If selection falls through to cred-2, the per-model
    # state didn't persist.
    if entry2 is not None:
        assert entry2.id != "cred-1", (
            "after reload, cred-1's Fable exhaustion was lost; "
            "selection returned cred-1 for a Fable query that should "
            "have been blocked"
        )


def test_select_skips_credential_exhausted_for_requested_model(
    tmp_path, monkeypatch
):
    """The selection function must consult per-model exhaustion state.
    A query for model X must skip a credential where X is exhausted,
    even if the credential is fine for model Y.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    _anthropic_pool_entry("cred-1", "primary", 0),
                    _anthropic_pool_entry("cred-2", "secondary", 1),
                ],
            },
        },
    )

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    entry = pool.select()
    pool._mark_exhausted(
        entry,
        status_code=429,
        error_context={
            "reason": "rate_limit",
            "reset_at": time.time() + 3600,
        },
        model="claude-fable-5",
    )

    # Fable query — cred-1 should be skipped (Fable exhausted), cred-2 selected.
    fable_entry = pool.select(model="claude-fable-5")
    if fable_entry is not None:
        assert fable_entry.id != "cred-1", (
            f"cred-1 has Fable exhaustion but was returned for a Fable "
            f"query. Selection must consult per-model state, not "
            f"credential-wide state."
        )

    # Sonnet query — cred-1 should be available (no sonnet exhaustion).
    sonnet_entry = pool.select(model="claude-sonnet-4.6")
    assert sonnet_entry is not None, (
        "sonnet query got no credential back; cred-1 should be "
        "available for sonnet (no sonnet-specific exhaustion)"
    )