"""Model-scoped credential-pool rotation.

Regression for #146385: a key-level 429 rotation (and the unmatched-identity
rotation) used to select without the request's model, so a cooldown recorded
for an unrelated model blocked every credential and rotation returned None
even though a credential could serve the request's model. The model-scoped
branch already forwarded the model; these tests pin the same contract for the
other two rotation paths, plus the conservative model-ignorant behaviour that
must stay when no model is known.
"""

from __future__ import annotations

import time

from agent.credential_pool import STATUS_EXHAUSTED, CredentialPool, PooledCredential

_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _entry(i: int, *, token: str | None = None, cooldowns: dict | None = None) -> PooledCredential:
    return PooledCredential(
        provider="openai-codex",
        id=f"cred-{i}",
        label=f"acct-{i}",
        auth_type="api_key",
        priority=i,
        source="manual",
        access_token=token or f"tok-{i}-1234567890",
        base_url=_BASE_URL,
        model_cooldowns=cooldowns,
    )


def _quota_429() -> dict:
    return {
        "reset_at": time.time() + 7000,
        "reason": "usage_limit_reached",
        "message": "The usage limit has been reached",
    }


def _cooldown(model: str) -> dict:
    return {model: time.time() + 86400 * 365}


def _pool_with_cooldowns(model: str) -> CredentialPool:
    return CredentialPool("openai-codex", [_entry(0, cooldowns=_cooldown(model)), _entry(1, cooldowns=_cooldown(model))])


def test_key_level_429_rotates_past_unrelated_model_cooldown():
    pool = _pool_with_cooldowns("other-model")
    rotated = pool.mark_exhausted_and_rotate(
        status_code=429, error_context=_quota_429(), credential_id="cred-0",
        failure_reason="rate_limit", model="current-model",
    )
    assert rotated is not None and rotated.id == "cred-1"
    assert next(e for e in pool.entries() if e.id == "cred-0").last_status == STATUS_EXHAUSTED


def test_key_level_429_blocked_by_same_model_cooldown():
    pool = _pool_with_cooldowns("current-model")
    rotated = pool.mark_exhausted_and_rotate(
        status_code=429, error_context=_quota_429(), credential_id="cred-0",
        failure_reason="rate_limit", model="current-model",
    )
    assert rotated is None


def test_key_level_429_without_model_keeps_conservative_selection():
    pool = _pool_with_cooldowns("other-model")
    rotated = pool.mark_exhausted_and_rotate(
        status_code=429, error_context=_quota_429(), credential_id="cred-0",
        failure_reason="rate_limit",
    )
    assert rotated is None


def test_key_level_429_marks_key_sharing_siblings_exhausted():
    pool = CredentialPool("openai-codex", [
        _entry(0, token="tok-shared-1234567890", cooldowns=_cooldown("other-model")),
        _entry(1, token="tok-shared-1234567890", cooldowns=_cooldown("other-model")),
    ])
    rotated = pool.mark_exhausted_and_rotate(
        status_code=429, error_context=_quota_429(), credential_id="cred-0",
        failure_reason="rate_limit", model="current-model",
    )
    assert rotated is None
    assert all(e.last_status == STATUS_EXHAUSTED for e in pool.entries())


def test_unmatched_identity_rotates_past_unrelated_model_cooldown():
    pool = _pool_with_cooldowns("other-model")
    rotated = pool.mark_exhausted_and_rotate(
        status_code=401, error_context={"message": "bad key"},
        api_key_hint="tok-no-such-key", model="current-model",
    )
    assert rotated is not None and rotated.id == "cred-0"
    assert all(e.last_status is None for e in pool.entries())


def test_unmatched_identity_blocked_by_same_model_cooldown():
    pool = _pool_with_cooldowns("current-model")
    rotated = pool.mark_exhausted_and_rotate(
        status_code=401, error_context={"message": "bad key"},
        api_key_hint="tok-no-such-key", model="current-model",
    )
    assert rotated is None
