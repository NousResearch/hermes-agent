"""Credential rotation must preserve model-scoped pool eligibility."""

from agent.credential_pool import CredentialPool, PooledCredential, STATUS_EXHAUSTED


TARGET_MODEL = "gpt-6.1-sol"
OTHER_MODEL = "gpt-6-astra-900k"


def _entry(index: int) -> PooledCredential:
    return PooledCredential(
        provider="anthropic",
        id=f"cred-{index}",
        label=f"account-{index}",
        auth_type="api_key",
        priority=index,
        source="manual",
        access_token=f"key-{index}",
    )


def _pool_with_unrelated_cooldown() -> CredentialPool:
    pool = CredentialPool("anthropic", [_entry(0), _entry(1)])
    # Both accounts cannot serve OTHER_MODEL, but remain valid for TARGET_MODEL.
    for index in (0, 1):
        pool.mark_exhausted_and_rotate(
            status_code=429,
            failure_reason="rate_limit",
            api_key_hint=f"key-{index}",
            credential_id=f"cred-{index}",
            model=OTHER_MODEL,
        )
    assert pool.select(model=TARGET_MODEL).id == "cred-0"
    assert pool.select() is None
    return pool


def test_credential_wide_rotation_uses_requested_model_eligibility():
    pool = _pool_with_unrelated_cooldown()

    rotated = pool.mark_exhausted_and_rotate(
        status_code=401,
        api_key_hint="key-0",
        credential_id="cred-0",
        model=TARGET_MODEL,
    )

    assert rotated is not None and rotated.id == "cred-1"
    assert next(entry for entry in pool.entries() if entry.id == "cred-0").last_status == STATUS_EXHAUSTED


def test_unmatched_rotation_uses_requested_model_eligibility():
    pool = _pool_with_unrelated_cooldown()

    rotated = pool.mark_exhausted_and_rotate(
        status_code=401,
        credential_id="missing-credential",
        model=TARGET_MODEL,
    )

    assert rotated is not None and rotated.id == "cred-0"
