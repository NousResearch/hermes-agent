"""Regression coverage for Anthropic model-scoped 429 cooldowns (#111769)."""

from agent.credential_pool import CredentialPool, PooledCredential, STATUS_EXHAUSTED


def _pool():
    return CredentialPool(
        "anthropic",
        [
            PooledCredential(
                provider="anthropic", id="credential", label="credential",
                auth_type="api_key", priority=0, source="manual", access_token="key",
            )
        ],
    )


def test_anthropic_429_only_cools_the_failed_model(monkeypatch):
    pool = _pool()
    monkeypatch.setattr("agent.credential_pool.time.time", lambda: 1_000.0)

    pool.mark_exhausted_and_rotate(
        status_code=429, api_key_hint="key", model="claude-sonnet-4",
    )

    entry = pool.entries()[0]
    assert entry.last_status is None
    assert pool.select(model="claude-haiku-4").id == entry.id
    assert pool.select(model="claude-sonnet-4") is None
    assert pool.next_available_at(model="claude-sonnet-4") == 4_600.0


def test_anthropic_model_cooldown_expiry_restores_selection(monkeypatch):
    pool = _pool()
    now = [1_000.0]
    monkeypatch.setattr("agent.credential_pool.time.time", lambda: now[0])
    pool.mark_exhausted_and_rotate(status_code=429, api_key_hint="key", model="claude-sonnet-4")

    now[0] = 4_601.0
    assert pool.select(model="claude-sonnet-4") is pool.entries()[0]


def test_reset_status_clears_anthropic_model_cooldown(monkeypatch):
    pool = _pool()
    monkeypatch.setattr("agent.credential_pool.time.time", lambda: 1_000.0)
    pool.mark_exhausted_and_rotate(status_code=429, api_key_hint="key", model="claude-sonnet-4")

    pool.reset_status("credential")

    assert pool.select(model="claude-sonnet-4") is not None


def test_persisted_model_cooldowns_merge_by_model():
    from hermes_cli.auth import _merge_disk_cooldown_state

    merged = _merge_disk_cooldown_state(
        {"access_token": "key", "model_cooldowns": {"claude-haiku-4": 2_000.0}},
        {"access_token": "key", "model_cooldowns": {"claude-sonnet-4": 3_000.0}},
        "anthropic",
    )

    assert merged["model_cooldowns"] == {"claude-haiku-4": 2_000.0, "claude-sonnet-4": 3_000.0}


def test_anthropic_401_and_billing_402_remain_credential_wide():
    for status_code, failure_reason in ((401, None), (402, "billing")):
        pool = _pool()
        pool.mark_exhausted_and_rotate(
            status_code=status_code, api_key_hint="key", model="claude-sonnet-4",
            failure_reason=failure_reason,
        )
        assert pool.entries()[0].last_status == STATUS_EXHAUSTED
        assert pool.select(model="claude-haiku-4") is None
