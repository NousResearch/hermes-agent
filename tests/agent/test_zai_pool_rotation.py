from agent import credential_pool as cp


def _entry(entry_id: str, label: str, priority: int, token: str) -> cp.PooledCredential:
    return cp.PooledCredential(
        provider="zai",
        id=entry_id,
        label=label,
        auth_type="api_key",
        priority=priority,
        source="manual",
        access_token=token,
    )


def test_zai_1302_rotates_to_backup_when_pool_has_two_keys(monkeypatch):
    primary = _entry("max", "glm-2", 0, "key-max")
    backup = _entry("lite", "glm-old", 1, "key-lite")
    pool = cp.CredentialPool("zai", [primary, backup])
    monkeypatch.setattr(pool, "_persist", lambda *args, **kwargs: None)
    pool._current_id = primary.id

    selected = pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"code": "1302", "message": "rate limited"},
        api_key_hint=primary.runtime_api_key,
        credential_id=primary.id,
    )

    assert selected is not None
    assert selected.id == backup.id


def test_zai_1302_single_key_reprobes_immediately(monkeypatch):
    primary = _entry("max", "glm-2", 0, "key-max")
    pool = cp.CredentialPool("zai", [primary])
    monkeypatch.setattr(pool, "_persist", lambda *args, **kwargs: None)
    pool._current_id = primary.id

    selected = pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"code": "1302", "message": "rate limited"},
        api_key_hint=primary.runtime_api_key,
        credential_id=primary.id,
    )

    assert selected is not None
    assert selected.id == primary.id
