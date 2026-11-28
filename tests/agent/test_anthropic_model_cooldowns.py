"""A generic Anthropic 429 benches only the model that was rate-limited (#111769, #61451).

Real ``agent.credential_pool`` against a real temp auth store: one API-key credential,
one ``mark_exhausted_and_rotate(status_code=429, model=A)``.
"""
import json

import pytest

KEY = "sk-ant-api03-synthetic-test-key-0000"
MODEL_A = "claude-sonnet-4-5"
MODEL_B = "claude-haiku-4-5"


def _load_pool(tmp_path, monkeypatch, rows):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "fakehome"))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": rows}}))
    from agent.credential_pool import load_pool
    return load_pool("anthropic")


def _row(eid, key, priority, **extra):
    return {"id": eid, "label": eid, "auth_type": "api_key", "priority": priority, "source": "manual",
            "access_token": key, **extra}


@pytest.fixture
def pool(tmp_path, monkeypatch):
    return _load_pool(tmp_path, monkeypatch, [_row("seat", KEY, 0)])


def test_generic_429_benches_only_the_rate_limited_model(pool, monkeypatch):
    from agent.anthropic_credentials import resolve_anthropic_token
    from agent.credential_pool import load_pool

    ctx = {"message": "This request would exceed your account's rate limit. Please try again later."}
    assert pool.mark_exhausted_and_rotate(
        status_code=429, error_context=ctx, api_key_hint=KEY, failure_reason="rate_limit", model=MODEL_A,
    ) is None  # a sole credential has nothing to rotate to for model A

    assert pool.entries()[0].last_status is None  # credential-wide state untouched
    assert pool.select(model=MODEL_A) is None
    assert pool.select(model=MODEL_B) is not None
    assert pool.select() is None  # a caller that names no model honours every active cooldown
    assert pool.next_available_at(model=MODEL_A) is not None
    assert pool.next_available_at(model=MODEL_B) is None

    # The cooldown is persisted, so another process (and the env/borrowed token resolver,
    # which reads the store fresh) sees the same per-model verdict.
    fresh = load_pool("anthropic")
    assert fresh.select(model=MODEL_A) is None and fresh.select(model=MODEL_B) is not None
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    assert resolve_anthropic_token(model=MODEL_A) is None
    assert resolve_anthropic_token(model=MODEL_B) == KEY
    assert resolve_anthropic_token() == KEY  # model-less diagnostics keep the key


@pytest.mark.parametrize("status_code, failure_reason", [(401, None), (402, "billing"), (429, "billing")])
def test_auth_and_billing_failures_stay_credential_wide(pool, status_code, failure_reason):
    from agent.credential_pool import STATUS_EXHAUSTED

    pool.mark_exhausted_and_rotate(
        status_code=status_code, api_key_hint=KEY, failure_reason=failure_reason, model=MODEL_A,
    )
    assert pool.entries()[0].last_status == STATUS_EXHAUSTED
    assert not pool.entries()[0].model_cooldowns
    assert pool.select(model=MODEL_B) is None


KEY_BACKUP = "sk-ant...0001"


def _two_row_pool(tmp_path, monkeypatch, backup_cooldowns):
    """A primary row and a backup row; the backup carries per-model cooldowns only."""
    import time

    until = time.time() + 3600
    return _load_pool(tmp_path, monkeypatch, [
        _row("primary", KEY, 0),
        _row("backup", KEY_BACKUP, 1, model_cooldowns={model: until for model in backup_cooldowns}),
    ])


KEY_LEVEL_FAILURES = [(401, None), (402, "billing"), (429, "billing")]


@pytest.mark.parametrize("status_code, failure_reason", KEY_LEVEL_FAILURES)
def test_key_level_rotation_takes_a_row_cooling_down_only_for_another_model(
    tmp_path, monkeypatch, status_code, failure_reason,
):
    pool = _two_row_pool(tmp_path, monkeypatch, backup_cooldowns=[MODEL_B])
    next_entry = pool.mark_exhausted_and_rotate(
        status_code=status_code, api_key_hint=KEY, failure_reason=failure_reason, model=MODEL_A,
    )
    assert next_entry is not None and next_entry.id == "backup"


@pytest.mark.parametrize("status_code, failure_reason", KEY_LEVEL_FAILURES)
def test_key_level_rotation_skips_a_row_cooling_down_for_the_requested_model(
    tmp_path, monkeypatch, status_code, failure_reason,
):
    pool = _two_row_pool(tmp_path, monkeypatch, backup_cooldowns=[MODEL_A])
    assert pool.mark_exhausted_and_rotate(
        status_code=status_code, api_key_hint=KEY, failure_reason=failure_reason, model=MODEL_A,
    ) is None


@pytest.mark.parametrize("status_code, failure_reason", KEY_LEVEL_FAILURES)
def test_key_level_rotation_skips_a_row_for_another_endpoint(tmp_path, monkeypatch, status_code, failure_reason):
    import time

    api, proxy = "https://api.anthropic.com", "https://old-proxy.example.com"
    pool = _load_pool(tmp_path, monkeypatch, [
        _row("primary", KEY, 0, base_url=api),
        _row("proxy", KEY_BACKUP, 1, base_url=proxy, model_cooldowns={MODEL_B: time.time() + 3600}),
        _row("backup", "sk-ant...0002", 2, base_url=api),
    ])
    next_entry = pool.mark_exhausted_and_rotate(
        status_code=status_code, api_key_hint=KEY, failure_reason=failure_reason, model=MODEL_A, endpoint_url=api,
    )
    assert next_entry is not None and next_entry.id == "backup"


@pytest.mark.parametrize("status_code, failure_reason", KEY_LEVEL_FAILURES)
def test_key_level_rotation_without_a_model_honours_every_cooldown(
    tmp_path, monkeypatch, status_code, failure_reason,
):
    pool = _two_row_pool(tmp_path, monkeypatch, backup_cooldowns=[MODEL_B])
    assert pool.mark_exhausted_and_rotate(
        status_code=status_code, api_key_hint=KEY, failure_reason=failure_reason,
    ) is None
