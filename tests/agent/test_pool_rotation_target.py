"""The 429 recovery hinge asks the pool for a rotation TARGET, not mere availability.

``_pool_may_recover_from_rate_limit`` decides whether a rate-limited turn waits for
credential-pool rotation (sleeping through ``Retry-After``, up to 600s) or activates
``fallback_model`` right away. Waiting only pays off when another credential can serve
the rate-limited model. Regression for the two ways the old ``has_available()`` probe said
"wait" with nowhere to go: a bench a sibling process persisted for that model (this pool
never adopts peer cooldown state), and a duplicate seat sharing the failed key.
Related: #11314, #13636, #117933.

Real ``agent.credential_pool`` against a real temp auth store; two ``load_pool`` instances
stand in for two processes sharing one ``auth.json``.
"""
import json
import time

import pytest

from run_agent import _pool_may_recover_from_rate_limit

KEY_A = "sk-ant...000a"
KEY_B = "sk-ant...000b"
MODEL_X = "claude-opus-4-5"
MODEL_Y = "claude-sonnet-4-5"


def _entry(entry_id: str, key: str, priority: int) -> dict:
    return {"id": entry_id, "label": entry_id, "auth_type": "api_key", "priority": priority,
            "source": "manual", "access_token": key}


@pytest.fixture
def auth_store(tmp_path, monkeypatch):
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
    return root / "auth.json"


def _write_pool(auth_file, entries):
    auth_file.write_text(json.dumps({"credential_pool": {"anthropic": entries}}))


def _load():
    from agent.credential_pool import load_pool
    return load_pool("anthropic")


def test_peer_bench_for_the_rate_limited_model_ends_the_wait(auth_store):
    _write_pool(auth_store, [_entry("seat-a", KEY_A, 0), _entry("seat-b", KEY_B, 1)])
    session_pool = _load()  # a long-lived session's pool, loaded before the peer's 429
    peer_pool = _load()  # another process sharing the same auth.json

    ctx = {"message": "This request would exceed your account's rate limit."}
    peer_pool.mark_exhausted_and_rotate(
        status_code=429, error_context=ctx, api_key_hint=KEY_B, failure_reason="rate_limit", model=MODEL_X)

    # The in-memory verdict is stale by design: seat-b still looks usable for MODEL_X here.
    assert session_pool.has_available(model=MODEL_X) is True
    # seat-a just 429'd for MODEL_X; the only other seat is benched for it on disk -> fall back now.
    assert _pool_may_recover_from_rate_limit(
        session_pool, model=MODEL_X, failed_key=KEY_A, failed_entry_id="seat-a") is False
    # The peer bench is per model: a different model can still rotate onto seat-b.
    assert _pool_may_recover_from_rate_limit(
        session_pool, model=MODEL_Y, failed_key=KEY_A, failed_entry_id="seat-a") is True
    # Read-only probe: nothing was adopted into the session pool or persisted back.
    assert session_pool.entries()[1].model_cooldowns in (None, {})


def test_lapsed_peer_bench_does_not_block(auth_store):
    _write_pool(auth_store, [_entry("seat-a", KEY_A, 0), _entry("seat-b", KEY_B, 1)])
    pool = _load()
    rows = json.loads(auth_store.read_text())
    rows["credential_pool"]["anthropic"][1]["model_cooldowns"] = {MODEL_X: time.time() - 5}
    auth_store.write_text(json.dumps(rows))

    assert _pool_may_recover_from_rate_limit(
        pool, model=MODEL_X, failed_key=KEY_A, failed_entry_id="seat-a") is True


def test_failed_credential_and_its_duplicate_seat_are_not_targets(auth_store):
    _write_pool(auth_store, [_entry("seat-a", KEY_A, 0), _entry("seat-a-dup", KEY_A, 1)])
    pool = _load()
    assert len(pool.entries()) == 2 and pool.has_available(model=MODEL_X) is True

    # Two rows, one quota: rotating onto the duplicate retries the account that just 429'd.
    assert _pool_may_recover_from_rate_limit(
        pool, model=MODEL_X, failed_key=KEY_A, failed_entry_id="seat-a") is False
    # ... whereas a genuinely different seat is a target.
    _write_pool(auth_store, [_entry("seat-a", KEY_A, 0), _entry("seat-b", KEY_B, 1)])
    assert _pool_may_recover_from_rate_limit(
        _load(), model=MODEL_X, failed_key=KEY_A, failed_entry_id="seat-a") is True


def test_model_less_caller_keeps_the_unscoped_verdict(auth_store):
    _write_pool(auth_store, [_entry("seat-a", KEY_A, 0), _entry("seat-b", KEY_B, 1)])
    pool = _load()
    assert _pool_may_recover_from_rate_limit(pool, failed_key=KEY_A, failed_entry_id="seat-a") is True
    pool.mark_exhausted_and_rotate(status_code=429, api_key_hint=KEY_B, failure_reason="rate_limit", model=MODEL_X)
    # Without a model every active cooldown blocks the entry (the conservative unscoped rule).
    assert _pool_may_recover_from_rate_limit(pool, failed_key=KEY_A, failed_entry_id="seat-a") is False


def test_unreadable_store_falls_open_to_the_in_memory_verdict(auth_store):
    _write_pool(auth_store, [_entry("seat-a", KEY_A, 0), _entry("seat-b", KEY_B, 1)])
    pool = _load()
    auth_store.write_text("{not json")
    assert _pool_may_recover_from_rate_limit(
        pool, model=MODEL_X, failed_key=KEY_A, failed_entry_id="seat-a") is True
