"""Cross-process model cooldown resets stay authoritative (#128995).

A model-scoped cooldown keeps last_status healthy, so the credential-wide reset tombstone
path does not cover it. These tests pin both sides of the race: stale pre-reset snapshots
must lose, while a genuinely new model failure after the reset must still persist.
"""
import json
import time

import pytest

KEY = "sk-ant-api03-reset-race-test-key"
MODEL_A = "claude-sonnet-4-5"
MODEL_B = "claude-haiku-4-5"


@pytest.fixture
def live_pool(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(fake_home))
    monkeypatch.setenv("HERMES_HOME", str(root))
    for name in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]

    (root / "auth.json").write_text(json.dumps({"credential_pool": {"anthropic": [{
        "id": "seat",
        "label": "seat",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": KEY,
    }]}}))

    from agent.credential_pool import load_pool
    return root, load_pool("anthropic")


def _cool(pool, model):
    return pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"message": "rate limit"},
        api_key_hint=KEY,
        failure_reason="rate_limit",
        model=model,
    )


def _disk_row():
    from hermes_cli.auth import read_credential_pool
    return read_credential_pool("anthropic")[0]


def test_stale_flush_cannot_resurrect_model_cooldown_after_reset(live_pool):
    _, live = live_pool
    from agent.credential_pool import load_pool

    assert _cool(live, MODEL_A) is None
    assert live.select(model=MODEL_A) is None

    fresh = load_pool("anthropic")
    assert fresh.reset_statuses() == 1
    cleared = _disk_row()
    assert not cleared.get("model_cooldowns")
    assert cleared.get("status_cleared_at")

    # Process A still holds its pre-reset map. An unrelated flush must not put it back on disk.
    live._persist()
    after_flush = _disk_row()
    assert not after_flush.get("model_cooldowns")
    assert after_flush.get("status_cleared_at") == cleared.get("status_cleared_at")

    # The same already-running process must observe the reset without a restart.
    assert live.select(model=MODEL_A) is not None


def test_failure_after_reset_survives_without_reviving_older_model(live_pool):
    _, live = live_pool
    from agent.credential_pool import load_pool

    assert _cool(live, MODEL_A) is None
    fresh = load_pool("anthropic")
    assert fresh.reset_statuses() == 1
    reset_at = _disk_row().get("status_cleared_at")
    assert reset_at

    # A real failure after the reset is newer authority. Only MODEL_B should return.
    assert _cool(live, MODEL_B) is None
    row = _disk_row()
    cooldowns = row.get("model_cooldowns") or {}
    assert MODEL_A not in cooldowns
    assert cooldowns.get(MODEL_B, 0) > time.time()
    observed = row.get("model_cooldown_observed_at") or {}
    assert observed.get(MODEL_B, 0) > reset_at

    assert live.select(model=MODEL_A) is not None
    assert live.select(model=MODEL_B) is None


def test_reset_tombstone_clears_legacy_model_cooldown_without_observation_metadata(live_pool):
    root, _ = live_pool
    from agent.credential_pool import load_pool

    # Pre-fix rows have model_cooldowns but no per-model observation metadata.
    store = json.loads((root / "auth.json").read_text())
    row = store["credential_pool"]["anthropic"][0]
    row["model_cooldowns"] = {MODEL_A: time.time() + 7200}
    row.pop("model_cooldown_observed_at", None)
    (root / "auth.json").write_text(json.dumps(store))

    stale = load_pool("anthropic")
    assert stale.select(model=MODEL_A) is None
    fresh = load_pool("anthropic")
    assert fresh.reset_statuses() == 1

    stale._persist()
    assert not (_disk_row().get("model_cooldowns") or {})
    assert stale.select(model=MODEL_A) is not None
