"""Per-model cooldowns must not bench a login for models it never serves, nor for model-less callers.

Live incident: a delegate_task child routed ``claude-fable-5-1`` onto the ``openai-codex`` (ChatGPT
Pro) batch provider. The Codex entitlement 400 recorded a one-year cooldown for that Claude id on
the OpenAI login, and because ``peek()`` / ``select()`` without a model honour every active model
cooldown, the usage reader and other model-less callers saw "no openai-codex login" for a year.
"""
import json
import time
import types

import pytest

from agent.credential_pool_model_cooldowns import (
    COOLDOWN_KIND_ENTITLEMENT, COOLDOWN_KIND_WINDOW, MODEL_ENTITLEMENT_BENCH_SECONDS,
    MODEL_WINDOW_COOLDOWN_MAX_SECONDS, cooldown_kind, merge_model_cooldown_kinds, model_cooldown_until,
    provider_serves_model,
)

CODEX_MODEL = "gpt-5.3-codex"
CLAUDE_MODEL = "claude-fable-5-1"
FAKE_KEY_A = "fixture-a"


@pytest.fixture
def pool(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [
        {"id": "cred-0", "label": "acct-0", "auth_type": "oauth", "priority": 0, "source": "manual",
         "access_token": FAKE_KEY_A, "refresh_token": "rt-0", "expires_at_ms": 4_000_000_000_000},
    ]}}))
    from agent.credential_pool import load_pool
    return load_pool("openai-codex")


def test_provider_serves_model_vendor_check():
    assert provider_serves_model("openai-codex", CODEX_MODEL)
    assert not provider_serves_model("openai-codex", CLAUDE_MODEL)
    assert provider_serves_model("anthropic", "claude-sonnet-4-5")
    assert not provider_serves_model("anthropic", CODEX_MODEL)
    # Aggregators / multi-vendor / unknown stay permissive (today's behaviour).
    assert provider_serves_model("openrouter", CLAUDE_MODEL)
    assert provider_serves_model("nous", CLAUDE_MODEL)
    assert provider_serves_model("openai-codex", "some-unknown-id")
    assert provider_serves_model(None, CLAUDE_MODEL) and provider_serves_model("openai-codex", None)


def test_foreign_model_entitlement_400_records_no_cooldown(pool):
    assert pool.mark_exhausted_and_rotate(
        status_code=400, api_key_hint=FAKE_KEY_A, credential_id="cred-0", failure_reason="model_entitlement",
        model=CLAUDE_MODEL,
    ) is None
    entry = pool.entries()[0]
    assert not entry.model_cooldowns and entry.last_status is None
    assert pool.peek() is not None and pool.select() is not None
    from agent.credential_pool import load_pool
    assert not load_pool("openai-codex").entries()[0].model_cooldowns  # nothing persisted either


def test_foreign_model_goes_straight_to_session_marker(pool):
    from agent.fallback_cooldown import _is_entitlement_rejected, _mark_entitlement_rejected_model

    agent = types.SimpleNamespace(provider="openai-codex", model=CLAUDE_MODEL, _credential_pool=pool,
                                  _buffer_diagnostic_status=lambda *_a, **_k: None)
    err = types.SimpleNamespace(status_code=400, message=(
        f"The '{CLAUDE_MODEL}' model is not supported when using Codex with a ChatGPT account."))
    assert _mark_entitlement_rejected_model(agent, err) is True
    assert _is_entitlement_rejected(agent, "openai-codex", CLAUDE_MODEL)


def test_entitlement_bench_does_not_hide_login_from_model_less_callers(pool):
    pool.mark_exhausted_and_rotate(
        status_code=400, api_key_hint=FAKE_KEY_A, credential_id="cred-0", failure_reason="model_entitlement",
        model=CODEX_MODEL,
    )
    entry = pool.entries()[0]
    assert entry.model_cooldowns[CODEX_MODEL] > time.time() + 300 * 24 * 3600  # bench kept for its model
    assert pool.select(model=CODEX_MODEL) is None
    assert pool.peek() is not None  # usage reader / status keep seeing the login
    assert model_cooldown_until(entry, None) is None
    assert model_cooldown_until(entry, CODEX_MODEL) is not None


def test_window_cooldown_still_blocks_model_less_callers_and_is_capped(pool):
    far = time.time() + MODEL_ENTITLEMENT_BENCH_SECONDS  # a bogus reset_at a year out
    pool.mark_exhausted_and_rotate(
        status_code=402, api_key_hint=FAKE_KEY_A, credential_id="cred-0", failure_reason="usage_window",
        model=CODEX_MODEL, error_context={"usage_window": True, "reset_at": far},
    )
    until = pool.entries()[0].model_cooldowns[CODEX_MODEL]
    assert until <= time.time() + MODEL_WINDOW_COOLDOWN_MAX_SECONDS + 1
    assert model_cooldown_until(pool.entries()[0], None) is not None  # conservative for unscoped routes
    assert pool.entries()[0].model_cooldown_kinds == {CODEX_MODEL: COOLDOWN_KIND_WINDOW}


def _advance_clock(monkeypatch, seconds):
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() + seconds)


@pytest.mark.parametrize("days_later", [183, 300, 364])
def test_entitlement_kind_holds_for_the_whole_bench(pool, monkeypatch, days_later):
    """The kind is recorded, not inferred from time left, so past a 182d
    threshold the bench still spares model-less callers and still blocks its own model."""
    pool.mark_exhausted_and_rotate(
        status_code=400, api_key_hint=FAKE_KEY_A, credential_id="cred-0", failure_reason="model_entitlement",
        model=CODEX_MODEL,
    )
    assert pool.entries()[0].model_cooldown_kinds == {CODEX_MODEL: COOLDOWN_KIND_ENTITLEMENT}
    from agent.credential_pool import load_pool
    reloaded = load_pool("openai-codex")  # the kind survives a persist round-trip
    _advance_clock(monkeypatch, days_later * 24 * 3600)
    entry = reloaded.entries()[0]
    assert model_cooldown_until(entry, None) is None
    assert reloaded.peek() is not None
    assert model_cooldown_until(entry, CODEX_MODEL) is not None
    assert reloaded.select(model=CODEX_MODEL) is None
    assert reloaded.limit_state([CODEX_MODEL]) is None  # not a window: the picker says nothing


def test_entitlement_bench_ends_at_expiry(pool, monkeypatch):
    pool.mark_exhausted_and_rotate(
        status_code=400, api_key_hint=FAKE_KEY_A, credential_id="cred-0", failure_reason="model_entitlement",
        model=CODEX_MODEL,
    )
    _advance_clock(monkeypatch, MODEL_ENTITLEMENT_BENCH_SECONDS + 60)
    assert model_cooldown_until(pool.entries()[0], CODEX_MODEL) is None


def test_window_kind_blocks_model_less_callers_until_reset(pool, monkeypatch):
    reset = time.time() + 2 * 24 * 3600
    pool.mark_exhausted_and_rotate(
        status_code=402, api_key_hint=FAKE_KEY_A, credential_id="cred-0", failure_reason="usage_window",
        model=CODEX_MODEL, error_context={"usage_window": True, "reset_at": reset},
    )
    _advance_clock(monkeypatch, 24 * 3600)
    entry = pool.entries()[0]
    assert model_cooldown_until(entry, None) is not None
    assert pool.limit_state([CODEX_MODEL]) == {"scope": "models", "models": {CODEX_MODEL: pytest.approx(reset, abs=2)}}


def test_legacy_row_without_kind_gets_kind_fixed_at_load():
    from agent.credential_pool import PooledCredential

    now = time.time()
    legacy = PooledCredential.from_dict("openai-codex", {
        "id": "x", "access_token": FAKE_KEY_A,
        "model_cooldowns": {"far": now + 300 * 86400, "near": now + 3600, "month": now + 30 * 86400},
    })
    assert legacy.model_cooldown_kinds == {
        "far": COOLDOWN_KIND_ENTITLEMENT, "near": COOLDOWN_KIND_WINDOW, "month": COOLDOWN_KIND_WINDOW,
    }
    # A legacy 30-day window (older writers had no cap) is a window, clamped like a new one.
    month = (legacy.model_cooldowns or {})["month"]
    assert month <= time.time() + MODEL_WINDOW_COOLDOWN_MAX_SECONDS + 1
    assert model_cooldown_until(legacy, None) == pytest.approx(month)
    assert model_cooldown_until(legacy, "far") == pytest.approx(now + 300 * 86400)


@pytest.mark.parametrize("days_later", [1, 200, 358, 364])
def test_legacy_entitlement_stays_entitlement_for_its_whole_life(tmp_path, monkeypatch, days_later):
    """Review finding: a kindless entitlement row must not turn into a window in its last 8 days."""
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    now = time.time()
    until = now + MODEL_ENTITLEMENT_BENCH_SECONDS - 60  # written by an older version a minute ago
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [
        {"id": "cred-0", "label": "acct-0", "auth_type": "oauth", "priority": 0, "source": "manual",
         "access_token": FAKE_KEY_A, "refresh_token": "rt-0", "expires_at_ms": 4_000_000_000_000,
         "model_cooldowns": {CODEX_MODEL: until}},
    ]}}))
    from agent.credential_pool import load_pool

    load_pool("openai-codex")  # upgrade: kind fixed and persisted
    stored = json.loads((root / "auth.json").read_text())["credential_pool"]["openai-codex"][0]
    assert stored["model_cooldown_kinds"] == {CODEX_MODEL: COOLDOWN_KIND_ENTITLEMENT}
    _advance_clock(monkeypatch, days_later * 86400)
    entry = load_pool("openai-codex").entries()[0]
    assert model_cooldown_until(entry, None) is None  # model-less callers never blocked
    assert model_cooldown_until(entry, CODEX_MODEL) == pytest.approx(until)  # model stays benched


def test_load_drops_foreign_model_cooldown_rows():
    from agent.credential_pool import PooledCredential

    now = time.time()
    entry = PooledCredential.from_dict("openai-codex", {
        "id": "x", "access_token": FAKE_KEY_A,
        "model_cooldowns": {CLAUDE_MODEL: now + 300 * 86400, CODEX_MODEL: now + 3600},
        "model_cooldown_kinds": {CLAUDE_MODEL: COOLDOWN_KIND_ENTITLEMENT, CODEX_MODEL: COOLDOWN_KIND_WINDOW},
    })
    assert entry.model_cooldowns == {CODEX_MODEL: pytest.approx(now + 3600)}
    assert entry.model_cooldown_kinds == {CODEX_MODEL: COOLDOWN_KIND_WINDOW}


def test_kindless_merge_row_uses_legacy_rule_not_window_cap():
    from agent.credential_pool import PooledCredential

    now = time.time()
    entry = PooledCredential.from_dict("openai-codex", {"id": "x", "access_token": FAKE_KEY_A})
    entry.model_cooldowns = {"m": now + 30 * 86400}  # e.g. adopted from an older peer's disk write
    assert cooldown_kind(entry, "m", now + 30 * 86400, now) == COOLDOWN_KIND_WINDOW
    assert model_cooldown_until(entry, None) is not None


def test_kind_follows_the_winning_reset_in_merges():
    early, late = 1_000.0, 2_000.0
    merged = merge_model_cooldown_kinds(
        ({"m": late, "n": early}, {"m": COOLDOWN_KIND_ENTITLEMENT, "n": COOLDOWN_KIND_WINDOW}),
        ({"m": early, "n": late}, {"m": COOLDOWN_KIND_WINDOW, "n": COOLDOWN_KIND_ENTITLEMENT}),
    )
    assert merged == {"m": COOLDOWN_KIND_ENTITLEMENT, "n": COOLDOWN_KIND_ENTITLEMENT}
    assert merge_model_cooldown_kinds(({"m": late}, None)) == {}


def test_disk_merge_keeps_kinds():
    from hermes_cli.auth import _merge_disk_cooldown_state

    disk = {"id": "x", "model_cooldowns": {"a": 5e9}, "model_cooldown_kinds": {"a": COOLDOWN_KIND_ENTITLEMENT}}
    mem = {"id": "x", "model_cooldowns": {"b": 4e9}, "model_cooldown_kinds": {"b": COOLDOWN_KIND_WINDOW}}
    out = _merge_disk_cooldown_state(mem, disk, "openai-codex")
    assert out["model_cooldowns"] == {"a": 5e9, "b": 4e9}
    assert out["model_cooldown_kinds"] == {"a": COOLDOWN_KIND_ENTITLEMENT, "b": COOLDOWN_KIND_WINDOW}


def test_disk_merge_later_kindless_cooldown_clears_stale_entitlement_kind():
    from agent.credential_pool import PooledCredential
    from hermes_cli.auth import _merge_disk_cooldown_state

    now = time.time()
    disk = {"id": "x", "model_cooldowns": {"a": now + 3 * 86400}}  # later reset, no kind (legacy writer)
    mem = {"id": "x", "model_cooldowns": {"a": now + 86400}, "model_cooldown_kinds": {"a": COOLDOWN_KIND_ENTITLEMENT}}
    out = _merge_disk_cooldown_state(mem, disk, "openai-codex")
    assert out["model_cooldowns"] == {"a": pytest.approx(now + 3 * 86400)}
    # The kindless disk row is classed (3 days left -> window) before the merge: never the stale
    # in-memory "entitlement".
    assert out["model_cooldown_kinds"] == {"a": COOLDOWN_KIND_WINDOW}
    entry = PooledCredential.from_dict("openai-codex", {
        "label": "x", "auth_type": "oauth", "priority": 0, "source": "manual", "access_token": FAKE_KEY_A, **out,
    })
    assert model_cooldown_until(entry, None) is not None  # model-less callers stay blocked


def test_disk_merge_does_not_undo_migration():
    """Review finding: an un-migrated disk row must not win the merge back over its migrated form."""
    from hermes_cli.auth import _merge_disk_cooldown_state

    now = time.time()
    disk = {"id": "x", "model_cooldowns": {"m": now + 30 * 86400, CLAUDE_MODEL: now + 300 * 86400}}
    mem = {"id": "x", "model_cooldowns": {"m": now + MODEL_WINDOW_COOLDOWN_MAX_SECONDS},
           "model_cooldown_kinds": {"m": COOLDOWN_KIND_WINDOW}}
    out = _merge_disk_cooldown_state(mem, disk, "openai-codex")
    assert set(out["model_cooldowns"]) == {"m"}  # foreign row is not restored
    assert out["model_cooldowns"]["m"] <= time.time() + MODEL_WINDOW_COOLDOWN_MAX_SECONDS + 1
    assert out["model_cooldown_kinds"] == {"m": COOLDOWN_KIND_WINDOW}


def test_legacy_window_cap_is_fixed_once_not_sliding(tmp_path, monkeypatch):
    """Review finding: after load_pool persists, the 8-day cap must not move with each later load."""
    root = tmp_path / "hermes-root"
    root.mkdir()
    (tmp_path / "fakehome").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    monkeypatch.setenv("HERMES_HOME", str(root))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    now = time.time()
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"openai-codex": [
        {"id": "cred-0", "label": "acct-0", "auth_type": "oauth", "priority": 0, "source": "manual",
         "access_token": FAKE_KEY_A, "refresh_token": "rt-0", "expires_at_ms": 4_000_000_000_000,
         "model_cooldowns": {CODEX_MODEL: now + 30 * 86400, CLAUDE_MODEL: now + 300 * 86400}},
    ]}}))
    from agent.credential_pool import load_pool

    load_pool("openai-codex")
    stored = json.loads((root / "auth.json").read_text())["credential_pool"]["openai-codex"][0]
    first = stored["model_cooldowns"][CODEX_MODEL]
    assert set(stored["model_cooldowns"]) == {CODEX_MODEL}
    assert first <= now + MODEL_WINDOW_COOLDOWN_MAX_SECONDS + 60
    assert stored["model_cooldown_kinds"] == {CODEX_MODEL: COOLDOWN_KIND_WINDOW}
    _advance_clock(monkeypatch, 3 * 86400)
    entry = load_pool("openai-codex").entries()[0]
    assert (entry.model_cooldowns or {})[CODEX_MODEL] == pytest.approx(first)
    stored = json.loads((root / "auth.json").read_text())["credential_pool"]["openai-codex"][0]
    assert stored["model_cooldowns"][CODEX_MODEL] == pytest.approx(first)


def test_reset_clears_kinds(pool):
    pool.mark_exhausted_and_rotate(
        status_code=400, api_key_hint=FAKE_KEY_A, credential_id="cred-0", failure_reason="model_entitlement",
        model=CODEX_MODEL,
    )
    assert pool.reset_statuses() >= 1
    entry = pool.entries()[0]
    assert not entry.model_cooldowns and not entry.model_cooldown_kinds
