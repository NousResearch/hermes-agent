"""Tests for model-scoped exhaustion in the credential pool.

A 429 rate limit on a known model must bench only the (key, model) pair so
the key stays selectable for other models.  Callers that do not pass a
model keep the exact legacy key-level behavior.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Optional

import pytest


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _two_key_pool_payload() -> dict:
    """Two healthy manual api_key entries with distinct access tokens."""
    return {
        "version": 1,
        "credential_pool": {
            "custom": [
                {
                    "id": "cred-a",
                    "label": "key-a",
                    "auth_type": "api_key",
                    "priority": 0,
                    "source": "manual",
                    "access_token": "tok-a",
                    "last_status": "ok",
                    "last_status_at": None,
                    "last_error_code": None,
                },
                {
                    "id": "cred-b",
                    "label": "key-b",
                    "auth_type": "api_key",
                    "priority": 1,
                    "source": "manual",
                    "access_token": "tok-b",
                    "last_status": "ok",
                    "last_status_at": None,
                    "last_error_code": None,
                },
            ]
        },
    }


def _load_pool(tmp_path, monkeypatch, payload: Optional[dict] = None):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(tmp_path, payload or _two_key_pool_payload())
    from agent.credential_pool import load_pool

    return load_pool("custom")


def _entry(pool, entry_id: str):
    return next(e for e in pool.entries() if e.id == entry_id)


def _selected_id(pool, model=None) -> str:
    """select() returns Optional; tests here always expect an entry."""
    entry = pool.select(model=model)
    assert entry is not None
    return entry.id


def test_model_scoped_429_benches_pair_only_and_rotates_for_that_model(tmp_path, monkeypatch):
    """A 429 with a known model must not flip the key-level status.

    The failing key is benched only for that model; rotation for the model
    lands on the other key, and the failing key stays selectable for other
    models.
    """
    pool = _load_pool(tmp_path, monkeypatch)
    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        api_key_hint="tok-a",
        model="gemini-3.5-flash-lite",
    )
    assert next_entry is not None
    assert next_entry.id == "cred-b"

    key_a = _entry(pool, "cred-a")
    # Key-level status untouched: the key is not exhausted for other models.
    assert key_a.last_status == "ok"
    bench = key_a.model_exhaustions["gemini-3.5-flash-lite"]
    assert bench["status_code"] == 429
    assert bench["until"] > time.time()

    # The benched model is routed to the other key...
    assert _selected_id(pool, "gemini-3.5-flash-lite") == "cred-b"
    # ...but another model may still use key A.
    assert _selected_id(pool, "gemini-3.5-pro") == "cred-a"


def test_select_skips_model_benched_key_for_that_model_only(tmp_path, monkeypatch):
    """Active per-model benching blocks only that model, not others."""
    payload = _two_key_pool_payload()
    payload["credential_pool"]["custom"][0]["model_exhaustions"] = {
        "gemini-x": {
            "until": time.time() + 3600,
            "status_code": 429,
            "reason": None,
            "reset_at": None,
        }
    }
    pool = _load_pool(tmp_path, monkeypatch, payload)

    assert _selected_id(pool, "gemini-x") == "cred-b"
    assert _selected_id(pool, "gemini-y") == "cred-a"
    # No-model selection ignores per-model benchings entirely.
    assert _selected_id(pool) == "cred-a"

def test_model_429_prefers_provider_reset_at_over_default_ttl(tmp_path, monkeypatch):
    """The provider's reset_at from error_context wins over the 1h TTL."""
    pool = _load_pool(tmp_path, monkeypatch)
    reset_at = time.time() + 4 * 3600  # clearly beyond the 1h default TTL

    pool.mark_exhausted_and_rotate(
        status_code=429,
        api_key_hint="tok-a",
        model="gemini-x",
        error_context={"reset_at": reset_at, "reason": "rate_limit_exceeded"},
    )

    bench = _entry(pool, "cred-a").model_exhaustions["gemini-x"]
    assert bench["until"] == pytest.approx(reset_at, abs=2)
    assert bench["reset_at"] == pytest.approx(reset_at, abs=2)
    assert bench["reason"] == "rate_limit_exceeded"


def test_expired_model_bench_clears_and_is_pruned(tmp_path, monkeypatch):
    """A past-until bench no longer blocks selection and is pruned on clear."""
    payload = _two_key_pool_payload()
    payload["credential_pool"]["custom"][0]["model_exhaustions"] = {
        "gemini-x": {
            "until": time.time() - 60,
            "status_code": 429,
            "reason": None,
            "reset_at": None,
        }
    }
    pool = _load_pool(tmp_path, monkeypatch, payload)

    # The expired bench no longer blocks the model.
    assert _selected_id(pool, "gemini-x") == "cred-a"

    # Selection runs with clear_expired=True, which pruned the stale entry.
    key_a = _entry(pool, "cred-a")
    assert "gemini-x" not in key_a.model_exhaustions


def test_model_bench_survives_persist_and_reload(tmp_path, monkeypatch):
    """Model benchings round-trip through auth.json with a real reload."""
    pool = _load_pool(tmp_path, monkeypatch)
    pool.mark_exhausted_and_rotate(
        status_code=429,
        api_key_hint="tok-a",
        model="gemini-x",
    )

    stored = json.loads((tmp_path / "hermes" / "auth.json").read_text(encoding="utf-8"))
    stored_a = next(
        e for e in stored["credential_pool"]["custom"] if e["id"] == "cred-a"
    )
    assert stored_a["model_exhaustions"]["gemini-x"]["status_code"] == 429

    from agent.credential_pool import load_pool

    pool2 = load_pool("custom")
    assert _entry(pool2, "cred-a").model_exhaustions["gemini-x"]["until"] > time.time()
    assert _selected_id(pool2, "gemini-x") == "cred-b"
    assert _selected_id(pool2, "gemini-y") == "cred-a"


def test_auth_reset_clears_model_exhaustions(tmp_path, monkeypatch):
    """`hermes auth reset` clears per-model benchings for every entry."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _write_auth_store(
        tmp_path,
        {
            "version": 1,
            "credential_pool": {
                "anthropic": [
                    {
                        "id": "cred-1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "tok-1",
                        "last_status": "ok",
                        "last_status_at": None,
                        "last_error_code": None,
                    },
                    {
                        "id": "cred-2",
                        "label": "secondary",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "tok-2",
                        "last_status": "ok",
                        "last_status_at": None,
                        "last_error_code": None,
                    },
                ]
            },
        },
    )
    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    pool.mark_exhausted_and_rotate(
        status_code=429,
        api_key_hint="tok-1",
        model="gemini-x",
    )
    assert _entry(pool, "cred-1").model_exhaustions

    from hermes_cli.auth_commands import auth_reset_command

    auth_reset_command(SimpleNamespace(provider="anthropic"))

    pool2 = load_pool("anthropic")
    assert _entry(pool2, "cred-1").model_exhaustions == {}
    # The key is usable again for the previously benched model.
    assert _selected_id(pool2, "gemini-x") == "cred-1"


def test_429_without_model_keeps_key_level_behavior(tmp_path, monkeypatch):
    """model=None callers keep the exact legacy key-level exhaustion."""
    pool = _load_pool(tmp_path, monkeypatch)
    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        api_key_hint="tok-a",
    )
    assert next_entry is not None
    assert next_entry.id == "cred-b"

    key_a = _entry(pool, "cred-a")
    assert key_a.last_status == "exhausted"
    assert key_a.model_exhaustions == {}

    # Key-level exhaustion blocks the key for every model.
    assert _selected_id(pool) == "cred-b"
    assert _selected_id(pool, "gemini-x") == "cred-b"
    assert _selected_id(pool, "gemini-y") == "cred-b"


def test_non_429_with_model_keeps_key_level_behavior(tmp_path, monkeypatch):
    """Only 429 with a model benches per-model; 401 stays key-level."""
    pool = _load_pool(tmp_path, monkeypatch)
    pool.mark_exhausted_and_rotate(
        status_code=401,
        api_key_hint="tok-a",
        model="gemini-x",
    )

    key_a = _entry(pool, "cred-a")
    assert key_a.last_status == "exhausted"
    assert key_a.model_exhaustions == {}
    assert _selected_id(pool, "gemini-x") == "cred-b"


def test_model_bench_dominates_when_key_level_cooldown_expired(tmp_path, monkeypatch):
    """The later of key-level cooldown and per-model bench gates selection.

    Key A's key-level 429 cooldown has elapsed, so a plain selection would
    reset it to OK; the still-active per-model bench for gemini-x must keep
    blocking that model while other models see the key as healthy.
    """
    payload = _two_key_pool_payload()
    entry_a = payload["credential_pool"]["custom"][0]
    entry_a["last_status"] = "exhausted"
    entry_a["last_status_at"] = time.time() - 4000  # beyond the 1h TTL
    entry_a["last_error_code"] = 429
    entry_a["model_exhaustions"] = {
        "gemini-x": {
            "until": time.time() + 3600,
            "status_code": 429,
            "reason": None,
            "reset_at": None,
        }
    }
    pool = _load_pool(tmp_path, monkeypatch, payload)

    assert _selected_id(pool, "gemini-x") == "cred-b"
    assert _selected_id(pool, "gemini-y") == "cred-a"


def test_resolve_provider_client_threads_model_into_openrouter_select(monkeypatch):
    """The main openrouter client-build path must select the pool with the
    model so per-model benches are honored at selection time, not only at
    rotation time (review fix: model was in scope but not passed)."""
    from agent import auxiliary_client as mod

    captured = {}

    def fake_try_openrouter(*, explicit_api_key=None, model=None):
        captured["model"] = model
        return object(), "some-default-model"

    monkeypatch.setattr(mod, "_try_openrouter", fake_try_openrouter)

    client, final_model = mod.resolve_provider_client(
        "openrouter", "google/gemini-3-flash-preview", False,
    )
    assert client is not None
    assert captured["model"] == "google/gemini-3-flash-preview"
    assert final_model is not None


def test_per_model_bench_propagates_to_same_key_siblings(tmp_path, monkeypatch):
    """A per-model bench must cover every entry sharing the failed key.

    Duplicate entries (an explicit pool entry plus a model_config entry
    auto-seeded from the same model.api_key) carry the identical runtime
    key. Benching only the matched entry would let model-aware rotation
    reselect the same depleted key through its sibling.
    """
    payload = _two_key_pool_payload()
    # cred-b shares cred-a's runtime key.
    payload["credential_pool"]["custom"][1]["access_token"] = "tok-a"
    pool = _load_pool(tmp_path, monkeypatch, payload)

    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        api_key_hint="tok-a",
        model="gemini-x",
    )
    # Both entries benched for gemini-x: nothing left to rotate to.
    assert next_entry is None

    key_a = _entry(pool, "cred-a")
    key_b = _entry(pool, "cred-b")
    assert key_a.last_status == "ok"
    assert key_b.last_status == "ok"
    assert "gemini-x" in key_a.model_exhaustions
    assert "gemini-x" in key_b.model_exhaustions
    # No entry selectable for the benched model...
    assert pool.select(model="gemini-x") is None
    # ...but both remain selectable for other models.
    assert _selected_id(pool, "gemini-y") in {"cred-a", "cred-b"}
