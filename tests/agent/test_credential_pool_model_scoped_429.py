"""Model-scoped 429 cooldowns (issue #61451).

A 429 that is attributable to a model-specific quota bucket (e.g. the
separate Fable/Mythos-class quota on Anthropic subscriptions) must not park
the whole credential: other models whose shared unified budget is still
free should keep working. These tests cover the (credential, model)
cooldown path and the header heuristic that decides attribution.
"""

from __future__ import annotations

import json
import time

import pytest


@pytest.fixture(autouse=True)
def _clean_anthropic_env(monkeypatch, tmp_path):
    """Keep host-machine Anthropic credentials out of the seeded pool."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    # A real ~/.claude/.credentials.json would seed an extra claude_code entry.
    monkeypatch.setenv("HOME", str(tmp_path))


def _write_auth_store(tmp_path, payload: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "auth.json").write_text(json.dumps(payload, indent=2))


def _two_entry_anthropic_store(tmp_path) -> None:
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
                        "access_token": "***",
                    },
                    {
                        "id": "cred-2",
                        "label": "secondary",
                        "auth_type": "api_key",
                        "priority": 1,
                        "source": "manual",
                        "access_token": "***",
                    },
                ]
            },
        },
    )


def test_model_scoped_429_cools_only_that_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _two_entry_anthropic_store(tmp_path)

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    assert pool.select().id == "cred-1"

    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reset_at": time.time() + 3600},
        model="claude-fable-5",
        model_scoped=True,
    )

    # Rotates to the second credential for the rate-limited model...
    assert next_entry is not None
    assert next_entry.id == "cred-2"
    # ...but cred-1 is NOT exhausted: it stays available for other models.
    entry_1 = next(e for e in pool.entries() if e.id == "cred-1")
    assert entry_1.last_status is None
    assert pool._available_entries(model="claude-opus-4-8")[0].id == "cred-1"
    # For the rate-limited model itself, cred-1 is skipped.
    fable_available = pool._available_entries(model="claude-fable-5")
    assert [e.id for e in fable_available] == ["cred-2"]


def test_model_scoped_cooldown_not_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _two_entry_anthropic_store(tmp_path)

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    pool.select()
    pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context=None,
        model="claude-fable-5",
        model_scoped=True,
    )

    # A freshly loaded pool (e.g. another process) sees no exhaustion at all —
    # the cooldown is in-memory by design, no PooledCredential schema change.
    reloaded = load_pool("anthropic")
    assert all(e.last_status is None for e in reloaded.entries())


def test_model_scoped_cooldown_expires(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _two_entry_anthropic_store(tmp_path)

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    pool.select()
    pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reset_at": time.time() - 1},  # already elapsed
        model="claude-fable-5",
        model_scoped=True,
    )
    available = pool._available_entries(model="claude-fable-5")
    assert [e.id for e in available] == ["cred-1", "cred-2"]
    assert not pool._model_cooldowns


def test_all_credentials_cooled_for_model_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _two_entry_anthropic_store(tmp_path)

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    pool.select()
    reset_at = time.time() + 3600
    assert (
        pool.mark_exhausted_and_rotate(
            status_code=429,
            error_context={"reset_at": reset_at},
            model="claude-fable-5",
            model_scoped=True,
        ).id
        == "cred-2"
    )
    # Second credential trips the same model bucket → nothing left for
    # that model, but the pool is still fully available for others.
    assert (
        pool.mark_exhausted_and_rotate(
            status_code=429,
            error_context={"reset_at": reset_at},
            model="claude-fable-5",
            model_scoped=True,
        )
        is None
    )
    assert [e.id for e in pool._available_entries(model="claude-opus-4-8")] == [
        "cred-1",
        "cred-2",
    ]


def test_credential_wide_429_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _two_entry_anthropic_store(tmp_path)

    from agent.credential_pool import STATUS_EXHAUSTED, load_pool

    pool = load_pool("anthropic")
    pool.select()
    next_entry = pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context=None,
        model="claude-fable-5",
        model_scoped=False,
    )
    assert next_entry.id == "cred-2"
    entry_1 = next(e for e in pool.entries() if e.id == "cred-1")
    assert entry_1.last_status == STATUS_EXHAUSTED


class _Headers(dict):
    """Case-preserving dict standing in for httpx.Headers in tests."""


@pytest.mark.parametrize(
    "headers,expected",
    [
        # Unified budgets clearly free → the 429 must be model-scoped.
        ({"anthropic-ratelimit-unified-5h-utilization": "0.10",
          "anthropic-ratelimit-unified-7d-utilization": "0.48"}, True),
        # Percent-style values are tolerated.
        ({"anthropic-ratelimit-unified-5h-utilization": "10%",
          "anthropic-ratelimit-unified-7d-utilization": "48"}, True),
        # A unified budget at/over the ceiling → credential-wide.
        ({"anthropic-ratelimit-unified-5h-utilization": "1.0",
          "anthropic-ratelimit-unified-7d-utilization": "0.48"}, False),
        # No relevant headers → stay conservative (credential-wide).
        ({}, False),
        ({"x-ratelimit-reset": "123"}, False),
        # Unparseable values are ignored; nothing parseable → credential-wide.
        ({"anthropic-ratelimit-unified-5h-utilization": "garbage"}, False),
    ],
)
def test_is_model_scoped_429_headers(headers, expected):
    from agent.credential_pool import is_model_scoped_429

    assert is_model_scoped_429(_Headers(headers)) is expected


def test_is_model_scoped_429_none_headers():
    from agent.credential_pool import is_model_scoped_429

    assert is_model_scoped_429(None) is False


def test_is_model_scoped_429_context():
    from agent.credential_pool import is_model_scoped_429_context

    assert is_model_scoped_429_context(
        {"anthropic_unified_utilizations": [0.10, 0.48]}
    ) is True
    assert is_model_scoped_429_context(
        {"anthropic_unified_utilizations": [0.10, 1.0]}
    ) is False
    assert is_model_scoped_429_context({}) is False
    assert is_model_scoped_429_context(None) is False


def test_extract_api_error_context_captures_unified_utilizations():
    from agent.agent_runtime_helpers import extract_api_error_context

    class _Response:
        headers = _Headers(
            {
                "anthropic-ratelimit-unified-5h-utilization": "0.10",
                "anthropic-ratelimit-unified-7d-utilization": "48%",
            }
        )

    class _Error(Exception):
        response = _Response()

    context = extract_api_error_context(_Error("rate limited"))
    assert context["anthropic_unified_utilizations"] == [0.10, 0.48]
