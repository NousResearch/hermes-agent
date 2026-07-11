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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_anthropic_env(monkeypatch, tmp_path):
    """Keep host-machine Anthropic credentials out of the seeded pool."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    # A real ~/.claude/.credentials.json would seed an extra claude_code entry.
    monkeypatch.setenv("HOME", str(tmp_path))
    from agent import credential_pool

    with credential_pool._MODEL_COOLDOWN_LOCK:
        credential_pool._MODEL_COOLDOWNS.clear()


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


def test_model_scoped_cooldown_survives_pool_reload(tmp_path, monkeypatch):
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

    # Auxiliary client eviction reloads the pool.  The process-local registry
    # must keep the attributed cooldown without changing auth.json's schema.
    reloaded = load_pool("anthropic")
    assert all(e.last_status is None for e in reloaded.entries())
    assert [
        e.id for e in reloaded._available_entries(model="anthropic/CLAUDE-FABLE-5")
    ] == ["cred-2"]
    assert [e.id for e in reloaded._available_entries(model="claude-opus-4-8")] == [
        "cred-1",
        "cred-2",
    ]


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
    from agent.credential_pool import _MODEL_COOLDOWNS

    assert not _MODEL_COOLDOWNS


def test_public_selection_seams_honor_canonical_model_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _two_entry_anthropic_store(tmp_path)

    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    assert pool.select(model="Anthropic/Claude-Fable-5").id == "cred-1"
    pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"reset_at": time.time() + 3600},
        model="Anthropic/Claude-Fable-5",
        model_scoped=True,
    )

    reloaded = load_pool("anthropic")
    assert reloaded.has_available(model="claude-fable-5") is True
    assert reloaded.select(model="claude-fable-5").id == "cred-2"
    assert reloaded.peek(model="CLAUDE-FABLE-5").id == "cred-2"


def test_main_recovery_attributes_anthropic_429_and_survives_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _two_entry_anthropic_store(tmp_path)

    from agent.agent_runtime_helpers import recover_with_credential_pool
    from agent.credential_pool import load_pool

    pool = load_pool("anthropic")
    first = pool.select(model="claude-fable-5")
    agent = SimpleNamespace(
        _credential_pool=pool,
        provider="anthropic",
        model="anthropic/claude-fable-5",
        _swap_credential=MagicMock(),
    )

    recovered, retried = recover_with_credential_pool(
        agent,
        status_code=429,
        has_retried_429=True,
        error_context={
            "reset_at": time.time() + 3600,
            "anthropic_unified_utilizations": [0.1, 0.4],
        },
    )

    assert recovered is True
    assert retried is False
    assert first.id == "cred-1"
    assert agent._swap_credential.call_args.args[0].id == "cred-2"
    reloaded = load_pool("anthropic")
    assert reloaded.select(model="claude-fable-5").id == "cred-2"


def test_auxiliary_recovery_evicts_client_without_losing_model_cooldown(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _two_entry_anthropic_store(tmp_path)

    from agent import auxiliary_client
    from agent.credential_pool import load_pool

    class _Response:
        headers = {
            "anthropic-ratelimit-unified-5h-utilization": "0.10",
            "anthropic-ratelimit-unified-7d-utilization": "0.40",
            "x-ratelimit-reset": str(time.time() + 7200),
        }

    class _RateLimitError(Exception):
        status_code = 429
        response = _Response()

    with patch.object(auxiliary_client, "_evict_cached_clients") as evict:
        assert auxiliary_client._recover_provider_pool(
            "anthropic",
            _RateLimitError("rate limited"),
            model="Anthropic/Claude-Fable-5",
        ) is True

    evict.assert_called_once_with("anthropic")
    from agent.credential_pool import _MODEL_COOLDOWNS

    assert next(iter(_MODEL_COOLDOWNS.values())) > time.time() + 7100
    # The client rebuild loads a new pool instance. It must choose cred-2 for
    # the same canonical model while leaving cred-1 usable by Opus.
    reloaded = load_pool("anthropic")
    assert reloaded.select(model="claude-fable-5").id == "cred-2"
    other_model_pool = load_pool("anthropic")
    assert other_model_pool.select(model="claude-opus-4-8").id == "cred-1"


@pytest.mark.parametrize("status_code", [401, 402, 403])
def test_non_429_failures_remain_credential_wide(tmp_path, monkeypatch, status_code):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    _two_entry_anthropic_store(tmp_path)

    from agent.credential_pool import STATUS_EXHAUSTED, load_pool

    pool = load_pool("anthropic")
    pool.select(model="claude-fable-5")
    pool.mark_exhausted_and_rotate(
        status_code=status_code,
        error_context=None,
        model="claude-fable-5",
        model_scoped=True,
    )
    first = next(entry for entry in pool.entries() if entry.id == "cred-1")
    assert first.last_status == STATUS_EXHAUSTED


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
