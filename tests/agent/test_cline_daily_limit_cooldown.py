"""Regression coverage for Cline daily-quota cooldown parsing."""

from __future__ import annotations

import time
from types import SimpleNamespace


def test_extract_api_error_context_parses_cline_daily_limit_wait():
    from agent.agent_runtime_helpers import extract_api_error_context

    error = SimpleNamespace(
        status_code=429,
        body={
            "error": {
                "message": (
                    "Daily free limit reached on model "
                    "deepseek/deepseek-v4-flash-0731. Try again in 23h 26m"
                )
            }
        },
        response=SimpleNamespace(headers={}),
    )

    before = time.time()
    context = extract_api_error_context(error)
    after = time.time()

    assert context["message"].startswith("Daily free limit reached")
    assert before + (23 * 3600 + 26 * 60) <= context["reset_at"] <= after + (23 * 3600 + 26 * 60)


def test_mark_exhausted_uses_message_reset_over_default_ttl(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))

    from agent.credential_pool import CredentialPool, PooledCredential

    pool = CredentialPool(
        "custom:cline",
        [
            PooledCredential(
                provider="custom:cline",
                id="cline-1",
                label="cline",
                auth_type="api_key",
                priority=0,
                source="test",
                access_token="key",
            )
        ],
    )
    before = time.time()
    pool.mark_exhausted_and_rotate(
        status_code=429,
        api_key_hint="key",
        error_context={
            "message": "Daily free limit reached. Try again in 23h 26m",
        },
    )
    after = time.time()

    entry = pool.entries()[0]
    assert entry.last_status == "exhausted"
    expected_wait = 23 * 3600 + 26 * 60
    assert entry.last_error_reset_at is not None
    assert before + expected_wait <= entry.last_error_reset_at <= after + expected_wait
    assert pool.has_available() is False
    assert pool.next_available_at() == entry.last_error_reset_at