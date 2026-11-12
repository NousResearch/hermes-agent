"""OpenCode Go rate limits are model-scoped when the provider classifies them as rate_limit.

A 429 on beam-free must not bench the same GO credential for iceman-contributor-free.
Confirmed billing/entitlement failures remain credential-wide or explicitly scoped.
"""
from __future__ import annotations

import json

import pytest

KEY = "oc_sk_test-only"


@pytest.fixture
def go_pool(tmp_path, monkeypatch):
    root = tmp_path / "hermes-root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    for var in ("OPENCODE_GO_API_KEY", "OPENCODE_ZEN_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None  # type: ignore[attr-defined]
    (root / "auth.json").write_text(json.dumps({"credential_pool": {"opencode-go": [{
        "id": "go-seat", "label": "go-seat", "auth_type": "api_key", "priority": 0,
        "source": "manual", "access_token": KEY,
        "base_url": "https://opencode.ai/zen/go/v1",
    }]}}))
    from agent.credential_pool import load_pool
    return load_pool("opencode-go")


def test_rate_limit_on_one_go_model_keeps_sibling_model_usable(go_pool):
    """A generic GO model-rate-limit must cool only that model, not the subscription key."""
    from agent.credential_pool import STATUS_EXHAUSTED

    next_entry = go_pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"message": "Rate limit exceeded. Please try again later."},
        api_key_hint=KEY,
        failure_reason="rate_limit",
        model="beam-free",
    )

    assert next_entry is None  # the only key is cooled for beam-free
    entry = go_pool.entries()[0]
    assert entry.last_status != STATUS_EXHAUSTED  # the credential itself remains healthy
    assert go_pool.select(model="beam-free") is None
    assert go_pool.select(model="iceman-contributor-free") is not None


def test_confirmed_go_billing_429_still_benches_the_credential(go_pool):
    """A billing/quota 429 is account-wide; model scoping must not hide it."""
    from agent.credential_pool import STATUS_EXHAUSTED

    go_pool.mark_exhausted_and_rotate(
        status_code=429,
        error_context={"message": "Monthly allowance exhausted"},
        api_key_hint=KEY,
        failure_reason="billing",
        model="beam-free",
    )

    entry = go_pool.entries()[0]
    assert entry.last_status == STATUS_EXHAUSTED
    assert not entry.model_cooldowns
    assert go_pool.select(model="iceman-contributor-free") is None
