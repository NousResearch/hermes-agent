"""A model-level Zen refusal must not invalidate a working credential (#124021)."""
import json
from types import SimpleNamespace

import openai
import httpx
import pytest

from agent.agent_runtime_helpers import extract_api_error_context, recover_with_credential_pool
from agent.credential_pool import load_pool
from agent.error_classifier import FailoverReason, classify_api_error


BASE = "https://opencode.ai/zen"
MODEL = "claude-restricted"


def refusal(message="Upstream request failed: Model access is disabled", status=403):
    body = {"type": "error", "error": {"type": "api_error", "message": message}}
    response = httpx.Response(status, request=httpx.Request("POST", BASE + "/v1/messages"), json=body)
    return openai.PermissionDeniedError(message, response=response, body=body)


@pytest.mark.parametrize("count", [1, 2])
def test_model_refusal_preserves_other_models_after_pool_reload(tmp_path, monkeypatch, count):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    entries = [
        {"id": f"key-{i}", "label": f"key-{i}", "auth_type": "api_key", "priority": i,
         "source": "manual", "access_token": f"fixture-key-{i}", "base_url": BASE}
        for i in range(count)
    ]
    (home / "auth.json").write_text(json.dumps({"credential_pool": {"opencode-zen": entries}}))
    pool = load_pool("opencode-zen")
    first = pool.select(model=MODEL)
    assert first is not None
    swaps = []
    agent = SimpleNamespace(
        provider="opencode-zen", model=MODEL, base_url=BASE,
        api_key=first.runtime_api_key, _credential_pool=pool, _credential_pool_entry_id=first.id,
        _swap_credential=lambda entry: swaps.append(entry.id),
    )
    error = refusal()
    verdict = classify_api_error(error, provider=agent.provider, model=MODEL, base_url=BASE)
    assert verdict.reason == FailoverReason.model_entitlement
    assert verdict.should_fallback and verdict.should_rotate_credential and not verdict.retryable
    recovered, _ = recover_with_credential_pool(
        agent, status_code=403, has_retried_429=False,
        classified_reason=verdict.reason, error_context=extract_api_error_context(error),
    )
    assert recovered is (count > 1)
    assert swaps == (["key-1"] if count > 1 else [])
    reloaded = load_pool("opencode-zen")
    failed = next(entry for entry in reloaded.entries() if entry.id == first.id)
    assert failed.last_status is None
    assert failed.failure_reason is None
    assert MODEL in failed.model_cooldowns
    assert reloaded.select(model="claude-available").id == first.id
    selected = reloaded.select(model=MODEL)
    assert (selected.id if selected else None) == ("key-1" if count > 1 else None)


@pytest.mark.parametrize("provider,message,status,expected", [
    ("opencode-zen", "Model access is disabled", 403, FailoverReason.model_entitlement),
    ("zen", "Model access is disabled", 403, FailoverReason.model_entitlement),
    ("opencode-go", "Model access is disabled", 403, FailoverReason.auth),
    ("opencode-zen", "Invalid API key", 403, FailoverReason.auth),
    ("opencode-zen", "Account access is disabled", 403, FailoverReason.auth),
    ("opencode-zen", "Model access is disabled; insufficient credits", 403, FailoverReason.billing),
    ("opencode-zen", "Model access is disabled; request blocked", 403, FailoverReason.upstream_blocked),
    ("opencode-zen", "Model access is disabled", 401, FailoverReason.auth),
    ("anthropic", "Model access is disabled", 403, FailoverReason.auth),
])
def test_unrelated_refusals_keep_their_verdict(provider, message, status, expected):
    verdict = classify_api_error(refusal(message, status), provider=provider, model=MODEL)
    assert verdict.reason == expected
