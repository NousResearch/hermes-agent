"""Native Gemini quota scope and retry timing across the real recovery boundaries."""
from functools import partial
from types import SimpleNamespace
import time
from datetime import datetime, timezone

import httpx
import pytest
from openai import RateLimitError

from agent.agent_runtime_helpers import extract_api_error_context, recover_with_credential_pool
from agent.credential_pool import CredentialPool, PooledCredential, STATUS_EXHAUSTED, load_pool, _normalize_error_context
from agent.error_classifier import FailoverReason, classify_api_error
from agent.fallback_cooldown import _arm_rate_limit_cooldown
from agent.gemini_native_adapter import GeminiAPIError, GeminiNativeClient, gemini_http_error
from agent.turn_recovery import compute_error_backoff, recover_after_classification, route_classified_error
from agent.turn_retry_state import TurnRetryState

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-test-pro"


def quota_body(*, limit=0, scope="model", delay="3.6787s"):
    violation = {
        "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_input_token_count",
        "quotaId": "GenerateContentInputTokensPerModelPerMinute-FreeTier",
        "quotaDimensions": {"model": MODEL, "location": "global"},
        "quotaValue": str(limit),
    }
    violations = [violation]
    if scope == "mixed":
        violations.append({"quotaMetric": "project_requests", "quotaDimensions": {"project": "offline"}})
    if scope == "unknown":
        violation.pop("quotaDimensions")
    return {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": f"Quota exceeded for metric: generate_content_free_tier_input_token_count, limit: {limit}, model: {MODEL}. Please retry in 3.6787s.",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": violations},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
        ],
    }}


def recovery_agent(pool=None):
    agent = SimpleNamespace(
        provider="gemini", model=MODEL, api_mode="chat_completions", base_url=BASE_URL, _credential_pool=pool,
        _credential_pool_entry_id=None, api_key="offline-key", _fallback_activated=False,
        _primary_runtime={"provider": "gemini", "model": MODEL}, _rate_limit_backoff_count=3,
        _buffer_status=lambda message: None, _emit_status=lambda message: None,
        _client_log_context=lambda: "offline Gemini regression",
    )
    agent._recover_with_credential_pool = partial(recover_with_credential_pool, agent)
    return agent


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("scope,limit", [("model", 0), ("model", 250000), ("text", 0), ("text", 250000), ("mixed", 0), ("unknown", 0), ("malformed", 0), ("auth", 0), ("billing", 0)])
def test_quota_scope_survives_native_error_and_pool_recovery(tmp_path, monkeypatch, stream, scope, limit):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entries = [PooledCredential(
        provider="gemini", id=f"key-{i}", label=f"offline {i}", auth_type="api_key",
        priority=i, source="manual", access_token=f"offline-key-{i}",
    ) for i in range(7)]
    pool = CredentialPool("gemini", entries)
    pool._persist()
    agent = recovery_agent(pool)
    sent_keys = []

    def upstream(request):
        sent_keys.append(request.headers["x-goog-api-key"])
        if f"/models/{MODEL}:" in request.url.path:
            body = quota_body(limit=limit, scope=scope)
            status = 429
            if scope == "text":
                body["error"]["details"].pop(0)
            elif scope == "malformed":
                body["error"]["details"][0]["violations"] = ["invalid violation"]
            elif scope in {"auth", "billing"}:
                status = 401 if scope == "auth" else 402
                body["error"]["message"] = "Invalid API key" if status == 401 else "Insufficient credits"
            return httpx.Response(status, json=body)
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": "Flash still works"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 154803, "candidatesTokenCount": 3, "totalTokenCount": 154806},
        })

    for entry in entries:
        agent.api_key, agent._credential_pool_entry_id = entry.runtime_api_key, entry.id
        with GeminiNativeClient(api_key=entry.runtime_api_key, http_client=httpx.Client(transport=httpx.MockTransport(upstream))) as client:
            with pytest.raises(GeminiAPIError) as raised:
                result = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hello"}], stream=stream)
                if stream:
                    list(result)
            error = raised.value
            classified = classify_api_error(error, provider=agent.provider, model=MODEL)
            if scope not in {"model", "text"}:
                assert classified.reason != FailoverReason.upstream_rate_limit
                assert classified.should_rotate_credential
                if scope not in {"auth", "billing"}:
                    agent._swap_credential = lambda credential: True
                    agent._recover_with_credential_pool(
                        status_code=429, has_retried_429=True, classified_reason=classified.reason,
                        error_context=extract_api_error_context(error),
                    )
                    assert any(row.last_status == STATUS_EXHAUSTED for row in pool.entries())
                return
            assert classified.reason == FailoverReason.upstream_rate_limit
            assert not classified.should_rotate_credential
            assert classified.retryable is (limit != 0)
            recovered, _ = agent._recover_with_credential_pool(
                status_code=429, has_retried_429=True, classified_reason=classified.reason,
                error_context=extract_api_error_context(error),
            )
            assert not recovered
            # Exercise the actual eager-fallback decision, not a copied policy predicate.
            agent._fallback_index, agent._fallback_chain = 0, [{"provider": "gemini", "model": "gemini-test-flash"}]
            attempts = []
            agent._try_activate_fallback = lambda **kwargs: attempts.append(kwargs) or False
            for retry_count in (1, 3):
                attempts.clear()
                route_classified_error(
                    agent, error, classified, TurnRetryState(), error_msg=str(error),
                    error_context=extract_api_error_context(error), recovered_with_pool=False,
                    base_url=BASE_URL, model=MODEL, messages=[], api_messages=[], system_message=None,
                    active_system_prompt="", conversation_history=[], retry_count=retry_count,
                    max_retries=3, compression_attempts=0, max_compression_attempts=1,
                    api_call_count=retry_count, effective_task_id=None,
                )
                assert bool(attempts) is (limit == 0 or retry_count == 3)
            result = client.chat.completions.create(model="gemini-test-flash", messages=[{"role": "user", "content": "hello"}])
            assert result.choices[0].message.content == "Flash still works"
            assert result.usage.prompt_tokens == 154803
            assert sent_keys[-2:] == [entry.runtime_api_key, entry.runtime_api_key]
    # In-memory and persisted pool state stay usable for other models, including after reload.
    assert all(entry.last_status != STATUS_EXHAUSTED for entry in pool.entries())
    reloaded = load_pool("gemini")
    assert reloaded.has_available()
    assert {entry.id for entry in reloaded.entries()} >= {entry.id for entry in entries}
    assert all(entry.last_status != STATUS_EXHAUSTED for entry in reloaded.entries())


@pytest.mark.parametrize("retry_info,message_delay,header,expected", [
    (None, "3.6787", None, 3.6787),
    ("3s", "3.6787", None, 3.6787),
    ("28.91s", "28.91", "2", 28.91),
    ("3s", "3", "30", 30.0),
    ({"seconds": "3", "nanos": 678700000}, "3", None, 3.6787),
    ("NaNs", "25.57", None, 25.57),
    ("-5s", "27.86", "NaN", 27.86),
    ("750s", "750", None, 750.0),
    ("3s", "3", "Sat, 12 Sep 2026 00:00:30 GMT", 30.0),
])
def test_body_retry_floor_reaches_backoff_reset_and_fallback(monkeypatch, retry_info, message_delay, header, expected):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 12, tzinfo=timezone.utc)
    monkeypatch.setattr("agent.retry_utils.datetime", FixedDateTime)
    body = quota_body(limit=250000, delay=retry_info)
    body["error"]["message"] = f"A free_tier quota limited this request. Please retry in {message_delay}s."
    if retry_info is None:
        body["error"]["details"].pop()
    response = httpx.Response(429, json=body, headers={} if header is None else {"Retry-After": header})
    error = gemini_http_error(response)
    assert error.retry_after == pytest.approx(expected)
    assert "cannot sustain an agent session" not in str(error)
    assert "free tier is exhausted" not in str(error)
    agent = recovery_agent()
    wait = compute_error_backoff(
        agent, error, retry_count=1, max_retries=3, is_rate_limited=True,
        is_zai_coding_overload=False, base_url=BASE_URL, model=MODEL,
    )
    assert wait >= expected
    before = time.time()
    context = extract_api_error_context(error)
    after = time.time()
    assert before + expected <= context["reset_at"] <= after + expected
    # Both legacy text consumers must also understand Google's fractional 'retry in' wording.
    for parse in (extract_api_error_context, _normalize_error_context):
        raw = RuntimeError(f"Please retry in {message_delay}s") if parse is extract_api_error_context else {"message": f"Please retry in {message_delay}s"}
        before = time.time()
        context = parse(raw)
        after = time.time()
        assert before + float(message_delay) <= context["reset_at"] <= after + float(message_delay)
    classified = classify_api_error(error, provider="gemini", model=MODEL)
    # Native transport semantics also survive a custom/unresolved provider label.
    assert classify_api_error(error, provider="custom-gemini-relay", model=MODEL).reason == classified.reason
    recover_after_classification(
        agent, error, classified, TurnRetryState(), status_code=429,
        error_context=extract_api_error_context(error), messages=[], api_messages=[],
    )
    # Falling back after retries must not convert a provider delay into level-3 / 480s cooldown.
    cooldown = _arm_rate_limit_cooldown(agent, classified.reason)
    assert cooldown is not None and cooldown > 0
    # Absolute monotonic deadline arithmetic can overshoot the source delay by a few ULPs.
    assert cooldown <= expected + 1e-6
    # The recorded deadline belongs to the failed model, not another same-provider fallback.
    agent.model = "gemini-test-flash"
    agent._rate_limit_backoff_count = 0
    assert _arm_rate_limit_cooldown(agent, classified.reason) != cooldown


def test_sdk_gemini_retryinfo_reaches_actual_backoff():
    body = quota_body(limit=250000, delay="30s")
    response = httpx.Response(
        429, json=body, headers={"Retry-After": "2"},
        request=httpx.Request("POST", BASE_URL + "/openai/chat/completions"),
    )
    error = RateLimitError("Gemini quota", response=response, body=body)
    classified = classify_api_error(error, provider="gemini", model=MODEL)
    assert classified.reason == FailoverReason.upstream_rate_limit
    assert classified.error_context["retry_after"] == pytest.approx(30.0)
    wait = compute_error_backoff(
        recovery_agent(), error, retry_count=1, max_retries=3,
        is_rate_limited=True, is_zai_coding_overload=False,
        base_url=BASE_URL + "/openai", model=MODEL,
        error_context=classified.error_context,
    )
    assert wait == pytest.approx(30.0)
