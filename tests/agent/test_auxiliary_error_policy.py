"""Auxiliary recovery decisions come from the shared classifier, driven end to end through ``call_llm``."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from agent.auxiliary_client import _try_openrouter, async_call_llm, call_llm
from agent.credential_pool import CredentialPool, PooledCredential as Key
from agent.error_classifier import FailoverReason
from agent.errors import InvalidProviderResponseError
from tests.agent.conftest import sdk_status_error as sdk

_REQ = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
_MESSAGES = [{"role": "user", "content": "hi"}]


class _Plain(Exception):
    pass


def _ok():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fallback"))])


def run_ladder(error, *, provider="openrouter", retries=1, is_async=False, fallback_error=None, **call_kw):
    """``call_llm`` / ``async_call_llm`` on a client whose requests raise ``error`` (a list: one per request)."""
    create = AsyncMock if is_async else MagicMock
    primary = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="sk-primary")
    primary.chat.completions.create = create(side_effect=error)
    fallback = MagicMock(base_url="https://api.example.com/v1")
    fallback.chat.completions.create = create(side_effect=fallback_error, return_value=_ok())
    marks, rotations, refreshes = [], [], []
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=(provider, "m", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "m")),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
        patch("agent.auxiliary_client._transient_retry_count", return_value=retries),
        patch("agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value="openrouter"),
        patch("agent.auxiliary_client._recover_provider_pool", side_effect=lambda _p, e, **_k: rotations.append(e)),
        patch("agent.auxiliary_client._mark_provider_unhealthy", side_effect=lambda *a, **kw: marks.append(a)),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_main_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_payment_fallback", return_value=(fallback, "fb", "fallback")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(fallback, "fb", "fallback")),
        patch("agent.auxiliary_rate_limit_backoff.rate_limit_wait", return_value=0.0),
        patch("agent.auxiliary_client._refresh_provider_credentials",
              side_effect=lambda slug, **_kw: refreshes.append(slug) or True),
    ):
        try:
            if is_async:
                result = asyncio.run(async_call_llm(messages=_MESSAGES, **call_kw))
            else:
                result = call_llm(messages=_MESSAGES, **call_kw)
        except Exception as exc:  # the outcome under test
            result = exc
    return SimpleNamespace(attempts=primary.chat.completions.create.call_count,
                           fallback_attempts=fallback.chat.completions.create.call_count,
                           fell_back=fallback.chat.completions.create.call_count > 0,
                           unhealthy=bool(marks), rotated=bool(rotations), refreshed=refreshes, result=result)


# (case id, error, route provider, expected outcome): the approved behaviour decisions.
BEHAVIOUR = [
    ("dns-refused", _Plain("[Errno 8] nodename nor servname provided, or not known"), "openrouter",
     dict(attempts=2, fell_back=True)),
    ("http-500", sdk(500, "Internal Server Error"), "openrouter", dict(attempts=2, fell_back=True)),
    ("http-503", sdk(503, "Service Unavailable"), "openrouter", dict(attempts=2, fell_back=True)),
    ("http-408", sdk(408, "Request Timeout"), "openrouter", dict(attempts=2, fell_back=True)),
    ("429-overloaded", sdk(429, "The service may be temporarily overloaded, please try again later"),
     "openrouter", dict(attempts=2, fell_back=True, rotated=False)),
    ("404-model-not-found", sdk(404, "model_not_found: The model 'x' does not exist"), "openrouter",
     dict(attempts=1, fell_back=True)),
    ("404-free-tier-model", sdk(404, "model not found (free tier)"), "openrouter", dict(attempts=1, fell_back=True)),
    ("400-billing-body", sdk(400, "Insufficient credits. Add more using https://openrouter.ai/credits"),
     "openrouter", dict(attempts=1, fell_back=True, unhealthy=True, rotated=True)),
    ("429-daily-window", sdk(429, "RESOURCE_EXHAUSTED: You have reached your daily limit."), "openrouter",
     dict(fell_back=True, unhealthy=True)),
    ("429-resource-exhausted", sdk(429, "RESOURCE_EXHAUSTED: Worker local total request limit reached (32/32)"),
     "openrouter", dict(fell_back=True, rotated=True, unhealthy=False)),
    ("429-rate-limit-with-billing-words", sdk(429, "Rate limit exceeded: insufficient credits for burst"),
     "openrouter", dict(fell_back=True, rotated=True, unhealthy=False)),
    ("403-generic-auto", sdk(403, "Forbidden"), "auto", dict(attempts=1, fell_back=True, rotated=False)),
    ("403-quota", sdk(403, "Quota exceeded for this key"), "openrouter", dict(fell_back=True, unhealthy=True)),
    ("xai-bad-credentials", sdk(403, "unauthenticated:bad-credentials"), "xai-oauth",
     dict(refreshed=["xai-oauth"])),
    ("account-gated-model", sdk(400, "model kimi-k2 is not supported for this account"), "openrouter",
     dict(attempts=1, fell_back=True, rotated=True)),
    ("feature-restriction", sdk(400, "Tool use is not supported with this model"), "openrouter",
     dict(attempts=1, fell_back=True, rotated=False)),
    ("parameter-400-no-rung", sdk(400, "Unsupported parameter: 'temperature' is not supported with this model."),
     "openrouter", dict(attempts=1, fell_back=False, rotated=False)),
    ("statusless-account-wording", openai.APIError("model x-1 is not supported when using this account", _REQ,
                                                   body={"message": "gated"}), "openrouter",
     dict(fell_back=True, rotated=True)),
    ("statusless-data-policy-denied", openai.APIError("denied", _REQ, body={"code": "data_policy_denied"}), "nous",
     dict(attempts=1, fell_back=False)),
    ("statusless-sse-404-body", openai.APIError("Not Found", _REQ, body={"code": 404, "message": "Not Found"}),
     "openrouter", dict(attempts=1, fell_back=True)),
    ("response-ended-prematurely", _Plain("RemoteProtocolError: response ended prematurely"), "openrouter",
     dict(attempts=2, fell_back=True)),
    ("openrouter-upstream-429", sdk(429, "Provider returned error", wrapped=True, code=429,
                                    metadata={"raw": "rate-limited upstream", "provider_name": "DeepInfra"}),
     "openrouter", dict(attempts=1, fell_back=True, rotated=False)),
    # An interrupted task stops: no same-provider retry, no other provider.
    ("interrupted", InterruptedError("Codex auxiliary Responses stream interrupted"), "openrouter",
     dict(attempts=1, fell_back=False)),
]


@pytest.mark.parametrize("case,error,provider,expected", BEHAVIOUR, ids=[c[0] for c in BEHAVIOUR])
def test_behaviour(case, error, provider, expected):
    outcome = run_ladder(error, provider=provider)
    assert {key: getattr(outcome, key) for key in expected} == expected, (case, outcome.result)


@pytest.mark.parametrize("retry_error,fell_back", [  # a terminal verdict (no retry, no fallback) holds after a rung
    pytest.param(sdk(400, "Bad Request"), True, id="format-error"),
    pytest.param(sdk(403, "No endpoint qualifies.", code="data_policy_denied"), False, id="data-policy-denied"),
    pytest.param(sdk(400, "invalid tool call arguments"), False, id="malformed-tool-args"),
])
def test_spent_parameter_rung(retry_error, fell_back):
    first = sdk(400, "Unsupported parameter: 'temperature' is not supported with this model.")
    outcome = run_ladder([first, retry_error], provider="nous", temperature=0.3)
    assert (outcome.attempts, outcome.fell_back, outcome.rotated) == (2, fell_back, False)


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streamed"])
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_each_failed_request_is_classified_once(monkeypatch, is_async, stream):
    """The primary, its same-provider retry and the fallback candidate fail: one hook call per failed request."""
    seen = []
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_error_classification", lambda **kw: seen.append(kw["error"]))
    with patch("agent.auxiliary_client._aux_progress_active", return_value=stream):
        outcome = run_ladder([sdk(503, "down"), sdk(503, "down")], is_async=is_async,
                             fallback_error=[sdk(404, "model not found"), sdk(404, "model not found")])
    assert len(set(map(id, seen))) == len(seen) == outcome.attempts + outcome.fallback_attempts


def test_plugin_verdict_steers_and_retryable_false_is_honoured(monkeypatch):
    from hermes_cli import plugins

    verdict = {"reason": FailoverReason.server_error, "retryable": False, "should_fallback": True}
    monkeypatch.setattr(plugins, "get_plugin_error_classification",
                        lambda **kw: verdict if "custom hiccup" in kw["error_message"] else None)
    outcome = run_ladder(sdk(400, "custom hiccup"))
    assert (outcome.attempts, outcome.fell_back) == (1, True)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_streamed_attempt_is_judged_by_the_routes_provider_profile(monkeypatch, is_async):
    """An explicit route (no auto-routing tag) still gets its profile's verdict on a failed stream negotiation."""
    import providers

    hook = lambda *_a, **_k: {"reason": "server_error", "retryable": False, "should_fallback": True}
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: SimpleNamespace(classify_api_error=hook))
    with patch("agent.auxiliary_client._aux_progress_active", return_value=True):
        outcome = run_ladder(sdk(400, "custom hiccup"), is_async=is_async)
    assert (outcome.attempts, outcome.fell_back) == (1, True)  # no non-streaming resend before recovery


@pytest.mark.parametrize("hook,expected", [
    ({"reason": FailoverReason.model_not_found, "should_fallback": False}, (1, False)),
    ({"reason": FailoverReason.rate_limit, "retryable": False, "should_rotate_credential": True}, (1, True)),
], ids=["should-fallback-false", "rate-limit-retryable-false"])
def test_plugin_hook_overrides_win_over_aux_defaults(monkeypatch, hook, expected):
    from hermes_cli import plugins

    monkeypatch.setattr(plugins, "get_plugin_error_classification", lambda **kw: hook)
    outcome = run_ladder(sdk(400, "plugin-owned failure"))
    assert (outcome.attempts, outcome.fell_back) == expected


@pytest.mark.parametrize("retry_outcome", [ConnectionError("Connection refused"), TimeoutError("timed out")])
def test_failure_after_a_401_refresh_does_not_bench_the_refreshed_key(retry_outcome):
    outcome = run_ladder([sdk(401, "invalid api key"), retry_outcome, retry_outcome])
    assert outcome.refreshed == ["openrouter"]
    assert (outcome.rotated, outcome.fell_back) == (False, True)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_429_then_generic_403_reraises_the_403(is_async):
    outcome = run_ladder([sdk(429, "Rate limit exceeded"), sdk(403, "Forbidden")], is_async=is_async)
    assert isinstance(outcome.result, openai.PermissionDeniedError) and not outcome.fell_back


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_opencode_go_candidate_classifies_on_its_concrete_provider(monkeypatch, is_async):
    """A sync candidate tagged opencode-go (converted to async by ``async_call_llm``) classifies on Go's
    profile: the upstream throttle quarantines the candidate without rotating its key."""
    import model_tools  # registers the bundled provider profiles
    from agent import auxiliary_error_policy as aux_errors

    seen, real = [], aux_errors.classify_api_error
    monkeypatch.setattr(aux_errors, "classify_api_error",
                        lambda exc, **kw: seen.append((kw["provider"], (v := real(exc, **kw)).reason)) or v)
    body = {"error": {"type": "server_error", "message": "Upstream request failed: Rate limit exceeded."}}
    transport = httpx.MockTransport(lambda request: httpx.Response(429, json=body, request=request))
    go = openai.OpenAI(api_key="sk-go", base_url="https://opencode.ai/zen/go/v1", max_retries=0,
                       http_client=httpx.Client(transport=transport))
    go._hermes_aux_effective_provider = "opencode-go"  # type: ignore[attr-defined]
    primary = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="sk-primary")
    primary.chat.completions.create = (AsyncMock if is_async else MagicMock)(side_effect=sdk(503, "down"))
    with (
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openrouter", "m", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "m")),
        patch("agent.auxiliary_client._openai_http_client_kwargs",
              return_value={"http_client": httpx.AsyncClient(transport=transport)}),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value=None),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(go, "go-model", "auto")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")),
        patch("agent.auxiliary_client._mark_provider_unhealthy"),
        pytest.raises(openai.APIError),
    ):
        if is_async:
            asyncio.run(async_call_llm(task="title_generation", messages=_MESSAGES))
        else:
            call_llm(task="title_generation", messages=_MESSAGES)
    assert ("opencode-go", FailoverReason.upstream_rate_limit) in seen


def _wire(handle, *, key, url="https://openrouter.ai/api/v1"):
    return openai.OpenAI(api_key=key, base_url=url, max_retries=0, http_client=httpx.Client(
        transport=httpx.MockTransport(handle)))


def test_refreshed_nous_stream_error_falls_back():
    """401, Nous refresh, then the refreshed key's SSE ``error`` event: classified as itself (fallback)."""
    sse = lambda request: httpx.Response(200, content=b'data: {"error": "backend worker failed"}\n\n',
                                         headers={"content-type": "text/event-stream"}, request=request)
    primary = _wire(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}, request=r), key="sk-old")
    refreshed = _wire(sse, key="sk-fresh", url="https://inference-api.nousresearch.com/v1")
    fallback = MagicMock(base_url="https://api.example.com/v1")
    fallback.chat.completions.create = MagicMock(return_value=_ok())
    with (
        patch("agent.auxiliary_client._aux_progress_active", return_value=True),
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "m", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "m")),
        patch("agent.auxiliary_client._refresh_nous_auxiliary_client", return_value=(refreshed, "m2")),
        patch("agent.auxiliary_client._refresh_nous_recommended_model", return_value=None),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value=None),
        patch("agent.auxiliary_client._mark_provider_unhealthy"),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(fallback, "fb", "fallback")),
    ):
        assert call_llm(task="title_generation", messages=_MESSAGES).choices[0].message.content == "fallback"


def test_invalid_answer_classifies_as_incomplete_response():
    from agent.error_classifier import classify_api_error

    assert classify_api_error(InvalidProviderResponseError("x")).reason is FailoverReason.incomplete_response


def test_tls_certificate_failure_keeps_compressions_session_preserving_abort():
    import ssl

    from agent.context_compressor import ContextCompressor

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2,
                                       abort_on_summary_failure=False)
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(12)]
    error = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    with patch("agent.context_compressor.call_llm", side_effect=error):
        assert compressor.compress(msgs, current_tokens=999999, force=True) == msgs
    assert compressor._last_summary_network_failure and compressor._last_compress_aborted


def test_relay_connection_reset_502_keeps_compressions_session_preserving_abort():
    from agent.context_compressor import ContextCompressor

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2,
                                       abort_on_summary_failure=False)
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(12)]
    with patch("agent.context_compressor.call_llm", side_effect=sdk(502, "upstream: connection reset by peer")):
        assert compressor.compress(msgs, current_tokens=999999, force=True) == msgs
    assert compressor._last_summary_network_failure


def test_pinned_custom_provider_entitlement_benches_only_the_model():
    from agent.auxiliary_client import _recover_provider_pool
    from agent.auxiliary_error_policy import classify_attempt

    error = sdk(400, "model kimi-k2 is not supported for this account")
    pool = MagicMock()
    with patch("agent.auxiliary_client.pool_for_aux_provider", return_value=pool):
        _recover_provider_pool("custom:relay", error, verdict=classify_attempt(error, "custom:relay", model="kimi-k2"))
    assert pool.mark_exhausted_and_rotate.call_args.kwargs["model"] == "kimi-k2"
    key = Key("openrouter", "k1", "k1", "api_key", 0, "manual", "sk-or", model_cooldowns={"openai/gpt-gated": 9e9})
    with patch("agent.auxiliary_client.load_pool", lambda _p: CredentialPool("openrouter", [key])):  # sibling models keep the key
        assert [_try_openrouter(model=m)[0] is None for m in ("gpt-ok", "gpt-gated")] == [False, True]


def test_missing_model_on_a_candidate_hides_only_that_model(monkeypatch):
    from agent import auxiliary_client as aux

    monkeypatch.setattr(aux, "_aux_unhealthy_until", {})
    monkeypatch.setattr(aux, "_aux_unhealthy_reason", {})
    aux._quarantine_fallback_candidate("title_generation", "fallback_chain[0](openrouter)", "openrouter",
                                   sdk(404, "model not found"), reason="model not found", model="old-model")
    assert aux._is_provider_unhealthy("openrouter", model="old-model")
    assert not aux._is_provider_unhealthy("openrouter", model="new-model")
    assert not aux._is_provider_unhealthy("openrouter")
    aux._quarantine_fallback_candidate("t", "fb", "openrouter", sdk(404, "x"), reason="model not found", model="openai/gpt-retired")
    assert aux._is_provider_unhealthy("openrouter", model="gpt-retired")  # resolved name written, configured name read
    with patch.object(aux, "_get_provider_chain", return_value=[("openrouter", lambda: (MagicMock(), "openai/gpt-retired")), (
            "nous", lambda: (MagicMock(), "m"))]), patch.object(aux, "_discovery_chain_allowed"):
        assert aux._try_payment_fallback("zai")[2] == "nous"  # discovery skips the hidden model too


def test_a_rebuilt_retry_is_classified_on_the_request_it_sent():  # e.g. a fast-model task rebuilt on the main model
    models, clients = [], [MagicMock(base_url="https://openrouter.ai/api/v1", **{"chat.completions.create.side_effect": sdk(
        400, f"model {model} is not supported for this account")}) for model in ("kimi-k2", "glm-5")]
    with patch.multiple(
        "agent.auxiliary_client", _resolve_task_provider_model=lambda *_a: ("openrouter", "kimi-k2", None, None, None),
        _get_cached_client=lambda *_a, **_k: (clients[0], "kimi-k2"), _prepare_same_provider_retry=lambda **_k: (
            clients[1], {"model": "glm-5"}), _recover_provider_pool=lambda _p, _e, **kw: models.append(kw["verdict"].model) or True,
    ), pytest.raises(openai.APIError):
        call_llm(task="title_generation", messages=_MESSAGES)
    assert models == ["kimi-k2", "glm-5"]  # the verdict models handed to pool rotation, in order


def _candidate_outcome(first, refreshed):
    from agent.auxiliary_client import _call_fallback_candidate_sync

    marks, raised = [], None
    with (
        patch("agent.auxiliary_client._mark_provider_unhealthy", side_effect=lambda *a, **k: marks.append(k)),
        patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True),
        patch("agent.auxiliary_client._get_cached_client", return_value=(refreshed, "fb-model")),
    ):
        try:
            _call_fallback_candidate_sync(
                first, "fb-model", "openrouter", task="title_generation", messages=_MESSAGES, temperature=None,
                max_tokens=16, tools=None, effective_timeout=30.0, effective_extra_body={}, reasoning_config=None)
        except Exception as exc:  # the outcome under test
            raised = type(exc)
    return raised, [(k["reason"], k["ttl"]) for k in marks]


@pytest.mark.parametrize("body", [b"null", b'{"success": false, "data": {"error": {"message": "crashed"}}}'])
def test_refreshed_candidates_bad_answer_is_not_a_stale_credential(body):
    """401, refresh, then a bad answer: the same outcome the answer gets with no 401 before it."""
    bad = lambda: _wire(lambda r: httpx.Response(200, content=body, headers={"content-type": "application/json"},
                                                 request=r), key="sk-fresh")
    stale = _wire(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}, request=r), key="sk-old")
    after_refresh = _candidate_outcome(stale, bad())
    assert ("stale fallback credential", None) not in after_refresh[1]
    assert after_refresh == _candidate_outcome(bad(), None)
