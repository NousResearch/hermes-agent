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

_REQ = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
_MESSAGES = [{"role": "user", "content": "hi"}]


class _Plain(Exception):
    pass


def sdk(status: int, message: str, *, wrapped: bool = False, **error_fields):
    """An error shaped like openai-python's ``_make_status_error`` output (``wrapped``: a relay that keeps
    the ``{"error": ...}`` envelope on the exception body)."""
    body = {"message": message, **error_fields}
    if wrapped:
        body = {"error": body}
    by_status = {400: openai.BadRequestError, 401: openai.AuthenticationError, 403: openai.PermissionDeniedError,
                 404: openai.NotFoundError, 429: openai.RateLimitError}
    cls = by_status.get(status, openai.InternalServerError if status >= 500 else openai.APIStatusError)
    return cls(f"Error code: {status} - {{'error': {body!r}}}", response=httpx.Response(status, request=_REQ), body=body)


def _ok():
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fallback"))])


def run_ladder(error, *, provider="openrouter", retries=1, is_async=False, fallback_error=None, pool=None, call=None,
               client=None, **call_kw):
    """``call_llm`` / ``async_call_llm`` on a client whose requests raise ``error`` (a list: one per request).
    ``pool``: a real ``CredentialPool`` that recovery rotates (its ``sk-primary`` key sent the request).
    ``call``: the caller that reaches the ladder instead of a bare ``call_llm`` (a compression, say)."""
    create = AsyncMock if is_async else MagicMock
    primary = client or MagicMock(base_url="https://openrouter.ai/api/v1", api_key="sk-primary")
    if client is None:
        primary.chat.completions.create = create(side_effect=error)
    fallback = MagicMock(base_url="https://api.example.com/v1")
    fallback.chat.completions.create = create(side_effect=fallback_error, return_value=_ok())
    marks, rotations, refreshes = [], [], []
    recovery = (patch("agent.auxiliary_client.load_pool", lambda _p: pool) if pool is not None else
                patch("agent.auxiliary_client._recover_provider_pool", side_effect=lambda _p, e, **_k: rotations.append(e)))
    with (
        recovery,
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=(provider, "m", None, None, None)),
        patch("agent.auxiliary_client._get_cached_client", return_value=(primary, "m")),
        patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
        patch("agent.auxiliary_client._transient_retry_count", return_value=retries),
        patch("agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0),
        patch("agent.auxiliary_client._recoverable_pool_provider", return_value=pool.provider if pool else "openrouter"),
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
            if call is not None:
                result = call()
            elif is_async:
                result = asyncio.run(async_call_llm(messages=_MESSAGES, **call_kw))
            else:
                result = call_llm(messages=_MESSAGES, **call_kw)
        except Exception as exc:  # the outcome under test
            result = exc
    return SimpleNamespace(attempts=getattr(primary.chat.completions.create, "call_count", None),
                           fallback_attempts=fallback.chat.completions.create.call_count,
                           fell_back=fallback.chat.completions.create.call_count > 0,
                           unhealthy=bool(marks), rotated=bool(rotations), refreshed=refreshes, result=result)


def _gemini_prompt_block():
    """The native Gemini adapter's error for a 200 answer carrying ``promptFeedback.blockReason``."""
    from agent.gemini_native_adapter import raise_on_prompt_block

    try:
        raise_on_prompt_block({"promptFeedback": {"blockReason": "SAFETY"}})
    except Exception as exc:  # the adapter's own error, not a hand-built one
        return exc
    raise AssertionError("the adapter did not raise on a prompt block")


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
    ("data-policy-denied-quota-words", sdk(403, "quota exceeded", code="data_policy_denied"), "nous",
     dict(attempts=1, fell_back=False, unhealthy=False, rotated=False)),
    ("statusless-sse-404-body", openai.APIError("Not Found", _REQ, body={"code": 404, "message": "Not Found"}),
     "openrouter", dict(attempts=1, fell_back=True)),
    ("response-ended-prematurely", _Plain("RemoteProtocolError: response ended prematurely"), "openrouter",
     dict(attempts=2, fell_back=True)),
    ("openrouter-upstream-429", sdk(429, "Provider returned error", wrapped=True, code=429,
                                    metadata={"raw": "rate-limited upstream", "provider_name": "DeepInfra"}),
     "openrouter", dict(attempts=1, fell_back=True, rotated=False)),
    # An upstream's quota wording does not bench the healthy OpenRouter key's provider.
    ("openrouter-upstream-quota", sdk(429, "Provider returned error", wrapped=True, code=429, metadata={
        "raw": '{"error": {"message": "Quota exceeded for this model"}}', "provider_name": "DeepInfra"}),
     "openrouter", dict(fell_back=True, unhealthy=False, rotated=False)),
    # A safety refusal is of the content itself: it is never resent to another provider.
    ("gemini-prompt-blocked", _gemini_prompt_block(), "gemini", dict(attempts=1, fell_back=False)),
    ("400-content-policy", sdk(400, "Your request was flagged by our safety system"), "openrouter",
     dict(attempts=1, fell_back=False)),
    # An interrupted task stops: no same-provider retry, no other provider.
    ("interrupted", InterruptedError("Codex auxiliary Responses stream interrupted"), "openrouter",
     dict(attempts=1, fell_back=False)),
]


@pytest.mark.parametrize("case,error,provider,expected", BEHAVIOUR, ids=[c[0] for c in BEHAVIOUR])
def test_behaviour(case, error, provider, expected):
    outcome = run_ladder(error, provider=provider)
    assert {key: getattr(outcome, key) for key in expected} == expected, (case, outcome.result)


# (case id, summary request's error, compression aborts with the transcript intact): the ladder hands a safety
# refusal back to the compressor; only Gemini's prompt block aborts, other refusals keep the fallback summary.
COMPRESSION_BEHAVIOUR = [
    ("gemini-prompt-blocked", _gemini_prompt_block(), True),
    ("400-content-policy", sdk(400, "Your request was flagged by our safety system"), False),
]


@pytest.mark.parametrize("case,error,aborted", COMPRESSION_BEHAVIOUR, ids=[c[0] for c in COMPRESSION_BEHAVIOUR])
def test_compression_behaviour(case, error, aborted):
    from agent.context_compressor import ContextCompressor

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="gemini-2.5-flash", provider="gemini", quiet_mode=True, protect_first_n=2,
                                       protect_last_n=2, abort_on_summary_failure=False)
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(12)]
    outcome = run_ladder(error, provider="gemini",
                         call=lambda: compressor.compress(msgs, current_tokens=999999, force=True))
    assert (outcome.fell_back, outcome.result == msgs, compressor._last_compress_aborted) == (False, aborted, aborted)


_OUT_OF_EXTRA_USAGE = "You're out of extra usage. Add more at claude.ai/settings/usage and keep going."
# (case id, error, route provider, pooled keys, the failed key's bench): sized from the verdict as the main loop's
# credential recovery sizes it, as (stored status, pool failure reason, seconds) or the models cooled down.
POOL_BENCH = [
    # Anthropic's "out of extra usage" 400 can be a content filter on a healthy account (#82154).
    ("unverified-billing-400-sole", sdk(400, _OUT_OF_EXTRA_USAGE), "anthropic", 1, (400, "billing_unverified", 60)),
    ("unverified-billing-400-pool", sdk(400, _OUT_OF_EXTRA_USAGE), "anthropic", 2, (400, "billing_unverified", 60)),
    ("billing-403-sole", sdk(403, "Key limit exceeded (total limit)."), "openrouter", 1, (403, "billing", 3600)),
    ("billing-402-sole", sdk(402, "Insufficient credits"), "openrouter", 1, (402, "billing", 3600)),
    ("anthropic-per-model-429", sdk(429, "Rate limit exceeded"), "anthropic", 1, ["m"]),
]


@pytest.mark.parametrize("case,error,provider,keys,bench", POOL_BENCH, ids=[c[0] for c in POOL_BENCH])
def test_pool_bench(case, error, provider, keys, bench):
    from agent.credential_pool import _exhausted_until

    pool = CredentialPool(provider, [Key(provider, f"k{i}", f"k{i}", "api_key", i, "manual", f"sk-{i}" if i else "sk-primary")
                                     for i in range(keys)])
    run_ladder(error, provider=provider, pool=pool)
    failed = pool.entries()[0]
    until = _exhausted_until(failed, sole_credential=keys == 1)
    if until is None:
        benched = sorted(failed.model_cooldowns or {})
    else:
        benched = (failed.last_error_code, failed.failure_reason, round(until - (failed.last_status_at or 0)))
    assert benched == bench, case


_MTA = "invalid tool call arguments: max_tokens must be an integer"


@pytest.mark.parametrize("retry_errors,attempts,fell_back", [  # a terminal verdict (no retry, no fallback) holds after a rung
    pytest.param([sdk(400, "Bad Request")], 2, True, id="format-error"),
    pytest.param([sdk(403, "No endpoint qualifies.", code="data_policy_denied")], 2, False, id="data-policy-denied"),
    pytest.param([sdk(400, "invalid tool call arguments")], 2, False, id="malformed-tool-args"),
    # A message naming another parameter still takes that rung (format_error, as on main), then stops.
    pytest.param([sdk(400, _MTA), sdk(400, _MTA)], 3, False, id="malformed-tool-args-naming-max-tokens"),
])
def test_spent_parameter_rung(retry_errors, attempts, fell_back):
    first = sdk(400, "Unsupported parameter: 'temperature' is not supported with this model.")
    outcome = run_ladder([first, *retry_errors], provider="nous", temperature=0.3, max_tokens=16)
    assert (outcome.attempts, outcome.fell_back, outcome.rotated) == (attempts, fell_back, False)


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


@pytest.mark.parametrize("route_tag", ["openrouter", "api-key", None])
def test_pool_recovery_reuses_the_failed_requests_verdict(monkeypatch, route_tag):
    """Credential-pool recovery on an auto route sizes the bench from the verdict the ladder holds: one hook
    call per failed request, whatever label the client carries."""
    seen = []
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_error_classification", lambda **kw: seen.append(kw["error"]))
    client = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="sk-primary")
    client._hermes_aux_effective_provider = route_tag
    client._hermes_aux_backend_provider = None
    client.chat.completions.create = MagicMock(side_effect=[sdk(402, "Insufficient credits")] * 2)
    pool = CredentialPool("openrouter", [Key("openrouter", f"k{i}", f"k{i}", "api_key", i, "manual",
                                             f"sk-{i}" if i else "sk-primary") for i in range(2)])
    outcome = run_ladder(None, provider="auto", pool=pool, client=client)
    assert len(seen) == outcome.attempts == 2


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_observers_see_the_providers_own_error_when_its_stream_was_judged(monkeypatch, is_async):
    """A stream negotiation is classified where it failed; ``post_auxiliary_call`` still reports the provider's error."""
    from hermes_cli import plugins
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    manager = PluginManager()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    monkeypatch.setattr(plugins, "_plugin_managers_by_home", {})
    seen = []
    PluginContext(PluginManifest(name="aux-observer", source="user"), manager).register_hook(
        "post_auxiliary_call", lambda **kw: seen.append(kw["error_type"]))
    with patch("agent.auxiliary_client._aux_progress_active", return_value=True):
        run_ladder([sdk(503, "down"), sdk(503, "down")], is_async=is_async,
                   fallback_error=[sdk(404, "model not found"), sdk(404, "model not found")])
    assert set(seen) == {"InternalServerError", "NotFoundError"}


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


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_streamed_certificate_failure_is_not_resent_non_streaming(is_async):
    import ssl

    error = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    with patch("agent.auxiliary_client._aux_progress_active", return_value=True):
        outcome = run_ladder(error, is_async=is_async)
    assert (outcome.attempts, outcome.fell_back) == (1, True)  # one doomed connection, then the next provider


@pytest.mark.parametrize("hook,expected", [
    ({"reason": FailoverReason.model_not_found, "should_fallback": False}, (1, False)),
    ({"reason": FailoverReason.rate_limit, "retryable": False, "should_rotate_credential": True}, (1, True)),
], ids=["should-fallback-false", "rate-limit-retryable-false"])
def test_plugin_hook_overrides_win_over_aux_defaults(monkeypatch, hook, expected):
    from hermes_cli import plugins

    monkeypatch.setattr(plugins, "get_plugin_error_classification", lambda **kw: hook)
    outcome = run_ladder(sdk(400, "plugin-owned failure"))
    assert (outcome.attempts, outcome.fell_back) == expected


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


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_discovered_api_key_client_classifies_on_its_concrete_provider(monkeypatch, is_async):
    """Auto discovery's API-key step builds the OpenCode Go client itself (no hand-set tag): its upstream
    throttle is judged on Go's profile, not under the route label "api-key", so the key is not rotated."""
    import model_tools  # registers the bundled provider profiles
    from agent import auxiliary_client as aux, auxiliary_error_policy as aux_errors

    seen, real = [], aux_errors.classify_api_error
    monkeypatch.setattr(aux_errors, "classify_api_error", lambda exc, **kw: seen.append(
        (kw["provider"], (v := real(exc, **kw)).reason, v.should_rotate_credential)) or v)
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "sk-go")
    monkeypatch.setattr(aux, "_client_cache", {})
    body = {"error": {"type": "server_error", "message": "Upstream request failed: Rate limit exceeded."}}
    transport = httpx.MockTransport(lambda request: httpx.Response(429, json=body, request=request))
    none = lambda *_a, **_k: (None, None)
    with (
        patch.multiple(aux, _try_openrouter=none, _try_nous=none, _try_custom_endpoint=none),
        patch("agent.auxiliary_client._openai_http_client_kwargs", lambda _url, async_mode=False: {
            "http_client": (httpx.AsyncClient if async_mode else httpx.Client)(transport=transport)}),
        patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("auto", None, None, None, None)),
        patch("agent.auxiliary_client._transient_retry_count", return_value=0),
        patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_main_fallback_chain", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_payment_fallback", return_value=(None, None, "")),
        patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")),
        patch("agent.auxiliary_client._mark_provider_unhealthy"),
        pytest.raises(openai.APIError),
    ):
        if is_async:
            asyncio.run(async_call_llm(task="title_generation", messages=_MESSAGES))
        else:
            call_llm(task="title_generation", messages=_MESSAGES)
    assert seen and set(seen) == {("opencode-go", FailoverReason.upstream_rate_limit, False)}


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


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_refreshed_nous_client_under_auto_classifies_as_nous(monkeypatch, is_async):
    from agent import auxiliary_client as aux

    seen = []
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_error_classification", lambda **kw: seen.append(kw["provider"]))
    with patch.object(aux, "_resolve_nous_runtime_api", return_value=("sk-fresh", "https://inference-api.nousresearch.com/v1")):
        client, _model = aux._refresh_nous_auxiliary_client(cache_provider="auto", model="m", async_mode=is_async)
    aux._classify_failure(sdk(500, "down"), "auto", client, "m", None)
    assert seen == ["nous"]


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


def test_model_entitlement_cooldown_keeps_the_key_for_its_other_models():
    """A ChatGPT account refused one Codex model (#71970), or an Anthropic key one Claude model, still serves aux
    calls on its other models."""
    from agent.auxiliary_client import _build_codex_client, resolve_provider_client

    key = Key("openai-codex", "k1", "k1", "api_key", 0, "manual", "tok", model_cooldowns={"gpt-5.5-pro": 9e9})
    with (patch("agent.auxiliary_client.load_pool", lambda _p: CredentialPool("openai-codex", [key])),
          patch("agent.auxiliary_client._read_codex_singleton_token", return_value=None)):
        assert [_build_codex_client(m)[0] is None for m in ("gpt-5.4-mini", "gpt-5.5-pro")] == [False, True]
        assert [resolve_provider_client("openai-codex", model=m, raw_codex=True)[0] is None
                for m in ("gpt-5.4-mini", "gpt-5.5-pro")] == [False, True]
    key = Key("anthropic", "k1", "k1", "api_key", 0, "manual", "sk-ant-api", model_cooldowns={"claude-opus-4-6": 9e9})
    with (patch("agent.auxiliary_client.load_pool", lambda _p: CredentialPool("anthropic", [key])),
          patch("agent.anthropic_credentials.resolve_anthropic_token", return_value=None)):
        assert [resolve_provider_client("anthropic", model=m)[0] is None
                for m in ("claude-haiku-4-5", "claude-opus-4-6")] == [False, True]


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


@pytest.mark.parametrize("retry_outcome,mark", [(ConnectionError("Connection refused"), ("connection error", 60)),
                                                (TimeoutError("timed out"), ("request timed out", None))])
def test_failure_after_a_candidates_401_refresh_does_not_bench_the_refreshed_key(retry_outcome, mark):
    """401, refresh, then a transport failure: the candidate is held for that failure, not as a stale credential."""
    stale = _wire(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}, request=r), key="sk-old")
    refreshed = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="sk-fresh")
    refreshed.chat.completions.create.side_effect = retry_outcome
    assert _candidate_outcome(stale, refreshed)[1] == [mark]


@pytest.mark.parametrize("body", [b"null", b'{"success": false, "data": {"error": {"message": "crashed"}}}'])
def test_refreshed_candidates_bad_answer_is_not_a_stale_credential(body):
    """401, refresh, then a bad answer: the same outcome the answer gets with no 401 before it."""
    bad = lambda: _wire(lambda r: httpx.Response(200, content=body, headers={"content-type": "application/json"},
                                                 request=r), key="sk-fresh")
    stale = _wire(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}, request=r), key="sk-old")
    after_refresh = _candidate_outcome(stale, bad())
    assert ("stale fallback credential", None) not in after_refresh[1]
    assert after_refresh == _candidate_outcome(bad(), None)


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_a_rebuilt_retrys_invalid_answer_is_classified_on_the_request_it_sent(monkeypatch, is_async):
    seen = []
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_error_classification", lambda **kw: seen.append(kw["model"]))
    create = AsyncMock if is_async else MagicMock
    first = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="k")
    first.chat.completions.create = create(side_effect=sdk(401, "invalid api key"))
    retry = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="k2")
    retry.chat.completions.create = create(return_value=SimpleNamespace(choices=None))
    with patch.multiple(
        "agent.auxiliary_client", _resolve_task_provider_model=lambda *_a: ("openrouter", "kimi-k2", None, None, None),
        _get_cached_client=lambda *_a, **_k: (first, "kimi-k2"), _refresh_provider_credentials=lambda *_a, **_k: True,
        _prepare_same_provider_retry=lambda **_k: (retry, {"model": "glm-5", "messages": _MESSAGES}),
        _recoverable_pool_provider=lambda *_a, **_k: None, _try_main_agent_model_fallback=lambda *_a, **_k: (None, None, ""),
        _try_configured_fallback_chain=lambda *_a, **_k: (None, None, ""),
    ), pytest.raises(Exception):
        if is_async:
            asyncio.run(async_call_llm(task="title_generation", messages=_MESSAGES))
        else:
            call_llm(task="title_generation", messages=_MESSAGES)
    assert seen[-1] == "glm-5", seen


def test_stream_watchdog_timeout_is_not_judged_by_the_error_it_cut_off():
    """The watchdog fires, then the stream surfaces a 401: the timeout is the verdict (no refresh)."""
    from agent import auxiliary_error_policy as policy, auxiliary_stream_watchdog as wd

    watchdogs = []

    class Recorder(wd.ChatStreamWatchdog):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            watchdogs.append(self)

    def chunks():
        yield "chunk"
        watchdogs[0].fired = True
        raise openai.APIError("Error code: 401 - invalid api key", _REQ, body={"code": 401, "message": "invalid"})

    with patch.object(wd, "ChatStreamWatchdog", Recorder), pytest.raises(TimeoutError) as caught:
        wd.consume_chat_stream(chunks(), SimpleNamespace(finish_reason=None, feed=lambda _c: True), (30.0, None))
    assert caught.value.__context__ is None  # the classifier walks __context__
    verdict = policy.classify_attempt(caught.value, "openrouter")
    assert policy.fallback_label(policy.Failure(caught.value, verdict)) == "request timed out"
    assert not policy.auth_refresh(verdict)



_POLICY_REFUSALS = [
    (403, {"error": {"code": "data_policy_denied", "message": "No eligible data-policy route for max_tokens=16"}}),
    (400, {"error": {"message": "Your request was flagged by our safety system. Unsupported parameter: 'temperature'"}}),
    None,  # an interrupted task
]


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streamed"])
@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("refusal", _POLICY_REFUSALS, ids=["data-policy-denied", "content-policy", "interrupted"])
def test_a_refusal_no_resend_can_change_is_sent_once(refusal, is_async, stream):
    """No parameter strip or non-streaming resend re-asks after a policy refusal or a cancelled task."""
    sent = []

    def handle(request):
        sent.append(request)
        if refusal is None:
            raise InterruptedError("auxiliary stream interrupted")
        return httpx.Response(refusal[0], json=refusal[1], request=request)

    sdk_client, http = (openai.AsyncOpenAI, httpx.AsyncClient) if is_async else (openai.OpenAI, httpx.Client)
    client = sdk_client(api_key="sk", base_url="https://openrouter.ai/api/v1", max_retries=0,
                        http_client=http(transport=httpx.MockTransport(handle)))
    if refusal is None:  # the SDK would wrap it as a connection error: raise it as the stream guard does
        client.chat.completions.create = (AsyncMock if is_async else MagicMock)(
            side_effect=lambda **kw: handle(None))
    with patch("agent.auxiliary_client._aux_progress_active", return_value=stream):
        outcome = run_ladder(None, provider="nous", is_async=is_async, client=client,
                             max_tokens=16, temperature=0.3)
    assert (len(sent), outcome.fell_back) == (1, False), outcome.result
