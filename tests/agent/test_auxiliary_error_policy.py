"""Auxiliary recovery decisions come from the shared classifier, driven end to end through ``call_llm``."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from hermes_agent.agent.auxiliary_client import _try_openrouter, async_call_llm, call_llm
from hermes_agent.agent.credential_pool import CredentialPool, PooledCredential as Key
from hermes_agent.agent.error_classifier import FailoverReason
from hermes_agent.agent.errors import InvalidProviderResponseError

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
    recovery = (patch("hermes_agent.agent.auxiliary_client.load_pool", lambda _p: pool) if pool is not None else
                patch("hermes_agent.agent.auxiliary_client._recover_provider_pool", side_effect=lambda _p, e, **_k: rotations.append(e)))
    with (
        recovery,
        patch("hermes_agent.agent.auxiliary_client._resolve_task_provider_model", return_value=(provider, "m", None, None, None)),
        patch("hermes_agent.agent.auxiliary_client._get_cached_client", return_value=(primary, "m")),
        patch("hermes_agent.agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
        patch("hermes_agent.agent.auxiliary_client._transient_retry_count", return_value=retries),
        patch("hermes_agent.agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0),
        patch("hermes_agent.agent.auxiliary_client._recoverable_pool_provider", return_value=pool.provider if pool else "openrouter"),
        patch("hermes_agent.agent.auxiliary_client._mark_provider_unhealthy", side_effect=lambda *a, **kw: marks.append(a)),
        patch("hermes_agent.agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("hermes_agent.agent.auxiliary_client._try_main_fallback_chain", return_value=(None, None, "")),
        patch("hermes_agent.agent.auxiliary_client._try_payment_fallback", return_value=(fallback, "fb", "fallback")),
        patch("hermes_agent.agent.auxiliary_client._try_main_agent_model_fallback", return_value=(fallback, "fb", "fallback")),
        patch("hermes_agent.agent.auxiliary_rate_limit_backoff.rate_limit_wait", return_value=0.0),
        patch("hermes_agent.agent.auxiliary_client._refresh_provider_credentials",
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
    from hermes_agent.agent.gemini_native_adapter import raise_on_prompt_block

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
    ("content-policy-quota-words", sdk(403, "Your request was flagged by our safety system. Quota exceeded."),
     "openrouter", dict(attempts=1, fell_back=False, unhealthy=False, rotated=False)),
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
    from hermes_agent.agent.context_compressor import ContextCompressor

    with patch("hermes_agent.agent.context_compressor.get_model_context_length", return_value=100000):
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
    from hermes_agent.agent.credential_pool import _exhausted_until

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
    with patch("hermes_agent.agent.auxiliary_client._aux_progress_active", return_value=stream):
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
@pytest.mark.parametrize("route", ["custom", "auto"])
def test_opencode_go_upstream_throttle_keeps_the_key_on_an_unnamed_route(monkeypatch, route, is_async):
    """A ``custom`` or untagged auto route at the main runtime's OpenCode Go host: the one classification per request
    is made on opencode-go, so Go's per-model upstream 429 rotates no key and the call falls back."""
    from hermes_agent import model_tools  # registers the bundled provider profiles
    from hermes_agent.agent import auxiliary_client as aux

    seen, real_pool_provider = [], aux._recoverable_pool_provider
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_error_classification", lambda **kw: seen.append(kw["error"]))
    go_url = "https://opencode.ai/zen/go/v1"
    client = MagicMock(base_url=go_url, api_key="sk-0", _hermes_aux_effective_provider=None,
                       _hermes_aux_backend_provider=None)
    client.chat.completions.create = (AsyncMock if is_async else MagicMock)(
        side_effect=[sdk(429, "Upstream request failed: Rate limit exceeded.", type="server_error")] * 3)
    pool = CredentialPool("opencode-go", [Key("opencode-go", f"k{i}", f"k{i}", "api_key", i, "manual", f"sk-{i}")
                                          for i in range(2)])
    runtime = {"provider": "opencode-go", "base_url": go_url}

    def call():  # the real host resolution, from the main runtime
        with patch("hermes_agent.agent.auxiliary_client._recoverable_pool_provider", real_pool_provider):
            if is_async:
                return asyncio.run(async_call_llm(messages=_MESSAGES, main_runtime=runtime))
            return call_llm(messages=_MESSAGES, main_runtime=runtime)
    outcome = run_ladder(None, provider=route, pool=pool, client=client, call=call, is_async=is_async)
    assert outcome.fell_back and [k.last_status for k in pool.entries()] == [None, None]
    assert len(seen) == outcome.attempts


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("zen_url,go_url", [
    ("https://opencode.ai/zen/v1", "https://opencode.ai/zen/go/v1"),  # Zen beside Go
    ("https://proxy.example:9443/zen/go/v1", "https://proxy.example:8443/zen/go/v1"),  # same path, other port
    ("http://proxy.example/zen/go/v1", "https://proxy.example/zen/go/v1"),  # same path, other scheme
], ids=["other-path", "other-port", "other-scheme"])
def test_a_custom_zen_endpoint_is_not_classified_as_the_main_runtimes_go(is_async, zen_url, go_url):
    """OpenCode Zen and Go share opencode.ai: a custom Zen endpoint with its own key, while the main runtime is Go,
    is judged as custom, so its upstream 429 gets the same-key retry and the second request answers. So is
    an endpoint on Go's path at another origin."""
    from hermes_agent import model_tools  # registers the bundled provider profiles

    sent, statuses = [], iter([429, 200])

    def handle(request):
        sent.append(request.url.path)
        if next(statuses) == 429:
            return httpx.Response(429, json={"error": {"type": "server_error", "message":
                                                       "Upstream request failed: Rate limit exceeded."}}, request=request)
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "fast-1",
                                         "choices": [{"index": 0, "finish_reason": "stop", "message": {
                                             "role": "assistant", "content": "ok"}}]}, request=request)

    zen = _wire(handle, key="sk-zen", url=zen_url, is_async=is_async)
    runtime = {"provider": "opencode-go", "base_url": go_url, "api_key": "sk-go"}
    with patch("hermes_agent.agent.auxiliary_rate_limit_backoff.rate_limit_wait", return_value=0.0):
        assert _routed_call([zen] * 2, provider="custom", is_async=is_async, main_runtime=runtime) == "ok"
    assert sent == [httpx.URL(zen_url).path + "/chat/completions"] * 2


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_a_custom_zen_endpoints_payment_error_benches_that_endpoint_not_go(is_async):
    """The same Zen endpoint under a Go session answers 402: the endpoint that sent it is benched, Go stays eligible."""
    from hermes_agent.agent import auxiliary_client as aux

    health, zen_url = {}, "https://opencode.ai/zen/v1"
    zen = _wire(lambda r: httpx.Response(402, json={"error": {"message": "Insufficient Balance"}}, request=r),
                key="sk-zen", url=zen_url, is_async=is_async)
    runtime = {"provider": "opencode-go", "base_url": "https://opencode.ai/zen/go/v1", "api_key": "sk-go"}
    assert _routed_call([zen] * 2, provider="custom", is_async=is_async, health=health, main_runtime=runtime) == "fallback"
    with patch.object(aux, "_aux_unhealthy_until", health):
        assert (aux._is_provider_unhealthy("opencode-go"), aux._is_provider_unhealthy("custom", zen_url)) == (False, True)


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
    with patch("hermes_agent.agent.auxiliary_client._aux_progress_active", return_value=True):
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
    from hermes_agent import providers

    hook = lambda *_a, **_k: {"reason": "server_error", "retryable": False, "should_fallback": True}
    monkeypatch.setattr(providers, "get_provider_profile", lambda name: SimpleNamespace(classify_api_error=hook))
    with patch("hermes_agent.agent.auxiliary_client._aux_progress_active", return_value=True):
        outcome = run_ladder(sdk(400, "custom hiccup"), is_async=is_async)
    assert (outcome.attempts, outcome.fell_back) == (1, True)  # no non-streaming resend before recovery


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_streamed_certificate_failure_is_not_resent_non_streaming(is_async):
    import ssl

    error = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    with patch("hermes_agent.agent.auxiliary_client._aux_progress_active", return_value=True):
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
@pytest.mark.parametrize("effective,backend", [("opencode-go", None), ("api-key", "opencode-go")],
                         ids=["effective-provider", "backend-tag"])
def test_opencode_go_candidate_classifies_on_its_concrete_provider(monkeypatch, is_async, effective, backend):
    """A sync candidate tagged opencode-go (converted to async by ``async_call_llm``) classifies on Go's
    profile: the upstream throttle quarantines the candidate without rotating its key. A discovered candidate keeps
    the route label ``api-key`` as its effective provider and names Go in its backend tag."""
    from hermes_agent import model_tools  # registers the bundled provider profiles
    from hermes_agent.agent import auxiliary_error_policy as aux_errors

    seen, real = [], aux_errors.classify_api_error
    monkeypatch.setattr(aux_errors, "classify_api_error", lambda exc, **kw: seen.append(
        (kw["provider"], (v := real(exc, **kw)).reason, v.should_rotate_credential)) or v)
    body = {"error": {"type": "server_error", "message": "Upstream request failed: Rate limit exceeded."}}
    transport = httpx.MockTransport(lambda request: httpx.Response(429, json=body, request=request))
    go = openai.OpenAI(api_key="sk-go", base_url="https://opencode.ai/zen/go/v1", max_retries=0,
                       http_client=httpx.Client(transport=transport))
    go._hermes_aux_effective_provider = effective  # type: ignore[attr-defined]
    go._hermes_aux_backend_provider = backend  # type: ignore[attr-defined]
    primary = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="sk-primary")
    primary.chat.completions.create = (AsyncMock if is_async else MagicMock)(side_effect=sdk(503, "down"))
    with (
        patch("hermes_agent.agent.auxiliary_client._resolve_task_provider_model", return_value=("openrouter", "m", None, None, None)),
        patch("hermes_agent.agent.auxiliary_client._get_cached_client", return_value=(primary, "m")),
        patch("hermes_agent.agent.auxiliary_client._openai_http_client_kwargs",
              return_value={"http_client": httpx.AsyncClient(transport=transport)}),
        patch("hermes_agent.agent.auxiliary_client._transient_retry_count", return_value=0),
        patch("hermes_agent.agent.auxiliary_client._recoverable_pool_provider", return_value=None),
        patch("hermes_agent.agent.auxiliary_client._try_configured_fallback_chain", return_value=(go, "go-model", "auto")),
        patch("hermes_agent.agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")),
        patch("hermes_agent.agent.auxiliary_client._mark_provider_unhealthy"),
        pytest.raises(openai.APIError),
    ):
        if is_async:
            asyncio.run(async_call_llm(task="title_generation", messages=_MESSAGES))
        else:
            call_llm(task="title_generation", messages=_MESSAGES)
    assert ("opencode-go", FailoverReason.upstream_rate_limit, False) in seen


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_discovered_api_key_client_classifies_on_its_concrete_provider(monkeypatch, is_async):
    """Auto discovery's API-key step builds the OpenCode Go client itself (no hand-set tag): its upstream
    throttle is judged on Go's profile, not under the route label "api-key", so the key is not rotated."""
    from hermes_agent import model_tools  # registers the bundled provider profiles
    from hermes_agent.agent import auxiliary_client as aux, auxiliary_error_policy as aux_errors

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
        patch("hermes_agent.agent.auxiliary_client._openai_http_client_kwargs", lambda _url, async_mode=False: {
            "http_client": (httpx.AsyncClient if async_mode else httpx.Client)(transport=transport)}),
        patch("hermes_agent.agent.auxiliary_client._resolve_task_provider_model", return_value=("auto", None, None, None, None)),
        patch("hermes_agent.agent.auxiliary_client._transient_retry_count", return_value=0),
        patch("hermes_agent.agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("hermes_agent.agent.auxiliary_client._try_main_fallback_chain", return_value=(None, None, "")),
        patch("hermes_agent.agent.auxiliary_client._try_payment_fallback", return_value=(None, None, "")),
        patch("hermes_agent.agent.auxiliary_client._try_main_agent_model_fallback", return_value=(None, None, "")),
        patch("hermes_agent.agent.auxiliary_client._mark_provider_unhealthy"),
        pytest.raises(openai.APIError),
    ):
        if is_async:
            asyncio.run(async_call_llm(task="title_generation", messages=_MESSAGES))
        else:
            call_llm(task="title_generation", messages=_MESSAGES)
    assert seen and set(seen) == {("opencode-go", FailoverReason.upstream_rate_limit, False)}


def _wire(handle, *, key, url="https://openrouter.ai/api/v1", is_async=False):
    sdk_client, http = (openai.AsyncOpenAI, httpx.AsyncClient) if is_async else (openai.OpenAI, httpx.Client)
    return sdk_client(api_key=key, base_url=url, max_retries=0, http_client=http(transport=httpx.MockTransport(handle)))


def test_refreshed_nous_stream_error_falls_back():
    """401, Nous refresh, then the refreshed key's SSE ``error`` event: classified as itself (fallback)."""
    sse = lambda request: httpx.Response(200, content=b'data: {"error": "backend worker failed"}\n\n',
                                         headers={"content-type": "text/event-stream"}, request=request)
    primary = _wire(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}, request=r), key="sk-old")
    refreshed = _wire(sse, key="sk-fresh", url="https://inference-api.nousresearch.com/v1")
    fallback = MagicMock(base_url="https://api.example.com/v1")
    fallback.chat.completions.create = MagicMock(return_value=_ok())
    with (
        patch("hermes_agent.agent.auxiliary_client._aux_progress_active", return_value=True),
        patch("hermes_agent.agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "m", None, None, None)),
        patch("hermes_agent.agent.auxiliary_client._get_cached_client", return_value=(primary, "m")),
        patch("hermes_agent.agent.auxiliary_client._refresh_nous_auxiliary_client", return_value=(refreshed, "m2")),
        patch("hermes_agent.agent.auxiliary_client._refresh_nous_recommended_model", return_value=None),
        patch("hermes_agent.agent.auxiliary_client._transient_retry_count", return_value=0),
        patch("hermes_agent.agent.auxiliary_client._recoverable_pool_provider", return_value=None),
        patch("hermes_agent.agent.auxiliary_client._mark_provider_unhealthy"),
        patch("hermes_agent.agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")),
        patch("hermes_agent.agent.auxiliary_client._try_main_agent_model_fallback", return_value=(fallback, "fb", "fallback")),
    ):
        assert call_llm(task="title_generation", messages=_MESSAGES).choices[0].message.content == "fallback"


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_refreshed_nous_client_under_auto_classifies_as_nous(monkeypatch, is_async):
    from hermes_agent.agent import auxiliary_client as aux

    seen = []
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_error_classification", lambda **kw: seen.append(kw["provider"]))
    with patch.object(aux, "_resolve_nous_runtime_api", return_value=("sk-fresh", "https://inference-api.nousresearch.com/v1")):
        client, _model = aux._refresh_nous_auxiliary_client(cache_provider="auto", model="m", async_mode=is_async)
    aux._classify_failure(sdk(500, "down"), "auto", client, "m")
    assert seen == ["nous"]


def test_invalid_answer_classifies_as_incomplete_response():
    from hermes_agent.agent.error_classifier import classify_api_error

    assert classify_api_error(InvalidProviderResponseError("x")).reason is FailoverReason.incomplete_response


def test_tls_certificate_failure_keeps_compressions_session_preserving_abort():
    import ssl

    from hermes_agent.agent.context_compressor import ContextCompressor

    with patch("hermes_agent.agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2,
                                       abort_on_summary_failure=False)
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(12)]
    error = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    with patch("hermes_agent.agent.context_compressor.call_llm", side_effect=error):
        assert compressor.compress(msgs, current_tokens=999999, force=True) == msgs
    assert compressor._last_summary_network_failure and compressor._last_compress_aborted


def test_relay_connection_reset_502_keeps_compressions_session_preserving_abort():
    from hermes_agent.agent.context_compressor import ContextCompressor

    with patch("hermes_agent.agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2,
                                       abort_on_summary_failure=False)
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(12)]
    with patch("hermes_agent.agent.context_compressor.call_llm", side_effect=sdk(502, "upstream: connection reset by peer")):
        assert compressor.compress(msgs, current_tokens=999999, force=True) == msgs
    assert compressor._last_summary_network_failure


def test_pinned_custom_provider_entitlement_benches_only_the_model():
    from hermes_agent.agent.auxiliary_client import _recover_provider_pool
    from hermes_agent.agent.auxiliary_error_policy import classify_attempt

    error = sdk(400, "model kimi-k2 is not supported for this account")
    pool = MagicMock()
    with patch("hermes_agent.agent.auxiliary_client.pool_for_aux_provider", return_value=pool):
        _recover_provider_pool("custom:relay", error, verdict=classify_attempt(error, "custom:relay", model="kimi-k2"))
    assert pool.mark_exhausted_and_rotate.call_args.kwargs["model"] == "kimi-k2"
    key = Key("openrouter", "k1", "k1", "api_key", 0, "manual", "sk-or", model_cooldowns={"openai/gpt-gated": 9e9})
    with patch("hermes_agent.agent.auxiliary_client.load_pool", lambda _p: CredentialPool("openrouter", [key])):  # sibling models keep the key
        assert [_try_openrouter(model=m)[0] is None for m in ("gpt-ok", "gpt-gated")] == [False, True]


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_a_key_cooled_down_for_the_model_is_not_peeked_after_a_sibling_selected_it(monkeypatch, is_async):
    """k0 is refused for gpt-gated and still current after serving gpt-ok: a gpt-gated aux call peeks k1 and sends
    its one request with k1."""
    from hermes_agent.agent import auxiliary_client as aux

    keys = [Key("openrouter", "k0", "k0", "api_key", 0, "manual", "cooled-key",
                model_cooldowns={"openai/gpt-gated": 9e9}),
            Key("openrouter", "k1", "k1", "api_key", 1, "manual", "eligible-key")]
    pool = CredentialPool("openrouter", keys)
    assert pool.select(model="openai/gpt-ok").id == "k0"
    assert [pool.peek(model="openai/gpt-gated").id, aux._peek_pool_entry("openrouter", pool, "gpt-gated").id,
            pool.peek().id] == ["k1", "k1", "k0"]
    sent, ok = [], {"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
    transport = httpx.MockTransport(lambda r: sent.append(r.headers["authorization"]) or httpx.Response(200, json=ok))
    monkeypatch.setattr(aux, "_client_cache", {})
    with (patch("hermes_agent.agent.auxiliary_client.load_pool", lambda _p: pool),
          patch("hermes_agent.agent.auxiliary_client._openai_http_client_kwargs", lambda _url, async_mode=False: {
              "http_client": (httpx.AsyncClient if async_mode else httpx.Client)(transport=transport)}),
          patch("hermes_agent.agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp)):
        call = (async_call_llm if is_async else call_llm)(provider="openrouter", model="openai/gpt-gated",
                                                          messages=_MESSAGES)
        asyncio.run(call) if is_async else call
    assert sent == ["Bearer eligible-key"]


def test_model_entitlement_cooldown_keeps_the_key_for_its_other_models():
    """A ChatGPT account refused one Codex model (#71970), or an Anthropic key one Claude model, still serves aux
    calls on its other models."""
    from hermes_agent.agent.auxiliary_client import _build_codex_client, resolve_provider_client

    key = Key("openai-codex", "k1", "k1", "api_key", 0, "manual", "tok", model_cooldowns={"gpt-5.5-pro": 9e9})
    with (patch("hermes_agent.agent.auxiliary_client.load_pool", lambda _p: CredentialPool("openai-codex", [key])),
          patch("hermes_agent.agent.auxiliary_client._read_codex_singleton_token", return_value=None)):
        assert [_build_codex_client(m)[0] is None for m in ("gpt-5.4-mini", "gpt-5.5-pro")] == [False, True]
        assert [resolve_provider_client("openai-codex", model=m, raw_codex=True)[0] is None
                for m in ("gpt-5.4-mini", "gpt-5.5-pro")] == [False, True]
    key = Key("anthropic", "k1", "k1", "api_key", 0, "manual", "sk-ant-api", model_cooldowns={"claude-opus-4-6": 9e9})
    with (patch("hermes_agent.agent.auxiliary_client.load_pool", lambda _p: CredentialPool("anthropic", [key])),
          patch("hermes_agent.agent.anthropic_credentials.resolve_anthropic_token", return_value=None)):
        assert [resolve_provider_client("anthropic", model=m)[0] is None
                for m in ("claude-haiku-4-5", "claude-opus-4-6")] == [False, True]


def test_missing_model_on_a_candidate_hides_only_that_model(monkeypatch):
    from hermes_agent.agent import auxiliary_client as aux

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
        "hermes_agent.agent.auxiliary_client", _resolve_task_provider_model=lambda *_a: ("openrouter", "kimi-k2", None, None, None),
        _get_cached_client=lambda *_a, **_k: (clients[0], "kimi-k2"), _prepare_same_provider_retry=lambda **_k: (
            clients[1], {"model": "glm-5"}), _recover_provider_pool=lambda _p, _e, **kw: models.append(kw["verdict"].model) or True,
    ), pytest.raises(openai.APIError):
        call_llm(task="title_generation", messages=_MESSAGES)
    assert models == ["kimi-k2", "glm-5"]  # the verdict models handed to pool rotation, in order


def _candidate_outcome(first, refreshed):
    from hermes_agent.agent.auxiliary_client import _call_fallback_candidate_sync

    marks, raised = [], None
    with (
        patch("hermes_agent.agent.auxiliary_client._mark_provider_unhealthy", side_effect=lambda *a, **k: marks.append(k)),
        patch("hermes_agent.agent.auxiliary_client._refresh_provider_credentials", return_value=True),
        patch("hermes_agent.agent.auxiliary_client._get_cached_client", return_value=(refreshed, "fb-model")),
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
        "hermes_agent.agent.auxiliary_client", _resolve_task_provider_model=lambda *_a: ("openrouter", "kimi-k2", None, None, None),
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
    from hermes_agent.agent import auxiliary_error_policy as policy, auxiliary_stream_watchdog as wd

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

    client = _wire(handle, key="sk", is_async=is_async)
    if refusal is None:  # the SDK would wrap it as a connection error: raise it as the stream guard does
        client.chat.completions.create = (AsyncMock if is_async else MagicMock)(
            side_effect=lambda **kw: handle(None))
    with patch("hermes_agent.agent.auxiliary_client._aux_progress_active", return_value=stream):
        outcome = run_ladder(None, provider="nous", is_async=is_async, client=client,
                             max_tokens=16, temperature=0.3)
    assert (len(sent), outcome.fell_back) == (1, False), outcome.result


def _routed_call(clients, *, provider, is_async, pool=None, health=None, **call_kw):
    """``call_llm`` on ``provider`` whose client builds return ``clients`` in turn (an auto route re-resolving
    its retry); the configured chains are empty and the last-resort fallback answers. ``health``: a fresh dict the
    real provider-health marks land in (else marking is a no-op)."""
    builds, fallback = iter(clients), MagicMock(base_url="https://api.example.com/v1")
    fallback.chat.completions.create = (AsyncMock if is_async else MagicMock)(return_value=_ok())
    marks = {"_mark_provider_unhealthy": lambda *_a, **_k: None} if health is None else {
        "_aux_unhealthy_until": health, "_aux_unhealthy_reason": {}, "_aux_unhealthy_logged_at": {}}
    with patch.multiple(
        "hermes_agent.agent.auxiliary_client", _resolve_task_provider_model=lambda *_a: (provider, "fast-1", None, None, None),
        _get_cached_client=lambda *_a, **_k: (next(builds), "fast-1"), _transient_retry_count=lambda *_a: 0,
        _refresh_provider_credentials=lambda *_a, **_k: True, **marks,
        _try_configured_fallback_chain=lambda *_a, **_k: (None, None, ""),
        _try_main_fallback_chain=lambda *_a, **_k: (None, None, ""),
        _try_payment_fallback=lambda *_a, **_k: (fallback, "fb", "fallback"),
        _try_main_agent_model_fallback=lambda *_a, **_k: (fallback, "fb", "fallback"),
    ), patch("hermes_agent.agent.auxiliary_client.load_pool", lambda _p: pool):
        if is_async:
            result = asyncio.run(async_call_llm(task="title_generation", messages=_MESSAGES, **call_kw))
        else:
            result = call_llm(task="title_generation", messages=_MESSAGES, **call_kw)
    return result.choices[0].message.content


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("retry_url,benched", [("https://api.deepseek.com/v1", None), ("https://openrouter.ai/api/v1", 402)],
                         ids=["retry-on-another-backend", "retry-on-the-same-pool"])
def test_a_rebuilt_retry_benches_only_the_key_that_sent_it(is_async, retry_url, benched):
    """The pool rotates off k0 (model not served to this account); the rebuilt retry's 402 benches k1 only when
    k1 sent it: an auto route that re-resolved to another backend leaves this pool's unused key alone, and the
    provider-health mark lands on the backend that answered 402, not on OpenRouter."""
    from hermes_agent.agent import auxiliary_client as aux

    sent, health = [], {}
    def handle(status, message):
        return lambda r: sent.append(r.url.host) or httpx.Response(status, json={"error": {"message": message}}, request=r)
    pool = CredentialPool("openrouter", [Key("openrouter", f"k{i}", f"k{i}", "api_key", i, "manual", f"sk-{i}")
                                         for i in range(2)])
    clients = [_wire(handle(400, "model fast-1 is not supported for this account"), key="sk-0", is_async=is_async),
               _wire(handle(402, "Insufficient Balance"), key="sk-1", url=retry_url, is_async=is_async)]
    if benched is None:  # api-key discovery tags the client with the provider it was built for
        clients[1]._hermes_aux_backend_provider = "deepseek"
    assert _routed_call(clients, provider="auto", is_async=is_async, pool=pool, health=health) == "fallback"
    k0, k1 = pool.entries()
    assert sent == ["openrouter.ai", httpx.URL(retry_url).host]
    assert (sorted(k0.model_cooldowns or {}), k1.last_error_code) == (["fast-1"], benched)
    with patch.object(aux, "_aux_unhealthy_until", health):
        assert [aux._is_provider_unhealthy(p) for p in ("openrouter", "deepseek")] == [bool(benched), not benched]


_TEMPERATURE_400 = "Unsupported parameter: 'temperature' is not supported with this model."


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("third,answer", [(200, "ok"), (400, "fallback")], ids=["refreshed-request-answers",
                                                                                  "rejected-again-falls-back"])
@pytest.mark.parametrize("second,provider", [(401, "openrouter"), (402, "openrouter"), (401, "nous"), (429, "openrouter")],
                         ids=["refresh", "rotation", "nous-refresh", "rate-limit-backoff"])
def test_a_parameter_repair_survives_a_credential_refresh(is_async, third, answer, second, provider):
    """temperature 400 -> the repaired request 401s (refresh, Nous refresh), 402s (pool rotation) or 429s (backoff):
    the resent request still omits temperature. If the provider rejects it anyway, that is a parameter-named 400
    after a spent rung: model incompatible, fall back."""
    sent, statuses = [], iter([400, second, third])
    ok = {"id": "x", "object": "chat.completion", "created": 0, "model": "fast-1", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}

    def handle(request):
        sent.append(json.loads(request.content))
        status = next(statuses)
        message = {400: _TEMPERATURE_400, 401: "invalid api key", 402: "Insufficient credits",
                   429: "Rate limit exceeded"}.get(status)
        return httpx.Response(status, json={"error": {"message": message}} if message else ok, request=request)

    client = _wire(handle, key="sk-0", is_async=is_async)
    pool = CredentialPool("openrouter", [Key("openrouter", f"k{i}", f"k{i}", "api_key", i, "manual", f"sk-{i}")
                                         for i in range(2)])
    with patch.multiple("hermes_agent.agent.auxiliary_client", _refresh_nous_auxiliary_client=lambda **_k: (client, "fast-1"),
                        _nous_portal_account_has_fresh_paid_access=lambda: False), \
            patch("hermes_agent.agent.auxiliary_rate_limit_backoff.rate_limit_wait", return_value=0.0):
        assert _routed_call([client] * 2, provider=provider, is_async=is_async, temperature=0.3,
                            pool=pool if provider == "openrouter" else None) == answer
    assert len(sent) == 3 and ["temperature" in body for body in sent] == [True, False, False]


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("refreshed,answer", [(200, "fresh"), (400, "next")], ids=["refreshed-request-answers",
                                                                                 "rejected-again-takes-next"])
def test_a_fallback_candidates_repair_survives_its_credential_refresh(is_async, refreshed, answer):
    """primary 503 -> candidate temperature 400 -> the stripped request 401s -> refresh: the refreshed request still
    omits temperature; if it is rejected anyway, that candidate is quarantined and the next one answers."""
    sent, statuses = [], {"sk-primary": iter([503] * 2), "sk-old": iter([400, 401]), "sk-fresh": iter([refreshed]),
                          "sk-next": iter([200])}

    def handle(request):
        key = request.headers["authorization"].removeprefix("Bearer ")
        sent.append((key, "temperature" in json.loads(request.content)))
        status = next(statuses[key])
        message = {400: _TEMPERATURE_400 if key == "sk-old" else "Invalid request.", 401: "invalid api key",
                   503: "down"}.get(status)
        return httpx.Response(status, json={"error": {"message": message}} if message else {
            "id": "x", "object": "chat.completion", "created": 0, "model": "fb", "choices": [{"index": 0, "message": {
                "role": "assistant", "content": key.removeprefix("sk-")}, "finish_reason": "stop"}]}, request=request)

    keys = iter(["sk-primary", "sk-fresh"])  # the primary's client, then the candidate's refreshed rebuild
    candidates = iter([(_wire(handle, key="sk-old"), "fb", "openrouter"),
                       (_wire(handle, key="sk-next", url="https://api.example.com/v1"), "fb", "custom")])
    with patch.multiple(
        "hermes_agent.agent.auxiliary_client", _resolve_task_provider_model=lambda *_a: ("openrouter", "fast-1", None, None, None),
        _get_cached_client=lambda *_a, **_k: (_wire(handle, key=next(keys), is_async=is_async), "fb"),
        _transient_retry_count=lambda *_a: 0, _refresh_provider_credentials=lambda *_a, **_k: True,
        _mark_provider_unhealthy=lambda *_a, **_k: None, _recoverable_pool_provider=lambda *_a, **_k: None,
        _try_configured_fallback_chain=lambda *_a, **_k: next(candidates, (None, None, "")),
        _try_main_agent_model_fallback=lambda *_a, **_k: (None, None, ""),
        _openai_http_client_kwargs=lambda *_a, **_k: {"http_client": httpx.AsyncClient(
            transport=httpx.MockTransport(handle))}):
        call = async_call_llm(task="title_generation", messages=_MESSAGES, temperature=0.3) if is_async else None
        result = asyncio.run(call) if is_async else call_llm(task="title_generation", messages=_MESSAGES, temperature=0.3)
    assert result.choices[0].message.content == answer
    fallbacks = [request for request in sent if request[0] != "sk-primary"]
    assert fallbacks == [("sk-old", True), ("sk-old", False), ("sk-fresh", False)] + [("sk-next", True)] * (
        answer == "next")


def _refreshed_candidate_walk(is_async, next_status):
    """Primary 503 -> the REAL configured fallback_chain: [0] old-model temperature 400 -> stripped 401 -> refresh
    rebuilds on new-model -> 404, then [1] answers with ``next_status``. Returns (answer or error, sent, marked)."""
    from hermes_agent.agent import auxiliary_client as aux

    health, sent = {}, []
    statuses = {"sk-primary": iter([503] * 2), "sk-old": iter([400, 401]), "sk-fresh": iter([404]),
                "sk-next": iter([next_status])}

    def handle(request):
        key, body = request.headers["authorization"].removeprefix("Bearer "), json.loads(request.content)
        sent.append((key, body["model"], "temperature" in body))
        status = next(statuses[key])
        message = {400: _TEMPERATURE_400, 401: "invalid api key", 404: "model_not_found: The model 'new-model' does not exist",
                   503: "down"}.get(status)
        return httpx.Response(status, json={"error": {"message": message}} if message else {
            "id": "x", "object": "chat.completion", "created": 0, "model": "fb", "choices": [{"index": 0, "message": {
                "role": "assistant", "content": key.removeprefix("sk-")}, "finish_reason": "stop"}]}, request=request)

    chain = [{"provider": "openrouter", "model": "old-model"},
             {"provider": "custom", "model": "fb", "base_url": "https://api.example.com/v1", "api_key": "sk-next"}]
    builds = iter([("sk-primary", "fast-1"), ("sk-fresh", "new-model")])  # the primary, then the refreshed rebuild

    def build(*_a, **_k):
        key, model = next(builds)
        return _wire(handle, key=key, is_async=is_async), model

    def resolve(provider, model=None, explicit_base_url=None, explicit_api_key=None, **_k):  # a chain entry's client
        return _wire(handle, key=explicit_api_key or "sk-old", url=explicit_base_url or "https://openrouter.ai/api/v1"), model

    with patch.multiple(
        "hermes_agent.agent.auxiliary_client", _resolve_task_provider_model=lambda *_a: ("openrouter", "fast-1", None, None, None),
        _get_cached_client=build, _transient_retry_count=lambda *_a: 0, _refresh_provider_credentials=lambda *_a, **_k: True,
        _aux_unhealthy_until=health, _aux_unhealthy_reason={}, _aux_unhealthy_logged_at={},
        _recoverable_pool_provider=lambda *_a, **_k: None, resolve_provider_client=resolve,
        _get_auxiliary_task_config=lambda *_a, **_k: {"fallback_chain": chain},
        _try_main_agent_model_fallback=lambda *_a, **_k: (None, None, ""),
        _openai_http_client_kwargs=lambda *_a, **_k: {"http_client": httpx.AsyncClient(
            transport=httpx.MockTransport(handle))}):
        call = async_call_llm(task="title_generation", messages=_MESSAGES, temperature=0.3) if is_async else None
        try:
            result = asyncio.run(call) if is_async else call_llm(task="title_generation", messages=_MESSAGES, temperature=0.3)
            outcome = result.choices[0].message.content
        except openai.APIStatusError as exc:
            outcome = exc.status_code
        route = "https://openrouter.ai/api/v1"
        marked = [aux._is_provider_unhealthy("openrouter", route, model) for model in ("new-model", "old-model")]
    return outcome, sent, marked


_REFRESHED_WALK = [("sk-old", "old-model", True), ("sk-old", "old-model", False), ("sk-fresh", "new-model", False),
                   ("sk-next", "fb", True)]


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_a_refreshed_candidate_on_another_model_quarantines_the_model_that_failed(is_async):
    """new-model is quarantined on that route, old-model stays eligible, and the walk still reaches the next entry
    (old-model's re-selected lane does not end it)."""
    outcome, sent, marked = _refreshed_candidate_walk(is_async, 200)
    assert outcome == "next" and marked == [True, False]
    assert [request for request in sent if request[0] != "sk-primary"] == _REFRESHED_WALK


@pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])
def test_a_walk_where_every_entry_fails_tries_each_lane_once(is_async):
    """Every entry fails: each lane is sent once, then the primary error is raised."""
    outcome, sent, marked = _refreshed_candidate_walk(is_async, 503)
    assert outcome == 503 and marked == [True, False]
    assert [request for request in sent if request[0] != "sk-primary"] == _REFRESHED_WALK
