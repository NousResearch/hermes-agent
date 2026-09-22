"""The auxiliary recovery ladder must not let a failed auth-refresh retry escape.

Both auth-refresh rungs in ``agent/auxiliary_client.py`` perform their retry with a
bare ``yield`` inside a ``return`` statement:

    if _is_auth_error(first_err) and client_is_nous:
        step = _refreshed_nous_step(...)
        if step is not None:
            return (yield step), None          # nous rung
    ...
            return (yield _LadderStep(
                "retry_same_provider", ...)), None   # generic credential rung

``_rung()`` exists to convert a retry failure into ``(None, exc)`` -- but only when the
rung's accept predicate claims the error -- so the caller can fall through to the next
rung. An unclaimed failure (a 500, a malformed response) re-raises on purpose, since
``_ladder_provider_fallback`` only acts on the reasons in ``_FALLBACK_REASONS``.
Used this way no exception is caught: when the
refreshed client also fails (e.g. an out-of-credit 404 on a stale Nous runtime token),
the error escapes ``_aux_recovery_ladder`` and ``_ladder_provider_fallback`` never
runs -- the configured ``auxiliary.<task>.fallback_chain`` is silently skipped. The
rung right below (credential-pool rotation) documents the intended behavior with "then
fall through to the provider fallback" and guards its retry with try/except.

The first test drives the real ladder generator with a scripted driver. The second
exercises the real path end to end: a temp ``HERMES_HOME`` whose ``config.yaml``
declares a ``fallback_chain``, the real client construction and HTTP layer against a
local endpoint, with only the credential sources stubbed (the Nous portal account
probe and the runtime-credential fetch are external boundaries).
"""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
from typing import Optional

import pytest
import yaml

import agent.auxiliary_client as aux

AUX_MODEL = "z-ai/glm-5.3-flash"
FALLBACK_MODEL = "fallback-model"
NOUS_HOST = "inference-api.nousresearch.com"


class _ApiError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _auth_error():
    return _ApiError("Error code: 401 - Unauthorized", status_code=401)


def _credit_error():
    return _ApiError(
        "Error code: 404 - Model '%s' requires available credits. "
        "Your account balance is too low to use paid models." % AUX_MODEL,
        status_code=404,
    )


class _FakeClient:
    api_key = "sk-test"
    base_url = "https://%s/v1" % NOUS_HOST


class _ExplicitProviderClient:
    api_key = "stale-key"
    base_url = "https://vertex.example/v1"


def _ladder(base_info=("https://%s/v1" % NOUS_HOST), resolved_provider="nous"):
    return aux._aux_recovery_ladder(
        _auth_error(),
        client=_FakeClient(),
        kwargs={"model": AUX_MODEL},
        task="compression",
        async_mode=False,
        base_info=base_info,
        resolved_provider=resolved_provider,
        resolved_model=AUX_MODEL,
        resolved_base_url=None,
        resolved_api_key=None,
        resolved_api_mode=None,
        final_model=AUX_MODEL,
        max_tokens=None,
        main_runtime=None,
        route_info={},
    )


@pytest.fixture
def hermetic(monkeypatch):
    """Keep the ladder off the network and record the provider-fallback rung."""
    chain_calls = []

    def _fake_provider_fallback(first_err, route):
        """Stands in for the last rung: a generator that performs no steps."""
        chain_calls.append(first_err)
        yield from ()
        return "chain-response"

    monkeypatch.setattr(aux, "_recoverable_pool_provider", lambda *a, **kw: None)
    monkeypatch.setattr(aux, "_nous_portal_account_has_fresh_paid_access", lambda: False)
    monkeypatch.setattr(aux, "_ladder_provider_fallback", _fake_provider_fallback)
    return chain_calls


@pytest.mark.parametrize(
    "rung,retry_succeeds",
    [("nous", False), ("nous", True), ("provider_credential", False)],
)
def test_post_refresh_retry_owns_the_ladder_outcome(
        rung, retry_succeeds, monkeypatch, hermetic):
    """A failed retry resumes the ladder; a successful one returns its response."""
    if rung == "nous":
        monkeypatch.setattr(aux, "_refresh_nous_auxiliary_client",
                            lambda **kwargs: (_FakeClient(), AUX_MODEL))
        expected_step, expected_base = "call", ("https://%s/v1" % NOUS_HOST)
    else:
        monkeypatch.setattr(aux, "_auth_refresh_provider_for_route",
                            lambda *a, **kw: "codex")
        monkeypatch.setattr(aux, "_refresh_provider_credentials", lambda *a, **kw: True)
        monkeypatch.setattr(aux, "_evict_cached_clients", lambda *a, **kw: None)
        expected_step, expected_base = "retry_same_provider", "https://openrouter.ai/api/v1"

    ladder = _ladder(
        base_info=expected_base,
        resolved_provider="nous" if rung == "nous" else "openrouter",
    )
    performed = []
    failure = _credit_error()

    def perform(step):
        performed.append(step.kind)
        if retry_succeeds:
            return "refreshed-client-response"
        raise failure

    if retry_succeeds:
        assert aux._drive_ladder(ladder, perform) == "refreshed-client-response"
        assert hermetic == [], "a successful retry must not reach the fallback chain"
        return

    try:
        result = aux._drive_ladder(ladder, perform)
    except _ApiError as exc:
        pytest.fail(
            "the ladder let %r escape instead of falling through to the configured "
            "fallback chain (steps performed: %s)" % (exc, performed)
        )

    assert performed == [expected_step], "the post-refresh retry is the only request"
    assert hermetic, "the configured fallback chain must be consulted"
    assert "requires available credits" in str(hermetic[0])
    assert result == "chain-response"


@pytest.mark.parametrize("spare_survives", [True, False])
def test_explicit_provider_auth_uses_its_configured_task_fallback(monkeypatch, spare_survives):
    """An explicit route may leave a 401 only through its own configured chain — including the
    re-walk after a chain entry is quarantined mid-request; an exhausted chain raises the primary
    error instead of spilling onto discovery / the main model."""
    dead_client, fallback_client = _ExplicitProviderClient(), _FakeClient()
    chain = [("fallback_chain[0](custom:dead)", dead_client)]
    if spare_survives:
        chain.append(("fallback_chain[1](custom:backup)", fallback_client))
    monkeypatch.setattr(
        aux,
        "_get_auxiliary_task_config",
        lambda task: {"fallback_chain": [{"provider": "custom:dead"}, {"provider": "custom:backup"}]},
    )
    monkeypatch.setattr(aux, "_auth_refresh_provider_for_route", lambda *args, **kwargs: "vertex")
    monkeypatch.setattr(aux, "_refresh_provider_credentials", lambda *args, **kwargs: False)
    monkeypatch.setattr(aux, "_recoverable_pool_provider", lambda *args, **kwargs: None)
    for name in ("_try_payment_fallback", "_try_main_fallback_chain", "_try_main_agent_model_fallback"):
        monkeypatch.setattr(aux, name, lambda *a, _n=name, **k: pytest.fail(f"{_n} must stay gated for explicit auth"))

    def configured_chain(*args, **kwargs):
        if not chain:
            return None, None, ""
        label, client = chain.pop(0)
        return client, FALLBACK_MODEL, label

    monkeypatch.setattr(aux, "_try_configured_fallback_chain", configured_chain)
    ladder = aux._aux_recovery_ladder(
        _auth_error(),
        client=_ExplicitProviderClient(),
        kwargs={"model": AUX_MODEL},
        task="compression",
        async_mode=False,
        base_info="https://vertex.example/v1",
        resolved_provider="vertex",
        resolved_model=AUX_MODEL,
        resolved_base_url=None,
        resolved_api_key=None,
        resolved_api_mode=None,
        final_model=AUX_MODEL,
        max_tokens=None,
        main_runtime=None,
        route_info={},
    )

    def perform(step):
        assert step.kind == "fallback"
        if step.args[0] is dead_client:
            return None  # quarantined mid-request → the ladder re-walks the chain
        assert step.args == (fallback_client, FALLBACK_MODEL, "fallback_chain[1](custom:backup)")
        return "fallback-response"

    if spare_survives:
        assert aux._drive_ladder(ladder, perform) == "fallback-response"
    else:
        with pytest.raises(_ApiError, match="Unauthorized"):
            aux._drive_ladder(ladder, perform)
    assert not chain


def _model_not_found_error():
    return _ApiError(
        "Error code: 400 - {'error': {'message': 'unknown provider for "
        "model gemini-flash-lite-latest', 'type': 'invalid_request_error', "
        "'code': 'model_not_found', 'param': 'model'}}",
        status_code=400,
    )


def test_explicit_provider_model_not_found_uses_task_fallback_chain(monkeypatch):
    """A 400 model_not_found on an explicit aux route must walk fallback_chain.

    Hub compression pinned gemini-flash-lite-latest via cliproxyapi. When the
    router said unknown provider, the ladder skipped the configured chain and
    the Telegram session could not compress. Capacity-class: the named model
    cannot be served, so the task chain is in play.
    """
    fallback_client = _FakeClient()
    monkeypatch.setattr(
        aux,
        "_get_auxiliary_task_config",
        lambda task: {
            "fallback_chain": [
                {"provider": "cliproxyapi", "model": "gemini-3.7-flash-high"},
            ]
        },
    )
    monkeypatch.setattr(aux, "_auth_refresh_provider_for_route", lambda *a, **k: None)
    monkeypatch.setattr(aux, "_recoverable_pool_provider", lambda *a, **k: None)
    monkeypatch.setattr(
        aux,
        "_try_configured_fallback_chain",
        lambda *a, **k: (fallback_client, FALLBACK_MODEL, "fallback_chain[0](cliproxyapi)"),
    )
    for name in ("_try_payment_fallback", "_try_main_fallback_chain"):
        monkeypatch.setattr(
            aux, name, lambda *a, _n=name, **k: pytest.fail("%s must stay gated" % _n)
        )

    ladder = aux._aux_recovery_ladder(
        _model_not_found_error(),
        client=_ExplicitProviderClient(),
        kwargs={"model": "gemini-flash-lite-latest"},
        task="compression",
        async_mode=False,
        base_info="http://127.0.0.1:8318/v1",
        resolved_provider="cliproxyapi",
        resolved_model="gemini-flash-lite-latest",
        resolved_base_url=None,
        resolved_api_key=None,
        resolved_api_mode=None,
        final_model="gemini-flash-lite-latest",
        max_tokens=None,
        main_runtime=None,
        route_info={},
    )

    def perform(step):
        assert step.kind == "fallback"
        assert step.args == (
            fallback_client, FALLBACK_MODEL, "fallback_chain[0](cliproxyapi)",
        )
        return "fallback-response"

    assert aux._drive_ladder(ladder, perform) == "fallback-response"


def test_explicit_provider_auth_never_uses_an_unconfigured_fallback(monkeypatch):
    """A 401 without a task chain preserves the explicit-provider boundary."""
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {})
    monkeypatch.setattr(aux, "_auth_refresh_provider_for_route", lambda *args, **kwargs: "vertex")
    monkeypatch.setattr(aux, "_refresh_provider_credentials", lambda *args, **kwargs: False)
    monkeypatch.setattr(aux, "_recoverable_pool_provider", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        aux,
        "_try_main_agent_model_fallback",
        lambda *args, **kwargs: pytest.fail("explicit auth must not use the main-agent fallback"),
    )
    ladder = aux._aux_recovery_ladder(
        _auth_error(),
        client=_ExplicitProviderClient(),
        kwargs={"model": AUX_MODEL},
        task="compression",
        async_mode=False,
        base_info="https://vertex.example/v1",
        resolved_provider="vertex",
        resolved_model=AUX_MODEL,
        resolved_base_url=None,
        resolved_api_key=None,
        resolved_api_mode=None,
        final_model=AUX_MODEL,
        max_tokens=None,
        main_runtime=None,
        route_info={},
    )

    with pytest.raises(_ApiError, match="Unauthorized"):
        aux._drive_ladder(ladder, lambda step: pytest.fail("no fallback request expected"))


@pytest.fixture
def nous_ladder_endpoint(monkeypatch):
    """A local endpoint that answers the Nous host, recording every request.

    The routing decision that selects the auth-refresh rung matches on the base URL
    host, so the endpoint is addressed as the real Nous host and ``getaddrinfo`` is
    redirected to the loopback server.
    """
    aux.shutdown_cached_clients()
    aux._reset_aux_unhealthy_cache()
    requests = []
    resolve_address = socket.getaddrinfo

    def local_nous_address(host, *args, **kwargs):
        if host in (NOUS_HOST, NOUS_HOST.encode()):
            host = "127.0.0.1"
        return resolve_address(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", local_nous_address)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost,%s" % NOUS_HOST)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            model = payload.get("model")
            if model is None:
                self._send(400, {"error": {"message": "unexpected payload without model"}})
                return
            if model == AUX_MODEL:
                # Stale runtime token: 401 on the first attempt, then the refreshed
                # client hits a model the account cannot pay for.
                if sum(1 for _p, body in requests if body["model"] == AUX_MODEL) == 1:
                    self._send(401, {"error": {"message": "Unauthorized", "type": "authentication_error"}})
                    return
                self._send(404, {"error": {
                    "message": "Model '%s' requires available credits. Your account "
                               "balance is too low to use paid models." % AUX_MODEL,
                    "type": "invalid_request_error",
                    "code": "insufficient_credits",
                }})
                return
            self._send(200, {
                "id": "chatcmpl-ladder",
                "created": 1,
                "model": model,
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "The task is complete."},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            })

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield "http://%s:%d" % (NOUS_HOST, server.server_port), requests
    finally:
        aux.shutdown_cached_clients()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_auth_refresh_retry_failure_reaches_the_configured_chain_over_http(
        tmp_path, monkeypatch, nous_ladder_endpoint):
    """End to end: the configured chain must serve the retry the refresh could not."""
    host_url, requests = nous_ladder_endpoint
    local_url = "http://127.0.0.1:%s" % host_url.rsplit(":", 1)[1]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("AUX_FB_KEY", "fallback-test-key")
    config = {
        "model": {"provider": "nous", "default": AUX_MODEL},
        "providers": {"aux-fb": {"base_url": local_url + "/v1", "key_env": "AUX_FB_KEY"}},
        "auxiliary": {
            "compression": {
                "provider": "custom",
                "model": AUX_MODEL,
                "base_url": host_url + "/v1",
                "timeout": 20,
                "fallback_chain": [
                    {"provider": "custom:aux-fb", "model": FALLBACK_MODEL,
                     "base_url": local_url + "/v1", "timeout": 20}
                ],
            }
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    # Credential boundaries: the account probe and the runtime-credential fetch.
    monkeypatch.setattr(aux, "_nous_portal_account_has_fresh_paid_access", lambda: False)
    monkeypatch.setattr(
        aux, "_resolve_nous_runtime_api",
        lambda **kwargs: ("fresh-nous-key", host_url + "/v1"),
    )

    response = aux.call_llm(
        task="compression",
        messages=[{"role": "user", "content": "Summarize the conversation so far."}],
        timeout=20,
    )

    assert response.choices[0].message.content == "The task is complete."
    seen = [
        body.get("model") for path, body in requests if path == "/v1/chat/completions"
    ]
    assert seen == [AUX_MODEL, AUX_MODEL, FALLBACK_MODEL], (
        "401, then the refreshed retry fails on credits, then the configured chain: %r"
        % (seen,)
    )


def test_exhausted_ladder_raises_the_narrowed_error(monkeypatch, hermetic):
    """No chain answers: the retry's own failure surfaces, not the healed 401."""
    monkeypatch.setattr(aux, "_refresh_nous_auxiliary_client",
                        lambda **kwargs: (_FakeClient(), AUX_MODEL))

    def _no_chain(first_err, route):
        hermetic.append(first_err)
        yield from ()
        return None

    monkeypatch.setattr(aux, "_ladder_provider_fallback", _no_chain)
    failure = _credit_error()

    def perform(step):
        raise failure

    with pytest.raises(_ApiError) as raised:
        aux._drive_ladder(_ladder(), perform)

    assert raised.value is failure, (
        "the ladder must surface the actionable retry failure, got %r" % (raised.value,))
    assert hermetic == [failure]
