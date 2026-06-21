"""Tests for the egress credential broker (agent/secret_broker.py).

All key-like strings in this file are synthetic fixtures.
"""

from __future__ import annotations

import concurrent.futures

import httpx
import pytest

import agent.secret_broker as secret_broker
from agent.secret_broker import (
    SecretBroker,
    apply_to_client_kwargs,
    broker_enabled,
    clear_broker_config_cache,
    get_broker,
    install_broker_signal_handler,
    install_request_hook,
    register_aws_credential_triplet,
)
from hermes_constants import get_hermes_home


@pytest.fixture(autouse=True)
def _reset_broker_config_cache():
    """Each test starts with empty broker config and secret state."""
    clear_broker_config_cache()
    with secret_broker._broker_lock:
        secret_broker._broker = SecretBroker()
    try:
        yield
    finally:
        clear_broker_config_cache()
        with secret_broker._broker_lock:
            secret_broker._broker = SecretBroker()


def _enable_broker() -> None:
    """Write a config.yaml that turns the broker on (hermetic HERMES_HOME)."""
    (get_hermes_home() / "config.yaml").write_text(
        "security:\n  credential_broker:\n    enabled: true\n", encoding="utf-8"
    )
    clear_broker_config_cache()


# ── SecretBroker core ───────────────────────────────────────────────────────


def test_register_is_deduplicated():
    broker = SecretBroker()
    a = broker.register("sk-secret")
    b = broker.register("sk-secret")
    assert a == b
    assert broker.is_placeholder(a)
    assert not broker.is_placeholder("sk-secret")
    assert not broker.is_placeholder("hermes-broker-not-a-real-placeholder")


def test_register_evicts_fifo_at_max_entries():
    """a long-running broker must not grow unbounded.

    Once max_entries is hit, the oldest entry evicts FIFO-style. Both
    direction maps must drop the evicted entry to keep them in sync.
    """
    broker = SecretBroker(max_entries=3)
    p1 = broker.register("first")
    p2 = broker.register("second")
    p3 = broker.register("third")
    # All resolvable.
    assert broker.resolve(p1) == "first"
    assert broker.resolve(p2) == "second"
    assert broker.resolve(p3) == "third"

    # Insert past cap — oldest ("first") evicts.
    p4 = broker.register("fourth")
    assert broker.resolve(p1) is None, "evicted secret should no longer resolve"
    assert broker.resolve(p2) == "second"
    assert broker.resolve(p3) == "third"
    assert broker.resolve(p4) == "fourth"
    # _by_secret and _by_placeholder stay in sync.
    assert "first" not in broker._by_secret
    assert p1 not in broker._by_placeholder


def test_register_dedup_refreshes_recency():
    """re-registering a still-active secret must refresh
    its recency so a burst of one-shots doesn't age out the hot key.
    """
    broker = SecretBroker(max_entries=3)
    p1 = broker.register("hot-key")
    broker.register("filler-a")
    broker.register("filler-b")
    # Re-register "hot-key" → moves it to the back (most recent).
    p1_again = broker.register("hot-key")
    assert p1 == p1_again
    # Insert a new entry — "filler-a" (now the oldest) should evict,
    # not the "hot-key" we just refreshed.
    broker.register("new-key")
    assert broker.resolve(p1) == "hot-key", (
        "hot-key was evicted despite being refreshed by re-registration"
    )


def test_resolve_round_trip():
    broker = SecretBroker()
    ph = broker.register("sk-secret-value")
    assert broker.resolve(ph) == "sk-secret-value"
    assert broker.resolve("hermes-broker-" + "0" * 32) is None


def test_resolve_in_rewrites_embedded_placeholder():
    broker = SecretBroker()
    ph = broker.register("sk-xyz")
    assert broker.resolve_in(f"Bearer {ph}") == "Bearer sk-xyz"
    assert broker.resolve_in("Bearer untouched") == "Bearer untouched"


def test_register_aws_credential_triplet_handles_optional_token():
    broker = get_broker()

    id_placeholder, secret_placeholder, token_placeholder = (
        register_aws_credential_triplet("aws-id-none", "aws-secret-none")
    )
    assert broker.resolve(id_placeholder) == "aws-id-none"
    assert broker.resolve(secret_placeholder) == "aws-secret-none"
    assert token_placeholder is None

    _id_empty, _secret_empty, empty_token_placeholder = (
        register_aws_credential_triplet("aws-id-empty", "aws-secret-empty", "")
    )
    assert empty_token_placeholder is None

    _id_token, _secret_token, real_token_placeholder = (
        register_aws_credential_triplet(
            "aws-id-token",
            "aws-secret-token",
            "aws-session-token",
        )
    )
    assert real_token_placeholder is not None
    assert SecretBroker.is_placeholder(real_token_placeholder)
    assert broker.resolve(real_token_placeholder) == "aws-session-token"


# ── httpx request hook ──────────────────────────────────────────────────────


def test_request_hook_rewrites_authorization_header():
    broker = SecretBroker()
    ph = broker.register("sk-real-key")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    install_request_hook(client, broker)
    client.get("https://api.test/v1", headers={"Authorization": f"Bearer {ph}"})
    assert seen["auth"] == "Bearer sk-real-key"


def test_request_hook_rewrites_x_api_key_header():
    broker = SecretBroker()
    ph = broker.register("sk-ant-real")
    seen = {}

    def handler(request):
        seen["key"] = request.headers.get("x-api-key")
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    install_request_hook(client, broker)
    client.get("https://api.test/v1", headers={"x-api-key": ph})
    assert seen["key"] == "sk-ant-real"


def test_request_hook_is_idempotent():
    broker = SecretBroker()
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    install_request_hook(client, broker)
    install_request_hook(client, broker)
    assert len(client.event_hooks["request"]) == 1


def test_request_hook_concurrent_install_is_idempotent():
    broker = SecretBroker()
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(install_request_hook, client, broker) for _ in range(32)]
        for future in futures:
            future.result()

    assert len(client.event_hooks["request"]) == 1


@pytest.mark.asyncio
async def test_request_hook_works_on_async_client():
    broker = SecretBroker()
    ph = broker.register("sk-async-key")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    install_request_hook(client, broker)
    await client.get("https://api.test/v1", headers={"Authorization": f"Bearer {ph}"})
    await client.aclose()
    assert seen["auth"] == "Bearer sk-async-key"


# ── apply_to_client_kwargs (config-gated integration) ───────────────────────


def test_apply_is_noop_when_broker_disabled():
    kwargs = {
        "api_key": "sk-plain",
        "http_client": httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
    }
    apply_to_client_kwargs(kwargs)
    assert kwargs["api_key"] == "sk-plain"


def test_apply_placeholders_key_and_installs_hook_when_enabled():
    _enable_broker()
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    kwargs = {"api_key": "sk-real-enabled", "http_client": client}
    apply_to_client_kwargs(kwargs)
    assert SecretBroker.is_placeholder(kwargs["api_key"])
    assert get_broker().resolve(kwargs["api_key"]) == "sk-real-enabled"
    assert getattr(client, "_hermes_broker_hooked", False) is True


def test_apply_installs_hook_for_already_placeholder_key():
    _enable_broker()
    placeholder = get_broker().register("sk-real-reused")
    captured = {}

    def _handler(request):
        captured["authorization"] = request.headers.get("authorization")
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(_handler))
    kwargs = {"api_key": placeholder, "http_client": client}
    apply_to_client_kwargs(kwargs)

    assert kwargs["api_key"] == placeholder
    assert getattr(client, "_hermes_broker_hooked", False) is True
    client.get("https://example.invalid", headers={"Authorization": f"Bearer {placeholder}"})
    assert captured["authorization"] == "Bearer sk-real-reused"


def test_apply_skips_when_no_http_client():
    _enable_broker()
    kwargs = {"api_key": "sk-real-nohttp"}
    apply_to_client_kwargs(kwargs)
    # No Hermes httpx client to carry the egress hook — fail-open, key unchanged.
    assert kwargs["api_key"] == "sk-real-nohttp"


def test_broker_enabled_caches_load_config(monkeypatch):
    calls = {"n": 0}

    def _load_config():
        calls["n"] += 1
        return {}

    monkeypatch.setattr("hermes_cli.config.load_config", _load_config)
    assert broker_enabled() is False
    assert broker_enabled() is False
    assert calls["n"] == 1


def test_clear_broker_config_cache_forces_reload(monkeypatch):
    calls = {"n": 0}

    def _load_config():
        calls["n"] += 1
        return {}

    monkeypatch.setattr("hermes_cli.config.load_config", _load_config)
    broker_enabled()
    clear_broker_config_cache()
    broker_enabled()
    assert calls["n"] == 2


# ── AWS Bedrock broker wiring ───────────────────────────────────────────────


def _emit_bedrock_event(client, event_name, **kwargs):
    """Emit a botocore event on *client*'s meta event emitter."""
    client.meta.events.emit(event_name, **kwargs)


def _reset_boto3_default_session():
    """Force boto3 to re-read AWS credential env vars changed by a test."""
    try:
        import boto3
        boto3.DEFAULT_SESSION = None
    except ImportError:
        pass


def _sign_bedrock_request(signer, *, operation_name: str = "InvokeModel"):
    """Sign a synthetic Bedrock request without making a network call."""
    import botocore.awsrequest as ar

    request = ar.AWSRequest(
        method="POST",
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/test/invoke",
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    signer.sign(operation_name, request)
    return request


def test_bedrock_runtime_client_swaps_bearer_at_send_time(monkeypatch):
    """Broker on + AWS_BEARER_TOKEN_BEDROCK → before-send rewrites placeholder→real."""
    pytest.importorskip("boto3")
    pytest.importorskip("botocore")
    import botocore.awsrequest as ar
    from agent import bedrock_adapter
    from agent.secret_broker import SecretBroker

    _enable_broker()
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "real-bearer-XYZ")
    # Defend against the SigV4 branch picking up a stray access key
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    _reset_boto3_default_session()
    bedrock_adapter.reset_client_cache()

    client = bedrock_adapter._get_bedrock_runtime_client("us-east-1")

    # The signer's auth token has been swapped for a placeholder.
    placeholder_token = client._request_signer._auth_token.token
    assert SecretBroker.is_placeholder(placeholder_token)
    assert placeholder_token != "real-bearer-XYZ"

    # Build a fake outgoing request as botocore would (after BearerAuth wrote
    # the placeholder into the header) and emit before-send.
    headers = {"Authorization": f"Bearer {placeholder_token}"}
    req = ar.AWSPreparedRequest("POST", "https://example.invalid", headers, None, None)
    _emit_bedrock_event(
        client,
        "before-send.bedrock-runtime.InvokeModel",
        request=req,
    )

    assert req.headers["Authorization"] == "Bearer real-bearer-XYZ"
    # Idempotency marker exists.
    assert getattr(client.meta.events, "_hermes_broker_hooked", False) is True


def test_bedrock_control_client_swaps_bearer_at_send_time(monkeypatch):
    """Same as above but for the control-plane client."""
    pytest.importorskip("boto3")
    pytest.importorskip("botocore")
    import botocore.awsrequest as ar
    from agent import bedrock_adapter
    from agent.secret_broker import SecretBroker

    _enable_broker()
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "real-bearer-CTRL")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    _reset_boto3_default_session()
    bedrock_adapter.reset_client_cache()

    client = bedrock_adapter._get_bedrock_control_client("us-east-1")
    placeholder_token = client._request_signer._auth_token.token
    assert SecretBroker.is_placeholder(placeholder_token)

    headers = {"Authorization": f"Bearer {placeholder_token}"}
    req = ar.AWSPreparedRequest("GET", "https://example.invalid", headers, None, None)
    _emit_bedrock_event(
        client,
        "before-send.bedrock.ListFoundationModels",
        request=req,
    )
    assert req.headers["Authorization"] == "Bearer real-bearer-CTRL"


def test_bedrock_runtime_client_resolves_sigv4_only_for_signing(monkeypatch):
    """Broker on + AWS_ACCESS_KEY_ID pair → signer stays placeholder-only."""
    pytest.importorskip("boto3")
    pytest.importorskip("botocore")
    from agent import bedrock_adapter
    from agent.secret_broker import SecretBroker

    _enable_broker()
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "real-key-ABC")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "real-secret-XYZ")
    _reset_boto3_default_session()
    bedrock_adapter.reset_client_cache()

    client = bedrock_adapter._get_bedrock_runtime_client("us-east-1")
    signer = client._request_signer

    # The credentials on the signer are placeholders, not the real values.
    assert SecretBroker.is_placeholder(signer._credentials.access_key)
    assert SecretBroker.is_placeholder(signer._credentials.secret_key)
    assert signer._credentials.access_key != "real-key-ABC"

    request = _sign_bedrock_request(signer)
    auth = request.headers["Authorization"]

    assert "Credential=real-key-ABC/" in auth
    assert SecretBroker.is_placeholder(signer._credentials.access_key)
    assert SecretBroker.is_placeholder(signer._credentials.secret_key)
    assert signer._credentials.access_key != "real-key-ABC"
    assert signer._credentials.secret_key != "real-secret-XYZ"
    assert getattr(client.meta.events, "_hermes_broker_hooked", False) is True


def test_bedrock_runtime_client_preserves_sigv4_session_token(monkeypatch):
    """Static AWS session tokens are placeholders at rest and real while signing."""
    pytest.importorskip("boto3")
    pytest.importorskip("botocore")
    from agent import bedrock_adapter
    from agent.secret_broker import SecretBroker

    _enable_broker()
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "real-key-TOKEN")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "real-secret-TOKEN")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "real-session-token")
    _reset_boto3_default_session()
    bedrock_adapter.reset_client_cache()

    client = bedrock_adapter._get_bedrock_runtime_client("us-east-1")
    signer = client._request_signer

    assert SecretBroker.is_placeholder(signer._credentials.access_key)
    assert SecretBroker.is_placeholder(signer._credentials.secret_key)
    assert SecretBroker.is_placeholder(signer._credentials.token)

    request = _sign_bedrock_request(signer)

    assert "Credential=real-key-TOKEN/" in request.headers["Authorization"]
    assert request.headers["X-Amz-Security-Token"] == "real-session-token"
    assert SecretBroker.is_placeholder(signer._credentials.token)
    assert signer._credentials.token != "real-session-token"


def test_bedrock_runtime_client_sigv4_concurrent_signing_stays_placeholder(monkeypatch):
    """Concurrent signing must not mutate shared signer credentials to real keys."""
    pytest.importorskip("boto3")
    pytest.importorskip("botocore")
    from agent import bedrock_adapter
    from agent.secret_broker import SecretBroker

    _enable_broker()
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "real-key-CONCURRENT")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "real-secret-CONCURRENT")
    _reset_boto3_default_session()
    bedrock_adapter.reset_client_cache()

    client = bedrock_adapter._get_bedrock_runtime_client("us-east-1")
    signer = client._request_signer

    def _sign_once():
        return _sign_bedrock_request(signer).headers["Authorization"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        auth_headers = list(pool.map(lambda _i: _sign_once(), range(2)))

    assert all("Credential=real-key-CONCURRENT/" in auth for auth in auth_headers)
    assert SecretBroker.is_placeholder(signer._credentials.access_key)
    assert SecretBroker.is_placeholder(signer._credentials.secret_key)


def test_bedrock_sigv4_hook_skips_non_placeholder_credentials(monkeypatch):
    """Refreshable/non-placeholder credential sources are left to botocore."""
    pytest.importorskip("boto3")
    pytest.importorskip("botocore")
    import boto3

    from agent.secret_broker import install_bedrock_event_hook

    _enable_broker()
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "plain-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "plain-secret")
    _reset_boto3_default_session()

    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    original_credentials = client._request_signer._credentials

    install_bedrock_event_hook(client, mode="sigv4")

    assert client._request_signer._credentials is original_credentials
    assert client._request_signer._credentials.access_key == "plain-key"


def test_bedrock_broker_disabled_is_passthrough(monkeypatch):
    """Broker off → no hook installed, signer carries the real creds as before."""
    pytest.importorskip("boto3")
    # Broker stays off — no _enable_broker() call.
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "passthrough-real")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    from agent import bedrock_adapter
    _reset_boto3_default_session()
    bedrock_adapter.reset_client_cache()

    client = bedrock_adapter._get_bedrock_runtime_client("us-east-1")

    # Real token sits on the signer; broker hook was not installed.
    assert client._request_signer._auth_token.token == "passthrough-real"
    assert getattr(client.meta.events, "_hermes_broker_hooked", False) is False


def test_bedrock_broker_install_failure_falls_open(monkeypatch):
    """If install_bedrock_event_hook raises, the client is still returned and usable."""
    pytest.importorskip("boto3")
    _enable_broker()
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "fallopen-real")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    _reset_boto3_default_session()

    import agent.secret_broker as broker_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated install failure")

    monkeypatch.setattr(broker_mod, "install_bedrock_event_hook", _boom)
    from agent import bedrock_adapter
    bedrock_adapter.reset_client_cache()

    client = bedrock_adapter._get_bedrock_runtime_client("us-east-1")
    # We must still get back a usable boto3 client object; the broker install
    # failure has been audited (best-effort) and swallowed.
    assert client is not None
    assert hasattr(client, "meta")
    assert hasattr(client, "invoke_model")


# ── Native Anthropic SDK broker wiring ──────────────────────────────────────


def test_anthropic_native_client_broker_swap():
    """Broker on + plain Anthropic API key → SDK client carries placeholder, header carries real."""
    pytest.importorskip("anthropic")
    from agent.anthropic_adapter import build_anthropic_client

    _enable_broker()
    seen = {}

    def handler(request):
        seen["x-api-key"] = request.headers.get("x-api-key")
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-test",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    # Build the real client via the chokepoint — this installs the broker
    # block and constructs an httpx.Client. We then monkey-replace its
    # transport with a MockTransport so we never make a real network call.
    client = build_anthropic_client("sk-ant-api-real-anthropic-key")
    real_http = client._client
    real_http._transport = httpx.MockTransport(handler)

    # The SDK-visible api_key is a placeholder, not the real one.
    assert SecretBroker.is_placeholder(client.api_key)
    assert client.api_key != "sk-ant-api-real-anthropic-key"
    assert get_broker().resolve(client.api_key) == "sk-ant-api-real-anthropic-key"

    # Fire a request; the egress hook resolves the placeholder to the real key.
    client.messages.create(
        model="claude-test",
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert seen["x-api-key"] == "sk-ant-api-real-anthropic-key"


def test_anthropic_native_client_broker_disabled_passthrough():
    """Broker off → SDK client carries the real key, no broker applied."""
    pytest.importorskip("anthropic")
    from agent.anthropic_adapter import build_anthropic_client

    # No _enable_broker() call — broker stays off.
    client = build_anthropic_client("sk-ant-api-passthrough-key")
    assert client.api_key == "sk-ant-api-passthrough-key"
    assert not SecretBroker.is_placeholder(client.api_key)


def test_anthropic_native_client_broker_handles_auth_token_path():
    """OAuth token path (kwargs['auth_token']) → broker swaps auth_token, header carries real bearer."""
    pytest.importorskip("anthropic")
    from agent.anthropic_adapter import build_anthropic_client

    _enable_broker()
    seen = {}

    def handler(request):
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "id": "msg_oauth",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-test",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    # sk-ant-oat-* triggers the OAuth (auth_token) branch in build_anthropic_client.
    client = build_anthropic_client("sk-ant-oat-real-oauth-token")
    client._client._transport = httpx.MockTransport(handler)

    # The SDK-visible auth_token is a placeholder.
    assert SecretBroker.is_placeholder(client.auth_token)
    assert client.auth_token != "sk-ant-oat-real-oauth-token"
    assert get_broker().resolve(client.auth_token) == "sk-ant-oat-real-oauth-token"
    # api_key path is untouched for OAuth tokens.
    assert client.api_key is None

    client.messages.create(
        model="claude-test",
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert seen["authorization"] == "Bearer sk-ant-oat-real-oauth-token"


# ── SIGHUP reload handler ─────────────────────────────────
#
# install_broker_signal_handler() installs a POSIX SIGHUP handler that
# invalidates the broker_enabled cache so operators can flip
# security.credential_broker.enabled in config.yaml and signal the gateway
# without a process restart. Tests pin: invalidation, prior-handler chaining,
# idempotence, and Windows fall-through.


import os as _os
import signal as _signal


@pytest.fixture
def _restore_sighup():
    """Save and restore the SIGHUP handler + the install-once flag."""
    import agent.secret_broker as sb

    prior_handler = (
        _signal.getsignal(_signal.SIGHUP) if hasattr(_signal, "SIGHUP") else None
    )
    prior_installed = sb._signal_handler_installed
    try:
        yield
    finally:
        if hasattr(_signal, "SIGHUP"):
            try:
                _signal.signal(_signal.SIGHUP, prior_handler)
            except (TypeError, ValueError):
                _signal.signal(_signal.SIGHUP, _signal.SIG_DFL)
        sb._signal_handler_installed = prior_installed


@pytest.mark.skipif(
    not hasattr(_signal, "SIGHUP"),
    reason="SIGHUP is POSIX-only; skipped on Windows",
)
def test_install_broker_signal_handler_invalidates_cache(monkeypatch, _restore_sighup):
    """SIGHUP delivery must drop the broker_enabled cache so the next call
    re-reads config.yaml."""
    import agent.secret_broker as sb

    # Force a clean install.
    sb._signal_handler_installed = False
    calls = {"n": 0}

    def _load_config():
        calls["n"] += 1
        return {}

    monkeypatch.setattr("hermes_cli.config.load_config", _load_config)

    assert install_broker_signal_handler() is True
    # Populate the cache.
    broker_enabled()
    assert calls["n"] == 1

    # Fire SIGHUP at ourselves; Python's signal machinery runs the handler at
    # the next bytecode boundary. A bare attribute read after os.kill is
    # enough to ensure delivery on CPython.
    _os.kill(_os.getpid(), _signal.SIGHUP)
    _ = sb._broker_enabled_cache  # bytecode boundary for signal delivery

    # Cache was cleared by the handler; the next broker_enabled() reloads.
    broker_enabled()
    assert calls["n"] == 2


@pytest.mark.skipif(
    not hasattr(_signal, "SIGHUP"),
    reason="SIGHUP is POSIX-only; skipped on Windows",
)
def test_install_broker_signal_handler_chains_prior(_restore_sighup):
    """A pre-existing SIGHUP handler must still fire after the broker handler.

    Operators sometimes wire SIGHUP into systemd / logrotate config-reload
    flows; clobbering theirs would be a footgun.
    """
    import agent.secret_broker as sb

    sb._signal_handler_installed = False
    prior_called = {"n": 0}

    def _prior_handler(signum, frame):
        prior_called["n"] += 1

    _signal.signal(_signal.SIGHUP, _prior_handler)

    assert install_broker_signal_handler() is True
    _os.kill(_os.getpid(), _signal.SIGHUP)
    _ = sb._broker_enabled_cache  # bytecode boundary

    assert prior_called["n"] == 1, (
        "prior SIGHUP handler was not chained — operator-installed "
        "reload hooks would be silently clobbered"
    )


@pytest.mark.skipif(
    not hasattr(_signal, "SIGHUP"),
    reason="SIGHUP is POSIX-only; skipped on Windows",
)
def test_install_broker_signal_handler_idempotent(_restore_sighup):
    """A second call must not re-install — the first call's chained handler
    stays put so we don't recursively chain ourselves on every call.
    """
    import agent.secret_broker as sb

    sb._signal_handler_installed = False
    assert install_broker_signal_handler() is True
    handler_after_first = _signal.getsignal(_signal.SIGHUP)
    assert install_broker_signal_handler() is True
    handler_after_second = _signal.getsignal(_signal.SIGHUP)
    assert handler_after_first is handler_after_second, (
        "second call should not re-register the handler"
    )


@pytest.mark.skipif(
    hasattr(_signal, "SIGHUP"),
    reason="POSIX has SIGHUP — only test the Windows fall-through path here",
)
def test_install_broker_signal_handler_windows_returns_false():
    """On Windows (no SIGHUP attribute), the function must return False
    without raising so gateway startup proceeds normally.
    """
    assert install_broker_signal_handler() is False
