"""Published auxiliary egress, endpoint and scoped recovery invariants."""
from agent import auxiliary_egress_recovery
import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from agent.llm_egress_firewall import EgressBlocked
from agent.llm_egress_runtime import authorize_agent_sdk_kwargs
from agent.auxiliary_egress_recovery import auxiliary_egress_binding, authorize_auxiliary_request
from agent.auxiliary_client import call_llm, _RELAY_AUX_CALL_CONTEXT


def _aux_egress_response(content="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model="gpt-5.4",
    )


@pytest.mark.parametrize("remote_provider,has_local", [
    ("nous", True), ("nous", False),
    ("screening-remote", True), ("screening-remote", False),
    ("opencode-free", False),
    ("profile-custom-env", True), ("profile-custom-config", True),
    ("profile-custom-env-live", True), ("profile-custom-config-live", True),
])
def test_blocked_recovery_screens_remote_auth_before_resolving_local(
    monkeypatch, tmp_path, remote_provider, has_local
):
    import hermes_yaml as yaml
    import agent.auxiliary_client as auxiliary
    from agent.auxiliary_egress_recovery import local_fallback_steps

    home = tmp_path / "screening-home"
    home.mkdir()
    task = "compression" if remote_provider == "opencode-free" else "title_generation"
    chain = [{"provider": remote_provider, "model": "remote-model",
              "base_url": "http://127.0.0.1:11434/v1"}]
    if remote_provider == "screening-remote":
        chain[0].pop("base_url")
    if remote_provider == "opencode-free":
        chain[0].update(model="minimax-m2.5-free", api_mode="anthropic_messages", api_key="synthetic-key")
    profile_case = remote_provider.startswith("profile-custom-")
    if profile_case:
        chain[0] = {"provider": "custom", "model": "local-model"}
    elif has_local:
        chain.append({"provider": "custom", "model": "local-model",
                      "base_url": "http://127.0.0.1:11434/v1", "api_key": "local-test-key"})
    config = {
        "model": {"provider": "nous", "default": "main-remote-model"},
        "auxiliary": {task: {"fallback_chain": chain}},
        "providers": {"screening-remote": {"base_url": "https://remote.invalid/v1",
                                                "key_env": "SCREENING_REMOTE_API_KEY"}},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    auth_reads = []
    # Observe the credential I/O boundary; use the real config/router/SDK.
    def read_remote_auth():
        auth_reads.append("nous")
        return None
    monkeypatch.setattr(auxiliary, "_read_nous_auth", read_remote_auth)
    import hermes_cli.runtime_provider as runtime_provider
    import agent.model_metadata as model_metadata
    key_reads = []
    import hermes_cli.runtime_provider_custom as custom_owner
    original_getenv = custom_owner.get_secret_str
    def observe_key(name, default=""):
        if name == "SCREENING_REMOTE_API_KEY":
            key_reads.append(name)
            return "synthetic-key"
        return original_getenv(name, default)
    monkeypatch.setattr(custom_owner, "get_secret_str", observe_key)
    context_reads = []
    original_context = auxiliary.get_model_context_length
    def observe_context(*args, **kwargs):
        context_reads.append(kwargs.get("base_url"))
        return original_context(*args, **kwargs)
    monkeypatch.setattr(auxiliary, "get_model_context_length", observe_context)
    remote_catalog = MagicMock(side_effect=RuntimeError("unexpected catalog I/O"))
    monkeypatch.setattr("agent.models_dev.requests.get", remote_catalog)
    monkeypatch.setattr("httpx.Client.get", remote_catalog)
    route = SimpleNamespace(task=task, resolved_provider="openai-codex",
                            final_model="blocked-model", route_info={},
                            base_info="https://chatgpt.com/backend-api/codex",
                            main_runtime=None, async_mode=False)
    if profile_case:
        from pathlib import Path
        from agent import secret_scope
        from gateway.run import _profile_runtime_scope
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        other = tmp_path / "other-profile"
        other.mkdir()
        bases = {home: "http://127.0.0.1:11434/v1", other: "https://remote-profile.invalid/v1"}
        for profile, base in bases.items():
            profile_config = config
            custom_env = f"CUSTOM_BASE_URL={base}\n"
            if "config" in remote_provider:
                profile_config = {**config, "model": {"provider": "custom", "default": "configured-model",
                                                       "base_url": base, "api_key": "profile-test-key"}}
                custom_env = ""
            (profile / "config.yaml").write_text(yaml.safe_dump(profile_config))
            (profile / ".env").write_text(f"{custom_env}OPENAI_BASE_URL={base}\nOPENAI_API_KEY=profile-test-key\n")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://launch-profile.invalid/v1")
        main_token = None
        if remote_provider.endswith("live"):
            main_token = auxiliary.set_runtime_main("custom", "main-remote-model",
                                                   base_url="https://remote-main.invalid/v1", api_key="main-remote-key")
        multiplex = secret_scope.is_multiplex_active()
        secret_scope.set_multiplex_active(True)
        try:
            for profile in (home, other, home):
                with _profile_runtime_scope(profile, hydrate_secrets=False):
                    ordinary, model = auxiliary.resolve_provider_client("custom", "local-model")
                    assert ordinary is not None
                    try:
                        assert str(ordinary.base_url).rstrip("/") == bases[profile]
                        assert model == "local-model"
                    finally:
                        ordinary.close()
                    ordinary_key_reads = list(key_reads)
                    ladder = local_fallback_steps(route, lambda kind, args: SimpleNamespace(kind=kind, args=args))
                    if profile == other:
                        assert list(ladder) == []
                        assert key_reads == ordinary_key_reads
                        continue
                    client, model, _ = next(ladder).args
                    try:
                        assert str(client.base_url).rstrip("/") == bases[profile]
                        assert client.api_key == "profile-test-key"
                        assert model == "local-model"
                        with pytest.raises(StopIteration) as result:
                            ladder.send("local response")
                        assert result.value.value == "local response"
                    finally:
                        client.close()
                    assert key_reads == ordinary_key_reads
        finally:
            secret_scope.set_multiplex_active(multiplex)
            if main_token is not None:
                auxiliary.reset_runtime_main(main_token)
    else:
        ladder = local_fallback_steps(route, lambda kind, args: SimpleNamespace(kind=kind, args=args))
        if has_local:
            step = next(ladder)
            client, model, _ = step.args
            assert str(client.base_url) == "http://127.0.0.1:11434/v1/"
            assert model == "local-model"
            with pytest.raises(StopIteration) as result:
                ladder.send("local response")
            assert result.value.value == "local response"
            client.close()
        else:
            assert list(ladder) == []
    assert auth_reads == []
    if not profile_case:
        assert key_reads == []
    remote_catalog.assert_not_called()
    assert context_reads == []


@pytest.mark.parametrize("base_url", ["http://127.0.0.1:18434/v1", "https://remote-main.invalid/v1", None, "", "  "])
def test_blocked_custom_main_preserves_live_endpoint_and_key(monkeypatch, tmp_path, base_url):
    import hermes_yaml as yaml
    import agent.auxiliary_client as auxiliary
    import agent.model_metadata as model_metadata

    home = tmp_path / "main-pin-home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"model": {
        "provider": "custom", "default": "configured-model", "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "environment-key",
    }}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://127.0.0.1:12434/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:12434/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    final_http = MagicMock(side_effect=RuntimeError("unexpected metadata I/O"))
    monkeypatch.setattr("agent.models_dev.requests.get", final_http)
    monkeypatch.setattr("httpx.Client.get", final_http)
    runtime = {"provider": "custom", "model": "live-model", "api_key": "live-main-key"}
    if base_url is not None:
        runtime["base_url"] = base_url
    expected_base = base_url.strip() if base_url and base_url.strip() else "http://127.0.0.1:12434/v1"
    expected_key = runtime["api_key"] if base_url and base_url.strip() else "environment-key"
    with auxiliary.scoped_runtime_main(runtime):
        ordinary, model = auxiliary.resolve_provider_client("custom", "live-model", main_runtime=runtime)
        assert ordinary is not None
        try:
            assert str(ordinary.base_url).rstrip("/") == expected_base
            assert ordinary.api_key == expected_key
            assert model == runtime["model"]
        finally:
            ordinary.close()
        client, model, _ = auxiliary_egress_recovery.try_main_agent_model_fallback("openai-codex", "title_generation", local_only=True)
        if expected_base.startswith("https:"):
            assert client is None
        else:
            assert client is not None
            try:
                assert str(client.base_url).rstrip("/") == expected_base
                assert client.api_key == expected_key
                assert model == runtime["model"]
            finally:
                client.close()
    final_http.assert_not_called()


def _run_aux_codex_call(
    monkeypatch,
    tmp_path,
    *,
    content="safe request",
    provider="openai-codex",
    base_url="https://chatgpt.com/backend-api/codex",
    api_mode="codex_responses",
    client_out=None,
):
    client = MagicMock()
    client.base_url = base_url
    client.chat.completions.create.return_value = _aux_egress_response()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "agent.auxiliary_client._resolve_task_provider_model",
        lambda *args, **kwargs: (provider, "gpt-5.4", None, None, api_mode),
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._get_cached_client",
        lambda *args, **kwargs: (client, "gpt-5.4"),
    )
    if client_out is not None:
        client_out.append(client)
    return client, call_llm(
        task="compression",
        provider=provider,
        model="gpt-5.4",
        main_runtime={"session_id": "session-aux"},
        messages=[{"role": "user", "content": content}],
    )


def test_codex_auxiliary_egress_authorizes_once_with_complete_identity(
    monkeypatch, tmp_path
):
    client, response = _run_aux_codex_call(monkeypatch, tmp_path)

    assert response.choices[0].message.content == "ok"
    client.chat.completions.create.assert_called_once()
    receipt_path = tmp_path / "egress" / "llm-egress-receipts.jsonl"
    receipt = json.loads(receipt_path.read_text().splitlines()[0])
    assert receipt["decision"] == "allow"
    assert receipt["provider"] == "openai-codex"
    assert receipt["session_id"] == "session-aux"
    assert receipt["turn_id"]
    assert receipt["request_id"]
    assert receipt["policy_digest"]


def test_codex_auxiliary_egress_blocks_secret_before_provider_callback(
    monkeypatch, tmp_path
):
    clients = []
    with pytest.raises(EgressBlocked):
        _run_aux_codex_call(
            monkeypatch,
            tmp_path,
            content="token=super-secret-value",
            client_out=clients,
        )

    assert clients[0].chat.completions.create.call_count == 0


def test_nous_auxiliary_egress_is_provider_bound(monkeypatch, tmp_path):
    client, response = _run_aux_codex_call(
        monkeypatch,
        tmp_path,
        provider="nous",
        base_url="https://inference-api.nousresearch.com/v1",
        api_mode="chat_completions",
    )

    assert response.choices[0].message.content == "ok"
    client.chat.completions.create.assert_called_once()
    receipt = json.loads(
        (tmp_path / "egress" / "llm-egress-receipts.jsonl").read_text().splitlines()[0]
    )
    assert receipt["provider"] == "nous"
    assert receipt["destination_class"] == "remote"


def test_nous_auxiliary_egress_blocks_secret_before_provider_callback(monkeypatch, tmp_path):
    clients = []
    with pytest.raises(EgressBlocked):
        _run_aux_codex_call(
            monkeypatch,
            tmp_path,
            content="token=super-secret-value",
            provider="nous",
            base_url="https://inference-api.nousresearch.com/v1",
            api_mode="chat_completions",
            client_out=clients,
        )
    assert clients[0].chat.completions.create.call_count == 0


def test_nous_anthropic_messages_auxiliary_egress_blocks_secret(monkeypatch, tmp_path):
    clients = []
    with pytest.raises(EgressBlocked):
        _run_aux_codex_call(
            monkeypatch,
            tmp_path,
            content="token=super-secret-value",
            provider="nous",
            base_url="https://inference-api.nousresearch.com/anthropic",
            api_mode="anthropic_messages",
            client_out=clients,
        )
    assert clients[0].chat.completions.create.call_count == 0


def test_direct_anthropic_auxiliary_egress_blocks_secret(monkeypatch, tmp_path):
    clients = []
    with pytest.raises(EgressBlocked):
        _run_aux_codex_call(
            monkeypatch,
            tmp_path,
            content="token=super-secret-value",
            provider="anthropic",
            base_url="https://api.anthropic.com/v1",
            api_mode="anthropic_messages",
            client_out=clients,
        )
    assert clients[0].chat.completions.create.call_count == 0


def test_local_auxiliary_route_bypasses_remote_egress(monkeypatch, tmp_path):
    dispatch = MagicMock(side_effect=AssertionError("local route must not use remote egress"))
    monkeypatch.setattr("agent.llm_egress_runtime.dispatch_authorized_agent_request", dispatch)
    client, response = _run_aux_codex_call(
        monkeypatch,
        tmp_path,
        provider="custom",
        base_url="http://127.0.0.1:11434/v1",
        api_mode="chat_completions",
    )

    assert response.choices[0].message.content == "ok"
    client.chat.completions.create.assert_called_once()
    dispatch.assert_not_called()


def test_only_compression_auxiliary_binding_gets_larger_exact_grant_caps():
    client = SimpleNamespace(base_url="https://chatgpt.com/backend-api/codex")
    compression_token = _RELAY_AUX_CALL_CONTEXT.set({"task": "compression"})
    try:
        compression_agent, _ = auxiliary_egress_binding(
            client,
            provider="openai-codex",
            model="gpt-5.4",
            api_mode="codex_responses",
        )
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(compression_token)

    vision_token = _RELAY_AUX_CALL_CONTEXT.set({"task": "vision"})
    try:
        vision_agent, _ = auxiliary_egress_binding(
            client,
            provider="openai-codex",
            model="gpt-5.4",
            api_mode="codex_responses",
        )
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(vision_token)

    assert compression_agent._llm_egress_max_granted_serialized_bytes == 2_000_000
    assert compression_agent._llm_egress_max_granted_conservative_tokens == 666_667
    assert compression_agent._llm_egress_max_serialized_bytes == 2_000_000
    assert compression_agent._llm_egress_max_conservative_tokens == 666_667
    assert compression_agent._llm_egress_max_sanitized_bytes == 2_000_000
    assert compression_agent._llm_egress_max_sanitized_segment_bytes == 32_768
    assert not hasattr(vision_agent, "_llm_egress_max_granted_serialized_bytes")
    assert not hasattr(vision_agent, "_llm_egress_max_granted_conservative_tokens")
    assert not hasattr(vision_agent, "_llm_egress_max_sanitized_bytes")
    assert not hasattr(vision_agent, "_llm_egress_max_sanitized_segment_bytes")


def _bound_aux_agent(task: str, tmp_path):
    client = SimpleNamespace(base_url="https://chatgpt.com/backend-api/codex")
    token = _RELAY_AUX_CALL_CONTEXT.set({"task": task})
    try:
        agent, route = auxiliary_egress_binding(
            client,
            provider="openai-codex",
            model="gpt-5.4",
            api_mode="codex_responses",
        )
    finally:
        _RELAY_AUX_CALL_CONTEXT.reset(token)
    agent._llm_egress_state_dir = tmp_path / task
    return agent, route


def _many_bounded_sanitized_messages():
    return [
        {"role": "user", "content": f"segment {index}. " + "ordinary sentence. " * 240}
        for index in range(12)
    ]


def test_compression_allows_many_bounded_sanitized_segments_but_vision_denies(
    tmp_path,
):
    request = {"model": "gpt-5.4", "messages": _many_bounded_sanitized_messages()}
    compression_agent, compression_route = _bound_aux_agent("compression", tmp_path)
    vision_agent, vision_route = _bound_aux_agent("vision", tmp_path)

    authorized, decision = authorize_agent_sdk_kwargs(
        compression_agent,
        request,
        route=compression_route,
    )
    assert authorized == request
    assert decision.decision.serialized_bytes > 32_768

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(vision_agent, request, route=vision_route)
    assert "sanitized_bytes_exceeded" in exc_info.value.decision.reason_codes


def test_compression_still_denies_one_sanitized_segment_over_32768_bytes(tmp_path):
    compression_agent, compression_route = _bound_aux_agent("compression", tmp_path)
    oversized = "ordinary sentence. " * 2_000

    with pytest.raises(ValueError, match="sanitized segment exceeds byte cap"):
        authorize_agent_sdk_kwargs(
            compression_agent,
            {"model": "gpt-5.4", "messages": [{"role": "user", "content": oversized}]},
            route=compression_route,
        )


@pytest.mark.parametrize(
    ("unsafe", "reason"),
    [
        ("token=super-secret-value", "secret_detected"),
        ("cHJpdmF0ZSBzb3VyY2UgdGhhdCBtdXN0IG5vdCBsZWF2ZQ==", "base64_payload"),
        ("Read /Users/private/repository/file.py", "private_absolute_path"),
    ],
)
def test_compression_aggregate_capacity_does_not_bypass_scans(tmp_path, unsafe, reason):
    compression_agent, compression_route = _bound_aux_agent("compression", tmp_path)
    messages = _many_bounded_sanitized_messages()
    messages.append({"role": "user", "content": unsafe})

    with pytest.raises(EgressBlocked) as exc_info:
        authorize_agent_sdk_kwargs(
            compression_agent,
            {"model": "gpt-5.4", "messages": messages},
            route=compression_route,
        )
    assert reason in exc_info.value.decision.reason_codes


def test_blocked_remote_aux_call_is_fallback_eligible_without_retry(monkeypatch, tmp_path):
    primary = MagicMock()
    primary.base_url = "https://chatgpt.com/backend-api/codex"
    primary.chat.completions.create.return_value = _aux_egress_response()
    fallback = MagicMock()
    fallback.base_url = "http://127.0.0.1:11434/v1"
    fallback.chat.completions.create.return_value = _aux_egress_response("fallback")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "agent.auxiliary_client._resolve_task_provider_model",
        lambda *args, **kwargs: ("openai-codex", "gpt-5.4", None, None, "codex_responses"),
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._get_cached_client",
        lambda provider, *args, **kwargs: (primary, "gpt-5.4"),
    )
    monkeypatch.setattr(
        "agent.auxiliary_egress_recovery.try_configured_fallback_chain",
        lambda *args, **kwargs: (None, None, ""),
    )
    monkeypatch.setattr(
        "agent.auxiliary_egress_recovery.try_main_agent_model_fallback",
        lambda *args, **kwargs: (fallback, "local-model", "custom"),
    )
    monkeypatch.setattr(
        "agent.auxiliary_client._transient_retry_count",
        lambda: (_ for _ in ()).throw(AssertionError("blocked request must not retry")),
    )

    response = call_llm(
        task="compression",
        provider="openai-codex",
        model="gpt-5.4",
        main_runtime={"session_id": "session-fallback"},
        messages=[{"role": "user", "content": "token=super-secret-value"}],
    )

    assert response.choices[0].message.content == "fallback"
    primary.chat.completions.create.assert_not_called()
    fallback.chat.completions.create.assert_called_once()


def _jwt_with_claims(claims: dict) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{header}.{payload}.sig"


class _FakeAnthropicStream:
    def __init__(self, final_message):
        self._final_message = final_message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_final_message(self):
        return self._final_message


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip provider env vars so each test starts clean."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "OPENAI_MODEL", "LLM_MODEL", "NOUS_INFERENCE_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
        "NVIDIA_API_KEY", "NVIDIA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    # Module-level unhealthy cache (10-min TTL) leaks between tests;
    # earlier tests that call _mark_provider_unhealthy() poison the
    # cache for later ones, causing _resolve_auto_route to skip providers
    # that the test patched to return valid clients.
    import agent.auxiliary_client as _aux_mod
    _aux_mod._aux_unhealthy_until.clear()
    _aux_mod._aux_unhealthy_logged_at.clear()
    yield
    _aux_mod._aux_unhealthy_until.clear()
    _aux_mod._aux_unhealthy_logged_at.clear()


@pytest.mark.parametrize("provider", ["nous", "nous-portal", "nousresearch"])
def test_all_nous_aliases_require_authorization_before_callback(monkeypatch, tmp_path, provider):
    import agent.auxiliary_client as module
    client = SimpleNamespace(base_url="https://inference-api.nousresearch.com/v1")
    monkeypatch.setattr(module, "get_hermes_home", lambda: tmp_path)
    requests = []
    def refuse(agent, kwargs, callback, *, route):
        assert route.provider == provider
        raise RuntimeError("synthetic authorization refusal")
    monkeypatch.setattr("agent.llm_egress_runtime.dispatch_authorized_agent_request", refuse)
    with pytest.raises(RuntimeError, match="synthetic authorization refusal"):
        authorize_auxiliary_request(
            client, {"model": "synthetic-model", "messages": []},
            lambda request: requests.append(request), provider=provider,
            api_mode="chat_completions", metadata=None,
        )
    assert requests == []


@pytest.mark.parametrize("provider", ["nous", "nous-portal", "nousresearch"])
@pytest.mark.parametrize("local_auxiliary", [False, True])
def test_auxiliary_sdk_url_binds_actual_endpoint(
    monkeypatch, tmp_path, provider, local_auxiliary
):
    import httpx
    from openai import OpenAI
    from agent import auxiliary_client

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    endpoint = (
        "http://127.0.0.1:11434/v1" if local_auxiliary
        else "https://inference-api.nousresearch.com/v1"
    )
    request = {
        "model": "synthetic-model",
        "messages": [{"role": "user", "content": "token=synthetic-secret"}],
    }
    with OpenAI(api_key="synthetic-key", base_url=endpoint) as client:
        assert isinstance(client.base_url, httpx.URL)
        callback = MagicMock(return_value="local")
        monkeypatch.setattr(client.chat.completions, "create", callback)
        with auxiliary_client.scoped_runtime_main(
            {"provider": provider, "base_url": "http://127.0.0.1:11434/v1"}
        ):
            if local_auxiliary:
                assert auxiliary_client._relay_sync_completion(
                    client, request, provider=provider
                ) == "local"
                callback.assert_called_once()
                sent = callback.call_args.kwargs
                assert sent["model"] == request["model"]
                assert sent.get("extra_body", {}).get("messages", sent["messages"]) == request["messages"]
            else:
                with pytest.raises(EgressBlocked):
                    auxiliary_client._relay_sync_completion(
                        client, request, provider=provider
                    )
                callback.assert_not_called()




@pytest.mark.parametrize("case,top,metadata", [
    ("named-stale", False, True), ("named-stale", True, False),
    ("named-match", False, True), ("named-match", True, False),
    ("auto-match", False, True), ("auto-match", True, False),
    ("llamacpp", False, True), ("llama.cpp", True, False),
    ("llama-cpp", False, True), ("named-override", True, True),
])
def test_current_router_identity_endpoint_and_local_vision_parity(
    monkeypatch, tmp_path, case, top, metadata
):
    import io
    import urllib.error
    import urllib.request
    import hermes_yaml as yaml
    from agent import auxiliary_client as auxiliary
    from agent.auxiliary_egress_recovery import local_fallback_entry, local_main_supports_vision
    from hermes_cli.runtime_provider import resolve_runtime_provider

    home = tmp_path / "live-owner"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    local = "http://127.0.0.1:18434/v1"
    alias = case if case.startswith("llama") else "custom:localbox"
    named_base = "https://remote.invalid/v1" if case == "named-override" else local
    cfg = {
        "model": {"provider": alias if case == "named-match" else "auto" if case == "auto-match" else "openrouter",
                  "default": "live" if case in {"named-match", "auto-match"} else "stale", "supports_vision": top},
        "providers": {"localbox": {"base_url": named_base, "api_key": "named-key",
                                     "models": {"live": {"supports_vision": metadata}}},
                      "custom": {"base_url": "https://unrelated.invalid/v1", "api_key": "unrelated-key"},
                      **({alias: {"models": {"live": {"supports_vision": metadata}}}} if case.startswith("llama") else {})},
        "local_runtime": {"enabled": False, "detect_ports": [18434]},
    }
    if case == "auto-match":
        cfg["model"].update(base_url=local, api_key="auto-key")
    (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    probes = []
    def final_local_http(url, *args, **kwargs):
        target = str(getattr(url, "full_url", url))
        probes.append(target)
        if target.startswith("http://127.0.0.1:18434/"):
            response = io.BytesIO(json.dumps({"build_info": "test-llama", "data": []}).encode())
            response.status = 200
            return response
        raise urllib.error.URLError("no server")
    monkeypatch.setattr(urllib.request, "urlopen", final_local_http)
    entry = {"provider": alias, "model": "live"}
    if case in {"named-override", "auto-match"}:
        entry["base_url"] = local
    screened = local_fallback_entry(entry)
    assert screened is not None and screened["base_url"] == local
    ordinary = resolve_runtime_provider(requested="auto" if case == "auto-match" else alias,
                                        target_model="live", explicit_base_url=entry.get("base_url"))
    assert ordinary["base_url"].rstrip("/") == screened["base_url"]
    runtime = {**ordinary, "provider": "custom", "model": "live",
               "requested_provider": ordinary.get("requested_provider") or alias}
    with auxiliary.scoped_runtime_main(runtime):
        expected = top if case in {"named-match", "auto-match"} else metadata
        assert local_main_supports_vision("custom", "live", screened) is expected
    assert all(target.startswith("http://127.0.0.1:") for target in probes)
