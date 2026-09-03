"""Regression tests for the Actual Computer provider wiring."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from agent.auxiliary_client import _normalize_aux_provider
from hermes_cli import runtime_provider as rp
from hermes_cli.auth import (
    ACTUAL_LOCAL_NOAUTH_PLACEHOLDER,
    DEFAULT_ACTUAL_BASE_URL,
    DEFAULT_ACTUAL_LOCAL_BASE_URL,
    get_api_key_provider_status,
    normalize_actual_base_url,
    resolve_api_key_provider_credentials,
    resolve_provider,
)
from hermes_cli.models import normalize_provider as normalize_model_provider
from hermes_cli.models import provider_model_ids
from hermes_cli.providers import determine_api_mode
from hermes_cli.providers import get_label
from hermes_cli.providers import normalize_provider as normalize_overlay_provider
from providers import get_provider_profile


def _clear_actual_env(monkeypatch):
    monkeypatch.delenv("ACTUAL_API_KEY", raising=False)
    monkeypatch.delenv("ACTUAL_BASE_URL", raising=False)
    monkeypatch.delenv("ACTUAL_API_MODE", raising=False)


def test_actual_aliases_and_profile_metadata():
    profile = get_provider_profile("actual-computer")

    assert profile is not None
    assert profile.name == "actual"
    assert profile.display_name == "Actual Computer"
    assert profile.base_url == DEFAULT_ACTUAL_BASE_URL
    assert profile.api_mode == "chat_completions"
    assert profile.auth_type == "api_key"
    assert profile.env_vars == ("ACTUAL_API_KEY", "ACTUAL_BASE_URL")
    assert normalize_overlay_provider("aci") == "actual"
    assert normalize_model_provider("actualcomputer") == "actual"
    assert resolve_provider("actual-computer") == "actual"
    assert _normalize_aux_provider("aci") == "actual"
    assert get_label("actual") == "Actual Computer"
    assert determine_api_mode("actual", "https://api.actual.inc") == "chat_completions"


def test_actual_base_url_normalization():
    assert (
        normalize_actual_base_url("https://api.actual.inc") == DEFAULT_ACTUAL_BASE_URL
    )
    assert (
        normalize_actual_base_url("https://api.actual.inc/v1")
        == DEFAULT_ACTUAL_BASE_URL
    )
    assert (
        normalize_actual_base_url("http://127.0.0.1:8080")
        == DEFAULT_ACTUAL_LOCAL_BASE_URL
    )
    assert (
        normalize_actual_base_url("http://127.0.0.1:8080/v1")
        == DEFAULT_ACTUAL_LOCAL_BASE_URL
    )
    assert (
        normalize_actual_base_url("http://localhost:8080/")
        == "http://localhost:8080/v1"
    )


def test_actual_credentials_default_to_hosted_api(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")

    creds = resolve_api_key_provider_credentials("actual")

    assert creds["provider"] == "actual"
    assert creds["api_key"] == "actual-test-key"
    assert creds["base_url"] == DEFAULT_ACTUAL_BASE_URL


def test_actual_local_loopback_allows_no_auth(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:8080")

    creds = resolve_api_key_provider_credentials("actual")
    status = get_api_key_provider_status("actual")

    assert creds["api_key"] == ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
    assert creds["base_url"] == DEFAULT_ACTUAL_LOCAL_BASE_URL
    assert creds["source"] == "local-offline"
    assert status["configured"] is True
    assert status["logged_in"] is True
    assert status["key_source"] == "local-offline"
    assert status["base_url"] == DEFAULT_ACTUAL_LOCAL_BASE_URL


def test_actual_runtime_uses_hosted_default(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "actual", "default": "actual/test-model"},
    )

    resolved = rp.resolve_runtime_provider(requested="actual")

    assert resolved["provider"] == "actual"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["api_key"] == "actual-test-key"
    assert resolved["base_url"] == DEFAULT_ACTUAL_BASE_URL


def test_actual_runtime_repairs_stale_responses_mode(monkeypatch, caplog):
    _clear_actual_env(monkeypatch)
    caplog.set_level(logging.INFO, logger="hermes_cli.auth")
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "actual",
            "default": "actual/test-model",
            "api_mode": "codex_responses",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="actual")
    explicit = rp.resolve_runtime_provider(
        requested="actual",
        explicit_api_key="actual-test-key",
        explicit_base_url="https://api.actual.inc/v1",
    )

    assert resolved["api_mode"] == "chat_completions"
    assert explicit["api_mode"] == "chat_completions"
    assert "persisted api_mode=codex_responses" in caplog.text


def test_actual_runtime_responses_escape_hatch(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")
    monkeypatch.setenv("ACTUAL_API_MODE", "codex_responses")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "actual", "default": "actual/test-model"},
    )

    resolved = rp.resolve_runtime_provider(requested="actual")

    assert resolved["api_mode"] == "codex_responses"


def test_actual_hostname_detection_preserves_custom_responses_route():
    base_url = "https://api.actual.inc/v1"

    assert rp._detect_api_mode_for_url(base_url) == "codex_responses"
    assert rp._fallback_api_mode("custom", base_url) == "codex_responses"
    assert rp._resolve_plain_custom_api_mode({}, base_url) == "codex_responses"


def test_actual_runtime_uses_local_env_without_key(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:8080")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "actual", "default": "actual/local-model"},
    )

    resolved = rp.resolve_runtime_provider(requested="actual")

    assert resolved["provider"] == "actual"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["api_key"] == ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
    assert resolved["base_url"] == DEFAULT_ACTUAL_LOCAL_BASE_URL


def test_actual_runtime_uses_local_config_without_key(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "actual",
            "base_url": "http://127.0.0.1:8080",
            "default": "actual/local-model",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="actual")

    assert resolved["provider"] == "actual"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["api_key"] == ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
    assert resolved["base_url"] == DEFAULT_ACTUAL_LOCAL_BASE_URL


def test_actual_runtime_normalizes_explicit_hosted_base_url(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "actual", "default": "actual/test-model"},
    )

    resolved = rp.resolve_runtime_provider(
        requested="actual",
        explicit_api_key="actual-test-key",
        explicit_base_url="https://api.actual.inc",
    )

    assert resolved["provider"] == "actual"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["api_key"] == "actual-test-key"
    assert resolved["base_url"] == DEFAULT_ACTUAL_BASE_URL
    assert resolved["source"] == "explicit"


def test_actual_profile_fetch_models_normalizes_env_base_url(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:8080")
    profile = get_provider_profile("actual")
    seen = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"data": [{"id": "actual/local-model"}]}).encode()

    def _open(req, timeout=0):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", _open)

    assert profile.fetch_models(api_key=None, timeout=1.5) == ["actual/local-model"]
    assert seen["url"] == DEFAULT_ACTUAL_LOCAL_BASE_URL + "/models"
    assert seen["auth"] is None
    assert seen["timeout"] == 1.5


def test_actual_profile_fetch_models_sends_credential_only_to_original_origin(
    monkeypatch,
):
    """fetch_models must route through the shared redirect-credential guard.

    ActualProfile overrides ProviderProfile.fetch_models with its own
    base_url resolution, and previously called raw urllib.request.urlopen
    directly instead of the base class's open_credentialed_url — losing the
    protection that strips the Authorization header when a redirect leaves
    the original host. Exercises the real SafeCredentialRedirectHandler
    (no mocking of open_credentialed_url itself) against a local HTTP
    server that 302s to a different origin, mirroring
    test_urllib_security.py's end-to-end redirect tests.
    """
    import http.server
    import threading

    _clear_actual_env(monkeypatch)
    profile = get_provider_profile("actual")

    source_auth_headers: list[str | None] = []
    target_auth_headers: list[str | None] = []

    class _RedirectTargetHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            target_auth_headers.append(self.headers.get("Authorization"))
            body = json.dumps({"data": [{"id": "should-not-be-trusted"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    target_server = http.server.HTTPServer(("127.0.0.1", 0), _RedirectTargetHandler)
    target_thread = threading.Thread(target=target_server.serve_forever, daemon=True)
    target_thread.start()
    target_port = target_server.server_address[1]

    class _RedirectingHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            source_auth_headers.append(self.headers.get("Authorization"))
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target_port}/models")
            self.end_headers()

        def log_message(self, *_args):
            pass

    redirect_server = http.server.HTTPServer(("127.0.0.1", 0), _RedirectingHandler)
    redirect_thread = threading.Thread(
        target=redirect_server.serve_forever, daemon=True
    )
    redirect_thread.start()
    redirect_port = redirect_server.server_address[1]

    try:
        result = profile.fetch_models(
            api_key="actual-secret-token",
            base_url=f"http://127.0.0.1:{redirect_port}",
            timeout=5.0,
        )
    finally:
        redirect_server.shutdown()
        target_server.shutdown()
        redirect_thread.join(timeout=2.0)
        target_thread.join(timeout=2.0)

    assert result == ["should-not-be-trusted"], (
        "sanity check: the redirect must actually have been followed"
    )
    assert source_auth_headers == ["Bearer actual-secret-token"]
    assert target_auth_headers == [None], (
        "Authorization header leaked to a different origin after a redirect"
    )


def test_actual_provider_model_ids_use_local_profile_catalog(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_BASE_URL", "http://127.0.0.1:8080")
    profile = get_provider_profile("actual")

    with patch.object(
        profile, "fetch_models", return_value=["actual/local-model"]
    ) as fetch:
        assert provider_model_ids("actual") == ["actual/local-model"]

    fetch.assert_called_once_with(
        api_key=ACTUAL_LOCAL_NOAUTH_PLACEHOLDER,
        base_url=DEFAULT_ACTUAL_LOCAL_BASE_URL,
    )


def test_actual_hosted_model_ids_send_resolved_credential(monkeypatch):
    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_API_KEY", "actual-test-key")
    profile = get_provider_profile("actual")

    with patch.object(
        profile, "fetch_models", return_value=["actual/hosted-model"]
    ) as fetch:
        assert provider_model_ids("actual") == ["actual/hosted-model"]

    fetch.assert_called_once_with(
        api_key="actual-test-key",
        base_url=DEFAULT_ACTUAL_BASE_URL,
    )


def test_actual_hosted_model_ids_do_not_probe_without_credentials(monkeypatch):
    _clear_actual_env(monkeypatch)
    profile = get_provider_profile("actual")

    with patch.object(profile, "fetch_models") as fetch:
        assert provider_model_ids("actual") == []

    fetch.assert_not_called()


def test_actual_profile_translates_explicit_reasoning_controls():
    profile = get_provider_profile("actual")

    assert profile.build_api_kwargs_extras(reasoning_config=None) == ({}, {})
    assert profile.supported_reasoning_efforts("zai-org/GLM-5.3") == (
        "none",
        "low",
        "medium",
        "high",
        "max",
    )
    for config, expected_toggle, expected_effort in (
        ({"enabled": True, "effort": "high"}, "enabled", "high"),
        ({"enabled": True, "effort": "xhigh"}, "enabled", "high"),
        ({"enabled": True, "effort": "ultra"}, "enabled", "max"),
        ({"enabled": False, "effort": "high"}, "disabled", None),
        ({"enabled": True, "effort": "none"}, "disabled", "none"),
    ):
        extra_body, top_level = profile.build_api_kwargs_extras(
            reasoning_config=config,
            model="zai-org/GLM-5.3",
        )
        assert extra_body == {"thinking": {"type": expected_toggle}}
        if expected_effort is None:
            assert "reasoning_effort" not in top_level
        else:
            assert top_level["reasoning_effort"] == expected_effort


def test_actual_hosted_client_uses_scoped_macos_certifi(monkeypatch):
    import certifi

    profile = get_provider_profile("actual")
    monkeypatch.setattr(sys, "platform", "darwin")

    assert profile.build_client_kwargs_extras(base_url=DEFAULT_ACTUAL_BASE_URL) == {
        "ssl_ca_cert": certifi.where()
    }
    assert (
        profile.build_client_kwargs_extras(base_url=DEFAULT_ACTUAL_LOCAL_BASE_URL) == {}
    )


def test_actual_client_tls_default_does_not_override_explicit_config(monkeypatch):
    from agent.agent_runtime_helpers import create_openai_client

    captured: list[dict] = []

    def fake_resolve_httpx_verify(**kwargs):
        captured.append(kwargs)
        return "resolved-verify"

    class FakeAgent:
        provider = "actual"

        @staticmethod
        def _build_keepalive_http_client(base_url, *, verify):
            assert base_url == DEFAULT_ACTUAL_BASE_URL
            assert verify == "resolved-verify"
            return "http-client"

        @staticmethod
        def _client_log_context():
            return "test"

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        "agent.ssl_verify.resolve_httpx_verify", fake_resolve_httpx_verify
    )
    monkeypatch.setattr("agent.auxiliary_client._validate_proxy_env_urls", lambda: None)
    monkeypatch.setattr("agent.auxiliary_client._validate_base_url", lambda _url: None)
    monkeypatch.setattr("run_agent.OpenAI", lambda **kwargs: kwargs)

    defaults = create_openai_client(
        FakeAgent(),
        {"api_key": "test", "base_url": DEFAULT_ACTUAL_BASE_URL},
        reason="test",
        shared=False,
    )
    explicit = create_openai_client(
        FakeAgent(),
        {
            "api_key": "test",
            "base_url": DEFAULT_ACTUAL_BASE_URL,
            "ssl_ca_cert": "/explicit/corporate-ca.pem",
        },
        reason="test",
        shared=False,
    )

    assert Path(captured[0]["ca_bundle"]).name == "cacert.pem"
    assert captured[0]["base_url"] == DEFAULT_ACTUAL_BASE_URL
    assert captured[1]["ca_bundle"] == "/explicit/corporate-ca.pem"
    assert defaults["http_client"] == "http-client"
    assert explicit["http_client"] == "http-client"


def test_actual_oneshot_reasoning_override_reaches_agent(monkeypatch):
    from hermes_cli import oneshot

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._session_messages = []

        def run_conversation(self, _prompt):
            return {"final_response": "ok"}

        def shutdown_memory_provider(self, *_args):
            return None

        def close(self):
            return None

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "actual-test-key",
            "base_url": DEFAULT_ACTUAL_BASE_URL,
            "provider": "actual",
            "requested_provider": "actual",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(oneshot, "get_fallback_chain", lambda _cfg: [])
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

    response, _result = oneshot._run_agent(
        "hello",
        model="zai-org/GLM-5.3",
        provider="actual",
        reasoning="ultra",
        toolsets=["terminal"],
        use_config_toolsets=False,
    )

    assert response == "ok"
    assert captured["reasoning_config"] == {"enabled": True, "effort": "ultra"}


def test_oneshot_dispatch_forwards_reasoning_override(monkeypatch):
    from hermes_cli import main as main_mod
    from hermes_cli import oneshot

    captured = {}

    def fake_run_oneshot(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return 0

    class OneshotExit(Exception):
        pass

    def fake_exit(_rc):
        raise OneshotExit

    monkeypatch.setattr(oneshot, "run_oneshot", fake_run_oneshot)
    monkeypatch.setattr(main_mod, "_cleanup_oneshot_runtime", lambda: None)
    monkeypatch.setattr(main_mod, "_exit_after_oneshot", fake_exit)

    with pytest.raises(OneshotExit):
        main_mod._run_and_exit_oneshot(
            "hello",
            model="zai-org/GLM-5.3",
            provider="actual",
            reasoning="high",
        )

    assert captured["reasoning"] == "high"


def test_actual_agent_side_routing_keeps_chat_completions_for_any_model():
    from run_agent import AIAgent

    for model in ("qwen3.8-27b-Q4_K_M", "zai-org/GLM-5.3", "gpt-5.4"):
        assert not AIAgent._provider_model_requires_responses_api(
            model,
            provider=" Actual ",
        )


def test_actual_agent_init_repairs_stale_responses_mode():
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="actual-test-key",
            base_url=DEFAULT_ACTUAL_BASE_URL,
            provider="actual",
            api_mode="codex_responses",
            model="gpt-5.4",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.api_mode == "chat_completions"


def test_actual_agent_init_honors_responses_escape_hatch(monkeypatch):
    from run_agent import AIAgent

    _clear_actual_env(monkeypatch)
    monkeypatch.setenv("ACTUAL_API_MODE", "codex_responses")
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="actual-test-key",
            base_url=DEFAULT_ACTUAL_BASE_URL,
            provider="actual",
            api_mode="chat_completions",
            model="zai-org/GLM-5.3",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.api_mode == "codex_responses"


def test_actual_chat_completions_wire_replays_reasoning_through_tool_turn(
    monkeypatch,
):
    import http.server
    import threading

    from openai import OpenAI

    from agent.transports.chat_completions import ChatCompletionsTransport

    requests: list[tuple[str, dict]] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            requests.append((self.path, request))

            if len(requests) == 1:
                message = {
                    "role": "assistant",
                    "content": None,
                    "reasoning": "I should call the probe tool.",
                    "reasoning_content": "I should call the probe tool.",
                    "tool_calls": [
                        {
                            "id": "call_actual_probe",
                            "type": "function",
                            "function": {
                                "name": "actual_probe",
                                "arguments": '{"value":"ok"}',
                            },
                        }
                    ],
                }
                finish_reason = "tool_calls"
            else:
                message = {
                    "role": "assistant",
                    "content": "ACTUAL_CHAT_OK",
                    "reasoning": "The tool result confirms the answer.",
                    "reasoning_content": "The tool result confirms the answer.",
                }
                finish_reason = "stop"

            body = json.dumps({
                "id": f"chatcmpl-actual-{len(requests)}",
                "object": "chat.completion",
                "created": 1,
                "model": request["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        _clear_actual_env(monkeypatch)
        base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        monkeypatch.setenv("ACTUAL_BASE_URL", base_url)
        monkeypatch.setattr(
            rp,
            "_get_model_config",
            lambda: {"provider": "actual", "default": "actual/test-model"},
        )

        resolved = rp.resolve_runtime_provider(requested="actual")
        profile = get_provider_profile(resolved["provider"])
        transport = ChatCompletionsTransport()
        client = OpenAI(
            api_key=resolved["api_key"],
            base_url=resolved["base_url"],
            max_retries=0,
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "actual_probe",
                    "description": "Return the supplied value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ]
        messages = [{"role": "user", "content": "Use actual_probe."}]
        params = {
            "provider_profile": profile,
            "reasoning_config": {"enabled": True, "effort": "high"},
            "base_url": resolved["base_url"],
        }

        first_raw = client.chat.completions.create(
            **transport.build_kwargs(
                model="test-model",
                messages=messages,
                tools=tools,
                **params,
            )
        )
        first = transport.normalize_response(first_raw)
        assert first.reasoning == "I should call the probe tool."
        assert first.reasoning_content == "I should call the probe tool."
        assert first.tool_calls and first.tool_calls[0].name == "actual_probe"

        tool_call = first.tool_calls[0]
        messages.extend([
            {
                "role": "assistant",
                "content": first.content,
                "reasoning": first.reasoning,
                "reasoning_content": first.reasoning_content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": '{"value":"ok"}',
            },
        ])
        second_raw = client.chat.completions.create(
            **transport.build_kwargs(
                model="test-model",
                messages=messages,
                tools=tools,
                **params,
            )
        )
        second = transport.normalize_response(second_raw)
    finally:
        server.shutdown()
        thread.join(timeout=2.0)

    assert resolved["api_mode"] == "chat_completions"
    assert [path for path, _ in requests] == [
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]
    assert requests[0][1]["thinking"] == {"type": "enabled"}
    assert requests[0][1]["reasoning_effort"] == "high"
    assert requests[1][1]["reasoning_effort"] == "high"
    replayed = requests[1][1]["messages"][-2]
    assert replayed["reasoning"] == "I should call the probe tool."
    assert replayed["reasoning_content"] == "I should call the probe tool."
    assert second.content == "ACTUAL_CHAT_OK"
    assert second.reasoning == "The tool result confirms the answer."
    assert second.reasoning_content == "The tool result confirms the answer."


def test_actual_chat_completion_without_reasoning_keeps_final_content():
    from types import SimpleNamespace

    from agent.transports.chat_completions import ChatCompletionsTransport

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content="ACTUAL_NO_REASONING_OK",
                    reasoning=None,
                    reasoning_content=None,
                    tool_calls=None,
                ),
            )
        ],
        usage=None,
    )

    normalized = ChatCompletionsTransport().normalize_response(response)

    assert normalized.content == "ACTUAL_NO_REASONING_OK"
    assert normalized.reasoning is None
    assert normalized.reasoning_content is None


def test_actual_runtime_config_local_base_url_without_key(monkeypatch):
    """Config-driven loopback base_url (not just env) reaches the no-auth path."""
    _clear_actual_env(monkeypatch)
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "actual",
            "base_url": "http://localhost:8080",
            "default": "actual/local-model",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="actual")

    assert resolved["api_key"] == ACTUAL_LOCAL_NOAUTH_PLACEHOLDER
    assert resolved["api_mode"] == "chat_completions"
