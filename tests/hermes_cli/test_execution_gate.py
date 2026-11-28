import io
import json
from argparse import Namespace

import pytest

from hermes_cli.execution_gate import (
    choose_execution_mode,
    direct_completion,
    resolve_direct_route,
    run_pre_agent_oneshot,
)

CONFIG = {
    "model": {"provider": "nous"},
    "providers": {
        "zerogpu": {
            "base_url": "https://api.zerogpu.ai/v1",
            "api_mode": "chat_completions",
            "key_env": "ZGPU_API_KEY",
            "default_model": "meta-llama/llama-3.1-8b-instruct",
        }
    },
}


def fake_opener(captured):
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fetch(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return Response(
            json.dumps(
                {
                    "choices": [{"message": {"content": "OK"}}],
                    "usage": {"prompt_tokens": 38, "completion_tokens": 2},
                }
            ).encode()
        )

    return fetch


def test_routing_is_conservative():
    assert (
        choose_execution_mode("Reply exactly OK", "auto", direct_available=True).mode
        == "direct"
    )
    assert (
        choose_execution_mode("Reply exactly OK", "auto", direct_available=False).mode
        == "agent"
    )
    assert (
        choose_execution_mode(
            "Find and fix bugs then commit", "auto", direct_available=True
        ).mode
        == "agent"
    )
    assert (
        choose_execution_mode(
            "Reply to the customer email", "auto", direct_available=True
        ).mode
        == "agent"
    )
    assert (
        choose_execution_mode("Reply exactly OK", "agent", direct_available=True).mode
        == "agent"
    )
    assert choose_execution_mode("test", "bounded").mode == "bounded"
    with pytest.raises(ValueError, match="EXECUTION_MODE_INVALID"):
        choose_execution_mode("test", "unsafe")


def test_route_requires_explicit_custom_provider_and_safe_url():
    assert resolve_direct_route(CONFIG) is None
    route = resolve_direct_route(CONFIG, provider="custom:zerogpu")
    assert route["model"] == "meta-llama/llama-3.1-8b-instruct"
    assert route["key_env"] == "ZGPU_API_KEY"
    assert (
        resolve_direct_route(
            {
                "providers": {
                    "evil": {
                        "base_url": "http://localhost/v1",
                        "key_env": "TEST_KEY",
                        "api_mode": "chat_completions",
                        "model": "foo",
                    }
                }
            },
            provider="custom:evil",
        )
        is None
    )


def test_direct_is_one_call_no_tools_or_false_billing():
    seen = {}
    route = resolve_direct_route(CONFIG, provider="custom:zerogpu")
    result = direct_completion(
        "Reply exactly OK",
        route,
        "secret-value",
        opener=fake_opener(seen),
        clock=lambda: 12.0,
    )
    assert result["api_calls"] == 1
    assert result["input_tokens"] == 38
    assert result["output_tokens"] == 2
    assert result["cost_status"] == "unknown"
    assert result["estimated_cost_usd"] is None
    assert seen["body"]["messages"] == [{"role": "user", "content": "Reply exactly OK"}]
    assert "tools" not in seen["body"]
    assert seen["timeout"] == 20
    assert "secret-value" not in json.dumps(result)


def test_gate_routes_early_and_writes_usage(tmp_path, capsys):
    assert (
        run_pre_agent_oneshot(
            Namespace(execution_mode="bounded", oneshot="x", toolsets=None)
        )
        == 2
    )
    args = Namespace(
        execution_mode="auto",
        oneshot="Reply exactly OK",
        provider="custom:zerogpu",
        model=None,
        usage_file=str(tmp_path / "usage.json"),
    )
    seen = {}
    code = run_pre_agent_oneshot(
        args,
        load_config=lambda: CONFIG,
        get_secret=lambda name, default: "key",
        opener=fake_opener(seen),
    )
    assert code == 0
    assert capsys.readouterr().out.strip() == "OK"
    usage = json.loads((tmp_path / "usage.json").read_text())
    assert usage["api_calls"] == 1 and usage["input_tokens"] == 38


def test_bounded_limit_and_auto_fallback_to_full_agent():
    args = Namespace(
        execution_mode="bounded",
        oneshot="something",
        max_turns=None,
        toolsets="clarify",
    )
    assert run_pre_agent_oneshot(args) is None and args.max_turns == 2
    args.max_turns = 1
    assert run_pre_agent_oneshot(args) is None and args.max_turns == 1
    assert (
        run_pre_agent_oneshot(
            Namespace(execution_mode="bounded", oneshot="x", toolsets=None)
        )
        == 2
    )
    args = Namespace(
        execution_mode="auto",
        oneshot="Fix repository",
        provider="custom:zerogpu",
        model=None,
    )
    assert run_pre_agent_oneshot(args, load_config=lambda: CONFIG) is None


def test_http_transport_rejects_redirect_and_uses_hermes_user_agent(monkeypatch):
    import httpx
    from urllib.request import Request
    from hermes_cli.execution_gate import _single_request

    seen = {}

    def fake_request(method, url, **kwargs):
        seen.update({"method": method, "url": url, **kwargs})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "OK"}}]},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(httpx, "request", fake_request)
    req = Request(
        "https://example.test/v1/chat/completions",
        data=b"{}",
        headers={"Authorization": "Bearer unit-test"},
    )
    with _single_request(req, 12) as body:
        assert json.load(body)["choices"][0]["message"]["content"] == "OK"
    assert seen["follow_redirects"] is False
    assert seen["timeout"] == 12
    assert seen["headers"]["User-Agent"] == "Hermes-ExecutionGate/1.0"
