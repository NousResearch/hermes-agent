"""``hermes fallback check`` (#63852): per-entry readiness through the fallback switch's own resolvers."""
from __future__ import annotations

import json
import types

import httpx
import pytest
from openai import OpenAI

import hermes_yaml as yaml
from hermes_cli import fallback_check as fc

_KEY = "sk-fbcheck" + "K" * 30


# tests/conftest.py already points HERMES_HOME at a per-test dir and blanks credential env vars.
def _config(chain):
    from hermes_constants import get_hermes_home
    data = {"custom_providers": [{"name": "local-oai", "base_url": "http://127.0.0.1:9/v1", "api_key": _KEY}],
            "fallback_providers": chain}
    (get_hermes_home() / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _route_to(monkeypatch, client, sent_model=None, seen=None):
    from hermes_agent.agent.route_binding import RouteEntryClient

    def route(entry, provider, model, raw_codex=True):
        if seen is not None:
            seen.append((provider, model, raw_codex))
        return RouteEntryClient(client, sent_model or model, False, "chat_completions")
    monkeypatch.setattr("hermes_agent.agent.route_binding.resolve_route_entry_client", route)


def _openai_client(handler, calls):
    def record(request):
        calls.append(json.loads(request.content))
        return handler(request)
    return OpenAI(api_key=_KEY, base_url="http://stand.in/v1", max_retries=0,
                  http_client=httpx.Client(transport=httpx.MockTransport(record)))


def _ready(model="m", provider="local-oai"):
    return fc.EntryCheck(provider=provider, model=model, configured=True, auth_ready=True)


def _completion(model):
    return httpx.Response(200, json={
        "id": "c1", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}]})


@pytest.mark.parametrize("requested,returned,ok", [
    ("gpt-4o", "gpt-4o", True),
    ("gpt-4o", "gpt-4o-2024-08-06", True),
    ("claude-sonnet-4", "claude-sonnet-4-20250514", True),
    ("anthropic/claude-sonnet-4", "claude-sonnet-4", True),
    ("meta/llama-3:free", "llama-3", True),
    ("gpt-4o", "", True),
    ("gpt-4o", "gpt-4o-mini", False),
    ("claude-sonnet-4", "claude-haiku-4", False),
    ("gpt-5", "gpt-5.5", False),
    ("org-a/model", "org-b/model", False),
    ("qwen3:8b", "qwen3:32b", False),
    ("qwen3:8b", "qwen3:8b", True),
    ("google/gemini-2.5-pro", "publishers/google/models/gemini-2.5-pro", True),
    ("publishers/google/models/gemini-2.5-pro", "google/gemini-2.5-flash", False),
    ("google/gemini-2.5-pro", "projects/p/locations/us-central1/publishers/google/models/gemini-2.5-pro", True),
])
def test_answered_by_requested_model(requested, returned, ok):
    assert fc.answered_by_requested_model(requested, returned) is ok


def test_static_unknown_provider_is_a_config_failure():
    result = fc.check_entry_static({"provider": "no-such-provider", "model": "m"})
    assert (result.configured, result.ready, result.status) == (False, False, "config")


def test_static_missing_credential_is_an_auth_failure():
    result = fc.check_entry_static({"provider": "deepseek", "model": "deepseek-chat"})
    assert (result.configured, result.auth_ready, result.status) == (True, False, "auth")


def test_static_named_custom_provider_uses_its_own_endpoint_and_key():
    _config([])
    result = fc.check_entry_static({"provider": "local-oai", "model": "m"})
    assert result.ready and result.inference_ready is None
    assert result.base_url == "http://127.0.0.1:9/v1"


def test_static_entry_key_env_counts_as_credential(monkeypatch):
    monkeypatch.setenv("FB_CHECK_KEY", _KEY)
    result = fc.check_entry_static({"provider": "deepseek", "model": "deepseek-chat", "key_env": "FB_CHECK_KEY"})
    assert result.auth_ready, result.error


def test_live_mismatch_fails_and_sends_exactly_one_request(monkeypatch):
    calls: list = []
    _route_to(monkeypatch, _openai_client(lambda request: _completion("some-other-model"), calls))
    entry = {"provider": "local-oai", "model": "requested-model"}
    result = fc.check_entry_live(entry, _ready("requested-model"))
    assert (result.inference_ready, result.status, result.model_returned) == (False, "model_mismatch", "some-other-model")
    assert [c["model"] for c in calls] == ["requested-model"]


def test_live_provider_error_is_classified_redacted_and_not_retried(monkeypatch):
    calls: list = []
    echoed = "sk-echoed" + "Z" * 30
    _route_to(monkeypatch, _openai_client(lambda request: httpx.Response(
        401, json={"error": {"message": f"bad key {echoed}", "type": "authentication_error"}}), calls))
    result = fc.check_entry_live({"provider": "local-oai", "model": "m"},
                                 _ready())
    assert result.status == "live_auth" and echoed not in result.error
    assert len(calls) == 1


def _oai(url):
    return OpenAI(api_key=_KEY, base_url=url)


@pytest.mark.parametrize("provider,model,client,trusted", [
    ("x", "m", object(), False),  # Anthropic Messages / Codex Responses adapters echo the requested model
    ("x", "m", _oai("https://api.example.com/v1"), True),
    # Azure reports the base model, not the deployment, on every Azure host family and sovereign cloud.
    ("x", "m", _oai("https://res.openai.azure.com/openai/v1"), False),
    ("x", "m", _oai("https://res.cognitiveservices.azure.com/openai/v1"), False),
    ("x", "m", _oai("https://res.openai.azure.us/openai/v1"), False),
    ("azure-foundry", "m", _oai("https://ai.contoso.example/v1"), False),
    ("x", "m", _oai("http://127.0.0.1:8080/v1"), False),  # local servers report placeholders
    ("openrouter", "openrouter/auto", _oai("https://openrouter.ai/api/v1"), False),  # answers as its pick
    ("openrouter", "@preset/mine", _oai("https://openrouter.ai/api/v1"), False),
    ("openrouter", "openai/gpt-4o@preset/mine", _oai("https://openrouter.ai/api/v1"), False),
    ("azure", "m", _oai("https://ai.contoso.example/v1"), False),  # alias of azure-foundry
    ("bedrock", "arn:aws:bedrock:us-east-1:1:inference-profile/x", _oai("https://bedrock-mantle.example/v1"), False),
])
def test_reported_model_is_evidence_only_where_it_proves_a_swap(provider, model, client, trusted):
    assert fc._reported_model_is_evidence(provider, model, client) is trusted


def test_live_passes_the_swap_shape_and_provider_timeout(monkeypatch):
    calls, seen = [], []
    _route_to(monkeypatch, _openai_client(lambda request: _completion("m"), calls), seen=seen)
    monkeypatch.setattr("hermes_cli.timeouts.get_provider_request_timeout", lambda provider, model: 7.0)
    sent = {}
    import hermes_agent.agent.auxiliary_client as aux
    real = aux._build_call_kwargs
    monkeypatch.setattr(aux, "_build_call_kwargs", lambda *a, **k: sent.update(k) or real(*a, **k))
    _config([])
    static = fc.check_entry_static({"provider": "Local-OAI", "model": "m"})
    result = fc.check_entry_live({"provider": "Local-OAI", "model": "m"}, static)
    assert result.provider == "local-oai" and result.status == "ready"
    assert seen == [("local-oai", "m", False)] and sent["timeout"] == 7.0


def test_live_compares_against_the_model_actually_sent(monkeypatch):
    calls: list = []
    _route_to(monkeypatch, _openai_client(lambda request: _completion("vendor/normalized-id"), calls),
              sent_model="vendor/normalized-id")
    result = fc.check_entry_live({"provider": "local-oai", "model": "Alias"},
                                 _ready("Alias"))
    assert result.status == "ready" and [c["model"] for c in calls] == ["vendor/normalized-id"]


def test_cli_json_exit_code_and_no_raw_entry_key(monkeypatch, capsys):
    _config([{"provider": "local-oai", "model": "good", "api_key": _KEY},
             {"provider": "no-such-provider", "model": "m"}])
    calls: list = []
    _route_to(monkeypatch, _openai_client(lambda request: _completion("good"), calls))
    with pytest.raises(SystemExit) as exc:
        fc.cmd_fallback_check(types.SimpleNamespace(live=True, json=True))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert [(r["model"], r["status"]) for r in payload["results"]] == [("good", "ready"), ("m", "config")]
    assert payload["ready"] is False and {"profile", "checked_at", "live"} <= payload.keys()
    assert _KEY not in out and len(calls) == 1


def test_cli_all_ready_exits_zero_without_inference_in_static_mode(monkeypatch, capsys):
    _config([{"provider": "local-oai", "model": "good"}])
    monkeypatch.setattr("hermes_agent.agent.route_binding.resolve_route_entry_client",
                        lambda *a, **k: pytest.fail("static mode must not build an inference client"))
    fc.cmd_fallback_check(types.SimpleNamespace(live=False, json=False))
    assert "ready (static)" in capsys.readouterr().out


def test_parser_wires_check_subcommand():
    import argparse
    from hermes_cli.subcommands.fallback import build_fallback_parser
    parser = argparse.ArgumentParser()
    build_fallback_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["fallback", "check", "--live", "--json"])
    assert (args.fallback_command, args.live, args.json) == ("check", True, True)


_ANSWER = {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}


@pytest.mark.parametrize("body,status", [
    ({"choices": [{"index": 0, "message": {"role": "assistant",
                                          "content": "Connect timeout, please try again later."},
                   "finish_reason": "stop"}]}, "live_response"),  # router timeout shim
    ({"choices": [{"index": 0, "finish_reason": "stop"}]}, "live_response"),
    ({"choices": [{"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "stop"}]},
     "live_response"),
    ({"choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]},
     "live_response"),
    # Partial text under a failed status is not an answer.
    ({"choices": [_ANSWER], "status": "failed"}, "live_response"),
    # An HTTP-200 relay of an upstream failure is classified like the request error it reports.
    ({"choices": [], "error": {"code": "invalid_api_key", "message": "Incorrect API key provided", "status": 401}},
     "live_auth"),
    # An always-present error object with empty fields is not a failure.
    ({"choices": [_ANSWER], "error": {"message": None, "code": None}}, "ready"),
    ({"choices": [_ANSWER], "error": {"code": 0, "message": ""}}, "ready"),
    ({"success": True, "data": {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                                "choices": [_ANSWER]}}, "ready"),
    # Envelope normalization drops status/error, so both are read from the envelope first.
    ({"success": True, "data": {"choices": [_ANSWER], "status": "failed"}}, "live_response"),
    ({"success": True, "data": {"choices": [_ANSWER], "error": {"code": "invalid_api_key", "message": "x",
                                                               "status": 401}}}, "live_auth"),
    ({"success": False, "data": {"error": {"code": "invalid_api_key", "status": 401, "message": "denied"}}},
     "live_auth"),  # classified from the code
    ({"success": False, "data": {"error": {"message": "Unauthorized: invalid api key"}}}, "live_auth"),  # message
    # OpenRouter relays an upstream failure on the choice, possibly beside partial text.
    ({"choices": [{**_ANSWER, "finish_reason": "error", "error": {"code": 502, "message": "upstream died"}}]},
     "live_response_or_error"),
    ({"choices": [{**_ANSWER, "finish_reason": "error"}]}, "live_response"),
    ({"choices": [{**_ANSWER, "error": {"code": 401, "message": "Invalid API key"}}]}, "live_auth"),
    ({"success": False, "data": {"detail": "nope"}}, "live_unknown"),  # no error object: still a failure
    # Responses-style text outside ``choices`` is an answer, as on the aux path.
    ({"choices": [], "object": "response", "status": "completed", "output_text": "OK"}, "ready"),
    ({"choices": [], "object": "response", "status": "failed", "output_text": "OK"}, "live_response"),
    # A malformed Responses-style body is no answer, not a crash.
    ({"choices": [], "object": "response", "output": [{"type": "message", "content": 42}]}, "live_response"),
])
def test_live_judges_the_answer_body(monkeypatch, body, status):
    _route_to(monkeypatch, _openai_client(lambda request: httpx.Response(
        200, json={"id": "c", "object": "chat.completion", "created": 0, "model": "m", **body}), []))
    result = fc.check_entry_live({"provider": "local-oai", "model": "m"}, _ready())
    if status == "live_response_or_error":  # a choice-level error is classified, never ready
        assert result.inference_ready is False and result.status.startswith("live_"), result.status
    else:
        assert result.status == status, result.error


def test_live_redacts_a_credential_shaped_reported_model(monkeypatch):
    leaked = "sk-proj-" + "L" * 40
    _route_to(monkeypatch, _openai_client(lambda request: _completion(leaked), []))
    result = fc.check_entry_live({"provider": "local-oai", "model": "m"},
                                 _ready())
    assert result.status == "model_mismatch"
    assert leaked not in result.model_returned and leaked not in result.error


def test_live_scrubs_the_entry_key_even_without_a_known_prefix(monkeypatch):
    plain = "mango-example-12345"
    _route_to(monkeypatch, _openai_client(lambda request: httpx.Response(
        401, json={"error": {"message": f"Rejected credential {plain}", "type": "auth"}}), []))
    result = fc.check_entry_live({"provider": "local-oai", "model": "m", "api_key": plain},
                                 _ready())
    assert result.status == "live_auth" and plain not in result.error


def test_codex_wrapper_inner_client_loses_sdk_retries():
    from hermes_agent.agent.auxiliary_client import CodexAuxiliaryClient
    wrapped = fc._without_sdk_retries(CodexAuxiliaryClient(OpenAI(api_key=_KEY, base_url="http://h/v1"), "m"))
    assert isinstance(wrapped, CodexAuxiliaryClient) and wrapped._real_client.max_retries == 0
    assert wrapped.chat.completions._model == "m"


def test_live_keeps_aws_instance_metadata_and_skips_auto_and_moa(monkeypatch):
    seen = {}
    monkeypatch.setattr("hermes_cli.doctor_connectivity.run_probes",
                        lambda probes, **kw: seen.update(kw, names=[n for n, _ in probes]) or [])
    ready = lambda entry: fc.EntryCheck(provider=entry["provider"], model="m", configured=True, auth_ready=True)
    monkeypatch.setattr(fc, "check_entry_static", ready)
    chain = [{"provider": "auto", "model": "m"}, {"provider": "moa", "model": "p"}, {"provider": "bedrock", "model": "m"}]
    assert [r.status for r in fc.run_checks(chain, live=True)] == ["ready (static)"] * 3
    assert seen == {"aws_imds": True, "names": ["bedrock"]}


def test_no_latency_when_no_request_was_sent(monkeypatch):
    _route_to(monkeypatch, None)
    result = fc.check_entry_live({"provider": "local-oai", "model": "m"}, _ready())
    assert (result.status, result.latency_ms) == ("auth", None)


def test_a_judging_error_fails_the_entry_not_the_command(monkeypatch):
    _route_to(monkeypatch, _openai_client(lambda request: _completion("m"), []))
    monkeypatch.setattr("hermes_agent.agent.auxiliary_client.extract_content_or_reasoning",
                        lambda response: (_ for _ in ()).throw(TypeError("odd body")))
    result = fc.check_entry_live({"provider": "local-oai", "model": "m"}, _ready())
    assert result.inference_ready is False and result.status.startswith("live_") and result.latency_ms is not None
