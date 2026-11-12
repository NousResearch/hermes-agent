"""One MCP tool whose schema the provider rejects must not take down every request.

Anthropic's validator is stricter than JSON Schema's meta-schema, so a schema can pass every local check and
still 400 the whole request with ``tools.N.custom.input_schema: JSON schema is invalid``. Only the rejection
names the tool. Hermes drops that one MCP tool, tells the user, and retries.

The turn tests run the real conversation loop against a fake provider that rejects any request carrying the bad
tool, so they assert on what reached the wire, not on what the recovery logged.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery_tool_prune import recover_rejected_tool_schema
from agent.turn_retry_state import TurnRetryState
from run_agent import AIAgent
from tests.fakes.fake_llm_provider import Error, FakeLLMServer, Text
from tools.registry import registry

_REJECTION = "tools.{}.custom.input_schema: JSON schema is invalid. It must match JSON Schema draft 2020-12"
_BAD = "mcp_srv_bad_tool"
_ALSO_BAD = "mcp_srv_also_bad"
_GOOD = "mcp_srv_good_tool"
_BUILTIN = "builtin_tool"


class _APIError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code
        self.body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
        self.response = SimpleNamespace(headers={})


def _openai_tool(name: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": name, "parameters": {"type": "object", "properties": {}}}}


def _handler(args, **_kw):
    return "{}"


@pytest.fixture
def registered_tools():
    for name in (_BAD, _ALSO_BAD, _GOOD):
        registry.register(name=name, toolset="mcp-srv", schema=_openai_tool(name)["function"], handler=_handler)
    registry.register(name=_BUILTIN, toolset="core", schema=_openai_tool(_BUILTIN)["function"], handler=_handler)
    try:
        yield
    finally:
        for name in (_BAD, _ALSO_BAD, _GOOD, _BUILTIN):
            registry.deregister(name)


def _stub_agent(*names: str) -> SimpleNamespace:
    warnings: list[str] = []
    return SimpleNamespace(
        tools=[_openai_tool(n) for n in names], valid_tool_names=set(names), log_prefix="", warnings=warnings,
        _emit_warning=warnings.append,
    )


def _recover(agent, message: str, sent: list, retry: TurnRetryState | None = None) -> bool:
    error = _APIError(message)
    classified = classify_api_error(error, provider="anthropic", model="m")
    return recover_rejected_tool_schema(agent, error, classified, retry or TurnRetryState(), {"tools": sent})


# --- classification ---------------------------------------------------------------------------------------------


def test_rejection_is_classified_as_not_worth_retrying_unchanged():
    verdict = classify_api_error(_APIError(_REJECTION.format(4)), provider="anthropic", model="m")
    assert verdict.reason == FailoverReason.tool_schema_rejected
    assert verdict.retryable is False


def test_a_body_that_only_echoes_the_path_is_not_this_rejection(registered_tools):
    """A proxy can echo the request into an unrelated 400. Matching that would delete a healthy tool."""
    echoed = "invalid temperature. Received messages=[...see tools.0.custom.input_schema...]"
    agent = _stub_agent(_GOOD)

    verdict = classify_api_error(_APIError(echoed), provider="anthropic", model="m")
    assert verdict.reason != FailoverReason.tool_schema_rejected
    assert _recover(agent, echoed, [_openai_tool(_GOOD)]) is False
    assert agent.valid_tool_names == {_GOOD}


def test_other_statuses_and_messages_are_not_this_rejection():
    other_status = classify_api_error(_APIError(_REJECTION.format(4), 500), provider="anthropic", model="m")
    assert other_status.reason != FailoverReason.tool_schema_rejected
    other_message = classify_api_error(_APIError("messages.0: invalid"), provider="anthropic", model="m")
    assert other_message.reason != FailoverReason.tool_schema_rejected


# --- which tool, and whether to drop it ---------------------------------------------------------------------------


def test_index_maps_to_the_array_as_sent_not_to_agent_tools(registered_tools):
    """The wire array can differ from ``agent.tools``: here the same tools, in another order."""
    agent = _stub_agent(_GOOD, _BAD)
    sent = [{"name": _BAD, "input_schema": {}}, {"name": _GOOD, "input_schema": {}}]

    assert _recover(agent, _REJECTION.format(0), sent) is True

    assert [t["function"]["name"] for t in agent.tools] == [_GOOD]
    assert agent.valid_tool_names == {_GOOD}


def test_chat_completions_shaped_tools_are_read_too(registered_tools):
    """An OpenAI-compatible relay in front of Anthropic returns the same rejection for an OpenAI-shaped array."""
    agent = _stub_agent(_BAD, _GOOD)

    assert _recover(agent, _REJECTION.format(0), [_openai_tool(_BAD), _openai_tool(_GOOD)]) is True

    assert [t["function"]["name"] for t in agent.tools] == [_GOOD]


def test_oauth_wire_name_resolves_to_the_registered_tool(registered_tools):
    """Anthropic OAuth sends ``mcp_srv_x`` as ``mcp__srv_x``."""
    agent = _stub_agent(_BAD, _GOOD)

    assert _recover(agent, _REJECTION.format(0), [{"name": "mcp__srv_bad_tool"}, {"name": _GOOD}]) is True

    assert [t["function"]["name"] for t in agent.tools] == [_GOOD]


def test_builtin_tool_is_never_dropped(registered_tools):
    """Hermes's own tool failing is a bug to surface; quietly losing it could be worse than the failed turn."""
    agent = _stub_agent(_BUILTIN, _GOOD)

    assert _recover(agent, _REJECTION.format(0), [_openai_tool(_BUILTIN), _openai_tool(_GOOD)]) is False

    assert agent.valid_tool_names == {_BUILTIN, _GOOD}
    assert [t["function"]["name"] for t in agent.tools] == [_BUILTIN, _GOOD]
    assert not getattr(agent, "_rejected_schema_tools", None)
    assert len(agent.warnings) == 1 and _BUILTIN in agent.warnings[0], "the log is not the only place it is named"


@pytest.mark.parametrize("sent", [[], [_openai_tool(_GOOD)]], ids=["empty", "index-past-end"])
def test_index_that_maps_to_nothing_declines(registered_tools, sent):
    agent = _stub_agent(_GOOD)

    assert _recover(agent, _REJECTION.format(3), sent) is False

    assert agent.valid_tool_names == {_GOOD}


def test_prunes_are_capped_per_call(registered_tools):
    agent = _stub_agent(_BAD, _ALSO_BAD, _GOOD)
    retry = TurnRetryState(tool_schema_prunes=4)
    sent = [_openai_tool(_BAD), _openai_tool(_ALSO_BAD), _openai_tool(_GOOD)]

    assert _recover(agent, _REJECTION.format(0), sent, retry) is True
    assert _recover(agent, _REJECTION.format(1), sent, retry) is False, "the fifth prune is the last"

    assert agent.valid_tool_names == {_ALSO_BAD, _GOOD}


def test_shared_tool_dicts_are_not_mutated(registered_tools):
    """``get_tool_definitions`` hands out shallow copies of a process-wide cache: the dicts are every session's."""
    agent = _stub_agent(_BAD, _GOOD)
    before = agent.tools
    kept = before[1]

    assert _recover(agent, _REJECTION.format(0), [_openai_tool(_BAD), _openai_tool(_GOOD)]) is True

    assert [t["function"]["name"] for t in before] == [_BAD, _GOOD], "the original list must not change"
    assert agent.tools[0] is kept


# --- through the real loop -------------------------------------------------------------------------------------


def _rejecting_provider(record):
    """Reject the first tool in the request that is known bad, the way the real validator names one at a time."""
    tools = record["body"].get("tools") or []
    names = [t["function"]["name"] for t in tools]
    for index, name in enumerate(names):
        if name in (_BAD, _ALSO_BAD):
            return Error(status=400, message=_REJECTION.format(index))
    return Text("PONG")


def _agent_for(srv, *names: str) -> AIAgent:
    agent = AIAgent(
        base_url=srv.base_url, api_key="test-key", model="fake-model", provider="custom", quiet_mode=True,
        skip_context_files=True, skip_memory=True, enabled_toolsets=[],
    )
    agent.tools = [_openai_tool(n) for n in names]
    agent.valid_tool_names = set(names)
    return agent


def _sent_names(requests: list) -> list[list[str]]:
    return [[t["function"]["name"] for t in (r["tools"] or [])] for r in requests]


def test_the_retry_really_goes_out_without_the_tool(registered_tools):
    with FakeLLMServer(_rejecting_provider) as srv:
        agent = _agent_for(srv, _GOOD, _BAD)
        result = agent.run_conversation("Reply with exactly PONG.")
        requests = srv.main_requests()

    assert _sent_names(requests) == [[_GOOD, _BAD], [_GOOD]]
    assert result["final_response"] == "PONG"


def test_several_bad_tools_are_pruned_one_at_a_time(registered_tools):
    with FakeLLMServer(_rejecting_provider) as srv:
        agent = _agent_for(srv, _BAD, _GOOD, _ALSO_BAD)
        result = agent.run_conversation("Reply with exactly PONG.")
        requests = srv.main_requests()

    assert _sent_names(requests) == [[_BAD, _GOOD, _ALSO_BAD], [_GOOD, _ALSO_BAD], [_GOOD]]
    assert result["final_response"] == "PONG"


def test_the_next_turn_does_not_pay_for_the_rejection_again(registered_tools):
    with FakeLLMServer(_rejecting_provider) as srv:
        agent = _agent_for(srv, _GOOD, _BAD)
        agent.run_conversation("Reply with exactly PONG.")
        agent.run_conversation("Again.")
        requests = srv.main_requests()

    assert _sent_names(requests) == [[_GOOD, _BAD], [_GOOD], [_GOOD]]


def test_a_rejected_builtin_fails_the_turn_without_repeating_the_request(registered_tools):
    def _reject_builtin(record):
        names = [t["function"]["name"] for t in record["body"].get("tools") or []]
        return Error(status=400, message=_REJECTION.format(names.index(_BUILTIN)))

    with FakeLLMServer(_reject_builtin) as srv:
        agent = _agent_for(srv, _BUILTIN, _GOOD)
        result = agent.run_conversation("Reply with exactly PONG.")
        requests = srv.main_requests()

    assert len(requests) == 1, "an identical request would be rejected identically"
    assert result.get("failed") is True, result
    assert [t["function"]["name"] for t in agent.tools] == [_BUILTIN, _GOOD]


def test_the_user_is_told_through_the_rail_every_surface_reads(registered_tools):
    """``_vprint`` reaches the CLI only; a gateway or desktop user would lose a tool without a word."""
    seen: list[str] = []
    with FakeLLMServer(_rejecting_provider) as srv:
        agent = _agent_for(srv, _GOOD, _BAD)
        agent.status_callback = lambda _kind, message: seen.append(str(message))
        agent.run_conversation("Reply with exactly PONG.")

    assert any(_BAD in message for message in seen), seen


def test_an_mcp_refresh_does_not_bring_the_tool_back(monkeypatch, registered_tools):
    """A real build with tool search off, then the refresh compaction runs: it reads the registry afresh."""
    import model_tools
    from tools import tool_search
    from tools.mcp_tool_agent import refresh_agent_mcp_tools

    monkeypatch.setattr(tool_search, "load_config", lambda *a, **k: tool_search.ToolSearchConfig.from_raw({"enabled": "off"}))
    model_tools._clear_tool_defs_cache()
    with FakeLLMServer(_rejecting_provider) as srv:
        agent = AIAgent(
            base_url=srv.base_url, api_key="test-key", model="fake-model", provider="custom", quiet_mode=True,
            skip_context_files=True, skip_memory=True, enabled_toolsets=["mcp-srv"],
        )
        assert {_BAD, _ALSO_BAD, _GOOD} <= agent.valid_tool_names, "the build must carry the tools for this to test anything"
        agent.run_conversation("Reply with exactly PONG.")
        refresh_agent_mcp_tools(agent, content_aware=True)
        after_refresh = len(srv.main_requests())
        agent.run_conversation("Again.")
        next_turn = srv.main_requests()[after_refresh:]
    model_tools._clear_tool_defs_cache()

    assert len(next_turn) == 1, "no second rejection"
    assert {_BAD, _ALSO_BAD}.isdisjoint(_sent_names(next_turn)[0])
    assert _GOOD in _sent_names(next_turn)[0]


class _FakeAnthropic:
    """The native Messages API, which rejects the bad tool the way the real one names it."""

    def __init__(self):
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _send(self, status: int, payload: bytes, content_type: str = "application/json"):
                self.send_response(status)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                outer.requests.append(body)
                for index, tool in enumerate(body.get("tools") or []):
                    if tool.get("name") == _BAD:
                        error = {"type": "invalid_request_error", "message": _REJECTION.format(index)}
                        return self._send(400, json.dumps({"type": "error", "error": error}).encode())
                message = {"id": "m", "type": "message", "role": "assistant", "model": "x", "stop_sequence": None,
                           "content": [], "stop_reason": None, "usage": {"input_tokens": 1, "output_tokens": 1}}
                if not body.get("stream"):
                    final = {**message, "content": [{"type": "text", "text": "PONG"}], "stop_reason": "end_turn"}
                    return self._send(200, json.dumps(final).encode())
                events = [
                    {"type": "message_start", "message": message},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "PONG"}},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                     "usage": {"output_tokens": 1}},
                    {"type": "message_stop"},
                ]
                stream = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
                return self._send(200, stream, "text/event-stream")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "non-streaming"])
def test_native_anthropic_retry_goes_out_without_the_tool(registered_tools, streaming):
    provider = _FakeAnthropic()
    try:
        agent = AIAgent(
            base_url=provider.url, api_key="sk-ant-api03-test", model="claude-sonnet-4-5", provider="anthropic",
            api_mode="anthropic_messages", quiet_mode=True, skip_context_files=True, skip_memory=True,
            enabled_toolsets=[],
        )
        agent.tools = [_openai_tool(_GOOD), _openai_tool(_BAD)]
        agent.valid_tool_names = {_GOOD, _BAD}
        agent._disable_streaming = not streaming
        result = agent.run_conversation("Reply with exactly PONG.")
    finally:
        provider.server.shutdown()

    sent = [[t["name"] for t in r.get("tools") or []] for r in provider.requests]
    assert [names for names in sent if names] == [[_GOOD, _BAD], [_GOOD]]
    assert result["final_response"] == "PONG"
