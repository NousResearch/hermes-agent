"""Ollama wire alias for the Tool Search bridge ``tool_call`` (agent/chat_completion_wire_aliases.py).

Captured on Ollama 0.40.6 (Qwen 3, production CLI catalog, tool execution blocked): with the
bridge advertised as ``tool_call``, the model generated a valid ``web_search`` call and the
response named it ``tool_call`` (0/9 intact); with the bridge advertised as
``call_deferred_tool`` the same generations came back as ``web_search`` (9/9). Hermes then
rejected the reattributed call because ``tool_call`` requires ``calls``. The fake server below
models that observed reattribution; it does not claim to reproduce Ollama's parser internals.
"""

from __future__ import annotations

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from agent.chat_completion_wire_aliases import (
    OLLAMA_TOOL_CALL_ALIAS,
    alias_history_tool_names,
    alias_tool_call_for_ollama,
    apply_wire_tool_aliases,
    canonical_stream_tool_name,
    ollama_tool_call_alias_route,
)
from agent.transports import get_transport

_STANDALONE_TOOL_CALL = re.compile(r"(?<![A-Za-z0-9_<>/])tool_call(?![A-Za-z0-9_>])")


def _td(name, description="", properties=None, required=None):
    params = {"type": "object", "properties": properties or {}}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


def _real_bridge_defs():
    """Bridge declarations exactly as Tool Search assembles them (real descriptions)."""
    from tools.registry import discover_builtin_tools, registry
    from tools.tool_search import ToolSearchConfig, assemble_tool_defs

    discover_builtin_tools()
    registry.register(name="mcp_wirealias_probe", handler=lambda args, **kw: "{}",
                      schema=_td("mcp_wirealias_probe", "Deferred probe")["function"], toolset="mcp-wirealias-probe")
    try:
        assembled = assemble_tool_defs(
            [_td("web_search", "Search the web", {"query": {"type": "string"}}, ["query"]),
             _td("mcp_wirealias_probe", "Deferred probe")],
            context_length=200_000, config=ToolSearchConfig.from_raw({"enabled": "on"}))
    finally:
        registry.deregister("mcp_wirealias_probe")
    assert assembled.activated
    return assembled.tool_defs


def _response(name, arguments):
    tc = SimpleNamespace(id="call_1", function=SimpleNamespace(name=name, arguments=arguments))
    msg = SimpleNamespace(tool_calls=[tc], content=None, reasoning=None, reasoning_content=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=None)


class TestCapturedCondition:
    def test_reattributed_web_search_is_rejected_not_guessed(self):
        """The captured control: the reply names ``tool_call`` with web_search's arguments. Without
        an alias it stays ``tool_call`` and bridge validation rejects it; nothing remaps it."""
        from tools.tool_search_validation import normalize_tool_call_entries

        transport = get_transport("chat_completions")
        transport._last_wire_aliases = {}
        captured_args = '{"query":"Who wrote Pride and Prejudice?","limit":5}'
        call = transport.normalize_response(_response("tool_call", captured_args)).tool_calls[0]
        assert call.name == "tool_call"
        assert call.arguments == captured_args
        entries, error = normalize_tool_call_entries(json.loads(call.arguments))
        assert entries == [] and "requires 'calls'" in error


class TestAliasDeclarations:
    def test_bridge_advertised_under_noncolliding_name_with_coherent_descriptions(self):
        tools = _real_bridge_defs()
        before = json.dumps(tools, sort_keys=True)
        wire, alias_map = alias_tool_call_for_ollama(tools)
        names = [t["function"]["name"] for t in wire]
        assert alias_map == {OLLAMA_TOOL_CALL_ALIAS: "tool_call"}
        assert OLLAMA_TOOL_CALL_ALIAS in names and "tool_call" not in names
        assert "web_search" in names and "tool_search" in names and "tool_describe" in names
        for tool in wire:
            assert not _STANDALONE_TOOL_CALL.search(tool["function"].get("description", ""))
        described = " ".join(t["function"]["description"] for t in wire)
        assert f"`{OLLAMA_TOOL_CALL_ALIAS}`" in described
        assert json.dumps(tools, sort_keys=True) == before  # input never mutated
        canonical = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        assert {OLLAMA_TOOL_CALL_ALIAS if n == "tool_call" else n: p for n, p in canonical.items()} == \
            {t["function"]["name"]: t["function"]["parameters"] for t in wire}

    def test_markup_and_longer_identifiers_are_not_references(self):
        text = "Emit <tool_call></tool_call>; see tool_calls, post_tool_call and `tool_call`."
        wire, _ = alias_tool_call_for_ollama([_td("tool_call", text), _td("other", text)])
        for tool in wire:
            assert tool["function"]["description"] == text.replace("`tool_call`", f"`{OLLAMA_TOOL_CALL_ALIAS}`")

    def test_no_bridge_no_change(self):
        tools = [_td("web_search"), _td("tool_describe")]
        wire, alias_map = alias_tool_call_for_ollama(tools)
        assert wire is tools and alias_map == {}

    def test_real_tool_owning_the_alias_keeps_its_identity(self):
        tools = [_td(OLLAMA_TOOL_CALL_ALIAS, "a real plugin tool"), _td("tool_call")]
        wire, alias_map = alias_tool_call_for_ollama(tools)
        names = [t["function"]["name"] for t in wire]
        assert names == [OLLAMA_TOOL_CALL_ALIAS, f"{OLLAMA_TOOL_CALL_ALIAS}_2"]
        assert alias_map == {f"{OLLAMA_TOOL_CALL_ALIAS}_2": "tool_call"}
        transport = get_transport("chat_completions")
        transport._last_wire_aliases = alias_map
        assert transport.normalize_response(_response(OLLAMA_TOOL_CALL_ALIAS, "{}")).tool_calls[0].name == OLLAMA_TOOL_CALL_ALIAS
        assert transport.normalize_response(_response(f"{OLLAMA_TOOL_CALL_ALIAS}_2", "{}")).tool_calls[0].name == "tool_call"


class TestNormalizeAndHistory:
    def test_alias_reply_maps_to_canonical_with_arguments_and_validation_intact(self):
        from tools.tool_search_validation import normalize_tool_call_entries

        transport = get_transport("chat_completions")
        transport._last_wire_aliases = {OLLAMA_TOOL_CALL_ALIAS: "tool_call"}
        good = '{"calls":[{"name":"mcp_x","arguments":{"q":"a"}}]}'
        call = transport.normalize_response(_response(OLLAMA_TOOL_CALL_ALIAS, good)).tool_calls[0]
        assert (call.name, call.arguments) == ("tool_call", good)
        bad = '{"queries":["who wrote frankenstein"]}'
        call = transport.normalize_response(_response(OLLAMA_TOOL_CALL_ALIAS, bad)).tool_calls[0]
        assert call.name == "tool_call"
        assert "requires 'calls'" in normalize_tool_call_entries(json.loads(call.arguments))[1]
        assert transport.normalize_response(_response("web_search", '{"query":"q"}')).tool_calls[0].name == "web_search"

    def test_unadvertised_alias_is_never_reversed(self):
        transport = get_transport("chat_completions")
        transport._last_wire_aliases = {}
        assert transport.normalize_response(_response(OLLAMA_TOOL_CALL_ALIAS, "{}")).tool_calls[0].name == OLLAMA_TOOL_CALL_ALIAS

    def test_history_wire_copy_is_deterministic_and_stored_history_untouched(self):
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "tool_call", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a", "name": "tool_call", "content": "r"},
            {"role": "tool", "tool_call_id": "b", "content": "r"},
        ]
        snapshot = json.dumps(history, sort_keys=True)
        names = {"tool_call": OLLAMA_TOOL_CALL_ALIAS}
        wire = alias_history_tool_names(history, names)
        assert json.dumps(history, sort_keys=True) == snapshot
        assert [tc["function"]["name"] for tc in wire[1]["tool_calls"]] == [OLLAMA_TOOL_CALL_ALIAS, "web_search"]
        assert wire[2]["name"] == OLLAMA_TOOL_CALL_ALIAS
        assert wire[0] is history[0] and wire[3] is history[3]
        assert json.dumps(alias_history_tool_names(history, names)) == json.dumps(wire)
        plain = [{"role": "user", "content": "q"}]
        assert alias_history_tool_names(plain, names) is plain

    def test_stream_progress_names_are_canonical(self):
        agent = SimpleNamespace(_ollama_wire_tool_aliases={OLLAMA_TOOL_CALL_ALIAS: "tool_call"})
        assert canonical_stream_tool_name(agent, OLLAMA_TOOL_CALL_ALIAS) == "tool_call"
        assert canonical_stream_tool_name(agent, "web_search") == "web_search"
        assert canonical_stream_tool_name(agent, None) is None
        assert canonical_stream_tool_name(SimpleNamespace(), OLLAMA_TOOL_CALL_ALIAS) == OLLAMA_TOOL_CALL_ALIAS


def _agent(base_url, api_mode="chat_completions", provider="custom"):
    return SimpleNamespace(base_url=base_url, api_mode=api_mode, provider=provider, api_key="",
                           _base_url_hostname=base_url.split("//")[-1].split("/")[0].split(":")[0], log_prefix="")


class TestRouteScope:
    @pytest.fixture
    def probe(self, monkeypatch):
        calls, verdict = [], {"value": "ollama"}

        def fake_detect(base_url, api_key=""):
            calls.append(base_url)
            return verdict["value"]

        monkeypatch.setattr("agent.model_metadata.detect_local_server_type", fake_detect)
        return calls, verdict

    def test_detected_local_ollama_is_aliased_once_per_route(self, probe):
        calls, _ = probe
        agent = _agent("http://127.0.0.1:11434/v1")
        assert ollama_tool_call_alias_route(agent) and ollama_tool_call_alias_route(agent)
        assert len(calls) == 1
        agent.base_url = "http://127.0.0.1:11435/v1"
        ollama_tool_call_alias_route(agent)
        assert len(calls) == 2

    @pytest.mark.parametrize("server_type", ["lm-studio", "vllm", "llamacpp", None])
    def test_other_local_servers_on_the_ollama_port_are_not_aliased(self, probe, server_type):
        probe[1]["value"] = server_type
        assert not ollama_tool_call_alias_route(_agent("http://localhost:11434/v1"))

    @pytest.mark.parametrize("base_url", ["https://ollama.com/v1", "https://api.example.com/v1"])
    def test_cloud_and_public_endpoints_are_never_probed_or_aliased(self, probe, base_url):
        assert not ollama_tool_call_alias_route(_agent(base_url))
        assert probe[0] == []

    def test_non_chat_completions_mode_is_not_aliased(self, probe):
        assert not ollama_tool_call_alias_route(_agent("http://127.0.0.1:11434/v1", api_mode="codex_responses"))

    def test_unrelated_route_request_is_unchanged(self, probe):
        probe[1]["value"] = "vllm"
        agent, transport = _agent("http://127.0.0.1:8000/v1"), get_transport("chat_completions")
        tools, msgs = [_td("tool_call"), _td("web_search")], [{"role": "user", "content": "q"}]
        out_tools, out_msgs = apply_wire_tool_aliases(agent, transport, tools, msgs)
        assert out_tools is tools and out_msgs is msgs and transport._last_wire_aliases == {}

    def test_xai_keeps_its_own_alias_and_not_this_one(self, probe):
        agent, transport = _agent("https://api.x.ai/v1", provider="xai"), get_transport("chat_completions")
        out_tools, _ = apply_wire_tool_aliases(agent, transport, [_td("tool_search"), _td("tool_call")], [])
        assert [t["function"]["name"] for t in out_tools] == ["hermes_tool_search", "tool_call"]
        assert transport._last_wire_aliases == {"hermes_tool_search": "tool_search"}
        assert probe[0] == []


# --- multi-turn exchange against a local fake Ollama server ---------------------------------

class _FakeOllama(BaseHTTPRequestHandler):
    """Answers Ollama's /api/tags probe and streams scripted chat replies. A scripted call
    names what the model GENERATED; like the captured server, the reply is reattributed to
    ``tool_call`` whenever a tool of that name is advertised."""
    script: list = []
    chat_requests: list = []

    def do_GET(self):
        if self.path.rstrip("/").endswith("/api/tags"):
            self._json(200, {"models": [{"name": "qwen-test"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode() or "{}")
        if "messages" not in body:
            return self._json(404, {"error": "not found"})
        type(self).chat_requests.append(body)
        step = type(self).script.pop(0) if type(self).script else {"text": "done"}
        advertised = [t["function"]["name"] for t in body.get("tools") or []]
        chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
        if "text" in step:
            chunks.append({"choices": [{"index": 0, "delta": {"content": step["text"]}, "finish_reason": None}]})
            finish = "stop"
        else:
            generated = step["generated"]
            if generated == "<bridge>":
                generated = next(n for n in advertised if n == "tool_call" or n.startswith(OLLAMA_TOOL_CALL_ALIAS))
            returned = "tool_call" if "tool_call" in advertised else generated
            chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": f"call_{len(type(self).chat_requests)}", "type": "function",
                "function": {"name": returned, "arguments": step["arguments"]}}]}, "finish_reason": None}]})
            finish = "tool_calls"
        chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                       "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
        if not body.get("stream", False):
            tool_calls = [tc for chunk in chunks for choice in chunk["choices"]
                          for tc in choice["delta"].get("tool_calls", [])]
            message = {"role": "assistant", "content": step.get("text"),
                       "tool_calls": tool_calls or None}
            return self._json(200, {"id": "nonstream", "object": "chat.completion", "model": "qwen-test",
                                    "choices": [{"index": 0, "message": message, "finish_reason": finish}]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps({'id': 'm', **chunk})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def _json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a, **kw):
        pass


@pytest.fixture
def fake_ollama(tmp_path):
    from tools.registry import registry

    _FakeOllama.script, _FakeOllama.chat_requests = [], []
    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with open(os.path.join(os.environ["HERMES_HOME"], "config.yaml"), "w", encoding="utf-8") as fh:
        fh.write('tools:\n  tool_search:\n    enabled: "on"\n')
    echo_calls = []

    def echo(args, **kw):
        echo_calls.append(args)
        return json.dumps({"ok": True, "echo": args.get("text")})

    registry.register(name="mcp_wirealias_echo", handler=echo, toolset="mcp-wirealias",
                      schema=_td("mcp_wirealias_echo", "Echo text back",
                                 {"text": {"type": "string", "description": "Text"}}, ["text"])["function"])
    target = tmp_path / "note.txt"
    target.write_text("hello from the fixture", encoding="utf-8")
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_address[1]}/v1", handler=_FakeOllama,
                              echo_calls=echo_calls, target=target)
    finally:
        server.shutdown()
        registry.deregister("mcp_wirealias_echo")


def _make_agent(url):
    from run_agent import AIAgent

    return AIAgent(api_key="ollama", base_url=url, provider="custom", model="qwen-test", max_iterations=8,
                   enabled_toolsets=["mcp-wirealias", "file"], quiet_mode=True, skip_context_files=True,
                   skip_memory=True, save_trajectories=False, platform="cli")


def _tool_messages(result):
    return [m for m in result["messages"] if m.get("role") == "tool"]


class TestMultiTurnAgainstFakeOllama:
    def test_without_alias_the_eager_call_is_reattributed_and_rejected(self, fake_ollama, monkeypatch):
        monkeypatch.setattr("agent.chat_completion_wire_aliases.ollama_tool_call_alias_route", lambda agent: False)
        fake_ollama.handler.script = [
            {"generated": "read_file", "arguments": json.dumps({"path": str(fake_ollama.target)})}, {"text": "done"}]
        result = _make_agent(fake_ollama.url).run_conversation("read the note")
        first = fake_ollama.handler.chat_requests[0]
        assert "tool_call" in [t["function"]["name"] for t in first["tools"]]
        assert "requires 'calls'" in _tool_messages(result)[0]["content"]

    def test_describe_invoke_and_eager_call_survive_a_multi_turn_exchange(self, fake_ollama):
        fake_ollama.handler.script = [
            {"generated": "tool_describe", "arguments": json.dumps({"names": ["mcp_wirealias_echo"]})},
            {"generated": "<bridge>", "arguments": json.dumps(
                {"calls": [{"name": "mcp_wirealias_echo", "arguments": {"text": "ping"}}]})},
            {"generated": "read_file", "arguments": json.dumps({"path": str(fake_ollama.target)})},
            {"text": "done"},
        ]
        result = _make_agent(fake_ollama.url).run_conversation("use the echo tool, then read the note")
        reqs = fake_ollama.handler.chat_requests
        assert len(reqs) == 4
        wire_names = [t["function"]["name"] for t in reqs[0]["tools"]]
        assert OLLAMA_TOOL_CALL_ALIAS in wire_names and "tool_call" not in wire_names and "read_file" in wire_names
        # Session-stable tool surface: every request advertises byte-identical tools.
        assert all(json.dumps(r["tools"], sort_keys=True) == json.dumps(reqs[0]["tools"], sort_keys=True) for r in reqs)
        # The deferred tool ran exactly once through the alias, with its arguments intact.
        assert fake_ollama.echo_calls == [{"text": "ping"}]
        tool_results = _tool_messages(result)
        assert "mcp_wirealias_echo" in tool_results[0]["content"]
        assert "ping" in tool_results[1]["content"]
        assert "hello from the fixture" in tool_results[2]["content"]
        # Stored history keeps canonical names; the next request's wire history uses the alias.
        stored = [tc["function"]["name"] for m in result["messages"] if m.get("role") == "assistant"
                  for tc in m.get("tool_calls") or []]
        assert stored == ["tool_describe", "tool_call", "read_file"]
        wire_hist = [tc["function"]["name"] for m in reqs[3]["messages"] if m.get("role") == "assistant"
                     for tc in m.get("tool_calls") or []]
        assert wire_hist == ["tool_describe", OLLAMA_TOOL_CALL_ALIAS, "read_file"]
        assert result["final_response"] == "done"


class TestAdditionalCompatibilityAcceptance:
    def test_nonstream_multi_turn_preserves_arguments_ids_and_canonical_history(self, fake_ollama):
        args = {"calls": [{"name": "mcp_wirealias_echo", "arguments": {"text": " Unicode: café \u2603 "}}]}
        fake_ollama.handler.script = [
            {"generated": "<bridge>", "arguments": json.dumps(args, ensure_ascii=False)},
            {"text": "done"},
        ]
        agent = _make_agent(fake_ollama.url)
        agent._disable_streaming = True
        result = agent.run_conversation("echo the fixture text")
        reqs = fake_ollama.handler.chat_requests
        assert len(reqs) == 2 and all(not r.get("stream") for r in reqs)
        assert fake_ollama.echo_calls == [args["calls"][0]["arguments"]]
        assistant = next(m for m in result["messages"] if m.get("tool_calls"))
        call = assistant["tool_calls"][0]
        assert call["function"]["name"] == "tool_call"
        assert call["function"]["arguments"] == json.dumps(args, ensure_ascii=False)
        delivered = next(m for m in result["messages"] if m.get("role") == "tool")
        assert delivered["tool_call_id"] == call["id"]
        assert reqs[1]["tools"] == reqs[0]["tools"]
        assert next(m for m in reqs[1]["messages"] if m.get("tool_calls"))["tool_calls"][0]["function"]["name"] == OLLAMA_TOOL_CALL_ALIAS

    def test_denied_deferred_target_does_not_execute_through_wire_alias(self, fake_ollama, monkeypatch):
        from hermes_cli import plugins
        from tools import approval

        original = plugins._get_pre_tool_call_directive_details
        prompts = []

        def directive(tool_name, args, **kwargs):
            if tool_name == "mcp_wirealias_echo":
                return plugins._PreToolCallDirective(action="approve", message="fixture requires approval")
            return original(tool_name, args, **kwargs)

        def deny(tool_name, *args, **kwargs):
            prompts.append(tool_name)
            return {"approved": False, "message": "DENIED by acceptance fixture"}

        monkeypatch.setattr(plugins, "_get_pre_tool_call_directive_details", directive)
        monkeypatch.setattr(approval, "request_tool_approval", deny)
        fake_ollama.handler.script = [
            {"generated": "<bridge>", "arguments": json.dumps(
                {"calls": [{"name": "mcp_wirealias_echo", "arguments": {"text": "must not land"}}]})},
            {"text": "blocked"},
        ]
        result = _make_agent(fake_ollama.url).run_conversation("try the protected fixture")
        assert prompts == ["mcp_wirealias_echo"]
        assert fake_ollama.echo_calls == []
        assert "DENIED" in _tool_messages(result)[0]["content"]
        assert OLLAMA_TOOL_CALL_ALIAS in [t["function"]["name"] for t in fake_ollama.handler.chat_requests[0]["tools"]]
