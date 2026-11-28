"""The max-iterations summary call treats ``reasoning_details`` per api_mode: the
anthropic_messages converter rebuilds signed thinking blocks from it, so the summary messages
must keep it; a strict chat-completions route drops it on the wire via the same kwargs
builder the main loop uses (hermes-agent#70233)."""

import copy
import json
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from agent import prompt_cache_boundary
from agent.anthropic_message_convert import convert_tools_to_anthropic
from agent.anthropic_thinking_replay import remember_rejected_thinking
from agent.chat_completion_helpers import _build_api_kwargs_for_mode, _iteration_summary_api_messages
from agent.prompt_cache_boundary import register_stable_prefix
from run_agent import AIAgent

_HISTORY = [
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
     "reasoning_details": [{"type": "thinking", "thinking": "x", "signature": "SIG"}]},
    {"role": "tool", "tool_call_id": "t1", "content": "r"},
]


@pytest.fixture
def make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _make(base_url, provider, model="m"):
        agent = AIAgent(api_key="k", base_url=base_url, provider=provider, model=model, quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
        agent._cached_system_prompt = "SYS"
        return agent
    return _make


def test_anthropic_summary_messages_keep_reasoning_details(make_agent):
    agent = make_agent("https://api.anthropic.com", "anthropic")
    assert agent.api_mode == "anthropic_messages"
    out = _iteration_summary_api_messages(agent, [dict(m) for m in _HISTORY])
    assistant = next(m for m in out if m.get("role") == "assistant")
    assert assistant["reasoning_details"] == _HISTORY[1]["reasoning_details"]


def test_strict_chat_route_summary_wire_drops_reasoning_details(make_agent):
    agent = make_agent("https://api.groq.com/openai/v1", "custom")
    assert agent.api_mode == "chat_completions"
    api_messages = _iteration_summary_api_messages(agent, [dict(m) for m in _HISTORY])
    kwargs = _build_api_kwargs_for_mode(agent, api_messages)
    assert all("reasoning_details" not in m for m in kwargs["messages"])


class TestSummaryPrefixParity:
    """The summary request must close with the same normalization the main send
    path applies (assemble_api_request): canonical tool-call argument JSON and
    stripped string content — otherwise the summary's prefix diverges from every
    prior request and prefix-caching providers re-read the whole conversation
    (hermes-agent#123002)."""

    def test_canonicalizes_tool_call_arguments_and_strips_content(self, make_agent):
        agent = make_agent("https://api.groq.com/openai/v1", "custom")
        history = [
            {"role": "user", "content": "  q  "},
            {"role": "assistant", "content": "  thinking out loud  ", "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": '{"b": 1, "a": 2}'}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "  r  "},
        ]
        snapshot = copy.deepcopy(history)
        out = _iteration_summary_api_messages(agent, history)

        assistant = next(m for m in out if m.get("role") == "assistant")
        args = assistant["tool_calls"][0]["function"]["arguments"]
        assert args == json.dumps({"a": 2, "b": 1}, separators=(",", ":"), sort_keys=True)

        # Same whitespace contract as the main path: string content stripped on the
        # wire copy, history untouched.
        by_role = {m["role"]: m for m in out if m.get("role") in {"user", "tool"}}
        assert by_role["user"]["content"] == "q"
        assert by_role["tool"]["content"] == "r"
        assert history == snapshot  # send-path rewrites never reach the transcript

    def test_strips_lone_surrogates_like_the_send_path(self, make_agent):
        agent = make_agent("https://api.groq.com/openai/v1", "custom")
        history = [
            {"role": "user", "content": "clip \ud800 paste"},
            {"role": "assistant", "content": "ok \ud83d", "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": '{"k": "v\ud800"}'}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "r"},
            {"role": "user", "content": [{"type": "text", "text": "part \ud800"}]},
        ]
        snapshot = copy.deepcopy(history)
        out = _iteration_summary_api_messages(agent, history)
        # Main path's third pass rewrites lone surrogates to U+FFFD; anything else diverges the
        # prefix or makes the SDK's ensure_ascii=False utf-8 encode raise.
        assert next(m for m in out if m["role"] == "user")["content"] == "clip \ufffd paste"
        assistant = next(m for m in out if m["role"] == "assistant")
        assert assistant["content"] == "ok \ufffd"
        # Canonicalization runs first, so argument surrogates are already ASCII \udXXX escapes.
        assert assistant["tool_calls"][0]["function"]["arguments"] == '{"k":"v\\ud800"}'
        assert out[-1]["content"][0]["text"] == "part \ufffd"
        json.dumps(out, ensure_ascii=False).encode("utf-8")
        # The sanitizer is in-place and list-content parts are shared nested dicts:
        # history must keep its stored bytes.
        assert history == snapshot


def _texts(content):
    """Block texts, or the string itself: block boundaries are compared, cache markers are not."""
    return [content] if isinstance(content, str) else [block["text"] for block in content]


class TestAnthropicSummaryThinkingBinding:
    """Models that preserve thinking replay the tool loop's signed blocks in the summary request,
    and Anthropic checks each one against the system text, the tool definitions and every earlier
    message the loop sent. A difference there 400s ("Invalid `signature` in `thinking` block") and
    the turn ends without a summary. So the summary keeps the loop's tools and its durable
    rejected-thinking suppression, and it also keeps the loop's tool_choice and prompt-cache parts
    so that it reads the loop's cached prefix."""

    _STATIC = "You are Hermes.\n\nStable instructions."
    _VOLATILE = "\n\nSession details."
    _SCAFFOLD = "[Skill: triage]\n\nInspect the report carefully.\n\n"
    _TAIL = "ticket=42"
    _TOOL = {"type": "function", "function": {  # lone surrogate: the kept tools must pass the outbound scrub
        "name": "f", "description": "lone surrogate \ud83d here", "parameters": {"type": "object", "properties": {}}}}

    @pytest.fixture
    def anthropic_agent(self, make_agent, monkeypatch):
        agent = make_agent("https://api.anthropic.com", "anthropic", model="claude-opus-5-5")
        assert agent._use_prompt_caching and agent._use_native_cache_layout
        agent.tools = [copy.deepcopy(self._TOOL)]
        agent._cached_system_prompt = self._STATIC + self._VOLATILE
        agent._cached_system_prompt_static = self._STATIC
        monkeypatch.setattr(prompt_cache_boundary, "_prefixes", OrderedDict())  # an isolated scaffold registry
        register_stable_prefix(self._SCAFFOLD)  # as the skill and cron builders do
        return agent

    def _summary_request(self, agent, monkeypatch):
        sent = []

        def execute_current(request, callback, **kwargs):
            sent.append(request)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="Summary")], stop_reason="end_turn")

        monkeypatch.setattr("agent.relay_llm.execute_current", execute_current)
        history = [{"role": "user", "content": self._SCAFFOLD + self._TAIL}, *copy.deepcopy(_HISTORY[1:])]
        assert agent._handle_max_iterations(history, 60) == "Summary"
        return sent[-1]

    def test_summary_request_keeps_the_loops_prefix(self, anthropic_agent, monkeypatch):
        kw = self._summary_request(anthropic_agent, monkeypatch)
        # The loop's tools, scrubbed at the loop's outbound chokepoint (#50959 class), and the loop's tool_choice.
        expected_tools = convert_tools_to_anthropic(anthropic_agent.tools)
        expected_tools[0]["description"] = "lone surrogate \ufffd here"
        assert [{k: v for k, v in t.items() if k != "cache_control"} for t in kw.get("tools", [])] == expected_tools
        assert kw.get("tool_choice") == {"type": "auto"}
        assert _texts(kw["system"]) == [self._STATIC, self._VOLATILE]
        assert _texts(kw["messages"][0]["content"]) == [self._SCAFFOLD, self._TAIL]
        # Stripping thinking is not the fix: the API asks for the in-flight tool loop's thinking back.
        first = next(m for m in kw["messages"] if m["role"] == "assistant")["content"][0]
        assert (first["type"], first.get("signature")) == ("thinking", "SIG")

    def test_summary_skips_thinking_already_rejected(self, anthropic_agent, monkeypatch):
        # A signature 400 earlier in the session recorded the block, so the main loop no longer sends it.
        remember_rejected_thinking(anthropic_agent, copy.deepcopy(_HISTORY))
        kw = self._summary_request(anthropic_agent, monkeypatch)
        blocks = [b for m in kw["messages"] if isinstance(m["content"], list) for b in m["content"]]
        assert "SIG" not in [b.get("signature") for b in blocks]
        assert {"tool_use", "tool_result"} <= {b["type"] for b in blocks}
