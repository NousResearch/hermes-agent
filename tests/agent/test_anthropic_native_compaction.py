"""Anthropic on-demand compaction as the compression summary writer (compression.anthropic_native)."""

import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import anthropic
import httpx
import pytest

from agent import anthropic_native_compaction as anc
from agent.anthropic_native_compaction import (
    COMPACTION_BETA, COMPACTION_EFFORT, COMPACTION_MAX_TOKENS, INSTRUCTIONS_MAX_CHARS, AnthropicNativeSummary,
    CompactionSeed, NativeSummary, build_compaction_kwargs, compaction_summary_text, compaction_usage,
    model_supports_compaction, native_compaction_eligible, native_summary_for, record_compaction_seed,
)
from agent.context_compressor import ContextCompressor


def _agent(**overrides):
    attrs = dict(
        anthropic_native_compaction=True, compression_enabled=True, api_mode="anthropic_messages",
        base_url="https://api.anthropic.com", _use_prompt_caching=True, _cache_ttl="5m",
        model="claude-opus-5-5", session_id="s1", _anthropic_client=None,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _history(n=6):
    rows = [{"role": "user", "content": "start"}]
    for i in range(1, n):
        rows.append({"role": "assistant" if i % 2 else "user", "content": f"turn {i}"})
    return rows


def _seed_kwargs():
    return {
        "model": "claude-opus-5-5", "system": [{"type": "text", "text": "sys"}], "tools": [{"name": "t"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}],
        "max_tokens": 128000, "stream": True, "stop_sequences": ["X"], "thinking": {"type": "adaptive"},
        "output_config": {"effort": "xhigh", "format": {"type": "json_schema"}},
        "extra_headers": {"anthropic-beta": "interleaved-thinking-2025-05-14,oauth-2025-04-20"},
    }


def _api_error(status, message):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError(message, response=httpx.Response(status, request=request), body=None)


class _RawResponse:
    def __init__(self, data):
        self.http_response = SimpleNamespace(json=lambda: data)


class _FakeClient:
    def __init__(self, data=None, exc=None):
        self.calls = []
        self._data, self._exc = data, exc
        self.messages = SimpleNamespace(with_raw_response=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return _RawResponse(self._data)


def _ok_response(text="## Goal\nNative summary."):
    return {
        "model": "claude-opus-5-5", "stop_reason": "compaction",
        "content": [{"type": "compaction", "content": text, "signature": "sig"}],
        "usage": {"input_tokens": 0, "output_tokens": 0, "iterations": [
            {"type": "compaction", "input_tokens": 62, "cache_read_input_tokens": 38003,
             "cache_creation_input_tokens": 0, "output_tokens": 285}]},
    }


class TestEligibility:
    @pytest.mark.parametrize("model,expected", [
        ("claude-opus-5-5", True), ("claude-opus-4-6", True), ("claude-opus-4.8", True), ("claude-opus-4-5", False),
        ("claude-opus-4-20250514", False), ("claude-sonnet-4-6", True), ("claude-sonnet-4-5", False),
        ("claude-haiku-5-5", True), ("claude-haiku-4-5", False), ("claude-fable-5-1", True),
        ("claude-mythos-preview", True), ("claude-opus-5-5[1m]", True), ("gpt-6.1-sol", False), (None, False),
    ])
    def test_model_matrix(self, model, expected):
        assert model_supports_compaction(model) is expected

    def test_eligible_agent(self):
        assert native_compaction_eligible(_agent()) is True

    @pytest.mark.parametrize("override", [
        {"anthropic_native_compaction": False}, {"compression_enabled": False}, {"api_mode": "chat_completions"},
        {"base_url": "https://api.minimax.io/anthropic"}, {"_use_prompt_caching": False}, {"_cache_ttl": None},
        {"model": "claude-opus-4-5"}, {"_anthropic_native_compaction_rejected": True},
    ])
    def test_each_gate_blocks(self, override):
        assert native_compaction_eligible(_agent(**override)) is False


class TestSeedCoverage:
    def test_seed_recorded_only_when_eligible(self):
        agent = _agent(anthropic_native_compaction=False)
        record_compaction_seed(agent, _seed_kwargs(), _history())
        assert getattr(agent, "_anthropic_compaction_seed", None) is None
        agent = _agent()
        record_compaction_seed(agent, _seed_kwargs(), _history())
        assert isinstance(agent._anthropic_compaction_seed, CompactionSeed)

    def test_covered_prefix_counts_rows_of_the_request(self):
        agent, history = _agent(), _history(6)
        record_compaction_seed(agent, _seed_kwargs(), history)
        later = history + [{"role": "assistant", "content": "reply"}, {"role": "tool", "content": "r", "tool_call_id": "c"}]
        assert AnthropicNativeSummary(agent).covered_prefix(later) == 6

    def test_reloaded_equal_history_still_matches(self):
        agent, history = _agent(), _history(6)
        record_compaction_seed(agent, _seed_kwargs(), history)
        reloaded = json.loads(json.dumps(history))  # gateway turns reload rows from the DB
        assert AnthropicNativeSummary(agent).covered_prefix(reloaded) == 6

    @pytest.mark.parametrize("mutate", [
        lambda h: h[2].update(content="edited"), lambda h: h.pop(), lambda h: h.insert(1, {"role": "user", "content": "x"}),
    ])
    def test_changed_history_is_refused(self, mutate):
        agent, history = _agent(), _history(6)
        record_compaction_seed(agent, _seed_kwargs(), history)
        changed = json.loads(json.dumps(history))
        mutate(changed)
        assert AnthropicNativeSummary(agent).covered_prefix(changed) is None

    @pytest.mark.parametrize("override", [{"session_id": "s2"}, {"model": "claude-sonnet-5-5"}])
    def test_other_session_or_model_is_refused(self, override):
        agent, history = _agent(), _history(6)
        record_compaction_seed(agent, _seed_kwargs(), history)
        for key, value in override.items():
            setattr(agent, key, value)
        assert AnthropicNativeSummary(agent).covered_prefix(history) is None

    @pytest.mark.parametrize("ttl,age,ok", [("5m", 230, True), ("5m", 250, False), ("1h", 3500, True), ("1h", 3560, False)])
    def test_cache_lifetime(self, ttl, age, ok):
        agent, history = _agent(_cache_ttl=ttl), _history(6)
        record_compaction_seed(agent, _seed_kwargs(), history)
        agent._anthropic_compaction_seed.captured_at = time.monotonic() - age
        assert (AnthropicNativeSummary(agent).covered_prefix(history) == 6) is ok

    def test_factory_needs_a_seed(self):
        agent = _agent()
        assert native_summary_for(agent) is None
        record_compaction_seed(agent, _seed_kwargs(), _history())
        assert isinstance(native_summary_for(agent), AnthropicNativeSummary)


class TestRequestShape:
    def test_prefix_kept_and_rejected_fields_removed(self):
        seed = _seed_kwargs()
        kwargs = build_compaction_kwargs(seed, "INSTR")
        # Same objects up to the last message: the cache this request wrote is read back.
        assert kwargs["system"] is seed["system"] and kwargs["tools"] is seed["tools"]
        assert kwargs["messages"] is seed["messages"] and kwargs["thinking"] is seed["thinking"]
        assert "stream" not in kwargs and "stop_sequences" not in kwargs
        assert kwargs["output_config"] == {"effort": COMPACTION_EFFORT}
        assert kwargs["max_tokens"] == COMPACTION_MAX_TOKENS
        assert kwargs["extra_body"]["compaction"] == {"type": "summarize", "instructions": "INSTR"}
        assert kwargs["extra_headers"]["anthropic-beta"].split(",") == [
            "interleaved-thinking-2025-05-14", "oauth-2025-04-20", COMPACTION_BETA]
        assert seed["output_config"]["effort"] == "xhigh" and "stream" in seed  # the seed is never mutated

    def test_client_betas_used_when_request_has_none(self):
        seed = {k: v for k, v in _seed_kwargs().items() if k != "extra_headers"}
        client = SimpleNamespace(_custom_headers={"anthropic-beta": "a,b"})
        assert build_compaction_kwargs(seed, "I", client=client)["extra_headers"]["anthropic-beta"] == f"a,b,{COMPACTION_BETA}"

    @pytest.mark.parametrize("tool_choice,kept", [({"type": "auto"}, True), ({"type": "any"}, False), ({"type": "tool", "name": "t"}, False)])
    def test_forced_tool_choice_dropped(self, tool_choice, kept):
        kwargs = build_compaction_kwargs({**_seed_kwargs(), "tool_choice": tool_choice}, "I")
        assert ("tool_choice" in kwargs) is kept

    def test_context_management_and_task_budget_remaining_dropped(self):
        seed = {**_seed_kwargs(), "context_management": {"edits": []},
                "output_config": {"task_budget": {"total": 10, "remaining": 5}}}
        kwargs = build_compaction_kwargs(seed, "I")
        assert "context_management" not in kwargs
        assert kwargs["output_config"] == {"task_budget": {"total": 10}}

    def test_budget_thinking_gets_room(self):
        kwargs = build_compaction_kwargs({**_seed_kwargs(), "thinking": {"type": "enabled", "budget_tokens": 32000}}, "I")
        assert kwargs["max_tokens"] > 32000


class TestResponse:
    def test_summary_and_iteration_usage(self):
        data = _ok_response()
        assert compaction_summary_text(data) == "## Goal\nNative summary."
        assert compaction_usage(data) == {"input_tokens": 62, "output_tokens": 285, "cache_read_input_tokens": 38003,
                                          "cache_creation_input_tokens": 0}

    @pytest.mark.parametrize("stop", ["max_tokens", "refusal", "tool_use", "end_turn", "model_context_window_exceeded"])
    def test_no_summary_without_compaction_stop(self, stop):
        assert compaction_summary_text({**_ok_response(), "stop_reason": stop}) is None

    def test_blank_block_is_no_summary(self):
        assert compaction_summary_text(_ok_response(text="  ")) is None


class TestSummarize:
    def _ready(self, client):
        agent = _agent(_anthropic_client=client)
        record_compaction_seed(agent, _seed_kwargs(), _history())
        return agent

    def test_success_bills_usage_and_consumes_seed(self):
        client = _FakeClient(_ok_response())
        agent = self._ready(client)
        with patch("agent.aux_accounting.record_aux_usage") as billed:
            result = AnthropicNativeSummary(agent).summarize("INSTR")
        assert isinstance(result, NativeSummary) and result.text == "## Goal\nNative summary."
        assert result.usage["cache_read_input_tokens"] == 38003
        assert client.calls[0]["extra_body"]["compaction"]["instructions"] == "INSTR"
        assert client.calls[0]["timeout"] > 0
        assert agent._anthropic_compaction_seed is None
        response, task = billed.call_args.args
        assert task == "compression" and response.usage["output_tokens"] == 285
        assert billed.call_args.kwargs["provider"] == "anthropic"

    def test_no_block_falls_back(self):
        agent = self._ready(_FakeClient({**_ok_response(), "stop_reason": "max_tokens", "content": []}))
        with patch("agent.aux_accounting.record_aux_usage"):
            assert AnthropicNativeSummary(agent).summarize("INSTR") is None
        assert not getattr(agent, "_anthropic_native_compaction_rejected", False)

    def test_structural_rejection_disables_for_session(self):
        exc = _api_error(400, "Error code: 400 - compaction parameter requires anthropic-beta: compact-2026-09-04")
        agent = self._ready(_FakeClient(exc=exc))
        assert AnthropicNativeSummary(agent).summarize("INSTR") is None
        assert agent._anthropic_native_compaction_rejected is True
        assert native_compaction_eligible(agent) is False

    def test_transient_error_falls_back_without_disabling(self):
        agent = self._ready(_FakeClient(exc=_api_error(529, "overloaded_error: compaction_unavailable")))
        assert AnthropicNativeSummary(agent).summarize("INSTR") is None
        assert not getattr(agent, "_anthropic_native_compaction_rejected", False)

    def test_oversized_instructions_never_sent(self):
        client = _FakeClient(_ok_response())
        agent = self._ready(client)
        assert AnthropicNativeSummary(agent).summarize("x" * (INSTRUCTIONS_MAX_CHARS + 1)) is None
        assert client.calls == []


# --- compressor integration -------------------------------------------------------------------------------------

def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=8000):
        return ContextCompressor(model="claude-opus-5-5", threshold_percent=0.75, protect_first_n=2, protect_last_n=4,
                                 quiet_mode=True)


def _conversation():
    rows = [{"role": "system", "content": "You are helpful."}]
    for i in range(14):
        rows.append({"role": "user", "content": f"Question {i}: " + "detail " * 60})
        rows.append({"role": "assistant", "content": f"Answer {i}: " + "fact " * 60})
    return rows


class _FakeNative:
    def __init__(self, covered, result):
        self.covered, self.result, self.instructions = covered, result, None

    def covered_prefix(self, messages):
        return self.covered(len(messages))

    def summarize(self, instructions):
        self.instructions = instructions
        return self.result


def _aux_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])


def _joined(messages):
    return "\n".join(str(m.get("content")) for m in messages)


class TestCompressorIntegration:
    def test_native_summary_replaces_aux_call(self):
        comp, messages = _compressor(), _conversation()
        before = json.loads(json.dumps(messages))
        native = _FakeNative(lambda n: n - 1, NativeSummary("## Goal\nNATIVE-CHECKPOINT", "claude-opus-5-5", 1200,
                                                            {"input_tokens": 10, "cache_read_input_tokens": 9000}))
        with patch("agent.context_compressor.call_llm") as aux:
            result = comp.compress(messages, current_tokens=7000, force=True, native_summary=native)
        aux.assert_not_called()
        assert "NATIVE-CHECKPOINT" in _joined(result)
        assert len(result) < len(messages)
        assert "Do not call tools" in native.instructions and "## Goal" in native.instructions
        assert len(native.instructions) <= INSTRUCTIONS_MAX_CHARS
        assert messages == before
        assert result[-1]["role"] == before[-1]["role"] and result[-1]["content"] == before[-1]["content"]
        telemetry = comp._last_compression_telemetry
        assert isinstance(telemetry, dict)
        assert telemetry["aux_provider"] == "anthropic" and telemetry["aux_model"] == "claude-opus-5-5"
        assert telemetry["aux_prompt_tokens"] == 9010 and telemetry["method"] == "llm_summary"
        assert telemetry["messages_before"] == len(messages) and telemetry["messages_after"] == len(result)

    def test_native_failure_keeps_aux_summarizer(self):
        comp, messages = _compressor(), _conversation()
        native = _FakeNative(lambda n: n, None)
        with patch("agent.context_compressor.call_llm", return_value=_aux_response("## Goal\nAUX-CHECKPOINT")) as aux:
            result = comp.compress(messages, current_tokens=7000, force=True, native_summary=native)
        assert native.instructions is not None and aux.call_count == 1
        assert "AUX-CHECKPOINT" in _joined(result)

    def test_unexpected_native_error_keeps_aux_summarizer(self):
        comp, messages = _compressor(), _conversation()
        native = _FakeNative(lambda n: n, None)
        native.summarize = lambda instructions: (_ for _ in ()).throw(TypeError("bad client double"))
        with patch("agent.context_compressor.call_llm", return_value=_aux_response("## Goal\nAUX-CHECKPOINT")) as aux:
            result = comp.compress(messages, current_tokens=7000, force=True, native_summary=native)
        assert aux.call_count == 1 and "AUX-CHECKPOINT" in _joined(result)

    def test_rows_outside_the_kept_tail_block_native(self):
        comp, messages = _compressor(), _conversation()
        # The captured request stopped 10 rows ago: more rows than the kept tail, so they would be lost.
        native = _FakeNative(lambda n: n - 10, NativeSummary("NATIVE", "m", 1, {}))
        with patch("agent.context_compressor.call_llm", return_value=_aux_response("## Goal\nAUX-CHECKPOINT")) as aux:
            result = comp.compress(messages, current_tokens=7000, force=True, native_summary=native)
        assert native.instructions is None and aux.call_count == 1
        assert "AUX-CHECKPOINT" in _joined(result)

    def test_focus_topic_reaches_native_instructions(self):
        comp, messages = _compressor(), _conversation()
        native = _FakeNative(lambda n: n, NativeSummary("## Goal\nN", "m", 1, {}))
        with patch("agent.context_compressor.call_llm"):
            comp.compress(messages, current_tokens=7000, force=True, focus_topic="billing module", native_summary=native)
        assert 'FOCUS TOPIC: "billing module"' in native.instructions

    def test_auto_derived_focus_never_reaches_native_instructions(self):
        # The auto focus quotes recent user rows; conversation text in compaction.instructions made the API refuse.
        comp, messages = _compressor(), _conversation()
        assert comp._derive_auto_focus_topic(messages)  # the auxiliary path would get one
        native = _FakeNative(lambda n: n, NativeSummary("## Goal\nN", "m", 1, {}))
        with patch("agent.context_compressor.call_llm"):
            comp.compress(messages, current_tokens=7000, force=True, native_summary=native)
        assert "FOCUS TOPIC" not in native.instructions and "Question 13" not in native.instructions

    def test_memory_section_dropped_before_exceeding_limit(self):
        comp = _compressor()
        without = comp._build_native_summary_instructions(2000, None, "", True)
        with_memory = comp._build_native_summary_instructions(2000, None, "provider fact " * 50, True)
        assert len(with_memory) > len(without)  # the memory section rides along while it fits
        with patch("agent.anthropic_native_compaction.INSTRUCTIONS_MAX_CHARS", len(without) + 5):
            assert comp._build_native_summary_instructions(2000, None, "provider fact " * 50, True) == without


# --- agent wiring -----------------------------------------------------------------------------------------------

def test_resolve_compress_call_passes_native_only_to_engines_that_accept_it():
    from agent.conversation_compression_call import _resolve_compress_call

    class _Engine:
        def compress(self, messages, current_tokens=None, focus_topic=None, force=False, native_summary=None):
            return messages

    class _LegacyEngine:
        def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
            return messages

    for engine, expected in ((_Engine(), True), (_LegacyEngine(), False)):
        agent = _agent(context_compressor=engine)
        record_compaction_seed(agent, _seed_kwargs(), _history())
        _, kwargs = _resolve_compress_call(agent, approx_tokens=1, focus_topic=None, force=False, memory_context="",
                                           bypass_cooldown=False)
        assert ("native_summary" in kwargs) is expected


def test_config_reaches_agent_and_real_request_is_a_valid_seed(tmp_path, monkeypatch):
    from run_agent import AIAgent

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("compression:\n  anthropic_native: true\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    agent = AIAgent(api_key="sk-ant-api03-test", base_url="https://api.anthropic.com", api_mode="anthropic_messages",
                    model="claude-opus-5-5", provider="anthropic", quiet_mode=True, skip_context_files=True,
                    skip_memory=True, enabled_toolsets=[])
    assert agent.anthropic_native_compaction is True
    assert native_compaction_eligible(agent) is True
    history = [{"role": "user", "content": "hello"}]
    kwargs = agent._build_api_kwargs(list(history))
    record_compaction_seed(agent, kwargs, history)
    compaction = build_compaction_kwargs(agent._anthropic_compaction_seed.kwargs, "I", client=agent._anthropic_client)
    assert compaction["messages"] is kwargs["messages"] and compaction.get("system") is kwargs.get("system")
    assert COMPACTION_BETA in compaction["extra_headers"]["anthropic-beta"]


def test_default_config_keeps_feature_off():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["compression"]["anthropic_native"] is False
    assert anc.native_summary_for(_agent(anthropic_native_compaction=False)) is None


def test_live_opt_in_is_enabled_and_unset_returns_to_off():
    from tui_gateway.session_compression import _apply_live_compression_config

    agent = _agent(context_compressor=_compressor(), anthropic_native_compaction=False, provider="anthropic")
    _apply_live_compression_config(agent, {"compression": {"anthropic_native": True}})
    assert agent.anthropic_native_compaction is True
    _apply_live_compression_config(agent, {"compression": {}})
    assert agent.anthropic_native_compaction is False
