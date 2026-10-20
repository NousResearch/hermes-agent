"""Tests for compression.reuse_main_system_prompt (summary prefix-cache reuse).

When opted in and the summary routes to the main model, the aux summary call must carry
the session's frozen system prompt as its system message (so prefix-caching providers
serve the system block from KV cache), keep the summary instruction as the final user
message, and include a no-tool-call guard in the prompt. Default off = byte-identical
request shape to before (single user message).
"""

from __future__ import annotations

# Pre-import at collection time: the lazy agent.conversation_loop import inside
# _generate_summary runs process_bootstrap's payload selection, which reads the install's
# own manifest.json beside the checkout — legitimate on a sealed-payload install but
# flagged by the home I/O guard when it happens mid-test (same pre-existing failure as
# test_compression_interrupt_protection.py). Importing here moves the read to collection.
import agent.conversation_loop
from unittest.mock import patch

import agent.auxiliary_client as aux


def _mk_compressor(**kwargs):
    from agent.context_compressor import ContextCompressor
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(model="main-model", quiet_mode=True, **kwargs)


class _Resp:
    class _Choice:
        class _Msg:
            content = "[CONTEXT SUMMARY]: ok"
        message = _Msg()
    choices = [_Choice()]


def _capture_kwargs(compressor, messages=None):
    seen = {}

    def fake_call_llm(**kwargs):
        seen.update(kwargs)
        return _Resp()

    msgs = messages or [
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "done"},
    ]
    with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
        summary = compressor._generate_summary(msgs)
    assert summary is not None
    return seen


class TestSummarySystemPromptReuse:

    def test_default_off_request_unchanged(self):
        c = _mk_compressor()
        c._summary_system_prompt_source = "BIG FROZEN SYSTEM PROMPT"
        kwargs = _capture_kwargs(c)
        assert kwargs["messages"] == [{"role": "user", "content": kwargs["messages"][0]["content"]}]
        assert kwargs["messages"][0]["role"] == "user"

    def test_on_prepends_system_message_instruction_last(self):
        c = _mk_compressor(reuse_main_system_prompt=True)
        c._summary_system_prompt_source = "BIG FROZEN SYSTEM PROMPT"
        kwargs = _capture_kwargs(c)
        msgs = kwargs["messages"]
        assert msgs[0] == {"role": "system", "content": "BIG FROZEN SYSTEM PROMPT"}
        assert msgs[-1]["role"] == "user"
        assert "BIG FROZEN" not in msgs[-1]["content"]

    def test_on_but_no_prompt_available_stays_user_only(self):
        c = _mk_compressor(reuse_main_system_prompt=True)
        kwargs = _capture_kwargs(c)
        assert all(m["role"] == "user" for m in kwargs["messages"])

    def test_explicit_aux_model_disables_reuse(self):
        c = _mk_compressor(reuse_main_system_prompt=True)
        c._summary_system_prompt_source = "BIG FROZEN SYSTEM PROMPT"
        with patch.object(aux, "_get_auxiliary_task_config", return_value={"model": "small-4b"}):
            kwargs = _capture_kwargs(c)
        assert all(m["role"] == "user" for m in kwargs["messages"])

    def test_auto_aux_model_keeps_reuse(self):
        c = _mk_compressor(reuse_main_system_prompt=True)
        c._summary_system_prompt_source = "BIG FROZEN SYSTEM PROMPT"
        with patch.object(aux, "_get_auxiliary_task_config", return_value={"model": "auto"}):
            kwargs = _capture_kwargs(c)
        assert kwargs["messages"][0]["role"] == "system"

    def test_no_tool_guard_present_only_when_reuse_active(self):
        c_on = _mk_compressor(reuse_main_system_prompt=True)
        c_off = _mk_compressor()
        guard = "NO tools available"
        assert guard in _capture_kwargs(c_on)["messages"][-1]["content"]
        assert guard not in _capture_kwargs(c_off)["messages"][-1]["content"]

    def test_session_reset_clears_published_prompt(self):
        c = _mk_compressor(reuse_main_system_prompt=True)
        c._summary_system_prompt_source = "BIG FROZEN SYSTEM PROMPT"
        c.on_session_reset()
        assert c._main_system_prompt_for_summary() == ""


class TestCompressContextPublishesPromptSource:
    """compress_context() must publish the agent's frozen system prompt onto the compressor."""

    def test_cached_prompt_published(self):
        from agent.conversation_compression import compress_context

        class _Agent:
            model = "main-model"
            api_mode = "chat_completions"
            context_compressor = _mk_compressor(reuse_main_system_prompt=True)
            _cached_system_prompt = "FROZEN"

        agent = _Agent()
        # Abort as early as possible — we only assert the publish side effect.
        with patch("agent.conversation_compression._begin_compression_attempt") as begin:
            begin.return_value = None
            try:
                compress_context(agent, [], "FALLBACK SP")
            except Exception:
                pass
        assert agent.context_compressor._summary_system_prompt_source == "FROZEN"

    def test_falls_back_to_system_message_when_uncached(self):
        from agent.conversation_compression import compress_context

        class _Agent:
            model = "main-model"
            api_mode = "chat_completions"
            context_compressor = _mk_compressor(reuse_main_system_prompt=True)

        agent = _Agent()
        with patch("agent.conversation_compression._begin_compression_attempt") as begin:
            begin.return_value = None
            try:
                compress_context(agent, [], "LIVE SP")
            except Exception:
                pass
        assert agent.context_compressor._summary_system_prompt_source == "LIVE SP"
