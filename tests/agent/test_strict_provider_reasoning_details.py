"""Strict chat-completions hosts reject ``reasoning_details`` with 422 extra_forbidden.

Regression for #130757: Mistral (``mistral-small-latest``) strictly validates message
schemas and rejects ``messages[i].*.reasoning_details`` with::

    HTTP 422: {"detail": [{"type": "extra_forbidden",
      "loc": ["body", "messages", 2, "assistant", "reasoning_details"],
      "msg": "Extra inputs are not permitted"}]}

The chat-completions transport drops the field on the wire for every route that does
not replay it (OpenRouter / Nous Portal do); this pins that contract for Mistral and
other strict hosts, plus the one-shot retry net that strips a leaked field after a
422, without touching durable history.
"""

import copy
from types import SimpleNamespace

import pytest

from agent.transports.chat_completions import ChatCompletionsTransport
from agent.turn_retry_state import TurnRetryState

_MISTRAL = "https://api.mistral.ai/v1"
_GROQ = "https://api.groq.com/openai/v1"
_CEREBRAS = "https://api.cerebras.ai/v1"
_OPENROUTER = "https://openrouter.ai/api/v1"
_NOUS = "https://inference-api.nousresearch.com/v1"

_RD = [{"type": "thinking", "thinking": "x", "signature": "SIG"}]


def _history():
    return [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "reasoning_details": list(_RD)},
        {"role": "user", "content": "follow"},
    ]


def _mistral_422(text="reasoning_details"):
    err = SimpleNamespace()
    err.body = {
        "detail": [{
            "type": "extra_forbidden",
            "loc": ["body", "messages", 2, "assistant", text],
            "msg": "Extra inputs are not permitted",
        }]
    }
    return err


class TestStrictHostsStripReasoningDetails:
    @pytest.mark.parametrize("base_url", [_MISTRAL, _GROQ, _CEREBRAS, "https://gw.example.com/v1"])
    def test_wire_drops_field_history_untouched(self, base_url):
        transport = ChatCompletionsTransport()
        history = _history()
        snapshot = copy.deepcopy(history)
        kwargs = transport.build_kwargs("mistral-small-latest", history, base_url=base_url)
        assert all("reasoning_details" not in m for m in kwargs["messages"])
        assert history == snapshot

    def test_convert_messages_returns_original_when_clean(self):
        transport = ChatCompletionsTransport()
        clean = [{"role": "user", "content": "hi"}]
        assert transport.convert_messages(clean, base_url=_MISTRAL) is clean

    def test_trajectory_reasoning_never_reaches_wire(self):
        transport = ChatCompletionsTransport()
        history = [{"role": "assistant", "content": "x", "reasoning": "internal CoT"}]
        kwargs = transport.build_kwargs("m", history, base_url=_MISTRAL)
        assert all("reasoning" not in m for m in kwargs["messages"])


class TestReplayRoutesKeepReasoningDetails:
    @pytest.mark.parametrize("base_url", [_OPENROUTER, _NOUS])
    def test_wire_keeps_field(self, base_url):
        transport = ChatCompletionsTransport()
        kwargs = transport.build_kwargs("m", _history(), base_url=base_url)
        assert kwargs["messages"][1]["reasoning_details"] == _RD


@pytest.fixture
def make_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def _make(base_url=_MISTRAL, provider="mistral", model="mistral-small-latest"):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="k", base_url=base_url, provider=provider, model=model,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
        agent._cached_system_prompt = "SYS"
        return agent
    return _make


class TestMistralEndToEnd:
    def test_build_kwargs_strips_but_history_keeps(self, make_agent):
        agent = make_agent()
        assert agent.api_mode == "chat_completions"
        history = _history()
        snapshot = copy.deepcopy(history)
        from agent.turn_context import build_api_messages
        agent._current_turn_timestamp = 0
        agent.ephemeral_system_prompt = ""
        api_messages, _ = build_api_messages(
            agent, history, current_turn_user_idx=2, ext_prefetch_cache=None,
            plugin_user_context=None, moa_config=None, active_system_prompt="SYS",
        )
        kwargs = agent._build_api_kwargs(api_messages)
        assert all("reasoning_details" not in m for m in kwargs["messages"] if isinstance(m, dict))
        assert history == snapshot

    def test_tool_call_turn_strips(self, make_agent):
        agent = make_agent()
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
             "reasoning_details": list(_RD)},
            {"role": "tool", "tool_call_id": "t1", "content": "r"},
        ]
        snapshot = copy.deepcopy(history)
        kwargs = agent._build_api_kwargs(history)
        assert all("reasoning_details" not in m for m in kwargs["messages"] if isinstance(m, dict))
        assert history == snapshot


class TestStrictRecoveryNet:
    @staticmethod
    def _mistral_error():
        err = RuntimeError(
            'HTTP 422: {"detail": [{"type": "extra_forbidden", '
            '"loc": ["body", "messages", 2, "assistant", "reasoning_details"], '
            '"msg": "Extra inputs are not permitted"}]}'
        )
        err.body = _mistral_422().body
        err.status_code = 422
        return err

    def test_strips_and_retries_once(self):
        from types import SimpleNamespace as NS
        from agent.error_classifier import FailoverReason
        from agent.turn_recovery import _recover_format_errors
        agent = NS(
            log_prefix="", api_mode="chat_completions",
            codex_responses_native_compaction=False,
            _vprint=lambda *a, **k: None, _buffer_vprint=lambda *a, **k: None,
        )
        messages = _history()
        canonical = copy.deepcopy(messages)
        api_messages = copy.deepcopy(messages)
        retry = TurnRetryState()
        classified = NS(reason=FailoverReason.format_error)
        assert _recover_format_errors(agent, self._mistral_error(), classified, retry, messages, api_messages) is True
        assert retry.reasoning_details_retry_attempted is True
        assert all("reasoning_details" not in m for m in api_messages)
        assert messages == canonical
        # One-shot: second attempt does not loop.
        assert _recover_format_errors(agent, self._mistral_error(), classified, retry, messages, api_messages) is False

    def test_unrelated_format_error_does_not_strip(self):
        from types import SimpleNamespace as NS
        from agent.error_classifier import FailoverReason
        from agent.turn_recovery import _recover_format_errors
        agent = NS(
            log_prefix="", api_mode="chat_completions",
            codex_responses_native_compaction=False,
            _vprint=lambda *a, **k: None, _buffer_vprint=lambda *a, **k: None,
        )
        api_messages = _history()
        retry = TurnRetryState()
        classified = NS(reason=FailoverReason.format_error)
        assert _recover_format_errors(
            agent, RuntimeError("invalid_request_error: unknown parameter foo"),
            classified, retry, [], api_messages,
        ) is False
        assert retry.reasoning_details_retry_attempted is False
        assert "reasoning_details" in api_messages[1]

    def test_matcher_requires_both_signals(self):
        from agent.turn_recovery import _is_reasoning_details_forbidden
        assert _is_reasoning_details_forbidden(self._mistral_error()) is True
        assert _is_reasoning_details_forbidden(RuntimeError("reasoning_details present")) is False
        assert _is_reasoning_details_forbidden(RuntimeError("extra_forbidden elsewhere")) is False
