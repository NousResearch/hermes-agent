"""#86070 — dedicated large-context overflow model, not fallback_providers.

Context overflow stays compress-only at the classifier
(``should_compress=True``, ``should_fallback=False``). Recovery is a
separate opt-in: ``agent._large_context_model = {provider, model}`` plus
optional ``base_url`` / ``api_key``. On a real overflow,
``try_activate_overflow_model`` calls ``agent.switch_model`` so the
session sticks on the larger window.

Do NOT route overflow through ``fallback_providers`` /
``_try_activate_fallback``. Those remain the 429 / transport path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import run_agent
from agent.error_classifier import FailoverReason, classify_api_error
from agent.model_metadata import (
    is_output_cap_error,
    parse_available_output_tokens_from_error,
)
from run_agent import AIAgent


pytestmark = pytest.mark.usefixtures("_no_compression_sleep")


@pytest.fixture()
def _no_compression_sleep(monkeypatch):
    """Short-circuit the 2s pause between compression retries."""
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(run_agent, "jittered_backoff", lambda *a, **k: 0.0)


def _overflow_error(message="context length exceeded"):
    """Ollama / OpenAI-compat 400 for a genuine window overflow (#86070)."""
    err = Exception(message)
    err.status_code = 400
    return err


def _rate_limit_error():
    err = Exception("Error code: 429 - rate limit exceeded")
    err.status_code = 429
    err.body = {"error": {"message": "rate limit exceeded"}}
    err.response = SimpleNamespace(headers={})
    return err


# Anthropic phrasing that both is_output_cap_error and
# parse_available_output_tokens_from_error recognize.
_OUTPUT_CAP_MSG = (
    "max_tokens: 32768 > context_window: 200000 - "
    "input_tokens: 190000 = available_tokens: 10000"
)


def _output_cap_error():
    err = Exception(_OUTPUT_CAP_MSG)
    err.status_code = 400
    return err


def _mock_response(content="ok from fallback"):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    resp = SimpleNamespace(choices=[choice], model="minimax-m3")
    resp.usage = None
    return resp


def _prefill():
    return [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]


def _unchanged_compress(msgs, *_a, **_k):
    """Compressor that cannot shrink — forces compression_exhausted."""
    return list(msgs), "unchanged"


def _make_agent(*, fallback_model=None, context_length=32768, large_context_model=None):
    """Primary = small-window local model; optional larger-window fallback."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            provider="ollama",
            model="qwen2.5-coder:14b",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = True
        agent.save_trajectories = False
        agent._large_context_model = large_context_model
        if getattr(agent, "context_compressor", None) is not None:
            agent.context_compressor.context_length = context_length
            agent.context_compressor.threshold_tokens = int(context_length * 0.5)
        return agent


def _agent_from_config(cfg):
    """Construct AIAgent so ``agent_init`` loads ``context_overflow`` from *cfg*.

    Unlike ``_make_agent``, this does not overwrite ``_large_context_model``.
    """
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            provider="ollama",
            model="qwen2.5-coder:14b",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        return agent


LARGE_FALLBACK = [{"provider": "minimax", "model": "minimax-m3"}]
LARGE_CONTEXT = {
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "api_key": "test-gemini-key-1234567890",
}


# ── Config load (Task 2) ─────────────────────────────────────────────────


class TestLargeContextConfigLoad:
    """``context_overflow.large_context`` is loaded onto the agent at init."""

    def test_full_large_context_dict_is_loaded_onto_agent(self):
        agent = _agent_from_config({"context_overflow": {"large_context": LARGE_CONTEXT}})
        assert agent._large_context_model == {
            "provider": LARGE_CONTEXT["provider"],
            "model": LARGE_CONTEXT["model"],
            "base_url": LARGE_CONTEXT["base_url"],
            "api_key": LARGE_CONTEXT["api_key"],
        }
        assert agent._overflow_model_activated is False

    def test_omitted_context_overflow_leaves_large_context_none(self):
        agent = _agent_from_config({})
        assert agent._large_context_model is None
        assert agent._overflow_model_activated is False

    def test_empty_large_context_is_none(self):
        agent = _agent_from_config({"context_overflow": {"large_context": {}}})
        assert agent._large_context_model is None

    def test_missing_model_is_none(self):
        agent = _agent_from_config(
            {"context_overflow": {"large_context": {"provider": "gemini"}}}
        )
        assert agent._large_context_model is None

    def test_empty_and_whitespace_fields_are_none(self):
        agent = _agent_from_config(
            {
                "context_overflow": {
                    "large_context": {"provider": "  ", "model": ""},
                }
            }
        )
        assert agent._large_context_model is None

    def test_whitespace_is_stripped_on_loaded_fields(self):
        agent = _agent_from_config(
            {
                "context_overflow": {
                    "large_context": {
                        "provider": "  gemini  ",
                        "model": "  gemini-2.5-flash  ",
                        "base_url": "  https://example.invalid/v1  ",
                    }
                }
            }
        )
        assert agent._large_context_model == {
            "provider": "gemini",
            "model": "gemini-2.5-flash",
            "base_url": "https://example.invalid/v1",
        }
        assert "api_key" not in agent._large_context_model

    def test_non_dict_sections_are_treated_as_empty(self):
        agent = _agent_from_config({"context_overflow": "off"})
        assert agent._large_context_model is None
        agent = _agent_from_config(
            {"context_overflow": {"large_context": "gemini-2.5-flash"}}
        )
        assert agent._large_context_model is None


# ── Layer 1: classifier ──────────────────────────────────────────────────


class TestOverflowClassifiesCompressNotFallback:
    """Overflow stays compress-only; 429 stays a fallback candidate."""

    def test_ollama_context_length_exceeded_is_compress_only(self):
        result = classify_api_error(
            _overflow_error(),
            provider="ollama",
            model="qwen2.5-coder:14b",
        )
        assert result.reason == FailoverReason.context_overflow
        assert result.should_compress is True
        # Dedicated large-context recovery is a separate opt-in, not
        # should_fallback. Overflow must not enter fallback_providers.
        assert result.should_fallback is False

    def test_rate_limit_still_is_a_fallback_candidate(self):
        """Control: 429 must keep should_fallback so we didn't over-read the classifier."""
        result = classify_api_error(_rate_limit_error(), provider="ollama")
        assert result.reason == FailoverReason.rate_limit
        assert result.should_fallback is True
        assert result.should_compress is False


# ── Layer 2: conversation-loop eager-fallback gate ───────────────────────


class TestOverflowMissesEagerFallbackGate:
    """Mirror the gate in agent/conversation_loop.py (~3162).

    Overflow is neither rate-limited nor a transport failure, so the
    fallback chain is never consulted even when it is populated.
    """

    @staticmethod
    def _should_eager_fallback(classified, retry_count=0):
        is_rate_limited = classified.reason in {
            FailoverReason.rate_limit,
            FailoverReason.billing,
            FailoverReason.upstream_rate_limit,
        }
        is_transport_failure = classified.reason in {
            FailoverReason.timeout,
            FailoverReason.overloaded,
        }
        return is_rate_limited or (is_transport_failure and retry_count >= 2)

    def test_overflow_does_not_enter_eager_fallback_gate(self):
        classified = classify_api_error(_overflow_error())
        assert self._should_eager_fallback(classified) is False
        assert self._should_eager_fallback(classified, retry_count=5) is False

    def test_rate_limit_does_enter_eager_fallback_gate(self):
        classified = classify_api_error(_rate_limit_error())
        assert self._should_eager_fallback(classified) is True


# ── Layer 3: run_conversation — must pass today ──────────────────────────


class TestOverflowWithoutLargeContextModel:
    """No ``_large_context_model`` → compress until exhausted; never fallback_providers."""

    def test_overflow_without_large_context_model_never_activates_fallback(self):
        agent = _make_agent(fallback_model=LARGE_FALLBACK, large_context_model=None)
        assert agent._fallback_chain, "repro requires a configured fallback chain"
        assert agent._fallback_index < len(agent._fallback_chain)
        assert agent._large_context_model is None

        agent.client.chat.completions.create.side_effect = _overflow_error()

        with (
            patch.object(agent, "_try_activate_fallback", return_value=True) as mock_fallback,
            patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch,
            patch.object(agent, "_compress_context", side_effect=_unchanged_compress) as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=_prefill())

        assert mock_fallback.call_count == 0
        assert mock_switch.call_count == 0
        assert mock_compress.called, "overflow without a large-context model must compress"
        assert agent.model == "qwen2.5-coder:14b"
        assert result.get("failed") is True
        assert result.get("compression_exhausted") is True
        assert result.get("completed") is False
        err = (result.get("error") or result.get("final_response") or "")
        assert "cannot compress further" in err.lower() or "compression attempts" in err.lower()


class TestRateLimitIgnoresLargeContextModel:
    """429 still uses fallback_providers even when a large-context model is set."""

    def test_rate_limit_with_fallback_chain_and_large_context_model_uses_fallback(self):
        agent = _make_agent(
            fallback_model=LARGE_FALLBACK,
            large_context_model=LARGE_CONTEXT,
        )
        agent.client.chat.completions.create.side_effect = [
            _rate_limit_error(),
            _mock_response("ok after fallback"),
        ]

        with (
            patch.object(agent, "_try_activate_fallback", return_value=True) as mock_fallback,
            patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch,
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch(
                "agent.conversation_loop._sync_failover_system_message",
                side_effect=lambda _a, _m, prompt: prompt,
            ),
        ):
            result = agent.run_conversation("hello")

        assert mock_fallback.called, "429 must still activate fallback_providers"
        assert mock_switch.call_count == 0, (
            "overflow helper / switch_model is not the 429 recovery path"
        )
        assert mock_compress.call_count == 0
        assert result.get("compression_exhausted") is not True
        assert result.get("completed") is True
        assert result.get("final_response") == "ok after fallback"


# ── Layer 4: run_conversation large-context overflow switch ──────────────


class TestLargeContextOverflowContract:
    """run_conversation switches to the configured large-context model on overflow."""

    def test_overflow_switches_to_configured_large_context_model(self):
        """With ``_large_context_model`` set, overflow must switch_model and complete."""
        agent = _make_agent(large_context_model=LARGE_CONTEXT)
        large_client = MagicMock()
        large_client.chat.completions.create.return_value = _mock_response(
            "ok from large context"
        )
        # Primary keeps overflowing until switch_model rebuilds the client.
        agent.client.chat.completions.create.side_effect = _overflow_error()

        with (
            patch.object(agent, "_create_openai_client", return_value=large_client),
            patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch,
            patch.object(agent, "_try_activate_fallback", return_value=True) as mock_fallback,
            patch.object(agent, "_compress_context", side_effect=_unchanged_compress) as mock_compress,
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(large_client, None),
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=1_048_576),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=_prefill())

        mock_switch.assert_called()
        switch_kwargs = mock_switch.call_args.kwargs
        switch_args = mock_switch.call_args.args

        def _switch_arg(name, index):
            if name in switch_kwargs:
                return switch_kwargs[name]
            if len(switch_args) > index:
                return switch_args[index]
            return None

        assert _switch_arg("new_model", 0) == LARGE_CONTEXT["model"]
        assert _switch_arg("new_provider", 1) == LARGE_CONTEXT["provider"]
        assert _switch_arg("api_key", 2) == LARGE_CONTEXT["api_key"]
        assert _switch_arg("base_url", 3) == LARGE_CONTEXT["base_url"]
        assert mock_fallback.call_count == 0
        assert mock_compress.call_count == 0, (
            "overflow with _large_context_model must escalate before compression"
        )
        assert result.get("compression_exhausted") is not True
        assert result.get("failed") is not True
        assert result.get("completed") is True
        assert agent.model == "gemini-2.5-flash"
        assert agent.provider == "gemini"
        assert result.get("final_response") == "ok from large context"

    def test_overflow_skips_switch_when_large_window_too_small(self):
        """Large-model window < request tokens → compress, do not switch."""
        agent = _make_agent(large_context_model=LARGE_CONTEXT)
        agent.client.chat.completions.create.side_effect = _overflow_error()

        def _tiny_window(model="", *a, **k):
            if "gemini" in str(model).lower():
                return 8
            return 32768

        bulky = (
            "A long prior turn that is clearly larger than an 8-token window. "
            * 40
        )
        prefill = [
            {"role": "user", "content": bulky},
            {"role": "assistant", "content": bulky},
        ]

        with (
            patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch,
            patch.object(agent, "_try_activate_fallback", return_value=True) as mock_fallback,
            patch.object(agent, "_compress_context", side_effect=_unchanged_compress) as mock_compress,
            patch("agent.model_metadata.get_model_context_length", side_effect=_tiny_window),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(MagicMock(), None),
            ),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=prefill)

        assert mock_switch.call_count == 0
        assert mock_fallback.call_count == 0
        assert mock_compress.called
        assert result.get("compression_exhausted") is True
        assert agent.model == "qwen2.5-coder:14b"

    def test_already_on_large_context_model_does_not_switch_again(self):
        """Already on the overflow provider/model → compress, no second switch."""
        agent = _make_agent(large_context_model=LARGE_CONTEXT)
        agent.provider = "gemini"
        agent.model = "gemini-2.5-flash"
        agent.client.chat.completions.create.side_effect = _overflow_error()

        with (
            patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch,
            patch.object(agent, "_try_activate_fallback", return_value=True) as mock_fallback,
            patch.object(agent, "_compress_context", side_effect=_unchanged_compress) as mock_compress,
            patch("agent.model_metadata.get_model_context_length", return_value=1_048_576),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(MagicMock(), None),
            ),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=_prefill())

        assert mock_switch.call_count == 0
        assert mock_fallback.call_count == 0
        assert mock_compress.called
        assert result.get("compression_exhausted") is True
        assert agent.model == "gemini-2.5-flash"
        assert agent.provider == "gemini"

    def test_output_cap_400_does_not_switch_to_large_context_model(self):
        """Output-cap 400 is not a context overflow — do not call switch_model."""
        assert is_output_cap_error(_OUTPUT_CAP_MSG) is True
        assert parse_available_output_tokens_from_error(_OUTPUT_CAP_MSG) == 10000

        agent = _make_agent(large_context_model=LARGE_CONTEXT)
        agent.max_tokens = 32768
        # parse_available_output_tokens_from_error → 10000; loop retries at 10000-64.
        reduced_cap = 10000 - 64

        def _create_side_effect(*_a, **kwargs):
            # _ephemeral_max_output_tokens is consumed while building kwargs,
            # before create() runs — detect the reduced cap on the retry.
            max_tokens = kwargs.get("max_tokens")
            if max_tokens is None:
                max_tokens = kwargs.get("max_completion_tokens")
            if max_tokens == reduced_cap:
                return _mock_response("ok after output cap retry")
            raise _output_cap_error()

        agent.client.chat.completions.create.side_effect = _create_side_effect

        with (
            patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch,
            patch.object(agent, "_try_activate_fallback", return_value=True) as mock_fallback,
            patch.object(agent, "_compress_context", side_effect=_unchanged_compress),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(MagicMock(), None),
            ),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert mock_switch.call_count == 0
        assert mock_fallback.call_count == 0
        assert result.get("compression_exhausted") is not True
        assert result.get("completed") is True
        assert result.get("final_response") == "ok after output cap retry"
        assert agent.model == "qwen2.5-coder:14b"

    def test_overflow_switches_when_compression_disabled(self):
        """compression.enabled false + large_context set: switch, do not terminal-error.

        The disabled-compaction path currently returns compaction_disabled.
        With a configured large-context model, escalate first. If the helper
        returns False, the existing terminal error stays.
        """
        agent = _make_agent(large_context_model=LARGE_CONTEXT)
        agent.compression_enabled = False
        large_client = MagicMock()
        large_client.chat.completions.create.return_value = _mock_response(
            "ok from large context"
        )
        agent.client.chat.completions.create.side_effect = _overflow_error()

        with (
            patch.object(agent, "_create_openai_client", return_value=large_client),
            patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch,
            patch.object(agent, "_try_activate_fallback", return_value=True) as mock_fallback,
            patch.object(agent, "_compress_context", side_effect=_unchanged_compress) as mock_compress,
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(large_client, None),
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=1_048_576),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=_prefill())

        mock_switch.assert_called()
        assert mock_fallback.call_count == 0
        assert mock_compress.call_count == 0
        assert result.get("compaction_disabled") is not True
        assert result.get("compression_exhausted") is not True
        assert result.get("failed") is not True
        assert result.get("completed") is True
        assert agent.model == "gemini-2.5-flash"
        assert agent.provider == "gemini"
        assert result.get("final_response") == "ok from large context"


class TestOverflowSwitchIsSessionSticky:
    """switch_model rewrites ``_primary_runtime``; restore must keep the large model.

    This can pass today by calling ``switch_model`` directly — the overflow
    loop is not required for the stickiness invariant.
    """

    def test_restore_primary_runtime_keeps_large_model_after_overflow_switch(self):
        agent = _make_agent(large_context_model=LARGE_CONTEXT)
        assert agent._fallback_activated is False
        original_model = agent.model
        original_provider = agent.provider

        with (
            patch.object(agent, "_create_openai_client", return_value=MagicMock()),
            patch("agent.model_metadata.get_model_context_length", return_value=1_048_576),
            patch("agent.credential_pool.load_pool", return_value=None),
            patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
        ):
            agent.switch_model(
                new_model=LARGE_CONTEXT["model"],
                new_provider=LARGE_CONTEXT["provider"],
                api_key=LARGE_CONTEXT["api_key"],
                base_url=LARGE_CONTEXT["base_url"],
            )

        assert agent.model == "gemini-2.5-flash"
        assert agent.provider == "gemini"
        assert agent._fallback_activated is False
        assert agent._primary_runtime["model"] == "gemini-2.5-flash"
        assert agent._primary_runtime["provider"] == "gemini"
        assert agent._primary_runtime["model"] != original_model
        assert agent._primary_runtime["provider"] != original_provider

        restored = agent._restore_primary_runtime()

        assert restored is False
        assert agent._fallback_activated is False
        assert agent.model == "gemini-2.5-flash"
        assert agent.provider == "gemini"
        assert agent._primary_runtime["model"] == "gemini-2.5-flash"
        assert agent._primary_runtime["provider"] == "gemini"


# ── Helper-direct (try_activate_overflow_model) ──────────────────────────


class TestTryActivateOverflowModel:
    """Unit tests that call ``try_activate_overflow_model`` directly."""

    @staticmethod
    def _switch_arg(mock_switch, name, index):
        switch_kwargs = mock_switch.call_args.kwargs
        switch_args = mock_switch.call_args.args
        if name in switch_kwargs:
            return switch_kwargs[name]
        if len(switch_args) > index:
            return switch_args[index]
        return None

    def test_missing_large_context_model_returns_false(self):
        from agent.chat_completion_helpers import try_activate_overflow_model

        agent = _make_agent(large_context_model=None)
        with patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch:
            assert try_activate_overflow_model(agent, 100) is False
        mock_switch.assert_not_called()

    def test_incomplete_dict_returns_false(self):
        from agent.chat_completion_helpers import try_activate_overflow_model

        agent = _make_agent(large_context_model={"provider": "gemini"})
        with patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch:
            assert try_activate_overflow_model(agent, 100) is False
        mock_switch.assert_not_called()

    def test_already_on_gemini_flash_returns_false(self):
        from agent.chat_completion_helpers import try_activate_overflow_model

        agent = _make_agent(large_context_model=LARGE_CONTEXT)
        agent.provider = "gemini"
        agent.model = "gemini-2.5-flash"
        with (
            patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch,
            patch("agent.model_metadata.get_model_context_length", return_value=1_048_576),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(MagicMock(), None),
            ),
        ):
            assert try_activate_overflow_model(agent, 100) is False
        mock_switch.assert_not_called()
        assert agent._overflow_model_activated is False

    def test_overflow_already_activated_returns_false(self):
        from agent.chat_completion_helpers import try_activate_overflow_model

        agent = _make_agent(large_context_model=LARGE_CONTEXT)
        agent._overflow_model_activated = True
        with (
            patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch,
            patch("agent.model_metadata.get_model_context_length", return_value=1_048_576),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(MagicMock(), None),
            ),
        ):
            assert try_activate_overflow_model(agent, 100) is False
        mock_switch.assert_not_called()

    def test_window_smaller_than_request_returns_false(self):
        from agent.chat_completion_helpers import try_activate_overflow_model

        agent = _make_agent(large_context_model=LARGE_CONTEXT)
        with (
            patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch,
            patch("agent.model_metadata.get_model_context_length", return_value=8),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(MagicMock(), None),
            ),
        ):
            assert try_activate_overflow_model(agent, 10000) is False
        mock_switch.assert_not_called()
        assert agent._overflow_model_activated is False

    def test_unknown_window_allows_switch(self):
        from agent.chat_completion_helpers import try_activate_overflow_model

        agent = _make_agent(large_context_model=LARGE_CONTEXT)
        with (
            patch.object(agent, "switch_model") as mock_switch,
            patch("agent.model_metadata.get_model_context_length", return_value=None),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(MagicMock(), None),
            ),
        ):
            assert try_activate_overflow_model(agent, 10**9) is True
        mock_switch.assert_called()
        assert agent._overflow_model_activated is True
        assert agent._fallback_activated is False

    def test_resolve_none_returns_false(self):
        from agent.chat_completion_helpers import try_activate_overflow_model

        agent = _make_agent(large_context_model=LARGE_CONTEXT)
        with (
            patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch,
            patch("agent.model_metadata.get_model_context_length", return_value=1_048_576),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(None, None),
            ),
        ):
            assert try_activate_overflow_model(agent, 100) is False
        mock_switch.assert_not_called()
        assert agent._overflow_model_activated is False

    def test_success_switches_and_rewrites_primary_runtime(self):
        from agent.chat_completion_helpers import try_activate_overflow_model

        agent = _make_agent(large_context_model=LARGE_CONTEXT)
        large_client = MagicMock()
        with (
            patch.object(agent, "_create_openai_client", return_value=large_client),
            patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch,
            patch("agent.model_metadata.get_model_context_length", return_value=1_048_576),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(large_client, None),
            ),
            patch("agent.credential_pool.load_pool", return_value=None),
            patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
        ):
            assert try_activate_overflow_model(agent, 100) is True

        mock_switch.assert_called()
        assert self._switch_arg(mock_switch, "new_model", 0) == LARGE_CONTEXT["model"]
        assert self._switch_arg(mock_switch, "new_provider", 1) == LARGE_CONTEXT["provider"]
        assert self._switch_arg(mock_switch, "api_key", 2) == LARGE_CONTEXT["api_key"]
        assert self._switch_arg(mock_switch, "base_url", 3) == LARGE_CONTEXT["base_url"]
        assert agent._overflow_model_activated is True
        assert agent._fallback_activated is False
        assert agent.model == "gemini-2.5-flash"
        assert agent.provider == "gemini"
        assert agent._primary_runtime["model"] == "gemini-2.5-flash"
        assert agent._primary_runtime["provider"] == "gemini"

    def test_success_uses_resolved_client_creds_when_yaml_omits_them(self):
        """Documented yaml is {provider, model}; creds come from the resolver."""
        from agent.chat_completion_helpers import try_activate_overflow_model

        agent = _make_agent(
            large_context_model={"provider": "gemini", "model": "gemini-2.5-flash"}
        )
        resolved_key = "resolved-gemini-key"
        resolved_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        large_client = MagicMock()
        large_client.api_key = resolved_key
        large_client.base_url = resolved_url

        with (
            patch.object(agent, "_create_openai_client", return_value=large_client),
            patch.object(agent, "switch_model", wraps=agent.switch_model) as mock_switch,
            patch("agent.model_metadata.get_model_context_length", return_value=1_048_576),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(large_client, None),
            ),
            patch("agent.credential_pool.load_pool", return_value=None),
            patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
        ):
            assert try_activate_overflow_model(agent, 100) is True

        mock_switch.assert_called()
        assert self._switch_arg(mock_switch, "new_model", 0) == "gemini-2.5-flash"
        assert self._switch_arg(mock_switch, "new_provider", 1) == "gemini"
        assert self._switch_arg(mock_switch, "api_key", 2) == resolved_key
        assert self._switch_arg(mock_switch, "base_url", 3) == resolved_url
        assert agent._overflow_model_activated is True
        assert agent._fallback_activated is False
        assert agent.model == "gemini-2.5-flash"
        assert agent.provider == "gemini"

    def test_switch_model_exception_does_not_activate(self):
        from agent.chat_completion_helpers import try_activate_overflow_model

        agent = _make_agent(large_context_model=LARGE_CONTEXT)
        with (
            patch.object(agent, "switch_model", side_effect=RuntimeError("swap failed")),
            patch("agent.model_metadata.get_model_context_length", return_value=1_048_576),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(MagicMock(), None),
            ),
        ):
            assert try_activate_overflow_model(agent, 100) is False
        assert agent._overflow_model_activated is False
        assert agent._fallback_activated is False
        assert agent.model == "qwen2.5-coder:14b"
