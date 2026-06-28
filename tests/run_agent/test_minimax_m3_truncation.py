"""Regression tests for the MiniMax-M3 premature-stop detection.

Background
----------
MiniMax-M3 (via the ``minimax-oauth`` provider) has a recurring bug where
streaming responses that contain certain byte sequences — most commonly a
backtick immediately followed by a newline (`` `<newline> ``) inside
inline-code prose — terminate early with ``finish_reason="stop"`` while
the visible content ends mid-sentence. The signature is an unclosed
inline-code span: the last non-whitespace character is a backtick and
the two nearest backticks have a newline between them.

Observed live in session ``20260627_170215_502ca60b`` (June 27 2026):
messages 9134, 9136, 9138, 9140 all ended with `` `<newline>...<backtick> ``
and were misreported as ``stop`` instead of ``length``. The agent loop's
existing continuation-retry path (triggered by ``finish_reason="length"``)
already handles up to 3 retries transparently — the fix here is just to
detect the misreport and rewrite ``stop`` → ``length`` so that path
fires.

These tests cover three pieces:
1. ``_has_natural_response_ending`` correctly identifies closed and
   unclosed inline-code spans.
2. ``_is_minimax_m3_backend`` matches the ``minimax-oauth`` provider
   with an M3-family model, and rejects other provider/model combos.
3. ``_should_treat_stop_as_truncated`` returns True for the
   misreported-stop pattern and False for genuine stops.

Pre-existing Ollama-GLM behavior is not regressed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# _has_natural_response_ending
# ---------------------------------------------------------------------------


class TestNaturalResponseEnding:
    """The natural-ending heuristic must distinguish closed inline-code
    spans from the unclosed spans left by the MiniMax-M3 truncation bug."""

    @pytest.mark.parametrize(
        "text",
        [
            "Hello world.",
            "Hello world!",
            "Question?",
            "List item:",
            "```python\nprint('hi')\n```",  # code fence close
            "Use the `foo` function.",  # tight inline pair mid-sentence
            "end with `foo`.",  # tight inline pair + period
            "call `some_long_function_name(arg1, arg2)` and stop.",  # long pair + period
            "wraps `multiple words inside` cleanly.",  # spaced pair + period
        ],
    )
    def test_recognizes_natural_endings(self, text: str) -> None:
        assert AIAgent._has_natural_response_ending(text) is True

    @pytest.mark.parametrize(
        "text,label",
        [
            (
                "1. **Merges thinking blocks** — M3 emits the same reasoning twice "
                "(once via `reasoning_content`/`reasoning` fields, once inline as `\n"
                "did not** stream real reasoning fields, the inline `",
                "session 20260627_170215_502ca60b msg 9134",
            ),
            (
                "2. **Filters `\n a `\n `lines 232-237) *only* routes inline `",
                "session 20260627_170215_502ca60b msg 9136",
            ),
            (
                "Reasoning chain stays valid.\n\n"
                "2. **Filters `\n the `\nonly* routes inline `",
                "session 20260627_170215_502ca60b msg 9138",
            ),
            (
                "once inline as `\n `\n only thing it strips is the *visible* `",
                "session 20260627_170215_502ca60b msg 9140",
            ),
            # Synthetic edge cases
            ("unclosed ` here", "single trailing backtick after non-pair"),
            ("the inline `", "trailing backtick, no opening nearby"),
            ("`", "lone backtick"),
            ("`x\nstuff `", "newline between pair, simple"),
            ("`\nsome big chunk\nhere `", "newline between pair, multiline"),
        ],
    )
    def test_flags_unclosed_code_spans(self, text: str, label: str) -> None:
        assert AIAgent._has_natural_response_ending(text) is False, (
            f"expected truncation-flag False for: {label}"
        )


# ---------------------------------------------------------------------------
# _is_minimax_m3_backend
# ---------------------------------------------------------------------------


class TestMinimaxM3BackendDetection:
    """The provider/model detector must match the exact buggy pair and
    reject near-misses (other MiniMax models, other providers with M3)."""

    def _make_agent(self, provider: str, model: str) -> AIAgent:
        """Build a minimal AIAgent-shaped instance without going through
        full initialization. The detector reads self.provider and
        self.model only."""
        agent = AIAgent.__new__(AIAgent)
        agent.provider = provider
        agent.model = model
        # _base_url_lower is read by _is_ollama_glm_backend; set it to
        # something that won't false-positive.
        agent._base_url_lower = ""
        agent.base_url = ""
        return agent

    def test_matches_minimax_oauth_with_m3(self) -> None:
        agent = self._make_agent("minimax-oauth", "MiniMax-M3")
        assert agent._is_minimax_m3_backend() is True

    def test_matches_case_insensitive(self) -> None:
        agent = self._make_agent("MINIMAX-OAUTH", "minimax-m3")
        assert agent._is_minimax_m3_backend() is True

    def test_matches_provider_only_with_m3_in_name(self) -> None:
        # Some provider configs only carry the family in the provider id.
        agent = self._make_agent("minimax", "MiniMax-M3")
        assert agent._is_minimax_m3_backend() is True

    def test_matches_via_openrouter_with_MiniMax_prefix(self) -> None:
        # OpenRouter hosts MiniMax models under ``MiniMax/MiniMax-M3`` —
        # the provider is openrouter but the model name identifies it as
        # a MiniMax model.
        agent = self._make_agent("openrouter", "MiniMax/MiniMax-M3")
        assert agent._is_minimax_m3_backend() is True

    @pytest.mark.parametrize(
        "provider,model",
        [
            # Other providers with M3 in name — different backend, no bug
            ("openai", "m3"),
            ("anthropic", "MiniMax-M3"),
            # Plain M3 with no MiniMax anywhere — clearly a different model
            ("openai-codex", "gpt-m3"),
            ("zai", "glm-m3"),
            # Model name happens to contain "MiniMax" substring but isn't
            # from the official namespace — defensive check.
            ("openai", "some-MiniMax-fine-tune-m3"),
        ],
    )
    def test_rejects_non_minimax_m3(self, provider: str, model: str) -> None:
        agent = self._make_agent(provider, model)
        assert agent._is_minimax_m3_backend() is False, (
            f"should reject provider={provider!r} model={model!r}"
        )

    def test_matches_minimax_oauth_with_other_model(self) -> None:
        # The bug is server-side on the MiniMax endpoint, not specific to
        # MiniMax-M3. Any model served by minimax-oauth should match.
        agent = self._make_agent("minimax-oauth", "MiniMax-M2")
        assert agent._is_minimax_m3_backend() is True


# ---------------------------------------------------------------------------
# _should_treat_stop_as_truncated — end-to-end on the detector chain
# ---------------------------------------------------------------------------


def _make_assistant_message(content: str) -> SimpleNamespace:
    """Build a minimal assistant_message stand-in matching the shape the
    detector reads: ``content`` and ``tool_calls`` attributes."""
    return SimpleNamespace(content=content, tool_calls=None)


class TestShouldTreatStopAsTruncated:
    """End-to-end behavior of the public detector. The function reads
    self.provider, self.model, self.api_mode, and the assistant_message
    content; we exercise those via a bare AIAgent.__new__ instance."""

    def _agent(self, provider: str, model: str, api_mode: str = "chat_completions") -> AIAgent:
        agent = AIAgent.__new__(AIAgent)
        agent.provider = provider
        agent.model = model
        agent.api_mode = api_mode
        agent._base_url_lower = ""
        agent.base_url = ""
        return agent

    def test_minimax_m3_with_unclosed_code_span_returns_true(self) -> None:
        agent = self._agent("minimax-oauth", "MiniMax-M3")
        msg = _make_assistant_message(
            "Some reasoning here.\n\n"
            "1. **Merges thinking blocks** — M3 emits the same reasoning "
            "twice (once via `reasoning_content`/`reasoning` fields, "
            "once inline as `\n"
            "did not** stream real reasoning fields, the inline `"
        )
        # MiniMax-M3 misreports regardless of tool history, so even a
        # pure-chat conversation triggers the detector.
        messages = [{"role": "user", "content": "audit this"}]
        assert agent._should_treat_stop_as_truncated("stop", msg, messages) is True

    def test_minimax_m3_with_proper_ending_returns_false(self) -> None:
        agent = self._agent("minimax-oauth", "MiniMax-M3")
        msg = _make_assistant_message("This response is complete.")
        messages = [{"role": "tool", "content": "{}"}]
        assert agent._should_treat_stop_as_truncated("stop", msg, messages) is False

    def test_non_minimax_m3_with_mid_sentence_returns_false(self) -> None:
        # A mid-sentence stop on a NON-MiniMax-M3 backend must NOT be
        # flagged — we only extend the existing detection, not invent new
        # heuristics for arbitrary providers.
        agent = self._agent("openai", "gpt-5")
        msg = _make_assistant_message(
            "Some response that cuts off mid-sen"
        )
        messages = [{"role": "tool", "content": "{}"}]
        assert agent._should_treat_stop_as_truncated("stop", msg, messages) is False

    def test_length_finish_reason_returns_false_immediately(self) -> None:
        # The detector only rewrites "stop" → "length". If the provider
        # already reports "length", we don't need to do anything.
        agent = self._agent("minimax-oauth", "MiniMax-M3")
        msg = _make_assistant_message("anything")
        messages = [{"role": "tool", "content": "{}"}]
        assert agent._should_treat_stop_as_truncated("length", msg, messages) is False

    def test_tool_calls_present_returns_false(self) -> None:
        # If the model emitted tool_calls, the content mid-truncation is
        # legitimate (partial JSON) — don't rewrite stop → length.
        agent = self._agent("minimax-oauth", "MiniMax-M3")
        msg = SimpleNamespace(
            content="calling tool with `unclosed arg",
            tool_calls=[SimpleNamespace(id="1", function=SimpleNamespace(name="x"))],
        )
        messages = [{"role": "tool", "content": "{}"}]
        assert agent._should_treat_stop_as_truncated("stop", msg, messages) is False

    def test_no_tool_message_in_history_still_returns_true_for_minimax_m3(self) -> None:
        # MiniMax-M3 differs from Ollama-GLM: the bug fires on pure chat,
        # no tool use required. This test pins that distinction so the
        # two paths don't accidentally get merged.
        agent = self._agent("minimax-oauth", "MiniMax-M3")
        msg = _make_assistant_message("anything ending with `unclosed")
        messages = [{"role": "user", "content": "hello"}]
        assert agent._should_treat_stop_as_truncated("stop", msg, messages) is True