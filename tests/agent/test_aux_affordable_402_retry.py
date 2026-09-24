"""Credit-limited 402 handling: clamp to the affordable budget and retry.

Regression tests for the masoria stall (Aug 2026): compression fell back to
OpenRouter, which defaulted the omitted output cap to the model's full 65,536
window and rejected with ``402 ... can only afford 7117`` — three times in a
row — on an account whose balance easily covered a summary.

Two coordinated fixes:

1. ``_build_call_kwargs`` preserves an explicit ``max_tokens`` for OpenRouter
   routes (salvage of PR #41055 by @liuhao1024, issue #41035).
2. ``_create_with_progress`` retries ONCE with the provider-stated affordable
   budget when a 402 names one (pattern proven in closed PR #49785 for the
   main loop; this is the auxiliary-path equivalent).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.auxiliary_client import (
    _affordable_max_tokens_from_error,
    _build_call_kwargs,
    _create_with_progress,
)


class _Payment402(Exception):
    status_code = 402


_OPENROUTER_402 = (
    "Error code: 402 - {'error': {'message': 'This request requires more "
    "credits, or fewer max_tokens. You requested up to 65536 tokens, but can "
    "only afford 7117. To increase, visit https://openrouter.ai/settings/"
    "credits and add more credits', 'code': 402}}"
)


class TestAffordableExtraction:
    def test_extracts_budget_minus_margin(self):
        assert _affordable_max_tokens_from_error(_Payment402(_OPENROUTER_402)) == 7117 - 64

    def test_plain_exhaustion_returns_none(self):
        err = _Payment402("Error code: 402 - insufficient funds")
        assert _affordable_max_tokens_from_error(err) is None

    def test_tiny_budget_treated_as_exhaustion(self):
        err = _Payment402("You requested up to 65536 tokens, but can only afford 100.")
        assert _affordable_max_tokens_from_error(err) is None

    def test_non_payment_error_returns_none(self):
        assert _affordable_max_tokens_from_error(TimeoutError(
            "Codex auxiliary Responses stream stalled: no new output for 60.0s"
        )) is None

    def test_comma_grouped_count(self):
        err = _Payment402("can only afford 12,345 tokens")
        assert _affordable_max_tokens_from_error(err) == 12345 - 64


class _FlakyClient:
    """402s with the affordable message until max_tokens fits the budget."""

    def __init__(self, affordable=7117):
        self.calls = []
        self._affordable = affordable
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        cap = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        if cap is None or cap > self._affordable:
            raise _Payment402(_OPENROUTER_402)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                index=0,
                message=SimpleNamespace(role="assistant", content="summary"),
                finish_reason="stop",
            )],
            model=kwargs.get("model"), usage=None,
        )


class TestCreateWithProgressAffordableRetry:
    def test_uncapped_402_retries_once_with_affordable_cap(self):
        client = _FlakyClient()
        response = _create_with_progress(
            client,
            {"model": "google/gemini-3.6-flash",
             "messages": [{"role": "user", "content": "summarize"}]},
            "compression",
        )
        assert response.choices[0].message.content == "summary"
        assert len(client.calls) == 2
        assert client.calls[1]["max_tokens"] == 7117 - 64

    def test_already_affordable_cap_does_not_spin(self):
        """A 402 on a request already within budget re-raises immediately."""
        client = _FlakyClient(affordable=100)  # everything 402s
        with pytest.raises(_Payment402):
            _create_with_progress(
                client,
                {"model": "m", "max_tokens": 5000,
                 "messages": [{"role": "user", "content": "x"}]},
                "compression",
            )
        # 5000 > 7053 is false → within stated budget → single attempt.
        assert len(client.calls) == 1

    def test_plain_exhaustion_402_is_not_retried(self):
        class _Broke:
            def __init__(self):
                self.calls = []
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                self.calls.append(kwargs)
                raise _Payment402("Error code: 402 - insufficient funds")

        client = _Broke()
        with pytest.raises(_Payment402):
            _create_with_progress(
                client,
                {"model": "m", "messages": [{"role": "user", "content": "x"}]},
                "compression",
            )
        assert len(client.calls) == 1

    def test_retry_402_surfaces_without_spinning(self):
        """If the clamped retry 402s again, it propagates (single retry)."""

        class _Always402:
            def __init__(self):
                self.calls = []
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                self.calls.append(kwargs)
                raise _Payment402(_OPENROUTER_402)

        client = _Always402()
        with pytest.raises(_Payment402):
            _create_with_progress(
                client,
                {"model": "m", "messages": [{"role": "user", "content": "x"}]},
                "compression",
            )
        assert len(client.calls) == 2


class TestOpenRouterMaxTokensPreserved:
    """Salvage of PR #41055 (@liuhao1024): OpenRouter keeps an explicit cap."""

    def test_openrouter_provider_includes_max_tokens(self):
        kwargs = _build_call_kwargs(
            provider="openrouter",
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=2000,
        )
        assert kwargs.get("max_tokens") == 2000 or kwargs.get("max_completion_tokens") == 2000

    def test_openrouter_base_url_includes_max_tokens(self):
        kwargs = _build_call_kwargs(
            provider="openai",
            model="google/gemini-3.6-flash",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=2000,
            base_url="https://openrouter.ai/api/v1",
        )
        assert kwargs.get("max_tokens") == 2000 or kwargs.get("max_completion_tokens") == 2000

    def test_generic_provider_still_omits_max_tokens(self):
        kwargs = _build_call_kwargs(
            provider="openai",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=2000,
        )
        assert "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs

    def test_none_max_tokens_never_included_for_openrouter(self):
        kwargs = _build_call_kwargs(
            provider="openrouter",
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=None,
        )
        assert "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs
