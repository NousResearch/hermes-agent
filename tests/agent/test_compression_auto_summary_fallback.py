"""#116472: an ``auto``-resolved compression summary model that fails (e.g. a proxy channel
answering HTTP 200 with empty content) must fall back to the main model. ``auto`` resolves a
model per call without setting ``summary_model``, so the retry gate previously saw "no separate
model" and re-hit the same bad route forever; the resolved model is also recorded so the
user-visible warning names it instead of only appearing in errors.log.
"""

import time
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


def _msgs():
    return [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": "ok"},
    ]


def _compressor(**kwargs):
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(quiet_mode=True, **kwargs)


def test_fallback_arms_a_recovery_window():
    """The aux-route skip is time-bounded, not a permanent latch (#123362)."""
    c = _compressor(model="main-model")
    assert getattr(c, "_summary_aux_route_retry_at", 0.0) == 0.0

    c._fallback_to_main_for_compression(Exception("quota"), "failed", failed_model="bad/route")

    assert c._summary_aux_route_retry_at > time.monotonic()


def test_fallen_back_pins_the_main_route():
    """The main retry must pin the main model so a pinned aux route cannot be re-selected."""
    c = _compressor(
        model="main-model", provider="anthropic",
        base_url="https://api.anthropic.com", api_key="main-key",
    )
    c._summary_model_fallen_back = True
    c._summary_aux_route_retry_at = time.monotonic() + 600
    ok = MagicMock()
    ok.choices = [MagicMock()]
    ok.choices[0].message.content = "summary"

    with patch("agent.context_compressor.call_llm", return_value=ok) as mock_call:
        assert c._call_summary_llm("prompt", time.monotonic()) == "summary"

    kwargs = mock_call.call_args.kwargs
    assert kwargs["model"] == "main-model"
    assert kwargs["provider"] == "anthropic"
    assert kwargs["api_key"] == "main-key"


def test_recovery_window_clears_the_latch_before_the_next_attempt():
    """After the window elapses the attempt targets the aux route again."""
    c = _compressor(model="main-model")
    c._summary_model_fallen_back = True
    c._summary_aux_route_retry_at = time.monotonic() - 1
    seen = {}

    def _fake_call(prompt, started):
        seen["fallen_back"] = c._summary_model_fallen_back
        raise RuntimeError("still failing")

    c._call_summary_llm = _fake_call
    c._on_summary_failure = lambda *a, **k: None

    c._generate_summary(_msgs())

    assert seen["fallen_back"] is False


def test_auto_resolved_summary_model_falls_back_to_main_on_empty_content():
    calls = {"n": 0}

    def _route(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            # First (auto) attempt: a proxy channel answers 200 with empty content.
            kwargs["route_info"].update(provider="openrouter", model="z-ai/glm-5.3")
            return {"choices": [{"message": {"content": "   "}}]}
        # Main-model retry succeeds.
        kwargs["route_info"].update(provider="openrouter", model="main-model")
        ok = MagicMock()
        ok.choices = [MagicMock()]
        ok.choices[0].message.content = "summary via main model"
        return ok

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(model="main-model", quiet_mode=True)  # no override → provider: auto

    with patch("agent.context_compressor.call_llm", side_effect=_route) as mock_call:
        result = c._generate_summary(_msgs())

    assert mock_call.call_count == 2  # first auto route failed → retried on main
    # The main retry pins the main route explicitly, so a configured auxiliary route
    # cannot be re-selected and re-fail (#123362).
    assert mock_call.call_args_list[1].kwargs.get("model") == "main-model"
    assert result is not None and "summary via main model" in result
    # The model that actually failed (the auto-resolved one) is recorded for the user warning.
    assert c._last_aux_model_failure_model == "z-ai/glm-5.3"


def test_fallback_records_the_explicit_failed_model():
    c = _compressor(model="main-model")

    c._fallback_to_main_for_compression(Exception("boom"), "failed", failed_model="bad/route")

    assert c.summary_model == ""  # empty = use the main model
    assert c._summary_model_fallen_back is True
    assert c._last_aux_model_failure_model == "bad/route"
