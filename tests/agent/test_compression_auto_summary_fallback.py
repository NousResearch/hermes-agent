"""#116472: an ``auto``-resolved compression summary model that fails (e.g. a proxy channel
answering HTTP 200 with empty content) must fall back to the main model. ``auto`` resolves a
model per call without setting ``summary_model``, so the retry gate previously saw "no separate
model" and re-hit the same bad route forever; the resolved model is also recorded so the
user-visible warning names it instead of only appearing in errors.log.
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


def _msgs():
    return [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": "ok"},
    ]


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
    assert "model" not in mock_call.call_args_list[1].kwargs
    assert result is not None and "summary via main model" in result
    # The model that actually failed (the auto-resolved one) is recorded for the user warning.
    assert c._last_aux_model_failure_model == "z-ai/glm-5.3"


def test_fallback_records_the_explicit_failed_model():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(model="main-model", quiet_mode=True)

    c._fallback_to_main_for_compression(Exception("boom"), "failed", failed_model="bad/route")

    assert c.summary_model == ""  # empty = use the main model
    assert c._summary_model_fallen_back is True
    assert c._last_aux_model_failure_model == "bad/route"


def test_success_on_main_rearms_the_configured_summary_model():
    """Regression for #123362: the fallback cleared the configured summary model and
    latched for the compressor's lifetime; a success on main must re-arm the route so
    a recovered aux model is probed again instead of being abandoned forever."""
    calls = {"n": 0}

    def _route(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("aux model is not available")
        ok = MagicMock()
        ok.choices = [MagicMock()]
        ok.choices[0].message.content = "summary via main model"
        return ok

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(model="main-model", summary_model_override="aux/route", quiet_mode=True)

    with patch("agent.context_compressor.call_llm", side_effect=_route):
        result = c._generate_summary(_msgs())

    assert result is not None
    assert c._summary_model_fallen_back is False
    assert c.summary_model == "aux/route"  # re-armed for the next compression


def test_rearmed_route_is_probed_again_and_bounded():
    """The recovery loop costs one aux attempt + one main success per compression —
    bounded by the one-shot fallback machinery, never a tight retry loop."""
    calls = {"n": 0}

    def _route(**kwargs):
        calls["n"] += 1
        if calls["n"] % 2 == 1:  # the aux attempt of each cycle (re-arm makes it odd again)
            kwargs["route_info"].update(provider="openrouter", model="aux/route")
            raise RuntimeError("aux model is not available")
        kwargs["route_info"].update(provider="openrouter", model="main-model")
        ok = MagicMock()
        ok.choices = [MagicMock()]
        ok.choices[0].message.content = "summary"
        return ok

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(model="main-model", summary_model_override="aux/route", quiet_mode=True)

    with patch("agent.context_compressor.call_llm", side_effect=_route):
        assert c._generate_summary(_msgs()) is not None
        assert c._generate_summary(_msgs()) is not None

    assert calls["n"] == 4  # 2 per compression: aux attempt + main success


def test_wrapped_error_status_dug_from_cause_chain():
    """#123362: a wrapped RuntimeError must classify like the raw error it caused."""
    from agent.context_compressor import _exc_status_code, _is_summary_access_or_quota_error

    class _ApiError(Exception):
        def __init__(self, msg, status_code):
            super().__init__(msg)
            self.status_code = status_code

    raw = _ApiError("model is not available in the current token plan", 403)
    wrapped = RuntimeError("Context compression LLM call failed")
    wrapped.__cause__ = raw

    assert _exc_status_code(wrapped) == 403
    assert _is_summary_access_or_quota_error(wrapped) is True
