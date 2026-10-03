"""#123362: a temporary auxiliary compression-model failure must not permanently
downgrade compression to the main model. ``_fallback_to_main_for_compression()``
clears ``summary_model`` so the one-shot main-model retry runs; once that retry
succeeds, the operator-configured aux route must be re-armed so the next
compression attempt returns to it as soon as it is usable again.
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


class _AccessQuotaError(Exception):
    """Shaped like an HTTP 403 access/quota failure (status on the exception)."""

    status_code = 403


def _msgs():
    return [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": "ok"},
    ]


def _ok_response(content: str) -> MagicMock:
    ok = MagicMock()
    ok.choices = [MagicMock()]
    ok.choices[0].message.content = content
    return ok


def test_aux_route_is_rearmed_after_a_successful_main_model_fallback():
    """After the one-shot main-model retry succeeds, the NEXT attempt probes the
    configured aux route again instead of being permanently pinned to main."""
    calls = {"n": 0}

    def _route(**kwargs):
        calls["n"] += 1
        kwargs["route_info"].update(provider="openrouter", model=kwargs.get("model", "main-model"))
        if kwargs.get("model") == "aux-model":
            raise _AccessQuotaError("model is not available in the current token plan")
        return _ok_response("summary via main model")

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(model="main-model", quiet_mode=True, summary_model_override="aux-model")

    with patch("agent.context_compressor.call_llm", side_effect=_route) as mock_call:
        first = c._generate_summary(_msgs())
        second = c._generate_summary(_msgs())

    assert first is not None and "main model" in first
    assert second is not None and "main model" in second
    # Attempt 1: aux (fails) then main (succeeds). Attempt 2: aux again — the route was re-armed.
    assert [call.kwargs.get("model") for call in mock_call.call_args_list] == [
        "aux-model", None, "aux-model", None,
    ]
    assert c.summary_model == "aux-model"  # re-armed for the next attempt
    assert c._summary_model_fallen_back is False


def test_compression_returns_to_the_aux_route_once_it_recovers():
    """The configured aux route is used again as soon as it succeeds, instead of
    the compressor staying on the main model for the rest of its lifetime."""
    calls = {"n": 0}

    def _route(**kwargs):
        calls["n"] += 1
        kwargs["route_info"].update(provider="openrouter", model=kwargs.get("model", "main-model"))
        if kwargs.get("model") == "aux-model" and calls["n"] == 1:
            raise _AccessQuotaError("model is not available in the current token plan")
        if kwargs.get("model") == "aux-model":
            return _ok_response("summary via the recovered aux route")
        return _ok_response("summary via main model")

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(model="main-model", quiet_mode=True, summary_model_override="aux-model")

    with patch("agent.context_compressor.call_llm", side_effect=_route) as mock_call:
        first = c._generate_summary(_msgs())
        second = c._generate_summary(_msgs())

    # Attempt 1: aux fails once -> main succeeds; attempt 2: the recovered aux route is used.
    assert [call.kwargs.get("model") for call in mock_call.call_args_list] == [
        "aux-model", None, "aux-model",
    ]
    assert first is not None and "main model" in first
    assert second is not None and "recovered aux route" in second
