"""Auxiliary-summary fallback must actually reach the main model.

``_fallback_to_main_for_compression`` clears ``summary_model`` to mean "use the
main model". But ``call_llm(task="compression", ...)`` resolves an unspecified
route from ``auxiliary.compression`` in the config -- i.e. the model that just
failed. Without an explicit route the fallback re-calls the failing auxiliary
model, and since a second fallback is not allowed the summary aborts and
compression is skipped entirely.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor


def _compressor(**kwargs):
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(quiet_mode=True, **kwargs)


def _ok_response(text="summary via main model"):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    return response


def _quota_error():
    error = Exception("Error code: 429 - usage_limit_reached")
    error.status_code = 429
    return error


def _turns():
    return [
        {"role": "user", "content": "do something"},
        {"role": "assistant", "content": "ok"},
    ]


def test_quota_exhausted_aux_falls_back_to_named_main_model():
    """A 429 on the aux model must retry on an explicitly named main model."""
    compressor = _compressor(
        model="main-model",
        provider="main-provider",
        summary_model_override="quota-exhausted-aux",
    )

    with patch(
        "agent.context_compressor.call_llm",
        side_effect=[_quota_error(), _ok_response()],
    ) as mock_call:
        summary = compressor._generate_summary(_turns())

    assert summary is not None
    assert "summary via main model" in summary
    assert mock_call.call_count == 2

    first, second = mock_call.call_args_list
    assert first.kwargs.get("model") == "quota-exhausted-aux"
    # Without an explicit route the retry resolves auxiliary.compression again
    # and re-calls the exhausted model.
    assert second.kwargs.get("model") == "main-model"
    assert second.kwargs.get("provider") == "main-provider"
    main_runtime = second.kwargs.get("main_runtime")
    assert main_runtime is not None
    assert main_runtime["model"] == "main-model"
    assert main_runtime["provider"] == "main-provider"


def test_aux_quota_failure_does_not_stick_when_main_succeeds():
    """A terminal access/quota flag from the aux attempt must not survive a
    successful main-model retry, otherwise compress() aborts anyway."""
    compressor = _compressor(
        model="main-model",
        provider="main-provider",
        summary_model_override="quota-exhausted-aux",
    )

    with patch(
        "agent.context_compressor.call_llm",
        side_effect=[_quota_error(), _ok_response()],
    ):
        summary = compressor._generate_summary(_turns())

    assert summary is not None
    assert compressor._last_summary_auth_failure is False


def test_route_is_untouched_before_any_fallback():
    """No fallback yet: the configured aux route must be preserved verbatim."""
    compressor = _compressor(
        model="main-model",
        provider="main-provider",
        summary_model_override="healthy-aux",
    )

    call_kwargs = {"task": "compression"}
    compressor._apply_summary_route(call_kwargs)

    assert call_kwargs["model"] == "healthy-aux"
    assert "provider" not in call_kwargs


def test_route_is_untouched_when_no_aux_model_is_configured():
    """No aux model and no fallback: leave task resolution to call_llm."""
    compressor = _compressor(model="main-model", provider="main-provider")

    call_kwargs = {"task": "compression"}
    compressor._apply_summary_route(call_kwargs)

    assert call_kwargs == {"task": "compression"}


def test_micro_summarize_also_names_main_model_after_fallback():
    """The micro-summarization entry point shares the same defect and fix."""
    compressor = _compressor(
        model="main-model",
        provider="main-provider",
        summary_model_override="quota-exhausted-aux",
    )
    compressor._fallback_to_main_for_compression(_quota_error(), "failed")

    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="merged"))]
        )

    with patch("agent.auxiliary_client.call_llm", side_effect=_capture), patch.object(
        compressor, "_build_micro_summary_prompt", return_value=[{"role": "user", "content": "x"}]
    ):
        compressor._micro_summarize_one("an exchange")

    assert captured.get("model") == "main-model"
    assert captured.get("provider") == "main-provider"
    main_runtime = captured.get("main_runtime")
    assert main_runtime is not None
    assert main_runtime["model"] == "main-model"
    assert main_runtime["provider"] == "main-provider"
