"""The aux-summary fallback must actually reach the main model (#123362).

``_fallback_to_main_for_compression`` clears ``summary_model``, but an omitted route makes
``call_llm(task="compression")`` re-resolve ``auxiliary.compression`` from config — the model
that just failed — so the retry re-fails and later attempts abort without a second fallback.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor, pin_summary_route


def _compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="main-model", provider="main-provider", quiet_mode=True,
            base_url="https://main.example/v1", api_key="sk-MAIN-SECRET",
            summary_model_override="quota-exhausted-aux",
        )


def _access_error():
    error = Exception("Error code: 403 - model is not available in the current token plan")
    error.status_code = 403
    return error


def test_access_failed_aux_falls_back_to_named_main_route():
    ok = MagicMock()
    ok.choices = [MagicMock()]
    ok.choices[0].message.content = "summary via main model"
    compressor = _compressor()

    with patch("agent.context_compressor.call_llm", side_effect=[_access_error(), ok]) as mock_call:
        summary = compressor._generate_summary([
            {"role": "user", "content": "do something"}, {"role": "assistant", "content": "ok"},
        ])

    assert summary is not None and "summary via main model" in summary
    first, second = mock_call.call_args_list
    assert first.kwargs.get("model") == "quota-exhausted-aux"
    assert (second.kwargs.get("model"), second.kwargs.get("provider")) == ("main-model", "main-provider")


@pytest.mark.parametrize("refusal_type", ["auxiliary", "primary"])
def test_local_admission_refusal_does_not_fallback_to_main_model(refusal_type):
    from agent.auxiliary_client import LocalAuxiliaryAdmissionRequired
    from hermes_cli.middleware import LocalModelAdmissionBlockedError

    compressor = _compressor()
    refusal_class = (LocalAuxiliaryAdmissionRequired if refusal_type == "auxiliary"
                     else LocalModelAdmissionBlockedError)
    refusal = refusal_class("local model admission refused before provider I/O")
    with patch.object(compressor, "_transcript_has_real_user_turn", return_value=False), patch(
        "agent.context_compressor.call_llm", side_effect=refusal,
    ) as mock_call:
        try:
            compressor._generate_summary([
                {"role": "assistant", "content": "do something"},
            ])
        except LocalAuxiliaryAdmissionRequired as exc:
            assert "local model admission refused" in str(exc)
        except LocalModelAdmissionBlockedError as exc:
            assert "local model admission refused" in str(exc)
        else:
            raise AssertionError("admission refusal must stop compression without fallback")

    assert mock_call.call_count == 1
    assert mock_call.call_args.kwargs.get("model") == "quota-exhausted-aux"
    assert compressor.summary_model == "quota-exhausted-aux"
    assert compressor._last_summary_error == str(refusal)


def test_fallen_back_route_names_main_for_micro_and_never_rides_into_a_pin():
    compressor = _compressor()
    compressor._fallback_to_main_for_compression(_access_error(), "failed")
    captured = {}

    def _capture(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="merged " * 20), finish_reason="stop")])

    with patch("agent.auxiliary_client.call_llm", side_effect=_capture), patch.object(
        compressor, "_build_micro_summary_prompt", return_value=[{"role": "user", "content": "x"}]
    ):
        compressor._micro_summarize_one("an exchange")

    assert (captured.get("model"), captured.get("provider")) == ("main-model", "main-provider")

    # A KEYLESS stall-fallback pin (local server) replaces the whole route: the main key never rides along.
    pin = {"provider": "custom", "model": "llama3", "base_url": "http://other-host:8080/v1", "api_key": None}
    with patch("agent.context_compressor.call_llm", side_effect=_capture), pin_summary_route(pin):
        compressor._call_summary_llm("prompt", 0.0)

    assert {k: captured.get(k) for k in ("provider", "model", "base_url", "api_key")} == {
        "provider": "custom", "model": "llama3", "base_url": "http://other-host:8080/v1", "api_key": None,
    }
    assert "sk-MAIN-SECRET" not in repr({k: v for k, v in captured.items() if k != "main_runtime"})
