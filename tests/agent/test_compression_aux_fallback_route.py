"""The aux-summary fallback must actually reach the main model (#123362).

``_fallback_to_main_for_compression`` clears ``summary_model``, but an omitted route makes
``call_llm(task="compression")`` re-resolve ``auxiliary.compression`` from config — the model
that just failed — so the retry re-fails and later attempts abort without a second fallback.

After fallback, the pinned main route must not stamp a stored ``api_key`` snapshot onto the
call (#138469): a revoked OAuth token would 401 every compaction while ``auxiliary.compression``
stays ignored.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


def _auth_revoked_error():
    error = Exception(
        "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', "
        "'message': 'OAuth access token has been revoked.'}}"
    )
    error.status_code = 401
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
    # Fallen-back main route names the model/provider but does not pin a stored api_key (#138469).
    assert "api_key" not in second.kwargs


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
    assert "api_key" not in captured
    assert "sk-MAIN-SECRET" not in repr({k: v for k, v in captured.items() if k != "main_runtime"})

    # A KEYLESS stall-fallback pin (local server) replaces the whole route: the main key never rides along.
    pin = {"provider": "custom", "model": "llama3", "base_url": "http://other-host:8080/v1", "api_key": None}
    with patch("agent.context_compressor.call_llm", side_effect=_capture), pin_summary_route(pin):
        compressor._call_summary_llm("prompt", 0.0)

    assert {k: captured.get(k) for k in ("provider", "model", "base_url", "api_key")} == {
        "provider": "custom", "model": "llama3", "base_url": "http://other-host:8080/v1", "api_key": None,
    }
    assert "sk-MAIN-SECRET" not in repr({k: v for k, v in captured.items() if k != "main_runtime"})


def test_fallen_back_summary_route_omits_stored_api_key():
    """#138469: latched main route must not pin a revoked OAuth snapshot as explicit api_key."""
    compressor = _compressor()
    compressor.api_key = "sk-revoked-oauth-snapshot"
    compressor._fallback_to_main_for_compression(_auth_revoked_error(), "failed")
    call_kwargs = {"task": "compression", "api_key": "sk-preexisting-stale"}
    compressor._apply_summary_route(call_kwargs)
    assert call_kwargs["provider"] == "main-provider"
    assert call_kwargs["model"] == "main-model"
    assert call_kwargs["base_url"] == "https://main.example/v1"
    assert "api_key" not in call_kwargs


def test_auth_failure_on_fallen_back_route_clears_latch():
    """#138469: a 401 on the pinned main route clears the latch so aux.compression applies again."""
    compressor = _compressor()
    compressor._fallback_to_main_for_compression(_auth_revoked_error(), "failed")
    assert compressor._summary_model_fallen_back is True

    with patch("agent.context_compressor.call_llm", side_effect=_auth_revoked_error()):
        result = compressor._generate_summary([
            {"role": "user", "content": "do something"}, {"role": "assistant", "content": "ok"},
        ])

    assert result is None
    assert compressor._summary_model_fallen_back is False
    assert compressor._last_summary_auth_failure is True


def test_fallen_back_route_strips_snapshot_from_main_runtime():
    """#138469 review: the nested main_runtime snapshot must not survive as the session credential."""
    compressor = _compressor()
    compressor.api_key = "sk-revoked-oauth-snapshot"
    compressor._fallback_to_main_for_compression(_auth_revoked_error(), "failed")
    runtime = {"model": "main-model", "provider": "main-provider", "api_key": "sk-revoked-oauth-snapshot"}
    call_kwargs = {"task": "compression", "main_runtime": runtime}
    compressor._apply_summary_route(call_kwargs)
    assert "api_key" not in call_kwargs["main_runtime"]
    assert call_kwargs["main_runtime"]["provider"] == "main-provider"
    assert runtime["api_key"] == "sk-revoked-oauth-snapshot"  # caller's dict untouched

    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="summary " * 20), finish_reason="stop")])

    with patch("agent.context_compressor.call_llm", side_effect=_capture):
        compressor._call_summary_llm("prompt", 0.0)
    assert "sk-revoked-oauth-snapshot" not in repr(captured)


def test_fallen_back_custom_endpoint_keeps_main_runtime_key():
    """Custom endpoints cannot pool-resolve, so their main_runtime key is the only credential."""
    compressor = _compressor()
    compressor.provider = "custom:my-gateway"
    compressor._fallback_to_main_for_compression(_access_error(), "failed")
    call_kwargs = {"task": "compression", "main_runtime": {"provider": "custom:my-gateway", "api_key": "sk-custom"}}
    compressor._apply_summary_route(call_kwargs)
    assert "api_key" not in call_kwargs
    assert call_kwargs["main_runtime"]["api_key"] == "sk-custom"
