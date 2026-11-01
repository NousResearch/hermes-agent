"""Current-main regressions for #113322 / #130895 auxiliary compression fallback routing.

The retry must keep compression's non-route policy while treating the main runtime as a complete
route. Otherwise unset main fields inherit the failed auxiliary endpoint or wire mode and can send
the main credential to the wrong host.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.auxiliary_client import _resolve_task_provider_model
from agent.auxiliary_task_config import (
    AuthoritativeAuxiliaryTask,
    _get_auxiliary_task_config,
    authoritative_auxiliary_task,
)
from agent.context_compressor import ContextCompressor


CONFLICTING_COMPRESSION = {
    "provider": "google",
    "model": "aux-model",
    "base_url": "https://aux.invalid/v1",
    "api_key": "aux-key",
    "api_mode": "responses",
    "reasoning_effort": "high",
    "timeout": 417,
    "max_concurrency": 2,
    "extra_body": {"trace": "kept-policy"},
}


def _config():
    return {"auxiliary": {"compression": dict(CONFLICTING_COMPRESSION)}}


def _response(content: str, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )]
    )


def _messages():
    return [
        {"role": "user", "content": "preserve this task"},
        {"role": "assistant", "content": "working on it"},
    ]


def test_authoritative_task_strips_only_destination_fields():
    task = authoritative_auxiliary_task("compression")

    assert isinstance(task, AuthoritativeAuxiliaryTask)
    assert task == "compression"
    assert hash(task) == hash("compression")

    with patch("hermes_cli.config.load_config_readonly", return_value=_config()):
        effective = _get_auxiliary_task_config(task)
        resolved = _resolve_task_provider_model(
            task,
            provider="openrouter",
            model="main-model",
            api_key="main-key",
        )

    # Policy remains compression-specific.
    assert effective["timeout"] == 417
    assert effective["max_concurrency"] == 2
    assert effective["extra_body"] == {"trace": "kept-policy"}

    # Every destination-shaping field from the failed route is removed.
    for field in (
        "provider", "model", "base_url", "api_key", "api_mode",
        "key_env", "api_key_env", "reasoning_effort",
    ):
        assert field not in effective

    assert resolved == ("openrouter", "main-model", None, "main-key", None)


def test_truncated_aux_summary_retries_on_complete_main_route():
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        compressor = ContextCompressor(
            model="main-model",
            provider="openrouter",
            api_key="main-key",
            quiet_mode=True,
            summary_model_override="aux-model",
        )

    calls: list[dict] = []

    def fake_call_llm(**kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("model") == "main-model":
            return _response("complete summary via main model")
        route_info = kwargs.get("route_info")
        if isinstance(route_info, dict):
            route_info.update(provider="google", model="aux-model")
        return _response("partial summary", finish_reason="length")

    with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
        result = compressor._generate_summary(_messages())

    assert result is not None and "complete summary via main model" in result
    assert len(calls) == 2

    retry = calls[1]
    assert isinstance(retry["task"], AuthoritativeAuxiliaryTask)
    assert retry["task"] == "compression"
    assert retry.get("provider") == "openrouter"
    assert retry.get("model") == "main-model"
    assert retry.get("api_key") == "main-key"
    assert "base_url" not in retry
    assert "api_mode" not in retry

    with patch("hermes_cli.config.load_config_readonly", return_value=_config()):
        assert _resolve_task_provider_model(
            retry["task"],
            provider=retry.get("provider"),
            model=retry.get("model"),
            base_url=retry.get("base_url"),
            api_key=retry.get("api_key"),
        ) == ("openrouter", "main-model", None, "main-key", None)


def test_main_retry_is_not_fast_lane_certified_by_the_failed_aux_reasoning():
    """call_llm on the authoritative task must not let auxiliary.compression.reasoning_effort
    certify (and disable reasoning on) the main destination."""
    from unittest.mock import MagicMock

    from agent.auxiliary_client import call_llm

    config = {"auxiliary": {"compression": {"provider": "openrouter", "model": "main-model",
                                            "base_url": "https://aux.invalid/v1", "reasoning_effort": "none"}}}
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1"
    response = _response("ok")
    client.chat.completions.create.return_value = response
    with (
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("agent.auxiliary_client._get_cached_client", return_value=(client, "main-model")),
        patch("agent.auxiliary_client._validate_llm_response", return_value=response),
    ):
        call_llm(task=authoritative_auxiliary_task("compression"), provider="openrouter", model="main-model",
                 api_key="main-key", messages=_messages())

    request = client.chat.completions.create.call_args.kwargs
    assert "reasoning_effort" not in request
    assert "reasoning" not in request.get("extra_body", {})
