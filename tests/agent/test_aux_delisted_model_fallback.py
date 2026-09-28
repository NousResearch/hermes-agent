"""A delisted-model 404 must walk the configured task fallback chain.

Incident: OpenRouter removed ``nex-agi/nex-n2.5-mini:free`` from its catalogue;
every auxiliary call pinned to it failed with HTTP 404 ``No endpoints found for
<model>.`` That error matched none of the ``_FALLBACK_REASONS`` predicates, so
``_ladder_provider_fallback`` returned without consulting
``auxiliary.<task>.fallback_chain`` — the paid fallback leg behind titles sat
idle while the task failed outright.
"""

import pytest

import agent.auxiliary_client as aux

PRIMARY_MODEL = "nex-agi/nex-n2.5-mini:free"
FALLBACK_MODEL = "fallback-model"


class _ApiError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _delisted_error():
    return _ApiError(
        "Error code: 404 - {'error': {'message': "
        "'No endpoints found for %s.', 'code': 404}}" % PRIMARY_MODEL,
        status_code=404,
    )


def test_delisted_model_404_is_model_not_found():
    """The bare OpenRouter delisted-model 404 belongs to the not-found bucket.

    Capability-scoped cousins (image input, tool use) keep their own routing —
    the new phrase must not swallow them.
    """
    assert aux._is_model_not_found_error(_delisted_error()) is True
    for cousin in (
        "No endpoints found that support image input",
        "No endpoints found that support tool use",
    ):
        exc = _ApiError("Error code: 404 - %s" % cousin, status_code=404)
        assert aux._is_model_not_found_error(exc) is False


def test_delisted_404_reaches_configured_task_fallback(monkeypatch):
    """The ladder consults the task chain for a delisted 404 on an explicit route."""
    fallback_client = object()
    monkeypatch.setattr(
        aux,
        "_get_auxiliary_task_config",
        lambda task: {"fallback_chain": [{"provider": "custom:backup"}]},
    )
    monkeypatch.setattr(aux, "_recoverable_pool_provider", lambda *a, **kw: None)
    monkeypatch.setattr(
        aux,
        "_try_configured_fallback_chain",
        lambda *a, **kw: (fallback_client, FALLBACK_MODEL, "fallback_chain[0](custom:backup)"),
    )
    ladder = aux._aux_recovery_ladder(
        _delisted_error(),
        client=object(),
        kwargs={"model": PRIMARY_MODEL},
        task="title_generation",
        async_mode=False,
        base_info="https://openrouter.ai/api/v1",
        resolved_provider="custom:openrouter-free",
        resolved_model=PRIMARY_MODEL,
        resolved_base_url=None,
        resolved_api_key=None,
        resolved_api_mode=None,
        final_model=PRIMARY_MODEL,
        max_tokens=None,
        main_runtime=None,
        route_info={},
    )

    def perform(step):
        assert step.kind == "fallback"
        assert step.args == (fallback_client, FALLBACK_MODEL, "fallback_chain[0](custom:backup)")
        return "fallback-response"

    try:
        result = aux._drive_ladder(ladder, perform)
    except _ApiError as exc:
        pytest.fail(
            "the ladder let %r escape instead of consulting the configured "
            "fallback chain" % (exc,)
        )
    assert result == "fallback-response"
