"""The provider-fallback gate must stay open for unrecognised failures (#136849).

``_ladder_provider_fallback`` used to gate the configured ``fallback_chain`` on the failure
being *recognised*: an error matching no predicate in ``_FALLBACK_REASONS`` (a deprecated
model id's 404, a novel vendor status, a free-tier body without a status code) closed the
gate, the chain was never consulted, and the caller retried the same dead backend until its
attempt budget was spent. A configured chain is a user instruction — "if this backend cannot
serve the task, use this one instead" — and an unrecognised failure is precisely the case
the chain exists for; enumerating every recognisable failure is unwinnable. The single
provider-pinned case survives: a *recognised* auth error on an explicit provider with no
task chain (a bad credential is not evidence the endpoint cannot serve the task).
"""

import pytest

import agent.auxiliary_client as aux


class _ApiError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


def _deprecated_model_error():
    """A 404 naming a since-dropped model: no predicate in ``_FALLBACK_REASONS`` claims it."""
    return _ApiError(
        "Error code: 404 - The model `grok-2-deprecated` does not exist or you do not "
        "have access to it.",
        status_code=404,
    )


def _route(resolved_provider="openrouter"):
    return aux._LadderRoute(
        None, "compression", "", False, "https://openrouter.ai/api/v1", resolved_provider,
        "grok-2-deprecated", None, None, None, "grok-2-deprecated", None, {}, None)


def test_unrecognised_error_walks_its_configured_task_chain(monkeypatch, caplog):
    """A 404 for a deprecated model on an explicit provider must fall back through the
    task's configured chain (named "unrecognised error" in the logs) instead of closing
    the gate and letting the caller retry the dead backend."""
    monkeypatch.setattr(
        aux, "_get_auxiliary_task_config",
        lambda task: {"fallback_chain": [{"provider": "custom:backup"}]})
    fallback_client = object()
    monkeypatch.setattr(
        aux, "_try_configured_fallback_chain",
        lambda *a, **k: (fallback_client, "backup-model", "fallback_chain[0](custom:backup)"))
    monkeypatch.setattr(
        aux, "_try_main_agent_model_fallback",
        lambda *a, **k: pytest.fail("a task chain answered; the main-agent model must not be asked"))

    steps = []

    def perform(step):
        steps.append(step.kind)
        assert step.kind == "fallback"
        return "fallback-response"

    with caplog.at_level("INFO", logger="agent.auxiliary_client"):
        result = aux._drive_ladder(
            aux._ladder_provider_fallback(_deprecated_model_error(), _route()), perform)

    assert result == "fallback-response"
    assert steps == ["fallback"]
    assert "unrecognised error" in caplog.text, (
        f"the gate must name the failure class in its logs, not a bare None: {caplog.text}")


def test_auto_route_unrecognised_error_walks_the_chains(monkeypatch):
    """Under ``provider: auto`` an unrecognised failure reaches the configured chain like
    any recognised capacity error would."""
    fallback_client = object()
    monkeypatch.setattr(
        aux, "_try_configured_fallback_chain",
        lambda *a, **k: (fallback_client, "fb-model", "fallback_chain[0](openrouter)"))
    for name in ("_try_main_fallback_chain", "_try_payment_fallback"):
        monkeypatch.setattr(
            aux, name,
            lambda *a, _n=name, **k: (_n, pytest.fail("the configured chain already answered")))

    def perform(step):
        assert step.kind == "fallback"
        return "auto-fallback-response"

    result = aux._drive_ladder(
        aux._ladder_provider_fallback(_deprecated_model_error(), _route(resolved_provider="auto")),
        perform)
    assert result == "auto-fallback-response"


def test_recognised_auth_without_a_task_chain_stays_provider_pinned(monkeypatch):
    """The one provider-pinned case survives the gate inversion: a recognised 401 on an
    explicit provider with no configured task chain never consults any fallback layer."""
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {})
    monkeypatch.setattr(
        aux, "_try_configured_fallback_chain",
        lambda *a, **k: pytest.fail("a recognised auth error without a task chain is provider-pinned"))
    monkeypatch.setattr(
        aux, "_try_main_agent_model_fallback",
        lambda *a, **k: pytest.fail("explicit auth must not use the main-agent fallback"))

    assert aux._drive_ladder(
        aux._ladder_provider_fallback(
            _ApiError("Error code: 401 - Unauthorized", status_code=401), _route()),
        lambda step: pytest.fail("no fallback request expected")) is None
