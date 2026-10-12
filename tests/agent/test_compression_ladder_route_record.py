"""A recovery rung that swaps the model must record the route that served.

``_SummaryRouteRecord`` counts route records: ``call_llm`` records the route it prepared, then once more
for every recovery-ladder lane it tries. A second record is how the compressor learns that the ladder
left the pinned route, so the remaining summary chunks follow the route that served (or fail closed
when it cannot be reproduced) instead of returning to the failed one. The Nous rungs replace
``kwargs["model"]`` (stale-model self-heal, refreshed recommendation) and retry without recording:
the record kept naming the prepared model while another model produced the summary.
"""
from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import agent.auxiliary_client as aux
from agent.context_compressor import _SummaryRouteRecord

NOUS_BASE = "https://inference-api.nousresearch.com/v1"
STALE = "nous/stale-model"
HEALED = "nous/recommended-model"


class _ApiError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class _NousClient:
    api_key = "sk-test"
    base_url = NOUS_BASE


@pytest.fixture
def offline_ladder(monkeypatch):
    """No pool, no paid-account probe, and a provider-fallback rung that must not be reached."""
    reached_fallback = []

    def provider_fallback(first_err, route):
        reached_fallback.append(first_err)
        yield from ()

    monkeypatch.setattr(aux, "_recoverable_pool_provider", lambda *a, **k: None)
    monkeypatch.setattr(aux, "_nous_portal_account_has_fresh_paid_access", lambda: False)
    monkeypatch.setattr(aux, "_ladder_provider_fallback", provider_fallback)
    return reached_fallback


def _prepared_record():
    record = _SummaryRouteRecord()
    aux._record_route_info(record, "nous", STALE)  # what _plan_aux_call records before the first send
    assert record.records == 1
    return record


def _drive(first_err, record):
    kwargs = {"model": STALE, "messages": [{"role": "user", "content": "summarize"}]}
    sent = []

    def perform(step):
        assert step.kind == "call"
        sent.append(step.args[1]["model"])
        return "served"

    ladder = aux._aux_recovery_ladder(
        first_err, client=_NousClient(), kwargs=kwargs, task="compression", async_mode=False,
        base_info=NOUS_BASE, resolved_provider="nous", resolved_model=STALE, resolved_base_url=None,
        resolved_api_key=None, resolved_api_mode=None, final_model=STALE, max_tokens=None,
        main_runtime=None, route_info=record)
    return aux._drive_ladder(ladder, perform), sent


def test_stale_model_self_heal_records_the_model_that_served(offline_ladder, monkeypatch):
    monkeypatch.setattr(aux, "_refresh_nous_recommended_model", lambda **k: HEALED)
    record = _prepared_record()

    result, sent = _drive(_ApiError("Error code: 404 - model not found", status_code=404), record)

    assert (result, sent) == ("served", [HEALED])
    assert (record["provider"], record["model"]) == ("nous", HEALED)
    assert record.records == 2, "the compressor must see that the ladder left the prepared route"
    assert offline_ladder == []


def test_refreshed_nous_client_with_a_new_model_records_it(offline_ladder, monkeypatch):
    monkeypatch.setattr(aux, "_refresh_nous_auxiliary_client", lambda **k: (_NousClient(), HEALED))
    record = _prepared_record()

    result, sent = _drive(_ApiError("Error code: 401 - Unauthorized", status_code=401), record)

    assert (result, sent) == ("served", [HEALED])
    assert (record["provider"], record["model"]) == ("nous", HEALED)
    assert record.records == 2


def test_refreshed_nous_client_on_the_same_model_is_the_same_route(offline_ladder, monkeypatch):
    """Fresh credentials on the prepared model are not a lane change: no second record."""
    monkeypatch.setattr(aux, "_refresh_nous_auxiliary_client", lambda **k: (_NousClient(), STALE))
    record = _prepared_record()

    result, sent = _drive(_ApiError("Error code: 401 - Unauthorized", status_code=401), record)

    assert (result, sent) == ("served", [STALE])
    assert (record["provider"], record["model"]) == ("nous", STALE)
    assert record.records == 1


class _AuthExpired(Exception):
    status_code = 401

    def __init__(self):
        super().__init__("Error code: 401 - Provided authentication token is expired")


class _Answer:
    def __init__(self, text):
        self.choices = [MagicMock(message=MagicMock(content=text))]


def _retry_after_refresh(rebuilt_model):
    """``call_llm`` on a pinned route whose first send gets a 401; the same-provider retry runs on a
    client rebuilt after the credential refresh, which resolves its model again (``rebuilt_model``)."""
    stale, fresh = MagicMock(), MagicMock()
    stale.base_url = fresh.base_url = "https://api.anthropic.com"
    stale.api_key = "expired-token"
    stale.chat.completions.create.side_effect = _AuthExpired()
    fresh.chat.completions.create.return_value = _Answer("served")
    clients = iter([(stale, PREPARED), (fresh, rebuilt_model)])
    record = _SummaryRouteRecord()
    with patch.object(aux, "_resolve_task_provider_model", return_value=("anthropic", PREPARED, None, None, None)), \
         patch.object(aux, "_get_cached_client", side_effect=lambda *a, **k: next(clients)), \
         patch.object(aux, "_recoverable_pool_provider", return_value=None), \
         patch.object(aux, "_refresh_provider_credentials", return_value=True):
        result = aux.call_llm(task="compression", provider="anthropic", model=PREPARED,
                              messages=[{"role": "user", "content": "summarize"}], route_info=record)
    assert result.choices[0].message.content == "served"
    sent = [call.kwargs["model"] for call in fresh.chat.completions.create.call_args_list]
    return record, sent


PREPARED = "claude-prepared"


def test_a_same_provider_retry_on_a_rebuilt_model_records_it():
    record, sent = _retry_after_refresh("claude-rebuilt")

    assert sent == ["claude-rebuilt"]
    assert (record["provider"], record["model"]) == ("anthropic", "claude-rebuilt")
    assert record.records == 2, "the compressor must see that the retry left the prepared route"


def test_a_same_provider_retry_on_the_prepared_model_is_the_same_route():
    """Fresh credentials on the prepared model are not a lane change: no second record."""
    record, sent = _retry_after_refresh(PREPARED)

    assert sent == [PREPARED]
    assert (record["provider"], record["model"]) == ("anthropic", PREPARED)
    assert record.records == 1


class _PaymentRequired(Exception):
    status_code = 402

    def __init__(self):
        super().__init__("Error code: 402 - insufficient credits")


FALLBACK_LABEL, FALLBACK_MODEL = "fallback_chain[0](openrouter)", "fallback-model"


def _fallback_clients(rebuilt_model, *, async_mode):
    """Primary out of credit; the configured fallback answers its first send with a 401; the client rebuilt after
    the credential refresh resolves its model again (``rebuilt_model``) and serves."""
    primary, stale, fresh = MagicMock(), MagicMock(), MagicMock()
    primary.base_url = "https://api.anthropic.com"
    stale.base_url = fresh.base_url = "https://openrouter.ai/api/v1"
    stale.api_key = "expired-token"
    mock = AsyncMock if async_mode else MagicMock
    primary.chat.completions.create = mock(side_effect=_PaymentRequired())
    stale.chat.completions.create = mock(side_effect=_AuthExpired())
    fresh.chat.completions.create = mock(return_value=_Answer("served"))
    clients = iter([(primary, PREPARED), (fresh, rebuilt_model)])
    patches = [
        patch.object(aux, "_resolve_task_provider_model", return_value=("anthropic", PREPARED, None, None, None)),
        patch.object(aux, "_get_cached_client", side_effect=lambda *a, **k: next(clients)),
        patch.object(aux, "_recoverable_pool_provider", return_value=None),
        patch.object(aux, "_refresh_provider_credentials", return_value=True),
        patch.object(aux, "_try_configured_fallback_chain", return_value=(stale, FALLBACK_MODEL, FALLBACK_LABEL)),
        patch.object(aux, "_to_async_client", side_effect=lambda client, model, is_vision=False: (client, model)),
    ]
    return fresh, patches


def _fallback_after_refresh(rebuilt_model, *, async_mode=False):
    fresh, patches = _fallback_clients(rebuilt_model, async_mode=async_mode)
    record = _SummaryRouteRecord()
    request = dict(task="compression", provider="anthropic", model=PREPARED,
                   messages=[{"role": "user", "content": "summarize"}], route_info=record)
    with contextlib.ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        result = asyncio.run(aux.async_call_llm(**request)) if async_mode else aux.call_llm(**request)
    assert result.choices[0].message.content == "served"
    sent = [call.kwargs["model"] for call in fresh.chat.completions.create.call_args_list]
    return record, sent


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_a_fallback_rebuilt_on_another_model_records_the_model_it_serves(async_mode):
    record, sent = _fallback_after_refresh("fallback-rebuilt", async_mode=async_mode)

    assert sent == ["fallback-rebuilt"]
    assert (record["provider"], record["model"]) == ("openrouter", "fallback-rebuilt")
    assert record.records == 3, "prepared route, fallback lane, then the model the rebuilt fallback serves"


@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
def test_a_fallback_rebuilt_on_its_own_model_is_the_same_lane(async_mode):
    record, sent = _fallback_after_refresh(FALLBACK_MODEL, async_mode=async_mode)

    assert sent == [FALLBACK_MODEL]
    assert (record["provider"], record["model"]) == ("openrouter", FALLBACK_MODEL)
    assert record.records == 2


def test_model_swap_without_a_route_observer_is_harmless(offline_ladder, monkeypatch):
    monkeypatch.setattr(aux, "_refresh_nous_recommended_model", lambda **k: HEALED)
    kwargs = {"model": STALE, "messages": []}
    ladder = aux._aux_recovery_ladder(
        _ApiError("Error code: 404 - model not found", status_code=404), client=_NousClient(), kwargs=kwargs,
        task="compression", async_mode=False, base_info=NOUS_BASE, resolved_provider="nous",
        resolved_model=STALE, resolved_base_url=None, resolved_api_key=None, resolved_api_mode=None,
        final_model=STALE, max_tokens=None, main_runtime=None, route_info=None)
    assert aux._drive_ladder(ladder, lambda step: "served") == "served"
