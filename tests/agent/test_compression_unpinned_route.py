"""An ordinary (unpinned) multi-chunk summary follows the dispatch that rescued a chunk, and only then."""
from unittest.mock import MagicMock, patch

import pytest

from agent import auxiliary_client as ac
from agent.context_compressor import _SummaryRouteRecord
from tests.agent.test_compression_transport_bound import (
    BACKUP_URL, MAIN_URL, PIN_URL, SNAPSHOT,
    _client, _main_compressor, _ok, _payment_402, _two_chunk_source,
)


@pytest.mark.parametrize("chain_rescue", [True, False], ids=["task-chain", "main-runtime"])
def test_unpinned_plan_does_not_return_to_the_failed_primary(chain_rescue):
    primary_prompts, successful_prompts, dispatches = [], [], []

    def broken(**kwargs):
        primary_prompts.append(kwargs["messages"][0]["content"])
        raise _payment_402()

    def healthy(**kwargs):
        successful_prompts.append(kwargs["messages"][0]["content"])
        return _ok(SNAPSHOT + f"part {len(successful_prompts)}")

    clients = {
        PIN_URL: _client(PIN_URL, MagicMock(side_effect=broken), "synthetic-pin-key"),
        BACKUP_URL: _client(BACKUP_URL, MagicMock(side_effect=healthy), "synthetic-backup-key"),
        MAIN_URL: _client(MAIN_URL, MagicMock(side_effect=healthy), "synthetic-main-key"),
    }

    def configured(task=None, provider=None, model=None, base_url=None, api_key=None, *_a, **_kw):
        return provider or "custom", model or "pinned/model", base_url or PIN_URL, api_key, None

    def cached(provider, model=None, *_a, base_url=None, **_kw):
        return clients[str(base_url)], model

    real_call = ac.call_llm

    def spy(**kwargs):
        dispatches.append({k: kwargs.get(k) for k in ("provider", "model", "base_url", "api_key")})
        return real_call(**kwargs)

    main = clients[MAIN_URL], "main/model", "main-agent(custom)"
    chain = (clients[BACKUP_URL], "backup/model", "fallback_chain[1](custom)") if chain_rescue else (None, None, "")
    compressor = _main_compressor()
    with patch("agent.context_compressor.call_llm", side_effect=spy), \
         patch("agent.auxiliary_client._resolve_task_provider_model", side_effect=configured), \
         patch("agent.auxiliary_client._get_cached_client", side_effect=cached), \
         patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=chain), \
         patch("agent.auxiliary_client._try_main_fallback_chain", return_value=main), \
         patch("agent.auxiliary_client._try_payment_fallback", return_value=main), \
         patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=main):
        result = compressor._generate_summary(_two_chunk_source())
    assert result and compressor._previous_summary
    assert SNAPSHOT.strip() in compressor._previous_summary
    assert len(dispatches) >= 2, "the workload must exercise multiple real summary dispatches"
    destination = {"provider": "custom", "model": "backup/model" if chain_rescue else "main/model",
                   "base_url": BACKUP_URL if chain_rescue else MAIN_URL,
                   "api_key": "synthetic-backup-key" if chain_rescue else "synthetic-main-key"}
    assert dispatches[0]["provider"] is None, "no host pin is installed in this regression"
    assert dispatches[1:] == [destination] * (len(dispatches) - 1)
    assert len(primary_prompts) == 1
    assert len(successful_prompts) == len(dispatches)


def test_same_pair_rescue_off_the_main_host_never_returns_to_it():
    # Review N1: the failed primary IS the main runtime; the rescue serves the same pair on another host.
    main_prompts, dispatches = [], []

    def broken(**kwargs):
        main_prompts.append(kwargs["messages"][0]["content"])
        raise _payment_402()

    clients = {MAIN_URL: _client(MAIN_URL, MagicMock(side_effect=broken), "synthetic-main-key"),
               BACKUP_URL: _client(BACKUP_URL, MagicMock(side_effect=lambda **_kw: _ok(SNAPSHOT + "part")),
                                   "synthetic-backup-key")}

    def configured(task=None, provider=None, model=None, base_url=None, api_key=None, *_a, **_kw):
        return provider or "custom", model or "main/model", base_url or MAIN_URL, api_key, None

    def cached(provider, model=None, *_a, base_url=None, **_kw):
        return clients[str(base_url)], model

    real_call = ac.call_llm

    def spy(**kwargs):
        dispatches.append({k: kwargs.get(k) for k in ("provider", "model", "base_url", "api_key")})
        return real_call(**kwargs)

    rescue = clients[BACKUP_URL], "main/model", "fallback_providers[0](custom)"
    with patch("agent.context_compressor.call_llm", side_effect=spy), \
         patch("agent.auxiliary_client._resolve_task_provider_model", side_effect=configured), \
         patch("agent.auxiliary_client._get_cached_client", side_effect=cached), \
         patch("agent.auxiliary_client._try_configured_fallback_chain", return_value=(None, None, "")), \
         patch("agent.auxiliary_client._try_main_fallback_chain", return_value=rescue), \
         patch("agent.auxiliary_client._try_payment_fallback", return_value=rescue), \
         patch("agent.auxiliary_client._try_main_agent_model_fallback", return_value=rescue):
        assert _main_compressor()._generate_summary(_two_chunk_source())
    assert len(dispatches) >= 2
    assert len(main_prompts) == 1, "the host that answered 402 receives no later chunk"
    assert all((d["base_url"], d["api_key"]) == (BACKUP_URL, "synthetic-backup-key") for d in dispatches[1:]), dispatches


def test_unpinned_plan_without_a_hop_stays_unpinned():
    # Review N2: without a ladder hop nothing is pinned, so every chunk re-resolves its route and
    # credential (a refreshed token or rotated pool key is picked up, never a stale snapshot).
    dispatches = []
    client = _client(PIN_URL, MagicMock(return_value=_ok(SNAPSHOT + "checkpoint")))
    real_call = ac.call_llm

    def spy(**kwargs):
        dispatches.append({k: kwargs.get(k) for k in ("provider", "model", "base_url", "api_key")})
        return real_call(**kwargs)

    def configured(task=None, provider=None, model=None, base_url=None, api_key=None, *_a, **_kw):
        return provider or "custom", model or "pinned/model", base_url or PIN_URL, api_key, None

    with patch("agent.context_compressor.call_llm", side_effect=spy), \
         patch("agent.auxiliary_client._resolve_task_provider_model", side_effect=configured), \
         patch("agent.auxiliary_client._get_cached_client", side_effect=lambda _provider, model, **_kw: (client, model)):
        compressor = _main_compressor()
        assert compressor._generate_summary(_two_chunk_source())
    assert len(dispatches) >= 2
    assert all(d == {"provider": None, "model": None, "base_url": None, "api_key": None} for d in dispatches)


def test_route_record_exposes_only_the_pair_and_counts_records():
    record = _SummaryRouteRecord()
    ac._record_route_info(record, "custom", "primary/model")
    ac._record_route_info(record, "custom", "backup/model")
    assert dict(record) == {"provider": "custom", "model": "backup/model"}
    assert record.records == 2
