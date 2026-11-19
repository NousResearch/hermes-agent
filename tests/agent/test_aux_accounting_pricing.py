"""Auxiliary cost estimation remains best-effort across a config reload failure."""

from types import SimpleNamespace

from agent import aux_accounting, usage_pricing


def test_aux_cost_survives_config_reload_failure(monkeypatch):
    """A live auxiliary response must keep its known price if config becomes unreadable."""
    def unavailable_config():
        raise OSError("config temporarily unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config", unavailable_config)
    calls = []

    def estimate(model, usage, **kwargs):
        calls.append((model, usage, kwargs))
        return SimpleNamespace(amount_usd=0.125)

    monkeypatch.setattr(usage_pricing, "estimate_usage_cost", estimate)
    rows = []
    db = SimpleNamespace(record_auxiliary_usage=lambda *args, **kwargs: rows.append(kwargs))
    token = aux_accounting.set_accounting_context(db, "session-1")
    try:
        response = SimpleNamespace(
            model="local-model",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13),
        )
        aux_accounting.record_aux_usage(
            response, "title_generation", provider="custom",
            base_url="http://127.0.0.1:47970/v1",
        )
    finally:
        aux_accounting.reset_accounting_context(token)

    assert len(calls) == 1
    assert calls[0][0] == "local-model"
    assert calls[0][2] == {
        "provider": "custom", "base_url": "http://127.0.0.1:47970/v1", "api_key": None,
    }
    assert len(rows) == 1
    assert rows[0]["estimated_cost_usd"] == 0.125


def test_aux_cost_survives_an_unavailable_config_backend(monkeypatch):
    """The config backend's outage is a SystemExit subtype; the optional pricing-key lookup must not
    let it abort a successful auxiliary call or drop the row."""
    from hermes_cli.config_backend import ConfigBackendUnavailable

    def unavailable_config():
        raise ConfigBackendUnavailable("remote config unavailable")

    monkeypatch.setattr("hermes_cli.config.load_config", unavailable_config)
    calls = []
    monkeypatch.setattr(usage_pricing, "estimate_usage_cost",
                        lambda model, usage, **kw: calls.append(kw) or SimpleNamespace(amount_usd=0.125))
    rows = []
    db = SimpleNamespace(record_auxiliary_usage=lambda *args, **kwargs: rows.append(kwargs))
    token = aux_accounting.set_accounting_context(db, "session-1")
    try:
        response = SimpleNamespace(model="local-model", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3))
        aux_accounting.record_aux_usage(response, "title_generation", provider="custom",
                                        base_url="http://127.0.0.1:47970/v1")
    finally:
        aux_accounting.reset_accounting_context(token)

    assert [c["api_key"] for c in calls] == [None]
    assert [r["estimated_cost_usd"] for r in rows] == [0.125]


def _serving_vs_saved_route(monkeypatch, *, async_mode):
    """An aux call served at the configured main endpoint by a client whose static key differs from the
    saved one; that endpoint's /models prices the model only for the serving key."""
    from unittest.mock import AsyncMock, MagicMock

    from hermes_cli.config import save_config

    base = "http://127.0.0.1:47970/v1"
    monkeypatch.setenv("LAB_LLAMA_KEY", "saved-key-1111")
    save_config({"model": {"default": "qwen-lab", "provider": "custom", "base_url": base, "key_env": "LAB_LLAMA_KEY"}})
    response = SimpleNamespace(model="qwen-lab", choices=[SimpleNamespace(message=SimpleNamespace(content="t"))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13))
    client = MagicMock()
    client.base_url, client.api_key = base + "/", "serving-key-2222"
    client.chat.completions.create = AsyncMock(return_value=response) if async_mode else MagicMock(return_value=response)
    monkeypatch.setattr("agent.auxiliary_client._resolve_task_provider_model",
                        lambda *a, **k: ("custom", "qwen-lab", base, None, None))
    monkeypatch.setattr("agent.auxiliary_client._get_cached_client", lambda *a, **k: (client, "qwen-lab"))
    keys = []
    monkeypatch.setattr(usage_pricing, "estimate_usage_cost", lambda model, usage, **kw: keys.append(kw["api_key"]) or
                        SimpleNamespace(amount_usd=0.125 if kw["api_key"] == "serving-key-2222" else None))
    rows = []
    db = SimpleNamespace(record_auxiliary_usage=lambda *args, **kwargs: rows.append(kwargs))
    return aux_accounting.set_accounting_context(db, "session-1"), rows, keys


def test_sync_aux_pricing_uses_the_key_that_served_the_call(monkeypatch):
    from agent.auxiliary_client import call_llm

    token, rows, keys = _serving_vs_saved_route(monkeypatch, async_mode=False)
    try:
        call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])
    finally:
        aux_accounting.reset_accounting_context(token)
    assert keys == ["serving-key-2222"]
    assert [r["estimated_cost_usd"] for r in rows] == [0.125]
    assert "serving-key-2222" not in repr(rows)


def test_async_aux_pricing_uses_the_key_that_served_the_call(monkeypatch):
    import asyncio

    from agent.auxiliary_client import async_call_llm

    token, rows, keys = _serving_vs_saved_route(monkeypatch, async_mode=True)
    try:
        asyncio.run(async_call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}]))
    finally:
        aux_accounting.reset_accounting_context(token)
    assert keys == ["serving-key-2222"]
    assert [r["estimated_cost_usd"] for r in rows] == [0.125]


def test_aux_pricing_mints_nothing_for_a_credential_source_client(monkeypatch):
    """A client authenticating with a token source served the call: pricing stays keyless rather
    than minting a token or sending the saved key."""
    from openai import OpenAI

    minted = []
    client = OpenAI(api_key=lambda: minted.append(1) or "minted", base_url="http://127.0.0.1:47970/v1")
    monkeypatch.setenv("LAB_LLAMA_KEY", "saved-key-1111")
    from hermes_cli.config import save_config
    save_config({"model": {"default": "m", "provider": "custom", "base_url": "http://127.0.0.1:47970/v1",
                           "key_env": "LAB_LLAMA_KEY"}})
    keys = []
    monkeypatch.setattr(usage_pricing, "estimate_usage_cost", lambda model, usage, **kw: keys.append(kw["api_key"]) or
                        SimpleNamespace(amount_usd=None))
    token = aux_accounting.set_accounting_context(SimpleNamespace(record_auxiliary_usage=lambda *a, **k: None), "s")
    try:
        aux_accounting.record_aux_usage(
            SimpleNamespace(model="m", usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)), "title_generation",
            provider="custom", base_url=str(client.base_url), serving=aux_accounting.ServingCredential.of(client))
    finally:
        aux_accounting.reset_accounting_context(token)
    assert keys == [None] and minted == []


def test_a_serving_credential_never_shows_its_key_in_repr():
    cred = aux_accounting.ServingCredential.of(SimpleNamespace(base_url="http://127.0.0.1:47970/v1",
                                                                api_key="serving-key-2222"))
    assert cred.api_key == "serving-key-2222"
    assert "serving-key-2222" not in repr(cred) and "serving-key-2222" not in str(cred)


def _transient_error():
    import httpx
    import openai
    return openai.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:47970/v1/chat/completions"))


def test_sync_aux_pricing_after_a_same_provider_retry_uses_the_serving_key(monkeypatch):
    """The transient retry reuses the same client at the same URL; its success prices like the first try."""
    from agent import auxiliary_client
    from agent.auxiliary_client import call_llm

    token, rows, keys = _serving_vs_saved_route(monkeypatch, async_mode=False)
    client, _ = auxiliary_client._get_cached_client()
    client.chat.completions.create.side_effect = [_transient_error(), client.chat.completions.create.return_value]
    monkeypatch.setattr(auxiliary_client, "_TRANSIENT_RETRY_BACKOFF_BASE", 0.0)
    try:
        call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])
    finally:
        aux_accounting.reset_accounting_context(token)
    assert client.chat.completions.create.call_count == 2
    assert keys == ["serving-key-2222"]
    assert [r["estimated_cost_usd"] for r in rows] == [0.125]


def test_async_aux_pricing_after_a_same_provider_retry_uses_the_serving_key(monkeypatch):
    import asyncio

    from agent import auxiliary_client
    from agent.auxiliary_client import async_call_llm

    token, rows, keys = _serving_vs_saved_route(monkeypatch, async_mode=True)
    client, _ = auxiliary_client._get_cached_client()
    client.chat.completions.create.side_effect = [_transient_error(), client.chat.completions.create.return_value]
    try:
        asyncio.run(async_call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}]))
    finally:
        aux_accounting.reset_accounting_context(token)
    assert client.chat.completions.create.call_count == 2
    assert keys == ["serving-key-2222"]
    assert [r["estimated_cost_usd"] for r in rows] == [0.125]
