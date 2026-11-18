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
