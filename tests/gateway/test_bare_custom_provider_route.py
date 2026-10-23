"""Bare ``custom`` session routes must not lose the provider name.

A named custom endpoint reports the billing class ``custom`` at runtime. Persisting
that class verbatim makes the next turn call ``resolve_runtime_provider("custom")``,
which has no endpoint and no API key and raises
``provider 'custom' resolved without credentials``.

The identity therefore has to be recovered from ``(base_url, model)``: the endpoint
alone is ambiguous as soon as several entries share one gateway, which is the normal
shape for a relay that serves several model families under one base URL.
"""
from hermes_cli import runtime_provider as rp
from hermes_cli import runtime_provider_custom as rpc
from hermes_state_sessions import SessionSessionsMixin

BASE = "https://gateway.example.test/v1"


def _config():
    return {
        "custom_providers": [
            {
                "name": "hub-gpt",
                "base_url": BASE,
                "model": "gpt-x",
                "api_mode": "codex_responses",
                "models": {"gpt-x": {}},
            },
            {
                "name": "hub-deepseek",
                "base_url": BASE,
                "model": "deepseek-x",
                "models": {"deepseek-x": {}, "deepseek-pro": {}},
            },
        ]
    }


class _ConfigStub:
    def __init__(self, config):
        self._config = config

    def load_config(self):
        return self._config

    def get_compatible_custom_providers(self, config=None):
        return (config or self._config).get("custom_providers") or []


def _install(monkeypatch, config=None):
    stub = _ConfigStub(config or _config())
    monkeypatch.setattr(rp, "load_config", stub.load_config)
    monkeypatch.setattr(rp, "get_compatible_custom_providers", stub.get_compatible_custom_providers)
    return stub


def test_endpoint_alone_is_ambiguous(monkeypatch):
    """Two entries share one base URL, so the endpoint-only lookup cannot be unique."""
    _install(monkeypatch)
    assert rpc.find_custom_provider_identity(BASE) == "custom:hub-gpt"


def test_model_disambiguates_a_shared_endpoint(monkeypatch):
    _install(monkeypatch)
    assert rpc.find_custom_provider_identity_for(BASE, "deepseek-x") == "custom:hub-deepseek"
    assert rpc.find_custom_provider_identity_for(BASE, "gpt-x") == "custom:hub-gpt"


def test_canonical_identity_prefers_the_endpoint_model_pair(monkeypatch):
    """The model must win over the first sibling the endpoint match would return."""
    _install(monkeypatch)
    assert rpc.canonical_custom_identity(base_url=BASE, model="deepseek-pro") == "custom:hub-deepseek"


def test_canonical_identity_falls_back_to_the_endpoint(monkeypatch):
    _install(monkeypatch)
    assert rpc.canonical_custom_identity(base_url=BASE) == "custom:hub-gpt"


def test_update_session_model_never_persists_bare_custom(monkeypatch):
    """Without a recovered name the stored route must stay untouched."""
    db = object.__new__(SessionSessionsMixin)
    monkeypatch.setattr(db, "flush_token_counts", lambda: None, raising=False)
    captured = {}

    def _capture(session_id, patch, sql=None, params=None):
        captured.update(patch)

    monkeypatch.setattr(db, "_write_model_config_patch", _capture, raising=False)
    db.update_session_model("sid", "deepseek-x", provider="custom", base_url=BASE)
    assert "gateway_runtime" not in captured
    assert "provider" not in captured
    assert captured.get("model") == "deepseek-x"


def test_update_session_model_keeps_a_recovered_name(monkeypatch):
    db = object.__new__(SessionSessionsMixin)
    monkeypatch.setattr(db, "flush_token_counts", lambda: None, raising=False)
    captured = {}

    def _capture(session_id, patch, sql=None, params=None):
        captured.update(patch)

    monkeypatch.setattr(db, "_write_model_config_patch", _capture, raising=False)
    db.update_session_model("sid", "deepseek-x", provider="custom:hub-deepseek", base_url=BASE,
                            api_mode="chat_completions")
    assert captured["provider"] == "custom:hub-deepseek"
    assert captured["gateway_runtime"]["provider"] == "custom:hub-deepseek"
    assert captured["gateway_runtime"]["base_url"] == BASE
