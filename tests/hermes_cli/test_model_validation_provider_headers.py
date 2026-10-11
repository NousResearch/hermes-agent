"""Model validation keeps provider headers at the real HTTP request boundary (#57667)."""
import pytest
from hermes_cli.model_switch import switch_model

@pytest.mark.parametrize("alias_url, same_endpoint", [
    ("https://PROXY.example.com:443/v1/", True),
    ("https://proxy.example.com/v1", True),
    ("https://proxy.example.com/V1", False),
    ("https://proxy.example.com/v1?tenant=beta", False),
])
def test_direct_alias_keeps_auth_for_equivalent_endpoint(monkeypatch, alias_url, same_endpoint):
    """Equivalent aliases preserve tenant headers and Anthropic auth mode."""
    from hermes_cli.model_switch import DirectAlias
    import hermes_cli.model_switch as model_switch_mod

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"data":[{"id":"aliased-model"}]}'

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        return FakeResponse()

    monkeypatch.setattr(
        model_switch_mod,
        "DIRECT_ALIASES",
        {
            "equivalent": DirectAlias(
                "aliased-model",
                "tenant-proxy",
                alias_url,
            )
        },
    )
    monkeypatch.setattr(
        model_switch_mod,
        "resolve_alias",
        lambda *a, **k: ("tenant-proxy", "aliased-model", "equivalent"),
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda *a, **k: {
            "api_key": "proxy-key",
            "base_url": "https://proxy.example.com/v1",
            "api_mode": "anthropic_messages",
            "extra_headers": {"X-Tenant": "alpha"},
        },
    )
    monkeypatch.setattr("hermes_cli.models._urlopen_model_catalog_request", fake_urlopen)
    monkeypatch.setattr("hermes_cli.models.detect_provider_for_model", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)

    result = switch_model(
        raw_input="equivalent",
        current_provider="tenant-proxy",
        current_model="old-model",
        current_base_url="https://proxy.example.com/v1",
        current_api_key="proxy-key",
    )

    assert result.success is True
    if same_endpoint:
        assert captured["request"].get_header("X-tenant") == "alpha"
        assert captured["request"].get_header("X-api-key") == "proxy-key"
        assert captured["request"].get_header("Anthropic-version") == "2023-06-01"
        assert captured["request"].get_header("Authorization") is None
    else:
        # Same origin may keep the key under upstream's policy, but path/query-specific
        # tenant headers and the old wire protocol must not follow a different endpoint.
        assert captured["request"].get_header("X-tenant") is None
        assert captured["request"].get_header("X-api-key") is None
        assert result.api_mode != "anthropic_messages"


@pytest.mark.parametrize("mode", ["chat_completions", "anthropic_messages"])
def test_generic_validation_forwards_headers_without_an_alias(monkeypatch, mode):
    from hermes_cli.models_validate import validate_requested_model
    captured = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"data":[{"id":"tenant-model"}]}'

    def open_request(request, *, timeout):
        captured.append(request)
        return Response()

    monkeypatch.setattr("hermes_cli.models._urlopen_model_catalog_request", open_request)
    result = validate_requested_model(
        "tenant-model", "tenant-proxy", api_key="local-key",
        base_url="https://proxy.example.com/v1", api_mode=mode,
        headers={"X-Tenant": "alpha"},
    )
    assert result["accepted"]
    assert captured[-1].get_header("X-tenant") == "alpha"
    if mode == "anthropic_messages":
        assert captured[-1].get_header("X-api-key") == "local-key"
        assert captured[-1].get_header("Authorization") is None
    else:
        assert captured[-1].get_header("Authorization") == "Bearer local-key"
