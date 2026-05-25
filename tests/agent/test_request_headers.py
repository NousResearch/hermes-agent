from hermes_cli.config import get_compatible_custom_providers

from agent.request_headers import (
    apply_configured_default_headers,
    normalize_config_headers,
)


def test_normalize_config_headers_skips_reserved_and_invalid_headers():
    headers = normalize_config_headers(
        {
            "comate_custom_header": '{"username":"chengbo05","source":"openclaw"}',
            "Authorization": "Bearer leaked",
            "Bad\nName": "x",
            "X-Number": 42,
        },
        source="model.headers",
    )

    assert headers == {
        "comate_custom_header": '{"username":"chengbo05","source":"openclaw"}',
        "X-Number": "42",
    }


def test_configured_headers_merge_custom_provider_then_model_headers():
    config = {
        "model": {
            "provider": "custom",
            "default": "GLM-5-Turbo",
            "base_url": "http://127.0.0.1:18707/v1",
            "headers": {
                "comate_custom_header": "from-model",
                "Authorization": "reserved",
            },
        },
        "custom_providers": [
            {
                "name": "bdllm",
                "base_url": "http://127.0.0.1:18707/v1",
                "model": "GLM-5-Turbo",
                "headers": {
                    "X-Provider": "yes",
                    "comate_custom_header": "from-provider",
                },
            },
        ],
    }

    headers = apply_configured_default_headers(
        {"User-Agent": "HermesAgent/test"},
        provider="custom",
        base_url="http://127.0.0.1:18707/v1",
        model="GLM-5-Turbo",
        config=config,
    )

    assert headers["User-Agent"] == "HermesAgent/test"
    assert headers["X-Provider"] == "yes"
    assert headers["comate_custom_header"] == "from-model"
    assert "Authorization" not in headers


def test_custom_provider_normalizer_preserves_headers_from_providers_dict():
    providers = get_compatible_custom_providers(
        {
            "providers": {
                "bdllm": {
                    "base_url": "http://127.0.0.1:18707/v1",
                    "api_key": "test-key",
                    "headers": {"comate_custom_header": "value"},
                }
            }
        }
    )

    assert providers[0]["headers"] == {"comate_custom_header": "value"}
