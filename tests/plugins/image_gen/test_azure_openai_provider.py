"""Focused tests for Azure OpenAI image settings resolution."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_PLUGIN_PATH = (
    Path(__file__).resolve().parents[3]
    / "plugins"
    / "image_gen"
    / "azure-openai"
    / "__init__.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "plugins.image_gen.azure_openai_test_plugin", _PLUGIN_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
azure_plugin = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = azure_plugin
_SPEC.loader.exec_module(azure_plugin)


def _config(endpoint="canonical-endpoint", deployment="canonical-deployment", **extra):
    azure = {
        "endpoint": endpoint,
        "deployment_name": deployment,
        **extra,
    }
    return {"image_gen": {"azure_openai": azure}}


def test_resolver_trims_values_and_prefers_canonical_direct_azure_config():
    config = _config(
        endpoint="  https://canonical.openai.azure.com/  ",
        deployment="  canonical-deployment  ",
        api_version="  2025-04-01-preview  ",
    )
    environ = {
        "AZURE_OPENAI_IMAGE_KEY": "  secret-key  ",
        "AZURE_OPENAI_ENDPOINT": "https://compat.openai.azure.com",
        "AZURE_IMAGE_DEPLOYMENT_NAME": "compat-deployment",
    }

    result = azure_plugin.resolve_azure_image_settings(config, environ)

    assert result == azure_plugin.AzureImageSettings(
        endpoint="https://canonical.openai.azure.com",
        api_key="secret-key",
        deployment_name="canonical-deployment",
        api_version="2025-04-01-preview",
        endpoint_family="azure-openai",
    )


def test_resolver_normalizes_foundry_v1_and_ignores_azure_rest_api_version():
    result = azure_plugin.resolve_azure_image_settings(
        _config(
            endpoint=" https://example-resource.services.ai.azure.com/openai/v1/ ",
            deployment=" test-image-deployment ",
            api_version="2099-01-01-test",
        ),
        {"AZURE_OPENAI_IMAGE_KEY": "key"},
    )

    assert result == azure_plugin.AzureImageSettings(
        endpoint="https://example-resource.services.ai.azure.com/openai/v1",
        api_key="key",
        deployment_name="test-image-deployment",
        api_version=None,
        endpoint_family="foundry-v1",
    )


def test_resolver_uses_trimmed_compatibility_fallbacks_for_blank_config():
    result = azure_plugin.resolve_azure_image_settings(
        _config(endpoint=" \t", deployment="\n"),
        {
            "AZURE_OPENAI_IMAGE_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "  https://compat.openai.azure.com/ ",
            "AZURE_IMAGE_DEPLOYMENT_NAME": " compat-deployment ",
        },
    )

    assert isinstance(result, azure_plugin.AzureImageSettings)
    assert result.endpoint == "https://compat.openai.azure.com"
    assert result.deployment_name == "compat-deployment"
    assert result.api_version == azure_plugin.DEFAULT_AZURE_IMAGE_API_VERSION
    assert result.endpoint_family == "azure-openai"


@pytest.mark.parametrize(
    ("config", "environ", "field", "error_type", "canonical_key"),
    [
        (
            _config(endpoint="https://resource.openai.azure.com"),
            {"AZURE_OPENAI_IMAGE_KEY": "  "},
            "api_key",
            "auth_required",
            None,
        ),
        (
            _config(endpoint=" "),
            {"AZURE_OPENAI_IMAGE_KEY": "secret", "AZURE_OPENAI_ENDPOINT": ""},
            "endpoint",
            "configuration_error",
            "image_gen.azure_openai.endpoint",
        ),
        (
            _config(deployment=None),
            {
                "AZURE_OPENAI_IMAGE_KEY": "secret",
                "AZURE_IMAGE_DEPLOYMENT_NAME": "\t",
            },
            "deployment_name",
            "configuration_error",
            "image_gen.azure_openai.deployment_name",
        ),
        (
            _config(endpoint="https://example.invalid/openai/v1"),
            {"AZURE_OPENAI_IMAGE_KEY": "secret"},
            "endpoint",
            "configuration_error",
            "image_gen.azure_openai.endpoint",
        ),
    ],
)
def test_resolver_returns_field_specific_safe_results(
    config, environ, field, error_type, canonical_key
):
    secret = environ.get("AZURE_OPENAI_IMAGE_KEY")

    result = azure_plugin.resolve_azure_image_settings(config, environ)

    assert isinstance(result, azure_plugin.AzureImageConfigurationError)
    assert result.field == field
    assert result.error_type == error_type
    assert result.canonical_key == canonical_key
    if secret and secret.strip():
        assert secret.strip() not in result.message
        assert secret.strip() not in (result.setup_action or "")


def test_foundry_settings_allow_explicit_keyless_entra_path():
    result = azure_plugin.resolve_azure_image_settings(
        _config(
            endpoint="https://example-resource.services.ai.azure.com/openai/v1",
            deployment="test-image-deployment",
        ),
        {},
    )

    assert isinstance(result, azure_plugin.AzureImageSettings)
    assert result.endpoint_family == "foundry-v1"
    assert result.api_key is None


def test_resolver_treats_non_mapping_and_non_string_values_as_missing():
    result = azure_plugin.resolve_azure_image_settings(
        {"image_gen": {"azure_openai": "invalid"}},
        {
            "AZURE_OPENAI_IMAGE_KEY": "key",
            "AZURE_OPENAI_ENDPOINT": "https://resource.openai.azure.com",
            "AZURE_IMAGE_DEPLOYMENT_NAME": 42,
        },
    )

    assert isinstance(result, azure_plugin.AzureImageConfigurationError)
    assert result.field == "deployment_name"


class TestAzureOpenAIProviderRegistration:
    def test_identity_setup_schema_and_text_only_capabilities(self):
        provider = azure_plugin.AzureOpenAIImageGenProvider()

        assert provider.name == "azure-openai"
        assert provider.display_name == "Azure OpenAI"

        schema = provider.get_setup_schema()
        assert schema["name"] == "Azure OpenAI"
        assert schema["badge"] == "paid"
        assert schema["env_vars"] == [
            {
                "key": "AZURE_OPENAI_IMAGE_KEY",
                "prompt": "Azure OpenAI image API key (optional with Foundry Entra ID)",
                "password": True,
                "required": False,
            }
        ]
        assert schema["config_fields"] == [
            {
                "key": "image_gen.azure_openai.endpoint",
                "prompt": "Azure endpoint (resource root or Foundry /openai/v1)",
                "required": True,
                "normalize": azure_plugin.normalize_azure_image_setup_endpoint,
            },
            {
                "key": "image_gen.azure_openai.deployment_name",
                "prompt": "Azure image deployment",
                "required": True,
            },
        ]
        assert provider.capabilities() == {
            "modalities": ["text"],
            "max_reference_images": 0,
        }

    @pytest.mark.parametrize(
        ("endpoint", "key", "expected"),
        [
            ("https://resource.openai.azure.com", "secret", True),
            ("https://resource.services.ai.azure.com/openai/v1", "secret", True),
            ("https://resource.openai.azure.com", "", False),
        ],
    )
    def test_availability_matches_endpoint_client_and_auth(
        self, monkeypatch, endpoint, key, expected
    ):
        import hermes_cli.config as config_module

        class OpenAIModule:
            AzureOpenAI = object
            OpenAI = object

        monkeypatch.setitem(sys.modules, "openai", OpenAIModule())
        monkeypatch.setenv("AZURE_OPENAI_IMAGE_KEY", key)
        monkeypatch.setattr(
            config_module,
            "load_config",
            lambda: _config(endpoint=endpoint, deployment="image-deployment"),
        )

        assert azure_plugin.AzureOpenAIImageGenProvider().is_available() is expected

    def test_register_adds_exactly_one_provider(self):
        registered = []

        class Context:
            def register_image_gen_provider(self, provider):
                registered.append(provider)

        azure_plugin.register(Context())

        assert len(registered) == 1
        assert isinstance(registered[0], azure_plugin.AzureOpenAIImageGenProvider)
        assert registered[0].name == "azure-openai"


class TestAzureOpenAIRequestConstruction:
    @staticmethod
    def _install_fake_sdk(monkeypatch):
        calls = {"constructors": [], "requests": []}

        class Images:
            def generate(self, **kwargs):
                calls["requests"].append(kwargs)
                return object()

        class AzureOpenAI:
            def __init__(self, **kwargs):
                calls["constructors"].append(("AzureOpenAI", kwargs))
                self.images = Images()

        class OpenAI:
            def __init__(self, **kwargs):
                calls["constructors"].append(("OpenAI", kwargs))
                self.images = Images()

        class OpenAIModule:
            pass

        OpenAIModule.AzureOpenAI = AzureOpenAI
        OpenAIModule.OpenAI = OpenAI
        monkeypatch.setitem(sys.modules, "openai", OpenAIModule())
        return calls

    @staticmethod
    def _configure(
        monkeypatch,
        *,
        endpoint="https://resource.openai.azure.com",
        key=" resolved-key ",
        api_version="2024-10-21",
    ):
        import hermes_cli.config as config_module

        if key is None:
            monkeypatch.delenv("AZURE_OPENAI_IMAGE_KEY", raising=False)
        else:
            monkeypatch.setenv("AZURE_OPENAI_IMAGE_KEY", key)
        monkeypatch.setattr(
            config_module,
            "load_config",
            lambda: _config(
                endpoint=f" {endpoint} ",
                deployment=" azure-image-deployment ",
                api_version=f" {api_version} ",
            ),
        )

    def test_direct_azure_constructs_native_client_and_exact_request(
        self, monkeypatch
    ):
        calls = self._install_fake_sdk(monkeypatch)
        self._configure(monkeypatch, api_version="2025-04-01-preview")

        azure_plugin.AzureOpenAIImageGenProvider().generate(
            "  paint a moonlit harbor  ",
            aspect_ratio="square",
            model="generic-model-must-be-ignored",
            quality="unverified-param-must-not-be-forwarded",
        )

        assert calls["constructors"] == [
            (
                "AzureOpenAI",
                {
                    "api_key": "resolved-key",
                    "azure_endpoint": "https://resource.openai.azure.com",
                    "api_version": "2025-04-01-preview",
                },
            )
        ]
        assert calls["requests"] == [
            {
                "model": "azure-image-deployment",
                "prompt": "paint a moonlit harbor",
                "size": "1024x1024",
                "n": 1,
            }
        ]

    def test_foundry_constructs_openai_client_without_azure_rest_arguments(
        self, monkeypatch
    ):
        calls = self._install_fake_sdk(monkeypatch)
        self._configure(
            monkeypatch,
            endpoint="https://example-resource.services.ai.azure.com/openai/v1/",
            api_version="2099-01-01-test",
        )

        azure_plugin.AzureOpenAIImageGenProvider().generate(
            "  paint a moonlit harbor  ", aspect_ratio="square"
        )

        assert calls["constructors"] == [
            (
                "OpenAI",
                {
                    "base_url": (
                        "https://example-resource.services.ai.azure.com/openai/v1"
                    ),
                    "api_key": "resolved-key",
                },
            )
        ]
        assert calls["requests"] == [
            {
                "model": "azure-image-deployment",
                "prompt": "paint a moonlit harbor",
                "size": "1024x1024",
                "n": 1,
            }
        ]

    def test_foundry_keyless_uses_shared_bearer_token_provider(self, monkeypatch):
        from agent import azure_identity_adapter

        calls = self._install_fake_sdk(monkeypatch)
        token_provider = lambda: "token"  # noqa: E731 - SDK contract fixture
        scopes = []
        monkeypatch.setattr(
            azure_identity_adapter, "has_azure_identity_installed", lambda: True
        )
        monkeypatch.setattr(
            azure_identity_adapter,
            "build_token_provider",
            lambda *, scope: scopes.append(scope) or token_provider,
        )
        self._configure(
            monkeypatch,
            endpoint="https://resource.services.ai.azure.com/openai/v1",
            key=None,
        )

        azure_plugin.AzureOpenAIImageGenProvider().generate("draw a fox")

        assert scopes == ["https://ai.azure.com/.default"]
        assert calls["constructors"] == [
            (
                "OpenAI",
                {
                    "base_url": "https://resource.services.ai.azure.com/openai/v1",
                    "api_key": token_provider,
                },
            )
        ]

    def test_foundry_keyless_without_identity_is_actionable(self, monkeypatch):
        from agent import azure_identity_adapter

        calls = self._install_fake_sdk(monkeypatch)
        monkeypatch.setattr(
            azure_identity_adapter, "has_azure_identity_installed", lambda: False
        )
        self._configure(
            monkeypatch,
            endpoint="https://resource.services.ai.azure.com/openai/v1",
            key=None,
        )

        result = azure_plugin.AzureOpenAIImageGenProvider().generate("draw a fox")

        assert result["success"] is False
        assert result["error_type"] == "auth_required"
        assert "AZURE_OPENAI_IMAGE_KEY" in result["setup_action"]
        assert "azure-identity" in result["setup_action"]
        assert calls == {"constructors": [], "requests": []}

    @pytest.mark.parametrize(
        ("aspect_ratio", "expected_size"),
        [
            ("landscape", "1536x1024"),
            ("square", "1024x1024"),
            ("portrait", "1024x1536"),
            ("unsupported", "1536x1024"),
        ],
    )
    def test_maps_supported_sizes_and_defaults_invalid_aspects(
        self, monkeypatch, aspect_ratio, expected_size
    ):
        calls = self._install_fake_sdk(monkeypatch)
        self._configure(monkeypatch)

        azure_plugin.AzureOpenAIImageGenProvider().generate(
            "prompt", aspect_ratio=aspect_ratio
        )

        assert calls["requests"][0]["size"] == expected_size

    @pytest.mark.parametrize(
        ("image_url", "reference_image_urls"),
        [
            ("https://example.test/source.png", None),
            (None, ["https://example.test/reference.png"]),
        ],
    )
    def test_rejects_source_images_before_client_construction(
        self, monkeypatch, image_url, reference_image_urls
    ):
        calls = self._install_fake_sdk(monkeypatch)

        result = azure_plugin.AzureOpenAIImageGenProvider().generate(
            "  edit this image  ",
            image_url=image_url,
            reference_image_urls=reference_image_urls,
        )

        assert result["success"] is False
        assert result["error_type"] == "modality_unsupported"
        assert result["prompt"] == "edit this image"
        assert calls == {"constructors": [], "requests": []}


class TestAzureOpenAIResponseMaterialization:
    @staticmethod
    def _response(*, b64=None, url=None, revised_prompt=None, include_data=True):
        if not include_data:
            return type("Response", (), {"data": []})()
        image = type(
            "GeneratedImage",
            (),
            {"b64_json": b64, "url": url, "revised_prompt": revised_prompt},
        )()
        return type("Response", (), {"data": [image]})()

    @staticmethod
    def _install_fake_sdk(monkeypatch, response):
        class Images:
            def generate(self, **kwargs):
                return response

        class AzureOpenAI:
            def __init__(self, **kwargs):
                self.images = Images()

        class OpenAIModule:
            pass

        OpenAIModule.AzureOpenAI = AzureOpenAI
        monkeypatch.setitem(sys.modules, "openai", OpenAIModule())

    @staticmethod
    def _configure(monkeypatch):
        import hermes_cli.config as config_module

        monkeypatch.setenv("AZURE_OPENAI_IMAGE_KEY", "secret")
        monkeypatch.setattr(
            config_module,
            "load_config",
            lambda: _config(
                endpoint="https://resource.openai.azure.com",
                deployment="azure-image-deployment",
            ),
        )

    def test_base64_output_is_saved_under_profile_cache_with_response_identity(
        self, monkeypatch
    ):
        from hermes_constants import get_hermes_home

        self._install_fake_sdk(
            monkeypatch,
            self._response(b64="aGVybWVz", revised_prompt="a refined prompt"),
        )
        self._configure(monkeypatch)

        result = azure_plugin.AzureOpenAIImageGenProvider().generate(
            "  draw a lighthouse  ", aspect_ratio="portrait"
        )

        image_path = Path(result["image"])
        assert result == {
            "success": True,
            "image": str(image_path),
            "model": "azure-image-deployment",
            "prompt": "draw a lighthouse",
            "aspect_ratio": "portrait",
            "modality": "text",
            "provider": "azure-openai",
            "size": "1024x1536",
            "revised_prompt": "a refined prompt",
        }
        assert image_path.parent == get_hermes_home() / "cache" / "images"
        assert image_path.read_bytes() == b"hermes"

    def test_url_output_uses_shared_persistence_helper(self, monkeypatch, tmp_path):
        import agent.image_gen_provider as image_helpers

        original_url = "https://example.test/generated.png"
        cached_path = tmp_path / "cached.png"
        calls = []

        def save_url(url, **kwargs):
            calls.append((url, kwargs))
            return cached_path

        monkeypatch.setattr(image_helpers, "save_url_image", save_url)
        self._install_fake_sdk(monkeypatch, self._response(url=original_url))
        self._configure(monkeypatch)

        result = azure_plugin.AzureOpenAIImageGenProvider().generate("draw a fox")

        assert result["success"] is True
        assert result["image"] == str(cached_path)
        assert calls == [(original_url, {"prefix": "azure_openai"})]

    def test_url_cache_failure_preserves_original_url(self, monkeypatch):
        import agent.image_gen_provider as image_helpers

        original_url = "https://example.test/ephemeral.png"

        def fail_cache(*args, **kwargs):
            raise OSError("cache unavailable")

        monkeypatch.setattr(image_helpers, "save_url_image", fail_cache)
        self._install_fake_sdk(monkeypatch, self._response(url=original_url))
        self._configure(monkeypatch)

        result = azure_plugin.AzureOpenAIImageGenProvider().generate("draw a fox")

        assert result["success"] is True
        assert result["image"] == original_url

    @pytest.mark.parametrize(
        "response",
        [
            _response.__func__(include_data=False),
            _response.__func__(b64="%%%not-base64%%%"),
        ],
    )
    def test_absent_or_invalid_output_returns_empty_response(
        self, monkeypatch, response
    ):
        self._install_fake_sdk(monkeypatch, response)
        self._configure(monkeypatch)

        result = azure_plugin.AzureOpenAIImageGenProvider().generate("draw a fox")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"
        assert result["model"] == "azure-image-deployment"

    def test_base64_cache_failure_returns_sanitized_io_error(self, monkeypatch):
        import agent.image_gen_provider as image_helpers

        key = "cache-failure-azure-key-7Yq2"

        def fail_cache(*args, **kwargs):
            raise OSError(f"disk full while handling credential {key}")

        monkeypatch.setattr(image_helpers, "save_b64_image", fail_cache)
        self._install_fake_sdk(monkeypatch, self._response(b64="aGVybWVz"))
        self._configure(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_IMAGE_KEY", key)

        result = azure_plugin.AzureOpenAIImageGenProvider().generate("draw a fox")

        assert result["success"] is False
        assert result["error_type"] == "io_error"
        assert result["model"] == "azure-image-deployment"
        assert "[REDACTED]" in result["error"]
        assert key not in result["error"]


class TestAzureOpenAIFailureHandling:
    @staticmethod
    def _configure(monkeypatch, *, key, endpoint, deployment):
        import hermes_cli.config as config_module

        monkeypatch.delenv("AZURE_OPENAI_IMAGE_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_IMAGE_DEPLOYMENT_NAME", raising=False)
        if key is not None:
            monkeypatch.setenv("AZURE_OPENAI_IMAGE_KEY", key)
        monkeypatch.setattr(
            config_module,
            "load_config",
            lambda: _config(endpoint=endpoint, deployment=deployment),
        )

    @pytest.mark.parametrize(
        (
            "key",
            "endpoint",
            "deployment",
            "error_type",
            "setup_target",
            "config_key",
        ),
        [
            (
                None,
                "https://resource.openai.azure.com",
                "image-deployment",
                "auth_required",
                "AZURE_OPENAI_IMAGE_KEY",
                None,
            ),
            (
                "azure-secret",
                None,
                "image-deployment",
                "configuration_error",
                "image_gen.azure_openai.endpoint",
                "image_gen.azure_openai.endpoint",
            ),
            (
                "azure-secret",
                "https://resource.openai.azure.com",
                None,
                "configuration_error",
                "image_gen.azure_openai.deployment_name",
                "image_gen.azure_openai.deployment_name",
            ),
        ],
    )
    def test_missing_settings_return_actionable_standard_errors(
        self,
        monkeypatch,
        key,
        endpoint,
        deployment,
        error_type,
        setup_target,
        config_key,
    ):
        self._configure(
            monkeypatch,
            key=key,
            endpoint=endpoint,
            deployment=deployment,
        )

        result = azure_plugin.AzureOpenAIImageGenProvider().generate("draw a fox")

        assert result["success"] is False
        assert result["error_type"] == error_type
        assert setup_target in result["error"]
        assert setup_target in result["setup_action"]
        if config_key is None:
            assert "config_key" not in result
        else:
            assert result["config_key"] == config_key

    @pytest.mark.parametrize("failure_stage", ["construction", "generation"])
    def test_sdk_failures_are_api_errors_with_key_redacted_from_response_and_logs(
        self, monkeypatch, caplog, failure_stage
    ):
        key = "arbitrary-azure-image-key-9Zx7"
        self._configure(
            monkeypatch,
            key=key,
            endpoint="https://resource.openai.azure.com",
            deployment="image-deployment",
        )

        class Images:
            def generate(self, **kwargs):
                raise RuntimeError(f"Azure service echoed credential {key}")

        class AzureOpenAI:
            def __init__(self, **kwargs):
                if failure_stage == "construction":
                    raise RuntimeError(f"Azure client rejected credential {key}")
                self.images = Images()

        class OpenAIModule:
            pass

        OpenAIModule.AzureOpenAI = AzureOpenAI
        monkeypatch.setitem(sys.modules, "openai", OpenAIModule())

        with caplog.at_level("ERROR", logger=azure_plugin.__name__):
            result = azure_plugin.AzureOpenAIImageGenProvider().generate("draw a fox")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert result["model"] == "image-deployment"
        assert "[REDACTED]" in result["error"]
        assert key not in result["error"]
        assert key not in caplog.text
        assert "[REDACTED]" in caplog.text