"""Tests for plugin image_gen providers injecting themselves into the picker.

Covers `_plugin_image_gen_providers`, `_visible_providers`, and
`_toolset_needs_configuration_prompt` handling of plugin providers.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


class _FakeProvider(ImageGenProvider):
    def __init__(self, name: str, available: bool = True, schema=None, models=None):
        self._name = name
        self._available = available
        self._schema = schema or {
            "name": name.title(),
            "badge": "test",
            "tag": f"{name} test tag",
            "env_vars": [{"key": f"{name.upper()}_API_KEY", "prompt": f"{name} key"}],
        }
        self._models = models or [
            {"id": f"{name}-model-v1", "display": f"{name} v1",
             "speed": "~5s", "strengths": "test", "price": "$"},
        ]

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return self._available

    def list_models(self):
        return list(self._models)

    def default_model(self):
        return self._models[0]["id"] if self._models else None

    def get_setup_schema(self):
        return dict(self._schema)

    def generate(self, prompt, aspect_ratio="landscape", **kw):
        return {"success": True, "image": f"{self._name}://{prompt}"}


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


class TestPluginPickerInjection:
    def test_plugin_providers_returns_registered(self, monkeypatch):
        from hermes_cli import tools_config

        image_gen_registry.register_provider(_FakeProvider("myimg"))

        rows = tools_config._plugin_image_gen_providers()
        names = [r["name"] for r in rows]
        plugin_names = [r.get("image_gen_plugin_name") for r in rows]

        assert "Myimg" in names
        assert "myimg" in plugin_names

    def test_fal_surfaced_alongside_other_plugins(self, monkeypatch):
        from hermes_cli import tools_config

        # After #26241, FAL is itself a plugin (`plugins/image_gen/fal/`)
        # and the hardcoded `TOOL_CATEGORIES["image_gen"]` FAL row is
        # gone. The plugin-row builder therefore surfaces it like any
        # other backend — no deduplication step needed.
        image_gen_registry.register_provider(_FakeProvider("fal"))
        image_gen_registry.register_provider(_FakeProvider("openai"))

        rows = tools_config._plugin_image_gen_providers()
        names = [r.get("image_gen_plugin_name") for r in rows]
        assert "fal" in names
        assert "openai" in names

    def test_visible_providers_includes_plugins_for_image_gen(self, monkeypatch):
        from hermes_cli import tools_config

        image_gen_registry.register_provider(_FakeProvider("someimg"))

        cat = tools_config.TOOL_CATEGORIES["image_gen"]
        visible = tools_config._visible_providers(cat, {})
        plugin_names = [p.get("image_gen_plugin_name") for p in visible if p.get("image_gen_plugin_name")]
        assert "someimg" in plugin_names

    def test_visible_providers_does_not_inject_into_other_categories(self, monkeypatch):
        from hermes_cli import tools_config

        image_gen_registry.register_provider(_FakeProvider("someimg"))

        # Browser category must NOT see image_gen plugins.
        browser = tools_config.TOOL_CATEGORIES["browser"]
        visible = tools_config._visible_providers(browser, {})
        assert all(p.get("image_gen_plugin_name") is None for p in visible)

    def test_post_setup_propagated_when_declared(self, monkeypatch):
        from hermes_cli import tools_config

        image_gen_registry.register_provider(_FakeProvider(
            "xai_img",
            schema={
                "name": "xAI Grok Imagine",
                "badge": "paid",
                "tag": "grok image",
                "env_vars": [],
                "post_setup": "xai_grok",
            },
        ))

        rows = tools_config._plugin_image_gen_providers()
        match = next(r for r in rows if r.get("image_gen_plugin_name") == "xai_img")
        assert match["post_setup"] == "xai_grok"

    def test_post_setup_omitted_when_not_declared(self, monkeypatch):
        from hermes_cli import tools_config

        image_gen_registry.register_provider(_FakeProvider("plain_img"))

        rows = tools_config._plugin_image_gen_providers()
        match = next(r for r in rows if r.get("image_gen_plugin_name") == "plain_img")
        assert "post_setup" not in match


class TestPluginCatalog:
    def test_plugin_catalog_returns_models(self):
        from hermes_cli import tools_config

        image_gen_registry.register_provider(_FakeProvider("catimg"))

        catalog, default = tools_config._plugin_image_gen_catalog("catimg")
        assert "catimg-model-v1" in catalog
        assert default == "catimg-model-v1"

    def test_plugin_catalog_empty_for_unknown(self):
        from hermes_cli import tools_config

        catalog, default = tools_config._plugin_image_gen_catalog("does-not-exist")
        assert catalog == {}
        assert default is None


class TestConfigPrompt:
    def test_image_gen_satisfied_by_plugin_provider(self, monkeypatch, tmp_path):
        """When a plugin provider reports is_available(), the picker should
        not force a setup prompt on the user."""
        from hermes_cli import tools_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("FAL_KEY", raising=False)

        image_gen_registry.register_provider(_FakeProvider("avail-img", available=True))

        assert tools_config._toolset_needs_configuration_prompt("image_gen", {}) is False

    def test_image_gen_still_prompts_when_nothing_available(self, monkeypatch, tmp_path):
        from hermes_cli import tools_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("FAL_KEY", raising=False)

        image_gen_registry.register_provider(_FakeProvider("unavail-img", available=False))

        assert tools_config._toolset_needs_configuration_prompt("image_gen", {}) is True


class TestConfigWriting:
    def test_picking_plugin_provider_writes_provider_and_model(self, monkeypatch, tmp_path):
        """When a user picks a plugin-backed image_gen provider with no
        env vars needed, ``_configure_provider`` should write both
        ``image_gen.provider`` and ``image_gen.model``."""
        from hermes_cli import tools_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        image_gen_registry.register_provider(_FakeProvider("noenv", schema={
            "name": "NoEnv",
            "badge": "free",
            "tag": "",
            "env_vars": [],
        }))

        # Stub out the interactive model picker — no TTY in tests.
        monkeypatch.setattr(tools_config, "_prompt_choice", lambda *a, **kw: 0)

        config: dict = {}
        provider_row = {
            "name": "NoEnv",
            "env_vars": [],
            "image_gen_plugin_name": "noenv",
        }
        tools_config._configure_provider(provider_row, config)

        assert config["image_gen"]["provider"] == "noenv"
        assert config["image_gen"]["model"] == "noenv-model-v1"

    def test_reconfiguring_plugin_provider_writes_provider_and_model(self, monkeypatch, tmp_path):
        """The reconfigure path should switch image_gen away from managed FAL
        and onto the selected plugin provider."""
        from hermes_cli import tools_config

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        image_gen_registry.register_provider(_FakeProvider("testopenai"))
        monkeypatch.setattr(tools_config, "_prompt_choice", lambda *a, **kw: 0)
        monkeypatch.setattr(tools_config, "_prompt", lambda *a, **kw: "")
        monkeypatch.setattr(
            tools_config,
            "get_env_value",
            lambda key: "sk-test" if key == "OPENAI_API_KEY" else "",
        )

        config = {"image_gen": {"use_gateway": True}}
        provider_row = {
            "name": "OpenAI",
            "env_vars": [{"key": "OPENAI_API_KEY", "prompt": "OpenAI API key"}],
            "image_gen_plugin_name": "testopenai",
        }

        tools_config._reconfigure_provider(provider_row, config)

        assert config["image_gen"]["provider"] == "testopenai"
        assert config["image_gen"]["model"] == "testopenai-model-v1"
        assert config["image_gen"]["use_gateway"] is False

    def test_plugin_provider_active_overrides_managed_nous_active_label(self, monkeypatch):
        from hermes_cli import tools_config

        monkeypatch.setattr(
            tools_config,
            "get_nous_subscription_features",
            lambda config, **kwargs: SimpleNamespace(
                features={"image_gen": SimpleNamespace(managed_by_nous=True)}
            ),
        )

        config = {"image_gen": {"provider": "openai", "use_gateway": False}}
        nous_row = {
            "name": "Nous Subscription",
            "managed_nous_feature": "image_gen",
        }
        openai_row = {
            "name": "OpenAI",
            "image_gen_plugin_name": "openai",
        }

        assert tools_config._is_provider_active(openai_row, config) is True
        assert tools_config._is_provider_active(nous_row, config) is False

    def test_reconfiguring_fal_clears_plugin_provider(self, monkeypatch):
        from hermes_cli import tools_config

        monkeypatch.setattr(tools_config, "_prompt_choice", lambda *a, **kw: 0)
        monkeypatch.setattr(tools_config, "_prompt", lambda *a, **kw: "")
        monkeypatch.setattr(
            tools_config,
            "get_env_value",
            lambda key: "fal-key" if key == "FAL_KEY" else "",
        )

        config = {"image_gen": {"provider": "openai", "use_gateway": False}}
        provider_row = {
            "name": "FAL.ai",
            "env_vars": [{"key": "FAL_KEY", "prompt": "FAL API key"}],
            "imagegen_backend": "fal",
        }

        tools_config._reconfigure_provider(provider_row, config)

        assert config["image_gen"]["provider"] == "fal"
        assert config["image_gen"]["use_gateway"] is False


class TestAzureSetupMetadata:
    def test_azure_image_key_is_secret_tool_metadata_only(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        metadata = OPTIONAL_ENV_VARS["AZURE_OPENAI_IMAGE_KEY"]
        assert metadata["password"] is True
        assert metadata["category"] == "tool"
        assert "image_generate" in metadata["tools"]
        assert "AZURE_OPENAI_ENDPOINT" not in OPTIONAL_ENV_VARS
        assert "AZURE_IMAGE_DEPLOYMENT_NAME" not in OPTIONAL_ENV_VARS

    def test_image_provider_rows_preserve_non_secret_config_fields(self):
        from hermes_cli import tools_config

        fields = [
            {
                "key": "image_gen.azure_openai.endpoint",
                "prompt": "Azure endpoint",
                "required": True,
            },
        ]
        image_gen_registry.register_provider(
            _FakeProvider(
                "config-fields-provider",
                schema={
                    "name": "Azure OpenAI",
                    "env_vars": [{"key": "AZURE_OPENAI_IMAGE_KEY"}],
                    "config_fields": fields,
                },
            )
        )

        row = next(
            row
            for row in tools_config._plugin_image_gen_providers()
            if row["image_gen_plugin_name"] == "config-fields-provider"
        )

        assert row["config_fields"] == fields
        assert row["env_vars"] == [{"key": "AZURE_OPENAI_IMAGE_KEY"}]


class TestNonSecretProviderConfigFields:
    def test_initial_setup_discovers_azure_and_writes_nested_fields_to_active_profile(
        self, monkeypatch, tmp_path
    ):
        from hermes_cli import plugins as plugins_module
        from hermes_cli import tools_config
        from hermes_cli.config import read_raw_config

        profile_home = tmp_path / ".hermes" / "profiles" / "azure"
        profile_home.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile_home))

        # Exercise real bundled-manifest discovery rather than manually
        # registering the provider. Bundled backend manifests must auto-load
        # without a plugins.enabled opt-in before the picker is built.
        manager = plugins_module.PluginManager()
        monkeypatch.setattr(plugins_module, "_plugin_manager", manager)
        rows = tools_config._plugin_image_gen_providers()

        loaded = manager._plugins["image_gen/azure-openai"]
        assert loaded.manifest.source == "bundled"
        assert loaded.manifest.kind == "backend"
        assert loaded.manifest.requires_env == []
        assert loaded.enabled is True, f"error: {loaded.error}"

        provider_row = next(
            row
            for row in rows
            if row.get("image_gen_plugin_name") == "azure-openai"
        )
        assert provider_row["name"] == "Azure OpenAI"

        prompts = []
        values = iter([
            "https://example-resource.services.ai.azure.com/openai/v1",
            "test-image-deployment",
        ])

        def prompt(question, default=None, password=False):
            prompts.append((question, default, password))
            if password:
                return ""
            return next(values)

        monkeypatch.setattr(tools_config, "_prompt", prompt)
        monkeypatch.setattr(tools_config, "get_env_value", lambda key: None)
        monkeypatch.setattr(
            tools_config,
            "save_env_value",
            lambda *args, **kwargs: pytest.fail("config fields must not use credential storage"),
        )

        config = {}
        tools_config._configure_provider(provider_row, config)
        tools_config.save_config(config)

        saved = read_raw_config()
        azure = saved["image_gen"]["azure_openai"]
        assert azure == {
            "endpoint": "https://example-resource.services.ai.azure.com/openai/v1",
            "deployment_name": "test-image-deployment",
        }
        assert saved["image_gen"]["provider"] == "azure-openai"
        assert (profile_home / "config.yaml").exists()
        assert len(prompts) == 3
        assert prompts[0][2] is True
        assert all(password is False for _, _, password in prompts[1:])

    def test_reconfigure_updates_visible_fields_and_keeps_blank_existing_value(self, monkeypatch):
        from hermes_cli import tools_config

        image_gen_registry.register_provider(_FakeProvider("azure-openai"))
        answers = iter([
            "https://new-resource.openai.azure.com",
            "",
            "2025-04-01-preview",
        ])
        prompts = []

        def prompt(question, default=None, password=False):
            prompts.append((question, default, password))
            return next(answers)

        monkeypatch.setattr(tools_config, "_prompt", prompt)

        config = {
            "image_gen": {
                "provider": "azure-openai",
                "azure_openai": {
                    "endpoint": "https://old-resource.openai.azure.com",
                    "deployment_name": "existing-deployment",
                },
            },
        }
        provider_row = {
            "name": "Azure OpenAI",
            "env_vars": [],
            "config_fields": [
                {"key": "image_gen.azure_openai.endpoint", "required": True},
                {"key": "image_gen.azure_openai.deployment_name", "required": True},
                {"key": "image_gen.azure_openai.api_version", "required": False},
            ],
            "image_gen_plugin_name": "azure-openai",
        }

        tools_config._reconfigure_provider(provider_row, config)

        azure = config["image_gen"]["azure_openai"]
        assert azure["endpoint"] == "https://new-resource.openai.azure.com"
        assert azure["deployment_name"] == "existing-deployment"
        assert azure["api_version"] == "2025-04-01-preview"
        assert all(password is False for _, _, password in prompts)


def test_provider_readiness_is_endpoint_auth_aware(monkeypatch):
    from agent import azure_identity_adapter
    from hermes_cli import plugins as plugins_module
    from hermes_cli import tools_config

    manager = plugins_module.PluginManager()
    monkeypatch.setattr(plugins_module, "_plugin_manager", manager)
    provider = next(
        row
        for row in tools_config._plugin_image_gen_providers()
        if row.get("image_gen_plugin_name") == "azure-openai"
    )
    credentials = {"AZURE_OPENAI_IMAGE_KEY": None}
    monkeypatch.setattr(
        tools_config, "get_env_value", lambda key: credentials.get(key)
    )

    foundry_config = {
        "image_gen": {
            "azure_openai": {
                "endpoint": "https://example.services.ai.azure.com/openai/v1",
                "deployment_name": "image-deployment",
            },
        },
    }
    direct_config = {
        "image_gen": {
            "azure_openai": {
                "endpoint": "https://example.openai.azure.com",
                "deployment_name": "image-deployment",
            },
        },
    }

    monkeypatch.setattr(
        azure_identity_adapter, "has_azure_identity_installed", lambda: False
    )
    assert (
        tools_config.provider_readiness_status(provider, foundry_config)
        == "needs_auth"
    )

    monkeypatch.setattr(
        azure_identity_adapter, "has_azure_identity_installed", lambda: True
    )
    assert tools_config.provider_readiness_status(provider, foundry_config) == "ready"
    assert (
        tools_config.provider_readiness_status(provider, direct_config)
        == "needs_keys"
    )

    credentials["AZURE_OPENAI_IMAGE_KEY"] = "secret"
    monkeypatch.setattr(
        azure_identity_adapter, "has_azure_identity_installed", lambda: False
    )
    assert tools_config.provider_readiness_status(provider, foundry_config) == "ready"
    assert tools_config.provider_readiness_status(provider, direct_config) == "ready"
    assert tools_config.provider_readiness_status(provider, {}) == "needs_setup"
