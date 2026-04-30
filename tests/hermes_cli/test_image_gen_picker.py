"""Tests for plugin image_gen providers injecting themselves into the picker.

Covers `_plugin_image_gen_providers`, `_visible_providers`, and
`_toolset_needs_configuration_prompt` handling of plugin providers.
"""

from __future__ import annotations

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
def _reset_registry():
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

    def test_fal_skipped_to_avoid_duplicate(self, monkeypatch):
        from hermes_cli import tools_config

        # Simulate a FAL plugin being registered — the picker already has
        # hardcoded FAL rows in TOOL_CATEGORIES, so plugin-FAL must be
        # skipped to avoid showing FAL twice.
        image_gen_registry.register_provider(_FakeProvider("fal"))
        image_gen_registry.register_provider(_FakeProvider("openai"))

        rows = tools_config._plugin_image_gen_providers()
        names = [r.get("image_gen_plugin_name") for r in rows]
        assert "fal" not in names
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
            lambda config: SimpleNamespace(
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
