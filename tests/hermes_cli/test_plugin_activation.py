from __future__ import annotations

import pytest

from hermes_cli.plugin_activation import PluginActivationState


@pytest.mark.parametrize(
    ("state", "plugin", "expected"),
    [
        (
            PluginActivationState(),
            {"name": "web", "key": "web", "source": "bundled", "kind": "backend"},
            "enabled",
        ),
        (
            PluginActivationState(),
            {"name": "buzz", "key": "buzz", "source": "bundled", "kind": "platform"},
            "enabled",
        ),
        (
            PluginActivationState(),
            {
                "name": "openrouter",
                "key": "model-providers/openrouter",
                "source": "bundled",
                "kind": "model-provider",
            },
            "enabled",
        ),
        (
            PluginActivationState(),
            {
                "name": "cleanup",
                "key": "cleanup",
                "source": "bundled",
                "kind": "standalone",
            },
            "not enabled",
        ),
        (
            PluginActivationState(),
            {
                "name": "third-party",
                "key": "third-party",
                "source": "user",
                "kind": "backend",
            },
            "not enabled",
        ),
        (
            PluginActivationState(enabled=frozenset({"third-party"})),
            {
                "name": "third-party",
                "key": "third-party",
                "source": "user",
                "kind": "backend",
            },
            "enabled",
        ),
        (
            PluginActivationState(
                enabled=frozenset({"third-party"}),
                disabled=frozenset({"third-party"}),
            ),
            {
                "name": "third-party",
                "key": "third-party",
                "source": "user",
                "kind": "backend",
            },
            "disabled",
        ),
        (
            PluginActivationState(safe_mode=True),
            {"name": "web", "key": "web", "source": "bundled", "kind": "backend"},
            "not enabled",
        ),
        (
            PluginActivationState(safe_mode=True),
            {
                "name": "openrouter",
                "key": "model-providers/openrouter",
                "source": "bundled",
                "kind": "model-provider",
            },
            "enabled",
        ),
    ],
)
def test_activation_policy_matrix(state, plugin, expected):
    assert state.status(**plugin) == expected


def _reset_config_state(config_module) -> None:
    config_module._LOAD_CONFIG_CACHE.clear()
    config_module._LAST_EXPANDED_CONFIG_BY_PATH.clear()


def test_activation_accessor_uses_canonical_env_expansion(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_PLUGIN_KEY", "expanded-plugin")
    (tmp_path / "config.yaml").write_text(
        'plugins:\n  enabled: ["${TEST_PLUGIN_KEY}"]\n',
        encoding="utf-8",
    )

    from hermes_cli import config

    _reset_config_state(config)
    state = config.load_plugin_activation_state()

    assert state.enabled == frozenset({"expanded-plugin"})


def test_activation_accessor_preserves_canonical_last_known_good(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "plugins:\n  enabled: [user-plugin]\n  disabled: [model-providers/gmi]\n",
        encoding="utf-8",
    )

    from hermes_cli import config

    _reset_config_state(config)
    good = config.load_plugin_activation_state()
    config_path.write_text("plugins: [\n", encoding="utf-8")
    retained = config.load_plugin_activation_state()

    assert retained == good
    assert retained.enabled == frozenset({"user-plugin"})
    assert retained.disabled == frozenset({"model-providers/gmi"})


def test_fresh_corrupt_config_fails_closed_for_nonbundled(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("plugins: [\n", encoding="utf-8")

    from hermes_cli import config

    _reset_config_state(config)
    state = config.load_plugin_activation_state()

    assert not state.is_active(
        name="third-party",
        key="third-party",
        source="user",
        kind="backend",
    )
    assert state.is_active(
        name="bundled-backend",
        key="bundled-backend",
        source="bundled",
        kind="backend",
    )


def test_static_model_alias_cannot_revive_disabled_provider(monkeypatch):
    from hermes_cli import models

    monkeypatch.setattr(
        models,
        "_PROVIDER_MODELS",
        {"anthropic": ["claude-sonnet-4"]},
    )
    monkeypatch.setattr(
        models,
        "_provider_is_routable",
        lambda provider: provider != "anthropic",
    )

    assert models._resolve_static_model_alias("sonnet", set()) is None


def test_disabled_provider_catalogs_do_not_leak_static_or_disk_cache(monkeypatch):
    from hermes_cli import models

    provider = "disabled-cache-provider"
    fingerprint = "disabled-cache-fingerprint"
    monkeypatch.setattr(models, "_provider_is_routable", lambda _provider: False)
    monkeypatch.setattr(
        models,
        "_PROVIDER_MODELS",
        {provider: ["static-model-must-not-leak"]},
    )
    monkeypatch.setattr(
        models,
        "_load_provider_models_cache",
        lambda: {
            provider: {
                "fp": fingerprint,
                "at": models.time.time(),
                "models": ["cached-model-must-not-leak"],
            }
        },
    )
    monkeypatch.setattr(
        models,
        "_credential_fingerprint",
        lambda _provider: fingerprint,
    )

    assert models.provider_model_ids(provider) == []
    assert models.cached_provider_model_ids(provider) == []
