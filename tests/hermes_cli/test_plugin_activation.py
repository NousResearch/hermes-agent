from __future__ import annotations

import pytest

from hermes_cli.plugin_activation import PluginActivationState


@pytest.mark.parametrize(
    ("name", "key", "source", "kind", "enabled", "disabled", "safe_mode", "expected"),
    (
        ("web", None, "bundled", "backend", (), (), False, "enabled"),
        ("buzz", None, "bundled", "platform", (), (), False, "enabled"),
        ("openrouter", "model-providers/openrouter", "bundled", "model-provider", (), (), False, "enabled"),
        ("cleanup", None, "bundled", "standalone", (), (), False, "not enabled"),
        ("third-party", None, "user", "backend", (), (), False, "not enabled"),
        ("third-party", None, "user", "backend", ("third-party",), (), False, "enabled"),
        (
            "third-party", None, "user", "backend",
            ("third-party",), ("third-party",), False, "disabled",
        ),
        ("web", None, "bundled", "backend", (), (), True, "not enabled"),
        ("openrouter", "model-providers/openrouter", "bundled", "model-provider", (), (), True, "enabled"),
    ),
    ids=(
        "bundled-backend-default", "bundled-platform-default", "model-provider-default",
        "standalone-default-deny", "user-default-deny", "explicit-enable",
        "explicit-deny-wins", "safe-mode-backend-deny", "safe-mode-model-provider",
    ),
)
def test_activation_policy_matrix(
    name, key, source, kind, enabled, disabled, safe_mode, expected
):
    state = PluginActivationState(
        enabled=frozenset(enabled), disabled=frozenset(disabled), safe_mode=safe_mode
    )
    assert state.status(name=name, key=key or name, source=source, kind=kind) == expected


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


def test_safe_mode_does_not_read_user_plugin_configuration(
    monkeypatch,
):
    from hermes_cli import config

    monkeypatch.setenv("HERMES_SAFE_MODE", "1")

    def user_config_must_not_be_read():
        raise AssertionError("safe mode read user config")

    monkeypatch.setattr(
        config,
        "load_config_readonly",
        user_config_must_not_be_read,
    )
    state = config.load_plugin_activation_state()

    assert state == PluginActivationState(safe_mode=True)
    assert state.is_active(
        name="openrouter",
        key="model-providers/openrouter",
        source="bundled",
        kind="model-provider",
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
