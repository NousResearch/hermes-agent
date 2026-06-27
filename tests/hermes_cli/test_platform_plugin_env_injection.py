import yaml

import hermes_cli.config as config_mod


def _write_plugin_manifest(path, manifest):
    path.mkdir(parents=True, exist_ok=True)
    (path / "plugin.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")


def test_inject_platform_plugin_env_vars_scans_user_plugins(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    user_plugins = hermes_home / "plugins"

    _write_plugin_manifest(
        user_plugins / "demo-platform",
        {
            "name": "demo-platform",
            "label": "Demo Platform",
            "kind": "platform",
            "requires_env": [
                {
                    "name": "DEMO_PLATFORM_TOKEN",
                    "description": "Token for the demo platform",
                    "prompt": "Demo token",
                    "url": "https://example.com/demo",
                    "password": True,
                    "category": "messaging",
                }
            ],
            "optional_env": [
                {
                    "name": "DEMO_PLATFORM_ROOM",
                    "description": "Default room for notifications",
                    "prompt": "Demo room",
                    "password": False,
                }
            ],
        },
    )
    _write_plugin_manifest(
        user_plugins / "platforms" / "nested-platform",
        {
            "name": "nested-platform",
            "label": "Nested Platform",
            "kind": "platform",
            "requires_env": [
                {
                    "name": "NESTED_PLATFORM_SECRET",
                    "description": "Shared secret for the nested platform",
                    "prompt": "Nested secret",
                    "password": True,
                }
            ],
        },
    )
    _write_plugin_manifest(
        user_plugins / "image_gen" / "ignore-me",
        {
            "name": "ignore-me",
            "kind": "backend",
            "requires_env": [{"name": "IGNORE_ME_TOKEN"}],
        },
    )

    added_keys = [
        "DEMO_PLATFORM_TOKEN",
        "DEMO_PLATFORM_ROOM",
        "NESTED_PLATFORM_SECRET",
        "IGNORE_ME_TOKEN",
    ]
    original_flag = config_mod._platform_plugin_env_vars_injected
    try:
        monkeypatch.setattr(config_mod, "get_hermes_home", lambda: hermes_home)
        config_mod._platform_plugin_env_vars_injected = False
        for key in added_keys:
            config_mod.OPTIONAL_ENV_VARS.pop(key, None)

        config_mod._inject_platform_plugin_env_vars()

        assert config_mod.OPTIONAL_ENV_VARS["DEMO_PLATFORM_TOKEN"] == {
            "description": "Token for the demo platform",
            "prompt": "Demo token",
            "url": "https://example.com/demo",
            "password": True,
            "category": "messaging",
        }
        assert config_mod.OPTIONAL_ENV_VARS["DEMO_PLATFORM_ROOM"] == {
            "description": "Default room for notifications",
            "prompt": "Demo room",
            "url": None,
            "password": False,
            "category": "messaging",
        }
        assert config_mod.OPTIONAL_ENV_VARS["NESTED_PLATFORM_SECRET"] == {
            "description": "Shared secret for the nested platform",
            "prompt": "Nested secret",
            "url": None,
            "password": True,
            "category": "messaging",
        }
        assert "IGNORE_ME_TOKEN" not in config_mod.OPTIONAL_ENV_VARS
    finally:
        for key in added_keys:
            config_mod.OPTIONAL_ENV_VARS.pop(key, None)
        config_mod._platform_plugin_env_vars_injected = original_flag
