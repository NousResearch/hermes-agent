"""User-installed platform plugins feed OPTIONAL_ENV_VARS like bundled ones (#46600, redo of #46964).

The Desktop Gateway form and ``hermes config`` render env fields from ``OPTIONAL_ENV_VARS``; before
this, only ``plugins/platforms/*`` in the repo was scanned, so a third-party platform's
``requires_env`` prompts/descriptions/password flags never reached the UI.
"""

import hermes_cli.config as config_mod


def _manifest(path, text):
    path.mkdir(parents=True)
    (path / "plugin.yaml").write_text(text, encoding="utf-8")


def test_user_platform_plugins_inject_env_metadata_but_non_platforms_do_not(monkeypatch, tmp_path):
    home = tmp_path / "hermes-home"
    _manifest(home / "plugins" / "demo-platform", (
        "name: demo-platform\nkind: platform\nlabel: Demo Platform\n"
        "requires_env:\n  - name: DEMO_PLATFORM_TOKEN\n    description: Token for the demo platform\n"
        "    prompt: Demo token\n    url: https://example.invalid/demo\n"
        "optional_env:\n  - name: DEMO_PLATFORM_ROOM\n    password: false\n"))
    _manifest(home / "plugins" / "platforms" / "nested-platform", (
        "name: nested-platform\nrequires_env:\n  - NESTED_PLATFORM_SECRET\n"))
    _manifest(home / "plugins" / "ignore-me", "name: ignore-me\nkind: backend\nrequires_env:\n  - IGNORE_ME_TOKEN\n")
    keys = ["DEMO_PLATFORM_TOKEN", "DEMO_PLATFORM_ROOM", "NESTED_PLATFORM_SECRET", "IGNORE_ME_TOKEN"]
    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: home)
    for key in keys:
        monkeypatch.delitem(config_mod.OPTIONAL_ENV_VARS, key, raising=False)

    config_mod._inject_platform_plugin_env_vars()

    try:
        assert config_mod.OPTIONAL_ENV_VARS["DEMO_PLATFORM_TOKEN"] == {
            "description": "Token for the demo platform", "prompt": "Demo token",
            "url": "https://example.invalid/demo", "password": True, "category": "messaging"}
        assert config_mod.OPTIONAL_ENV_VARS["DEMO_PLATFORM_ROOM"]["password"] is False
        assert config_mod.OPTIONAL_ENV_VARS["NESTED_PLATFORM_SECRET"]["password"] is True  # category dir needs no kind
        assert "IGNORE_ME_TOKEN" not in config_mod.OPTIONAL_ENV_VARS
    finally:
        for key in keys:
            config_mod.OPTIONAL_ENV_VARS.pop(key, None)
