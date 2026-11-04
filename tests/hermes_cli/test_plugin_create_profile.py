"""``ctx.create_profile``: a plugin creates a profile that can answer, without holding a credential.

Contracts: the new profile carries the plugin's profile's provider key (and not its messaging bot
token), the receipt the plugin gets carries no credential value, ``mirror_credentials=False`` gives
the bare comment-only profile, and the source is the profile that LOADED the plugin, not whichever
home the calling thread holds. Plugins load through real discovery from temp homes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hermes_cli.plugins as plugins_mod
from agent.secret_scope import build_profile_secret_scope
from hermes_cli.plugins import PluginManager
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_PLUGIN = "CTX = None\n\ndef register(ctx):\n    global CTX\n    CTX = ctx\n"


def _write_home(home: Path, provider_key: str) -> None:
    plugin_dir = home / "plugins" / "maker"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(json.dumps({"name": "maker", "version": "0.1.0"}))
    (plugin_dir / "__init__.py").write_text(_PLUGIN)
    (home / "config.yaml").write_text(json.dumps({"plugins": {"enabled": ["maker"]}}))
    (home / ".env").write_text(f"OPENROUTER_API_KEY={provider_key}\nTELEGRAM_BOT_TOKEN=123:tg-FAKE\n")


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Default home and named profile ``work``, each loading the ``maker`` plugin with its own key."""
    default_home = tmp_path / ".hermes"
    work_home = default_home / "profiles" / "work"
    _write_home(default_home, "sk-or-FAKE-default")
    _write_home(work_home, "sk-or-FAKE-work")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: tmp_path / "empty-bundled")
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    plugins_mod._reset_plugin_managers_for_tests()
    yield default_home, work_home
    plugins_mod._reset_plugin_managers_for_tests()


def _plugin_ctx(home: Path):
    token = set_hermes_home_override(str(home))
    try:
        manager = plugins_mod.get_plugin_manager()
        manager.discover_and_load(force=True)
    finally:
        reset_hermes_home_override(token)
    return manager._plugins["maker"].module.CTX


def test_created_profile_has_the_provider_key_and_the_plugin_never_sees_it(homes):
    default_home, _ = homes
    result = _plugin_ctx(default_home).create_profile("scout", no_alias=True)

    scout = Path(result["path"])
    assert scout == default_home / "profiles" / "scout"
    secrets = build_profile_secret_scope(scout)
    assert secrets.get("OPENROUTER_API_KEY") == "sk-or-FAKE-default"
    assert "TELEGRAM_BOT_TOKEN" not in secrets  # one bot, one profile
    assert result["mirrored"]["env"] is True
    assert "sk-or-FAKE" not in json.dumps(result) and "tg-FAKE" not in json.dumps(result)


def test_mirror_off_leaves_the_bare_profile(homes):
    default_home, _ = homes
    result = _plugin_ctx(default_home).create_profile("bare", no_alias=True, mirror_credentials=False)

    bare = Path(result["path"])
    assert "OPENROUTER_API_KEY" not in build_profile_secret_scope(bare)
    assert not (bare / "auth.json").exists()
    assert result["mirrored"] == {"env": False, "auth": False, "model_inherited": False, "voice": False}


def test_source_is_the_plugins_own_profile_not_the_callers_home(homes):
    default_home, work_home = homes
    work_ctx, default_ctx = _plugin_ctx(work_home), _plugin_ctx(default_home)

    # The calling thread holds the default home (HERMES_HOME); the work plugin still copies work's key.
    from_work = Path(work_ctx.create_profile("from-work", no_alias=True)["path"])
    from_default = Path(default_ctx.create_profile("from-default", no_alias=True)["path"])

    assert build_profile_secret_scope(from_work).get("OPENROUTER_API_KEY") == "sk-or-FAKE-work"
    assert build_profile_secret_scope(from_default).get("OPENROUTER_API_KEY") == "sk-or-FAKE-default"


def test_a_taken_name_raises_and_changes_nothing(homes):
    default_home, _ = homes
    ctx = _plugin_ctx(default_home)
    ctx.create_profile("scout", no_alias=True)
    env_before = (default_home / "profiles" / "scout" / ".env").read_bytes()

    with pytest.raises((ValueError, FileExistsError)):
        ctx.create_profile("scout", no_alias=True)
    assert (default_home / "profiles" / "scout" / ".env").read_bytes() == env_before
