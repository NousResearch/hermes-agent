"""The Weixin ``LEFT_CORE`` row: homes core ran Weixin for get the ``weixin`` catalog plugin, others never.

Each in_use case is a home whose verdict on main came from a real probe: ``load_gateway_config()`` plus the
adapter core created and its connect preflight (token and account id, the QR login's saved account file
included). Real config/.env files under a temp home; the catalog lookup and the install are stand-ins.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import hermes_cli.left_core_migration as lcm
import hermes_yaml as yaml
from tests.hermes_cli.plugin_worker_support import (
    isolated_python as isolated_python,  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
    plugin_world as plugin_world,  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import
)

_CREDS = "WEIXIN_ACCOUNT_ID=acc1\nWEIXIN_TOKEN=tok1\n"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(lcm, "_attempted", set())
    for name in ("WEIXIN_ACCOUNT_ID", "WEIXIN_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    import hermes_cli.memory_provider_migration as mpm
    monkeypatch.setattr(mpm, "catalog_source", lambda name: name)


def _home(tmp_path: Path, *, env: str = "", config: str = "", saved_token: str = "") -> Path:
    home = tmp_path / "home"
    home.mkdir()
    if env:
        (home / ".env").write_text(env, encoding="utf-8")
    if config:
        (home / "config.yaml").write_text(config, encoding="utf-8")
    if saved_token:  # what the QR login (save_weixin_account) wrote
        accounts = home / "weixin" / "accounts"
        accounts.mkdir(parents=True)
        (accounts / "acc1.json").write_text(json.dumps({"token": saved_token, "base_url": ""}), encoding="utf-8")
    return home


def _install(home: Path):
    calls: list[str] = []

    def install(name):
        calls.append(name)
        (home / "plugins" / name).mkdir(parents=True, exist_ok=True)
        return {"ok": True}
    return install, calls


# The platform_toolsets block cli-config.yaml.example shipped while Weixin was in core.
_SHIPPED_PLATFORM_TOOLSETS = (
    "platform_toolsets:\n  cli: [hermes-cli]\n  telegram: [hermes-telegram]\n  weixin: [hermes-weixin]\n"
)


@pytest.mark.parametrize(("env", "config", "saved_token", "in_use"), [
    (_CREDS, "", "", True),
    ("", "", "", False),
    ("WEIXIN_TOKEN=tok1\n", "", "", False),  # no account: the adapter's preflight refused to connect
    ("WEIXIN_ACCOUNT_ID=acc1\n", "", "", False),  # no token anywhere
    ("WEIXIN_ACCOUNT_ID=acc1\n", "", "tok1", True),  # the QR login's saved token filled it in
    ("", "platforms:\n  weixin:\n    enabled: true\n    extra: {account_id: acc1}\n", "tok1", True),
    ("", "platforms:\n  weixin:\n    extra: {account_id: acc1, token: tok1}\n", "", True),
    # A block token with no enabled key: only extra/env reached the enable pass's probe, so it stayed off.
    ("", "platforms:\n  weixin:\n    token: tok1\n    extra: {account_id: acc1}\n", "", False),
    ("", "platforms:\n  weixin:\n    enabled: true\n    token: tok1\n    extra: {account_id: acc1}\n", "", True),
    ("WEIXIN_TOKEN=tok1\n", "platforms:\n  weixin:\n    extra: {account_id: acc1}\n", "", True),
    ("WEIXIN_ACCOUNT_ID=acc1\n", "platforms:\n  weixin:\n    token: tok1\n", "", True),
    ("", "platforms:\n  weixin:\n    enabled: true\n", "", False),
    # An explicit config.yaml disable vetoed env credentials, legacy and top-level blocks included.
    (_CREDS, "platforms:\n  weixin:\n    enabled: false\n", "", False),
    (_CREDS, "gateway:\n  weixin:\n    enabled: false\n", "", False),
    (_CREDS, "weixin:\n  enabled: false\n", "", False),
    (_CREDS, "platforms:\n  weixin:\n    enabled: false\nweixin:\n  enabled: true\n", "", True),
    # A saved home channel names Platform("weixin"), which exists only once the plugin loads.
    (_CREDS, "platforms:\n  weixin:\n    enabled: true\n    home_channel: {platform: weixin, chat_id: x}\n", "", True),
    ("", _SHIPPED_PLATFORM_TOOLSETS, "", False),
    ("", "platform_toolsets:\n  telegram: [hermes-weixin]\n", "", False),  # core tools only, never the adapter
    (_CREDS, "plugins:\n  disabled: [platforms/weixin]\n", "", False),
    (_CREDS, "plugins:\n  disabled: [weixin-platform]\n", "", False),
    (_CREDS, "plugins:\n  disabled: [weixin]\n", "", True),  # core's adapter had another name
])
def test_weixin_in_use_matches_what_core_ran(tmp_path, env, config, saved_token, in_use):
    assert lcm.weixin_in_use(_home(tmp_path, env=env, config=config, saved_token=saved_token)) is in_use


def test_a_disable_in_legacy_gateway_json_did_not_stop_env_credentials(tmp_path):
    """gateway.json's ``enabled: false`` is overwritten by the env enable (its marker never reached the pass)."""
    home = _home(tmp_path, env=_CREDS)
    (home / "gateway.json").write_text(json.dumps({"platforms": {"weixin": {"enabled": False}}}))
    assert lcm.weixin_in_use(home) is True


def test_a_weixin_home_gets_the_plugin_once(tmp_path):
    home = _home(tmp_path, env=_CREDS)
    install, calls = _install(home)
    said: list[str] = []
    assert lcm.migrate_home(home, install=install, say=said.append) == ["weixin"]
    assert lcm.migrate_home(home, install=install, say=said.append) == []
    assert calls == ["weixin"]
    assert "plugins.disabled" not in said[0]
    # No plugin toolset: nothing is recorded as off anywhere.
    from hermes_cli.config import read_user_config_raw
    assert "weixin" not in str(read_user_config_raw(home / "config.yaml").get("known_plugin_toolsets") or {})


def test_a_fresh_or_shipped_default_home_installs_nothing(tmp_path):
    for config in ("", _SHIPPED_PLATFORM_TOOLSETS):
        home = tmp_path / ("shipped" if config else "fresh")
        home.mkdir()
        if config:
            (home / "config.yaml").write_text(config, encoding="utf-8")
        said: list[str] = []
        assert lcm.migrate_home(home, install=lambda name: pytest.fail("installed"), say=said.append) == []
        assert said == []


def test_a_secondary_whose_credentials_come_from_its_secret_scope_uses_weixin(tmp_path):
    from agent.secret_scope import reset_secret_scope, set_secret_scope
    home = _home(tmp_path)
    other = tmp_path / "other"
    for bound, in_use in ((home, True), (other, False)):
        token = set_secret_scope({"WEIXIN_ACCOUNT_ID": "acc1", "WEIXIN_TOKEN": "tok1"}, profile_home=str(bound))
        try:
            assert lcm.weixin_in_use(home) is in_use
        finally:
            reset_secret_scope(token)


@pytest.mark.parametrize("name", ["platforms/weixin", "weixin-platform"])
def test_a_core_era_disable_keeps_weixin_off_through_both_installers(tmp_path, plugin_world, monkeypatch, name):
    """Core never ran Weixin for this home, so the migration skips it; the default installer, which runs
    beside it on every update, may place the plugin but its enable must not lift the disable core wrote
    (``platforms/weixin``, core's manifest name). Real catalog install: real clone of a local repo, real
    enable and config writes; only the https transport is redirected."""
    from hermes_cli import default_plugins, plugins_cmd, plugins_cmd_catalog
    from hermes_cli.plugin_catalog import PluginCatalogEntry
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    home = plugin_world.home
    (home / ".env").write_text(_CREDS, encoding="utf-8")
    (home / "config.yaml").write_text(f"plugins:\n  enabled: []\n  disabled: ['{name}']\n", encoding="utf-8")
    install, calls = _install(home)
    assert lcm.migrate_home(home, install=install, say=lambda _: None) == [] and calls == []

    repo = tmp_path / "weixin-repo"
    repo.mkdir()
    (repo / "plugin.yaml").write_text(yaml.safe_dump({"name": "weixin", "version": "1.0.0"}), encoding="utf-8")
    (repo / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")

    def git(*args):
        return subprocess.run(["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args],
                              cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q", "-b", "main")
    git("add", ".")
    git("commit", "-qm", "v1")
    entry = PluginCatalogEntry(name="weixin", repo="https://example.invalid/nous/weixin.git",
                               sha=git("rev-parse", "HEAD"), description="d", maintainer="NousResearch",
                               tier="official", default=True)
    monkeypatch.setattr(plugins_cmd_catalog, "get_live_catalog_entry", lambda n: entry if n == "weixin" else None)
    real = plugins_cmd._clone_plugin_repo
    monkeypatch.setattr(plugins_cmd, "_clone_plugin_repo",
                        lambda dest, _url, revision, subdir=None: real(dest, repo.as_uri(), revision, subdir))
    said: list[str] = []

    default_plugins.install_defaults(home, say=said.append, entries=[entry])

    plugins = (yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}).get("plugins") or {}
    assert plugins.get("disabled") == [name], f"the core-era disable was lifted: {plugins}"
    assert "weixin" not in (plugins.get("enabled") or []), plugins
    assert any(f"plugins.disabled lists '{name}', so it is installed but off" in line
               and "plugins enable weixin`" in line for line in said), said
