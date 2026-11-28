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
    # Send-only: an explicit config.yaml disable kept the adapter off, but core's send_message (and cron
    # delivery through it) still sent with env credentials, legacy and top-level blocks included.
    (_CREDS, "platforms:\n  weixin:\n    enabled: false\n", "", True),
    (_CREDS, "gateway:\n  weixin:\n    enabled: false\n", "", True),
    (_CREDS, "weixin:\n  enabled: false\n", "", True),
    ("WEIXIN_TOKEN=tok1\n", "platforms:\n  weixin:\n    enabled: false\n    extra: {account_id: acc1}\n", "", False),
    ("", "platforms:\n  weixin:\n    enabled: false\n    token: tok1\n    extra: {account_id: acc1}\n", "", False),
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


# The detached default-plugin child mid-install, in another process: holds the installer lock with weixin's tree
# placed and its ``installing`` marker written, until the test signals *release*; then enables it and lets go
# (``fails``: removes the tree instead, an install that broke before it enabled anything; ``refused``: keeps the
# tree and writes the default installer's ``placed`` marker, a failed enable; ``killed``: exits holding its
# ``installing`` marker, the tree left behind).
_INSTALLER_CHILD = """
import sys
from pathlib import Path
from hermes_cli.default_plugins import installer_lock, installing_marker
from tests.waiting import wait_until
home, signals, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
with installer_lock(5):
    (home / "plugins" / "weixin").mkdir(parents=True, exist_ok=True)
    installing_marker(home, "weixin").parent.mkdir(parents=True, exist_ok=True)
    installing_marker(home, "weixin").touch()
    (signals / "holding").touch()
    wait_until(lambda: (signals / "release").exists(), timeout=120, what="release signal")
    if mode == "killed":
        raise SystemExit(9)
    if mode == "fails":
        (home / "plugins" / "weixin").rmdir()
    elif mode == "refused":
        (home / "cache" / "default-plugins.placed-weixin").touch()
    else:
        (home / "config.yaml").write_text("platforms:\\n  weixin:\\n    enabled: false\\n"
                                          "plugins:\\n  enabled: [weixin]\\n_default_plugins_installed: [weixin]\\n")
    installing_marker(home, "weixin").unlink(missing_ok=True)  # a test may have cleared it
"""


def _send_only_home_with_installer_in_flight(tmp_path, *, mode: str = "places"):
    import os
    import sys
    from hermes_constants import get_hermes_home
    from tests.waiting import wait_until
    home = Path(get_hermes_home())
    (home / ".env").write_text(_CREDS, encoding="utf-8")
    (home / "config.yaml").write_text("platforms:\n  weixin:\n    enabled: false\n", encoding="utf-8")
    signals = tmp_path / "signals"
    signals.mkdir()
    child = subprocess.Popen([sys.executable, "-c", _INSTALLER_CHILD, str(home), str(signals), mode],
                             env={**os.environ, "HERMES_HOME": str(home)}, cwd=str(Path(lcm.__file__).parents[1]))
    wait_until(lambda: (signals / "holding").exists(), timeout=60, what="installer child holding the lock")
    return home, signals, child


def _run_send(monkeypatch, home, *, on_wait=None, install=None):
    """``hermes send --to weixin:wxid_x hi`` through cmd_send; the delivery is a stand-in recording whether the
    plugin was enabled in the home when it ran. *on_wait* runs as the send starts waiting on the installer lock;
    *install* stands in for the catalog install (default: none may run)."""
    import argparse
    from gateway.outbound import send as send_mod
    from hermes_cli import default_plugins, send_cmd
    seen: list[bool] = []
    real_lock = default_plugins.installer_lock

    def installer_lock(wait):
        if on_wait is None:
            pytest.fail("a send whose plugin needs nothing waited on the installer lock")
        on_wait()
        return real_lock(wait)
    monkeypatch.setattr(default_plugins, "installer_lock", installer_lock)
    monkeypatch.setattr(send_cmd, "_load_hermes_env", lambda: None)  # the predicate reads the home's .env itself
    monkeypatch.setattr(send_mod, "send_message_tool",
                        lambda args: (seen.append(lcm._plugins_listed(home, ("weixin",))[0]), '{"success": true}')[1])
    monkeypatch.setattr(lcm, "_install_into", lambda home, consent=None: install or pytest.fail("installed a second copy"))
    with pytest.raises(SystemExit):
        send_cmd.cmd_send(argparse.Namespace(to="weixin:wxid_x", message="hi", file=None, subject=None,
                                             mentions=None, list_targets=False, json=True, quiet=True))
    return seen


def test_the_first_send_waits_for_the_install_already_in_flight(tmp_path, monkeypatch):
    """An upgraded send-only home's first command is ``hermes send`` while the background default-plugin child
    is still installing weixin: the send waits for that install (the installer lock) and delivers through the
    plugin it placed, instead of failing on an unregistered platform. It never installs a second copy."""
    home, signals, child = _send_only_home_with_installer_in_flight(tmp_path)
    try:
        seen = _run_send(monkeypatch, home, on_wait=lambda: (signals / "release").touch())
    finally:
        (signals / "release").touch()
        child.wait(timeout=60)
    assert seen == [True]


@pytest.mark.parametrize("mode", ["fails", "refused", "killed"])
def test_a_send_whose_in_flight_install_failed_installs_weixin_itself(tmp_path, monkeypatch, mode):
    """The installer the send waited for stopped without enabling weixin (it failed before placing it, its enable
    was refused, or it was killed with the tree on disk): getting the lock is not success and a leftover tree is
    not an install, so the send rechecks and installs weixin itself, then delivers through it."""
    import pm.install
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: True)
    home, signals, child = _send_only_home_with_installer_in_flight(tmp_path, mode=mode)
    installed: list[str] = []

    def install(name):
        installed.append(name)
        (home / "plugins" / name).mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("platforms:\n  weixin:\n    enabled: false\nplugins:\n  enabled: [weixin]\n",
                                          encoding="utf-8")
        return {"ok": True}
    try:
        seen = _run_send(monkeypatch, home, on_wait=lambda: (signals / "release").touch(), install=install)
    finally:
        (signals / "release").touch()
        child.wait(timeout=60)
    assert installed == ["weixin"] and seen == [True]
    from hermes_cli.default_plugins import left_unfinished
    assert not left_unfinished(home, "weixin"), "the recovered install left the default installer's markers behind"


def test_a_ready_weixin_never_waits_on_a_leftover_installing_marker(tmp_path, monkeypatch):
    """An earlier ``hermes send`` recovered the install the default-plugin child was killed in, so weixin is enabled
    while that child's ``installing`` marker is still on disk; another process now holds the installer lock for
    something else. The ready plugin outranks the leftover marker: the send goes straight to its target."""
    home, signals, child = _send_only_home_with_installer_in_flight(tmp_path)
    try:
        (home / "config.yaml").write_text("platforms:\n  weixin:\n    enabled: false\nplugins:\n  enabled: [weixin]\n",
                                          encoding="utf-8")
        seen = _run_send(monkeypatch, home)
    finally:
        (signals / "release").touch()
        child.wait(timeout=60)
    assert seen == [True]


def test_a_send_that_outwaits_its_budget_says_the_installer_is_busy(tmp_path, monkeypatch, capsys):
    """Past the bounded wait the send fails, and says why: another process is installing plugins, retry; never
    only the generic unknown-platform error with an install command that would race the running install."""
    from gateway.outbound.targets import unknown_platform_error
    monkeypatch.setattr(lcm, "_STARTUP_LOCK_WAIT", 1.0)
    home, signals, child = _send_only_home_with_installer_in_flight(tmp_path)
    try:
        _run_send(monkeypatch, home, on_wait=lambda: None)
        error = unknown_platform_error("weixin")
    finally:
        (signals / "release").touch()
        child.wait(timeout=60)
    assert "installing plugins right now" in capsys.readouterr().err
    assert "installing plugins right now" in error and "retry" in error, error


@pytest.mark.parametrize("plugins", ["enabled: [weixin]", "disabled: [weixin]"])
def test_a_send_whose_plugin_is_ready_or_disabled_never_waits_on_a_busy_installer(tmp_path, monkeypatch, plugins):
    """Another process holds the installer lock (another plugin, another profile): a send whose weixin is already
    installed, or that the user disabled, goes straight to its target."""
    home, signals, child = _send_only_home_with_installer_in_flight(tmp_path)
    try:
        from hermes_cli.default_plugins import installing_marker
        installing_marker(home, "weixin").unlink()  # the child is busy with something else
        (home / "config.yaml").write_text(f"platforms:\n  weixin:\n    enabled: false\nplugins:\n  {plugins}\n"
                                          "_default_plugins_installed: [weixin]\n", encoding="utf-8")
        seen = _run_send(monkeypatch, home)
    finally:
        (signals / "release").touch()
        child.wait(timeout=60)
    assert seen == ["enabled" in plugins]


# A saved ``hermes-weixin`` pick (a platform_toolsets row, a cron job, a delegation parent) resolved to core tools
# on main, Weixin plugin or not: the bundle carried core tools only. It stays out of every listing.
def test_a_mixed_list_naming_hermes_weixin_keeps_its_core_tools_and_an_unchecked_one_stays_off():
    from hermes_cli import tools_config
    config = {"platform_toolsets": {"telegram": ["hermes-weixin", "file"]}}
    enabled = tools_config._get_platform_tools(config, "telegram")
    assert enabled >= {"file", "terminal", "web"}
    tools_config._save_platform_tools(config, "telegram", enabled - {"terminal"})
    assert "hermes-weixin" not in config["platform_toolsets"]["telegram"]
    assert "terminal" not in tools_config._get_platform_tools(config, "telegram")


def test_a_bare_hermes_weixin_row_reads_as_its_core_tools_without_the_plugin():
    from hermes_cli import tools_config
    from toolsets import TOOLSETS, get_toolset_names
    config = {"platform_toolsets": {"telegram": ["hermes-weixin"]}}
    enabled = tools_config._get_platform_tools(config, "telegram")
    assert enabled >= {"file", "terminal", "web"}
    tools_config._save_platform_tools(config, "telegram", enabled)
    assert tools_config._get_platform_tools(config, "telegram") >= {"file", "terminal", "web"}
    assert "hermes-weixin" not in get_toolset_names()
    assert "hermes-weixin" not in TOOLSETS["hermes-gateway"]["includes"]


def test_an_mcp_server_named_hermes_weixin_does_not_shadow_the_old_preset(monkeypatch):
    import toolsets
    from hermes_cli import tools_config
    real = toolsets.resolve_toolset
    monkeypatch.setattr(toolsets, "resolve_toolset",
                        lambda name, *a, **k: ["mcp_tool"] if name == "hermes-weixin" else real(name, *a, **k))
    config = {"platform_toolsets": {"telegram": ["hermes-weixin", "file"]}}
    assert tools_config._get_platform_tools(config, "telegram") >= {"file", "terminal", "web"}


@pytest.mark.parametrize("collision", [False, True])
def test_saved_weixin_preset_still_selects_core_tools_at_runtime(monkeypatch, collision):
    from cron.scheduler import _resolve_cron_enabled_toolsets
    from model_tools import _select_tool_names
    from tools.registry import ToolRegistry
    from toolsets import _resolve_toolset_memo

    registry = ToolRegistry()
    monkeypatch.setattr("tools.registry.registry", registry)
    _resolve_toolset_memo.clear()
    if collision:
        registry.register("mcp_fixture_echo", "mcp-hermes-weixin", {"name": "mcp_fixture_echo"}, lambda *_: "")
        registry.register_toolset_alias("hermes-weixin", "mcp-hermes-weixin")

    picks = _resolve_cron_enabled_toolsets({"enabled_toolsets": ["hermes-weixin", "file"]}, {})
    tools = _select_tool_names(picks, None, quiet_mode=True)
    assert {"terminal", "web_search"} <= tools
    assert ("mcp_fixture_echo" in tools) is collision


def test_a_retired_hermes_weixin_parent_still_grants_its_core_toolsets_to_a_child():
    from types import SimpleNamespace
    from tools.delegate_tool_toolsets import _resolve_child_toolsets
    parent = SimpleNamespace(enabled_toolsets=["hermes-weixin"], disabled_toolsets=[])
    child, _disabled = _resolve_child_toolsets(parent, ["terminal", "web"], "leaf")
    assert child == ["terminal", "web"]
