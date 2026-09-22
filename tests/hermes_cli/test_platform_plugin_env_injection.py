"""User-installed platform plugins get their manifest env vars surfaced like bundled ones.

Regression: ``_inject_platform_plugin_env_vars`` only scanned the bundled
``plugins/platforms/`` tree, so a platform plugin installed in
``~/.hermes/plugins/`` (the documented user location) that followed the docs
and declared ``requires_env``/``optional_env`` in its manifest got nothing —
the env vars never reached ``OPTIONAL_ENV_VARS``, so the Channels card and
``hermes config`` showed no fields for it. The docs promise parity
(``website/docs/developer-guide/adding-platform-adapters.md`` § Surfacing
Env Vars), the implementation delivered bundled-only.
"""
from __future__ import annotations

import os

import pytest

from hermes_cli import config as config_mod


@pytest.fixture()
def user_plugin_env(monkeypatch, tmp_path):
    """A user plugin dir with a platform plugin manifest and a decoy non-platform one."""
    home = tmp_path / ".hermes"
    plugins = home / "plugins"
    plugins.mkdir(parents=True)
    monkeypatch.setattr(config_mod, "get_hermes_home", lambda: home)

    xmpp_dir = plugins / "hermes-xmpp"
    xmpp_dir.mkdir()
    (xmpp_dir / "plugin.yaml").write_text(
        """
name: hermes-xmpp
label: XMPP/Jabber
kind: platform
version: 1.0.0
description: XMPP gateway adapter
requires_env:
  - name: XMPP_JID
    description: "The account JID"
    password: false
optional_env:
  - name: XMPP_ALLOWED_USERS
    description: "Comma-separated bare JIDs allowed to talk to the agent"
    password: false
  - name: XMPP_HOME_CHANNEL
    description: "Default JID for cron delivery"
    password: false
""",
        encoding="utf-8",
    )
    # Decoy: a user plugin that is NOT a platform — its env vars must not be injected.
    tool_dir = plugins / "my-tool"
    tool_dir.mkdir()
    (tool_dir / "plugin.yaml").write_text(
        """
name: my-tool
label: My Tool
kind: tool
version: 1.0.0
optional_env:
  - name: MY_TOOL_SETTING
    description: "Not a messaging var"
    password: false
""",
        encoding="utf-8",
    )
    yield home


def test_user_platform_plugin_env_vars_injected(user_plugin_env):
    """A user-installed kind:platform manifest's env entries reach OPTIONAL_ENV_VARS."""
    before = set(config_mod.OPTIONAL_ENV_VARS)
    config_mod._inject_platform_plugin_env_vars()
    injected = set(config_mod.OPTIONAL_ENV_VARS) - before
    assert {"XMPP_JID", "XMPP_ALLOWED_USERS", "XMPP_HOME_CHANNEL"} <= injected


def test_user_non_platform_plugin_env_vars_not_injected(user_plugin_env):
    """kind:tool user plugins stay out — the messaging catalog only surfaces platforms."""
    config_mod._inject_platform_plugin_env_vars()
    assert "MY_TOOL_SETTING" not in config_mod.OPTIONAL_ENV_VARS


def test_bundled_manifests_still_injected(user_plugin_env):
    """The bundled scan keeps working: at least one bundled platform var is present."""
    bundled_names = {
        "TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "SLACK_BOT_TOKEN",
    }
    present = {n for n in bundled_names if n in config_mod.OPTIONAL_ENV_VARS}
    # Bundled injection runs at import; if the environment strips it, run it again.
    if not present:
        config_mod._inject_platform_plugin_env_vars()
        present = {n for n in bundled_names if n in config_mod.OPTIONAL_ENV_VARS}
    assert present, "bundled platform manifests stopped being injected"