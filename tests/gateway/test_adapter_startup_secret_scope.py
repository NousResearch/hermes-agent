"""Regression tests — Slack-pattern scoped credential reads at adapter startup.

Class-closure follow-up to the profile secret-scope cluster (#76462, Slack
pattern #59739, WhatsApp ``_get_wsecret``): the 13 platform adapters below
read credentials at ``__init__`` / availability-check / standalone-send time
with bare ``os.getenv`` (SMS even used ``os.environ[...]``, which KeyErrors).
Under ``gateway.multiplex_profiles`` those reads leak the default profile's
credential into secondary profiles' adapters.

Each migrated module now carries a module-level ``_get_scoped_secret`` helper
mirroring ``gateway/platforms/whatsapp_common.py::_get_wsecret``:

- scope installed → scope is authoritative; scoped miss returns the default
  (NO borrow from ``os.environ``)
- unscoped under multiplex (the default profile's own startup loop) →
  fall back to ``os.getenv`` without raising ``UnscopedSecretError``

The helpers are imported and exercised directly rather than constructing every
adapter — construction pulls heavy platform dependencies (slack-bolt-style SDK
imports, webhook servers, sidecars) that the test environment doesn't need.
"""

import importlib

import pytest

from agent import secret_scope as ss

# (module path, representative credential env var owned by that adapter)
MIGRATED_ADAPTER_MODULES = [
    ("plugins.platforms.irc.adapter", "IRC_SERVER_PASSWORD"),
    ("plugins.platforms.line.adapter", "LINE_CHANNEL_ACCESS_TOKEN"),
    ("plugins.platforms.teams.adapter", "TEAMS_CLIENT_SECRET"),
    ("plugins.platforms.mattermost.adapter", "MATTERMOST_TOKEN"),
    ("plugins.platforms.ntfy.adapter", "NTFY_TOKEN"),
    ("plugins.platforms.homeassistant.adapter", "HASS_TOKEN"),
    ("plugins.platforms.sms.adapter", "TWILIO_AUTH_TOKEN"),
    ("plugins.platforms.dingtalk.adapter", "DINGTALK_CLIENT_SECRET"),
    ("plugins.platforms.feishu.adapter", "FEISHU_APP_SECRET"),
    ("plugins.platforms.wecom.adapter", "WECOM_SECRET"),
    ("plugins.platforms.photon.adapter", "PHOTON_PROJECT_SECRET"),
    ("plugins.platforms.photon.auth", "PHOTON_PROJECT_SECRET"),
    ("plugins.platforms.buzz.adapter", "BUZZ_PRIVATE_KEY"),
    ("gateway.platforms.bluebubbles", "BLUEBUBBLES_PASSWORD"),
    ("gateway.platforms.api_server", "API_SERVER_KEY"),
]

MODULE_IDS = [m for m, _ in MIGRATED_ADAPTER_MODULES]


@pytest.fixture(autouse=True)
def _reset_multiplex():
    """Multiplex mode off before and after every test."""
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


def _helper(module_name):
    mod = importlib.import_module(module_name)
    helper = getattr(mod, "_get_scoped_secret", None)
    assert helper is not None, (
        f"{module_name} must define the module-level _get_scoped_secret helper "
        "(Slack pattern #59739 / whatsapp_common._get_wsecret)"
    )
    return helper


@pytest.mark.parametrize(("module_name", "var"), MIGRATED_ADAPTER_MODULES, ids=MODULE_IDS)
def test_helper_exists(module_name, var):
    _helper(module_name)


@pytest.mark.parametrize(("module_name", "var"), MIGRATED_ADAPTER_MODULES, ids=MODULE_IDS)
def test_scoped_read_wins_over_environ(module_name, var, monkeypatch):
    """Scope installed under multiplex: the profile's value wins, not environ's."""
    helper = _helper(module_name)
    monkeypatch.setenv(var, "default-profile-value")
    ss.set_multiplex_active(True)
    tok = ss.set_secret_scope({var: "secondary-profile-value"})
    try:
        assert helper(var) == "secondary-profile-value"
    finally:
        ss.reset_secret_scope(tok)


@pytest.mark.parametrize(("module_name", "var"), MIGRATED_ADAPTER_MODULES, ids=MODULE_IDS)
def test_scoped_miss_returns_default_no_environ_borrow(module_name, var, monkeypatch):
    """Scope installed but key absent: return the default — never borrow the
    default profile's os.environ value into a secondary profile."""
    helper = _helper(module_name)
    monkeypatch.setenv(var, "default-profile-value")
    ss.set_multiplex_active(True)
    tok = ss.set_secret_scope({"SOME_OTHER_KEY": "x"})
    try:
        assert helper(var) is None
        assert helper(var, "") == ""
        assert helper(var, "sentinel") == "sentinel"
    finally:
        ss.reset_secret_scope(tok)


@pytest.mark.parametrize(("module_name", "var"), MIGRATED_ADAPTER_MODULES, ids=MODULE_IDS)
def test_unscoped_under_multiplex_falls_back_to_environ(module_name, var, monkeypatch):
    """The DEFAULT profile constructs its adapters unscoped under multiplexing.
    A bare get_secret would raise UnscopedSecretError and crash startup; the
    helper must fall back to os.environ (that profile's own value) instead."""
    helper = _helper(module_name)
    monkeypatch.setenv(var, "default-profile-own-value")
    ss.set_multiplex_active(True)
    assert ss.current_secret_scope() is None
    assert helper(var) == "default-profile-own-value"

    monkeypatch.delenv(var, raising=False)
    assert helper(var, "fallback-default") == "fallback-default"


@pytest.mark.parametrize(("module_name", "var"), MIGRATED_ADAPTER_MODULES, ids=MODULE_IDS)
def test_single_profile_legacy_environ_read(module_name, var, monkeypatch):
    """Multiplex off, no scope: legacy os.environ read keeps working."""
    helper = _helper(module_name)
    monkeypatch.setenv(var, "legacy-env-value")
    assert helper(var) == "legacy-env-value"
    monkeypatch.delenv(var, raising=False)
    assert helper(var, "d") == "d"
