"""Invariants for the shared adapter boilerplate in ``gateway/platforms/_shared.py``.

Contract under test: every plugin platform's env-driven enablement reads through the ONE
profile-scoped reader (never an ambient ``os.environ``/``get_env_value`` read), and the table-driven
``apply_yaml_bridge`` seeds ``extra`` for a secondary profile even though it must not write env.
"""

from __future__ import annotations

import importlib
import os

import pytest

import agent.secret_scope as ss
import gateway.platforms._shared as shared
import hermes_cli.config as cli_config

_ENV_ENABLEMENT_PLUGINS = ("buzz", "google_chat", "irc", "line", "ntfy", "photon", "simplex", "teams")


@pytest.fixture(autouse=True)
def _multiplex_off():
    prev = ss._MULTIPLEX_ACTIVE
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(prev)


@pytest.mark.parametrize("plugin", _ENV_ENABLEMENT_PLUGINS)
def test_env_enablement_reads_only_through_scoped_getter(plugin, monkeypatch):
    """A secondary profile is seeded only from its own scope: every var the hook consults must go through
    ``_shared.get_scoped_secret`` (a fake scope answers them) and never through a raw env read."""
    mod = importlib.import_module(f"plugins.platforms.{plugin}.adapter")
    # Anything the plugin resolves through raw env would be visible here; the scoped reader hides it.
    for var in list(os.environ):
        if var.startswith(plugin.upper().replace("GOOGLE_CHAT", "GOOGLE")):
            monkeypatch.delenv(var, raising=False)
    fake_scope = {
        "BUZZ_RELAY_URL": "wss://relay", "BUZZ_PRIVATE_KEY": "k" * 8, "BUZZ_CHANNELS": "a,b",
        "GOOGLE_CHAT_PROJECT_ID": "p", "GOOGLE_CHAT_SUBSCRIPTION_NAME": "s", "GOOGLE_CHAT_MAX_MESSAGES": "3",
        "IRC_SERVER": "irc.example", "IRC_CHANNEL": "#h", "IRC_PORT": "6697",
        "LINE_CHANNEL_ACCESS_TOKEN": "t", "LINE_CHANNEL_SECRET": "s", "LINE_PORT": "8081",
        "NTFY_TOPIC": "topic", "NTFY_TOKEN": "tok",
        "PHOTON_PROJECT_ID": "pid", "PHOTON_PROJECT_SECRET": "sec", "PHOTON_HOME_CHANNEL": "line-1",
        "SIMPLEX_WS_URL": "ws://x", "SIMPLEX_GROUP_ALLOWED": "g",
        "TEAMS_CLIENT_ID": "c", "TEAMS_CLIENT_SECRET": "s", "TEAMS_TENANT_ID": "t", "TEAMS_PORT": "3979",
    }
    scoped_hits: list[str] = []
    raw_hits: list[str] = []
    real = shared.get_scoped_secret

    def spy(name, default=None, **kw):
        scoped_hits.append(name)
        return fake_scope.get(name, default)

    monkeypatch.setattr(shared, "get_scoped_secret", spy)
    monkeypatch.setattr(shared, "_scoped_get_secret", lambda name, default=None: fake_scope.get(name, default))
    monkeypatch.setattr(cli_config, "get_env_value", lambda key: raw_hits.append(key))
    monkeypatch.setattr(os, "getenv", lambda key, default=None: raw_hits.append(key) or default)

    seed = mod._env_enablement()

    assert real is not shared.get_scoped_secret  # the spy really replaced the seam
    assert seed, f"{plugin}: fake scope holds the required vars, hook must enable"
    assert scoped_hits, f"{plugin}: enablement never touched the scoped reader"
    assert not raw_hits, f"{plugin}: raw env reads during enablement: {raw_hits}"


def test_home_channel_name_defaults_to_home_everywhere(monkeypatch):
    """One rule for ``home_channel.name`` across adapters: ``<HOME_ENV>_NAME`` else the literal ``Home``
    (irc/ntfy/buzz previously used the chat id as its own name)."""
    monkeypatch.setattr(shared, "get_scoped_secret", lambda n, d=None, **k: {"X_HOME": "room"}.get(n, d))
    seed = shared.seed_extra_from_env((), home_env="X_HOME")
    assert seed["home_channel"] == {"chat_id": "room", "name": "Home"}
    monkeypatch.setattr(shared, "get_scoped_secret", lambda n, d=None, **k: {"X_HOME": "room", "X_HOME_NAME": "Ops"}.get(n, d))
    assert shared.seed_extra_from_env((), home_env="X_HOME")["home_channel"]["name"] == "Ops"
    assert shared.seed_extra_from_env((), home_env="X_HOME", home_default="fallback")["home_channel"]["chat_id"] == "room"
    monkeypatch.setattr(shared, "get_scoped_secret", lambda n, d=None, **k: d)
    assert shared.seed_extra_from_env((), home_env="X_HOME", home_default="fallback")["home_channel"]["chat_id"] == "fallback"
    assert "home_channel" not in shared.seed_extra_from_env((), home_env="X_HOME")


def test_buzz_yaml_bridge_seeds_extra_for_a_secondary_profile(monkeypatch):
    """Under a secondary profile's scope the env write is skipped, so the hook MUST return the values for
    ``PlatformConfig.extra`` — a ``None`` return left a secondary Buzz profile with neither."""
    import plugins.platforms.buzz.adapter as buzz

    for var in ("BUZZ_RELAY_URL", "BUZZ_REPLY_TO_MODE", "BUZZ_CHANNELS", "BUZZ_REQUIRE_MENTION"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(shared, "profile_scoped", lambda: True)
    seeded = buzz._apply_yaml_config({}, {"extra": {
        "relay_url": "wss://p.relay", "reply_to_mode": "off", "channels": ["a", "b"], "require_mention": False}})
    assert seeded == {"relay_url": "wss://p.relay", "reply_to_mode": "off", "channels": ["a", "b"], "require_mention": False}
    assert not any(v in os.environ for v in ("BUZZ_RELAY_URL", "BUZZ_REPLY_TO_MODE", "BUZZ_CHANNELS", "BUZZ_REQUIRE_MENTION"))

    monkeypatch.setattr(shared, "profile_scoped", lambda: False)
    buzz._apply_yaml_config({}, {"extra": {"reply_to_mode": "off", "channels": ["a", "b"], "require_mention": False}})
    assert os.environ["BUZZ_REPLY_TO_MODE"] == "off"
    assert os.environ["BUZZ_CHANNELS"] == "a,b"
    assert os.environ["BUZZ_REQUIRE_MENTION"] == "false"
    for var in ("BUZZ_REPLY_TO_MODE", "BUZZ_CHANNELS", "BUZZ_REQUIRE_MENTION"):
        monkeypatch.delenv(var, raising=False)


def test_extra_or_secret_precedence_env_then_yaml_then_default(monkeypatch):
    """Explicit scoped env → the profile's YAML → default; a blank env value is unset (#108440, #109032)."""
    env: dict = {}
    monkeypatch.setattr(shared, "get_scoped_secret", lambda n, d=None, **k: env.get(n, d))
    # YAML alone: explicit False is a real value; blank/None fall to the default.
    assert shared.extra_or_secret({"require_mention": False}, "require_mention", "X", "true") is False
    assert shared.extra_or_secret({"require_mention": ""}, "require_mention", "X", "true") == "true"
    assert shared.extra_or_secret(None, "require_mention", "X", "true") == "true"
    # Readers where a blank YAML value means "cleared" (channel whitelists) keep it as a value.
    assert shared.extra_or_secret({"allowed_channels": ""}, "allowed_channels", "X", "dflt", blank_is_unset=False) == ""
    assert shared.extra_or_secret({}, "allowed_channels", "X", "dflt", blank_is_unset=False) == "dflt"
    # An explicit env value beats YAML in either direction; a blank env value does not.
    env["X"] = "false"
    assert shared.extra_or_secret({"reactions": True}, "reactions", "X", "true") == "false"
    env["X"] = "true"
    assert shared.extra_or_secret({"reactions": False}, "reactions", "X", "false") == "true"
    env["X"] = "  "
    assert shared.extra_or_secret({"reactions": False}, "reactions", "X", "true") is False


def test_extra_or_secret_scoped_miss_never_reads_launch_env(monkeypatch):
    """Under a secondary's scope the launch process's os.environ is another profile's value: a miss
    falls to the secondary's OWN YAML, then the default — never to os.environ."""
    monkeypatch.setenv("X_FLAG", "launch-value")
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({})
    try:
        assert shared.extra_or_secret({"flag": "yaml-value"}, "flag", "X_FLAG", "dflt") == "yaml-value"
        assert shared.extra_or_secret({}, "flag", "X_FLAG", "dflt") == "dflt"
    finally:
        ss.reset_secret_scope(token)
    # Unscoped (single-profile / default profile): env-over-YAML exactly as documented.
    ss.set_multiplex_active(False)
    assert shared.extra_or_secret({"flag": "yaml-value"}, "flag", "X_FLAG", "dflt") == "launch-value"


def test_external_fallback_consults_profile_scope_only_when_unscoped(monkeypatch):
    """Buzz's startup gate rung: a Bitwarden-managed key is visible through ``external_fallback`` before any
    scope exists, but never overrides an installed (authoritative) scope."""
    monkeypatch.delenv("K", raising=False)
    monkeypatch.setattr(shared, "_UNSCOPED_PROFILE_SECRETS", {"K": "external"})
    assert shared.get_scoped_secret("K") is None
    assert shared.get_scoped_secret("K", external_fallback=True) == "external"
    ss.set_multiplex_active(True)
    token = ss.set_secret_scope({"OTHER": "x"})
    try:
        assert shared.get_scoped_secret("K", "dflt", external_fallback=True) == "dflt"
    finally:
        ss.reset_secret_scope(token)


def test_yaml_env_setter_encodes_every_collection_shape_as_csv(monkeypatch):
    """Every collection shape the adapter readers accept must CSV-join, not fall through to ``str()``:
    the Slack ``reaction_triggers`` reader takes list/tuple/set, so a tuple/set bridged to env as
    Python repr (``"('eyes', 'rocket')"``) split into punctuated trigger names (#109900). Sets sort
    by text so the env value is deterministic regardless of iteration order."""
    set_env = shared.yaml_env_setter()
    os.environ.pop("X_SHAPES", None)  # pop, not monkeypatch.delenv: the setter's own writes must not be snapshotted for teardown restore
    set_env("X_SHAPES", ["eyes", "rocket"])
    assert os.environ["X_SHAPES"] == "eyes,rocket"
    for value in (("eyes", "rocket"), {"rocket", "eyes"}, frozenset({"rocket", "eyes"})):
        os.environ.pop("X_SHAPES", None)
        set_env("X_SHAPES", value)
        assert os.environ["X_SHAPES"] == "eyes,rocket", f"{type(value).__name__} must encode as sorted CSV"
    os.environ.pop("X_SHAPES", None)
    set_env("X_SHAPES", "plain")
    assert os.environ["X_SHAPES"] == "plain"
    os.environ.pop("X_SHAPES", None)
    set_env("X_SHAPES", None)  # None never writes
    assert "X_SHAPES" not in os.environ


def test_slack_yaml_bridge_reaction_triggers_round_trip(monkeypatch):
    """Adapter-level round-trip for every supported collection shape (#109900): the Slack bridge writes
    SLACK_REACTION_TRIGGERS, and a process that reads the var back (no ``extra`` seeded — e.g. a child
    process that only inherited env) must recover exactly the trigger names."""
    import plugins.platforms.slack.adapter as slack
    from gateway.config import Platform, PlatformConfig

    monkeypatch.delenv("SLACK_REACTION_TRIGGERS", raising=False)
    adapter = object.__new__(slack.SlackAdapter)
    adapter.platform = Platform.SLACK
    adapter.config = PlatformConfig(enabled=True)

    for value in (["eyes", "rocket"], ("eyes", "rocket"), {"rocket", "eyes"}):
        os.environ.pop("SLACK_REACTION_TRIGGERS", None)  # pop: the bridge's own writes must not be snapshotted for teardown restore
        seeded = slack._apply_yaml_config({}, {"reaction_triggers": value})
        assert seeded == {"reaction_triggers": value}
        assert adapter._slack_reaction_triggers() == {"eyes", "rocket"}
    os.environ.pop("SLACK_REACTION_TRIGGERS", None)
