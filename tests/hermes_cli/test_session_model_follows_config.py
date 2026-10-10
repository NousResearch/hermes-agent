"""A session's effective model follows its profile's CONFIG, unless a user picked it.

``sessions.model`` is written at creation from whatever ``model.default`` said at
that instant, so a config change leaves every existing row pinned to the OLD
default. A profile's ``config.yaml`` briefly carried a different ``model.default``
and every session built in the window kept billing it after the config was
reverted.

These tests pin the reconciliation at the resume seam:

  * a stored model that is only a CACHE of the config default is NOT restored —
    the profile's current default wins, and the row is repaired so the stale pin
    does not come back on the next resume;
  * a DELIBERATE pick (a row carrying a recorded route) still survives, and a
    launch ``-m`` still wins outright — this guard must not break user intent;
  * ``follow_profile_config`` (the gateway's own "this chat follows config"
    marker) makes a row a cache by declaration.

The classification function is pure and is pinned directly as well as through the
resume path.
"""

from types import SimpleNamespace
from unittest.mock import patch

# Imported at module scope (aliased: the stub locals below are named ``cli``):
# ``patch.dict("cli.CLI_CONFIG", ...)`` would otherwise resolve the dotted string at
# call time, forcing a fresh ``cli`` import *inside* the test's file-I/O guard —
# which the CLI's startup recovery probe trips.
import cli as cli_module

from hermes_cli.cli_model_switch_mixin import (
    CLIModelSwitchMixin,
    stored_model_is_config_cache,
)

# Config now says flash. A session created while config said v4-pro still caches v4-pro.
_CFG = {"model": {"provider": "deepseek", "default": "deepseek-flash"}}
_OLD_DEFAULT = "deepseek-v4-pro"


def _cli(model="deepseek-flash", provider="deepseek", agent=None, explicit_m=False, db=None):
    printed = []
    return SimpleNamespace(
        model=model, provider=provider, requested_provider=provider, api_key="k",
        base_url="https://api.deepseek.com/v1", api_mode="chat_completions",
        _explicit_api_key=None, _explicit_base_url=None, _explicit_model_override=explicit_m,
        agent=agent, _credential_pool=None, reasoning_config={"enabled": True, "effort": "high"},
        _session_db=db, session_id="s1",
        _console_print=lambda *_a, **_k: printed.append(_a), _printed=printed)


class _FakeDB:
    """Records what the repair writes, so the pin is provably fixed, not merely ignored."""

    def __init__(self):
        self.model_writes = []
        self.config_patches = []

    def update_session_model(self, sid, model):
        self.model_writes.append((sid, model))

    def patch_session_model_config(self, sid, patch):
        self.config_patches.append((sid, patch))

    def get_session(self, sid):
        return None

    def reopen_session(self, sid):
        pass


def _cache_row(**extra):
    """The live shape: a kanban row that cached the config default and records no route."""
    row = {"model": _OLD_DEFAULT, "source": "kanban",
           "model_config": {"max_iterations": 500,
                            "reasoning_config": {"enabled": True, "effort": "high"}}}
    row.update(extra)
    return row


# --------------------------------------------------------------------------- #
# the classification (pure)
# --------------------------------------------------------------------------- #

def test_a_kanban_row_with_no_recorded_route_is_a_config_cache():
    assert stored_model_is_config_cache(_cache_row()) is True


def test_a_row_with_a_recorded_route_is_a_deliberate_pick():
    assert stored_model_is_config_cache(
        {"model": "zai/glm-5.1", "source": "desktop",
         "model_config": {"provider": "zai", "base_url": "https://api.z.ai/v1"}}) is False
    assert stored_model_is_config_cache(
        {"model": "zai/glm-5.1", "source": "cli",
         "model_config": {"gateway_runtime": {"provider": "zai"}}}) is False


def test_follow_profile_config_marks_a_row_a_cache_even_with_a_route():
    assert stored_model_is_config_cache(
        {"model": "x", "source": "desktop",
         "model_config": {"follow_profile_config": True, "provider": "zai"}}) is True


def test_every_non_interactive_source_is_a_cache():
    for source in ("kanban", "cron", "oneshot", "bot_peer_dm", "peer", "webhook"):
        assert stored_model_is_config_cache({"model": "x", "source": source}) is True, source


def test_a_row_whose_model_config_is_a_json_string_is_parsed():
    assert stored_model_is_config_cache(
        {"model": "x", "source": "desktop", "model_config": '{"provider": "zai"}'}) is False


# --------------------------------------------------------------------------- #
# the resume seam
# --------------------------------------------------------------------------- #

def test_a_stale_cached_pin_is_reconciled_to_the_config_default_and_repaired():
    db = _FakeDB()
    cli = _cli(model="deepseek-flash", db=db)
    with patch.dict(cli_module.CLI_CONFIG, _CFG, clear=False):
        CLIModelSwitchMixin._restore_session_model(cli, _cache_row(), quiet=True)
    assert cli.model == "deepseek-flash", "config must win over the stale cache"
    assert db.model_writes == [("s1", "deepseek-flash")], "the stale pin must be repaired"


def test_a_deliberate_pick_still_survives_resume():
    db = _FakeDB()
    cli = _cli(model="deepseek-flash", db=db)
    cli.base_url = None
    row = {"model": "zai/glm-5.1", "source": "desktop",
           "model_config": {"gateway_runtime": {"provider": "zai", "base_url": "https://api.z.ai/v1"}}}
    with patch.dict(cli_module.CLI_CONFIG, _CFG, clear=False):
        CLIModelSwitchMixin._restore_session_model(cli, row, quiet=True)
    assert cli.model == "zai/glm-5.1", "an explicit pick is user intent and must not be clobbered"
    assert db.model_writes == []


def test_a_launch_model_flag_still_wins_outright():
    db = _FakeDB()
    cli = _cli(model="some/explicit", db=db, explicit_m=True)
    with patch.dict(cli_module.CLI_CONFIG, _CFG, clear=False):
        CLIModelSwitchMixin._restore_session_model(cli, _cache_row(), quiet=True)
    assert cli.model == "some/explicit"
    assert db.model_writes == []


def test_follow_profile_config_on_a_desktop_chat_is_reconciled():
    db = _FakeDB()
    cli = _cli(model="deepseek-flash", db=db)
    row = _cache_row(source="desktop")
    row["model_config"]["follow_profile_config"] = True
    with patch.dict(cli_module.CLI_CONFIG, _CFG, clear=False):
        CLIModelSwitchMixin._restore_session_model(cli, row, quiet=True)
    assert cli.model == "deepseek-flash"


def test_a_matching_pin_is_not_touched():
    db = _FakeDB()
    cli = _cli(model="deepseek-flash", db=db)
    row = _cache_row(model="deepseek-flash")
    with patch.dict(cli_module.CLI_CONFIG, _CFG, clear=False):
        CLIModelSwitchMixin._restore_session_model(cli, row, quiet=True)
    assert cli.model == "deepseek-flash"
    assert db.model_writes == [], "nothing to repair when the cache already matches config"


def test_the_reconcile_warns_rather_than_being_silent():
    cli = _cli(model="deepseek-flash", db=None)
    with patch.dict(cli_module.CLI_CONFIG, _CFG, clear=False):
        CLIModelSwitchMixin._restore_session_model(cli, _cache_row(), quiet=False)
    assert cli._printed, "a config change that overrides a pin must be announced, never silent"
