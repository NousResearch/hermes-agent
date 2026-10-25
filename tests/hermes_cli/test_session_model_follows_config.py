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

import pytest

# Imported at module scope (aliased: the stub locals below are named ``cli``):
# ``patch.dict("cli.CLI_CONFIG", ...)`` would otherwise resolve the dotted string at
# call time, forcing a fresh ``cli`` import *inside* the test's file-I/O guard —
# which the CLI's startup recovery probe trips.
import cli as cli_module

from hermes_cli.cli_model_switch_mixin import (
    SESSION_MODEL_POLICY_FOLLOW_CONFIG,
    SESSION_MODEL_POLICY_NEVER_PIN,
    CLIModelSwitchMixin,
    SessionModelPolicyError,
    resolve_session_model_policy,
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


# --------------------------------------------------------------------------- #
# the pin must not RETURN: the repair leaves a row that is still a cache
# --------------------------------------------------------------------------- #

def _apply_patch(config: dict, patch: dict) -> dict:
    """Emulate ``SessionDB._merge_model_config_json``: ``None`` deletes a key."""
    out = dict(config)
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
    return out


def test_the_repair_deletes_the_stale_route_instead_of_restamping_it():
    """The repair must not write the route it just rejected: a stamp IS provenance, so
    restamping would make the row a permanent "deliberate pick" and hand the pin back."""
    db = _FakeDB()
    cli = _cli(model="deepseek-flash", db=db)
    row = _cache_row(model_config={"provider": "zai", "base_url": "https://api.z.ai/v1",
                                   "gateway_runtime": {"provider": "zai"}})
    with patch.dict(cli_module.CLI_CONFIG, _CFG, clear=False):
        CLIModelSwitchMixin._restore_session_model(cli, row, quiet=True)
    assert cli.model == "deepseek-flash"
    sid, repaired = db.config_patches[-1]
    assert sid == "s1"
    assert repaired.get("provider") is None, "the stale route must be deleted, never re-stamped"
    assert repaired.get("base_url") is None
    assert repaired.get("gateway_runtime") is None
    repaired_row = dict(row, model="deepseek-flash",
                        model_config=_apply_patch(row["model_config"], repaired))
    assert stored_model_is_config_cache(repaired_row) is True, \
        "a repaired row carries no provenance, so it stays a cache"


def test_a_repaired_row_reconciles_again_when_the_config_moves():
    """Second config change, same row: the pin must not come back through the repair.

    An interactive source with no route is a cache; after the repair it must STILL be one —
    so this fails if the repair stamps the provider it just rejected.
    """
    db = _FakeDB()
    cli = _cli(model="deepseek-flash", db=db)
    row = _cache_row(source="cli")
    with patch.dict(cli_module.CLI_CONFIG, _CFG, clear=False):
        CLIModelSwitchMixin._restore_session_model(cli, row, quiet=True)
    _, repaired = db.config_patches[-1]
    row = dict(row, model="deepseek-flash",
               model_config=_apply_patch(row["model_config"], repaired))
    moved = {"model": {"provider": "deepseek", "default": "deepseek-v4-lite"}}
    cli2 = _cli(model="deepseek-v4-lite", db=db)
    with patch.dict(cli_module.CLI_CONFIG, moved, clear=False):
        CLIModelSwitchMixin._restore_session_model(cli2, row, quiet=True)
    assert cli2.model == "deepseek-v4-lite"
    assert db.model_writes[-1] == ("s1", "deepseek-v4-lite")


# --------------------------------------------------------------------------- #
# every non-interactive source is a cache, route key or not
# --------------------------------------------------------------------------- #

_NON_INTERACTIVE_SOURCES = ("kanban", "cron", "oneshot", "bot_peer_dm", "peer", "webhook",
                            "delegation", "subagent", "background_review")


def test_a_stray_route_key_cannot_rescue_a_non_interactive_row():
    for source in _NON_INTERACTIVE_SOURCES:
        row = {"model": _OLD_DEFAULT, "source": source,
               "model_config": {"provider": "zai", "base_url": "https://api.z.ai/v1",
                                "gateway_runtime": {"provider": "zai"}}}
        assert stored_model_is_config_cache(row) is True, source


def test_a_non_interactive_row_with_a_route_is_reconciled_on_resume():
    db = _FakeDB()
    cli = _cli(model="deepseek-flash", db=db)
    row = _cache_row(model_config={"provider": "zai"})
    with patch.dict(cli_module.CLI_CONFIG, _CFG, clear=False):
        CLIModelSwitchMixin._restore_session_model(cli, row, quiet=True)
    assert cli.model == "deepseek-flash"
    assert db.model_writes == [("s1", "deepseek-flash")]


def test_a_resumed_one_shot_never_restores_a_stored_pin():
    """One-shot IS a non-interactive source: no user was there to pick, so the stored
    route is a cache like any other — the ambient/config choice stands."""
    from hermes_cli.oneshot import _ModelChoice, _apply_stored_session_runtime

    choice = _ModelChoice(model="deepseek-flash", provider="deepseek", api_key="k")
    row = {"model": "zai/glm-5.1", "source": "oneshot",
           "model_config": {"provider": "zai", "base_url": "https://api.z.ai/v1"}}
    out = _apply_stored_session_runtime(choice, row, explicit_model=False)
    assert (out.model, out.provider) == ("deepseek-flash", "deepseek")


def test_an_explicit_one_shot_model_still_wins():
    from hermes_cli.oneshot import _ModelChoice, _apply_stored_session_runtime

    choice = _ModelChoice(model="deepseek-flash", provider="deepseek", api_key="k")
    row = {"model": "zai/glm-5.1", "source": "desktop", "model_config": {"provider": "zai"}}
    out = _apply_stored_session_runtime(choice, row, explicit_model=True)
    assert out.model == "deepseek-flash"


# --------------------------------------------------------------------------- #
# session.model_policy (config knob, ruling R3)
# --------------------------------------------------------------------------- #

_NEVER_PIN_CFG = {**_CFG, "session": {"model_policy": SESSION_MODEL_POLICY_NEVER_PIN}}


def test_the_default_policy_is_follow_config():
    assert resolve_session_model_policy({}) == SESSION_MODEL_POLICY_FOLLOW_CONFIG
    assert resolve_session_model_policy({"session": {}}) == SESSION_MODEL_POLICY_FOLLOW_CONFIG
    assert resolve_session_model_policy(
        {"session": {"model_policy": None}}) == SESSION_MODEL_POLICY_FOLLOW_CONFIG


def test_never_pin_is_read_from_the_session_section():
    assert resolve_session_model_policy(
        {"session": {"model_policy": "never_pin"}}) == SESSION_MODEL_POLICY_NEVER_PIN
    assert resolve_session_model_policy(
        {"session": {"model_policy": "  NEVER_PIN  "}}) == SESSION_MODEL_POLICY_NEVER_PIN


def test_an_unknown_policy_is_refused_rather_than_falling_back():
    with pytest.raises(SessionModelPolicyError):
        resolve_session_model_policy({"session": {"model_policy": "sometimes"}})
    with pytest.raises(SessionModelPolicyError):
        resolve_session_model_policy({"session": {"model_policy": ""}})


def test_the_config_validator_reports_an_unknown_policy_as_an_error():
    from hermes_cli.config import validate_config_structure

    issues = validate_config_structure({"session": {"model_policy": "sometimes"}})
    flagged = [i for i in issues if "session.model_policy" in i.message]
    assert flagged, "an unknown policy must be flagged, not silently ignored"
    assert all(i.severity == "error" for i in flagged)


def test_a_known_policy_passes_the_config_validator():
    from hermes_cli.config import validate_config_structure

    for value in ("follow_config", "never_pin"):
        issues = validate_config_structure({"session": {"model_policy": value}})
        assert not [i for i in issues if "session.model_policy" in i.message], value


def test_the_knob_is_registered_in_the_canonical_config_defaults():
    """Every reader has a registry entry: the knob is declared in DEFAULT_CONFIG and in the
    CLI loader's own defaults, so it inherits per-lane exactly like model.default."""
    from hermes_cli.cli_config_load import _cli_config_defaults
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["session"]["model_policy"] == SESSION_MODEL_POLICY_FOLLOW_CONFIG
    assert _cli_config_defaults()["session"]["model_policy"] == SESSION_MODEL_POLICY_FOLLOW_CONFIG


def test_a_temp_config_yaml_carries_the_knob_through_the_cli_loader(tmp_path, monkeypatch):
    import hermes_yaml as yaml

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cli_module, "_hermes_home", home)
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "deepseek-flash", "provider": "deepseek"},
        "session": {"model_policy": "never_pin"}}))
    cfg = cli_module.load_cli_config()
    assert resolve_session_model_policy(cfg) == SESSION_MODEL_POLICY_NEVER_PIN


def test_never_pin_makes_every_row_a_cache_even_a_deliberate_pick():
    row = {"model": "zai/glm-5.1", "source": "desktop",
           "model_config": {"provider": "zai", "base_url": "https://api.z.ai/v1"}}
    assert stored_model_is_config_cache(row) is False
    assert stored_model_is_config_cache(
        row, policy=SESSION_MODEL_POLICY_NEVER_PIN) is True


def test_never_pin_overrides_a_deliberate_pick_on_resume():
    db = _FakeDB()
    cli = _cli(model="deepseek-flash", db=db)
    cli.base_url = None
    row = {"model": "zai/glm-5.1", "source": "desktop",
           "model_config": {"provider": "zai", "base_url": "https://api.z.ai/v1"}}
    with patch.dict(cli_module.CLI_CONFIG, _NEVER_PIN_CFG, clear=False):
        CLIModelSwitchMixin._restore_session_model(cli, row, quiet=True)
    assert cli.model == "deepseek-flash"
    assert db.model_writes == [("s1", "deepseek-flash")]


def test_never_pin_with_no_configured_default_leaves_the_resolved_model_alone():
    db = _FakeDB()
    cli = _cli(model="deepseek-flash", db=db)
    row = {"model": "zai/glm-5.1", "source": "desktop", "model_config": {"provider": "zai"}}
    cfg = {"model": {}, "session": {"model_policy": SESSION_MODEL_POLICY_NEVER_PIN}}
    with patch.dict(cli_module.CLI_CONFIG, cfg, clear=False):
        CLIModelSwitchMixin._restore_session_model(cli, row, quiet=True)
    assert cli.model == "deepseek-flash", "never_pin must never fall back to the stored pin"


def test_follow_config_still_honours_a_deliberate_pick_under_the_knob():
    db = _FakeDB()
    cli = _cli(model="deepseek-flash", db=db)
    cli.base_url = None
    row = {"model": "zai/glm-5.1", "source": "desktop",
           "model_config": {"gateway_runtime": {"provider": "zai", "base_url": "https://api.z.ai/v1"}}}
    cfg = {**_CFG, "session": {"model_policy": SESSION_MODEL_POLICY_FOLLOW_CONFIG}}
    with patch.dict(cli_module.CLI_CONFIG, cfg, clear=False):
        CLIModelSwitchMixin._restore_session_model(cli, row, quiet=True)
    assert cli.model == "zai/glm-5.1"
    assert db.model_writes == []
