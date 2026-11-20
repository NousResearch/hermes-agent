"""``sessions.show_subagents`` decides whether human-facing session lists re-admit delegate runs (#97202).

The key takes ``false`` (none), ``"active"`` (only runs still in flight) or ``true`` (all).
"""

import pytest

from hermes_cli.session_listing import (
    RUNNING_SUBAGENTS, resolve_subagent_sidebar_mode, show_subagent_sessions, subagent_include_marker,
    subagent_listing_scope, subagent_sidebar_mode,
)


def _home(tmp_path, value):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    rendered = value if isinstance(value, str) else str(value).lower()
    (home / "config.yaml").write_text(f"sessions:\n  show_subagents: {rendered}\n")
    return home


@pytest.mark.parametrize(("value", "mode"), [
    (False, "none"), (None, "none"), ("", "none"), ("false", "none"), ("off", "none"), ("bogus", "none"),
    ("active", "active"), (" Active ", "active"),
    (True, "all"), ("true", "all"), ("yes", "all"), (1, "all"),
])
def test_show_subagents_value_resolves_to_a_mode(value, mode):
    assert resolve_subagent_sidebar_mode(value) == mode


def test_boolean_configs_keep_their_listing(tmp_path):
    # A config written before "active" existed reads exactly as before.
    assert subagent_sidebar_mode(_home(tmp_path, True)) == "all"
    assert show_subagent_sessions(_home(tmp_path, True)) is True
    assert subagent_sidebar_mode(_home(tmp_path, False)) == "none"
    assert show_subagent_sessions(_home(tmp_path, False)) is False


def test_unset_key_hides_subagents(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("model: m\n")
    assert subagent_sidebar_mode(home) == "none"
    assert subagent_listing_scope(home, exclude_sources=["cron", "subagent"]) == (False, ["cron", "subagent"])


def test_include_marker_per_mode(tmp_path):
    assert subagent_include_marker(_home(tmp_path, False)) is False
    assert subagent_include_marker(_home(tmp_path, "active")) == RUNNING_SUBAGENTS
    assert subagent_include_marker(_home(tmp_path, True)) is True


def test_listing_scope_follows_the_store_home_config(tmp_path):
    assert subagent_listing_scope(_home(tmp_path, False)) == (False, None)
    assert subagent_listing_scope(_home(tmp_path, False), exclude_sources=["cron", "subagent"]) == (
        False, ["cron", "subagent"])
    assert subagent_listing_scope(_home(tmp_path, "active")) == (RUNNING_SUBAGENTS, None)
    assert subagent_listing_scope(_home(tmp_path, "active"), exclude_sources=["cron", "subagent"]) == (
        RUNNING_SUBAGENTS, ["cron"])
    assert subagent_listing_scope(_home(tmp_path, True)) == (True, None)


def test_listing_scope_drops_the_subagent_exclusion_from_the_recents_shape(tmp_path):
    home = _home(tmp_path, True)
    assert subagent_listing_scope(home, exclude_sources=["cron", "subagent"]) == (True, ["cron"])
    assert subagent_listing_scope(home, exclude_sources=["subagent"]) == (True, None)


@pytest.mark.parametrize("value", [True, "active"])
def test_listing_scope_leaves_source_scoped_and_messaging_slices_alone(tmp_path, value):
    home = _home(tmp_path, value)
    assert subagent_listing_scope(home, source="cron") == (False, None)
    assert subagent_listing_scope(home, sources=["telegram"]) == (False, None)
    assert subagent_listing_scope(home, exclude_sources=["cron", "cli", "desktop"]) == (False, ["cron", "cli", "desktop"])
