"""``sessions.show_subagents`` decides whether human-facing session lists re-admit delegate runs (#97202)."""

from hermes_cli.session_listing import subagent_listing_scope


def _home(tmp_path, show):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(f"sessions:\n  show_subagents: {str(show).lower()}\n")
    return home


def test_listing_scope_follows_the_store_home_config(tmp_path):
    assert subagent_listing_scope(_home(tmp_path, False)) == (False, None)
    assert subagent_listing_scope(_home(tmp_path, True)) == (True, None)


def test_listing_scope_drops_the_subagent_exclusion_from_the_recents_shape(tmp_path):
    home = _home(tmp_path, True)
    assert subagent_listing_scope(home, exclude_sources=["cron", "subagent"]) == (True, ["cron"])
    assert subagent_listing_scope(home, exclude_sources=["subagent"]) == (True, None)


def test_listing_scope_leaves_source_scoped_and_messaging_slices_alone(tmp_path):
    home = _home(tmp_path, True)
    assert subagent_listing_scope(home, source="cron") == (False, None)
    assert subagent_listing_scope(home, sources=["telegram"]) == (False, None)
    assert subagent_listing_scope(home, exclude_sources=["cron", "cli", "desktop"]) == (False, ["cron", "cli", "desktop"])
