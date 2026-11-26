"""``sessions.show_oneshot`` decides whether human-facing session lists re-admit finite one-shot runs.

The opt-in mirrors ``sessions.show_subagents`` (#97202): one flag per internal source, read from the
store's OWN profile config, dropping that source from the exclusion the surface sent.
"""

from hermes_cli.session_listing import (
    session_listing_scope, show_oneshot_sessions, subagent_listing_scope,
)
from persistence.conversation.sessions import INTERNAL_LISTING_SOURCES


def _home(tmp_path, **sessions):
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    body = "".join(f"  {key}: {str(value).lower()}\n" for key, value in sessions.items())
    (home / "config.yaml").write_text(f"sessions:\n{body}" if body else "sessions: {}\n")
    return home


def test_oneshot_is_off_by_default_and_on_when_the_store_opts_in(tmp_path):
    assert show_oneshot_sessions(_home(tmp_path, show_oneshot=False)) is False
    assert show_oneshot_sessions(_home(tmp_path, show_oneshot=True)) is True
    # An unreadable/absent config keeps the default shape rather than failing the listing.
    assert show_oneshot_sessions(tmp_path / "missing") is False


def test_the_recents_shape_drops_the_oneshot_exclusion_when_the_flag_is_on(tmp_path):
    home = _home(tmp_path, show_oneshot=True)
    # The desktop recents shape: every local source it hides, one-shot runs included.
    recents = ["acp", "cron", "kanban", "oneshot", "subagent", "tool"]
    assert session_listing_scope(home, exclude_sources=recents) == (
        False, True, ["acp", "cron", "kanban", "subagent", "tool"])
    # Excluding only the one-shot source leaves nothing to exclude.
    assert session_listing_scope(home, exclude_sources=["oneshot"])[2] is None
    # No exclusions at all (the CLI/TUI listing): nothing to drop, the flag alone admits them.
    assert session_listing_scope(home, exclude_sources=None) == (False, True, None)


def test_the_flag_is_ignored_when_the_exclusion_does_not_name_oneshot(tmp_path):
    """The per-platform messaging slice hides local sources but not one-shot runs: its shape stands."""
    home = _home(tmp_path, show_oneshot=True)
    messaging = ["cron", "cli", "desktop"]
    assert session_listing_scope(home, exclude_sources=messaging) == (False, False, messaging)


def test_source_scoped_slices_are_never_widened(tmp_path):
    home = _home(tmp_path, show_oneshot=True)
    assert session_listing_scope(home, source="cron") == (False, False, None)
    assert session_listing_scope(home, sources=["telegram"]) == (False, False, None)


def test_the_two_flags_are_independent(tmp_path):
    """Only one opt-in on: the other source keeps its exclusion, so the surfaces stay separable."""
    oneshot_only = _home(tmp_path / "a", show_oneshot=True, show_subagents=False)
    assert session_listing_scope(oneshot_only, exclude_sources=["oneshot", "subagent"]) == (
        False, True, ["subagent"])

    subagent_only = _home(tmp_path / "b", show_oneshot=False, show_subagents=True)
    assert session_listing_scope(subagent_only, exclude_sources=["oneshot", "subagent"]) == (
        True, False, ["oneshot"])

    both = _home(tmp_path / "c", show_oneshot=True, show_subagents=True)
    assert session_listing_scope(both, exclude_sources=["oneshot", "subagent", "cron"]) == (
        True, True, ["cron"])


def test_without_either_flag_the_scope_is_unchanged(tmp_path):
    bare = _home(tmp_path)
    recents = ["cron", "oneshot", "subagent"]
    assert session_listing_scope(bare, exclude_sources=recents) == (False, False, recents)


def test_subagent_scope_keeps_its_two_value_contract(tmp_path):
    """The #97202 helper is left exactly as it was: show_oneshot must not leak into it."""
    home = _home(tmp_path, show_oneshot=True)
    recents = ["cron", "oneshot", "subagent"]
    assert subagent_listing_scope(home, exclude_sources=recents) == (False, recents)
    assert subagent_listing_scope(home, source="cron") == (False, None)


def test_the_recents_shape_the_surfaces_really_send_readmits_the_oneshot_run(tmp_path):
    """Criterion 6(b) against the shape the desktop/TUI actually ship.

    The surface builds its exclusion from the canonical internal-source list, so the assertion uses
    that tuple rather than a hand-typed one: if upstream ever drops ``oneshot`` from it, the opt-in
    would be a no-op and this test would say so. ``subagent_listing_scope`` is the #97202 helper and
    keeps its two-value contract (pinned above), so the re-admission is asserted on the helper that
    now owns both opt-ins, with the real shape as input.
    """
    home = _home(tmp_path, show_oneshot=True)
    shipped = [s for s in INTERNAL_LISTING_SOURCES if s != "acp"]  # desktop keeps acp local-side
    assert "oneshot" in shipped

    offline = session_listing_scope(home, exclude_sources=shipped)
    assert offline[1] is True
    assert "oneshot" not in (offline[2] or [])
    # Everything else the surface hides stays hidden.
    assert set(offline[2] or []) == set(shipped) - {"oneshot"}
