"""Regression test for #104849: skill command cache thrash from platform/home flapping."""
from pathlib import Path
from unittest.mock import patch
from agent.skill_commands import get_skill_commands, scan_skill_commands


def _make_skill(parent: Path, name: str) -> None:
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n---\n\n# {name}\n\nBody.\n"
    )


def test_cache_survives_platform_empty_string_none_flapping(tmp_path):
    """When platform flaps between "" and None, both share the same cache slot.

    Regression for #104849: `_set_session_context` in `tui_gateway/server.py`
    pins the platform to an empty string (`platform=""`), and
    `_resolve_skill_commands_platform()` normalizes that `""` to `None` in
    some request contexts. A single-slot cache treating them as distinct keys
    resulted in a rescan on every `commands.catalog` poll (~3x/5s).

    The fix: multi-slot cache keyed by `(platform, home)`, and normalize `""`
    to `None` so both spellings share one slot (scan once, hit forever).
    """
    import os
    import agent.skill_commands as sc_mod

    _make_skill(tmp_path, "test-skill")

    scan_count = 0
    original_scan = scan_skill_commands

    def counting_scan(*args, **kwargs):
        nonlocal scan_count
        scan_count += 1
        return original_scan(*args, **kwargs)

    with (
        patch("tools.skills_tool.SKILLS_DIR", tmp_path),
        patch.object(sc_mod, "_skill_commands_by_key", {}),
        patch("agent.skill_commands.scan_skill_commands", side_effect=counting_scan),
    ):
        # First call: platform implicitly None (no env var)
        with patch.dict(os.environ, {}, clear=False):
            cmds1 = get_skill_commands()
        assert "/test-skill" in cmds1
        assert scan_count == 1  # First scan

        # Second call: platform explicitly "" (what the TUI gateway sets)
        with patch.dict(os.environ, {"HERMES_PLATFORM": ""}, clear=False):
            cmds2 = get_skill_commands()
        # Both "" and None normalize to None — cache hit, NO rescan
        assert cmds2 is cmds1
        assert scan_count == 1  # Still the same scan

        # Third call: back to implicitly None (drop the env var)
        with patch.dict(os.environ, {}, clear=False):
            cmds3 = get_skill_commands()
        # Cache hit again — no new scan
        assert cmds3 is cmds1
        assert scan_count == 1  # Never rescanned


def test_cache_creates_multiple_slots_for_distinct_platform_or_home(tmp_path):
    """Each distinct (platform, home) identity gets its own cache slot.

    Ensures the multi-slot cache doesn't degrade into a single-slot
    cache that breaks the #14536 / #88023 fixes (platform/profile isolation).
    """
    import os
    import agent.skill_commands as sc_mod
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_a = tmp_path / "profile_a"
    profile_b = tmp_path / "profile_b"
    profile_a.mkdir()
    profile_b.mkdir()
    (profile_a / "config.yaml").write_text("{}\n")
    (profile_b / "config.yaml").write_text("{}\n")
    _make_skill(profile_a / "skills", "a-skill")
    _make_skill(profile_b / "skills", "b-skill")

    scan_count = 0
    original_scan = scan_skill_commands

    def counting_scan(*args, **kwargs):
        nonlocal scan_count
        scan_count += 1
        return original_scan(*args, **kwargs)

    with (
        patch.object(sc_mod, "_skill_commands_by_key", {}),
        patch("agent.skill_commands.scan_skill_commands", side_effect=counting_scan),
    ):
        # Scan profile A, platform 'telegram'
        token = set_hermes_home_override(profile_a)
        try:
            with patch.dict(os.environ, {"HERMES_PLATFORM": "telegram"}):
                cmds_a_telegram = get_skill_commands()
        finally:
            reset_hermes_home_override(token)
        assert "/a-skill" in cmds_a_telegram
        assert scan_count == 1  # First scan

        # Switch to profile B, same platform 'telegram' — different home → rescan
        token = set_hermes_home_override(profile_b)
        try:
            with patch.dict(os.environ, {"HERMES_PLATFORM": "telegram"}):
                cmds_b_telegram = get_skill_commands()
        finally:
            reset_hermes_home_override(token)
        assert "/b-skill" in cmds_b_telegram
        assert scan_count == 2  # New (platform, home) key → second scan

        # Switch to profile A, platform 'discord' — different platform → rescan
        token = set_hermes_home_override(profile_a)
        try:
            with patch.dict(os.environ, {"HERMES_PLATFORM": "discord"}):
                cmds_a_discord = get_skill_commands()
        finally:
            reset_hermes_home_override(token)
        assert "/a-skill" in cmds_a_discord
        assert scan_count == 3  # New platform key → third scan

        # Back to profile A, telegram — cache hit from the first scan
        token = set_hermes_home_override(profile_a)
        try:
            with patch.dict(os.environ, {"HERMES_PLATFORM": "telegram"}):
                cmds_a_telegram_again = get_skill_commands()
        finally:
            reset_hermes_home_override(token)
        # Same (platform='telegram', home=profile_a) as call 1 → cache hit
        assert cmds_a_telegram_again is cmds_a_telegram
        assert scan_count == 3  # No rescan

def test_cache_keys_project_as_third_dimension(tmp_path):
    """Two sessions in different repos share (platform, home) but must NOT share
    a cache slot: the project root is part of the identity (#114359). A 2-tuple
    (platform, home) key would serve repo A's project skills to repo B after a
    cache hit — the regression this guards against.
    """
    import agent.skill_commands as sc_mod
    from agent import skill_utils
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    for repo in (repo_a, repo_b):
        (repo / ".git").mkdir(parents=True)
    _make_skill((repo_a / ".hermes" / "skills"), "proj-a-skill")
    _make_skill((repo_b / ".hermes" / "skills"), "proj-b-skill")
    # A shared, otherwise-empty home: only the project dimension differs.
    home = tmp_path / "shared-home"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n")

    state = {"root": repo_a}

    def fake_find_project_root(start=None):
        return state["root"]

    def fake_project_dirs():
        d = state["root"] / ".hermes" / "skills"
        return [d] if d.is_dir() else []

    scan_count = 0
    original_scan = scan_skill_commands

    def counting_scan(*args, **kwargs):
        nonlocal scan_count
        scan_count += 1
        return original_scan(*args, **kwargs)

    token = set_hermes_home_override(home)
    try:
        with (
            patch.object(sc_mod, "_skill_commands_by_key", {}),
            patch("agent.skill_commands.scan_skill_commands", side_effect=counting_scan),
            patch.object(skill_utils, "find_project_root", fake_find_project_root),
            patch.object(skill_utils, "get_project_skills_dirs", fake_project_dirs),
        ):
            cmds_a = get_skill_commands()
            assert "/proj-a-skill" in cmds_a
            assert "/proj-b-skill" not in cmds_a
            assert scan_count == 1

            # Same platform/home, different project root → its OWN slot, own view.
            state["root"] = repo_b
            cmds_b = get_skill_commands()
            assert "/proj-b-skill" in cmds_b
            assert "/proj-a-skill" not in cmds_b
            assert scan_count == 2

            # Flapping back and forth hits the memoized slots — no rescan.
            state["root"] = repo_a
            assert get_skill_commands() is cmds_a
            state["root"] = repo_b
            assert get_skill_commands() is cmds_b
            assert scan_count == 2
    finally:
        reset_hermes_home_override(token)


def test_reload_invalidates_every_identity_slot(tmp_path):
    """reload_skills() clears the whole multi-slot dict: a skill edit can affect
    any (platform, home, project) identity, so every cached view must rescan."""
    import os
    import agent.skill_commands as sc_mod
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from agent.skill_commands import reload_skills

    profile_a = tmp_path / "profile_a"
    profile_b = tmp_path / "profile_b"
    profile_a.mkdir()
    profile_b.mkdir()
    (profile_a / "config.yaml").write_text("{}\n")
    (profile_b / "config.yaml").write_text("{}\n")
    _make_skill(profile_a / "skills", "a-skill")
    _make_skill(profile_b / "skills", "b-skill")

    with patch.object(sc_mod, "_skill_commands_by_key", {}):
        for profile, expected in ((profile_a, "/a-skill"), (profile_b, "/b-skill")):
            token = set_hermes_home_override(profile)
            try:
                assert expected in get_skill_commands()
            finally:
                reset_hermes_home_override(token)
        assert len(sc_mod._skill_commands_by_key) == 2

        # Add a skill to profile A, then reload from A: BOTH slots must drop.
        _make_skill(profile_a / "skills", "a-second-skill")
        token = set_hermes_home_override(profile_a)
        try:
            result = reload_skills()
            assert "/a-second-skill" in {item["name"] for item in result["added"]} or \
                result["added"], result
        finally:
            reset_hermes_home_override(token)
        # reload_skills() repopulated ONLY the current (A) slot with a fresh view
        # (it saw the skill added after the original scan); B's stale slot is gone.
        slots = sc_mod._skill_commands_by_key
        assert len(slots) == 1
        (a_cmds,) = slots.values()
        assert "/a-skill" in a_cmds and "/a-second-skill" in a_cmds
        # The next lookup from B rescans and still sees its own view.
        token = set_hermes_home_override(profile_b)
        try:
            assert "/b-skill" in get_skill_commands()
        finally:
            reset_hermes_home_override(token)
