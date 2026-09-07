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
