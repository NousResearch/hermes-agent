"""Tests for Discord /skill 32-char clamp collision warnings.

Discord's per-command name limit is 32 chars, so
``discord_skill_commands_by_category`` clamps skill slugs to that width
before deduping. When two skills share the same 32-char prefix, only
the first (alphabetical) wins; the second is dropped. Previously the
drop was silent — the ``hidden`` count incremented but nothing named
which skills collided, so authors had no way to discover the drop
short of noticing that their skill was missing from the autocomplete.

This module pins the upgraded behavior: a WARNING log with both full
cmd_keys + the clamped name, so whoever named the skills sees the
collision and can rename one.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch


def test_clamp_collision_emits_warning_naming_both_skills(
    tmp_path: Path, caplog
) -> None:
    """Two skills with identical first 32 chars — warning names both."""
    from hermes_cli.commands import discord_skill_commands_by_category

    # Craft cmd_keys that share the first 32 chars.
    # 40-char prefix 'skill-collision-prefix-identical-first-32'
    #   -> clamped to 'skill-collision-prefix-identical'
    prefix = "skill-collision-prefix-identical"  # exactly 32 chars
    name_a = prefix + "-alpha"  # /skill-collision-prefix-identical-alpha
    name_b = prefix + "-bravo"  # /skill-collision-prefix-identical-bravo
    assert name_a[:32] == name_b[:32] == prefix

    skills_dir = tmp_path / "skills"
    for nm in (name_a, name_b):
        d = skills_dir / "creative" / nm
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: x\n---\n")

    fake_cmds = {
        f"/{name_a}": {
            "name": name_a,
            "description": "Alpha",
            "skill_md_path": str(skills_dir / "creative" / name_a / "SKILL.md"),
        },
        f"/{name_b}": {
            "name": name_b,
            "description": "Bravo",
            "skill_md_path": str(skills_dir / "creative" / name_b / "SKILL.md"),
        },
    }

    with caplog.at_level(logging.WARNING, logger="hermes_cli.commands"), (
        patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds)
    ), patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        categories, uncategorized, hidden = discord_skill_commands_by_category(
            reserved_names=set(),
        )

    # One skill made it through, one was dropped (hidden counted).
    assert hidden == 1
    kept_names = [n for n, _d, _k in categories.get("creative", [])]
    assert len(kept_names) == 1
    # Alphabetical iteration means the -alpha variant wins the slot.
    assert kept_names[0] == prefix  # clamped

    # Exactly one warning, naming BOTH full cmd_keys and the clamped name.
    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "clamp" in r.getMessage()
    ]
    assert len(warnings) == 1, (
        f"expected exactly one clamp-collision warning, got {len(warnings)}: "
        f"{[r.getMessage() for r in warnings]}"
    )
    msg = warnings[0].getMessage()
    assert f"/{name_a}" in msg, f"winner not named in warning: {msg!r}"
    assert f"/{name_b}" in msg, f"loser not named in warning: {msg!r}"
    assert prefix in msg, f"clamped name not in warning: {msg!r}"


def test_clamp_collision_with_reserved_name_emits_distinct_warning(
    tmp_path: Path, caplog
) -> None:
    """A skill clashing with a reserved gateway command gets its own phrasing.

    The reserved-vs-skill case is operationally different — the fix is
    still "rename the skill," but there's no second skill to also
    rename. The warning should say so explicitly.
    """
    from hermes_cli.commands import discord_skill_commands_by_category

    # Reserved name 'help' is 4 chars — make a skill whose slug
    # clamps to 'help' (so, exactly 'help').
    reserved = "help"
    skills_dir = tmp_path / "skills"
    d = skills_dir / "creative" / reserved
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: x\n---\n")

    fake_cmds = {
        f"/{reserved}": {
            "name": reserved,
            "description": "desc",
            "skill_md_path": str(d / "SKILL.md"),
        },
    }

    with caplog.at_level(logging.WARNING, logger="hermes_cli.commands"), (
        patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds)
    ), patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        categories, uncategorized, hidden = discord_skill_commands_by_category(
            reserved_names={"help"},
        )

    # Skill dropped in favor of the reserved command.
    assert hidden == 1
    assert categories == {}
    assert uncategorized == []

    warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "reserved" in r.getMessage()
    ]
    assert len(warnings) == 1, (
        f"expected one reserved-name collision warning, got "
        f"{[r.getMessage() for r in warnings]}"
    )
    msg = warnings[0].getMessage()
    assert f"/{reserved}" in msg
    assert "reserved" in msg.lower()


def test_no_collision_no_warning(tmp_path: Path, caplog) -> None:
    """Sanity: two distinct-prefix skills produce zero warnings."""
    from hermes_cli.commands import discord_skill_commands_by_category

    skills_dir = tmp_path / "skills"
    for nm in ("alpha", "bravo"):
        d = skills_dir / "creative" / nm
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: x\n---\n")

    fake_cmds = {
        "/alpha": {
            "name": "alpha", "description": "",
            "skill_md_path": str(skills_dir / "creative" / "alpha" / "SKILL.md"),
        },
        "/bravo": {
            "name": "bravo", "description": "",
            "skill_md_path": str(skills_dir / "creative" / "bravo" / "SKILL.md"),
        },
    }

    with caplog.at_level(logging.WARNING, logger="hermes_cli.commands"), (
        patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds)
    ), patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        categories, uncategorized, hidden = discord_skill_commands_by_category(
            reserved_names=set(),
        )

    assert hidden == 0
    assert {n for n, _d, _k in categories["creative"]} == {"alpha", "bravo"}
    clamp_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and ("clamp" in r.getMessage() or "reserved" in r.getMessage())
    ]
    assert clamp_warnings == []


def test_long_skill_name_preserves_cmd_key_through_by_category(
    tmp_path: Path,
) -> None:
    """Skills with names > 32 chars must keep their original cmd_key.

    ``discord_skill_commands_by_category`` clamps the display name to 32
    chars but the third tuple element (cmd_key) must stay as the original
    ``/full-skill-name`` so that ``_skill_handler`` dispatches via
    ``_run_simple_slash`` with the full command, not the truncated one.

    This is the actual runtime path used by the Discord adapter via
    ``_refresh_skill_catalog_state``.
    """
    from hermes_cli.commands import discord_skill_commands_by_category

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    resolved = str(skills_dir.resolve())

    long_name = "generate-ascii-art-from-text-description-detailed"
    cmd_key = f"/{long_name}"
    fake_cmds = {
        cmd_key: {
            "name": long_name,
            "description": "Generate ASCII art from a text description",
            "skill_md_path": f"{resolved}/creative/{long_name}/SKILL.md",
            "skill_dir": f"{resolved}/creative/{long_name}",
        },
        "/short-skill": {
            "name": "short-skill",
            "description": "A short skill",
            "skill_md_path": f"{resolved}/creative/short-skill/SKILL.md",
            "skill_dir": f"{resolved}/creative/short-skill",
        },
    }

    with patch("agent.skill_commands.get_skill_commands", return_value=fake_cmds), \
         patch("tools.skills_tool.SKILLS_DIR", skills_dir):
        categories, uncategorized, hidden = discord_skill_commands_by_category(
            reserved_names=set(),
        )

    # Flatten (same as _refresh_skill_catalog_state does)
    entries = list(uncategorized)
    for cat_skills in categories.values():
        entries.extend(cat_skills)

    # Build lookup (same as _refresh_skill_catalog_state does)
    skill_lookup = {n: (d, k) for n, d, k in entries}

    # Find the long skill
    long_entry = [e for e in entries if e[2] == cmd_key]
    assert len(long_entry) == 1, f"Long skill should appear once, got: {long_entry}"

    display_name, desc, key = long_entry[0]
    assert len(display_name) <= 32, (
        f"Display name should be clamped to 32 chars, got {len(display_name)}"
    )
    assert key == cmd_key, (
        f"cmd_key must be the original /{long_name}, got {key!r}"
    )

    # Verify lookup works: clamped display name -> original cmd_key
    assert display_name in skill_lookup
    _desc, looked_up_key = skill_lookup[display_name]
    assert looked_up_key == cmd_key, (
        f"Lookup must map clamped name to original cmd_key, got {looked_up_key!r}"
    )

    # Short skill should also be present and correct
    short_entry = [e for e in entries if e[2] == "/short-skill"]
    assert len(short_entry) == 1
    assert short_entry[0][0] == "short-skill"
