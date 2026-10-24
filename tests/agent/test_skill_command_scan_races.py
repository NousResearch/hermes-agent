"""Disk edits and older catalog writers must not pin stale slash commands."""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Event

import pytest

from agent import skill_commands, skill_utils
from tools import skills_tool


def _write_skill(path, name):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: Test skill\n---\nBody\n")


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", tmp_path)
    monkeypatch.setattr(skill_utils, "get_project_skills_dirs", list)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", list)
    monkeypatch.setattr(skill_commands, "_skill_commands_by_key", {})
    monkeypatch.setattr(skill_commands, "_skill_command_disk_tags_by_key", {})
    monkeypatch.setattr(skills_tool, "_SKILLS_CACHE", {})
    return tmp_path


def test_older_command_scan_publishing_last_recovers_after_edit(skills_dir, monkeypatch):
    skill_md = skills_dir / "first" / "SKILL.md"
    _write_skill(skill_md, "before-edit")
    _write_skill(skills_dir / "second" / "SKILL.md", "second-skill")
    parsed, release = Event(), Event()
    original = skill_commands._scan_skill_md

    def pause_after_parsing(*args):
        original(*args)
        if args[0] == skill_md and not parsed.is_set():
            parsed.set()
            assert release.wait(10), "older command scan was not released"

    monkeypatch.setattr(skill_commands, "_scan_skill_md", pause_after_parsing)
    with ThreadPoolExecutor(max_workers=1) as pool:
        older = pool.submit(copy_context().run, skill_commands.get_skill_commands)
        try:
            assert parsed.wait(10), "older command scan did not parse the skill"
            _write_skill(skill_md, "after-edit")
            assert "/after-edit" in skill_commands.get_skill_commands()
        finally:
            release.set()
        assert "/before-edit" in older.result(timeout=10)

    for _ in range(3):
        commands = skill_commands.get_skill_commands()
        assert {"/after-edit", "/second-skill"} <= commands.keys()
        assert "/before-edit" not in commands


def test_older_catalog_ambiguity_cannot_poison_command_scan(skills_dir, monkeypatch):
    first = skills_dir / "first" / "SKILL.md"
    second = skills_dir / "second" / "SKILL.md"
    _write_skill(first, "shared-name")
    _write_skill(second, "shared-name")
    resolved, release = Event(), Event()
    original_resolve = skill_utils.resolve_skill_catalog
    original_catalog = skills_tool._skill_catalog

    def pause_after_resolution(entries):
        result = original_resolve(entries)
        if not resolved.is_set():
            resolved.set()
            assert release.wait(10), "older catalog scan was not released"
        return result

    monkeypatch.setattr(skill_utils, "resolve_skill_catalog", pause_after_resolution)
    with ThreadPoolExecutor(max_workers=1) as pool:
        older = pool.submit(copy_context().run, original_catalog)
        try:
            assert resolved.wait(10), "older catalog scan did not resolve duplicate names"
            # In-place edits do not change the catalog's directory signature.
            _write_skill(second, "renamed-skill")

            def catalog_after_older_publication(**kwargs):
                skills_tool.clear_skills_cache()
                release.set()
                assert all(row["status"] != "unique" for row in older.result(timeout=10))
                return original_catalog(**kwargs)

            monkeypatch.setattr(skills_tool, "_skill_catalog", catalog_after_older_publication)
            commands = skill_commands.get_skill_commands()
            assert {"/shared-name", "/renamed-skill"} <= commands.keys()
        finally:
            release.set()
        older.result(timeout=10)

    for _ in range(3):
        assert {"/shared-name", "/renamed-skill"} <= skill_commands.get_skill_commands().keys()
