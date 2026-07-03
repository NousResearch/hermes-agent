"""Tests for skill lifecycle plugin hooks.

Verifies that pre_skill_create and post_skill_create hooks fire during
skill_manage(action='create') with the documented kwargs, that plugins
can redirect/block/handle creation, and that a misbehaving hook never
breaks creation.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager
from tools.skill_manager_tool import (
    _create_skill,
    _edit_skill,
)

SKILL_CONTENT = """\
---
name: test-skill
description: A test skill for unit testing.
---

# Test Skill

Step 1: Do the thing.
"""

SKILL_CONTENT_2 = """\
---
name: test-skill
description: Updated description.
---

# Test Skill v2

Step 1: Do the new thing.
"""


@contextmanager
def _isolated_skills(tmp_path):
    """Patch SKILLS_DIR and get_all_skills_dirs so creation uses tmp_path."""
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    with patch("tools.skill_manager_tool.SKILLS_DIR", skills_dir), \
         patch("agent.skill_utils.get_all_skills_dirs", return_value=[skills_dir]):
        yield skills_dir


@pytest.fixture
def captured_hooks():
    """Register capturing callbacks for the two skill lifecycle hooks.

    Patches the plugin manager's _hooks dict directly (the same registry
    invoke_hook reads) and restores it afterward.
    """
    mgr = get_plugin_manager()
    events: list[tuple[str, dict]] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    for hook in ("pre_skill_create", "post_skill_create"):
        mgr._hooks.setdefault(hook, []).append(
            lambda _h=hook, **kw: events.append((_h, kw))
        )
    try:
        yield events
    finally:
        mgr._hooks = saved


# ── VALID_HOOKS registration ──


def test_hooks_are_registered_as_valid():
    """The two skill lifecycle hook names are part of VALID_HOOKS."""
    assert "pre_skill_create" in VALID_HOOKS
    assert "post_skill_create" in VALID_HOOKS


# ── Default behavior (no hooks) ──


def test_default_behavior_writes_to_skills_dir(tmp_path):
    """With no plugin registered, skill creation writes to SKILLS_DIR."""
    with _isolated_skills(tmp_path) as skills_dir:
        result = _create_skill("my-skill", SKILL_CONTENT)
    assert result["success"] is True
    assert (skills_dir / "my-skill" / "SKILL.md").exists()
    assert "hook_handled" not in result


def test_default_behavior_with_category(tmp_path):
    """With no plugin, category nesting works as before."""
    with _isolated_skills(tmp_path) as skills_dir:
        result = _create_skill("my-skill", SKILL_CONTENT, category="devops")
    assert result["success"] is True
    assert (skills_dir / "devops" / "my-skill" / "SKILL.md").exists()


# ── pre_skill_create — redirect ──


def test_redirect_writes_to_custom_dir(tmp_path, captured_hooks):
    """A plugin returning {'action': 'redirect', 'path': ...} writes there."""
    redirect_path = tmp_path / "custom-skills"

    def _redirect(**kw):
        return {"action": "redirect", "path": str(redirect_path / "my-skill")}

    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_skill_create", []).append(_redirect)
    try:
        with _isolated_skills(tmp_path):
            result = _create_skill("my-skill", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    assert result["success"] is True
    # Skill was written to the redirected path, not the default skills dir
    assert (redirect_path / "my-skill" / "SKILL.md").exists()


def test_redirect_still_fires_post_hook(tmp_path):
    """A redirect should still trigger post_skill_create with the right path."""
    redirect_path = tmp_path / "custom-skills"
    post_events: list[dict] = []

    def _redirect(**kw):
        return {"action": "redirect", "path": str(redirect_path / "my-skill")}

    def _on_post(**kw):
        post_events.append(kw)

    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_skill_create", []).append(_redirect)
    mgr._hooks.setdefault("post_skill_create", []).append(_on_post)
    try:
        with _isolated_skills(tmp_path):
            result = _create_skill("my-skill", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    assert result["success"] is True
    assert len(post_events) == 1
    assert post_events[0]["name"] == "my-skill"
    assert str(redirect_path / "my-skill") in post_events[0]["path"]
    assert post_events[0]["success"] is True


# ── pre_skill_create — handled ──


def test_handled_skips_hermes_write(tmp_path, captured_hooks):
    """A plugin returning {'action': 'handled'} skips the default write."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_skill_create", []).append(
        lambda **kw: {"action": "handled"}
    )
    try:
        with _isolated_skills(tmp_path) as skills_dir:
            result = _create_skill("my-skill", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    assert result["success"] is True
    assert result.get("hook_handled") is True
    # Nothing was written to the default skills dir
    assert not (skills_dir / "my-skill" / "SKILL.md").exists()


def test_handled_fires_post_hook(tmp_path):
    """A handled action should still fire post_skill_create."""
    post_events: list[dict] = []

    def _on_post(**kw):
        post_events.append(kw)

    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_skill_create", []).append(
        lambda **kw: {"action": "handled"}
    )
    mgr._hooks.setdefault("post_skill_create", []).append(_on_post)
    try:
        with _isolated_skills(tmp_path):
            result = _create_skill("my-skill", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    assert result["success"] is True
    assert len(post_events) == 1
    assert post_events[0]["name"] == "my-skill"
    assert post_events[0]["success"] is True


# ── pre_skill_create — block ──


def test_block_aborts_creation(tmp_path):
    """A plugin returning {'action': 'block', 'reason': ...} aborts creation."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_skill_create", []).append(
        lambda **kw: {"action": "block", "reason": "not allowed"}
    )
    try:
        with _isolated_skills(tmp_path) as skills_dir:
            result = _create_skill("my-skill", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    assert result["success"] is False
    assert "not allowed" in result["error"]
    assert not (skills_dir / "my-skill" / "SKILL.md").exists()


def test_block_without_reason_uses_default_message(tmp_path):
    """A block without a reason uses a default error message."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_skill_create", []).append(
        lambda **kw: {"action": "block"}
    )
    try:
        with _isolated_skills(tmp_path):
            result = _create_skill("my-skill", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    assert result["success"] is False
    assert "blocked" in result["error"].lower()


# ── Resilience ──


def test_misbehaving_hook_does_not_break_creation(tmp_path):
    """A hook callback that raises must not break creation."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_skill_create", []).append(
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    try:
        with _isolated_skills(tmp_path) as skills_dir:
            result = _create_skill("my-skill", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    # Despite the raising hook, creation succeeds with default behavior
    assert result["success"] is True
    assert (skills_dir / "my-skill" / "SKILL.md").exists()


def test_hook_returns_none_falls_through(tmp_path):
    """A hook returning None passes through to default behavior."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_skill_create", []).append(
        lambda **kw: None
    )
    try:
        with _isolated_skills(tmp_path) as skills_dir:
            result = _create_skill("my-skill", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    assert result["success"] is True
    assert (skills_dir / "my-skill" / "SKILL.md").exists()


def test_first_hook_wins_multiple_hooks(tmp_path):
    """When multiple hooks registered, the first non-None return wins."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _first(**kw):
        return {"action": "block", "reason": "first hook blocked"}

    def _second(**kw):
        return {"action": "redirect", "path": "/nowhere"}

    mgr._hooks.setdefault("pre_skill_create", []).append(_first)
    mgr._hooks.setdefault("pre_skill_create", []).append(_second)
    try:
        with _isolated_skills(tmp_path) as skills_dir:
            result = _create_skill("my-skill", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    # First hook blocks, second is never reached
    assert result["success"] is False
    assert "first hook blocked" in result["error"]
    assert not (skills_dir / "my-skill" / "SKILL.md").exists()


# ── pre_skill_edit ──


def test_edit_hook_is_registered_as_valid():
    """The edit hook name is part of VALID_HOOKS."""
    assert "pre_skill_edit" in VALID_HOOKS


def test_edit_handled_returns_success(tmp_path):
    """A plugin handling edit returns success without touching Hermes path."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_skill_edit", []).append(
        lambda **kw: {"action": "handled"}
    )
    try:
        with _isolated_skills(tmp_path):
            result = _edit_skill("any-skill", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    assert result["success"] is True
    assert result.get("hook_handled") is True


def test_edit_block_aborts(tmp_path):
    """A plugin blocking edit returns failure."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_skill_edit", []).append(
        lambda **kw: {"action": "block", "reason": "no edits allowed"}
    )
    try:
        with _isolated_skills(tmp_path):
            result = _edit_skill("any-skill", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    assert result["success"] is False
    assert "no edits allowed" in result["error"]


def test_edit_hook_none_falls_through_to_find_skill(tmp_path):
    """A hook returning None falls through to _find_skill (skill not found)."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_skill_edit", []).append(lambda **kw: None)
    try:
        with _isolated_skills(tmp_path):
            result = _edit_skill("nonexistent", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    # Falls through to _find_skill which finds nothing
    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_edit_misbehaving_hook_falls_through(tmp_path):
    """A raising hook in edit falls through to default _find_skill."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_skill_edit", []).append(
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    try:
        with _isolated_skills(tmp_path):
            result = _edit_skill("nonexistent", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    # Falls through despite the raising hook
    assert result["success"] is False
    assert "not found" in result["error"].lower()


# ── post_skill_edit ──


def test_post_edit_hook_is_registered_as_valid():
    """The post-edit hook name is part of VALID_HOOKS."""
    assert "post_skill_edit" in VALID_HOOKS


def test_post_edit_fires_on_successful_default_edit(tmp_path):
    """post_skill_edit fires after a successful default edit (no pre hook)."""
    events: list[dict] = []

    def _on_post(**kw):
        events.append(kw)

    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("post_skill_edit", []).append(_on_post)
    try:
        with _isolated_skills(tmp_path) as skills_dir:
            # First create a skill, then edit it
            c = _create_skill("edit-me", SKILL_CONTENT)
            assert c["success"]
            e = _edit_skill("edit-me", SKILL_CONTENT_2)
            assert e["success"]
    finally:
        mgr._hooks = saved

    assert len(events) == 1
    assert events[0]["name"] == "edit-me"
    assert events[0]["success"] is True
    assert "edit-me" in str(events[0]["path"])


def test_post_edit_does_not_fire_on_handled(tmp_path):
    """post_skill_edit does NOT fire when pre_skill_edit handled the edit."""
    events: list[dict] = []

    def _on_post(**kw):
        events.append(kw)

    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_skill_edit", []).append(
        lambda **kw: {"action": "handled"}
    )
    mgr._hooks.setdefault("post_skill_edit", []).append(_on_post)
    try:
        with _isolated_skills(tmp_path):
            result = _edit_skill("any-skill", SKILL_CONTENT)
    finally:
        mgr._hooks = saved

    assert result["success"] is True
    assert result.get("hook_handled") is True
    # No post hook should have been called since the edit was handled by plugin
    assert len(events) == 0
