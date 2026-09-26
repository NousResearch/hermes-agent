"""Contract tests for the 18 skill mutation lifecycle plugin hooks."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from hermes_cli.plugins import (
    SHELL_UNSUPPORTED_HOOKS,
    SKILL_MUTATION_ACTIONS,
    SKILL_MUTATION_GUARD_HOOKS,
    SKILL_MUTATION_HOOKS,
    SKILL_MUTATION_POST_HOOKS,
    SKILL_MUTATION_PRE_HOOKS,
    VALID_HOOKS,
    get_plugin_manager,
)
from tools.skill_manager_tool import (
    _create_skill,
    _find_skill,
    _record_success as record_success_impl,
    skill_manage,
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

ACTIONS = tuple(SKILL_MUTATION_ACTIONS)

GUARD_PAYLOAD_KEYS = {
    "create": {"name", "content", "category"},
    "edit": {"name", "content"},
    "patch": {"name", "old_string", "new_string", "file_path", "replace_all"},
    "write_file": {"name", "file_path", "file_content"},
    "remove_file": {"name", "file_path"},
    "delete": {"name", "absorbed_into"},
}
PRE_PAYLOAD_KEYS = {**GUARD_PAYLOAD_KEYS, "edit": GUARD_PAYLOAD_KEYS["edit"] | {"old_content"}}
POST_PAYLOAD_KEYS = {
    "create": {"name", "category", "path", "success", "error"},
    "edit": {"name", "path", "success", "error"},
    "patch": {"name", "file_path", "replace_all", "success", "error"},
    "write_file": {"name", "file_path", "success", "error"},
    "remove_file": {"name", "file_path", "success", "error"},
    "delete": {"name", "absorbed_into", "success", "error"},
}


@pytest.fixture(autouse=True)
def isolated_plugin_hooks(monkeypatch):
    """Use the real plugin dispatcher without discovering user plugins."""
    manager = get_plugin_manager()
    monkeypatch.setattr(manager, "_hooks", {})
    monkeypatch.setattr(manager, "_discovered", True)
    yield manager


@contextmanager
def isolated_skills(tmp_path: Path, *, extra_roots=()):
    root = tmp_path / "skills"
    root.mkdir(parents=True, exist_ok=True)
    roots = [root, *extra_roots]
    with (
        patch("tools.skill_manager_tool.SKILLS_DIR", root),
        patch("agent.skill_utils.get_all_skills_dirs", return_value=roots),
        patch("tools.skill_manager_tool._apply_skill_write_gate", return_value=None),
        patch("tools.skill_manager_tool._run_write_gate", return_value=None),
        patch("tools.skill_manager_tool._record_success") as record_success,
    ):
        yield root, record_success


def register_hook(name, callback):
    get_plugin_manager()._hooks.setdefault(name, []).append(callback)


def seed_operation(root: Path, action: str) -> None:
    if action == "create":
        return
    created = _create_skill("test-skill", SKILL_CONTENT)
    assert created["success"], created
    if action == "remove_file":
        target = root / "test-skill" / "references" / "note.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("note", encoding="utf-8")


def invoke_operation(action: str):
    kwargs: dict[str, Any] = {
        "create": {"content": SKILL_CONTENT},
        "edit": {"content": SKILL_CONTENT_2},
        "patch": {"old_string": "Do the thing.", "new_string": "Do it safely."},
        "write_file": {"file_path": "references/note.md", "file_content": "note"},
        "remove_file": {"file_path": "references/note.md"},
        "delete": {"absorbed_into": ""},
    }[action]
    return json.loads(skill_manage(action=action, name="test-skill", **kwargs))


def public_payload(payload):
    return {key: value for key, value in payload.items() if key != "telemetry_schema_version"}


def test_exact_hook_catalog_and_shell_support_boundary():
    expected = {
        *(f"pre_skill_{action}:guard" for action in ACTIONS),
        *(f"pre_skill_{action}" for action in ACTIONS),
        *(f"post_skill_{action}" for action in ACTIONS),
    }
    assert len(expected) == 18
    assert SKILL_MUTATION_HOOKS == expected
    assert expected <= VALID_HOOKS
    assert SKILL_MUTATION_GUARD_HOOKS | SKILL_MUTATION_PRE_HOOKS <= SHELL_UNSUPPORTED_HOOKS
    assert SKILL_MUTATION_POST_HOOKS.isdisjoint(SHELL_UNSUPPORTED_HOOKS)


def test_shell_hook_cli_refuses_python_only_skill_directive(capsys):
    from hermes_cli.hooks import _cmd_test

    _cmd_test(SimpleNamespace(event="pre_skill_delete", for_tool=None, payload_file=None))

    assert "Python plugins only" in capsys.readouterr().out


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("phase", ("guard", "pre"))
def test_pre_hook_payload_and_handled_contract(tmp_path, action, phase):
    seen = []
    hook = f"pre_skill_{action}:guard" if phase == "guard" else f"pre_skill_{action}"

    with isolated_skills(tmp_path) as (root, record_success):
        seed_operation(root, action)
        plugin_path = tmp_path / "plugin-owned" / "test-skill"
        plugin_path.mkdir(parents=True)
        register_hook(
            hook,
            lambda **kwargs: seen.append(public_payload(kwargs))
            or {"action": "handled", **({"path": str(plugin_path)} if action == "create" else {})},
        )
        result = invoke_operation(action)

    assert result["success"] is True
    assert result["hook_handled"] is True
    assert len(seen) == 1
    assert set(seen[0]) == (GUARD_PAYLOAD_KEYS if phase == "guard" else PRE_PAYLOAD_KEYS)[action]
    if action == "edit" and phase == "pre":
        assert seen[0]["old_content"] == SKILL_CONTENT
    record_success.assert_called_once()


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("phase", ("guard", "pre"))
def test_block_emits_one_failed_post_event(tmp_path, action, phase):
    posts = []
    hook = f"pre_skill_{action}:guard" if phase == "guard" else f"pre_skill_{action}"

    with isolated_skills(tmp_path) as (root, record_success):
        seed_operation(root, action)
        register_hook(hook, lambda **_: {"action": "block", "reason": "policy denied"})
        register_hook(f"post_skill_{action}", lambda **kwargs: posts.append(public_payload(kwargs)))
        result = invoke_operation(action)

    assert result["success"] is False
    assert "policy denied" in result["error"]
    assert len(posts) == 1
    assert set(posts[0]) == POST_PAYLOAD_KEYS[action]
    assert posts[0]["success"] is False
    assert "policy denied" in posts[0]["error"]
    record_success.assert_not_called()


@pytest.mark.parametrize("action", ACTIONS)
def test_success_emits_one_post_after_success_bookkeeping(tmp_path, action):
    order = []
    posts = []

    with isolated_skills(tmp_path) as (root, record_success):
        seed_operation(root, action)
        record_success.side_effect = lambda *args, **kwargs: order.append("record_success")
        register_hook(
            f"post_skill_{action}",
            lambda **kwargs: (order.append("post"), posts.append(public_payload(kwargs))),
        )
        result = invoke_operation(action)

    assert result["success"] is True, result
    assert order == ["record_success", "post"]
    assert len(posts) == 1
    assert set(posts[0]) == POST_PAYLOAD_KEYS[action]
    assert posts[0]["success"] is True
    assert posts[0]["error"] is None
    if action == "create":
        assert Path(posts[0]["path"]).is_absolute()


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("phase", ("guard", "pre"))
def test_invalid_and_raising_callbacks_fall_through(tmp_path, action, phase):
    hook = f"pre_skill_{action}:guard" if phase == "guard" else f"pre_skill_{action}"

    def raises(**_):
        raise RuntimeError("plugin boom")

    with isolated_skills(tmp_path) as (root, _record_success):
        seed_operation(root, action)
        register_hook(hook, raises)
        register_hook(hook, lambda **_: ["not", "a", "directive"])
        result = invoke_operation(action)

    assert result["success"] is True, result
    assert "hook_handled" not in result


def test_normal_pre_hook_does_not_bypass_existence_guard_but_guard_hook_can(tmp_path):
    pre_seen = []
    with isolated_skills(tmp_path):
        register_hook("pre_skill_edit", lambda **kwargs: pre_seen.append(kwargs) or {"action": "handled"})
        missing = invoke_operation("edit")
    assert missing["success"] is False
    assert pre_seen == []

    with isolated_skills(tmp_path):
        register_hook("pre_skill_edit:guard", lambda **_: {"action": "handled"})
        handled = invoke_operation("edit")
    assert handled["success"] is True
    assert handled["hook_handled"] is True


def test_create_redirect_requires_discoverable_absolute_target(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    target = external / "test-skill"
    posts = []

    with isolated_skills(tmp_path, extra_roots=(external,)):
        register_hook("pre_skill_create", lambda **_: {"action": "redirect", "path": str(target)})
        register_hook("post_skill_create", lambda **kwargs: posts.append(public_payload(kwargs)))
        result = invoke_operation("create")
        found = _find_skill("test-skill")

    assert result["success"] is True, result
    assert (target / "SKILL.md").is_file()
    assert found == {"path": target}
    assert posts[0]["path"] == str(target.resolve())


@pytest.mark.parametrize("directive", [
    {"action": "redirect"},
    {"action": "redirect", "path": "relative/test-skill"},
])
def test_invalid_create_redirect_is_a_tool_error(tmp_path, directive):
    posts = []
    with isolated_skills(tmp_path) as (root, record_success):
        register_hook("pre_skill_create", lambda **_: directive)
        register_hook("post_skill_create", lambda **kwargs: posts.append(public_payload(kwargs)))
        result = invoke_operation("create")

    assert result["success"] is False
    assert not (root / "test-skill").exists()
    assert len(posts) == 1 and posts[0]["success"] is False
    record_success.assert_not_called()


@pytest.mark.parametrize("directive", [
    {"action": "handled"},
    {"action": "handled", "path": "relative/test-skill"},
])
def test_handled_create_requires_an_absolute_reported_path(tmp_path, directive):
    posts = []
    with isolated_skills(tmp_path) as (root, record_success):
        register_hook("pre_skill_create", lambda **_: directive)
        register_hook("post_skill_create", lambda **kwargs: posts.append(public_payload(kwargs)))
        result = invoke_operation("create")

    assert result["success"] is False
    assert "handled" in result["error"] and "path" in result["error"]
    assert not (root / "test-skill").exists()
    assert len(posts) == 1 and posts[0]["success"] is False
    record_success.assert_not_called()


@pytest.mark.parametrize("action", ACTIONS)
def test_core_error_emits_one_failed_post_event(tmp_path, action):
    posts = []
    with isolated_skills(tmp_path) as (root, record_success):
        if action == "create":
            seed_operation(root, "edit")  # duplicate create
        elif action not in {"edit", "delete"}:
            seed_operation(root, action)
        register_hook(f"post_skill_{action}", lambda **kwargs: posts.append(public_payload(kwargs)))
        if action == "edit":
            result = invoke_operation("edit")  # missing skill
        elif action == "patch":
            result = json.loads(skill_manage(
                action="patch", name="test-skill", old_string="missing", new_string="replacement"))
        elif action == "write_file":
            result = json.loads(skill_manage(
                action="write_file", name="test-skill", file_path="outside.txt", file_content="x"))
        elif action == "remove_file":
            result = json.loads(skill_manage(
                action="remove_file", name="test-skill", file_path="references/missing.md"))
        else:
            result = invoke_operation(action)

    assert result["success"] is False
    assert len(posts) == 1
    assert posts[0]["success"] is False
    assert posts[0]["error"]
    record_success.assert_not_called()


@pytest.mark.parametrize("action", ACTIONS)
def test_raising_post_callback_never_breaks_mutation(tmp_path, action):
    def raises(**_):
        raise RuntimeError("observer boom")

    with isolated_skills(tmp_path) as (root, _record_success):
        seed_operation(root, action)
        register_hook(f"post_skill_{action}", raises)
        result = invoke_operation(action)

    assert result["success"] is True, result


@pytest.mark.parametrize("action", ACTIONS)
def test_traversal_name_never_reaches_a_skill_hook(tmp_path, action):
    seen = []
    with isolated_skills(tmp_path):
        register_hook(f"pre_skill_{action}:guard", lambda **kwargs: seen.append(kwargs))
        kwargs: dict[str, Any] = {
            "create": {"content": SKILL_CONTENT},
            "edit": {"content": SKILL_CONTENT_2},
            "patch": {"old_string": "x", "new_string": "y"},
            "write_file": {"file_path": "references/x.md", "file_content": "x"},
            "remove_file": {"file_path": "references/x.md"},
            "delete": {"absorbed_into": ""},
        }[action]
        result = json.loads(skill_manage(action=action, name="../../test-skill", **kwargs))

    assert result["success"] is False
    assert seen == []


def test_handled_success_preserves_prompt_cache_invalidation(tmp_path):
    plugin_path = tmp_path / "plugin-owned" / "test-skill"
    plugin_path.mkdir(parents=True)
    with (
        isolated_skills(tmp_path) as (_root, record_success),
        patch("agent.prompt_builder.clear_skills_system_prompt_cache") as clear_cache,
        patch("tools.skill_manager_tool._find_skill", return_value=None),
        patch("tools.skill_manager_tool._maybe_debounced_sync_push"),
        patch("tools.skill_ledger.record_mutation"),
        patch("tools.skill_usage.record_created"),
    ):
        record_success.side_effect = record_success_impl
        register_hook(
            "pre_skill_create",
            lambda **_: {"action": "handled", "path": str(plugin_path)},
        )
        result = invoke_operation("create")

    assert result["success"] is True and result["hook_handled"] is True
    clear_cache.assert_called_once_with(clear_snapshot=True)


def test_atomic_batch_flushes_posts_once_after_commit(tmp_path):
    posts = []
    with isolated_skills(tmp_path) as (root, _record_success):
        register_hook("post_skill_create", lambda **kwargs: posts.append(("create", public_payload(kwargs))))
        register_hook("post_skill_write_file", lambda **kwargs: posts.append(("write_file", public_payload(kwargs))))
        result = json.loads(skill_manage(action="", name="", operations=[
            {"name": "test-skill", "action": "create", "content": SKILL_CONTENT},
            {"name": "test-skill", "action": "write_file", "file_path": "references/note.md",
             "file_content": "note"},
        ]))

    assert result["success"] is True, result
    assert [name for name, _ in posts] == ["create", "write_file"]
    assert all(payload["success"] is True for _, payload in posts)
    assert (root / "test-skill" / "references" / "note.md").read_text(encoding="utf-8") == "note"


def test_atomic_batch_rollback_reports_each_attempt_once_as_failed(tmp_path):
    posts = []
    with isolated_skills(tmp_path) as (root, _record_success):
        register_hook("post_skill_create", lambda **kwargs: posts.append(("create", public_payload(kwargs))))
        register_hook("post_skill_patch", lambda **kwargs: posts.append(("patch", public_payload(kwargs))))
        result = json.loads(skill_manage(action="", name="", operations=[
            {"name": "test-skill", "action": "create", "content": SKILL_CONTENT},
            {"name": "test-skill", "action": "patch", "old_string": "missing text",
             "new_string": "replacement"},
        ]))

    assert result["success"] is False
    assert not (root / "test-skill").exists()
    assert [name for name, _ in posts] == ["create", "patch"]
    assert all(payload["success"] is False for _, payload in posts)
    assert all("batch aborted" in payload["error"] for _, payload in posts)


def test_atomic_batch_rejects_plugin_handled_directive(tmp_path):
    posts = []
    with isolated_skills(tmp_path) as (root, _record_success):
        register_hook("pre_skill_create", lambda **_: {
            "action": "handled",
            "path": str(tmp_path / "plugin-owned" / "test-skill"),
        })
        register_hook("post_skill_create", lambda **kwargs: posts.append(public_payload(kwargs)))
        result = json.loads(skill_manage(action="", name="", operations=[
            {"name": "test-skill", "action": "create", "content": SKILL_CONTENT},
            {"name": "test-skill", "action": "write_file", "file_path": "references/note.md",
             "file_content": "note"},
        ]))

    assert result["success"] is False
    assert "cannot be rolled back" in result["error"]
    assert not (root / "test-skill").exists()
    assert len(posts) == 1
    assert posts[0]["success"] is False
    assert "batch aborted" in posts[0]["error"]


def test_full_rewrite_patch_uses_edit_hooks_once(tmp_path):
    events = []
    with isolated_skills(tmp_path) as (root, _record_success):
        seed_operation(root, "edit")
        register_hook("pre_skill_edit", lambda **kwargs: events.append("pre_edit"))
        register_hook("post_skill_edit", lambda **kwargs: events.append("post_edit"))
        register_hook("pre_skill_patch", lambda **kwargs: events.append("pre_patch"))
        register_hook("post_skill_patch", lambda **kwargs: events.append("post_patch"))
        result = json.loads(skill_manage(
            action="patch", name="test-skill", content=SKILL_CONTENT_2))

    assert result["success"] is True, result
    assert events == ["pre_edit", "post_edit"]
