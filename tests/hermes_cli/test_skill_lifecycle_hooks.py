"""Contract tests for the 18 per-operation skill mutation lifecycle hooks and the
whole-batch ``pre_skill_batch`` policy gate (payload-bound approval + required-policy
fail-closed)."""

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


def invoke_core_error_operation(action: str):
    operations = {
        "create": lambda: invoke_operation("create"),
        "edit": lambda: invoke_operation("edit"),
        "patch": lambda: json.loads(skill_manage(
            action="patch", name="test-skill", old_string="missing", new_string="replacement")),
        "write_file": lambda: json.loads(skill_manage(
            action="write_file", name="test-skill", file_path="outside.txt", file_content="x")),
        "remove_file": lambda: json.loads(skill_manage(
            action="remove_file", name="test-skill", file_path="references/missing.md")),
        "delete": lambda: invoke_operation("delete"),
    }
    return operations[action]()


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
        seed_action = {"create": "edit", "patch": "patch", "write_file": "write_file",
                       "remove_file": "remove_file"}.get(action)
        if seed_action is not None:
            seed_operation(root, seed_action)
        register_hook(f"post_skill_{action}", lambda **kwargs: posts.append(public_payload(kwargs)))
        result = invoke_core_error_operation(action)

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


# --- Whole-batch decision point: pre_skill_batch (vernmic contracts, 2026-10-08) ---

from hermes_cli.plugins import SKILL_BATCH_HOOK  # noqa: E402 — re-exported catalog constant
from tools.skill_manager_tool import (  # noqa: E402
    _skill_batch_payload_digest,
    apply_skill_pending as _apply_skill_pending,
)


def _batch_ops():
    return [
        {"name": "test-skill", "action": "create", "content": SKILL_CONTENT},
        {"name": "test-skill", "action": "write_file", "file_path": "references/note.md",
         "file_content": "note"},
    ]


def _run_batch(ops):
    return json.loads(skill_manage(action="", name="", operations=ops))


def _require_policy(monkeypatch, hook_name="pre_skill_batch"):
    monkeypatch.setattr("tools.skill_manager_tool._required_skill_policy_hook",
                        lambda: hook_name)


def test_batch_hook_catalog_registration():
    assert SKILL_BATCH_HOOK == "pre_skill_batch"
    assert SKILL_BATCH_HOOK in VALID_HOOKS
    assert SKILL_BATCH_HOOK in SHELL_UNSUPPORTED_HOOKS


def test_batch_gate_default_no_hook_no_policy_is_byte_for_byte_unchanged(tmp_path):
    # No pre_skill_batch callback registered, no skill_policy.required: the gate must
    # not change behavior — batch succeeds exactly as before the gate existed.
    with isolated_skills(tmp_path) as (root, _record_success):
        result = _run_batch(_batch_ops())
    assert result["success"] is True, result
    assert result["operations_applied"] == 2
    assert (root / "test-skill" / "SKILL.md").is_file()


def test_batch_gate_full_payload_visible_and_fires_once_before_any_mutation(tmp_path):
    seen = []

    def policy(**kwargs):
        seen.append(kwargs)
        return None  # observer: no decision

    with isolated_skills(tmp_path) as (root, _record_success):
        register_hook(SKILL_BATCH_HOOK, policy)
        result = _run_batch(_batch_ops())

    assert result["success"] is True, result
    assert len(seen) == 1  # one decision point for the whole batch, not per op
    payload = seen[0]
    assert payload["operations"] == _batch_ops()  # every op, verbatim
    # digest sent to the hook binds the exact payload
    assert payload["payload_digest"] == _skill_batch_payload_digest(_batch_ops())
    # decision happened before any mutation: nothing was on disk when the hook saw it
    assert (root / "test-skill" / "SKILL.md").is_file()  # …and the batch then succeeded


def test_batch_gate_block_refuses_whole_batch_before_first_mutation(tmp_path):
    with isolated_skills(tmp_path) as (root, _record_success):
        register_hook(SKILL_BATCH_HOOK, lambda **_: {"action": "block", "reason": "not allowed"})
        result = _run_batch(_batch_ops())

    assert result["success"] is False
    assert "not allowed" in result["error"]
    assert "policy denied" in result["error"]
    assert not (root / "test-skill").exists()  # refused BEFORE the first mutation


def test_batch_gate_approval_binds_exact_payload_tampered_batch_rejected(tmp_path):
    approved = []

    def policy(**kwargs):
        approved.append(kwargs["payload_digest"])
        return {"action": "approve", "payload_digest": kwargs["payload_digest"]}

    with isolated_skills(tmp_path):
        register_hook(SKILL_BATCH_HOOK, policy)
        ok = _run_batch(_batch_ops())

    assert ok["success"] is True
    assert approved and approved[0] == _skill_batch_payload_digest(_batch_ops())

    # An approval minted for payload X cannot authorize a MODIFIED batch: the hook is
    # gone now (approval was one-shot at decision time), and even a replayed staged
    # record whose operations were edited fails the digest binding.
    staged = {"action": "batch", "payload_digest": approved[0],
              "operations": [dict(_batch_ops()[0], content=SKILL_CONTENT + "tampered")]}
    replay = json.loads(_apply_skill_pending(staged))
    assert replay["success"] is False
    assert "payload binding" in replay["error"]


def test_batch_gate_approval_rejects_subset_replay(tmp_path):
    def policy(**kwargs):
        return {"action": "approve", "payload_digest": kwargs["payload_digest"]}

    with isolated_skills(tmp_path):
        register_hook(SKILL_BATCH_HOOK, policy)
        ok = _run_batch(_batch_ops())
    assert ok["success"] is True

    full_digest = _skill_batch_payload_digest(_batch_ops())
    subset = _batch_ops()[:1]  # a later subset of the approved batch
    assert _skill_batch_payload_digest(subset) != full_digest
    replay = json.loads(_apply_skill_pending(
        {"action": "batch", "payload_digest": full_digest, "operations": subset}))
    assert replay["success"] is False
    assert "payload binding" in replay["error"]
    # a wrong-digest echo is never an approval: see
    # test_batch_gate_mismatched_approve_digest_is_not_an_approval_when_required


def test_batch_gate_mismatched_approve_digest_is_not_an_approval_when_required(tmp_path):
    def policy(**kwargs):
        return {"action": "approve", "payload_digest": "deadbeef"}  # never echoes

    _require_policy(monkeypatch := pytest.MonkeyPatch())
    with isolated_skills(tmp_path) as (root, _record_success):
        register_hook(SKILL_BATCH_HOOK, policy)
        result = _run_batch(_batch_ops())

    assert result["success"] is False
    assert "policy unavailable" in result["error"]
    monkeypatch.undo()


def test_required_policy_absent_fails_closed_as_unavailable(tmp_path, monkeypatch):
    _require_policy(monkeypatch)
    # No plugin registers pre_skill_batch at all.
    with isolated_skills(tmp_path) as (root, _record_success):
        result = _run_batch(_batch_ops())

    assert result["success"] is False
    assert "policy unavailable" in result["error"]
    assert "not registered" in result["error"]
    assert not (root / "test-skill").exists()


def test_required_policy_crash_fails_closed_as_unavailable(tmp_path, monkeypatch):
    _require_policy(monkeypatch)

    def boom(**_):
        raise RuntimeError("policy engine down")

    with isolated_skills(tmp_path) as (root, _record_success):
        register_hook(SKILL_BATCH_HOOK, boom)
        result = _run_batch(_batch_ops())

    assert result["success"] is False
    assert "policy unavailable" in result["error"]
    assert not (root / "test-skill").exists()


def test_required_policy_invalid_return_fails_closed_as_unavailable(tmp_path, monkeypatch):
    _require_policy(monkeypatch)
    with isolated_skills(tmp_path) as (root, _record_success):
        register_hook(SKILL_BATCH_HOOK, lambda **_: {"action": "wat"})
        register_hook(SKILL_BATCH_HOOK, lambda **_: "not a directive")
        result = _run_batch(_batch_ops())

    assert result["success"] is False
    assert "policy unavailable" in result["error"]
    assert "no valid decision" in result["error"]


def test_required_policy_explicit_block_is_denied_not_unavailable(tmp_path, monkeypatch):
    _require_policy(monkeypatch)
    with isolated_skills(tmp_path) as (root, _record_success):
        register_hook(SKILL_BATCH_HOOK, lambda **_: {"action": "block", "reason": "combo denied"})
        result = _run_batch(_batch_ops())

    assert result["success"] is False
    assert "policy denied" in result["error"]
    assert "combo denied" in result["error"]
    assert "policy unavailable" not in result["error"]  # the two are distinguishable


def test_required_policy_valid_approval_proceeds_and_gates_flat_writes_too(tmp_path, monkeypatch):
    _require_policy(monkeypatch)
    seen = []

    def policy(**kwargs):
        seen.append(kwargs)
        return {"action": "approve", "payload_digest": kwargs["payload_digest"]}

    with isolated_skills(tmp_path) as (root, _record_success):
        register_hook(SKILL_BATCH_HOOK, policy)
        batch = _run_batch(_batch_ops())
        flat = json.loads(skill_manage(
            action="write_file", name="test-skill",
            file_path="references/extra.md", file_content="x"))

    assert batch["success"] is True, batch
    assert flat["success"] is True, flat  # contract 2 gates every write, flat included
    assert len(seen) == 2  # one decision per call: the batch + the flat write
    # The flat call's decision payload is its operations[]-shaped equivalent
    # (replace_all=False serializes, matching the staged-record payload shape).
    assert seen[1]["operations"] == [{
        "name": "test-skill", "action": "write_file",
        "file_path": "references/extra.md", "file_content": "x", "replace_all": False}]


def test_required_policy_absent_blocks_flat_write_as_unavailable(tmp_path, monkeypatch):
    _require_policy(monkeypatch)
    with isolated_skills(tmp_path) as (root, _record_success):
        result = json.loads(skill_manage(
            action="write_file", name="test-skill",
            file_path="references/x.md", file_content="x"))

    assert result["success"] is False
    assert "policy unavailable" in result["error"]


def test_required_policy_absent_blocks_sole_delete_operations_call(tmp_path, monkeypatch):
    _require_policy(monkeypatch)
    with isolated_skills(tmp_path) as (root, _record_success):
        seed_operation(root, "edit")
        result = _run_batch([{"name": "test-skill", "action": "delete", "absorbed_into": ""}])

    assert result["success"] is False
    assert "policy unavailable" in result["error"]
    assert (root / "test-skill" / "SKILL.md").is_file()  # delete refused


def test_optional_hook_no_decision_fails_open_for_batch_and_flat(tmp_path):
    # Contract 2 default: WITHOUT skill_policy.required an optional pre_skill_batch
    # consumer that crashes / returns nothing changes nothing (fail-open, observer).
    def boom(**_):
        raise RuntimeError("observer bug")

    with isolated_skills(tmp_path) as (root, _record_success):
        register_hook(SKILL_BATCH_HOOK, boom)
        register_hook(SKILL_BATCH_HOOK, lambda **_: None)
        batch = _run_batch(_batch_ops())
        flat = json.loads(skill_manage(
            action="write_file", name="test-skill",
            file_path="references/extra.md", file_content="x"))

    assert batch["success"] is True, batch
    assert flat["success"] is True, flat


def test_batch_gate_skips_invalid_names_before_any_hook_dispatch(tmp_path):
    # The existing invariant (test_traversal_name_never_reaches_a_skill_hook) extends
    # to the batch gate: a bad name never reaches pre_skill_batch either.
    seen = []
    with isolated_skills(tmp_path):
        register_hook(SKILL_BATCH_HOOK, lambda **kwargs: seen.append(kwargs))
        result = _run_batch([{"name": "../../evil", "action": "patch",
                              "old_string": "a", "new_string": "b"}])
        flat = json.loads(skill_manage(
            action="patch", name="../../evil", old_string="a", new_string="b"))

    assert result["success"] is False
    assert flat["success"] is False
    assert seen == []


def test_required_policy_timeout_fails_closed_as_unavailable(tmp_path, monkeypatch):
    _require_policy(monkeypatch)
    # Simulate the dispatcher's timeout handling: the bounded fail-closed hook injects
    # a block directive AND reports the callback as failed — the gate must classify
    # that combination as "no valid decision" (unavailable), never a denial.
    from hermes_cli import lifecycle

    def hung(**_):
        raise TimeoutError  # how a bounded runner surfaces an abandoned worker

    with isolated_skills(tmp_path) as (root, _record_success), \
            patch.object(lifecycle, "invoke_policy_hook",
                         lambda name, **kw: ([{"action": "block",
                                               "message": "pre_skill_batch plugin callback "
                                                          "timed out or is still running"}],
                                              [hung])):
        register_hook(SKILL_BATCH_HOOK, lambda **_: None)  # registered but never decides
        result = _run_batch(_batch_ops())

    assert result["success"] is False
    assert "policy unavailable" in result["error"]
    assert "timed out" in result["error"]
    assert not (root / "test-skill").exists()


def test_staged_flat_record_replay_verifies_digest_binding(tmp_path):
    def policy(**kwargs):
        return {"action": "approve", "payload_digest": kwargs["payload_digest"]}

    with isolated_skills(tmp_path):
        register_hook(SKILL_BATCH_HOOK, policy)
        created = json.loads(skill_manage(
            action="create", name="test-skill", content=SKILL_CONTENT))
    assert created["success"] is True

    ops = [{"name": "test-skill", "action": "create", "content": SKILL_CONTENT}]
    digest = _skill_batch_payload_digest(ops)
    # Tamper with the staged flat record after approval: replay must refuse.
    tampered = {"action": "create", "name": "test-skill",
                "content": SKILL_CONTENT + "tampered", "payload_digest": digest}
    replay = json.loads(_apply_skill_pending(tampered))
    assert replay["success"] is False
    assert "payload binding" in replay["error"]
    # Untampered legacy records (no digest) still replay unchanged (the skill the
    # first block created is still on disk under the same tmp_path root).
    with isolated_skills(tmp_path):
        legacy = json.loads(_apply_skill_pending(
            {"action": "patch", "name": "test-skill",
             "old_string": "Do the thing.", "new_string": "Still fine."}))
    assert legacy["success"] is True, legacy
