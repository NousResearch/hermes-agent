"""Tests for pre-tool-call required argument validation.

Covers:
1. Unit tests for `validate_required_tool_args`:
   - conditional_required rules on skill_manage (create/edit missing content, patch missing old_string/new_string, write_file missing file_path/file_content)
   - schema.required on built-in tools (write_file missing content, write_file missing path)
   - valid tool calls (all required args supplied) returning None
   - schema-less / rule-less tools returning None (no false positives)
   - empty string vs None distinction (empty string preserved for tool semantics)
2. E2E tests for `handle_function_call`:
   - broken call blocked with tool_error containing actionable guidance
   - post_tool_call hook emitted with status="blocked" and error_type="missing_required_args"
   - blocked before edit approval and execution dispatch
   - plugin pre_tool_call hook argument modification (priority preserved)
   - opt-out via tool_loop_guardrails.pre_tool_validation config toggle
"""

import json
from unittest.mock import MagicMock, patch
import pytest

from model_tools import handle_function_call, validate_required_tool_args
from tools.registry import tool_error


# ─── Unit Tests: validate_required_tool_args ──────────────────────────────────


def test_skill_manage_create_missing_content():
    """skill_manage create without content must return a block message specifying content."""
    msg = validate_required_tool_args("skill_manage", {"action": "create", "name": "my-skill"})
    assert msg is not None
    assert "Tool skill_manage is missing required argument(s): content." in msg
    assert "content:" in msg


def test_skill_manage_edit_missing_content():
    """skill_manage edit without content must return a block message specifying content."""
    msg = validate_required_tool_args("skill_manage", {"action": "edit", "name": "my-skill"})
    assert msg is not None
    assert "Tool skill_manage is missing required argument(s): content." in msg
    assert "content:" in msg


def test_skill_manage_patch_missing_old_and_new_string():
    """skill_manage patch without old_string/new_string must report both as missing."""
    msg = validate_required_tool_args("skill_manage", {"action": "patch", "name": "my-skill"})
    assert msg is not None
    assert "old_string" in msg
    assert "new_string" in msg


def test_skill_manage_patch_missing_new_string_only():
    """skill_manage patch with old_string but without new_string must report new_string."""
    msg = validate_required_tool_args(
        "skill_manage",
        {"action": "patch", "name": "my-skill", "old_string": "def old(): pass"},
    )
    assert msg is not None
    assert "new_string" in msg
    assert "old_string" not in msg.split("missing required argument(s):")[1].split(".")[0]


def test_skill_manage_write_file_missing_file_path_and_file_content():
    """skill_manage write_file without file_path/file_content must report both."""
    msg = validate_required_tool_args("skill_manage", {"action": "write_file", "name": "my-skill"})
    assert msg is not None
    assert "file_path" in msg
    assert "file_content" in msg


def test_write_file_missing_content():
    """write_file without content must be blocked by schema.required check."""
    msg = validate_required_tool_args("write_file", {"path": "example.py"})
    assert msg is not None
    assert "Tool write_file is missing required argument(s): content." in msg


def test_write_file_missing_path():
    """write_file without path must be blocked by schema.required check."""
    msg = validate_required_tool_args("write_file", {"content": "print('hello')"})
    assert msg is not None
    assert "Tool write_file is missing required argument(s): path." in msg


def test_valid_calls_return_none():
    """Valid calls with all required args must return None (no block)."""
    assert validate_required_tool_args(
        "skill_manage",
        {"action": "create", "name": "my-skill", "content": "# My Skill"},
    ) is None

    assert validate_required_tool_args(
        "skill_manage",
        {"action": "patch", "name": "my-skill", "old_string": "foo", "new_string": "bar"},
    ) is None

    assert validate_required_tool_args(
        "skill_manage",
        {"action": "write_file", "name": "my-skill", "file_path": "references/ref.md", "file_content": "data"},
    ) is None

    assert validate_required_tool_args(
        "write_file",
        {"path": "test.txt", "content": "content here"},
    ) is None

    assert validate_required_tool_args(
        "web_search",
        {"query": "hermes agent"},
    ) is None


def test_tools_without_required_args_return_none():
    """Tools with empty schema.required and no conditional rules must never be blocked."""
    assert validate_required_tool_args("session_search", {}) is None
    assert validate_required_tool_args("skills_list", {}) is None
    assert validate_required_tool_args("unknown_custom_tool", {}) is None


def test_empty_string_versus_none_distinction():
    """Empty string is preserved for tool semantics (e.g. deletion in patch), while None is missing."""
    # new_string as empty string (valid replacement for deletion)
    assert validate_required_tool_args(
        "skill_manage",
        {"action": "patch", "name": "my-skill", "old_string": "to_remove", "new_string": ""},
    ) is None

    # new_string as explicit None -> treated as missing
    msg = validate_required_tool_args(
        "skill_manage",
        {"action": "patch", "name": "my-skill", "old_string": "to_remove", "new_string": None},
    )
    assert msg is not None
    assert "new_string" in msg


# ─── E2E Tests: handle_function_call ──────────────────────────────────────────


def test_e2e_handle_function_call_blocks_missing_args(monkeypatch, tmp_path):
    """handle_function_call returns tool_error JSON when missing required args."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result_raw = handle_function_call(
        "skill_manage",
        {"action": "create", "name": "broken-skill"},
        task_id="t-1",
        session_id="s-1",
    )
    result = json.loads(result_raw)
    assert "error" in result
    assert "Tool skill_manage is missing required argument(s): content." in result["error"]
    assert "Full SKILL.md content" in result["error"]


def test_e2e_post_tool_call_hook_emitted_on_validation_block(monkeypatch, tmp_path):
    """When validation blocks a tool call, post_tool_call observer hook is emitted with status='blocked'."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli import lifecycle

    hook_calls = []
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "post_tool_call")
    monkeypatch.setattr(
        lifecycle,
        "invoke_hook",
        lambda name, **kwargs: hook_calls.append((name, kwargs)) or [],
    )

    handle_function_call(
        "skill_manage",
        {"action": "create", "name": "broken-skill"},
        task_id="t-1",
        session_id="s-1",
        turn_id="turn-1",
    )

    post_calls = [c for c in hook_calls if c[0] == "post_tool_call"]
    assert len(post_calls) == 1
    _, kwargs = post_calls[0]
    assert kwargs["status"] == "blocked"
    assert kwargs["error_type"] == "missing_required_args"
    assert "content" in kwargs["error_message"]


def test_e2e_blocked_before_execution_and_edit_approval(monkeypatch, tmp_path):
    """Validation blocks broken calls before reaching registry.dispatch or edit approval."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    dispatch_mock = MagicMock()
    approval_mock = MagicMock()

    monkeypatch.setattr("model_tools.registry.dispatch", dispatch_mock)
    with patch("acp_adapter.edit_approval.maybe_require_edit_approval", approval_mock):
        result_raw = handle_function_call(
            "write_file",
            {"path": "foo.py"},  # missing content
            task_id="t-1",
            session_id="s-1",
        )

    assert "Tool write_file is missing required argument(s): content." in result_raw
    assert not dispatch_mock.called
    assert not approval_mock.called


def test_e2e_plugin_modify_supplies_missing_arg(monkeypatch, tmp_path):
    """When a plugin pre_tool_call hook modifies args to supply the missing arg, validation allows dispatch."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def fake_pre_dispatch(name, args, **kwargs):
        # Plugin supplies the missing "content"
        new_args = dict(args)
        new_args["content"] = "# Added by plugin"
        return None, new_args

    monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks", fake_pre_dispatch)
    monkeypatch.setattr(
        "model_tools.registry.dispatch",
        lambda name, args, **kw: json.dumps({"success": True, "saved_content": args.get("content")}),
    )

    result_raw = handle_function_call(
        "skill_manage",
        {"action": "create", "name": "plugin-assisted-skill"},  # missing content initially
        task_id="t-1",
        session_id="s-1",
    )
    result = json.loads(result_raw)
    assert result.get("success") is True
    assert result.get("saved_content") == "# Added by plugin"


def test_e2e_pre_tool_validation_disabled_in_config(monkeypatch, tmp_path):
    """When pre_tool_validation is disabled in config, validation is bypassed and dispatch is reached."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"tool_loop_guardrails": {"pre_tool_validation": False}},
    )

    dispatch_mock = MagicMock(return_value=json.dumps({"success": True, "custom": "dispatched"}))
    monkeypatch.setattr("model_tools.registry.dispatch", dispatch_mock)

    result_raw = handle_function_call(
        "skill_manage",
        {"action": "create", "name": "no-val-skill"},  # missing content, but validation disabled
        task_id="t-1",
        session_id="s-1",
    )
    result = json.loads(result_raw)
    assert result.get("custom") == "dispatched"
    assert dispatch_mock.called


def test_repeated_malformed_call_escalates_in_same_turn():
    """The exact same malformed call shape repeated in one turn escalates the message."""
    sess, turn = "s_rep", "t_rep"
    args = {"action": "create", "name": "x"}
    msgs = [
        validate_required_tool_args("skill_manage", args, session_id=sess, turn_id=turn)
        for _ in range(3)
    ]
    # 1st and 2nd are plain actionable blocks
    assert msgs[0] is not None and "stop retrying" not in msgs[0]
    assert msgs[1] is not None and "stop retrying" not in msgs[1]
    # 3rd escalates with a hard stop warning
    assert msgs[2] is not None
    assert "3rd time" in msgs[2] and "stop retrying" in msgs[2]


def test_repeated_block_count_is_per_turn():
    """The repeat counter resets per (session, turn) — a new turn starts fresh."""
    args = {"action": "create", "name": "x"}
    for i in range(2):
        m = validate_required_tool_args(
            "skill_manage", args, session_id="s_turn", turn_id=f"turn_{i}"
        )
        assert m is not None and "stop retrying" not in m
    # Same session, same shape, but a fresh turn: no escalation on first call
    m = validate_required_tool_args(
        "skill_manage", args, session_id="s_turn", turn_id="turn_new"
    )
    assert m is not None and "stop retrying" not in m


def test_validate_uses_resolved_guardrail_config():
    """The resolved ToolCallGuardrailConfig is the single authority for conditional rules."""
    from agent.tool_guardrails import ToolCallGuardrailConfig

    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "conditional_required": {
                "skill_manage": [
                    {"condition": {"action": "custom_op"}, "require": ["custom_field"]}
                ]
            }
        }
    )
    # Matches the rule from the resolved config object (not bundled defaults)
    m = validate_required_tool_args(
        "skill_manage",
        {"action": "custom_op"},
        session_id="s_cfg",
        turn_id="t_cfg",
        tlg=cfg,
    )
    assert m is not None and "custom_field" in m


def test_repeat_escalation_requires_session_and_turn_context():
    """Without session/turn context (e.g. MCP call paths) repeats never escalate,
    so long-lived processes don't collide on a shared global key."""
    args = {"action": "create", "name": "x"}
    for _ in range(3):
        m = validate_required_tool_args("skill_manage", args, session_id="", turn_id="")
        assert m is not None and "stop retrying" not in m
