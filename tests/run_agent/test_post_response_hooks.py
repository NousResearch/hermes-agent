"""Tests for agent/post_response_hooks.py — post-response hook extension point.

Covers HookResult dataclass, hook loading, system prompt injection,
response validation (pass/nudge/block actions), backward compat for
bool-returning hooks, security checks (world-writable refusal),
max_nudges config, and context schema — without hitting the network
(all I/O is mocked).
"""

import os
import platform
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.post_response_hooks import (
    DEFAULT_MAX_NUDGES,
    Hook,
    HookResult,
    build_system_prompt_additions,
    load_hooks,
    run_post_response_checks,
)


# ---------------------------------------------------------------------------
# HookResult dataclass tests
# ---------------------------------------------------------------------------


class TestHookResult:
    def test_defaults(self):
        r = HookResult(passed=True)
        assert r.passed is True
        assert r.action == ""
        assert r.data == {}

    def test_full_construction(self):
        r = HookResult(
            passed=False,
            action="block",
            data={"message": "Blocked: PII detected", "severity": "critical"},
        )
        assert r.passed is False
        assert r.action == "block"
        assert r.data["message"] == "Blocked: PII detected"
        assert r.data["severity"] == "critical"

    def test_data_defaults_to_empty_dict(self):
        r = HookResult(passed=True)
        assert r.data == {}
        # Mutating data dict on one instance doesn't affect another
        r2 = HookResult(passed=True)
        r.data["key"] = "value"
        assert r2.data == {}


# ---------------------------------------------------------------------------
# Hook.check() — HookResult and backward compat
# ---------------------------------------------------------------------------


class TestHookCheck:
    def test_check_returns_hookresult_when_no_check_fn(self):
        hook = Hook(module_name="noop")
        result = hook.check("any response", {})
        assert isinstance(result, HookResult)
        assert result.passed is True
        assert result.action == "pass"

    def test_check_delegates_to_check_fn(self):
        hook = Hook(module_name="gate", _check_fn=lambda r, c: "good" in r)
        result_good = hook.check("this is good", {})
        assert isinstance(result_good, HookResult)
        assert result_good.passed is True

        result_bad = hook.check("this is bad", {})
        assert isinstance(result_bad, HookResult)
        assert result_bad.passed is False

    def test_check_catches_exception_and_returns_pass(self):
        """Broken hook never crashes the agent — exception is swallowed."""
        def _boom(r, c):
            raise RuntimeError("boom")

        hook = Hook(module_name="broken", _check_fn=_boom)
        result = hook.check("anything", {})
        assert isinstance(result, HookResult)
        assert result.passed is True
        assert result.action == "pass"

    def test_check_catches_type_error(self):
        """Hook returning non-bool-coercible value doesn't crash."""
        def _bad_return(r, c):
            raise TypeError("bad coercion")

        hook = Hook(module_name="bad_type", _check_fn=_bad_return)
        result = hook.check("anything", {})
        assert result.passed is True

    # Backward compat: bool-returning hooks

    def test_bool_true_becomes_pass(self):
        hook = Hook(module_name="legacy_ok", _check_fn=lambda r, c: True)
        result = hook.check("response", {})
        assert isinstance(result, HookResult)
        assert result.passed is True
        assert result.action == "pass"

    def test_bool_false_becomes_nudge_with_nudge_message(self):
        hook = Hook(
            module_name="legacy_fail",
            nudge_message="Please elaborate.",
            _check_fn=lambda r, c: False,
        )
        result = hook.check("response", {})
        assert isinstance(result, HookResult)
        assert result.passed is False
        assert result.action == "nudge"
        assert result.data["message"] == "Please elaborate."

    def test_bool_false_without_nudge_message(self):
        hook = Hook(module_name="no_msg", _check_fn=lambda r, c: False)
        result = hook.check("response", {})
        assert result.passed is False
        assert result.action == "nudge"
        assert result.data["message"] == ""

    # New-style: HookResult-returning hooks

    def test_hookresult_nudge(self):
        def _check(r, c):
            return HookResult(
                passed=False,
                action="nudge",
                data={"message": "Too short, add detail."},
            )

        hook = Hook(module_name="nudger", _check_fn=_check)
        result = hook.check("short", {})
        assert result.passed is False
        assert result.action == "nudge"
        assert result.data["message"] == "Too short, add detail."
        assert result.data["hook_name"] == "nudger"

    def test_hookresult_block(self):
        def _check(r, c):
            return HookResult(
                passed=False,
                action="block",
                data={"message": "PII redacted version", "severity": "critical"},
            )

        hook = Hook(module_name="blocker", _check_fn=_check)
        result = hook.check("contains PII", {})
        assert result.passed is False
        assert result.action == "block"
        assert result.data["message"] == "PII redacted version"
        assert result.data["severity"] == "critical"
        assert result.data["hook_name"] == "blocker"

    def test_hookresult_pass(self):
        def _check(r, c):
            return HookResult(passed=True, action="pass")

        hook = Hook(module_name="passer", _check_fn=_check)
        result = hook.check("good", {})
        assert result.passed is True
        assert result.action == "pass"

    def test_hookresult_no_action_defaults_to_nudge_when_failed(self):
        """HookResult with passed=False and no action → defaults to nudge."""
        def _check(r, c):
            return HookResult(passed=False, data={"message": "fix me"})

        hook = Hook(module_name="default_action", _check_fn=_check)
        result = hook.check("bad", {})
        assert result.action == "nudge"

    def test_hookresult_no_action_defaults_to_pass_when_passed(self):
        """HookResult with passed=True and no action → defaults to pass."""
        def _check(r, c):
            return HookResult(passed=True)

        hook = Hook(module_name="default_pass", _check_fn=_check)
        result = hook.check("good", {})
        assert result.action == "pass"

    def test_unknown_return_type_treated_as_pass(self):
        """Hook returning non-bool, non-HookResult → logged and passed."""
        def _weird(r, c):
            return 42  # integer, not bool or HookResult

        hook = Hook(module_name="weird", _check_fn=_weird)
        result = hook.check("anything", {})
        assert result.passed is True
        assert result.action == "pass"


# ---------------------------------------------------------------------------
# load_hooks
# ---------------------------------------------------------------------------


class TestLoadHooks:
    def test_empty_config_returns_empty(self):
        assert load_hooks([]) == []

    def test_disabled_hook_is_skipped(self):
        configs = [{"module": "my_hook", "enabled": False}]
        assert load_hooks(configs) == []

    def test_missing_module_file_is_skipped(self, tmp_path):
        configs = [{"module": "nonexistent"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            (tmp_path / "hooks").mkdir()
            result = load_hooks(configs)
        assert result == []

    def test_module_without_hook_class_is_skipped(self, tmp_path):
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "empty_mod.py").write_text("x = 1\n")

        configs = [{"module": "empty_mod"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)
        assert result == []

    def test_valid_hook_is_loaded(self, tmp_path):
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "quality.py").write_text(
            "class Hook:\n"
            "    module_name = 'quality'\n"
            "    system_prompt_addition = 'Be thorough.'\n"
            "    nudge_message = 'Please elaborate.'\n"
            "    def check(self, response, context):\n"
            "        return len(response) > 10\n"
        )

        configs = [{"module": "quality"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)

        assert len(result) == 1
        assert result[0].module_name == "quality"
        assert result[0].system_prompt_addition == "Be thorough."
        assert result[0].nudge_message == "Please elaborate."
        assert result[0].max_nudges == DEFAULT_MAX_NUDGES
        # Bool return still works via backward compat
        check_result = result[0].check("short", {})
        assert check_result.passed is False
        check_result = result[0].check("this is a long enough response", {})
        assert check_result.passed is True

    def test_valid_hook_with_hookresult_return(self, tmp_path):
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "advanced.py").write_text(
            "from agent.post_response_hooks import HookResult\n"
            "class Hook:\n"
            "    module_name = 'advanced'\n"
            "    def check(self, response, context):\n"
            "        if 'PII' in response:\n"
            "            return HookResult(passed=False, action='block', data={'message': '[redacted]'})\n"
            "        if len(response) < 20:\n"
            "            return HookResult(passed=False, action='nudge', data={'message': 'Too short'})\n"
            "        return HookResult(passed=True, action='pass')\n"
        )

        configs = [{"module": "advanced"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)

        assert len(result) == 1
        hook = result[0]
        # Block
        r1 = hook.check("contains PII here", {})
        assert r1.action == "block"
        assert r1.data["message"] == "[redacted]"
        # Nudge
        r2 = hook.check("short", {})
        assert r2.action == "nudge"
        # Pass
        r3 = hook.check("this is a good long response without issues", {})
        assert r3.action == "pass"

    def test_invalid_config_entries_are_skipped(self):
        configs = ["not_a_dict", {"no_module_key": True}, None]
        assert load_hooks(configs) == []

    def test_multiple_hooks_ordering(self, tmp_path):
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        for name in ("alpha", "beta"):
            (hooks_dir / f"{name}.py").write_text(
                f"class Hook:\n"
                f"    module_name = '{name}'\n"
                f"    system_prompt_addition = ''\n"
                f"    nudge_message = ''\n"
            )

        configs = [{"module": "alpha"}, {"module": "beta"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)

        assert len(result) == 2
        assert result[0].module_name == "alpha"
        assert result[1].module_name == "beta"

    def test_max_nudges_from_config(self, tmp_path):
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "retry_hook.py").write_text(
            "class Hook:\n"
            "    module_name = 'retry_hook'\n"
        )

        configs = [{"module": "retry_hook", "max_nudges": 3}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)

        assert len(result) == 1
        assert result[0].max_nudges == 3

    def test_max_nudges_invalid_value_uses_default(self, tmp_path):
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "bad_cfg.py").write_text(
            "class Hook:\n"
            "    module_name = 'bad_cfg'\n"
        )

        configs = [{"module": "bad_cfg", "max_nudges": "not_a_number"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)

        assert len(result) == 1
        assert result[0].max_nudges == DEFAULT_MAX_NUDGES

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix file permissions only")
    def test_world_writable_hook_is_refused(self, tmp_path):
        """Security: world-writable hook files are not loaded."""
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        hook_file = hooks_dir / "unsafe.py"
        hook_file.write_text("class Hook:\n    module_name = 'unsafe'\n")
        hook_file.chmod(hook_file.stat().st_mode | stat.S_IWOTH)

        configs = [{"module": "unsafe"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)

        assert result == []

    def test_hook_with_import_error_is_skipped(self, tmp_path):
        """Hook that raises on import doesn't crash the loader."""
        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        (hooks_dir / "crasher.py").write_text("raise ImportError('missing dep')\n")

        configs = [{"module": "crasher"}]
        with patch("agent.post_response_hooks.get_hermes_home", return_value=tmp_path):
            result = load_hooks(configs)

        assert result == []


# ---------------------------------------------------------------------------
# build_system_prompt_additions
# ---------------------------------------------------------------------------


class TestBuildSystemPromptAdditions:
    def test_empty_hooks(self):
        assert build_system_prompt_additions([]) == ""

    def test_aggregates_additions(self):
        hooks = [
            Hook(module_name="a", system_prompt_addition="Rule A."),
            Hook(module_name="b", system_prompt_addition=""),
            Hook(module_name="c", system_prompt_addition="Rule C."),
        ]
        result = build_system_prompt_additions(hooks)
        assert result == "Rule A.\n\nRule C."

    def test_single_addition(self):
        hooks = [Hook(module_name="solo", system_prompt_addition="Only rule.")]
        assert build_system_prompt_additions(hooks) == "Only rule."


# ---------------------------------------------------------------------------
# run_post_response_checks — pass / nudge / block
# ---------------------------------------------------------------------------


class TestRunPostResponseChecks:
    def test_all_hooks_pass_returns_none(self):
        hooks = [
            Hook(module_name="a", _check_fn=lambda r, c: True),
            Hook(module_name="b", _check_fn=lambda r, c: True),
        ]
        assert run_post_response_checks(hooks, "response", {}) is None

    def test_first_failing_hook_wins(self):
        hooks = [
            Hook(module_name="pass", _check_fn=lambda r, c: True),
            Hook(
                module_name="fail1",
                nudge_message="Fix from fail1",
                _check_fn=lambda r, c: False,
            ),
            Hook(
                module_name="fail2",
                nudge_message="Fix from fail2",
                _check_fn=lambda r, c: False,
            ),
        ]
        result = run_post_response_checks(hooks, "response", {})
        assert result is not None
        assert result.passed is False
        assert result.action == "nudge"
        assert result.data["message"] == "Fix from fail1"

    def test_failing_hook_without_nudge_uses_default(self):
        hooks = [
            Hook(module_name="strict", nudge_message="", _check_fn=lambda r, c: False),
        ]
        result = run_post_response_checks(hooks, "response", {})
        assert result is not None
        assert "strict" in result.data["message"]
        assert "quality check" in result.data["message"]

    def test_empty_hooks_returns_none(self):
        assert run_post_response_checks([], "response", {}) is None

    def test_context_contains_expected_keys(self):
        """Context dict schema: user_message, messages, model."""
        received = {}

        def _capture(r, c):
            received.update(c)
            return True

        hooks = [Hook(module_name="spy", _check_fn=_capture)]
        ctx = {
            "user_message": "hello",
            "messages": [{"role": "user", "content": "hello"}],
            "model": "test-model",
        }
        run_post_response_checks(hooks, "response", ctx)
        assert received["user_message"] == "hello"
        assert received["model"] == "test-model"
        assert isinstance(received["messages"], list)

    def test_hook_exception_does_not_block_others(self):
        """Crashing hook is skipped (treated as pass), next hook still runs."""
        def _boom(r, c):
            raise ValueError("kaboom")

        hooks = [
            Hook(module_name="crasher", _check_fn=_boom),
            Hook(
                module_name="checker",
                nudge_message="check failed",
                _check_fn=lambda r, c: False,
            ),
        ]
        result = run_post_response_checks(hooks, "response", {})
        assert result is not None
        assert result.data["message"] == "check failed"

    # --- Block action ---

    def test_block_action_replaces_response(self):
        def _check(r, c):
            return HookResult(
                passed=False,
                action="block",
                data={"message": "[PII redacted]"},
            )

        hooks = [Hook(module_name="safety", _check_fn=_check)]
        result = run_post_response_checks(hooks, "sensitive data", {})
        assert result is not None
        assert result.action == "block"
        assert result.data["message"] == "[PII redacted]"

    def test_block_hook_without_message_uses_fallback(self):
        def _check(r, c):
            return HookResult(passed=False, action="block")

        hooks = [Hook(module_name="blocker", _check_fn=_check)]
        result = run_post_response_checks(hooks, "response", {})
        assert result.action == "block"
        # run_post_response_checks fills in fallback message
        assert "blocker" in result.data["message"]
        assert "quality check" in result.data["message"]

    # --- Nudge with structured data ---

    def test_nudge_with_checklist_data(self):
        def _check(r, c):
            return HookResult(
                passed=False,
                action="nudge",
                data={
                    "message": "Missing: propose solution",
                    "severity": "medium",
                    "checklist_remaining": ["propose solution", "add tests"],
                    "next_item": "propose solution",
                    "max_nudges": 3,
                },
            )

        hooks = [Hook(module_name="checklist", _check_fn=_check)]
        result = run_post_response_checks(hooks, "response", {})
        assert result.action == "nudge"
        assert result.data["severity"] == "medium"
        assert result.data["checklist_remaining"] == ["propose solution", "add tests"]

    # --- Pass action ---

    def test_pass_action_passes_through(self):
        def _check(r, c):
            return HookResult(passed=True, action="pass")

        hooks = [Hook(module_name="always_pass", _check_fn=_check)]
        result = run_post_response_checks(hooks, "response", {})
        assert result is None  # all passed

    # --- HookResult with no explicit action ---

    def test_no_action_defaults_to_nudge(self):
        def _check(r, c):
            return HookResult(passed=False, data={"message": "fix this"})

        hooks = [Hook(module_name="default_nudge", _check_fn=_check)]
        result = run_post_response_checks(hooks, "response", {})
        assert result is not None
        assert result.action == "nudge"

    def test_structured_data_preserved_in_block(self):
        """Metadata (severity, checklist) preserved through run_post_response_checks."""
        def _check(r, c):
            return HookResult(
                passed=False,
                action="block",
                data={
                    "message": "blocked",
                    "severity": "critical",
                    "checklist_remaining": [],
                },
            )

        hooks = [Hook(module_name="structured", _check_fn=_check)]
        result = run_post_response_checks(hooks, "response", {})
        assert result.action == "block"
        assert result.data["severity"] == "critical"
        assert result.data["checklist_remaining"] == []
