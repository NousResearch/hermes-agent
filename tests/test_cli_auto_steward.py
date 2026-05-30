from types import SimpleNamespace

import pytest

import cli as cli_mod
from cli import HermesCLI


class TestCLIAutoSteward:
    def _make_cli(self, enabled=True, max_hops=1, depth=0, opt_in_token="", armed=True):
        cli = HermesCLI.__new__(HermesCLI)
        cli._auto_steward_enabled = enabled
        cli._auto_steward_max_hops = max_hops
        cli._auto_steward_depth = depth
        cli._auto_steward_opt_in_token = opt_in_token
        cli._auto_steward_armed = armed
        return cli

    def test_response_requires_user_input_for_want_me_to(self):
        assert HermesCLI._response_requires_user_input(
            "Feature is done. Want me to switch the default token to something visible?"
        )

    def test_response_requires_user_input_for_confirmation_prompt(self):
        assert HermesCLI._response_requires_user_input(
            "I can keep going, but would you like me to deploy this now?"
        )

    def test_response_requires_user_input_for_missing_credentials(self):
        assert HermesCLI._response_requires_user_input(
            "I cannot proceed without your approval and the missing credential."
        )

    def test_response_without_question_does_not_require_user_input(self):
        assert not HermesCLI._response_requires_user_input(
            "Done: patched the bug, ran the focused test, and updated the docs."
        )

    def test_should_auto_steward_when_enabled_and_not_blocked(self):
        cli = self._make_cli(enabled=True, max_hops=2, depth=0)
        assert cli._should_auto_steward(
            "Done: inspected the repo. Remaining: apply the obvious safe patch and verify it.",
            {"failed": False, "partial": False, "interrupted": False},
        )

    def test_should_not_auto_steward_when_depth_budget_exhausted(self):
        cli = self._make_cli(enabled=True, max_hops=1, depth=1)
        assert not cli._should_auto_steward(
            "Done: inspected the repo. Remaining: apply the obvious safe patch.",
            {"failed": False, "partial": False, "interrupted": False},
        )

    def test_should_not_auto_steward_for_failed_result(self):
        cli = self._make_cli(enabled=True, max_hops=2, depth=0)
        assert not cli._should_auto_steward(
            "Error: provider timed out.",
            {"failed": True, "partial": False, "interrupted": False},
        )

    def test_response_looks_terminal_for_completion_language(self):
        assert HermesCLI._response_looks_terminal(
            "Done: prior request already completed exactly as asked. Remaining: none. Stopping because the work is complete."
        )

    def test_should_not_auto_steward_for_terminal_response(self):
        cli = self._make_cli(enabled=True, max_hops=3, depth=0)
        assert not cli._should_auto_steward(
            "Done: no active task beyond the prior request, which was completed. Remaining: none. Stopping because there is no further concrete task to execute.",
            {"failed": False, "partial": False, "interrupted": False},
        )

    def test_build_auto_steward_prompt_mentions_followthrough(self):
        cli = self._make_cli(enabled=True)
        prompt = cli._build_auto_steward_prompt()
        assert "stewardship followthrough" in prompt.lower()
        assert "do not stop" in prompt.lower()

    def test_message_is_auto_steward_prompt_ignores_whitespace_differences(self):
        cli = self._make_cli(enabled=True)
        prompt = cli._build_auto_steward_prompt().replace(" ", "\n")
        assert cli._message_is_auto_steward_prompt(prompt)

    def test_should_drop_unarmed_leaked_auto_steward_prompt(self):
        cli = self._make_cli(enabled=True, opt_in_token="\t", armed=False)
        assert cli._should_drop_unarmed_auto_steward_message(
            cli._build_auto_steward_prompt()
        )

    def test_should_not_drop_armed_auto_steward_prompt(self):
        cli = self._make_cli(enabled=True, opt_in_token="\t", armed=True)
        assert not cli._should_drop_unarmed_auto_steward_message(
            cli._build_auto_steward_prompt()
        )

    def test_should_not_drop_normal_message_without_token(self):
        cli = self._make_cli(enabled=True, opt_in_token="\t", armed=False)
        assert not cli._should_drop_unarmed_auto_steward_message("test")

    def test_should_not_auto_steward_when_token_required_but_missing(self):
        cli = self._make_cli(
            enabled=True, max_hops=2, depth=0, opt_in_token="\t", armed=False
        )
        assert not cli._should_auto_steward(
            "Done: did the thing. Remaining: more obvious safe steps available.",
            {"failed": False, "partial": False, "interrupted": False},
        )

    def test_should_auto_steward_when_token_required_and_armed(self):
        cli = self._make_cli(
            enabled=True, max_hops=2, depth=0, opt_in_token="\t", armed=True
        )
        assert cli._should_auto_steward(
            "Done: did the thing. Remaining: more obvious safe steps available.",
            {"failed": False, "partial": False, "interrupted": False},
        )

    def test_message_contains_opt_in_token_string(self):
        cli = self._make_cli(opt_in_token="\t")
        assert cli._message_contains_opt_in_token("hi\tthere")
        assert not cli._message_contains_opt_in_token("hi there")

    def test_message_contains_opt_in_token_multimodal_list(self):
        cli = self._make_cli(opt_in_token="\t")
        assert cli._message_contains_opt_in_token([
            {"type": "image", "image_url": "x"},
            {"type": "text", "text": "please continue\tand keep going"},
        ])
        assert not cli._message_contains_opt_in_token([
            {"type": "text", "text": "please continue and keep going"},
        ])

    def test_empty_opt_in_token_disables_gating(self):
        cli = self._make_cli(opt_in_token="", armed=False)
        assert cli._message_contains_opt_in_token("anything")
        assert cli._should_auto_steward(
            "Done: did the thing. Remaining: more obvious safe steps available.",
            {"failed": False, "partial": False, "interrupted": False},
        )
