"""Configurable /goal continuation prompts — ``goals.continuation_prompt*`` and ``goals.directives``.

The loop feeds a user-role continuation message back into the session every turn. These tests pin
the contract: the built-in wording still applies by default, a configured template wins, the
standing directives are appended to every variant (and to the first-turn kick), and bad config
falls back to the built-in prompt instead of losing the goal or crashing the loop.
"""

from __future__ import annotations

import pytest

CHARTER = ("Work from the project root. Use your best judgement on ambiguity and record the "
           "decision where the next session will read it; when something blocks you, write down "
           "what would unblock it, push it down the road and take the next item.")


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


def _write_config(home, goals_block: str) -> None:
    (home / "config.yaml").write_text(f"goals:\n{goals_block}", encoding="utf-8")


def _manager(session_id: str):
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_id=session_id)
    mgr.set("ship the release")
    return mgr


def _prompt(mgr) -> str:
    prompt = mgr.next_continuation_prompt()
    assert prompt is not None
    return prompt


class TestBuiltInPrompt:
    def test_default_prompt_is_unchanged(self, hermes_home):
        from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

        prompt = _prompt(_manager("goal-default"))
        assert prompt == CONTINUATION_PROMPT_TEMPLATE.format(goal="ship the release")

    def test_blank_keys_mean_built_in(self, hermes_home):
        _write_config(hermes_home, (
            "  continuation_prompt: ''\n"
            "  continuation_prompt_with_subgoals: ''\n"
            "  continuation_prompt_with_contract: ''\n"
            "  directives: ''\n"
        ))
        from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

        prompt = _prompt(_manager("goal-blank"))
        assert prompt == CONTINUATION_PROMPT_TEMPLATE.format(goal="ship the release")


class TestConfiguredTemplate:
    def test_continuation_prompt_replaces_the_wording(self, hermes_home):
        _write_config(hermes_home, (
            "  continuation_prompt: 'GOAL: {goal}\\nNext concrete step, please.'\n"
        ))
        prompt = _prompt(_manager("goal-template"))
        assert prompt.startswith("GOAL: ship the release")
        assert "Next concrete step, please." in prompt
        # The built-in framing is gone, not merely appended to.
        assert "[Continuing toward your standing goal]" not in prompt

    def test_subgoals_variant_template(self, hermes_home):
        _write_config(hermes_home, (
            "  continuation_prompt_with_subgoals: 'SUB {goal}\\nCRITERIA:\\n{subgoals_block}'\n"
        ))
        mgr = _manager("goal-subgoals")
        mgr.add_subgoal("write the changelog")
        prompt = _prompt(mgr)
        assert prompt.startswith("SUB ship the release")
        assert "write the changelog" in prompt

    def test_contract_variant_template(self, hermes_home):
        from hermes_cli.goals import GoalContract

        _write_config(hermes_home, (
            "  continuation_prompt_with_contract: 'CONTRACT {goal}\\n{contract_block}'\n"
        ))
        mgr = _manager("goal-contract")
        mgr.set_contract(GoalContract(outcome="release ships", verification="pytest -q"))
        prompt = _prompt(mgr)
        assert prompt.startswith("CONTRACT ship the release")
        assert "pytest -q" in prompt


class TestDirectives:
    def test_directives_append_to_every_variant(self, hermes_home):
        from hermes_cli.goals import GoalContract

        _write_config(hermes_home, f"  directives: '{CHARTER}'\n")

        base = _prompt(_manager("goal-dir-base"))
        assert "ship the release" in base and CHARTER in base

        sub_mgr = _manager("goal-dir-sub")
        sub_mgr.add_subgoal("write the changelog")
        assert CHARTER in _prompt(sub_mgr)

        con_mgr = _manager("goal-dir-con")
        con_mgr.set_contract(GoalContract(outcome="release ships", verification="pytest -q"))
        assert CHARTER in _prompt(con_mgr)

    def test_directives_are_not_duplicated(self, hermes_home):
        _write_config(hermes_home, (
            "  directives: 'Use your best judgement.'\n"
            "  continuation_prompt: '[Goal] {goal}\\n\\nUse your best judgement.'\n"
        ))
        prompt = _prompt(_manager("goal-dedup"))
        assert prompt.count("Use your best judgement.") == 1

    def test_kick_prompt_carries_directives(self, hermes_home):
        from hermes_cli.goals import goal_kick_prompt

        _write_config(hermes_home, f"  directives: '{CHARTER}'\n")
        kick = goal_kick_prompt("ship the release", "")
        assert kick.startswith("ship the release")
        assert CHARTER in kick

    def test_kick_prompt_repaste_guard_still_wins(self, hermes_home):
        from hermes_cli.goals import GOAL_ALREADY_SEEN_KICK, goal_kick_prompt

        _write_config(hermes_home, f"  directives: '{CHARTER}'\n")
        goal = "x" * 500
        assert goal_kick_prompt(goal, goal) == GOAL_ALREADY_SEEN_KICK


class TestBadConfigFallsBack:
    def test_template_without_goal_placeholder_falls_back(self, hermes_home):
        from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

        _write_config(hermes_home, "  continuation_prompt: 'Continue working on it.'\n")
        prompt = _prompt(_manager("goal-no-goal"))
        assert prompt == CONTINUATION_PROMPT_TEMPLATE.format(goal="ship the release")
        assert "ship the release" in prompt

    def test_malformed_template_falls_back(self, hermes_home):
        from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

        _write_config(hermes_home, "  continuation_prompt: 'Unclosed {goal brace'\n")
        prompt = _prompt(_manager("goal-malformed"))
        assert prompt == CONTINUATION_PROMPT_TEMPLATE.format(goal="ship the release")

    def test_non_string_value_falls_back(self, hermes_home):
        from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

        _write_config(hermes_home, "  continuation_prompt:\n    - not\n    - a string\n")
        prompt = _prompt(_manager("goal-nonstring"))
        assert prompt == CONTINUATION_PROMPT_TEMPLATE.format(goal="ship the release")
