"""Operator policy reaches real goal loops without crossing profile boundaries."""

import json
import os
import shlex
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from hermes_cli import goals
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@contextmanager
def _profile(home):
    token = set_hermes_home_override(home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def _write(home, settings):
    home.mkdir(exist_ok=True)
    (home / 'config.yaml').write_text(json.dumps({'goals': settings}), encoding='utf-8')


def _reply(verdict):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=json.dumps({'verdict': verdict, 'reason': 'next step'})
    ))])


def test_profile_policy_reaches_session_and_judge_without_changing_defaults(tmp_path, monkeypatch):
    launch, active, other = (tmp_path / name for name in ('launch', 'active', 'other'))
    _write(launch, {'autonomy': 'never_ask', 'continuation_instructions': 'launch-only'})
    settings = {'autonomy': 'best_judgement', 'continuation_instructions': ['Choose reversible steps.', 'Explain {why}.']}
    _write(active, settings)
    _write(other, {})
    monkeypatch.setenv('HERMES_HOME', str(launch))
    captured = []
    def call_llm(**kwargs):
        captured.append(kwargs['messages'])
        return _reply('continue')
    monkeypatch.setattr('agent.auxiliary_client.call_llm', call_llm)
    goals._DB_CACHE.clear()
    try:
        for index, home in enumerate((active, other, active)):
            with _profile(home):
                from hermes_cli.config_effective import load_user_config_effective
                assert load_user_config_effective()['goals'] == (settings if home == active else {})
                mgr = goals.GoalManager(session_id=f'policy-{index}')
                # The text is user-owned: never replace phrases inside the goal or format braces twice.
                objective = 'Write {result}. If you are blocked and need input from the user, say so clearly and stop.'
                mgr.set(objective)
                prompt = mgr.next_continuation_prompt()
                if home == other:
                    assert prompt == goals.CONTINUATION_PROMPT_TEMPLATE.format(goal=objective)
                else:
                    assert 'Goal autonomy (best_judgement)' in prompt
                    assert 'Choose reversible steps.\nExplain {why}.' in prompt
                assert objective in prompt
                assert 'launch-only' not in prompt
                mgr.add_subgoal('include {examples}')
                subgoal_prompt = mgr.next_continuation_prompt()
                mgr.set_contract(goals.GoalContract(verification='tests pass', stop_when='paid access required'))
                contract_prompt = mgr.next_continuation_prompt()
                assert 'paid access required' in contract_prompt
                assert 'include {examples}' in contract_prompt
                if home == active:
                    assert 'Goal autonomy (best_judgement)' in subgoal_prompt
                    assert 'If you hit the stated stop condition, say so clearly and stop.' in contract_prompt
                else:
                    assert subgoal_prompt == goals.CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
                        goal=objective, subgoals_block=mgr.state.render_subgoals_block())
                    assert 'Additional goal instructions:' not in contract_prompt
                assert mgr.evaluate_after_turn('Should I choose A or B?')['should_continue']
                messages = captured[-1]
                if home == active:
                    assert 'Goal autonomy: best_judgement.' in messages[1]['content']
                    assert 'not by itself BLOCKED' in messages[1]['content']
                    assert 'next step needs user input to proceed' not in messages[0]['content']
                else:
                    assert messages[0]['content'] == goals.JUDGE_SYSTEM_PROMPT
                    assert 'Goal autonomy:' not in messages[1]['content']
        # The real CLI loader also retains the registered operator keys.
        with _profile(active):
            import cli
            from hermes_cli.cli_config_load import load_cli_config
            monkeypatch.setattr(cli, '_hermes_home', active)
            monkeypatch.delenv('HERMES_IGNORE_USER_CONFIG', raising=False)
            assert load_cli_config()['goals']['continuation_instructions'] == settings['continuation_instructions']
        # Malformed/unknown values fail softly, including an unhashable enum value.
        for invalid in ({'autonomy': ['bad'], 'continuation_instructions': 12}, {'autonomy': 'unknown'}, []):
            _write(other, invalid)
            with _profile(other):
                mgr = goals.GoalManager(session_id='invalid')
                mgr.set('g')
                assert mgr.next_continuation_prompt() == goals.CONTINUATION_PROMPT_TEMPLATE.format(goal='g')
    finally:
        goals._DB_CACHE.clear()


@pytest.mark.skipif(os.name != 'posix', reason='real POSIX shell gate command')
def test_gate_and_worker_policy_preserve_evidence_inheritance_and_budgets(tmp_path, monkeypatch):
    home = tmp_path / 'profile'
    _write(home, {'autonomy': 'never_ask', 'continuation_instructions': 'Keep {evidence}.'})
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('TERMINAL_CWD', str(tmp_path))
    captured = []
    verdicts = iter(('continue', 'done', 'continue', 'done', 'continue'))
    def call_llm(**kwargs):
        captured.append(kwargs['messages'])
        return _reply(next(verdicts))
    monkeypatch.setattr('agent.auxiliary_client.call_llm', call_llm)
    goals._DB_CACHE.clear()
    try:
        with _profile(home):
            mgr = goals.GoalManager(session_id='gate-policy')
            mgr.set('repair gate', max_turns=3)
            command = f'{shlex.quote(sys.executable)} -c ' + shlex.quote('print("gate {evidence}"); raise SystemExit(1)')
            mgr.add_gate(command, max_retries=3)
            decision = mgr.evaluate_after_turn('working')
            prompt = decision['continuation_prompt']
            assert decision['verdict'] == 'gate_failed' and decision['should_continue']
            assert 'gate {evidence}' in prompt and 'Exit code: 1' in prompt
            assert 'Goal autonomy (never_ask)' in prompt and 'Keep {evidence}.' in prompt
            assert not captured  # A real failed gate never reaches the LLM judge.
            for override in ('', ['Worker {report}.']):
                _write(home, {'autonomy': 'never_ask', 'continuation_instructions': 'Keep {evidence}.', 'worker_instructions': override})
                prompts, blocks = [], []
                def run_turn(prompt):
                    prompts.append(prompt)
                    return 'working'
                result = goals.run_kanban_goal_loop(
                    task_id='policy-task', goal_text='finish it', run_turn=run_turn,
                    task_status_fn=lambda: 'done' if len(prompts) == 2 else 'running',
                    block_fn=blocks.append, max_turns=3, first_response='initial work',
                )
                assert result['outcome'] == 'completed_by_worker' and not blocks
                assert len(prompts) == 2 and 'task is still open' in prompts[1]
                for prompt in prompts:
                    assert 'Goal autonomy (never_ask)' in prompt
                    assert 'kanban_block' in prompt and 'kanban_request_review' in prompt
                    assert ('Worker {report}.' if override else 'Keep {evidence}.') in prompt
                    if override:
                        assert 'Keep {evidence}.' not in prompt
                assert 'Goal autonomy: never_ask.' in captured[-1][1]['content']
            turns, blocks = [], []
            result = goals.run_kanban_goal_loop(
                task_id='budget-task', goal_text='finish it', run_turn=turns.append,
                task_status_fn=lambda: 'running', block_fn=blocks.append, max_turns=1, first_response='initial work',
            )
            assert result['outcome'] == 'blocked_budget' and blocks and not turns
            # Session gate retries/turn budget still stop even in never_ask mode.
            mgr.evaluate_after_turn('working')
            assert not mgr.evaluate_after_turn('working')['should_continue']
    finally:
        goals._DB_CACHE.clear()
