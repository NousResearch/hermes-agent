"""A plugin session fork never spends more input tokens than the background review may.

``fork_session`` documents its aggregate input budget as the review's
(``auxiliary.background_review.max_input_tokens``) with ``input_budget`` only able to lower it.
The fork's routing block is ``{}`` without ``task`` and the plugin's own auxiliary task with one,
so the bound must be read from the user's ``background_review`` block, never from a block the
plugin wrote (its ``register_auxiliary_task`` defaults) or from an empty one.
"""

from __future__ import annotations

import os
from pathlib import Path

import hermes_yaml as yaml

from agent import background_review
from agent.side_question_lease import issue_side_question_token
from tests.agent import test_plugin_session_fork as _base
from tests.agent.test_plugin_session_fork import CAP, PLUGIN, _agent, _ctx, _install_plugin, _load, _settled

forks = _base.forks

_REGISTERS_TASK = (
    "import json\n\n\ndef register(ctx):\n"
    "    ctx.register_tool('cw_offer', 'cw', {'name': 'cw_offer', 'description': 'x', 'parameters': {\n"
    "        'type': 'object', 'properties': {}}}, lambda args, **kw: json.dumps({'ok': True}))\n"
    "    ctx.register_auxiliary_task('wisdom_fork', display_name='W', description='W',\n"
    "                                defaults={'max_input_tokens': 0})\n")


def _consented_with_review_bound(bound: int):
    _install_plugin(PLUGIN, declares=True, body=_REGISTERS_TASK)
    manager = _load([PLUGIN], granted=[PLUGIN])
    cfg_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["auxiliary"] = {"background_review": {"max_input_tokens": bound, "defer": "never"}}
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return manager


def _real_budget(forks, monkeypatch):
    """The scripted fork, but with the budget build_cache_parity_fork really computes."""
    built = background_review.build_cache_parity_fork

    def _build(parent, task_cfg, *, max_iterations, write_origin):
        fork, rt, routed = built(parent, task_cfg, max_iterations=max_iterations, write_origin=write_origin)
        fork._review_input_token_budget = background_review._review_input_token_budget(task_cfg, fork)
        return fork, rt, routed

    monkeypatch.setattr(background_review, "build_cache_parity_fork", _build)


def test_untasked_fork_keeps_the_users_review_bound(forks, monkeypatch):
    manager = _consented_with_review_bound(5000)
    _real_budget(forks, monkeypatch)
    parent = _agent()
    token = issue_side_question_token(parent, [], "turn-1")
    assert token is not None
    _settled(_ctx(manager).fork_session(token, "q"))
    assert forks.built[0]["fork"].seen["budget"] <= 5000


def test_a_plugin_task_cannot_lift_the_users_review_bound(forks, monkeypatch):
    manager = _consented_with_review_bound(5000)
    _real_budget(forks, monkeypatch)
    parent = _agent()
    token = issue_side_question_token(parent, [], "turn-1")
    _settled(_ctx(manager).fork_session(token, "q", task="wisdom_fork"))
    budget = forks.built[0]["fork"].seen["budget"]
    assert budget is not None and budget <= 5000
