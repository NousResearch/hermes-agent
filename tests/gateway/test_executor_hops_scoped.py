"""Regression coverage for the scoped executor hops on /review and /reload-skills.

257da5ca07 routed both hops through ``_run_in_executor_with_context`` instead
of a bare ``loop.run_in_executor(None, ...)``: a bare hop starts the worker
with an empty context, so ``get_hermes_home()``-relative reads (the reviewer
subagent's home + secret scope, ``skills.external_dirs``) resolved the launch
home instead of the routed profile under multiplex. These tests pin that the
handlers hop through the scoped helper and that the worker callable actually
executes inside it.
"""

import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.slash_commands import GatewaySlashCommandsMixin


class _CapturingScope:
    """Stand-in for ``_run_in_executor_with_context`` that really runs the fn."""

    def __init__(self):
        self.calls = []

    async def __call__(self, fn, *args, **kwargs):
        self.calls.append(fn)
        return fn()


def _review_host(agent):
    return SimpleNamespace(
        _session_key_for_source=lambda _source: "k1",
        _running_agents=set(),
        _agent_cache={"k1": (agent, 0.0)},
        _agent_cache_lock=threading.Lock(),
        _run_in_executor_with_context=_CapturingScope(),
    )


@pytest.mark.asyncio
async def test_review_hops_through_scoped_executor():
    """``/review`` must dispatch via ``_run_in_executor_with_context``."""
    import agent.review_engine as review_engine

    agent = SimpleNamespace(_session_messages=[{"role": "user", "content": "hello"}])
    host = _review_host(agent)
    event = SimpleNamespace(
        get_command_args=lambda: "focus the review", source=object()
    )

    with patch.object(
        review_engine, "start_review", return_value={"status": "dispatched"}
    ) as start_review:
        response = await GatewaySlashCommandsMixin._handle_review_command(host, event)

    scope = host._run_in_executor_with_context
    assert len(scope.calls) == 1, (
        "/review must hop through _run_in_executor_with_context, not a bare "
        "loop.run_in_executor — a bare hop runs the reviewer subagent under "
        "the launch home with no secret scope"
    )
    start_review.assert_called_once()
    assert response.startswith("⚖ Review subagent dispatched")


@pytest.mark.asyncio
async def test_reload_skills_hops_through_scoped_executor():
    """``/reload-skills`` must rescan via ``_run_in_executor_with_context``."""
    import agent.skill_commands as skill_commands

    captured = []

    async def fake_scoped(fn, *args, **kwargs):
        captured.append(fn)
        return fn()

    host = SimpleNamespace(
        _run_in_executor_with_context=fake_scoped,
        adapters={},
        _pending_skills_reload_notes={},
    )
    event = SimpleNamespace(get_command_args=lambda: "")

    with patch.object(
        skill_commands,
        "reload_skills",
        return_value={"added": [], "removed": [], "total": 0},
    ) as reload_skills:
        response = await GatewaySlashCommandsMixin._handle_reload_skills_command(
            host, event
        )
        patched_fn = skill_commands.reload_skills

    assert captured == [patched_fn], (
        "/reload-skills must hop through _run_in_executor_with_context — the "
        "rescan walks get_hermes_home()/skills, a contextvar override under "
        "multiplex"
    )
    reload_skills.assert_called_once()
    assert isinstance(response, str)
