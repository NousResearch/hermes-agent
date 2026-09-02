"""Regression tests for the single-query clarify guard (#94943).

``hermes chat -q`` wires the interactive prompt_toolkit clarify callback
unconditionally: a -q turn never builds the prompt_toolkit application, so
``CLIApp._clarify_callback`` polls its response queue with nothing able to
answer it — the turn hangs until ``agent.clarify_timeout`` expires (default
3600 s, 0 = unlimited). The gateway, cron jobs, the kanban dispatcher and
inter-agent wakeups all deliver work as ``hermes chat -q``, so an agent that
calls ``clarify`` in those turns stalls silently. The oneshot (-z) path
already answers immediately via ``_oneshot_clarify_callback``; this pins the
same headless behavior for -q, wired at the ``CLIAgentSetupMixin`` agent
construction site that already knows it is in single-query mode.
"""

from __future__ import annotations

import inspect


def test_no_choices_returns_immediate_headless_answer():
    from hermes_cli.cli_agent_setup_mixin import _single_query_clarify_callback

    result = _single_query_clarify_callback("Which timezone should I use?")
    assert result.startswith("[single-query mode: no user available")
    assert "most reasonable assumption" in result


def test_choices_return_pick_best_guidance():
    from hermes_cli.cli_agent_setup_mixin import _single_query_clarify_callback

    result = _single_query_clarify_callback(
        "Format?", choices=["json", "yaml"]
    )
    assert result.startswith("[single-query mode: no user available")
    assert "Pick the best option from ['json', 'yaml']" in result


def test_multi_select_choices_return_subset_guidance():
    from hermes_cli.cli_agent_setup_mixin import _single_query_clarify_callback

    result = _single_query_clarify_callback(
        "Which checks?", choices=["lint", "types", "tests"], multi_select=True
    )
    assert result.startswith("[single-query mode: no user available")
    assert "Pick the best subset from ['lint', 'types', 'tests']" in result


def test_callback_signature_matches_oneshot_contract():
    """The headless callback must be a drop-in wherever a clarify callback is
    accepted — same (question, choices, multi_select) shape as the oneshot
    one and the interactive CLIApp one."""
    from hermes_cli.cli_agent_setup_mixin import _single_query_clarify_callback
    from hermes_cli import oneshot

    ours = inspect.signature(_single_query_clarify_callback)
    oneshot_cb = inspect.signature(oneshot._oneshot_clarify_callback)
    assert list(ours.parameters) == list(oneshot_cb.parameters)


def test_agent_construction_gates_clarify_callback_on_single_query_mode():
    """The mixin's AIAgent construction site must route -q turns to the
    headless callback — the regression this pins is wiring the blocking
    interactive callback unconditionally (#94943). Source-level pin modeled
    on test_cli_active_agent_ref_wiring: _init_agent is too integration-heavy
    to drive with a stub, but the wiring contract is one expression."""
    from hermes_cli import cli_agent_setup_mixin as mixin_mod

    src = inspect.getsource(mixin_mod)
    assert "_single_query_clarify_callback" in src, (
        "the headless single-query clarify callback disappeared from the mixin"
    )
    init_agent_src = inspect.getsource(mixin_mod.CLIAgentSetupMixin._init_agent)
    assert 'clarify_callback=' in init_agent_src
    assert '"_single_query_mode"' in init_agent_src or "'_single_query_mode'" in init_agent_src, (
        "the clarify_callback wiring no longer consults _single_query_mode — "
        "-q turns would hang on the interactive modal again (#94943)"
    )
