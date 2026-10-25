"""The CLI adopts a resumed session's own effort before its first frame.

A local session owns its effort (a ``/reasoning`` pin frozen into the policy), but the resumed
agent is built lazily and ``reasoning_config`` still holds the launch profile's value. The
resolver falls back to the attachment's snapshot when no agent exists yet, and
``_restore_session_state`` calls it on the ``-c/--continue`` path, so both resume entries report
the level the next turn will run instead of the profile's.
"""

from types import SimpleNamespace

from hermes_cli.cli_model_switch_mixin import _sync_cli_reasoning_from_session

_HIGH = {"enabled": True, "effort": "high"}
_LOW = {"enabled": True, "effort": "low"}
_MAX = {"enabled": True, "effort": "max"}
_DISABLED = {"enabled": False}


def _cli(*, reasoning=_HIGH, agent=None, snapshot=None):
    store = SimpleNamespace()
    if snapshot is not None:
        store.current = snapshot
    return SimpleNamespace(reasoning_config=reasoning, agent=agent, _session_db=store)


def _agent(info):
    return SimpleNamespace(_state=lambda: SimpleNamespace(info=info))


def test_resume_snapshot_wins_before_the_agent_exists():
    cli = _cli(snapshot={"info": {"reasoning_effort": "max"}})
    _sync_cli_reasoning_from_session(cli)
    assert cli.reasoning_config == _MAX


def test_live_agent_wins_over_the_snapshot():
    cli = _cli(agent=_agent({"reasoning_effort": "low"}),
               snapshot={"info": {"reasoning_effort": "max"}})
    _sync_cli_reasoning_from_session(cli)
    assert cli.reasoning_config == _LOW


def test_local_cli_keeps_its_own_resolve():
    cli = _cli()
    _sync_cli_reasoning_from_session(cli)
    assert cli.reasoning_config == _HIGH


def test_disabled_snapshot_reports_disabled():
    cli = _cli(snapshot={"info": {"reasoning_effort": "none"}})
    _sync_cli_reasoning_from_session(cli)
    assert cli.reasoning_config == _DISABLED


def test_empty_snapshot_falls_through_to_the_resolve():
    cli = _cli(snapshot={"info": {}})
    _sync_cli_reasoning_from_session(cli)
    assert cli.reasoning_config == _HIGH


def test_restore_session_state_adopts_the_snapshot_before_the_first_frame():
    """``-c/--continue`` restores cwd/yolo/model here; the effort must be adopted in the same
    step, before the status line renders, or the first frame shows the launch profile's level."""
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    cli = _cli(snapshot={"info": {"reasoning_effort": "max"}})
    cli._restore_session_cwd = lambda meta, quiet=False: None
    cli._restore_session_yolo = lambda meta, quiet=False: None
    cli._restore_session_model = lambda meta, quiet=False: None
    CLIAgentSetupMixin._restore_session_state(cli, {})
    assert cli.reasoning_config == _MAX
