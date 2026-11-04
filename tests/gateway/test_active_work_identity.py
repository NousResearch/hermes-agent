"""``active-work`` and the restart guard must name a managed turn, never the gateway's own pid.

Observed defect (t_925a816f): ``hermes gateway restart`` printed

    Refusing to restart the gateway: 2 live session(s) would be interrupted.
      • managed run pid 3612
      • managed run pid 3612

3612 was the GATEWAY's pid, printed once per managed turn, with no session, admission or task
identity. An operator could not tell two real worker turns from a leak, so a ``--yes`` restart
was made blind (and killed two real runs). ``_describe_active_work`` only counted managed turns;
it never described them.

These tests pin the identity: the unit carries the session, its admission, the CHILD's pid and
the served profile, and the refusal text prints all of it.
"""
import threading
from types import SimpleNamespace

import pytest


def _worker(*, pid=4321, bound=True, session='local-abc', admission='adm-7', profile='/h/profiles/nico',
            started=None, stopped=False):
    closed = threading.Event()
    if stopped:
        closed.set()
    stop = threading.Event()
    return SimpleNamespace(
        closed=closed, stop=stop, process=SimpleNamespace(pid=pid),
        worker=(pid, 'birth-1') if bound else None,
        session_id=session, admission_id=admission, route='agent:nico:kanban',
        profile=profile, started_at=started, loss_cause=None,
    )


def _runner(workers, *, managed_total=None):
    """A runner whose one authority owns *workers*, with no chat/api/cron work of its own."""
    authority = SimpleNamespace(_managed_workers=workers, sessions={}, _mutation_tasks=(),
                                _bot_mailbox_operations=(), _side_agents={}, _cron_cancellations={})
    units = {'managed': len(workers) if managed_total is None else managed_total}
    return SimpleNamespace(session_authority=authority, _running_agents={},
                           _sessions_map=dict, _active_api_run_count=lambda: 0,
                           _active_deferred_agent_worker_count=lambda: 0,
                           _active_work_count=lambda: units['managed'],
                           session_authorities=None)


def _describe(runner):
    from gateway.run_shutdown import GatewayShutdownMixin
    return GatewayShutdownMixin._describe_active_work(runner)


def test_active_work_names_each_managed_turn_by_session_and_admission():
    import time
    runner = _runner({'local-abc': _worker(started=time.time() - 90)})
    managed = [u for u in _describe(runner) if u['kind'] == 'managed']
    assert len(managed) == 1
    unit = managed[0]
    assert unit['session'] == 'local-abc'
    assert unit['admission_id'] == 'adm-7'
    assert unit['pid'] == 4321                      # the child, not the gateway
    assert unit['profile'].endswith('profiles/nico')
    assert 89 <= unit['elapsed_s'] <= 120


def test_a_managed_turn_names_the_spawn_handle_before_it_is_reserved():
    unit = _describe(_runner({'s1': _worker(pid=999, bound=False)}))[0]
    assert unit['kind'] == 'managed' and unit['pid'] == 999


def test_a_closed_worker_is_not_counted_and_leaves_the_rest_of_the_total_named(monkeypatch):
    monkeypatch.setattr('gateway.run_runtime.managed_turn_count', lambda runner: 1)  # one owner mutation
    runner = _runner({'gone': _worker(stopped=True)})
    units = _describe(runner)
    assert [u['kind'] for u in units] == ['owner_work']   # unnamed remainder of the managed total


def test_active_work_keeps_the_gateway_total_truthful(monkeypatch):
    """Two named turns plus one unnamed owner mutation still report 3, so the refusal cannot undercount."""
    from hermes_cli.update_cmd_drain_report import describe_active_work_unit
    monkeypatch.setattr('gateway.run_runtime.managed_turn_count', lambda runner: 3)
    runner = _runner({'a': _worker(session='local-a'), 'b': _worker(session='local-b')})
    units = _describe(runner)
    assert [u['kind'] for u in units].count('managed') == 2
    assert [u['kind'] for u in units].count('owner_work') == 1
    rendered = [describe_active_work_unit(u) for u in units]
    assert any('managed turn local-a' in line for line in rendered)
    assert any(line.startswith('owner_work run') for line in rendered)


def test_the_restart_refusal_prints_the_turn_not_the_gateway_pid():
    """The exact shape an operator sees: the identity line, and never ``managed run pid <gateway>``."""
    import os
    from pathlib import Path
    from hermes_cli import gateway_lifecycle_guards as guards

    runner = _runner({'local-abc': _worker()})
    work = {'active_agents': 1, 'active_work': _describe(runner)}
    lines = guards._work_lines('restart', Path('/h'), work, 'hermes gateway {verb} --interrupt')
    text = '\n'.join(lines)
    assert 'managed turn local-abc' in text
    assert 'pid 4321' in text and 'admission adm-7' in text
    assert f'managed run pid {os.getpid()}' not in text
    assert 'managed run pid' not in text


def test_the_profile_scoped_verb_names_a_managed_turn_by_its_session(monkeypatch):
    """``active-work {"profile": ...}`` answers what unserving it stops: the managed turn, named."""
    from gateway.run_active_work_verb import active_work_verb
    from tools import async_delegation
    monkeypatch.setattr(async_delegation, "list_async_delegations", list)
    authority = SimpleNamespace(_managed_workers={'local-abc': _worker()}, _cron_cancellations={},
                                _side_agents={})
    runner = SimpleNamespace(_describe_active_work=list, _active_work_count=lambda: 1,
                             _served_profile_homes={"nico": "/h/profiles/nico"},
                             session_authorities=SimpleNamespace(for_home=lambda home: authority))
    answer = active_work_verb(runner, SimpleNamespace(call_soon_threadsafe=lambda fn: fn()))(
        {"profile": "nico"})
    unit = answer["active_work"][0]
    assert unit["kind"] == "managed" and unit["session"] == "local-abc"
    assert unit["admission_id"] == "adm-7" and unit["pid"] == 4321
