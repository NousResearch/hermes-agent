"""A client's declared per-turn values ride ``prompt.submit``'s ``turn_env``: exact-name allowlist from
``terminal.client_env_turn_passthrough``, bound to that ONE turn's tool subprocesses (a run identity
changes per turn, so the session's frozen ``client_env`` cannot carry it), never stored, never logged.
"""
import logging

import pytest

CONFIG = {'terminal': {'client_env_turn_passthrough': ['JOB_RUN_ID', 'JOB_API_KEY']}}
NAMES = frozenset({'JOB_RUN_ID'})


def test_collect_sends_only_declared_names_and_only_what_this_shell_exported():
    from agent.session_client_env import collect_turn_env
    environ = {'JOB_RUN_ID': 'r1', 'JOB_API_KEY': 'k1',
               'JOB_TASK_ID': 'not-declared', 'PATH': '/bin'}
    assert collect_turn_env(environ, CONFIG) == {'JOB_API_KEY': 'k1', 'JOB_RUN_ID': 'r1'}


def test_a_profile_that_declares_nothing_sends_nothing():
    from agent.session_client_env import collect_turn_env
    assert collect_turn_env({'JOB_RUN_ID': 'r1'}, {}) == {}
    assert collect_turn_env({'JOB_RUN_ID': 'r1'},
                            {'terminal': {'client_env_turn_passthrough': 'JOB_RUN_ID'}}) == {}


def test_control_names_and_malformed_names_are_never_declarable():
    from agent.session_client_env import configured_turn_names
    declared = configured_turn_names(
        {'terminal': {'client_env_turn_passthrough': ['HERMES_HOME', 'BAD NAME', '1BAD', 'OK_1']}})
    assert declared == frozenset({'OK_1'})


def test_the_gateway_keeps_only_declared_names_and_drops_the_rest():
    from agent.session_client_env import accepted_turn_env
    sent = {'JOB_RUN_ID': 'r1', 'LD_PRELOAD': '/evil.so'}
    assert accepted_turn_env(sent, NAMES) == {'JOB_RUN_ID': 'r1'}
    assert accepted_turn_env(None, NAMES) == {}


@pytest.mark.parametrize('value', [
    ['x'], {'A': 1}, {'A': 'x' * (33 * 1024)}, {f'N{i}': 'v' for i in range(40)}])
def test_a_map_that_is_not_bounded_flat_strings_is_refused(value):
    from agent.session_client_env import accepted_turn_env
    with pytest.raises(ValueError):
        accepted_turn_env(value, frozenset({'A'}))


def test_the_values_reach_this_turns_tool_subprocesses_and_no_later_one(monkeypatch):
    from agent.session_client_env import client_env_scope, turn_env_scope
    from tools.environments.local import _make_run_env
    monkeypatch.setenv('PATH', '/daemon/bin:/usr/bin')
    monkeypatch.delenv('JOB_RUN_ID', raising=False)
    turn = {'JOB_RUN_ID': 'run-1', 'JOB_API_KEY': 'secret'}
    with client_env_scope({'PATH': '/client/bin:/usr/bin'}), turn_env_scope(turn):
        env = _make_run_env({})
        assert (env['JOB_RUN_ID'], env['JOB_API_KEY']) == ('run-1', 'secret')
    with client_env_scope({'PATH': '/client/bin:/usr/bin'}):   # the next turn of the same session
        assert 'JOB_RUN_ID' not in _make_run_env({})
    with turn_env_scope(turn):   # a client that sent no client_env at all still gets its turn's values
        assert _make_run_env({})['JOB_RUN_ID'] == 'run-1'
    assert 'JOB_RUN_ID' not in _make_run_env({})


def test_the_values_are_never_logged(caplog):
    from agent.session_client_env import accepted_turn_env, turn_env_scope
    secret = 'job-run-secret-value'
    with caplog.at_level(logging.DEBUG):
        with turn_env_scope({'JOB_API_KEY': secret}):
            accepted_turn_env({'JOB_API_KEY': secret},
                              frozenset({'JOB_API_KEY'}))
    assert secret not in caplog.text


def test_a_second_turn_of_one_session_reads_its_own_values_not_the_snapshots(tmp_path):
    """The session's shell snapshot is written on the first command and re-sourced by every later
    one, so a per-turn value it captured then used to answer the next turn: a resumed session
    read the PREVIOUS run's run id and credential, and its mutation carried a dead run id.
    """
    import os

    from agent.session_client_env import turn_env_scope
    from tools.environments.local import LocalEnvironment

    if os.name == 'nt':
        pytest.skip('drives a real bash session snapshot')
    env = LocalEnvironment(cwd=str(tmp_path), timeout=20, env={'TERMINAL_TEMP_DIR': str(tmp_path)})
    assert env._snapshot_ready, 'the regression is about a sourced snapshot; login bash must work here'

    def read(script):
        return env.execute(script)['output'].strip()

    with turn_env_scope({'JOB_RUN_ID': 'run-1', 'JOB_TASK_ID': 'task-1'}):
        assert read('printf \'%s %s\' "$JOB_RUN_ID" "$JOB_TASK_ID"') == 'run-1 task-1'
    with turn_env_scope({'JOB_RUN_ID': 'run-2', 'JOB_TASK_ID': 'task-2'}):
        assert read('printf \'%s %s\' "$JOB_RUN_ID" "$JOB_TASK_ID"') == 'run-2 task-2'
    # A name this turn does not send must be unset, never the previous turn's value.
    with turn_env_scope({'JOB_RUN_ID': 'run-3'}):
        assert read('printf \'%s %s\' "$JOB_RUN_ID" "${JOB_TASK_ID-unset}"') == 'run-3 unset'

    # The values are never stored with the session: a later shell cannot source them back.
    snapshot = open(env._snapshot_path, encoding='utf-8').read()
    assert 'JOB_RUN_ID' not in snapshot and 'JOB_TASK_ID' not in snapshot
    # ...and a user's own export still persists across commands, as before.
    assert read('export USER_FLAG=kept; printf \'%s\' "$USER_FLAG"') == 'kept'
    assert read('printf \'%s\' "${USER_FLAG-unset}"') == 'kept'
