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
