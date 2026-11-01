"""A routing ACK cannot authorize a restart over a retained profile writer."""
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    from gateway import control_socket
    from hermes_cli import gateway as gw
    from hermes_cli.gateway_profile_lifecycle import profile_lifecycle

    root = tmp_path / '.hermes'
    secondary = root / 'profiles' / 'worker'
    secondary.mkdir(parents=True)
    (secondary / 'config.yaml').write_text('{}')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(secondary))
    owner = SimpleNamespace(home=root, profile_label='default', profiles=('default', 'worker'))
    monkeypatch.setattr(gw, '_current_profile_name', lambda: 'worker')
    monkeypatch.setattr(gw, 'find_gateway_pids', lambda **kwargs: [])
    monkeypatch.setattr(gw, '_served_by_another_host_gateway', lambda: owner)
    monkeypatch.setattr(gw, 'named_profile_served_by_running_multiplexer', lambda: True)
    calls = []

    def serve(home, name):
        assert home == root and name == 'worker'
        calls.append(('serve', name))
        return {'served': name, 'served_profiles': ['default', name]}

    monkeypatch.setattr(control_socket, 'request_serve_profile_hot', serve)

    def run(command, answer):
        def unserve(home, name):
            assert home == root and name == 'worker'
            calls.append(('unserve', name))
            return answer

        monkeypatch.setattr(control_socket, 'request_unserve_profile', unserve)
        assert profile_lifecycle(command, SimpleNamespace()) is True
        return calls

    return run, secondary


@pytest.mark.parametrize('receipt', [
    {},  # An older server reports routing only; settlement is UNKNOWN.
    {'writer_settlement': 'retained', 'reservation': 'retained'},
    {'writer_settlement': 'settled', 'reservation': 'retained'},
    {'writer_settlement': 'settled', 'reservation': 'released', 'pending': True},
])
def test_restart_does_not_serve_without_positive_retirement(lifecycle, receipt):
    run, _ = lifecycle
    calls = run('restart', {'unserved': 'worker', 'served_profiles': ['default'], **receipt})
    assert calls == [('unserve', 'worker')], 'routing removal must not authorize re-serve'


def test_restart_serves_only_after_writer_settlement_and_reservation_release(lifecycle):
    run, _ = lifecycle
    calls = run('restart', {
        'unserved': 'worker', 'served_profiles': ['default'],
        'writer_settlement': 'settled', 'reservation': 'released',
    })
    assert calls == [('unserve', 'worker'), ('serve', 'worker')]


def test_stop_reports_retained_ownership_instead_of_complete_stop(lifecycle, capsys):
    run, secondary = lifecycle
    calls = run('stop', {
        'unserved': 'worker', 'served_profiles': ['default'],
        'writer_settlement': 'retained', 'reservation': 'retained',
    })
    assert calls == [('unserve', 'worker')]
    assert (secondary / 'gateway.parked').exists()
    output = capsys.readouterr().out
    assert 'ownership retained' in output
    assert 'bots and cron are stopped' not in output
