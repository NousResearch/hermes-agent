"""Per-profile lifecycle never stops the shared host or hides an installed profile."""
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    from gateway import profile_serving
    root = tmp_path / '.hermes'
    secondary = root / 'profiles' / 'worker'
    secondary.mkdir(parents=True)
    (secondary / 'config.yaml').write_text('model: {default: worker-model}\n')
    (root / 'config.yaml').write_text('gateway: {multiplex_profiles: true}\n')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(profile_serving, 'get_active_profile_name', lambda: 'default')
    return root, secondary


@pytest.mark.parametrize('include_parked', [False, True])
def test_parked_is_not_served_but_remains_installed(homes, include_parked):
    from gateway import profile_serving
    root, secondary = homes
    (secondary / 'gateway.parked').write_text('provisioned offline\n')
    if include_parked:
        actual = profile_serving.profiles_to_serve(True, include_parked=True)
    else:
        actual = profile_serving.profiles_to_serve(True)
    assert dict(actual) == ({'default': root, 'worker': secondary} if include_parked else {'default': root})
    assert profile_serving.parked_marker_path(secondary) == secondary / 'gateway.parked'
    assert profile_serving.profile_is_parked(secondary)


@pytest.mark.parametrize('multiplex', [False, True])
def test_default_marker_is_ignored_with_one_warning(homes, caplog, multiplex):
    from gateway.profile_serving import profiles_to_serve
    root, _ = homes
    (root / 'gateway.parked').touch()
    for _ in range(2):
        assert 'default' in dict(profiles_to_serve(multiplex))
    warnings = [r for r in caplog.records if 'gateway.parked' in r.message]
    assert len(warnings) == 1
    assert warnings[0].levelname == 'WARNING'


@pytest.mark.parametrize('verb', ['stop', 'start', 'restart'])
def test_cli_lifecycle_orders_marker_before_socket(homes, monkeypatch, capsys, verb):
    from hermes_cli import gateway as gw
    from gateway import control_socket
    root, secondary = homes
    marker = secondary / 'gateway.parked'
    owner = SimpleNamespace(home=root, profile_label='default', profiles=('default', 'worker'),
                            describe=lambda: 'test host')
    monkeypatch.setenv('HERMES_HOME', str(secondary))
    monkeypatch.setattr(gw, '_current_profile_name', lambda: 'worker')
    monkeypatch.setattr(gw, '_refuse_from_inside_gateway', lambda *a: None)
    monkeypatch.setattr(gw, 'find_gateway_pids', lambda **kw: [])
    monkeypatch.setattr(gw, '_served_by_another_host_gateway', lambda *a: owner)
    monkeypatch.setattr(gw, 'named_profile_served_by_running_multiplexer', lambda *a: True)
    monkeypatch.setattr(gw, '_host_multiplexer_for_all_verb', lambda: owner)
    calls = []

    def unserve(home, name):
        assert home == root and name == 'worker'
        assert marker.exists() is (verb == 'stop')
        calls.append('unserve')
        return {'unserved': name, 'served_profiles': ['default']}

    def serve(home, name):
        assert home == root and name == 'worker'
        assert not marker.exists()
        calls.append('serve')
        return {'served': name, 'served_profiles': ['default', name]}

    monkeypatch.setattr(control_socket, 'request_unserve_profile', unserve, raising=False)
    monkeypatch.setattr(control_socket, 'request_serve_profile_hot', serve, raising=False)
    if verb == 'start':
        marker.touch()
    # Precedence: a standalone profile is never parked, even while a stale host record lists it.
    from hermes_cli.gateway_profile_lifecycle import profile_lifecycle
    (secondary / 'config.yaml').write_text('gateway: {standalone: true}\n')
    assert profile_lifecycle(verb, SimpleNamespace()) is False and calls == []
    assert marker.exists() is (verb == 'start')
    (secondary / 'config.yaml').write_text('model: {default: worker-model}\n')
    getattr(gw, '_cmd_' + verb)(SimpleNamespace())
    assert calls == {'stop': ['unserve'], 'start': ['serve'], 'restart': ['unserve', 'serve']}[verb]
    assert marker.exists() is (verb == 'stop')
    output = capsys.readouterr().out
    assert {'stop': "Profile 'worker' parked; its bots and cron are stopped.",
            'start': "Profile 'worker' served", 'restart': "Profile 'worker' restarted"}[verb] in output


def test_parked_status_and_topology_keep_roster(homes, monkeypatch, capsys):
    from hermes_cli import gateway as gw, profiles
    from profiles import current as profile_current
    from hermes_cli.web_server_gateway import _collect_profile_gateway_topology
    root, secondary = homes
    (secondary / 'gateway.parked').touch()
    monkeypatch.setenv('HERMES_HOME', str(secondary))
    monkeypatch.setattr(gw, '_current_profile_name', lambda: 'worker')
    monkeypatch.setattr(profile_current, 'get_active_profile_name', lambda: 'worker')
    monkeypatch.setattr(profiles, '_check_gateway_running', lambda home: False)
    gw._cmd_status(SimpleNamespace())
    assert 'parked (hermes -p worker gateway start)' in capsys.readouterr().out
    topology = _collect_profile_gateway_topology()
    assert topology['profiles'] == ['default', 'worker']
    assert topology['parked_profiles'] == ['worker']


def test_parked_profile_keeps_implicit_host_multiplexed(homes, monkeypatch):
    from gateway.config import GatewayConfig
    from hermes_cli import gateway_migrate
    from hermes_cli.gateway_multiplex_mode import resolve_multiplex_mode
    _, secondary = homes
    (secondary / 'gateway.parked').touch()
    monkeypatch.setattr(gateway_migrate, '_host_supports_migration', lambda: None)
    monkeypatch.setattr(gateway_migrate, 'build_migration_plan', lambda: SimpleNamespace(
        standalone_secondaries=[], blocked=False))
    config = GatewayConfig()
    assert resolve_multiplex_mode(config).enabled


def test_dashboard_exposes_parked_profile_and_start_unparks_it(homes, monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from gateway import host_attach
    from hermes_cli import web_server, profiles
    from hermes_cli.web_server_gateway import multiplexed_profile_refusal
    root, secondary = homes
    (secondary / 'gateway.parked').touch()
    monkeypatch.setattr(profiles, '_check_gateway_running', lambda home: False)
    with TestClient(web_server.app) as client:
        response = client.get('/api/status')
    assert response.status_code == 200
    assert response.json()['parked_profiles'] == ['worker']
    # The Start button: refused while no host can unpark it, allowed (spawns `-p worker gateway start`)
    # once the host multiplexer is live; a parked profile is not served, so this is not the served path.
    monkeypatch.setattr(host_attach, 'host_gateway', lambda: None)
    assert multiplexed_profile_refusal('worker', 'start')
    monkeypatch.setattr(host_attach, 'host_gateway', lambda: SimpleNamespace(home=root, pid=1))
    assert multiplexed_profile_refusal('worker', 'start') is None


@pytest.mark.parametrize('host_running', [False, True])
def test_start_unparks_without_host_rendezvous(homes, monkeypatch, capsys, host_running):
    from hermes_cli import gateway as gw, gateway_multiplex_served as served
    from gateway import control_socket
    root, secondary = homes
    marker = secondary / 'gateway.parked'
    marker.touch()
    monkeypatch.setenv('HERMES_HOME', str(secondary))
    monkeypatch.setattr(gw, '_current_profile_name', lambda: 'worker')
    monkeypatch.setattr(gw, '_host_multiplexer_for_all_verb', lambda: None)
    monkeypatch.setattr(served, 'live_default_gateway_pid', lambda: 42 if host_running else None)
    calls = []

    def serve(home, name):
        assert home == root and name == 'worker' and not marker.exists()
        calls.append('serve')
        return {'served': name}

    def normal_start(**kw):
        assert not marker.exists()
        calls.append('normal')

    monkeypatch.setattr(control_socket, 'request_serve_profile_hot', serve)
    monkeypatch.setattr(gw, '_guard_named_profile_under_multiplexer', normal_start)
    monkeypatch.setattr(gw, '_dispatch_via_service_manager_if_s6', lambda *a: True)
    gw._cmd_start(SimpleNamespace())
    assert not marker.exists()
    assert calls == (['serve'] if host_running else ['normal'])


def test_default_status_distinguishes_served_and_parked(homes, monkeypatch, capsys):
    from hermes_cli import gateway as gw
    from hermes_cli.gateway_profile_lifecycle import print_parked_status
    root, secondary = homes
    (secondary / 'gateway.parked').touch()
    monkeypatch.setattr(gw, '_current_profile_name', lambda: 'default')
    monkeypatch.setattr(gw, 'host_multiplexer_serving', lambda: SimpleNamespace(
        home=root, profiles=('default',)))
    assert print_parked_status() is False
    output = capsys.readouterr().out
    assert 'Served profiles: default' in output
    assert "Profile 'worker': parked" in output
