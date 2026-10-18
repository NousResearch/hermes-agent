"""The host multiplexer is never the pid a served profile's ``gateway stop`` may SIGTERM (#141133).

First blind spot: ``profile_lifecycle`` saw the host pid in the profile's runtime record and stood
down. Second one, covered here: the host probe or the control socket does not answer, so
``profile_lifecycle`` falls through and ``_cmd_stop``'s own guard does not fire either (it only
fires while ``find_gateway_pids()`` is empty, and that record names the host). ``stop_profile_gateway``
is the last place before the kill, so the refusal lives on the pid it is about to signal.
"""
from types import SimpleNamespace

import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    from hermes_cli import profiles
    root = tmp_path / '.hermes'
    secondary = root / 'profiles' / 'worker'
    secondary.mkdir(parents=True)
    (secondary / 'config.yaml').write_text('model: {default: worker-model}\n')
    (root / 'config.yaml').write_text('gateway: {multiplex_profiles: true}\n')
    monkeypatch.setattr('pathlib.Path.home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(profiles, 'get_active_profile_name', lambda: 'default')
    return root, secondary


def _host_owner(root, pid=4242):
    return SimpleNamespace(home=root, pid=pid, profile_label='default',
                           profiles=('default', 'worker'), describe=lambda: 'test host')


def test_stop_refuses_the_serving_host_when_the_socket_does_not_answer(homes, monkeypatch, capsys):
    from hermes_cli import gateway as gw
    root, secondary = homes
    host = _host_owner(root)
    monkeypatch.setenv('HERMES_HOME', str(secondary))
    monkeypatch.setattr(gw, '_host_multiplexer_for_all_verb', lambda: host)
    monkeypatch.setattr(gw, '_host_multiplexer_is_ours', lambda owner: False)
    monkeypatch.setattr(gw, '_mark_planned_stop',
                        lambda *a: pytest.fail('planned-stop marker written for the host pid'))
    monkeypatch.setattr(gw.os, 'kill', lambda *a: pytest.fail('stop_profile_gateway SIGTERMed the host pid'))
    monkeypatch.setattr('gateway.status.get_running_pid', lambda **kw: host.pid)

    assert gw.stop_profile_gateway() is False
    out = capsys.readouterr().out
    assert 'host gateway serving this profile' in out
    assert 'gateway stop --all' in out


def test_stop_from_the_hosts_own_home_still_stops_it(homes, monkeypatch):
    """The owner case is unchanged: the home that launched the host stops its own gateway."""
    from hermes_cli import gateway as gw
    root, secondary = homes
    host = _host_owner(root)
    killed = []
    monkeypatch.setenv('HERMES_HOME', str(secondary))
    monkeypatch.setattr(gw, '_host_multiplexer_for_all_verb', lambda: host)
    monkeypatch.setattr(gw, '_host_multiplexer_is_ours', lambda owner: True)
    monkeypatch.setattr(gw, '_mark_planned_stop', lambda *a: None)
    monkeypatch.setattr(gw.os, 'kill', lambda pid, sig: killed.append(pid))
    monkeypatch.setattr('gateway.status.get_running_pid', lambda **kw: host.pid)
    monkeypatch.setattr('gateway.status.remove_pid_file', lambda: None)
    monkeypatch.setattr('gateway.status._pid_exists', lambda pid: False)
    monkeypatch.setattr(gw, '_reap_unsupervised_gateway_orphans', lambda **kw: [])

    assert gw.stop_profile_gateway() is True
    assert killed == [host.pid]
