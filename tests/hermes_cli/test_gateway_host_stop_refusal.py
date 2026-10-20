"""The host multiplexer is never the pid a served profile's ``gateway stop`` may SIGTERM (#141133).

While the host serves a profile it publishes its own pid into that profile's runtime record, so
``get_running_pid()`` under ``-p <name>`` names the host. ``owned_gateway_pid`` (the single gate,
read by ``find_gateway_pids`` and ``stop_profile_gateway``) answers only on positive ownership
evidence; these cases drive it through the real host-record and launch-record readers.
"""
import json
import os
from types import SimpleNamespace

import pytest

from gateway import host_attach, host_rendezvous as hr, status
from hermes_cli import dashboard_procs, gateway as gw

HOST_PID = 4242


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
    monkeypatch.setattr(gw, '_get_service_pids', lambda **kw: [])
    monkeypatch.setattr(gw, '_scan_gateway_pids', lambda *a, **kw: [])
    return root, secondary


def _live_host(monkeypatch, home, *, socket_answers=False):
    """A live host record launched from *home*; its control socket answers only when asked to."""
    record = SimpleNamespace(pid=HOST_PID, home=str(home), role=hr.ROLE_GATEWAY)
    monkeypatch.setattr(hr, 'read_record', lambda role, **kw: record)
    monkeypatch.setattr(hr, 'record_names_live_owner', lambda rec: True)
    dials = []

    def probe(wait):
        dials.append(wait)
        return host_attach.HostGateway(HOST_PID, home, ('default', 'worker'), served_known=socket_answers)

    monkeypatch.setattr(host_attach, '_probe_host_gateway', probe)
    monkeypatch.setattr(status, 'get_running_pid', lambda **kw: HOST_PID)
    return dials


def test_stop_refuses_the_serving_host_when_the_socket_does_not_answer(homes, monkeypatch, capsys):
    root, secondary = homes
    _live_host(monkeypatch, root)
    monkeypatch.setenv('HERMES_HOME', str(secondary))
    monkeypatch.setattr(gw, '_mark_planned_stop', lambda *a: pytest.fail('planned-stop marker written for the host pid'))
    monkeypatch.setattr(gw.os, 'kill', lambda *a: pytest.fail('stop_profile_gateway SIGTERMed the host pid'))
    monkeypatch.setattr(gw, '_reap_unsupervised_gateway_orphans', lambda **kw: pytest.fail('host pid reached the reaper'))

    assert gw.find_gateway_pids() == []
    assert gw.stop_profile_gateway() is False
    # uninstall's per-profile kill reads the same gate.
    monkeypatch.setattr(gw, 'terminate_pid', lambda *a, **kw: pytest.fail('uninstall kill reached the host pid'))
    assert gw.kill_gateway_processes() == 0
    from hermes_cli.gateway_profile_lifecycle import report_profile_stop_failure
    with pytest.raises(SystemExit) as exc:
        report_profile_stop_failure()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert 'host gateway serving this profile' in out
    assert 'gateway stop --all' in out


def test_stop_from_the_hosts_own_home_still_stops_it(homes, monkeypatch):
    """The owner case is unchanged: the home that launched the host stops its own gateway."""
    root, _secondary = homes
    _live_host(monkeypatch, root, socket_answers=True)
    killed = []
    monkeypatch.setattr(gw, '_mark_planned_stop', lambda *a: None)
    monkeypatch.setattr(gw.os, 'kill', lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(status, 'remove_pid_file', lambda: None)
    monkeypatch.setattr(status, '_pid_exists', lambda pid: False)
    monkeypatch.setattr(gw, '_reap_unsupervised_gateway_orphans', lambda **kw: [])

    assert gw.stop_profile_gateway() is True
    assert killed == [HOST_PID]


def test_cmd_stop_guard_names_the_host_when_the_lock_carries_its_pid(homes, monkeypatch, capsys):
    """``--force`` skips the park path; ``_cmd_stop``'s "the host serves this profile" refusal still
    fires, because ``find_gateway_pids()`` no longer reports the host pid as this profile's gateway."""
    root, secondary = homes
    _live_host(monkeypatch, root)
    host = SimpleNamespace(home=root, pid=HOST_PID, profile_label='default',
                           profiles=('default', 'worker'), describe=lambda: 'test host')
    monkeypatch.setenv('HERMES_HOME', str(secondary))
    monkeypatch.setattr(gw, '_refuse_from_inside_gateway', lambda *a: None)
    monkeypatch.setattr(gw, '_current_profile_name', lambda: 'worker')
    monkeypatch.setattr(gw, '_served_by_another_host_gateway', lambda *a: host)
    monkeypatch.setattr(gw, 'stop_profile_gateway', lambda: pytest.fail('stop reached the host pid'))
    monkeypatch.setattr(gw, '_stop_installed_service', lambda *a: pytest.fail('stop reached the service'))

    with pytest.raises(SystemExit) as exc:
        gw._cmd_stop(SimpleNamespace(force=True))
    assert exc.value.code == gw.GATEWAY_FATAL_CONFIG_EXIT_CODE
    assert "The host gateway serves profile 'worker'" in capsys.readouterr().out


def test_sticky_profile_force_gateway_stays_stoppable(homes, monkeypatch):
    """A ``gateway run --force`` launched without ``-p`` through the sticky profile carries no launch
    identity in argv or env; its own ``gateway.pid`` (pid, start time, home) is the proof."""
    import subprocess
    import sys
    root, secondary = homes
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        pid = child.pid
        (root / 'active_profile').write_text('default\n')  # the sticky choice moved on after launch
        (secondary / 'gateway.pid').write_text(json.dumps({
            'pid': pid, 'kind': status._GATEWAY_KIND, 'argv': ['hermes', 'gateway', 'run', '--force'],
            'start_time': status._get_process_start_time(pid), 'hermes_home': str(secondary)}))
        monkeypatch.setenv('HERMES_HOME', str(secondary))
        monkeypatch.setattr(hr, 'read_record', lambda role, **kw: None)
        monkeypatch.setattr(dashboard_procs, '_pid_environ',
                            lambda p: {'HERMES_HOME': str(root), 'HOME': str(root.parent)})
        monkeypatch.setattr(status, 'get_running_pid', lambda **kw: pid)
        assert dashboard_procs._hermes_home_for_pid(pid, require_launch_identity=True) is None
        assert gw._owned_gateway_pid() == pid
        assert gw.find_gateway_pids() == [pid]
        # A recycled pid (another start time) is not that launch.
        monkeypatch.setattr(status, '_get_process_start_time', lambda p: -1)
        assert gw._owned_gateway_pid() is None
    finally:
        child.kill()
        child.wait(timeout=10)


def test_only_the_hosts_pid_dials_its_control_socket(homes, monkeypatch):
    """``gateway status`` across N profiles calls ``find_gateway_pids`` per profile: a pid that is not
    the live host's never dials, and the host's pid from another home is refused on the record alone."""
    root, secondary = homes
    dials = _live_host(monkeypatch, root, socket_answers=True)
    monkeypatch.setenv('HERMES_HOME', str(secondary))
    monkeypatch.setattr(status, 'get_running_pid', lambda **kw: 5151)
    monkeypatch.setattr(dashboard_procs, '_hermes_home_for_pid', lambda pid, **kw: str(secondary))
    assert gw.find_gateway_pids() == [5151]
    monkeypatch.setattr(status, 'get_running_pid', lambda **kw: HOST_PID)
    assert gw.find_gateway_pids() == []
    assert dials == []
