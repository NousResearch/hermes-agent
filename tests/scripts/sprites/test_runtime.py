from unittest.mock import Mock
import pytest

@pytest.mark.linux_only
def test_cold_stopped_service_waits_until_explicit_start(monkeypatch):
    from scripts.sprites import runtime
    marker = Mock()
    marker.exists.side_effect = [True, True, False]
    monkeypatch.setattr(runtime, 'STOPPED', marker)
    sleep = Mock()
    monkeypatch.setattr(runtime.time, 'sleep', sleep)
    monkeypatch.setattr(runtime, 'environment', lambda: {})
    monkeypatch.setattr(runtime.pwd, 'getpwnam', lambda name: Mock(pw_name=name, pw_uid=123, pw_gid=456))
    for method in ('initgroups', 'setgid', 'setuid', 'chdir'):
        monkeypatch.setattr(runtime.os, method, Mock())
    launch = Mock()
    monkeypatch.setattr(runtime.os, 'execve', launch)
    runtime.run('dashboard')
    assert sleep.call_count == 2
    launch.assert_called_once()


@pytest.mark.linux_only
def test_cold_profile_waits_for_its_own_start(tmp_path, monkeypatch):
    from scripts.sprites import runtime
    from hermes_cli import sprites_api
    from hermes_cli.sprites_services import SpritesServiceManager
    profile = tmp_path / 'profiles' / 'quiet'
    profile.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(sprites_api, 'request', Mock())
    manager = SpritesServiceManager()
    manager.stop('gateway-quiet')
    monkeypatch.setattr(runtime, 'STOPPED', tmp_path / 'nas-stopped')
    monkeypatch.setattr(runtime, 'environment', lambda: {})
    monkeypatch.setattr(runtime.pwd, 'getpwnam', lambda name: Mock(pw_name=name, pw_uid=123, pw_gid=456))
    for method in ('initgroups', 'setgid', 'setuid', 'chdir'):
        monkeypatch.setattr(runtime.os, method, Mock())
    launch = Mock()
    monkeypatch.setattr(runtime.os, 'execve', launch)
    def start_profile(_seconds):
        launch.assert_not_called()
        manager.start('gateway-quiet')
    sleep = Mock(side_effect=start_profile)
    monkeypatch.setattr(runtime.time, 'sleep', sleep)
    runtime.run('gateway', 'quiet')
    sleep.assert_called_once_with(1)
    assert launch.call_args.args[1][1:3] == ['-p', 'quiet']
