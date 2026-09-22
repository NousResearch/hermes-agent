from unittest.mock import Mock
import pytest

@pytest.mark.linux_only
def test_cold_stopped_service_waits_until_explicit_start(monkeypatch):
    from scripts.sprites import runtime
    marker = Mock()
    marker.exists.side_effect = [True, True, False]
    monkeypatch.setattr(runtime, 'Path', lambda path: marker)
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
