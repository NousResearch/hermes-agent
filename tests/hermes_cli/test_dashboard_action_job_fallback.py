"""Windows dashboard actions tolerate scheduled-task jobs that forbid breakaway.

Regression for #135179: a dashboard action spawned from a Task Scheduler job dies with
PermissionError [WinError 5] because CREATE_BREAKAWAY_FROM_JOB is denied in a job object
without breakaway rights. The documented fallback is
``windows_detach_flags_without_breakaway()``.
"""
from unittest.mock import Mock
import pytest

pytestmark = pytest.mark.platforms("windows")

@pytest.mark.parametrize("winerror,retries", [(5, True), (2, False)])
def test_action_spawn_permission_failure(monkeypatch, tmp_path, winerror, retries):
    from hermes_cli import web_server_gateway as gateway
    from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway
    monkeypatch.setattr(gateway, "_ACTION_LOG_DIR", tmp_path)
    monkeypatch.setattr(gateway, "_ACTION_PROCS", {})
    monkeypatch.setattr(gateway, "_ACTION_COMMANDS", {})
    monkeypatch.setattr(gateway, "_ACTION_RESULTS", {})
    error = PermissionError("spawn denied")
    error.winerror = winerror
    proc = Mock(pid=123)
    spawn = Mock(side_effect=[error, proc])
    monkeypatch.setattr(gateway.subprocess, "Popen", spawn)
    if retries:
        assert gateway._spawn_hermes_action(["gateway", "status"], "gateway-restart") is proc
        assert spawn.call_count == 2
        first, second = spawn.call_args_list
        assert first.args == second.args
        assert second.kwargs["creationflags"] == windows_detach_flags_without_breakaway()
        assert first.kwargs["env"] == second.kwargs["env"]
    else:
        with pytest.raises(PermissionError):
            gateway._spawn_hermes_action(["gateway", "status"], "gateway-restart")
        assert spawn.call_count == 1
    assert spawn.call_args.kwargs["stdout"].closed
