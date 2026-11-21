"""Regression: serialized Python process methods are not shell killer commands.

All commands in this module are scanner input data; none are executed.
"""
import logging
import shlex

import pytest

from cron import lifecycle_guard as guard


@pytest.mark.parametrize("member", ["server.kill()", "os.kill(pid, signal.SIGTERM)", "worker.pkill()", "proc.Kill()"])
@pytest.mark.parametrize("order", ["hermes gateway", "gateway hermes"])
def test_python_member_not_shell_killer(member, order):
    text = member + '; print("' + order + '")'
    assert guard.contains_gateway_lifecycle_command(text) is False
    assert guard.scan_gateway_lifecycle(text) == (False, None)


@pytest.mark.parametrize("verb", ["restart", "stop", "uninstall"])
def test_real_lifecycle_after_member_still_blocked(verb):
    text = 'server.kill(); hermes gateway ' + verb
    assert guard.contains_gateway_lifecycle_command(text) is True
    assert guard.scan_gateway_lifecycle(text) == (True, None)


@pytest.mark.parametrize("command", [
    "pkill -f 'hermes gateway'", "pkill -f 'gateway hermes'",
    "kill $(pgrep -f 'hermes gateway')", "/bin/kill $(pgrep -f 'gateway hermes')",
    "taskkill /F /IM hermes-gateway.exe", "Stop-Process -Name hermes-gateway",
    'pki"ll" -f "hermes gateway"', r"pki\ll -f 'hermes gateway'",
    "sudo pkill -f 'hermes gateway'", "pkill -9 python3", "killall python",
    "systemctl --user restart hermes-gateway", "launchctl bootout gui/501/ai.hermes.gateway",
    'subprocess.run(["pkill", "-f", "hermes gateway"])',
])
def test_real_dangerous_commands_still_blocked(command):
    assert guard.contains_gateway_lifecycle_command(command) is True
    assert guard.scan_gateway_lifecycle(command) == (True, None)


def test_referenced_script_still_blocked(tmp_path):
    script = tmp_path / "maintenance.sh"
    script.write_text("server.kill(); hermes gateway stop\n")
    assert guard.scan_gateway_lifecycle("bash " + shlex.quote(str(script))) == (True, None)


@pytest.mark.parametrize("order", ["hermes gateway", "gateway hermes"])
@pytest.mark.parametrize("shell", [
    "kill $(pgrep -f '{order}')",
    "/usr/bin/pkill -f '{order}'",
    "sudo pki\"ll\" -f '{order}'",
    "Stop-Process -Name {order}",
])
def test_real_killer_after_member_remains_blocked(order, shell, caplog):
    command = "server.kill(); " + shell.format(order=order)
    with caplog.at_level(logging.WARNING, logger=guard.__name__):
        assert guard.contains_gateway_lifecycle_command(command) is True
        assert guard.scan_gateway_lifecycle(command) == (True, None)
    assert "falling back to direct-scan verdict" not in caplog.text


@pytest.mark.parametrize("command", [
    'subprocess.run(["/usr/bin/pkill", "-f", "hermes gateway"])',
    'server.kill(); subprocess.run(["taskkill", "/IM", "hermes-gateway.exe"])',
    'server.kill(); pkill -9 python3',
    'server.kill(); Get-Process python | Stop-Process -Force',
])
def test_nested_and_interpreter_killers_remain_blocked(command, caplog):
    with caplog.at_level(logging.WARNING, logger=guard.__name__):
        assert guard.contains_gateway_lifecycle_command(command) is True
        assert guard.scan_gateway_lifecycle(command) == (True, None)
    assert "falling back to direct-scan verdict" not in caplog.text


def test_serialized_python_source_not_shell_killer(caplog):
    harness = 'server.kill(); server.wait()\nprint("hermes gateway timeout")\n'
    writer = 'from pathlib import Path\np=Path("bench.py")\np.write_text(' + repr(harness) + ')\n'
    command = 'printf %s ' + shlex.quote(writer) + ' | python3 remote.py'
    with caplog.at_level(logging.WARNING, logger=guard.__name__):
        assert guard.contains_gateway_lifecycle_command(command) is False
        assert guard.scan_gateway_lifecycle(command) == (False, None)
    assert "falling back to direct-scan verdict" not in caplog.text
