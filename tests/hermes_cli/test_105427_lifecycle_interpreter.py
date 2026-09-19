"""Regression for #105427: interpreter binary vs script classification."""
from pathlib import Path
from cron import lifecycle_guard as guard


def test_python_directory_literal_is_not_a_shell_command(tmp_path):
    data = tmp_path / "market-data"
    data.mkdir()
    script = tmp_path / "snapshot.py"
    script.write_text("from pathlib import Path\nROOT = Path(" + repr(data.as_posix()) + ")\n")
    source = script.read_text()
    assert not guard.contains_gateway_lifecycle_command(source)
    assert not guard.contains_launchctl_submit_command(source)
    guard.check_gateway_lifecycle("Read-only public snapshot.", str(script))


def test_large_absolute_interpreter_is_not_a_shell_script(tmp_path):
    interp = tmp_path / "python3"
    interp.write_bytes(b"\x00" * (guard._MAX_REFERENCED_SCRIPT_BYTES + 1))
    script = tmp_path / "snapshot.py"
    script.write_text('print("snapshot")\n')
    cmd = interp.as_posix() + " " + script.as_posix()
    assert not guard.contains_gateway_lifecycle_command_or_referenced_script(cmd, cwd=str(tmp_path))


def test_quoted_absolute_interpreter_is_not_a_shell_script(tmp_path):
    interp = tmp_path / "python3"
    interp.write_bytes(b"\x00" * (guard._MAX_REFERENCED_SCRIPT_BYTES + 1))
    script = tmp_path / "snapshot.py"
    script.write_text('print("snapshot")\n')
    cmd = '"' + interp.as_posix() + '" "' + script.as_posix() + '"'
    assert not guard.contains_gateway_lifecycle_command_or_referenced_script(cmd, cwd=str(tmp_path))


def test_wrapped_interpreter_is_not_a_shell_script(tmp_path):
    interp = tmp_path / "python3"
    interp.write_bytes(b"\x00" * (guard._MAX_REFERENCED_SCRIPT_BYTES + 1))
    script = tmp_path / "snapshot.py"
    script.write_text('print("snapshot")\n')
    cmd = "sudo " + interp.as_posix() + " " + script.as_posix()
    assert not guard.contains_gateway_lifecycle_command_or_referenced_script(cmd, cwd=str(tmp_path))


def test_real_direct_lifecycle_command_stays_blocked():
    try:
        guard.check_gateway_lifecycle("hermes gateway restart")
    except guard.GatewayLifecycleBlocked:
        return
    raise AssertionError("direct lifecycle not blocked")


def test_referenced_shell_restart_stays_blocked(tmp_path):
    script = tmp_path / "restart.sh"
    script.write_text("#!/bin/sh\nhermes gateway stop\n")
    assert guard.contains_gateway_lifecycle_command_or_referenced_script(
        "sh " + script.as_posix(), cwd=str(tmp_path))


def test_actual_oversized_shell_script_stays_fail_closed(tmp_path):
    script = tmp_path / "large.sh"
    script.write_bytes(b"#" * (guard._MAX_REFERENCED_SCRIPT_BYTES + 1))
    assert guard.contains_gateway_lifecycle_command_or_referenced_script(
        "sh " + script.as_posix(), cwd=str(tmp_path))


def test_oversized_file_named_python_as_shell_arg_stays_fail_closed(tmp_path):
    # `sh python3-oversized`: explicitly supplied AS the shell script -> still blocked.
    # Proves the fix is positional, not a name exemption.
    payload = tmp_path / "python3"
    payload.write_bytes(b"#" * (guard._MAX_REFERENCED_SCRIPT_BYTES + 1))
    assert guard.contains_gateway_lifecycle_command_or_referenced_script(
        "sh " + payload.as_posix(), cwd=str(tmp_path))


def test_python_script_arg_with_lifecycle_stays_blocked(tmp_path):
    # sabotage: interpreter is benign, script payload is evil -> must still block.
    interp = tmp_path / "python3"
    interp.write_bytes(b"\x7fELF" + b"\x00" * 64)
    evil = tmp_path / "evil.py"
    evil.write_text('import os\nos.system("hermes gateway restart")\n')
    cmd = interp.as_posix() + " " + evil.as_posix()
    assert guard.contains_gateway_lifecycle_command_or_referenced_script(cmd, cwd=str(tmp_path))


def test_python_c_payload_with_lifecycle_stays_blocked(tmp_path):
    interp = tmp_path / "python3"
    interp.write_bytes(b"\x7fELF" + b"\x00" * 64)
    cmd = interp.as_posix() + ' -c "hermes gateway restart"'
    assert guard.contains_gateway_lifecycle_command_or_referenced_script(cmd, cwd=str(tmp_path))
