"""Regression tests for the conftest live-system guard's argv handling.

The guard must treat only argv[0] of a list/tuple command as the executable
(arguments are data: a file named ``skill`` is not the ``skill`` binary),
while still scanning every token of wrapper invocations like ``bash -c``.
All blocked-case commands use patterns that match no real process, so a
guard regression cannot kill anything.
"""

import subprocess

import pytest


def test_argv_arguments_are_not_treated_as_executables(tmp_path):
    """A file argument whose basename is a killer name must not trip the
    guard (the path contains "hermes" via the pytest tmp root)."""
    target = tmp_path / "skill"
    target.write_text("just a filename\n")
    result = subprocess.run(["cat", str(target)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "just a filename" in result.stdout


def test_direct_killer_argv_is_still_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["pkill", "-f", "hermes-guard-regression-nomatch"])


def test_wrapped_killer_command_is_still_blocked():
    """argv[0]-only scanning must not exempt commands hidden behind a
    shell wrapper."""
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["bash", "-c", "pkill -f hermes-guard-regression-nomatch"])


def test_env_wrapped_killer_command_is_still_blocked():
    with pytest.raises(RuntimeError, match="live-system guard"):
        subprocess.run(["env", "GUARD_TEST=1", "pkill", "-f", "hermes-guard-regression-nomatch"])


def test_launchctl_kickstart_of_hermes_gateway_is_blocked():
    with pytest.raises(RuntimeError, match="host service control"):
        subprocess.run(["launchctl", "kickstart", "-k", "gui/501/ai.hermes.gateway"])


def test_launchctl_bootout_of_profile_gateway_is_blocked():
    with pytest.raises(RuntimeError, match="host service control"):
        subprocess.run(
            ["launchctl", "bootout", "gui/501/ai.hermes.gateway-ops"],
            capture_output=True,
        )


def test_shell_wrapped_launchctl_kickstart_is_blocked():
    with pytest.raises(RuntimeError, match="guard"):
        subprocess.run(
            ["bash", "-c", "launchctl kickstart -k gui/501/ai.hermes.gateway"]
        )


def test_launchctl_read_only_contact_is_blocked_too():
    with pytest.raises(RuntimeError, match="host service control"):
        subprocess.run(
            ["launchctl", "print", "gui/501/ai.hermes.gateway"],
            capture_output=True,
            text=True,
            check=False,
        )


def test_launchctl_mutation_of_non_hermes_label_is_blocked():
    with pytest.raises(RuntimeError, match="host service control"):
        subprocess.run(["launchctl", "bootout", "gui/501/com.example.nomatch"])


def test_systemctl_contact_for_non_hermes_unit_is_blocked():
    with pytest.raises(RuntimeError, match="host service control"):
        subprocess.run(["systemctl", "status", "com.example.nomatch"])
