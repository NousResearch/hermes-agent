"""Effective SSH configuration must not turn a remote exemption into a self restart.

Only guard functions run: no connection, command payload, Match exec or restart is executed.
"""
from pathlib import Path

import pytest

from cron import lifecycle_guard as guard

REMOTE = "192.168.3.184"
LOCAL = "192.168.9.9"


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".ssh").mkdir()
    monkeypatch.setattr(guard, "_local_host_identities", lambda: frozenset({LOCAL, "localhost", "::1"}))
    # This test's temporary destination never traverses the configured remote alias.
    monkeypatch.setattr(guard, "_iter_referenced_shell_scripts", lambda command, cwd=None: iter(()))
    # Test the system-config boundary using a real file, not the operator's config.
    monkeypatch.setattr(guard, "_SSH_SYSTEM_CONFIG", tmp_path / "system_config", raising=False)
    return tmp_path


@pytest.mark.parametrize("config,destination,blocked", [
    ("", REMOTE, False),
    (f"Host peer\n HostName {REMOTE}\n User user\n", "peer", False),
    (f"Host *\n HostName {LOCAL}\n", REMOTE, True),
    (f"Host *\n HostName {LOCAL}\nHost peer\n HostName {REMOTE}\n", "peer", True),
    (f"Host peer\n HostName {LOCAL}\n HostName {REMOTE}\n", "peer", True),
    (f"Host peer\n HostName {REMOTE}\nHost *\n ProxyCommand nc localhost 22\n", "peer", True),
    (f"Host peer\n HostName {REMOTE}\n ProxyJump localhost\n", "peer", True),
    (f"Host peer\n HostName {REMOTE}\n LocalCommand hermes gateway restart\n PermitLocalCommand yes\n", "peer", True),
    (f"Host peer\n HostName {REMOTE}\n KnownHostsCommand hermes gateway restart\n", "peer", True),
    (f"Host peer\n HostName {REMOTE}\n ControlPath /tmp/shared-control\n", "peer", True),
    (f"Host peer\n HostName {REMOTE}\nMatch all\n HostName {LOCAL}\n", "peer", True),
    (f"Include other\nHost peer\n HostName {REMOTE}\n", "peer", True),
    (f"Host peer\n HostName {REMOTE}\n CanonicalizeHostname yes\n", "peer", True),
    (f"Host peer !peer\n HostName {REMOTE}\n", "peer", True),
    (f"Host peer\n HostName next\nHost next\n HostName {REMOTE}\n", "peer", True),
    (f'Host peer\n "HostName" "{LOCAL}"\n HostName {REMOTE}\n', "peer", True),
    (f"Host peer\n HostName={REMOTE}\n", "peer", False),
    (f"Host peer* !peer-local\n HostName {REMOTE}\n", "peer-ok", False),
])
def test_effective_config_at_public_guard(config_home, config, destination, blocked):
    (config_home / ".ssh/config").write_text(config)
    # Keep each row isolated from the operator's real system policy in this test.
    (config_home / "system_config").write_text("")
    hostname = guard._ssh_config_hostname(destination)
    if destination == "peer" and not blocked:
        assert hostname is not None
    if not blocked:
        assert guard._ssh_destination_is_remote(destination), (
            f"unexpected config veto: {guard._ssh_config_hostname(destination)!r}"
        )
    command = f"ssh {destination} 'hermes gateway restart'"
    masked = guard._mask_remote_ssh_command_payloads(command)
    if not blocked:
        assert masked != command
    assert guard._lifecycle_command_scan_with_data_exemption(command) is blocked
    assert guard.contains_gateway_lifecycle_command_or_referenced_script(command) is blocked
    # System policy participates for both aliases and literal destinations.
    (config_home / "system_config").write_text("Host *\n ProxyCommand nc localhost 22\n")
    assert guard.contains_gateway_lifecycle_command_or_referenced_script(command)


@pytest.mark.parametrize("command", [
    'ssh 192.168.3.184 "$(hermes gateway restart)"',
    'ssh 192.168.3.184 "`hermes gateway restart`"',
    'ssh 192.168.3.184 "echo hermes gateway restart" | sh',
    "sudo ssh 192.168.3.184 'hermes gateway restart'",
    "HOME=/other ssh 192.168.3.184 'hermes gateway restart'",
    "ssh -S /tmp/shared 192.168.3.184 'hermes gateway restart'",
    "ssh -o 'Host\"Name\"=localhost' 192.168.3.184 'hermes gateway restart'",
])
def test_ambiguous_local_execution_stays_scanned(config_home, command):
    assert guard.contains_gateway_lifecycle_command_or_referenced_script(command)
