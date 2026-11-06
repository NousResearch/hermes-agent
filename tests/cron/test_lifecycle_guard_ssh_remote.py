"""Lifecycle guard: a lifecycle command run on a REMOTE host over ssh is not blocked.

``ssh <host> <command>`` runs <command> under the remote sshd; it cannot SIGTERM this gateway, so
restarting another machine's gateway is legitimate maintenance. Loopback targets, options that
execute text locally, local command substitution and later local segments stay blocked.
"""
import pytest

from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script as blocked

CMD = "hermes gateway restart"


@pytest.mark.parametrize("command", [
    'ssh box "systemctl restart hermes-gateway"',
    f"ssh box '{CMD}'",
    f"ssh -p 2222 user@box {CMD}",
    "/usr/bin/ssh -i ~/.ssh/key box 'launchctl kickstart -k gui/501/ai.hermes.gateway'",
    f"autossh -M 0 box {CMD}",
    f"ssh -o BatchMode=yes box {CMD}",
])
def test_remote_lifecycle_command_is_allowed(command):
    assert not blocked(command), command


@pytest.mark.parametrize("command", [
    f"ssh localhost {CMD}",
    f"ssh localhost.localdomain {CMD}",
    f"ssh 127.0.0.1 {CMD}",
    f"ssh user@127.0.0.2 {CMD}",
    f"ssh ::1 {CMD}",
    f"ssh [::1] {CMD}",
    f"ssh user@[::1] {CMD}",
    f"ssh ::ffff:127.0.0.1 {CMD}",
    f"ssh 0.0.0.0 {CMD}",
    f"REMOTE=/usr/bin/ssh; $REMOTE box {CMD}",
    f"ssh box true && {CMD}",
    f"ssh box true; {CMD}",
    f'ssh box "$({CMD})"',
    f"ssh box `{CMD}`",
    f"ssh -o ProxyCommand='{CMD}' box true",
    f"ssh -oLocalCommand=x box {CMD}",
    f"ssh -o PermitLocalCommand=yes box {CMD}",
])
def test_local_effect_stays_blocked(command):
    assert blocked(command), command


# Numeric spellings the C resolver maps to loopback/unspecified but a text regex misses
# (reported on upstream #123701). ssh resolves the host the same way, so these reach this machine.
@pytest.mark.parametrize("host", [
    "127.1",
    "2130706433",
    "127.0.2",
    "0x7f000001",
    "017700000001",
    "0",
    "::",
    "[::]",
    "::ffff:7f00:1",
    "[::ffff:7f00:1]",
    "0:0:0:0:0:ffff:7f00:1",
    "::ffff:0.0.0.0",
    "127.0.0.1.",
    "0x7f.1",
    "::127.0.0.1",
    "LOCALHOST.",
    "4294967296",
])
def test_numeric_loopback_spellings_stay_blocked(host):
    assert blocked(f"ssh {host} {CMD}"), host
    assert blocked(f"ssh user@{host} {CMD}"), host


@pytest.mark.parametrize("host", ["999.1", "1.2.3.4.5", "::ffff:7f00:1:x", "box$x"])
def test_unparseable_numeric_host_fails_closed(host):
    assert blocked(f"ssh '{host}' {CMD}"), host


@pytest.mark.parametrize("host", [
    "box", "build-01.example.com", "0xbeef.example", "10.0.0.5", "192.168.1.20", "fe80::1", "[2001:db8::7]",
])
def test_real_remote_host_still_allowed(host):
    assert not blocked(f"ssh {host} {CMD}"), host
