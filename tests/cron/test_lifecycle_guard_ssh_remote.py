"""The lifecycle guard treats an ssh remote command aimed at ANOTHER machine as remote.

A fleet where one box's agent restarts a sibling box's gateway over ssh
(``ssh linux-lan 'hermes gateway restart'``) cannot SIGTERM the gateway running the
guard, so blocking it is a false positive. The local destination forms (loopback,
this machine's own name/address, an ssh alias whose effective ``HostName`` is local)
and every shape whose destination cannot be determined stay blocked.
"""

from __future__ import annotations

import socket
import subprocess

import pytest

from cron import lifecycle_guard as guard

# Effective hostnames the fake `ssh -G` reports for ~/.ssh/config aliases.
_SSH_CONFIG = {
    "linux-lan": "192.0.2.10",
    "windows-lan": "192.0.2.20",
    "loop-alias": "localhost",
    "self-alias": "127.0.0.1",
}


@pytest.fixture
def resolver(monkeypatch):
    """Replace the `ssh -G` resolution with a fake config; records every lookup."""
    calls: list[tuple[str, tuple]] = []

    def fake(destination: str, options) -> str | None:
        calls.append((destination, tuple(options)))
        host = guard._ssh_host_from_destination(destination) or destination
        return _SSH_CONFIG.get(host, host)

    monkeypatch.setattr(guard, "_ssh_effective_hostname", fake)
    # Effective hostnames that are names resolve to a TEST-NET address (no real DNS in tests).
    monkeypatch.setattr(guard, "_resolve_host_addresses", lambda name: ["192.0.2.99"])
    return calls


def blocked(command: str) -> bool:
    return guard.contains_gateway_lifecycle_command(command)


def scan_blocked(command: str, **kwargs) -> bool:
    return guard.scan_gateway_lifecycle(command, **kwargs)[0]


REMOTE_ALLOWED = [
    "ssh linux-lan 'hermes gateway restart'",
    "ssh linux-lan 'XDG_RUNTIME_DIR=/run/user/1000 systemctl --user restart hermes-gateway'",
    "ssh windows-lan 'hermes gateway restart'",
    "ssh clowlinux@198.51.100.7 -p 2222 'hermes gateway restart'",
    "ssh -p 2222 clowlinux@198.51.100.7 'hermes gateway restart'",
    "ssh -p2222 -l clowlinux 198.51.100.7 hermes gateway restart",
    "ssh linux-lan 'hermes gateway stop'",
    "ssh windows-lan 'schtasks /End /TN Hermes_Gateway & schtasks /Run /TN Hermes_Gateway'",
    "ssh -tt linux-lan 'hermes gateway restart'",
    "ssh -o ConnectTimeout=5 -o BatchMode=yes linux-lan 'hermes gateway restart'",
    "ssh ssh://admin@linux-lan:2222 'hermes gateway restart'",
    "ssh linux-lan -- hermes gateway restart",
    "sudo ssh linux-lan 'hermes gateway restart'",
    "env FOO=1 ssh linux-lan 'hermes gateway restart'",
    "timeout 30 ssh linux-lan 'hermes gateway restart'",
    "ssh mac-mini launchctl kickstart -k gui/501/ai.hermes.gateway",
    "ssh windows-lan 'taskkill /F /IM python.exe'",
    "ssh linux-lan 'hermes gateway restart' 2>&1 | tail -5",
    "ssh linux-lan 'hermes gateway restart' && echo done",
]

LOCAL_BLOCKED = [
    # Own-gateway protection: unchanged.
    "hermes gateway restart",
    "launchctl kickstart -k gui/501/ai.hermes.gateway",
    # Loopback / unspecified destinations.
    "ssh localhost 'hermes gateway restart'",
    "ssh LOCALHOST.localdomain 'hermes gateway restart'",
    "ssh 127.0.0.1 'hermes gateway restart'",
    "ssh me@127.4.5.6 -p 22 'hermes gateway restart'",
    "ssh ::1 'hermes gateway restart'",
    "ssh 0.0.0.0 'hermes gateway restart'",
    "ssh ssh://localhost:22 'hermes gateway restart'",
    "ssh 'ssh://user@[::1]:2222' 'hermes gateway restart'",
    # Aliases whose effective HostName is local, and explicit HostName overrides.
    "ssh loop-alias 'hermes gateway restart'",
    "ssh self-alias 'hermes gateway restart'",
    "ssh -o HostName=localhost linux-lan 'hermes gateway restart'",
    "ssh -oHostname=127.0.0.1 linux-lan 'hermes gateway restart'",
    "ssh -o 'HostName ::1' linux-lan 'hermes gateway restart'",
    # Mixed: the local segment still runs here.
    "ssh linux-lan 'ls'; hermes gateway restart",
    "ssh linux-lan ls && launchctl kickstart -k gui/501/ai.hermes.gateway",
    "ssh linux-lan 'hermes gateway restart'; ssh localhost 'hermes gateway restart'",
    # Destination cannot be determined: fail closed.
    'ssh "$HOST" \'hermes gateway restart\'',
    "ssh -Z linux-lan 'hermes gateway restart'",
    "for h in a b; do ssh $h 'hermes gateway restart'; done",
    # ssh option values that execute locally are not remote payload.
    "ssh -o 'ProxyCommand=hermes gateway restart' linux-lan ls",
    # Remote output fed back into local execution.
    "ssh linux-lan echo hermes gateway restart | bash",
    "ssh linux-lan echo hermes gateway restart 2>&1 | sh",
    "launchctl kickstart $(ssh linux-lan echo ai.hermes.gateway)",
    # shlex/bash divergence: bash runs the tail locally, shlex would fold it into the ssh argv.
    "ssh linux-lan x#; hermes gateway restart",
    "ssh linux-lan echo $'\\''; hermes gateway restart #'",
]


@pytest.mark.parametrize("command", REMOTE_ALLOWED)
def test_non_local_ssh_payload_is_not_blocked(resolver, command):
    assert blocked(command) is False
    assert scan_blocked(command) is False


@pytest.mark.parametrize("command", LOCAL_BLOCKED)
def test_local_or_undeterminable_destination_stays_blocked(resolver, command):
    assert blocked(command) is True
    assert scan_blocked(command) is True


def test_own_hostname_is_local(resolver):
    own = socket.gethostname()
    short = own.split(".")[0]
    for host in {own, own.upper(), short, f"{short}.example.test"}:
        assert blocked(f"ssh {host} 'hermes gateway restart'") is True, host


def test_alias_resolving_to_own_hostname_is_local(monkeypatch):
    monkeypatch.setattr(guard, "_ssh_effective_hostname", lambda dest, opts: socket.gethostname())
    assert blocked("ssh some-alias 'hermes gateway restart'") is True


def test_unresolvable_destination_fails_closed(monkeypatch):
    """`ssh -G` failing (timeout, missing binary, bad config) means local."""
    monkeypatch.setattr(guard, "_ssh_effective_hostname", lambda dest, opts: None)
    assert blocked("ssh linux-lan 'hermes gateway restart'") is True


@pytest.mark.parametrize(
    ("addresses", "expected"),
    [(["127.0.0.1"], True), (["192.0.2.50", "::1"], True), (None, True), ([], True), (["192.0.2.50"], False)],
)
def test_effective_hostname_name_is_resolved(monkeypatch, addresses, expected):
    """A HostName that is a DNS/mDNS name for THIS box (not its gethostname) stays blocked;
    a failed lookup fails closed. The literal alias itself is never looked up."""
    looked_up: list[str] = []
    monkeypatch.setattr(guard, "_ssh_effective_hostname", lambda dest, opts: "box.tailnet.example")
    monkeypatch.setattr(guard, "_resolve_host_addresses", lambda name: looked_up.append(name) or addresses)
    assert blocked("ssh some-alias 'hermes gateway restart'") is expected
    assert looked_up == ["box.tailnet.example"]


def test_ssh_g_resolution_uses_effective_hostname(monkeypatch):
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        assert kwargs.get("timeout"), "ssh -G must be bounded"
        out = "user me\nhostname localhost\nport 22\n" if "loop-alias" in argv else "hostname 192.0.2.10\n"
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")

    monkeypatch.setattr(guard.subprocess, "run", fake_run)
    assert blocked("ssh loop-alias 'hermes gateway restart'") is True
    assert blocked("ssh -p 2222 linux-lan 'hermes gateway restart'") is False
    assert all(argv[:2] == ["ssh", "-G"] for argv in seen)
    assert any("-p" in argv and "2222" in argv for argv in seen)


@pytest.mark.parametrize(
    "failure",
    [subprocess.TimeoutExpired(["ssh"], 2), FileNotFoundError("ssh"), OSError("boom")],
)
def test_ssh_g_failure_fails_closed(monkeypatch, failure):
    def fake_run(argv, **kwargs):
        raise failure

    monkeypatch.setattr(guard.subprocess, "run", fake_run)
    assert blocked("ssh linux-lan 'hermes gateway restart'") is True


def test_ssh_g_nonzero_exit_fails_closed(monkeypatch):
    monkeypatch.setattr(
        guard.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 255, stdout="", stderr="bad"),
    )
    assert blocked("ssh linux-lan 'hermes gateway restart'") is True


def test_custom_config_file_is_not_evaluated(monkeypatch):
    """`-F` names a config the guard would have to evaluate (Match exec): fail closed."""
    def fake_run(argv, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError(f"ssh -G must not run: {argv}")

    monkeypatch.setattr(guard.subprocess, "run", fake_run)
    assert blocked("ssh -F /tmp/cfg linux-lan 'hermes gateway restart'") is True


def test_resolution_only_runs_for_lifecycle_payloads(resolver):
    assert blocked("ssh linux-lan 'ls -la'; ssh windows-lan uptime") is False
    assert resolver == []


def test_resolution_is_bounded(resolver):
    command = "; ".join(f"ssh host{i} 'hermes gateway restart'" for i in range(40))
    assert blocked(command) is True  # unresolved destinations beyond the cap fail closed
    assert len({dest for dest, _ in resolver}) <= guard._MAX_SSH_RESOLUTIONS


def test_remote_script_paths_are_not_read_or_refused(resolver, monkeypatch, tmp_path):
    read: list[str] = []
    real_read = guard._read_referenced_script

    def spy(path, **kwargs):
        read.append(str(path))
        return real_read(path, **kwargs)

    monkeypatch.setattr(guard, "_read_referenced_script", spy)
    for command in (
        "ssh linux-lan '~/.hermes/agent-hooks/restart.sh'",
        "ssh linux-lan bash ~/.hermes/agent-hooks/restart.sh",
        "ssh linux-lan 'bash ~/Library/Mobile Documents/restart.sh'",
        "ssh linux-lan sh -c 'hermes gateway restart'",
    ):
        assert guard.scan_gateway_lifecycle(command, cwd=str(tmp_path)) == (False, None), command
    assert not any("agent-hooks" in p or "Mobile Documents" in p for p in read), read

    remote_reads: list[str] = []
    assert guard.scan_gateway_lifecycle(
        "ssh linux-lan ./restart.sh", cwd=str(tmp_path),
        read_remote_script=lambda p: remote_reads.append(p) or "hermes gateway restart",
    ) == (False, None)
    assert remote_reads == []


def test_local_script_walk_applies_the_same_rule(resolver, tmp_path):
    remote = tmp_path / "fleet.sh"
    remote.write_text("#!/bin/sh\nssh linux-lan 'hermes gateway restart'\n")
    local = tmp_path / "self.sh"
    local.write_text("#!/bin/sh\nssh linux-lan uptime\nhermes gateway restart\n")
    assert scan_blocked(f"bash {remote}", cwd=str(tmp_path)) is False
    assert scan_blocked(f"bash {local}", cwd=str(tmp_path)) is True


def test_cron_job_spec_uses_the_same_rule(resolver):
    guard.check_gateway_lifecycle("Restart the Linux gateway:\nssh linux-lan 'hermes gateway restart'")
    with pytest.raises(guard.GatewayLifecycleBlocked):
        guard.check_gateway_lifecycle("Restart it:\nssh localhost 'hermes gateway restart'")


@pytest.mark.parametrize(
    ("destination", "host"),
    [
        ("linux-lan", "linux-lan"),
        ("me@linux-lan", "linux-lan"),
        ("a@b@linux-lan", "linux-lan"),
        ("ssh://me@linux-lan:2222", "linux-lan"),
        ("ssh://[::1]:22", "::1"),
        ("SSH://Host", "Host"),
    ],
)
def test_destination_host_parsing(destination, host):
    assert guard._ssh_host_from_destination(destination) == host
