"""Remote-SSH maintenance must not be misread as a local gateway self-restart.

A gateway on host A issuing ``ssh B 'systemctl restart hermes-gateway'`` restarts
B's gateway, not A's — SIGTERM does not propagate across the SSH boundary, so the
self-restart rationale does not apply. The guard must exempt the remote command
payload *only* when B is provably a different host, while still blocking:

- local self-restarts (loopback, localhost, this host's own names/IPs),
- ambiguous targets (host from a variable / command substitution),
- the local component of a compound command,
- referenced scripts (existing inspection is unchanged).

See report: profiles/marin/workspace/reports/remote-gateway-guard-bug.md
"""

from __future__ import annotations

import textwrap

import pytest

import cron.lifecycle_guard as lifecycle_guard

guard = lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script

# Captured before the autouse fixture pins it, so the getifaddrs test can exercise the real impl.
_REAL_LOCAL_IDENTITIES = lifecycle_guard._local_host_identities

# A fixed local identity set keeps the loopback/self tests deterministic across
# machines: 192.168.9.9 stands in for "this host's own LAN IP".
_FIXED_LOCAL = frozenset(
    {"localhost", "localhost.localdomain", "127.0.0.1", "::1", "0.0.0.0",
     "thishost", "thishost.local", "192.168.9.9"}
)


@pytest.fixture(autouse=True)
def _pin_local_identities(monkeypatch):
    monkeypatch.setattr(lifecycle_guard, "_local_host_identities", lambda: _FIXED_LOCAL)


# --- authorized remote maintenance is allowed ---------------------------------

@pytest.mark.parametrize("command", [
    "ssh jim@192.168.3.184 'systemctl --user restart hermes-gateway'",
    "ssh 192.168.3.184 systemctl restart hermes-gateway",
    "ssh -p 2222 -i ~/.ssh/id_ed25519 jim@192.168.3.184 'systemctl --user restart hermes-gateway'",
    "ssh -o StrictHostKeyChecking=no jim@192.168.3.184 'hermes gateway restart'",
    "sudo ssh jim@192.168.3.184 'systemctl restart hermes-gateway'",
    "ssh root@10.4.5.6 'launchctl kickstart -k gui/501/ai.hermes.gateway'",
])
def test_remote_host_maintenance_allowed(command):
    assert not guard(command), f"remote maintenance should be allowed: {command!r}"


def test_ssh_config_alias_to_remote_ip_allowed(monkeypatch, tmp_path):
    home = tmp_path
    ssh_dir = home / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "config").write_text(
        textwrap.dedent(
            """\
            Host jimbox
                HostName 192.168.3.184
                User jim
            """
        )
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Path.expanduser() keys off this on Windows
    assert not guard("ssh jimbox 'systemctl --user restart hermes-gateway'")


# --- the calling gateway stays protected --------------------------------------

@pytest.mark.parametrize("command", [
    # Direct local self-restart forms are untouched by the SSH path.
    "hermes gateway restart",
    "systemctl --user restart hermes-gateway",
    "launchctl kickstart -k gui/501/ai.hermes.gateway",
    # SSH back to the local host in its many spellings.
    "ssh localhost 'systemctl restart hermes-gateway'",
    "ssh root@127.0.0.1 'hermes gateway restart'",
    "ssh [::1] 'systemctl restart hermes-gateway'",
    "ssh 0.0.0.0 'systemctl restart hermes-gateway'",
    "ssh thishost 'systemctl restart hermes-gateway'",
    "ssh 192.168.9.9 'systemctl restart hermes-gateway'",  # this host's own LAN IP
    # Ambiguous target: cannot prove it is remote -> fail closed.
    "ssh $TARGET 'systemctl restart hermes-gateway'",
    "ssh \"$(cat host)\" 'systemctl restart hermes-gateway'",
    # Bare hostname with no ssh-config entry and no DNS resolution -> ambiguous.
    "ssh buildbox 'systemctl restart hermes-gateway'",
])
def test_local_self_restart_still_blocked(command):
    assert guard(command), f"self-restart must stay blocked: {command!r}"


def test_ssh_config_alias_to_loopback_still_blocked(monkeypatch, tmp_path):
    home = tmp_path
    ssh_dir = home / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "config").write_text("Host localbox\n    HostName 127.0.0.1\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Path.expanduser() keys off this on Windows
    assert guard("ssh localbox 'systemctl restart hermes-gateway'")


# --- adversarial: option overrides that redirect the connection (review Finding 1/4) ----------
# A global-looking destination token combined with an option that makes ssh actually dial the
# local host must NOT exempt the payload. These all restart the CALLING gateway if allowed.

@pytest.mark.parametrize("command", [
    "ssh -o HostName=127.0.0.1 8.8.8.8 'systemctl restart hermes-gateway'",
    "ssh -oHostName=127.0.0.1 8.8.8.8 'systemctl restart hermes-gateway'",       # attached
    "ssh -o Hostname=localhost 8.8.8.8 'hermes gateway restart'",                # case-insensitive key
    "ssh 8.8.8.8 -o HostName=127.0.0.1 'hermes gateway restart'",                # override AFTER dest
    "ssh -o HostName=192.168.9.9 8.8.8.8 'systemctl restart hermes-gateway'",    # own LAN IP via override
    "ssh -o ProxyJump=localhost 8.8.8.8 'systemctl restart hermes-gateway'",
    "ssh -o ProxyCommand=nc 8.8.8.8 'systemctl restart hermes-gateway'",
    "ssh -J localhost 8.8.8.8 'systemctl restart hermes-gateway'",
    "ssh -Jlocalhost 8.8.8.8 'hermes gateway restart'",                          # attached -J
    "ssh -F /tmp/eviltunnel.cfg 8.8.8.8 'systemctl restart hermes-gateway'",     # alternate config
    "ssh -F none jimbox 'systemctl restart hermes-gateway'",                     # config bypass
    # Space-separated -o key/value (ssh_config accepts whitespace, not just '=').
    "ssh -o 'HostName 127.0.0.1' 8.8.8.8 'systemctl restart hermes-gateway'",
    "ssh -o 'hostname 127.0.0.1' 8.8.8.8 'systemctl restart hermes-gateway'",
    "ssh 8.8.8.8 -o 'HostName 127.0.0.1' 'hermes gateway restart'",
    "ssh -o 'ProxyJump localhost' 8.8.8.8 'systemctl restart hermes-gateway'",
    # Bundled short-flag clusters ahead of an arg-taking -o / -J / -F (getopt bundling).
    "ssh -tqoHostName=127.0.0.1 8.8.8.8 'systemctl restart hermes-gateway'",
    "ssh -4oHostName=127.0.0.1 8.8.8.8 'hermes gateway restart'",
    "ssh -qoHostName=127.0.0.1 8.8.8.8 'systemctl restart hermes-gateway'",
    "ssh -CqoHostName=localhost 8.8.8.8 'systemctl restart hermes-gateway'",
    "ssh -qo 'HostName 127.0.0.1' 8.8.8.8 'systemctl restart hermes-gateway'",   # bundled + spaced value
    "ssh -qJlocalhost 8.8.8.8 'systemctl restart hermes-gateway'",               # bundled ProxyJump
    "ssh -CqFnone jimbox 'systemctl restart hermes-gateway'",                     # bundled -F none
    "slogin -tqoHostName=127.0.0.1 8.8.8.8 'hermes gateway restart'",
    # Leading '=' separator: ssh discards one leading '=' (`-o=HostName=x` sets HostName).
    "ssh -o=HostName=127.0.0.1 8.8.8.8 'systemctl restart hermes-gateway'",
    "ssh -o =HostName=127.0.0.1 8.8.8.8 'hermes gateway restart'",
    "ssh -qo=HostName=127.0.0.1 8.8.8.8 'systemctl restart hermes-gateway'",
    "ssh -o=ProxyJump=localhost 8.8.8.8 'systemctl restart hermes-gateway'",
    "ssh -o '=HostName 127.0.0.1' 8.8.8.8 'systemctl restart hermes-gateway'",
    # Quoted keyword — ssh strips quotes around the keyword (review Finding 8).
    "ssh -o '\"HostName\" 127.0.0.1' 8.8.8.8 'systemctl restart hermes-gateway'",
    "ssh -o 'HostName\"\" 127.0.0.1' 8.8.8.8 'systemctl restart hermes-gateway'",
    # LocalCommand runs on the CALLING host — payload must stay scanned (review Finding 7).
    "ssh jim@192.168.3.184 -o PermitLocalCommand=yes -o LocalCommand='systemctl restart hermes-gateway' uptime",
    "ssh -tq jim@192.168.3.184 -oPermitLocalCommand=yes -oLocalCommand='hermes gateway restart' echo done",
    "ssh jim@192.168.3.184 uptime -o PermitLocalCommand=yes -o LocalCommand='systemctl restart hermes-gateway'",
    "ssh jim@192.168.3.184 -o LocalCommand='hermes gateway restart'",            # login-only + LocalCommand
    "ssh -o KnownHostsCommand='systemctl restart hermes-gateway' jim@192.168.3.184 uptime",
])
def test_host_override_options_fail_closed(command):
    assert guard(command), f"host-override must fail closed (stay blocked): {command!r}"


def test_remote_operand_masked_but_trailing_option_value_scanned():
    # The genuinely-remote operand ('uptime') is exempt, but a LocalCommand option value carrying a
    # lifecycle command on the SAME line is still blocked — proving only operands are masked.
    assert not guard("ssh jim@192.168.3.184 uptime")
    assert guard(
        "ssh jim@192.168.3.184 -o LocalCommand='hermes gateway restart' uptime"
    )


# --- adversarial: alternate spellings of this host's own IP (review Finding 2) -----------------

@pytest.mark.parametrize("command", [
    "ssh ::ffff:192.168.9.9 'systemctl restart hermes-gateway'",       # IPv4-mapped own LAN IP
    "ssh [::ffff:192.168.9.9] 'systemctl restart hermes-gateway'",     # bracketed
    "ssh ::ffff:127.0.0.1 'systemctl restart hermes-gateway'",         # IPv4-mapped loopback
    "ssh [::1] 'hermes gateway restart'",
    "ssh 0:0:0:0:0:0:0:1 'systemctl restart hermes-gateway'",          # expanded ::1
])
def test_alternate_ip_spellings_of_local_blocked(command):
    assert guard(command), f"own/loopback IP spelling must stay blocked: {command!r}"


@pytest.mark.parametrize("dest", [
    "::ffff:192.168.9.9",   # own LAN IP, IPv4-mapped
    "::ffff:127.0.0.1",     # loopback, IPv4-mapped
    "0:0:0:0:0:0:0:1",      # expanded loopback
    "192.168.9.9.",         # trailing dot on own IP
])
def test_ssh_destination_alternate_local_spellings_are_not_remote(monkeypatch, dest):
    monkeypatch.setattr(lifecycle_guard, "_local_host_identities", lambda: _FIXED_LOCAL)
    assert lifecycle_guard._ssh_destination_is_remote(dest) is False


def test_slogin_remote_allowed_but_override_blocked():
    assert not guard("slogin jim@192.168.3.184 'systemctl --user restart hermes-gateway'")
    assert guard("slogin -o HostName=127.0.0.1 8.8.8.8 'systemctl restart hermes-gateway'")


def test_all_local_interface_ips_recognized_via_getifaddrs(monkeypatch):
    # A multi-homed host: the default route yields one IP but a second NIC has another. Full
    # interface enumeration (psutil.net_if_addrs) must recognize BOTH as local so `ssh <second-nic>
    # 'systemctl restart hermes-gateway'` is a self-restart and stays blocked.
    import socket

    import psutil

    import cron.lifecycle_guard as lg

    class _Addr:
        def __init__(self, address):
            self.address = address

    fake = {
        "lo0": [_Addr("127.0.0.1"), _Addr("::1")],
        "en0": [_Addr("192.168.3.152"), _Addr("aa:bb:cc:dd:ee:ff")],   # + a MAC, must be ignored
        "en1": [_Addr("192.168.3.214")],
        "utun3": [_Addr("100.90.1.2%utun3")],                          # VPN w/ IPv6-style zone id
    }
    # Restore the real implementation (the autouse fixture pinned a fixed set) so we exercise the
    # actual getifaddrs enumeration, with psutil/gethostname patched for determinism.
    monkeypatch.setattr(lg, "_local_host_identities", _REAL_LOCAL_IDENTITIES)
    monkeypatch.setattr(psutil, "net_if_addrs", lambda: fake)
    monkeypatch.setattr(socket, "gethostname", lambda: "")
    identities = lg._local_host_identities()
    assert "192.168.3.152" in identities
    assert "192.168.3.214" in identities
    assert "100.90.1.2" in identities            # zone id stripped
    assert "aa:bb:cc:dd:ee:ff" not in identities  # MAC ignored
    assert lg._ssh_destination_is_remote("192.168.3.214") is False
    assert lg._ssh_destination_is_remote("100.90.1.2") is False
    assert lg._ssh_destination_is_remote("192.168.3.184") is True  # a genuine peer stays remote


# --- compound commands keep protection for the local component ----------------

@pytest.mark.parametrize("command", [
    "ssh jim@192.168.3.184 'uptime'; hermes gateway restart",
    "hermes gateway restart && ssh jim@192.168.3.184 'uptime'",
    "ssh jim@192.168.3.184 'echo ok' | systemctl restart hermes-gateway",
    "ssh jim@192.168.3.184 'systemctl restart hermes-gateway' ; systemctl restart hermes-gateway",
])
def test_compound_local_component_blocked(command):
    assert guard(command), f"local component must stay blocked: {command!r}"


# --- referenced scripts keep their existing inspection ------------------------

def test_referenced_local_script_still_scanned(tmp_path):
    script = tmp_path / "restart.sh"
    script.write_text("#!/bin/sh\nhermes gateway restart\n")
    assert guard(f"bash {script}")


def test_remote_payload_does_not_expand_local_script_reference(tmp_path):
    # The remote command references a path that also exists locally; it must NOT
    # be read as a local script (it runs on the remote host). The command is a
    # benign remote copy, so nothing should block.
    local_twin = tmp_path / "deploy.sh"
    local_twin.write_text("#!/bin/sh\necho harmless\n")
    assert not guard(f"ssh jim@192.168.3.184 'bash {local_twin}'")


# --- direct unit coverage of the classifier -----------------------------------

@pytest.mark.parametrize("dest,expected", [
    ("192.168.3.184", True),
    ("jim@192.168.3.184", True),
    ("10.0.0.5", True),
    ("8.8.8.8", True),
    ("127.0.0.1", False),
    ("::1", False),
    ("0.0.0.0", False),
    ("localhost", False),
    ("thishost", False),
    ("192.168.9.9", False),          # pinned local identity
    ("$HOST", False),                 # variable -> ambiguous
    ("`hostname`", False),            # command substitution -> ambiguous
    ("buildbox", False),              # unresolvable bare name -> ambiguous
    ("", False),
])
def test_ssh_destination_is_remote(monkeypatch, dest, expected):
    monkeypatch.setattr(lifecycle_guard, "_local_host_identities", lambda: _FIXED_LOCAL)
    assert lifecycle_guard._ssh_destination_is_remote(dest) is expected


# --- only ssh/slogin get the remote exemption ---------------------------------

@pytest.mark.parametrize("command", [
    # scp/rsync/telnet and look-alike names are NOT remote-shell executors here, so their
    # arguments are never masked — a lifecycle command in that text stays fully scanned.
    "scp jim@192.168.3.184 'systemctl restart hermes-gateway'",
    "rsync jim@192.168.3.184 'hermes gateway restart'",
    "sshpass jim@192.168.3.184 'systemctl restart hermes-gateway'",
    "notssh jim@192.168.3.184 'systemctl restart hermes-gateway'",
])
def test_only_ssh_and_slogin_are_masked(command):
    assert guard(command), f"non-ssh executable must not gain the exemption: {command!r}"


# --- getifaddrs failure degrades to the probe fallback without crashing -------

def test_local_identities_fall_back_when_getifaddrs_fails(monkeypatch):
    import psutil

    def _boom():
        raise RuntimeError("simulated getifaddrs failure")

    monkeypatch.setattr(lifecycle_guard, "_local_host_identities", _REAL_LOCAL_IDENTITIES)
    monkeypatch.setattr(psutil, "net_if_addrs", _boom)
    identities = lifecycle_guard._local_host_identities()  # must not raise
    assert "127.0.0.1" in identities and "::1" in identities  # loopback still recognized
    # And a loopback self-restart stays blocked even with enumeration degraded.
    assert lifecycle_guard._ssh_destination_is_remote("127.0.0.1") is False
