"""Behaviour contracts for the gateway SSH attach path (hermes_cli/gateway_connections_ssh*.py).

The remote-shell pieces are exercised by running them in a real local ``sh``/``dash`` against a
temp ``HOME`` with a fake ``hermes`` executable, so the spawn command, lockfile publication and
marker gate are proven by execution rather than by string shape.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import gateway_connections_ssh as facade
from hermes_cli.gateway_connections_ssh_remote import (
    READY_RE,
    LockfileSkew,
    SpawnPlan,
    backend_profile_arg,
    build_spawn_command,
    classify_ssh_reuse_proof,
    expand_remote_path,
    fingerprint_token,
    lockfile_path,
    parse_lockfile,
    remote_install_root,
    spawn_log_path,
    spawn_token_path,
)
from hermes_cli.gateway_connections_ssh_transport import (
    SshConnection,
    SshEndpoint,
    SshReachError,
    build_control_args,
    build_exec_args,
    build_master_args,
    classify_ssh_error,
    control_socket_path,
    default_control_dir,
    forward_spec,
    login_shell_quote,
    redact_secrets,
    short_control_dir,
    validate_key_path,
    validate_ssh_target,
    with_remote_timeout,
)
from tests.waiting import wait_until

OID = "0123456789abcdef0123456789abcdef"
NONCE = "fedcba9876543210"
POSIX_SHELLS = [s for s in ("sh", "dash") if shutil.which(s)]


def _endpoint(**kw) -> SshEndpoint:
    return SshEndpoint(**{"host": "box.example", "user": "me", "port": 22, "control_path": "/c/s.sock", **kw})


# ---------------------------------------------------------------------------
# Argument building
# ---------------------------------------------------------------------------


def _opts(args: list[str]) -> list[str]:
    return [args[i + 1] for i, a in enumerate(args) if a == "-o"]


@pytest.mark.parametrize("build", [
    lambda e: build_exec_args(e, "true"),
    lambda e: build_master_args(e),
    lambda e: build_control_args(e, "check"),
])
def test_every_invocation_is_batch_mode_tofu_and_target_follows_double_dash(build):
    args = build(_endpoint(port=2222, key_path="/k/id"))
    opts = _opts(args)
    assert "BatchMode=yes" in opts and "StrictHostKeyChecking=accept-new" in opts
    assert "ControlMaster=auto" in opts and "ControlPath=/c/s.sock" in opts
    assert args[args.index("-p") + 1] == "2222" and args[args.index("-i") + 1] == "/k/id"
    assert args[args.index("--") + 1] == "me@box.example"


def test_default_port_and_no_key_add_no_host_flags():
    args = build_exec_args(_endpoint(user=""), "true")
    assert "-p" not in args and "-i" not in args
    assert args[-2:] == ["box.example", "true"]


def test_forward_binds_the_local_end_to_loopback_only():
    assert forward_spec(4000, 5000) == "127.0.0.1:4000:127.0.0.1:5000"
    args = build_control_args(_endpoint(), "forward", ("-L", forward_spec(4000, 5000)))
    assert args[:4] == ["-O", "forward", "-L", "127.0.0.1:4000:127.0.0.1:5000"]


# ---------------------------------------------------------------------------
# Target / key validation + redaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host,user,port", [
    ("-oProxyCommand=x", "", 22), ("ssh box", "", 22), ("10,0,0,1", "", 22), ("box\x07", "", 22),
    ("box", "-l", 22), ("box", "a@b", 22), ("box", "a b", 22), ("box", "", 0), ("box", "", 70000),
    ("box", "", "22"), ("", "", 22), ("ho$t", "", 22),
])
def test_unsafe_targets_are_rejected(host, user, port):
    with pytest.raises(SshReachError):
        validate_ssh_target(host, user, port)


def test_single_colon_host_never_echoes_the_suspect_segment():
    with pytest.raises(SshReachError) as exc:
        validate_ssh_target("box:hunter2", "", 22)
    assert "hunter2" not in str(exc.value)


@pytest.mark.parametrize("host", ["box.example", "10.0.0.1", "fe80::1%eth0", "::1", "my_host-1"])
def test_plausible_hosts_are_accepted(host):
    validate_ssh_target(host, "me", 22)


@pytest.mark.parametrize("key", ["-oFoo", "/k/id\n"])
def test_unsafe_key_paths_are_rejected(key):
    with pytest.raises(SshReachError):
        validate_key_path(key)


def test_redaction_masks_tokens_and_host_field_passwords():
    text = ("HERMES_DASHBOARD_SESSION_TOKEN=abc X-Hermes-Session-Token: def "
            "Authorization: Bearer ghi /x?token=jkl me@box:secret me@box:22")
    out = redact_secrets(text)
    for secret in ("abc", "def", "ghi", "jkl", "secret"):
        assert secret not in out
    assert "me@box:22" in out


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stderr,kind", [
    ("@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@ ... Permission denied", "host-key-changed"),
    ("Host key verification failed.", "host-key-changed"),
    ("Tailscale SSH requires an additional check. To authenticate, visit: https://login.tailscale.com/a/x",
     "interactive-auth"),
    ("me@box: Permission denied (publickey).", "auth-failed"),
    ("ssh: connect to host box port 22: Connection refused", "unreachable"),
    ("ssh: Could not resolve hostname box: Name or service not known", "unreachable"),
    ("mux_client_forward: forwarding request failed", "unknown"),
])
def test_ssh_stderr_classifies_to_an_actionable_kind(stderr, kind):
    assert classify_ssh_error(stderr) == kind


@pytest.mark.platforms("posix")
def test_a_real_refused_connection_raises_unreachable(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with socket_port() as port:
        pass  # released: nothing listens on it now
    ssh_binary = shutil.which("ssh")
    if not ssh_binary:
        pytest.skip("OpenSSH client not installed")
    # A refused dial never binds a socket, so the (possibly shared short) control dir stays empty.
    control_dir = default_control_dir(tmp_path)
    conn = SshConnection(SshEndpoint(host="127.0.0.1", port=port, control_path=str(
        control_socket_path("", "127.0.0.1", port, control_dir, ownership_id=OID))),
        ssh_binary=ssh_binary, connect_timeout_s=5)
    with pytest.raises(SshReachError) as exc:
        conn.open()
    assert exc.value.kind == "unreachable"


class socket_port:
    def __enter__(self) -> int:
        import socket

        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        return self._sock.getsockname()[1]

    def __exit__(self, *exc) -> None:
        self._sock.close()


# ---------------------------------------------------------------------------
# Control socket placement + ownership identity
# ---------------------------------------------------------------------------


def test_control_dir_stays_under_the_root_when_sun_path_fits():
    # Pure path arithmetic: nothing is created, so a short literal root is safe here.
    assert default_control_dir(Path("/srv/h")) == Path("/srv/h/gateway-ssh")


def test_control_dir_falls_back_to_the_short_dir_for_a_deep_root(tmp_path):
    assert default_control_dir(tmp_path / ("d" * 120)) == short_control_dir()


def test_control_socket_is_stable_and_scoped_by_owner_and_key():
    a = control_socket_path("me", "box", 22, Path("/c"), key_path="/k", ownership_id=OID)
    assert a == control_socket_path("me", "box", 22, Path("/c"), key_path="/k", ownership_id=OID)
    assert a != control_socket_path("me", "box", 22, Path("/c"), key_path="/k2", ownership_id=OID)
    assert a != control_socket_path("me", "box", 22, Path("/c"), key_path="/k", ownership_id="f" * 32)
    assert len(a.name) == len("0123456789abcdef.sock")


def test_a_profile_home_shares_its_roots_ownership_id(tmp_path, monkeypatch):
    root = tmp_path / "home"
    (root / "profiles" / "work").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "work"))
    assert facade.store_root() == root.resolve()
    oid = facade.ownership_id(facade.store_root(), "c1")
    assert len(oid) == 32 and oid == facade.ownership_id(root.resolve(), "c1")
    assert oid != facade.ownership_id(root.resolve(), "c2")


def test_unknown_connection_has_no_status_and_nothing_to_disconnect():
    assert facade.tunnel_status("never-opened") is None
    assert facade.disconnect("never-opened") is False


# ---------------------------------------------------------------------------
# Lockfile parse: skew (foreign/corrupt, fail closed) vs none (no record / not reusable)
# ---------------------------------------------------------------------------


def _lock(**overrides) -> dict:
    lock = {"schemaVersion": 2, "ownershipId": OID, "spawnNonce": NONCE, "pid": 4242, "port": 5555,
            "profile": "", "hermesPath": "~/bin/hermes", "hermesHome": "~/.hermes",
            "logPath": spawn_log_path(OID, NONCE), "tokenFingerprint": fingerprint_token("t"),
            "protocolVersion": 1, "devServer": "", "startedAt": "2026-01-01T00:00:00.000Z"}
    lock.update(overrides)
    return {k: v for k, v in lock.items() if v is not _DROP}


_DROP = object()


def test_a_valid_lockfile_parses_to_its_record():
    assert parse_lockfile(json.dumps(_lock()), OID)["pid"] == 4242


@pytest.mark.parametrize("text", [
    "{trunc", "[]", json.dumps(_lock(schemaVersion=3)), json.dumps(_lock(pid="4242")),
    json.dumps(_lock(pid=0)), json.dumps(_lock(port=70000)), json.dumps(_lock(ownershipId="f" * 32)),
    json.dumps(_lock(spawnNonce="xyz")), json.dumps(_lock(tokenFingerprint="raw-token")),
    json.dumps(_lock(logPath="~/elsewhere.log")), json.dumps(_lock(startedAt=_DROP)),
])
def test_foreign_or_corrupt_lockfiles_are_skew_not_absence(text):
    assert isinstance(parse_lockfile(text, OID), LockfileSkew)


@pytest.mark.parametrize("text", ["", "  \n", json.dumps(_lock(protocolVersion=0)),
                                  json.dumps(_lock(creationTime="windows:1"))])
def test_absent_or_protocol_stale_own_lockfiles_read_as_no_record(text):
    assert parse_lockfile(text, OID) is None


# ---------------------------------------------------------------------------
# Spawn command, executed in a real POSIX shell
# ---------------------------------------------------------------------------


def _plan(home: Path, hermes: Path, nonce: str = NONCE) -> SpawnPlan:
    return SpawnPlan(hermes_path=str(hermes), profile="Work", spawn_nonce=nonce, ownership_id=OID,
                     token_file_path=spawn_token_path(OID, nonce), log_path=spawn_log_path(OID, nonce),
                     hermes_home=str(home / ".hermes"),
                     lock_metadata={"ownershipId": OID, "spawnNonce": nonce, "port": 0, "profile": "Work",
                                    "hermesPath": str(hermes), "hermesHome": str(home / ".hermes"),
                                    "logPath": spawn_log_path(OID, nonce),
                                    "tokenFingerprint": fingerprint_token("t"), "protocolVersion": 1,
                                    "devServer": "", "startedAt": "2026-01-01T00:00:00.000Z"})


@pytest.fixture
def remote_home(tmp_path):
    home = tmp_path / "rhome"
    (home / ".hermes").mkdir(parents=True)
    hermes = tmp_path / "bin" / "hermes"
    hermes.parent.mkdir()
    # Records its argv + env, announces a port the way serve does, then idles like a backend.
    hermes.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$HOME/argv"; echo "DESKTOP=$HERMES_DESKTOP" >> "$HOME/argv"\n'
                      'echo "noise HERMES_DASHBOARD_READY port=43210"\nexec sleep 60\n')
    hermes.chmod(0o755)
    return home, hermes


def _run_remote(shell: str, command: str, home: Path) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home)}
    return subprocess.run([shell, "-c", command], env=env, capture_output=True, text=True, timeout=30, check=False)


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shell", POSIX_SHELLS)
def test_spawn_command_parses_under_posix_sh(shell, remote_home):
    home, hermes = remote_home
    assert subprocess.run([shell, "-n", "-c", build_spawn_command(_plan(home, hermes))], check=False).returncode == 0


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the serve is setsid-detached by design; the test reaps its own child
@pytest.mark.parametrize("shell", POSIX_SHELLS)
def test_spawn_command_detaches_serve_and_publishes_its_pid(shell, remote_home):
    home, hermes = remote_home
    result = _run_remote(shell, build_spawn_command(_plan(home, hermes)), home)
    assert result.returncode == 0, result.stderr
    pid = int(result.stdout.strip().splitlines()[-1])
    try:
        lock_file = home / lockfile_path(OID).removeprefix("~/")
        lock = parse_lockfile(lock_file.read_text(), OID)
        assert isinstance(lock, dict) and lock["pid"] == pid and lock["profile"] == "work"
        argv_file = home / "argv"
        wait_until(argv_file.exists, timeout=10)
        argv = argv_file.read_text().splitlines()
        assert argv[:2] == ["--profile", "work"]
        assert argv[2:9] == ["serve", "--isolated", "--host", "127.0.0.1", "--port", "0", "--ssh-session-token-file"]
        assert argv[9] == str(home / spawn_token_path(OID, NONCE).removeprefix("~/"))
        assert argv[10:12] == ["--ssh-owner-nonce", NONCE] and argv[12] == "DESKTOP=1"
        log = home / spawn_log_path(OID, NONCE).removeprefix("~/")
        wait_until(lambda: READY_RE.search(log.read_text()) is not None, timeout=10)
        assert int(READY_RE.search(log.read_text()).group(1)) == 43210

        # A live published owner makes a second spawn back off instead of starting a twin.
        second = _run_remote(shell, build_spawn_command(_plan(home, hermes, "0011223344556677")), home)
        assert second.stdout.strip().splitlines()[-1] == "EXISTING"
    finally:
        _kill(pid)


class _LocalShell:
    """``SshConnection.exec`` against this machine's ``sh`` with the fixture's remote HOME."""

    def __init__(self, home: Path):
        self.home = home

    def exec(self, remote_command: str, *, timeout_s=None, stdin_data=None) -> str:
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.home)}
        return subprocess.run(["sh", "-c", remote_command], env=env, input=stdin_data, capture_output=True,
                              text=True, timeout=30, check=True).stdout


@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # the serve is setsid-detached by design; the test reaps its own child
def test_a_quote_in_the_hermes_path_still_proves_ownership_of_our_serve(tmp_path):
    from hermes_cli.gateway_connections_ssh_remote import pid_is_our_serve

    home = tmp_path / "rhome"
    (home / ".hermes").mkdir(parents=True)
    hermes = tmp_path / "it's bin" / "hermes"
    hermes.parent.mkdir()
    hermes.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n")  # keeps its serve argv
    hermes.chmod(0o755)
    result = _run_remote("sh", build_spawn_command(_plan(home, hermes)), home)
    assert result.returncode == 0, result.stderr
    pid = int(result.stdout.strip().splitlines()[-1])
    try:
        lock = parse_lockfile((home / lockfile_path(OID).removeprefix("~/")).read_text(), OID)
        wait_until(lambda: NONCE.encode() in subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(pid)],
                                                            capture_output=True, check=False).stdout, timeout=10)
        assert pid_is_our_serve(_LocalShell(home), lock, OID) is True
        assert pid_is_our_serve(_LocalShell(home), {**lock, "spawnNonce": "0011223344556677"}, OID) is False
    finally:
        _kill(pid)


@pytest.mark.platforms("posix")
def test_spawn_command_refuses_while_an_update_marker_is_live(remote_home):
    home, hermes = remote_home
    marker = Path(remote_install_root(str(home / ".hermes"))) / ".hermes-update-in-progress"
    marker.write_text(f"{os.getpid()}\n{int(time.time())}\n")
    result = _run_remote("sh", build_spawn_command(_plan(home, hermes)), home)
    assert result.returncode == 75 and f"LIVE:{os.getpid()}" in result.stderr
    assert not (home / lockfile_path(OID).removeprefix("~/")).exists()
    assert not (home / "argv").exists()


# ---------------------------------------------------------------------------
# Remote shell helpers
# ---------------------------------------------------------------------------


@pytest.mark.platforms("posix")
def test_remote_watchdog_kills_a_hung_probe_and_keeps_a_healthy_ones_output():
    started = time.monotonic()
    hung = subprocess.run(["sh", "-c", with_remote_timeout("sleep 30", 1)], capture_output=True, timeout=20, check=False)
    assert hung.returncode != 0 and time.monotonic() - started < 15
    ok = subprocess.run(["sh", "-c", with_remote_timeout("echo hi", 5)], capture_output=True, text=True, timeout=20, check=False)
    assert ok.returncode == 0 and ok.stdout.strip() == "hi"


@pytest.mark.platforms("posix")
def test_login_shell_quote_round_trips_quotes_and_backslashes():
    value = "a'b\\c $HOME `x`"
    out = subprocess.run(["sh", "-c", f"printf %s {login_shell_quote(value)}"], capture_output=True, text=True, check=False)
    assert out.stdout == value


@pytest.mark.platforms("posix")
def test_expand_remote_path_expands_home_only_at_the_tilde(tmp_path):
    out = subprocess.run(["sh", "-c", f"printf %s {expand_remote_path('~/a b/$c')}"], capture_output=True,
                         text=True, check=False, env={"HOME": str(tmp_path), "PATH": os.environ.get("PATH", "")})
    assert out.stdout == f"{tmp_path}/a b/$c"


@pytest.mark.parametrize("path", ["", "relative/hermes", "~user/x", "/a\nb"])
def test_unsafe_remote_paths_are_rejected(path):
    with pytest.raises(SshReachError):
        expand_remote_path(path)


# ---------------------------------------------------------------------------
# READY scrape + ownership proof + profile arg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("log,port", [
    ("boot\nHERMES_DASHBOARD_READY port=4100\n", 4100),
    ("partial line HERMES_BACKEND_READY port=4200", 4200),
    ("XHERMES_DASHBOARD_READY port=4300", None),
    ("HERMES_DASHBOARD_READY port=", None),
])
def test_ready_port_is_scraped_from_merged_output(log, port):
    match = READY_RE.search(log)
    assert (int(match.group(1)) if match else None) == port


@pytest.mark.parametrize("proof,verdict", [
    ({"ok": True, "sshOwnerNonce": NONCE, "protocolVersion": 1, "runtimeIntact": True}, "authenticated-ok"),
    ({"ok": True, "sshOwnerNonce": NONCE, "protocolVersion": 1}, "authenticated-ok"),
    ({"ok": True, "sshOwnerNonce": NONCE, "protocolVersion": 1, "runtimeIntact": False}, "authenticated-stale"),
    ({"ok": True, "sshOwnerNonce": "0" * 16, "protocolVersion": 1}, "authenticated-stale"),
    ({"ok": True, "sshOwnerNonce": NONCE, "protocolVersion": 2}, "authenticated-stale"),
    (None, "authenticated-stale"),
])
def test_reuse_proof_requires_our_nonce_protocol_and_intact_runtime(proof, verdict):
    assert classify_ssh_reuse_proof(proof, NONCE) == verdict


@pytest.mark.parametrize("value,expected", [
    (" Work ", "work"), ("default", "default"), ("", None), ("a b", None), (0, None), ("-x", None),
])
def test_only_slug_profiles_cross_into_the_spawn_argv(value, expected):
    assert backend_profile_arg(value) == expected
