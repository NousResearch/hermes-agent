"""OpenSSH ControlMaster transport for gateway SSH connections.

Server-side port of Desktop's ``apps/desktop/electron/ssh-connection.ts`` (POSIX mux path
only). Uses the system ``ssh`` client so ``~/.ssh/config``, the agent, ProxyJump and hardware
keys work unchanged. Every invocation runs with ``BatchMode=yes`` (a spawned ssh must never hang
on a passphrase/2FA prompt), ``StrictHostKeyChecking=accept-new`` (TOFU; a changed key fails
closed) and a hard timeout: a half-open TCP connection after a network change can leave ssh
hanging forever, so a timeout is treated as a dead connection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
import stat
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT_S = 15.0
DEFAULT_EXEC_TIMEOUT_S = 20.0
DEFAULT_FORWARD_TIMEOUT_S = 15.0
# Remote-side watchdog for probe commands: killing the LOCAL ssh child leaves a hung remote CLI
# running as an orphan, so the remote kill must land before the local timeout.
REMOTE_PROBE_TIMEOUT_S = 15
CONTROL_PERSIST_S = 300
# ``ssh -O forward`` does not keep a ControlPersist master busy; refresh it while forwards exist
# so the idle timer cannot silently drop their listeners.
CONTROL_FORWARD_KEEPALIVE_S = min(60.0, CONTROL_PERSIST_S / 2)
SUN_PATH_MAX = 104

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_IPV6_RE = re.compile(r"^[0-9A-Fa-f:.]+(?:%[A-Za-z0-9._-]+)?$")


class SshReachError(RuntimeError):
    """A classified failure reaching or starting a remote Hermes over SSH."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# Validation + redaction
# ---------------------------------------------------------------------------

_REDACTIONS = (
    (re.compile(r"(HERMES_DASHBOARD_SESSION_TOKEN=)(\S+)"), r"\1<redacted>"),
    (re.compile(r"""(X-Hermes-Session-Token["']?\s*[:=]\s*["']?)([^\s"'&]+)""", re.IGNORECASE), r"\1<redacted>"),
    (re.compile(r"""(Authorization["']?\s*:\s*Bearer\s+)(\S+)""", re.IGNORECASE), r"\1<redacted>"),
    (re.compile(r"""([?&](?:token|ticket)=)([^\s&"']+)""", re.IGNORECASE), r"\1<redacted>"),
    # A mistyped password in the host field (user@host:SECRET) must never reach logs verbatim.
    (re.compile(r"(\S+@[^\s:]+):(?!\d+\b)[^\s:]+"), r"\1:<redacted>"),
)


def redact_secrets(text: object) -> str:
    out = "" if text is None else str(text)
    for pattern, repl in _REDACTIONS:
        out = pattern.sub(repl, out)
    return out


def _reject(message: str) -> SshReachError:
    return SshReachError("unknown", message)


def _validate_host(host: str) -> None:
    if not host or not isinstance(host, str):
        raise _reject("Unsafe SSH target: host is required.")
    if host.startswith("-"):
        raise _reject(f'Unsafe SSH target: host must not start with a dash ("{host}").')
    if _CONTROL_CHAR_RE.search(host):
        raise _reject("Unsafe SSH target: host contains control characters.")
    if re.search(r"\s", host):
        raise _reject("Invalid SSH host: contains whitespace. Enter only the destination (user@host or host).")
    if "," in host:
        raise _reject(f'Invalid SSH host "{host}": commas are not valid in a hostname or IP.')
    if ":" in host:
        # Only a bare IPv6 address may contain colons; a single colon is a failed host:port parse
        # or a pasted credential, so the suspect segment is never echoed.
        if host.count(":") < 2 or not _IPV6_RE.match(host):
            raise _reject(f'Invalid SSH host "{host.split(":")[0]}:<hidden>": unexpected ":" segment.')
    elif not _HOSTNAME_RE.match(host):
        raise _reject(f'Invalid SSH host "{redact_secrets(host)}": not a valid hostname or IP address.')


def validate_ssh_target(host: str, user: str, port: object) -> None:
    _validate_host(host)
    if user and _CONTROL_CHAR_RE.search(user):
        raise _reject("Unsafe SSH target: user contains control characters.")
    if user and user.startswith("-"):
        raise _reject(f'Unsafe SSH target: user must not start with a dash ("{user}").')
    if user and re.search(r"[\s@]", user):
        raise _reject(f'Invalid SSH user "{user}": contains whitespace or "@".')
    if type(port) is not int or not 1 <= port <= 65535:
        raise _reject(f"Unsafe SSH port: {port} (must be 1-65535).")


def validate_key_path(key_path: str) -> None:
    if not key_path:
        return
    if _CONTROL_CHAR_RE.search(key_path):
        raise _reject("Unsafe SSH key path: contains control characters.")
    if key_path.startswith("-"):
        raise _reject(f'Unsafe SSH key path: must not start with a dash ("{key_path}").')


# ---------------------------------------------------------------------------
# Control socket path
# ---------------------------------------------------------------------------


def short_control_dir() -> Path:
    # no-tmp: ok — AF_UNIX's 104-byte sun_path rules out a deep root; parent and child are checked before use.
    return Path(f"/tmp/hermes-ssh-{os.getuid()}")  # windows-footgun: ok — POSIX-only (win32 refused upstream)


def default_control_dir(root: Path) -> Path:
    """``<root>/gateway-ssh`` when a socket plus OpenSSH's temporary-listener suffix fits sun_path."""
    candidate = Path(root) / "gateway-ssh"
    probe = candidate / "0123456789abcdef.sock.0123456789abcdef"
    return candidate if len(os.fsencode(str(probe))) <= SUN_PATH_MAX else short_control_dir()


def control_socket_path(user: str, host: str, port: int, control_dir: Path, *, key_path: str = "",
                        ownership_id: str = "") -> Path:
    """Stable across reconnects (so ControlMaster reuse works), short (so sun_path fits)."""
    parts = [ownership_id, user or "", host, int(port), os.path.normpath(key_path) if key_path else ""]
    digest = hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:16]
    return Path(control_dir) / f"{digest}.sock"


def check_short_control_parent() -> None:
    # /tmp can be a symlink (macOS). Inspect the resolved directory before creating anything there;
    # the sticky bit protects an owned child from rename.
    parent = short_control_dir().parent
    st = os.stat(os.path.realpath(parent))
    writable_unsticky = (st.st_mode & 0o022) and not (st.st_mode & stat.S_ISVTX)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid not in (0, os.getuid()) or writable_unsticky:  # windows-footgun: ok — POSIX only
        raise _reject(f"Unsafe SSH control parent: {parent} must be owned by root or this user and sticky if writable.")


def prepare_control_dir(control_dir: Path) -> None:
    if control_dir == short_control_dir():
        check_short_control_parent()
    control_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = os.lstat(control_dir)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():  # windows-footgun: ok — POSIX only
        raise _reject(f"Unsafe SSH control dir: {control_dir} is not a directory owned by this user (no symlinks).")
    if st.st_mode & 0o777 != 0o700:
        os.chmod(control_dir, 0o700)


# ---------------------------------------------------------------------------
# Argument builders (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SshEndpoint:
    host: str
    user: str = ""
    port: int = 22
    key_path: str = ""
    control_path: str = ""

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host


def base_ssh_options(control_path: str, connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S) -> list[str]:
    connect_secs = max(1, round(connect_timeout_s))
    mux = ["-o", f"ControlPath={control_path}", "-o", "ControlMaster=auto",
           "-o", f"ControlPersist={CONTROL_PERSIST_S}"] if control_path else []
    return [*mux, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ExitOnForwardFailure=yes", "-o", f"ConnectTimeout={connect_secs}",
            # 15s keepalive, drop after 3 misses, so NAT/firewall timeouts can't silently kill it.
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "-o", "TCPKeepAlive=yes"]


def host_args(endpoint: SshEndpoint) -> list[str]:
    args: list[str] = []
    if endpoint.port and endpoint.port != 22:
        args += ["-p", str(endpoint.port)]
    if endpoint.key_path:
        validate_key_path(endpoint.key_path)
        args += ["-i", endpoint.key_path]
    return args


def build_exec_args(endpoint: SshEndpoint, remote_command: str,
                    connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S) -> list[str]:
    return [*base_ssh_options(endpoint.control_path, connect_timeout_s), *host_args(endpoint),
            "--", endpoint.target, remote_command]


def build_control_args(endpoint: SshEndpoint, op: str, extra: tuple[str, ...] = (),
                       connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S) -> list[str]:
    return ["-O", op, *extra, *base_ssh_options(endpoint.control_path, connect_timeout_s),
            *host_args(endpoint), "--", endpoint.target]


def build_master_args(endpoint: SshEndpoint, connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S) -> list[str]:
    # -M -N -f backgrounds ssh once the master is up, so the call returns when it is established.
    return ["-M", "-N", "-f", *base_ssh_options(endpoint.control_path, connect_timeout_s),
            *host_args(endpoint), "--", endpoint.target]


def forward_spec(local_port: int, remote_port: int, remote_host: str = "127.0.0.1") -> str:
    # The local end binds 127.0.0.1 ONLY, never 0.0.0.0: the tunnel must not re-expose the remote
    # dashboard to this machine's LAN.
    return f"127.0.0.1:{int(local_port)}:{remote_host}:{int(remote_port)}"


def login_shell_quote(value: str) -> str:
    """Quote for the login shell sshd invokes. fish treats ``\\`` inside ``'…'``; POSIX does not, so
    close the quote and escape outside it so both decode identically."""
    return "'" + re.sub(r"['\\]", lambda m: "'\\''" if m.group(0) == "'" else "'\\\\'", str(value)) + "'"


def with_remote_timeout(remote_command: str, timeout_s: int = REMOTE_PROBE_TIMEOUT_S) -> str:
    """Wrap ONE remote command in a POSIX-sh watchdog (no GNU ``timeout``: macOS lacks it).

    ``set -m`` gives the probe its own process group so the watchdog reaches a grandchild too;
    non-interactive zsh exits on monitor mode, so it is skipped there. The sleeper's stdio is
    detached so an orphaned ``sleep`` cannot hold the ssh channel open on the healthy path."""
    secs = int(timeout_s) if timeout_s and timeout_s > 0 else REMOTE_PROBE_TIMEOUT_S
    return (
        f'[ -n "${{ZSH_VERSION-}}" ] || set -m 2>/dev/null; ({remote_command}) </dev/null & __htp=$!; set +m 2>/dev/null; '
        f"(sleep {secs} </dev/null >/dev/null 2>&1; kill -9 -- -$__htp 2>/dev/null; kill -9 $__htp 2>/dev/null) & __htw=$!; "
        "wait $__htp; __htrc=$?; "
        "kill $__htw 2>/dev/null; wait $__htw 2>/dev/null; "
        "exit $__htrc"
    )


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

_HOST_KEY_RE = re.compile(r"REMOTE HOST IDENTIFICATION HAS CHANGED|Host key verification failed|"
                          r"Offending (?:key|ECDSA|RSA|ED25519)", re.IGNORECASE)
_INTERACTIVE_RE = re.compile(r"Tailscale SSH requires an additional check|"
                             r"To authenticate, visit:\s*https://login\.tailscale\.com/", re.IGNORECASE)
_AUTH_RE = re.compile(r"Permission denied|Too many authentication failures|no matching host key|publickey|"
                      r"password|keyboard-interactive", re.IGNORECASE)
_UNREACHABLE_RE = re.compile(r"Could not resolve hostname|Connection refused|Connection timed out|No route to host|"
                             r"Network is unreachable|Operation timed out|port \d+: Connection", re.IGNORECASE)


def classify_ssh_error(stderr: str) -> str:
    """Order matters: the host-key-change banner also mentions keys, so it is checked first."""
    text = str(stderr or "")
    for pattern, kind in ((_HOST_KEY_RE, "host-key-changed"), (_INTERACTIVE_RE, "interactive-auth"),
                          (_AUTH_RE, "auth-failed"), (_UNREACHABLE_RE, "unreachable")):
        if pattern.search(text):
            return kind
    return "unknown"


def ssh_error_message(kind: str, endpoint: SshEndpoint, stderr: str = "") -> str:
    target = endpoint.target
    detail = str(stderr or "").strip()
    port_arg = f" -p {endpoint.port}" if endpoint.port != 22 else ""
    messages = {
        "host-key-changed": (
            f"The host key for {target} has CHANGED since the last connection. This could be a "
            "man-in-the-middle attack, or the server was reinstalled. SSH refused to connect. Verify the "
            f"change is expected, then remove the old key with `ssh-keygen -R {endpoint.host}` and reconnect."
            f"\n\n{detail}"),
        "auth-failed": (
            f"SSH authentication to {target} failed. Hermes runs ssh non-interactively (BatchMode), so a "
            "key requiring a passphrase or 2FA must be loaded into ssh-agent first, or set an IdentityFile "
            f"in ~/.ssh/config. Original error: {detail}"),
        "interactive-auth": (
            f"Tailscale SSH requires an interactive browser check for {target}. In a terminal, run "
            f"`ssh{port_arg} {target} true`, complete the check, then retry."),
        "unreachable": (f"Could not reach {target} over SSH. Check the host, port, and network. "
                        f"Original error: {detail}"),
        "timeout": (f"SSH operation to {target} timed out. The connection may be half-open; "
                    "it will be re-established on the next attempt."),
    }
    return messages.get(kind, f"SSH error connecting to {target}: {detail or 'unknown failure'}")


@dataclass(frozen=True)
class SshResult:
    code: int | None
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        # A negative returncode is a signal death: never success.
        return self.code == 0


class _SshTimeout(Exception):
    def __init__(self, stderr: str):
        super().__init__("ssh timed out")
        # Only the classification survives: a Tailscale check line carries a one-time URL.
        self.stderr_kind = classify_ssh_error(stderr)


def run_ssh(ssh_binary: str, args: list[str], *, timeout_s: float, stdin_data: str | None = None) -> SshResult:
    try:
        proc = subprocess.run(
            [ssh_binary, *args], input=stdin_data.encode() if stdin_data is not None else None,
            stdin=subprocess.DEVNULL if stdin_data is None else None, capture_output=True,
            timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise _SshTimeout((exc.stderr or b"").decode(errors="replace")) from exc
    return SshResult(proc.returncode, proc.stdout.decode(errors="replace"), proc.stderr.decode(errors="replace"))


# ---------------------------------------------------------------------------
# SshConnection
# ---------------------------------------------------------------------------


class SshConnection:
    """One ControlMaster connection: open/exec/forward/cancel_forward/close/is_alive."""

    def __init__(self, endpoint: SshEndpoint, *, ssh_binary: str,
                 connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
                 exec_timeout_s: float = DEFAULT_EXEC_TIMEOUT_S,
                 forward_timeout_s: float = DEFAULT_FORWARD_TIMEOUT_S):
        validate_ssh_target(endpoint.host, endpoint.user, endpoint.port)
        validate_key_path(endpoint.key_path)
        if not endpoint.control_path:
            raise _reject("SshConnection requires a ControlMaster socket path.")
        self.endpoint = endpoint
        self.ssh_binary = ssh_binary
        self._connect_timeout_s = connect_timeout_s
        self._exec_timeout_s = exec_timeout_s
        self._forward_timeout_s = forward_timeout_s
        self._posix_remote: bool | None = None
        self._forwarded: set[str] = set()
        self._opened = False
        self._keepalive_stop: threading.Event | None = None

    def _log(self, message: str) -> None:
        logger.info(redact_secrets(f"[ssh] {message}"))

    def _fail(self, raw: SshResult | _SshTimeout, fallback_kind: str = "unknown") -> SshReachError:
        if isinstance(raw, _SshTimeout):
            kind = "interactive-auth" if raw.stderr_kind == "interactive-auth" else "timeout"
            return SshReachError(kind, ssh_error_message(kind, self.endpoint))
        stderr = raw.stderr.strip()
        if raw.code is not None and raw.code < 0 and not stderr:
            # A signal death with empty stderr is a local process death, not proof the host is down.
            return SshReachError("unknown", ssh_error_message("unknown", self.endpoint,
                                                              f"ssh process exited from signal {-raw.code}"))
        kind = classify_ssh_error(stderr) if stderr else fallback_kind
        return SshReachError(kind, ssh_error_message(kind, self.endpoint, stderr))

    def _run(self, args: list[str], timeout_s: float, stdin_data: str | None = None) -> SshResult:
        return run_ssh(self.ssh_binary, args, timeout_s=timeout_s, stdin_data=stdin_data)

    def _run_ok(self, args: list[str], timeout_s: float) -> bool:
        try:
            return self._run(args, timeout_s).ok
        except _SshTimeout:
            return False

    def open(self) -> None:
        """Start the persistent master (a live, exec-verified master is reused)."""
        prepare_control_dir(Path(self.endpoint.control_path).parent)
        if self.is_alive():
            # -O check passing is not proof the master works: a ControlPersist master can survive a
            # failed teardown with wedged channels. Verify with a real exec, else evict and redial.
            if self._run_ok(build_exec_args(self.endpoint, "exit 0", self._connect_timeout_s),
                            self._connect_timeout_s):
                self._opened = True
                return
            self._log("existing control master failed exec verification; evicting stale master")
            self._evict_stale_master()
        self._log(f"opening control master to {self.endpoint.target}:{self.endpoint.port}")
        try:
            result = self._run(build_master_args(self.endpoint, self._connect_timeout_s), self._connect_timeout_s)
        except _SshTimeout as exc:
            raise self._connect_failed(exc) from None
        if not result.ok:
            raise self._connect_failed(result)
        self._opened = True
        self._log("control master established")

    def _connect_failed(self, raw: SshResult | _SshTimeout) -> SshReachError:
        err = self._fail(raw, "unreachable")
        if isinstance(raw, SshResult):
            self._log(f"connect to {self.endpoint.target}:{self.endpoint.port} failed (kind={err.kind}, "
                      f"exit={raw.code}): {raw.stderr.strip()[-500:] or '(empty)'}")
        return err

    def is_alive(self) -> bool:
        return self._run_ok(build_control_args(self.endpoint, "check", (), self._connect_timeout_s),
                            self._connect_timeout_s)

    def _evict_stale_master(self) -> None:
        # -O exit, then drop the socket so ControlMaster=auto cannot re-attach to the corpse.
        self._run_ok(build_control_args(self.endpoint, "exit", (), self._connect_timeout_s), self._connect_timeout_s)
        Path(self.endpoint.control_path).unlink(missing_ok=True)

    def _probe_posix(self, timeout_s: float) -> None:
        try:
            probe = self._run(build_exec_args(self.endpoint, "uname -s", self._connect_timeout_s), timeout_s)
        except _SshTimeout:
            return  # transport failure: leave uncached, send this payload raw
        if probe.code != 255:
            self._posix_remote = probe.code == 0 and probe.stdout.strip() in ("Linux", "Darwin")

    def exec(self, remote_command: str, *, timeout_s: float | None = None, stdin_data: str | None = None) -> str:
        """Run one remote command over the master; stdout, or a classified SshReachError.

        sshd hands the command to the user's LOGIN shell; payloads are POSIX sh (fish rejects
        ``help="$(…)"``), so Linux/Darwin remotes get them wrapped in ``sh -c``."""
        budget = timeout_s if timeout_s is not None else self._exec_timeout_s
        if self._posix_remote is None:
            self._probe_posix(budget)
        command = f"sh -c {login_shell_quote(remote_command)}" if self._posix_remote else remote_command
        try:
            result = self._run(build_exec_args(self.endpoint, command, self._connect_timeout_s), budget, stdin_data)
        except _SshTimeout as exc:
            raise self._fail(exc) from None
        if not result.ok:
            raise self._fail(result)
        return result.stdout

    def forward(self, local_port: int, remote_port: int, remote_host: str = "127.0.0.1") -> None:
        spec = forward_spec(local_port, remote_port, remote_host)
        self._log(f"forwarding 127.0.0.1:{local_port} -> {remote_host}:{remote_port}")
        try:
            result = self._run(build_control_args(self.endpoint, "forward", ("-L", spec), self._connect_timeout_s),
                               self._forward_timeout_s)
        except _SshTimeout as exc:
            raise self._fail(exc) from None
        if not result.ok:
            raise self._fail(result)
        self._forwarded.add(spec)
        self._start_keepalive()

    def cancel_forward(self, local_port: int, remote_port: int, remote_host: str = "127.0.0.1") -> None:
        """Best-effort: close tears everything down anyway."""
        spec = forward_spec(local_port, remote_port, remote_host)
        if not self._run_ok(build_control_args(self.endpoint, "cancel", ("-L", spec), self._connect_timeout_s),
                            self._forward_timeout_s):
            self._log(f"cancel forward 127.0.0.1:{local_port} failed (ignored)")
        self._forwarded.discard(spec)
        if not self._forwarded:
            self._stop_keepalive()

    def _start_keepalive(self) -> None:
        if self._keepalive_stop is not None:
            return
        stop = threading.Event()
        self._keepalive_stop = stop

        def _refresh() -> None:
            # Remote liveness owns reconnect/teardown; a transient failed check is not expiry.
            while not stop.wait(CONTROL_FORWARD_KEEPALIVE_S):
                self.is_alive()

        from hermes_agent.agent.memory_provider import spawn_context_thread

        spawn_context_thread(_refresh, name="gateway-ssh-keepalive").start()

    def _stop_keepalive(self) -> None:
        if self._keepalive_stop is not None:
            self._keepalive_stop.set()
            self._keepalive_stop = None

    def close(self) -> None:
        """Exit the master (drops every forward with it). Never raises."""
        self._stop_keepalive()
        self._forwarded.clear()
        if not self._opened:
            return
        self._opened = False
        if self._run_ok(build_control_args(self.endpoint, "exit", (), self._connect_timeout_s),
                        self._connect_timeout_s):
            self._log("control master closed")
            return
        # A master that refuses -O exit is the wedge that poisons re-attach: disown its socket.
        self._log("close failed; removing control socket")
        Path(self.endpoint.control_path).unlink(missing_ok=True)


def pick_local_port() -> int:
    """A free 127.0.0.1 port. The release->forward TOCTOU window is retried upstream."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def resolve_ssh_binary() -> str:
    if sys.platform == "win32":
        raise SshReachError("unsupported-platform",
                            "Gateway SSH connections need OpenSSH ControlMaster, which Windows does not support.")
    from hermes_agent.hermes_platform.resolver import locate_command

    found = locate_command("ssh").command
    if not found:
        raise SshReachError("unknown", "The OpenSSH client (`ssh`) is not installed on the gateway host.")
    return found[0]
