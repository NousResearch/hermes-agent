"""Remote ``hermes serve`` lifecycle over an SSH ControlMaster (Linux/macOS remotes).

Server-side port of ``apps/desktop/electron/remote-lifecycle.ts``: locate the remote Hermes, gate
the platform, refuse while a remote install update is in progress, reuse a provably-owned
``serve --isolated`` through its lockfile + an authenticated ownership proof, otherwise spawn a
fresh detached one under the update-marker mutex and scrape its READY port, and reap a stale
backend only when its argv proves it is ours.

The remote paths below (``~/.hermes/desktop-ssh/...``) are REMOTE path strings: the remote
``serve`` validates ``--ssh-session-token-file`` against ``$HOME/.hermes/desktop-ssh`` on its own
host (``hermes_cli/main_dashboard.py::_read_ssh_session_token_file``).
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from hermes_cli.gateway_connections_ssh_programs import REMOTE_MARKER_GATE_PY
from hermes_cli.gateway_connections_ssh_transport import SshConnection, SshReachError, with_remote_timeout

LOCKFILE_SCHEMA_VERSION = 2
# Bumped when the client<->serve reuse contract changes so an old serve is never reattached.
PROTOCOL_VERSION = 1
# The remote log is `>> log 2>&1`: merged output, not line-accurate (backend-ready.ts).
READY_RE = re.compile(r"(?<!\w)HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)")
REMOTE_LOCK_DIR = "~/.hermes/desktop-ssh"
SUPPORTED_REMOTE_OS = ("Linux", "Darwin")
# A busy remote's cold boot can take 60-150s before READY; a warm start announces in <1s.
DEFAULT_READY_TIMEOUT_S = 120.0
READY_POLL_INTERVAL_S = 0.75
# macOS sshd starts non-interactive shells with a 256-FD soft limit; raise only the child's.
REMOTE_NOFILE_SOFT_LIMIT = 65_536
PROBE_VERDICT_ATTEMPTS = 3
PROBE_VERDICT_RETRY_S = 0.5
_PROFILE_ID_ARG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_CREATION_TIME_RE = re.compile(r"^(?:linux:[0-9]+|darwin:[A-Za-z0-9 :+-]+)$")


def _err(kind: str, message: str) -> SshReachError:
    return SshReachError(kind, message)


# ---------------------------------------------------------------------------
# Identity, paths, quoting
# ---------------------------------------------------------------------------


def backend_profile_arg(profile: object) -> str | None:
    """A profile safe for ``--profile <name>`` (CLI-normalized), else None (profile-id-guard.ts)."""
    if not isinstance(profile, str):
        return None
    name = profile.strip().lower()
    if not name:
        return None
    return name if name == "default" or _PROFILE_ID_ARG_RE.match(name) else None


def mint_token() -> str:
    return secrets.token_hex(32)


def fingerprint_token(token: str) -> str:
    """The lockfile never stores the raw secret on the remote."""
    return hashlib.sha256(str(token or "").encode()).hexdigest()[:32]


def validate_ownership_id(ownership_id: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{32}", str(ownership_id or "")):
        raise _err("unknown", "SSH ownership ID is invalid.")
    return ownership_id


def validate_spawn_nonce(nonce: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{16}", str(nonce or "")):
        raise _err("unknown", "SSH spawn nonce is invalid.")
    return nonce


def ownership_directory(ownership_id: str) -> str:
    return f"{REMOTE_LOCK_DIR}/{validate_ownership_id(ownership_id)}"


def lockfile_path(ownership_id: str) -> str:
    return f"{ownership_directory(ownership_id)}/backend.lock.json"


def connect_reservation_path(ownership_id: str) -> str:
    return f"{ownership_directory(ownership_id)}/.connect.lock"


def spawn_log_path(ownership_id: str, nonce: str) -> str:
    return f"{ownership_directory(ownership_id)}/{validate_spawn_nonce(nonce)}.log"


def spawn_token_path(ownership_id: str, nonce: str) -> str:
    return f"{ownership_directory(ownership_id)}/{validate_spawn_nonce(nonce)}.token"


def shq(value: object) -> str:
    """POSIX single-quote for a remote shell word."""
    return "'" + str(value).replace("'", "'\\''") + "'"


def validate_remote_path(path: str) -> None:
    s = str(path or "")
    if not s:
        raise _err("unknown", "Remote path must not be empty.")
    if re.search(r"[\x00\n\r]", s):
        raise _err("unknown", "Unsafe remote path: contains NUL or newline.")
    if not (s == "~" or s.startswith("~/") or s.startswith("/")):
        raise _err("unknown", f'Remote path must be absolute or start with ~/: "{s}"')


def expand_remote_path(path: str) -> str:
    """A complete shell word: ``"$HOME"'/rest'`` for ``~/rest`` so $HOME expands remotely."""
    validate_remote_path(path)
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"' + shq(path[1:])
    return shq(path)


def assert_safe_remote_home(home: str) -> str:
    value = str(home or "").strip()
    if not re.fullmatch(r"(/|~/)[A-Za-z0-9._/+-]+", value) or ".." in value:
        raise _err("unknown", "Unsafe remote Hermes home.")
    return value.rstrip("/")


def remote_install_root(home: str) -> str:
    value = assert_safe_remote_home(home)
    profile = re.fullmatch(r"(.*)/profiles/[^/]+", value)
    return profile.group(1) if profile else value


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def probe_remote_platform(ssh: SshConnection) -> tuple[str, str]:
    out = ssh.exec("uname -s; uname -m").strip().split("\n")
    os_name = out[0].strip() if out else ""
    arch = out[1].strip() if len(out) > 1 else ""
    if os_name not in SUPPORTED_REMOTE_OS:
        raise _err("unsupported-platform",
                   f'Unsupported remote platform "{os_name or "unknown"}". Gateway SSH connections support '
                   "Linux and macOS remote hosts.")
    return os_name, arch


def probe_remote_hermes_home(ssh: SshConnection) -> str:
    try:
        out = ssh.exec('echo "${HERMES_HOME:-$HOME/.hermes}"').strip().split("\n")[-1]
    except SshReachError as exc:
        raise _err("transient-transport-error", "Could not resolve the remote Hermes home.") from exc
    return out or "~/.hermes"


def assert_remote_install_update_clear(ssh: SshConnection, hermes_home: str, hermes_path: str = "") -> None:
    """Refuse reuse/spawn while the remote install is being mutated (fail closed on any doubt).

    The gate travels on stdin (``python3 -``), not argv: over a mux socket a large argv can leave
    no buffer room for ssh's SCM_RIGHTS descriptor pass (EMSGSIZE on macOS's 8 KB buffers)."""
    marker = expand_remote_path(f"{remote_install_root(hermes_home)}/.hermes-update-in-progress")
    hermes = f" '' {expand_remote_path(hermes_path)}" if hermes_path else ""
    try:
        lines = ssh.exec(f"python3 - {marker}{hermes}", stdin_data=REMOTE_MARKER_GATE_PY).strip().splitlines()
    except SshReachError as exc:
        raise _err("update-in-progress",
                   "Could not prove that the remote Hermes install is clear for SSH startup.") from exc
    observation = lines[-1] if lines else ""
    if observation == "CLEAR":
        return
    live = re.fullmatch(r"LIVE:([1-9][0-9]*)", observation)
    if live:
        message = f"Remote Hermes update process {live.group(1)} is still running; SSH startup is paused."
    elif observation == "HELD":
        message = "A process the remote Hermes update started still holds the install; SSH startup is paused."
    else:
        message = "The remote Hermes update marker is unreadable, malformed, or locked; refusing SSH startup."
    raise _err("update-in-progress", message)


def _is_executable(ssh: SshConnection, candidate: str) -> bool:
    try:
        validate_remote_path(candidate)
        return ssh.exec(f"[ -x {expand_remote_path(candidate)} ] && echo OK || true").strip() == "OK"
    except SshReachError:
        return False


def locate_hermes(ssh: SshConnection, remote_hermes_path: str) -> str:
    """An explicit path is honored strictly; a blank one auto-detects (login shell, then installs)."""
    if remote_hermes_path:
        if _is_executable(ssh, remote_hermes_path):
            return remote_hermes_path
        raise _err("hermes-not-found",
                   f'The Hermes path set for this connection is not an executable on the remote host: '
                   f'"{remote_hermes_path}". It must be the full path to the remote `hermes` binary, or blank '
                   "to auto-detect.")
    candidates: list[str] = []
    try:
        found = ssh.exec(f"bash -lc {shq('command -v hermes')}").strip()
        if found:
            candidates.append(found.split("\n")[-1].strip())
    except SshReachError:
        pass
    # The installer's command locations (scripts/install.sh): per-user, root/FHS, legacy venv.
    candidates += ["~/.local/bin/hermes", "/usr/local/bin/hermes", "~/.hermes/hermes-agent/venv/bin/hermes"]
    for candidate in candidates:
        if candidate and _is_executable(ssh, candidate):
            return candidate
    raise _err("hermes-not-found",
               "Hermes is not installed on the remote host (no `hermes` executable found). Install it there, "
               "or set the Hermes path explicitly on the connection.")


def probe_hermes_version(ssh: SshConnection, hermes_path: str) -> str:
    try:
        out = ssh.exec(with_remote_timeout(f"{expand_remote_path(hermes_path)} --version 2>&1")).strip()
    except SshReachError:
        return ""
    return out.split("\n")[0].strip()


def remote_supports_ssh_ownership(ssh: SshConnection, hermes_path: str) -> bool:
    # The `$( (` space is load-bearing: `$((` would parse as arithmetic expansion.
    hermes = expand_remote_path(hermes_path)
    out = ssh.exec(
        f'help="$( {with_remote_timeout(f"{hermes} serve --help 2>&1")} )"; '
        "printf '%s' \"$help\" | grep -q ssh-session-token-file && "
        "printf '%s' \"$help\" | grep -q ssh-owner-nonce && echo YES || echo NO")
    return out.strip().endswith("YES")


# ---------------------------------------------------------------------------
# Lockfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LockfileSkew:
    """A lockfile that EXISTS but is not this lineage's record (fork build, corruption).

    Never conflated with "no lockfile": every reap/cleanup path skips on skew, because killing or
    overwriting foreign state murders a tunnel some other client depends on."""

    reason: str


_LOCK_STR_FIELDS = ("profile", "hermesPath", "hermesHome", "logPath", "startedAt")


def _valid_int(value: object, lo: int, hi: int) -> bool:
    return type(value) is int and lo <= value <= hi


def parse_lockfile(text: str, ownership_id: str) -> dict | LockfileSkew | None:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return LockfileSkew("unparseable-json")
    if not isinstance(parsed, dict):
        return LockfileSkew("non-object")
    if parsed.get("schemaVersion") != LOCKFILE_SCHEMA_VERSION or type(parsed.get("schemaVersion")) is not int:
        return LockfileSkew(f"schema-version {json.dumps(parsed.get('schemaVersion'))}")
    if not _valid_int(parsed.get("pid"), 1, 4194304):
        return LockfileSkew("malformed-pid")
    # port 0 = spawn-in-progress record: valid ownership proof for cleanup, never reusable.
    if not _valid_int(parsed.get("port"), 0, 65535):
        return LockfileSkew("malformed-port")
    nonce = parsed.get("spawnNonce")
    if parsed.get("ownershipId") != ownership_id or not isinstance(nonce, str) or not re.fullmatch(r"[0-9a-f]{16}", nonce):
        return LockfileSkew("ownership-mismatch")
    fp = parsed.get("tokenFingerprint")
    if not isinstance(fp, str) or not re.fullmatch(r"[0-9a-f]{32}", fp):
        return LockfileSkew("malformed-token-fingerprint")
    if parsed.get("protocolVersion") != PROTOCOL_VERSION:
        # Our own lineage, protocol-incompatible: not reusable; routes connect to a fresh spawn.
        return None
    if parsed.get("logPath") != spawn_log_path(ownership_id, nonce):
        return LockfileSkew("log-path-mismatch")
    for name in _LOCK_STR_FIELDS:
        if not isinstance(parsed.get(name), str) or len(parsed[name]) > 1024:
            return LockfileSkew(f"malformed-field {name}")
    ct = parsed.get("creationTime")
    if ct is not None and (not isinstance(ct, str) or not _CREATION_TIME_RE.match(ct)):
        return None
    return parsed


def read_lockfile(ssh: SshConnection, ownership_id: str) -> dict | LockfileSkew | None:
    lpath = expand_remote_path(lockfile_path(ownership_id))
    try:
        raw = ssh.exec(f"if [ ! -e {lpath} ]; then exit 0; fi; cat {lpath}")
    except SshReachError as exc:
        raise _err("transient-transport-error", "Could not read the SSH backend ownership record.") from exc
    return parse_lockfile(raw, ownership_id)


def _lock_json(lock: dict) -> str:
    return json.dumps(lock, separators=(",", ":"), ensure_ascii=False)


def write_lockfile(ssh: SshConnection, ownership_id: str, lock: dict) -> None:
    directory = ownership_directory(ownership_id)
    temporary = f"{directory}/.{secrets.token_hex(8)}.lock.tmp"
    payload = _lock_json({**lock, "schemaVersion": LOCKFILE_SCHEMA_VERSION})
    ssh.exec(f"umask 077 && mkdir -p {expand_remote_path(directory)} && "
             f"printf '%s' {shq(payload)} > {expand_remote_path(temporary)} && "
             f"mv -f {expand_remote_path(temporary)} {expand_remote_path(lockfile_path(ownership_id))}")


def remove_lockfile(ssh: SshConnection, ownership_id: str) -> None:
    try:
        ssh.exec(f"rm -f {expand_remote_path(lockfile_path(ownership_id))}")
    except SshReachError:
        pass


# ---------------------------------------------------------------------------
# Process identity
# ---------------------------------------------------------------------------


def exec_probe_verdict(ssh: SshConnection, command: str, sentinels: tuple[str, ...], failure: str) -> str:
    """An exec that answers with neither sentinel (a channel dying mid-teardown) is indeterminate,
    not the negative verdict: retry briefly, then fail closed so callers keep the record."""
    for attempt in range(PROBE_VERDICT_ATTEMPTS):
        if attempt:
            time.sleep(PROBE_VERDICT_RETRY_S)
        try:
            out = ssh.exec(command).strip()
        except SshReachError as exc:
            raise _err("transient-transport-error", failure) from exc
        if out in sentinels:
            return out
    raise _err("transient-transport-error", failure)


def remote_pid_alive(ssh: SshConnection, pid: object) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    return exec_probe_verdict(ssh, f"kill -0 {pid} 2>/dev/null && echo ALIVE || echo DEAD",
                              ("ALIVE", "DEAD"), "Could not verify the SSH backend process.") == "ALIVE"


_CREATION_TIME_PY = """import subprocess,sys
pid={pid}
value=""
if sys.platform.startswith("linux"):
 try:
  raw=open(f"/proc/{{pid}}/stat","r",encoding="ascii").read()
  fields=raw[raw.rfind(")")+2:].split()
  value="linux:"+fields[19]
 except (OSError,IndexError,UnicodeError):pass
elif sys.platform=="darwin":
 try:
  started=subprocess.check_output(["ps","-o","lstart=","-p",str(pid)],text=True).strip()
  if started:value="darwin:"+started
 except (OSError,subprocess.CalledProcessError):pass
print(value)"""


def remote_process_creation_time(ssh: SshConnection, pid: int) -> str:
    """Kernel start identity fencing PID reuse ('' when unprovable on an unusual POSIX host)."""
    if type(pid) is not int or pid <= 0:
        return ""
    try:
        value = ssh.exec(f"python3 -c {shq(_CREATION_TIME_PY.format(pid=pid))}").strip()
    except SshReachError:
        return ""
    return value if _CREATION_TIME_RE.match(value) else ""


_OWNED_PY = r"""import os,shlex,subprocess,sys
pid={pid}
expected=os.path.expanduser({hermes_path})
hermes_home=os.path.expanduser({hermes_home}) if {hermes_home} else ""
expected_entries={{expected}}
if hermes_home:
 expected_entries.add(os.path.join(hermes_home,"hermes-agent","venv","bin","hermes"))
expected_token=os.path.expanduser({token_path})
expected_profile={profile}
nonce={nonce}
try:
 raw=open(f"/proc/{{pid}}/cmdline","rb").read()
 args=[x.decode("utf-8","surrogateescape") for x in raw.split(b"\0") if x]
except OSError:
 try:
  line=subprocess.check_output(["ps","-ww","-o","command=","-p",str(pid)],text=True).strip()
 except subprocess.CalledProcessError:
  # pid already gone — a dead process is FOREIGN, not a transport error
  print("FOREIGN");sys.exit(0)
 # ps joins argv with spaces and no quoting (macOS has no /proc), so a quote or space in the
 # Hermes path breaks shlex: cut the known entry path out before splitting the rest.
 head,rest=None,line
 for e in sorted(expected_entries,key=len,reverse=True):
  at=0 if line.startswith(e+" ") else line.find(" "+e+" ")+1
  if at>0 or line.startswith(e+" "):
   head,rest=line[:at].split()+[e],line[at+len(e)+1:]
   break
 try:
  args=(head or [])+shlex.split(rest)
 except ValueError:
  args=(head or [])+rest.split()
ok=False
try:
 profile_arg=args.index("--profile") if expected_profile else -1
 pval=profile_arg+1 if profile_arg>=0 else -1
 serve_pos=[i for i,a in enumerate(args) if a=="serve" and i!=pval]
 serve=serve_pos[0]
 owner=args.index("--ssh-owner-nonce",serve+1)
 token=args.index("--ssh-session-token-file",serve+1) if expected_token else -1
 isolated=args.index("--isolated",serve+1)
 serve_count=len(serve_pos)
 owner_count=args.count("--ssh-owner-nonce")
 token_count=args.count("--ssh-session-token-file")
 isolated_count=args.count("--isolated")
 profile_count=args.count("--profile")
 direct=args[0] in expected_entries
 python_entry=len(args)>1 and args[1] in expected_entries and os.path.basename(args[0]).startswith("python")
 token_ok=not expected_token or args[token+1]==expected_token
 isolated_ok=isolated_count==1 and isolated>serve
 profile_ok=(profile_count==1 and profile_arg<serve and args[profile_arg+1]==expected_profile) if expected_profile else profile_count==0
 spawn_proof=bool(expected_token) and owner_count==1 and token_count==1 and token_ok and profile_ok
 ok=(direct or python_entry or spawn_proof) and serve_count==1 and isolated_ok and owner_count==1 and args[owner+1]==nonce and token_ok and profile_ok
except (ValueError,IndexError):pass
print("OWNED" if ok else "FOREIGN")"""


def pid_is_our_serve(ssh: SshConnection, lock: dict, ownership_id: str) -> bool:
    """Provably ours only when the live argv carries our exact serve args + random nonce.

    A profile literally named ``serve`` puts its value before the subcommand, so the ``serve``
    index/count skip the ``--profile`` value token."""
    pid, nonce, hermes_path = lock.get("pid"), str(lock.get("spawnNonce") or ""), lock.get("hermesPath") or ""
    if not pid or not re.fullmatch(r"[0-9a-f]{16}", nonce) or not hermes_path:
        return False
    # Values enter the Python source as Python literals; shell quoting wraps only the whole script.
    script = _OWNED_PY.format(
        pid=int(pid), hermes_path=repr(str(hermes_path)), hermes_home=repr(str(lock.get("hermesHome") or "")),
        token_path=repr(spawn_token_path(ownership_id, nonce)), profile=repr(str(lock.get("profile") or "")),
        nonce=repr(nonce))
    return exec_probe_verdict(ssh, f"python3 -c {shq(script)}", ("OWNED", "FOREIGN"),
                              "Could not verify SSH backend process ownership.") == "OWNED"


def _terminate(ssh: SshConnection, pid: int) -> None:
    try:
        ssh.exec(f"kill {pid} && i=0; while kill -0 {pid} 2>/dev/null; do "
                 'i=$((i+1)); [ "$i" -ge 50 ] && exit 1; sleep 0.1; done')
        return
    except SshReachError:
        pass
    # A serve mid-turn can ride out SIGTERM past the graceful wait; giving up would reparent it to
    # pid 1 (a leak). Escalate and require a confirmed exit before the record counts as reclaimed.
    try:
        ssh.exec(f"kill -9 {pid} 2>/dev/null; i=0; while kill -0 {pid} 2>/dev/null; do "
                 'i=$((i+1)); [ "$i" -ge 20 ] && exit 1; sleep 0.1; done')
    except SshReachError as exc:
        raise _err("transient-transport-error", "Could not terminate the stale SSH backend.") from exc


def cleanup_stale(ssh: SshConnection, ownership_id: str, lock: dict | LockfileSkew | None,
                  pid_alive: bool = True) -> None:
    """Kill the stale serve ONLY if provably ours, then drop its log and the lockfile."""
    if isinstance(lock, LockfileSkew):
        return
    if lock and pid_alive and pid_is_our_serve(ssh, lock, ownership_id):
        _terminate(ssh, int(lock["pid"]))
    nonce = lock.get("spawnNonce") if lock else None
    if lock and nonce and lock.get("logPath") == spawn_log_path(ownership_id, nonce):
        try:
            ssh.exec(f"rm -f {expand_remote_path(lock['logPath'])}")
        except SshReachError:
            pass
    remove_lockfile(ssh, ownership_id)


def disconnect_remote(ssh: SshConnection, ownership_id: str) -> bool:
    """Kill our provably-owned serve before closing the transport: spawn detaches with
    setsid/nohup, so closing the tunnel alone leaves the serve at pid 1. True when a record was reclaimed."""
    lock = read_lockfile(ssh, ownership_id)
    if not lock or isinstance(lock, LockfileSkew):
        return False
    cleanup_stale(ssh, ownership_id, lock, remote_pid_alive(ssh, lock["pid"]))
    return True


# ---------------------------------------------------------------------------
# Spawn
# ---------------------------------------------------------------------------


def with_remote_update_mutex(command: str, marker_word: str, hermes_word: str) -> str:
    """Run ``command`` holding the updaters' marker lock (REMOTE_MARKER_GATE_PY). ``marker_word``
    and ``hermes_word`` are expand_remote_path() output, embedded raw so $HOME expands remotely."""
    return f"python3 -c {shq(REMOTE_MARKER_GATE_PY)} {marker_word} {shq(command)} {hermes_word}"


@dataclass(frozen=True)
class SpawnPlan:
    hermes_path: str
    profile: str
    spawn_nonce: str
    token_file_path: str
    log_path: str
    hermes_home: str
    ownership_id: str
    lock_metadata: dict = field(default_factory=dict)


def _spawn_payload_prefix(plan: SpawnPlan) -> tuple[str, str, str]:
    hermes = expand_remote_path(plan.hermes_path)
    pinned = backend_profile_arg(plan.profile)
    profile_args = f"--profile {shq(pinned)} " if pinned else ""
    log_path = expand_remote_path(plan.log_path)
    sub = (f"serve --isolated --host 127.0.0.1 --port 0 --ssh-session-token-file "
           f"{expand_remote_path(plan.token_file_path)} --ssh-owner-nonce {validate_spawn_nonce(plan.spawn_nonce)}")
    # A value the remote login shell exports is cleared first; gateway connections never hand a
    # dev renderer origin on (Desktop's HERMES_DESKTOP_DEV_SERVER is always empty here).
    dash = (f"ulimit -n {REMOTE_NOFILE_SOFT_LIMIT} 2>/dev/null || true; unset HERMES_DESKTOP_DEV_SERVER; "
            f"exec env HERMES_DESKTOP=1 {hermes} {profile_args}{sub}")
    detached_shell = f'eval "exec $1>&-"; {dash} </dev/null >> {log_path} 2>&1 & echo $!'
    # setsid (Linux) starts a new session; macOS has no setsid, so nohup. The inner shell
    # backgrounds Hermes and reports ITS pid, which is what the lock publishes.
    detached_spawn = (f'child=$("$(command -v setsid || echo nohup)" sh -c {shq(detached_shell)} '
                      'hermes-update-child "$1")')
    return hermes, log_path, detached_spawn


def build_spawn_command(plan: SpawnPlan) -> str:
    """The marker judge, ownership reservation, process creation and initial lockfile publication
    as ONE remote command under the marker lock, so a second client never sees an empty lock and
    spawns beside this one. A marker present after the spawn came from a writer that skipped the
    lock, so the child is reaped (exit 75). The reservation is an atomic mkdir reclaimed only when
    its owning remote shell is dead. Plain sh (dash): no ``${var//pat/rep}``."""
    hermes, log_path, detached_spawn = _spawn_payload_prefix(plan)
    marker = expand_remote_path(f"{remote_install_root(plan.hermes_home or '~/.hermes')}/.hermes-update-in-progress")
    marker_absent = f"[ ! -e {marker} ]"
    reap = 'kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true;'
    reservation = expand_remote_path(connect_reservation_path(plan.ownership_id))
    lock_path = expand_remote_path(lockfile_path(plan.ownership_id))
    token_path = expand_remote_path(plan.token_file_path)
    metadata = _lock_json({"schemaVersion": LOCKFILE_SCHEMA_VERSION,
                           **plan.lock_metadata, "profile": backend_profile_arg(plan.profile) or "",
                           "pid": "__PID__"})
    nonce = validate_spawn_nonce(plan.spawn_nonce)
    payload = (
        f'(umask 077 && mkdir -p "$(dirname {reservation})"); '
        f"reservation={reservation}; lock={lock_path}; owner_file={reservation}/owner; "
        f"reservation_nonce={shq(nonce)}; "
        'i=0; while ! mkdir "$reservation" 2>/dev/null; do '
        'owner_data=$(cat "$owner_file" 2>/dev/null || true); owner_pid=${owner_data%%:*}; '
        "case \"$owner_pid\" in ''|*[!0-9]*) ;; *) kill -0 \"$owner_pid\" 2>/dev/null || "
        '{ rm -rf "$reservation"; continue; };; esac; '
        'i=$((i+1)); [ "$i" -ge 600 ] && exit 75; sleep 0.05; done; '
        "printf '%s:%s' \"$$\" \"$reservation_nonce\" > \"$owner_file\"; "
        "trap 'rm -rf \"$reservation\"' EXIT; "
        'if [ -f "$lock" ]; then '
        "existing_pid=$(sed -n 's/.*\"pid\":\\([0-9][0-9]*\\).*/\\1/p' \"$lock\" | head -n 1); "
        "case \"$existing_pid\" in ''|*[!0-9]*) rm -f \"$lock\";; *) "
        f'if kill -0 "$existing_pid" 2>/dev/null; then rm -f {token_path}; printf EXISTING; exit 0; fi; '
        'rm -f "$lock";; esac; fi; '
        f'mkdir -p "$(dirname {log_path})" && '
        f"{detached_spawn}; "
        f"{marker_absent} || {{ {reap} exit 75; }}; "
        f"lock_json=$(printf '%s' {shq(metadata)} | sed \"s/\\\"pid\\\":\\\"__PID__\\\"/\\\"pid\\\":${{child}}/\"); "
        f'temporary_lock="${{lock}}.{nonce}.tmp"; '
        f"printf '%s' \"$lock_json\" > \"$temporary_lock\" && mv -f \"$temporary_lock\" \"$lock\" || {{ {reap} exit 76; }}; "
        'echo "$child"'
    )
    return with_remote_update_mutex(payload, marker, hermes)


_TOKEN_UPLOAD_PY = """import os,sys,stat
p=os.path.expanduser({path})
d=os.path.dirname(p)
n=os.path.basename(p)
os.makedirs(d,mode=0o700,exist_ok=True)
df=os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0)
dd=os.open(d,df)
try:
 s=os.fstat(dd)
 if not stat.S_ISDIR(s.st_mode):raise SystemExit("unsafe token directory")
 if hasattr(os,"getuid") and s.st_uid!=os.getuid():raise SystemExit("token directory owner mismatch")
 if (s.st_mode&0o777)!=0o700:os.fchmod(dd,0o700)
 fl=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0)
 now=__import__("time").time()
 for stale in os.listdir(dd):
  if stale.endswith(".token") and len(stale)==22:
   try:
    ss=os.stat(stale,dir_fd=dd,follow_symlinks=False)
    if stat.S_ISREG(ss.st_mode) and now-ss.st_mtime>3600:os.unlink(stale,dir_fd=dd)
   except OSError:pass
 fd=os.open(n,fl,0o600,dir_fd=dd)
 try:os.write(fd,sys.stdin.buffer.read())
 except BaseException:
  try:os.unlink(n,dir_fd=dd)
  except OSError:pass
  raise
 finally:os.close(fd)
finally:os.close(dd)"""


def _remove_remote(ssh: SshConnection, path: str) -> None:
    try:
        ssh.exec(f"rm -f {expand_remote_path(path)}")
    except SshReachError:
        pass


@dataclass(frozen=True)
class Spawned:
    existing: bool
    pid: int = 0
    spawn_nonce: str = ""
    log_path: str = ""
    token_file_path: str = ""


def spawn_remote_serve(ssh: SshConnection, *, hermes_path: str, profile: str, token: str, ownership_id: str,
                       hermes_home: str, assert_install_clear: Callable[[], None]) -> Spawned:
    if not remote_supports_ssh_ownership(ssh, hermes_path):
        raise _err("update-required", "The remote Hermes install does not support --ssh-session-token-file and "
                                      "--ssh-owner-nonce. Update Hermes on the remote host.")
    nonce = secrets.token_hex(8)
    plan = SpawnPlan(
        hermes_path=hermes_path, profile=profile, spawn_nonce=nonce, ownership_id=ownership_id,
        token_file_path=spawn_token_path(ownership_id, nonce), log_path=spawn_log_path(ownership_id, nonce),
        hermes_home=hermes_home,
        lock_metadata={"ownershipId": ownership_id, "spawnNonce": nonce, "port": 0, "profile": profile,
                       "hermesPath": hermes_path, "hermesHome": hermes_home,
                       "logPath": spawn_log_path(ownership_id, nonce), "tokenFingerprint": fingerprint_token(token),
                       "protocolVersion": PROTOCOL_VERSION, "devServer": "", "startedAt": _now_iso()})
    try:
        # The token travels on stdin, never argv.
        ssh.exec(f"python3 -c {shq(_TOKEN_UPLOAD_PY.format(path=repr(plan.token_file_path)))}", stdin_data=token)
        # Close the marker race after the token write and immediately before process creation.
        assert_install_clear()
        out = ssh.exec(build_spawn_command(plan))
    except SshReachError:
        _remove_remote(ssh, plan.token_file_path)
        raise
    lines = [line.strip() for line in out.strip().split("\n") if line.strip()]
    if lines and lines[-1] == "EXISTING":
        return Spawned(existing=True)
    if not lines or not lines[-1].isdigit() or int(lines[-1]) <= 0:
        _remove_remote(ssh, plan.token_file_path)
        raise _err("spawn-failed", "Failed to launch the remote Hermes serve (no pid returned).")
    return Spawned(False, int(lines[-1]), nonce, plan.log_path, plan.token_file_path)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def remote_profile_missing_error(output: str) -> SshReachError | None:
    match = re.search(r"""Error: Profile ['"]([^'"]+)['"] does not exist\.?(?: Create it with: ([^\n]+))?""",
                      str(output or ""), re.IGNORECASE)
    if not match:
        return None
    create = (match.group(2) or "").strip()
    hint = f" or create it with: {create}" if create else "."
    return _err("spawn-failed", f"The remote Hermes profile '{match.group(1)}' does not exist. "
                                f"Select an existing remote profile{hint}")


def scrape_ready_port(ssh: SshConnection, log_path: str, *, timeout_s: float = DEFAULT_READY_TIMEOUT_S,
                      is_alive: Callable[[], bool] | None = None) -> int:
    deadline = time.monotonic() + timeout_s
    remote_log = expand_remote_path(log_path)
    while time.monotonic() < deadline:
        try:
            tail = ssh.exec(f"cat {remote_log} 2>/dev/null || true")
        except SshReachError:
            tail = ""
        if is_alive is not None and not is_alive():
            raise remote_profile_missing_error(tail) or _err(
                "spawn-failed", "Remote Hermes serve exited before announcing its port.")
        match = READY_RE.search(tail)
        if match:
            return int(match.group(1))
        time.sleep(READY_POLL_INTERVAL_S)
    raise _err("ready-timeout", f"Timed out waiting for the remote Hermes serve to announce its port ({timeout_s:.0f}s).")


# ---------------------------------------------------------------------------
# Reuse proof + connect
# ---------------------------------------------------------------------------


def classify_ssh_reuse_proof(proof: Any, spawn_nonce: str) -> str:
    ok = (isinstance(proof, dict) and proof.get("ok") is True and proof.get("sshOwnerNonce") == spawn_nonce
          and proof.get("protocolVersion") == PROTOCOL_VERSION and proof.get("runtimeIntact") is not False)
    return "authenticated-ok" if ok else "authenticated-stale"


def lock_serves_connect(lock: dict, *, profile: str, reuse_token: str, hermes_path: str, hermes_home: str) -> bool:
    return (lock["port"] > 0 and lock["profile"] == profile and bool(reuse_token)
            and lock["tokenFingerprint"] == fingerprint_token(reuse_token)
            and lock["hermesPath"] == hermes_path and lock["hermesHome"] == hermes_home
            and (lock.get("devServer") or "") == "")


def is_forward_bind_collision(error: BaseException) -> bool:
    return bool(re.search(r"address already in use|cannot listen to port|bind.*failed", str(error), re.IGNORECASE))


@dataclass
class ConnectDeps:
    """What connect() needs beyond the opened SshConnection; the HTTP probes are injected so the
    facade owns httpx and tests can drive the lifecycle against a real sshd."""

    ssh: SshConnection
    ownership_id: str
    pick_local_port: Callable[[], int]
    wait_for_health: Callable[[str], None]
    probe_reuse_proof: Callable[[str, str, str], str]
    profile: str = ""
    remote_hermes_path: str = ""
    reuse_token: str = ""
    ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S


@dataclass(frozen=True)
class Connected:
    base_url: str
    token: str
    local_port: int
    remote_port: int
    pid: int
    reused: bool
    hermes_version: str
    hermes_path: str
    hermes_home: str
    spawn_nonce: str
    platform: tuple[str, str]


def open_forward(ssh: SshConnection, pick: Callable[[], int], remote_port: int, attempts: int = 3) -> int:
    for attempt in range(attempts):
        local_port = pick()
        try:
            ssh.forward(local_port, remote_port)
            return local_port
        except SshReachError as exc:
            if not is_forward_bind_collision(exc) or attempt == attempts - 1:
                raise
    raise AssertionError("unreachable")


def wait_for_remote_spawn_completion(ssh: SshConnection, ownership_id: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        lock = read_lockfile(ssh, ownership_id)
        if not lock or isinstance(lock, LockfileSkew):
            return False
        if lock["port"] > 0:
            return True
        time.sleep(READY_POLL_INTERVAL_S)
    raise _err("spawn-failed", "Timed out waiting for a concurrent SSH connection to publish its backend.")


@dataclass(frozen=True)
class _Remote:
    platform: tuple[str, str]
    hermes_home: str
    hermes_path: str
    hermes_version: str
    profile: str


def _skew_error(ownership_id: str, skew: LockfileSkew) -> SshReachError:
    path = lockfile_path(ownership_id)
    return _err("remote-lockfile-skew",
                f"The remote ownership record {path} does not match this Hermes build ({skew.reason}). It was "
                "probably written by a different build sharing this remote, or the file is corrupt. Refusing to "
                "reap or overwrite it. If nothing else uses this remote, delete that file there and reconnect.")


def _try_reuse(deps: ConnectDeps, remote: _Remote, lock: dict) -> Connected | None:
    """Attach to a live, provably-owned serve whose record serves this connect; None after reaping."""
    ssh, oid = deps.ssh, deps.ownership_id
    pid_alive = remote_pid_alive(ssh, lock["pid"])
    owned = pid_alive and pid_is_our_serve(ssh, lock, oid)
    reusable = owned and lock_serves_connect(lock, profile=remote.profile, reuse_token=deps.reuse_token,
                                             hermes_path=remote.hermes_path, hermes_home=remote.hermes_home)
    if not reusable:
        assert_remote_install_update_clear(ssh, remote.hermes_home, remote.hermes_path)
        cleanup_stale(ssh, oid, lock, pid_alive)
        return None
    if not lock.get("creationTime"):
        creation = remote_process_creation_time(ssh, lock["pid"])
        if creation:
            lock = {**lock, "creationTime": creation}
            write_lockfile(ssh, oid, lock)
    assert_remote_install_update_clear(ssh, remote.hermes_home, remote.hermes_path)
    local_port = open_forward(ssh, deps.pick_local_port, lock["port"])
    base_url = f"http://127.0.0.1:{local_port}"
    try:
        verdict = deps.probe_reuse_proof(base_url, deps.reuse_token, lock["spawnNonce"])
    except Exception as exc:
        ssh.cancel_forward(local_port, lock["port"])
        raise _err("transient-transport-error", "Could not verify the existing SSH backend.") from exc
    if verdict != "authenticated-ok":
        ssh.cancel_forward(local_port, lock["port"])
        assert_remote_install_update_clear(ssh, remote.hermes_home, remote.hermes_path)
        cleanup_stale(ssh, oid, lock)
        return None
    if not remote_pid_alive(ssh, lock["pid"]):
        ssh.cancel_forward(local_port, lock["port"])
        raise _err("spawn-failed", "The reused remote Hermes serve exited while it was being verified.")
    return Connected(base_url, deps.reuse_token, local_port, lock["port"], lock["pid"], True,
                     remote.hermes_version, remote.hermes_path, remote.hermes_home, lock["spawnNonce"],
                     remote.platform)


def _spawn_fresh(deps: ConnectDeps, remote: _Remote) -> Connected | None:
    """Spawn, scrape READY, forward, prove ownership, publish the port. None = a concurrent owner."""
    ssh, oid = deps.ssh, deps.ownership_id
    assert_remote_install_update_clear(ssh, remote.hermes_home, remote.hermes_path)
    token = mint_token()
    spawned = spawn_remote_serve(
        ssh, hermes_path=remote.hermes_path, profile=remote.profile, token=token, ownership_id=oid,
        hermes_home=remote.hermes_home,
        assert_install_clear=lambda: assert_remote_install_update_clear(ssh, remote.hermes_home, remote.hermes_path))
    if spawned.existing:
        if not deps.reuse_token:
            raise _err("remote-ownership-contended",
                       "Another SSH connection owns this remote serve; a session token is required to reuse it.")
        wait_for_remote_spawn_completion(ssh, oid, deps.ready_timeout_s)
        return None
    creation = remote_process_creation_time(ssh, spawned.pid)
    owned = {"ownershipId": oid, "spawnNonce": spawned.spawn_nonce, "pid": spawned.pid, "port": 0,
             "profile": remote.profile, "hermesPath": remote.hermes_path, "hermesHome": remote.hermes_home,
             "logPath": spawned.log_path, "tokenFingerprint": fingerprint_token(token),
             "protocolVersion": PROTOCOL_VERSION, "devServer": "", "startedAt": _now_iso(),
             **({"creationTime": creation} if creation else {})}
    local_port = remote_port = 0
    try:
        # Publish the record immediately (port=0) so an interrupted attempt leaves no lockless orphan.
        write_lockfile(ssh, oid, owned)
        remote_port = scrape_ready_port(ssh, spawned.log_path, timeout_s=deps.ready_timeout_s)
        local_port = open_forward(ssh, deps.pick_local_port, remote_port)
        base_url = f"http://127.0.0.1:{local_port}"
        deps.wait_for_health(base_url)
        if deps.probe_reuse_proof(base_url, token, spawned.spawn_nonce) != "authenticated-ok":
            raise _err("spawn-failed", "The remote serve on the announced port did not prove it is ours.")
        write_lockfile(ssh, oid, {**owned, "port": remote_port})
    except BaseException as exc:
        _rollback_spawn(ssh, oid, owned, spawned, local_port, remote_port, exc)
        raise
    return Connected(base_url, token, local_port, remote_port, spawned.pid, False, remote.hermes_version,
                     remote.hermes_path, remote.hermes_home, spawned.spawn_nonce, remote.platform)


def _rollback_spawn(ssh: SshConnection, oid: str, owned: dict, spawned: Spawned, local_port: int,
                    remote_port: int, error: BaseException) -> None:
    if local_port and remote_port:
        ssh.cancel_forward(local_port, remote_port)
    _remove_remote(ssh, spawned.token_file_path)
    # This record IS the child this attempt spawned: an unsettled liveness probe means "assume
    # alive" so the ownership proof re-runs and keeps the record when nothing can be proven.
    try:
        alive = remote_pid_alive(ssh, spawned.pid)
    except SshReachError:
        alive = True
    try:
        cleanup_stale(ssh, oid, owned, alive)
    except SshReachError as cleanup_error:
        error.__context__ = cleanup_error


def _probe_remote(deps: ConnectDeps) -> _Remote:
    ssh = deps.ssh
    platform = probe_remote_platform(ssh)
    hermes_home = probe_remote_hermes_home(ssh)
    assert_remote_install_update_clear(ssh, hermes_home)
    hermes_path = locate_hermes(ssh, deps.remote_hermes_path)
    return _Remote(platform, hermes_home, hermes_path, probe_hermes_version(ssh, hermes_path),
                   backend_profile_arg(deps.profile) or "")


def connect(deps: ConnectDeps) -> Connected:
    """Establish (or reuse) a remote serve and a tunnel to it over ``deps.ssh`` (already open)."""
    remote = _probe_remote(deps)
    for _ in range(3):
        lock = read_lockfile(deps.ssh, deps.ownership_id)
        if isinstance(lock, LockfileSkew):
            # Fail closed: no reap, no removal, no overwrite, no spawn on top of foreign state.
            raise _skew_error(deps.ownership_id, lock)
        result = _try_reuse(deps, remote, lock) if lock else None
        result = result or _spawn_fresh(deps, remote)
        if result is not None:
            return result
    raise _err("remote-ownership-contended", "The remote serve kept changing owners while connecting.")
