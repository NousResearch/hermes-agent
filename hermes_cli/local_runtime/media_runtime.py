"""Lifecycle controller for exactly ONE media sidecar process tree.

``MediaRuntimeController`` owns one UnSloth-Studio-style backend: it takes a
residency grant from :mod:`hermes_cli.local_runtime.admission`, launches the
backend owned (own session/process group handed to the owner-death supervisor)
and credential-safe on an ephemeral ``127.0.0.1`` port, proves health, heart-
beats the lease via host-driven :meth:`MediaRuntimeController.tick` (NO hidden
thread), detects crashes, and stops the tree with a VERIFIED exit before the
grant is ever handed back.

Hard rules encoded here (each has a test in
``tests/hermes_cli/test_media_runtime.py``):

* An admission ``Refused`` changes nothing: no process, state stays ``stopped``,
  the refusal reason surfaces in ``detail``.
* Every start mints a fresh ``instance_id`` and a fresh random password. The
  password reaches the child through its STDIN ONLY — never argv, never env,
  never a log line, never ``RuntimeStatus``/repr/exception text.
* Spawning goes through ``spawn_server(owned=True)`` so the whole tree dies
  with this process by any cause (#126631 discipline).
* Startup failure (health timeout or early child exit) tears the tree down,
  VERIFIES exit, and only then releases the grant — an unverified exit leaves
  the grant live so it quarantines-by-lease instead of risking double residency.
* Teardown mirrors ``LlamaServerSupervisor``: enumerate children FIRST,
  terminate, escalate to kill, then drain/release the owned process group.
* A ``FencedError`` (another controller took over) stops only our own sidecar;
  the new holder's record is never mutated.
* The bound address the child reports must be ``127.0.0.1`` and never the
  shared ``:8888`` default; anything else fails the start.

The controller never downloads anything, never loads models, never calls a
generation endpoint (only ``GET /api/health``), and never persists the grant
process identity beyond what admission needs.
"""
from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from hermes_cli.local_runtime.admission import (
    AdmissionError,
    AdmissionStore,
    FencedError,
    Grant,
    Refused,
)
from hermes_cli.local_runtime.processes import spawn_server

logger = logging.getLogger(__name__)

SHARED_DEFAULT_PORT = 8888
LOOPBACK_HOST = "127.0.0.1"
HEALTH_PATH = "/api/health"
DEFAULT_RESOURCE = "media:studio"
DEFAULT_ENGINE = "unsloth-studio"

#: Names a strict disposable child environment may still inherit: locale/TLS
#: plumbing only (ported from the proven Studio boot probe contract).
_STRICT_KEEP = re.compile(
    r"^(PATH|TERM|LANG|LC_[A-Z_]+|SSL_CERT_FILE|SSL_CERT_DIR|CURL_CA_BUNDLE|PATH_CA_BUNDLE)$"
)

_RUNNING_RE = re.compile(r"running on https?://([^\s:/]+):(\d+)", re.IGNORECASE)
_TAURI_PORT_RE = re.compile(r"TAURI_PORT=(\d+)")

# Credential scrubbing for any text we might surface (ported from the probe's
# sanitize_credentials; replacements hold no further matchable material).
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}(?:\.[A-Za-z0-9_\-\.]+)?")
_BEARER_RE = re.compile(r"(?i)(authorization\s*:\s*bearer\s+)\S+")
_KV_RE = re.compile(
    r"(?i)([\"']?(?:password|passwd|passphrase|bootstrap_password|secret|api_key|apikey|"
    r"access_token|refresh_token|token)[\"']?\s*[:=]\s*)([\"']?)(\S+?)([\"']?)(?=[\s,;\"']|$)"
)
_PHRASE_RE = re.compile(r"(?i)(password\s+is\s+)(\S+)")

_STATES = ("stopped", "starting", "running", "stopping", "failed")


def sanitize_credentials(text: str) -> str:
    """Redact JWTs, bearer tokens, and key/value or prose password values."""
    text = _JWT_RE.sub("[REDACTED:token]", text)
    text = _BEARER_RE.sub(r"\1[REDACTED:token]", text)
    text = _KV_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]{m.group(4)}", text)
    text = _PHRASE_RE.sub(r"\1[REDACTED]", text)
    return text


@dataclass(frozen=True)
class RuntimeStatus:
    """Public observation of the controller. NEVER carries a credential."""

    state: str = "stopped"          # stopped | starting | running | stopping | failed
    instance_id: str | None = None  # fresh uuid4 hex per start
    pid: int | None = None          # sidecar pid (None once cleanly stopped)
    base_url: str | None = None     # loopback base URL while live
    epoch: int | None = None        # admission fencing epoch of our grant
    started_at: float | None = None  # wall clock of the successful start
    exit_code: int | None = None    # last observed child exit code
    detail: str = ""                # sanitized human-readable status


def default_boot_id() -> str:
    """Opaque current-boot identity for admission (injectable override)."""
    import psutil

    return str(psutil.boot_time())


def default_process_identity(pid: int) -> str:
    """Opaque creation identity for ``pid`` (injectable override)."""
    import psutil

    return str(psutil.Process(pid).create_time())


def free_loopback_port() -> int:
    """A kernel-chosen free TCP port bound to loopback; never :8888."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOOPBACK_HOST, 0))
        port = sock.getsockname()[1]
    if port == SHARED_DEFAULT_PORT:  # astronomically unlikely; contract is absolute
        return free_loopback_port()
    return port


def parse_bound_endpoint(log_text: str) -> tuple[str | None, int | None]:
    """``(host, port)`` the child reported it is running on, or ``(None, None)``.

    The proven contract prints ``Unsloth Studio running on
    http://127.0.0.1:<port>``; ``TAURI_PORT=<port>`` is the host-less fallback.
    """
    match = _RUNNING_RE.search(log_text or "")
    if match:
        return match.group(1), int(match.group(2))
    match = _TAURI_PORT_RE.search(log_text or "")
    if match:
        return None, int(match.group(1))
    return None, None


def _probe_health(url: str, timeout: float = 1.5) -> bool:
    """True on HTTP 200 from the health endpoint; anything else keeps polling."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except urllib.error.HTTPError:
        return False
    except (urllib.error.URLError, OSError):
        return False


def disposable_environment(data_root: Path) -> dict[str, str]:
    """Strict-allowlist disposable child env built under ``data_root``.

    Only locale/TLS plumbing is inherited; everything else (PYTHONPATH, every
    credential-shaped name, arbitrary app config) is dropped. HOME/XDG/TMPDIR
    and the HF cache with hard-offline flags all live under ``data_root`` so
    the sidecar can never steer or touch the user's real trees.
    """
    env = {name: value for name, value in os.environ.items()
           if _STRICT_KEEP.match(name) and value}
    env.setdefault("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
    home = data_root / "home"
    hf_cache = data_root / "hf-cache"
    for sub in ("tmp", "config", "cache", "state", "data"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    (hf_cache / "hub").mkdir(parents=True, exist_ok=True)
    env.update({
        "HOME": str(home),
        "TMPDIR": str(home / "tmp"),
        "XDG_CONFIG_HOME": str(home / "config"),
        "XDG_CACHE_HOME": str(home / "cache"),
        "XDG_STATE_HOME": str(home / "state"),
        "XDG_DATA_HOME": str(home / "data"),
        "UNSLOTH_STUDIO_HOME": str(data_root),
        "STUDIO_HOME": str(data_root),
        "UNSLOTH_STUDIO_PROJECTS_HOME": str(data_root / "projects"),
        "DATA_DESIGNER_HOME": str(data_root / "data-designer"),
        "HF_HOME": str(hf_cache),
        "HF_HUB_CACHE": str(hf_cache / "hub"),
        # Hard offline: this controller must never download; make any attempt
        # fail fast inside the child instead of fetching silently.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    return env


def _default_spawn(cmd, **kwargs):
    """Production seam: ``spawn_server`` keeps the POSIX/Windows ownership split."""
    return spawn_server(cmd, **kwargs)


def _quiet(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


class _StartupFailed(Exception):
    """Internal control flow: startup proof failed with a useful detail."""

    def __init__(self, detail: str, *, exit_code: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.exit_code = exit_code


class MediaRuntimeController:
    """Owns one media sidecar tree: grant -> launch -> health -> lease -> stop.

    The host calls :meth:`tick` periodically (no hidden thread): it renews the
    admission lease and detects child crashes. All teardown verifies process
    exit before the grant is released — the admission store itself never
    touches processes, so that verification lives here.
    """

    def __init__(self, *, owner_scope: str, data_root: Path | str,
                 admission: AdmissionStore | None = None, spawn=None,
                 boot_id_fn=None, identity_fn=None, clock=time.monotonic,
                 health_timeout: float = 60.0, lease_seconds: float = 30.0) -> None:
        self._owner_scope = owner_scope
        self._data_root = Path(data_root)
        self._admission = admission
        self._spawn = spawn or _default_spawn
        self._boot_id_fn = boot_id_fn or default_boot_id
        self._identity_fn = identity_fn or default_process_identity
        self._clock = clock
        self._health_timeout = float(health_timeout)
        self._lease_seconds = float(lease_seconds)
        # lifecycle state (in memory; durable truth lives in the admission store)
        self._state = "stopped"
        self._instance_id: str | None = None
        self._resource = DEFAULT_RESOURCE
        self._engine = DEFAULT_ENGINE
        self._model = ""
        self._grant: Grant | None = None
        self._proc: subprocess.Popen | None = None
        self._job = None                      # Windows job object from spawn_server
        self._pid: int | None = None
        self._child_identity: str | None = None
        self._base_url: str | None = None
        self._started_at: float | None = None
        self._exit_code: int | None = None
        self._detail = ""
        self._log_path: Path | None = None
        self._log_handle = None

    # -- public API ---------------------------------------------------------

    def start(self, *, python: str | Path, run_py: str | Path,
              resource: str = DEFAULT_RESOURCE, engine: str = DEFAULT_ENGINE,
              model: str = "") -> RuntimeStatus:
        """Acquire the grant, launch the sidecar owned+health-checked.

        Returns the resulting :class:`RuntimeStatus` (never raises for launch
        problems: failure lands in ``state='failed'`` with a sanitized detail).
        """
        if self._state in ("starting", "running", "stopping"):
            return self.status()
        self._resource, self._engine, self._model = resource, engine, model
        self._instance_id = uuid.uuid4().hex
        self._grant = None
        self._proc = None
        self._job = None
        self._pid = None
        self._child_identity = None
        self._base_url = None
        self._started_at = None
        self._exit_code = None
        self._detail = ""
        self._state = "starting"
        password = secrets.token_urlsafe(32)
        try:
            return self._start_attempt(python, run_py, password)
        except _StartupFailed as failed:
            verified, rc = self._teardown_tree(timeout=10.0)
            self._exit_code = rc if rc is not None else failed.exit_code
            self._set_failed(f"{failed.detail}; {self._release_on_verified_exit(verified)}")
            return self.status()
        except Exception as exc:
            verified, rc = self._teardown_tree(timeout=10.0)
            self._exit_code = rc
            self._set_failed(f"start aborted: {type(exc).__name__}: {exc}; "
                             f"{self._release_on_verified_exit(verified)}")
            return self.status()
        finally:
            password = None  # never retained on the instance past the stdin write

    def status(self) -> RuntimeStatus:
        return RuntimeStatus(
            state=self._state,
            instance_id=self._instance_id,
            pid=self._pid,
            base_url=self._base_url,
            epoch=self._grant.epoch if self._grant is not None else None,
            started_at=self._started_at,
            exit_code=self._exit_code,
            detail=self._detail,
        )

    def tick(self) -> RuntimeStatus:
        """Host-driven heartbeat: renew the lease, detect crashes. No auto-restart."""
        if self._state != "running":
            return self.status()
        proc = self._proc
        if proc is None:
            self._set_failed("controller lost its process handle while running")
            return self.status()
        rc = proc.poll()
        if rc is not None:
            verified, wait_rc = self._teardown_tree(timeout=10.0)
            self._exit_code = wait_rc if wait_rc is not None else rc
            self._set_failed(f"sidecar crashed rc={self._exit_code}; "
                             f"{self._release_on_verified_exit(verified)}")
            return self.status()
        try:
            self._grant = self._admission_store().renew(
                self._grant, now=self._clock(), lease_seconds=self._lease_seconds)
        except FencedError as exc:
            # Another controller legitimately took over: stop OUR sidecar and
            # step down; the new holder's record is never touched.
            verified, wait_rc = self._teardown_tree(timeout=10.0)
            self._exit_code = wait_rc
            self._set_failed(f"grant fenced ({exc}); own sidecar stopped "
                             f"(exit verified={verified}); the newer holder was not touched")
        except AdmissionError as exc:
            verified, wait_rc = self._teardown_tree(timeout=10.0)
            self._exit_code = wait_rc
            self._set_failed(f"grant renewal failed ({exc}); own sidecar stopped; "
                             f"{self._release_on_verified_exit(verified)}")
        return self.status()

    def stop(self, timeout: float = 20.0) -> RuntimeStatus:
        """Terminate the tree (children first), VERIFY exit, only THEN release.

        If the exit cannot be verified the grant is deliberately NOT released:
        it stays live and quarantines-by-lease, which is safer than releasing
        residency while a process may still hold GPU memory.
        """
        if self._state == "stopped":
            return self.status()
        if self._state == "failed":
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._state = "stopped"
                self._pid = None
                self._base_url = None
                self._detail = "stopped (no live sidecar; grant already handed back or quarantined)"
                return self.status()
        self._state = "stopping"
        self._base_url = None
        self._detail = ""
        verified, rc = self._teardown_tree(timeout=timeout)
        self._exit_code = rc
        if verified:
            note = self._release_cleanly()
            self._pid = None
            if "NOT released" in note:
                self._set_failed(f"sidecar stopped but {note}")
            else:
                self._state = "stopped"
                self._detail = f"sidecar stopped with verified exit; {note}"
        else:
            self._detail = (f"sidecar exit NOT verified within {timeout}s (pid {self._pid}): "
                            "grant NOT released, it stays live and quarantines-by-lease")
        return self.status()

    def log_tail(self, max_lines: int = 50) -> str:
        """Credential-scrubbed tail of the captured sidecar stdout/stderr."""
        if self._log_path is None or not self._log_path.exists():
            return ""
        text = self._log_path.read_text(encoding="utf-8", errors="replace")
        return sanitize_credentials("\n".join(text.splitlines()[-max_lines:]))

    # -- start internals ------------------------------------------------------

    def _admission_store(self) -> AdmissionStore:
        if self._admission is None:
            self._admission = AdmissionStore()
        return self._admission

    def _start_attempt(self, python, run_py, password: str) -> RuntimeStatus:
        store = self._admission_store()
        boot_id = self._boot_id_fn()
        outcome = store.acquire(
            self._resource, owner_scope=self._owner_scope,
            controller_id=self._instance_id, pid=os.getpid(),
            process_start_identity=self._safe_identity(os.getpid()) or "identity-unavailable",
            engine=self._engine, model=self._model, boot_id=boot_id,
            now=self._clock(),
            # the lease must outlive a full health wait without a heartbeat
            lease_seconds=self._lease_seconds + self._health_timeout)
        if isinstance(outcome, Refused):
            # State unchanged, no process: a refusal is not a failure.
            self._state = "stopped"
            self._detail = (f"start refused by admission: {outcome.reason} — {outcome.detail}")
            return self.status()
        self._grant = outcome

        port = free_loopback_port()
        if port == SHARED_DEFAULT_PORT:  # defensive; free_loopback_port guarantees this
            raise _StartupFailed("refusing the shared :8888 default port")
        data_root = self._data_root
        logs = data_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        self._log_path = logs / f"sidecar-{self._instance_id}.log"
        self._log_handle = open(self._log_path, "a", encoding="utf-8", errors="replace")
        # list-args, never a shell; the password is '-' (stdin), so the command
        # line itself is safe to record for forensics.
        cmd = [str(python), str(run_py), "--api-only", "-H", LOOPBACK_HOST,
               "-p", str(port), "--password", "-"]
        self._log_handle.write(f"\n# spawn: {cmd}\n")
        self._log_handle.flush()
        proc, job = self._spawn(cmd, owned=True, stdin=subprocess.PIPE,
                                stdout=self._log_handle, stderr=subprocess.STDOUT,
                                cwd=str(data_root), env=disposable_environment(data_root),
                                text=True)
        self._proc, self._job = proc, job
        self._pid = proc.pid
        self._child_identity = self._safe_identity(proc.pid)
        self._deliver_password(proc, password)

        deadline = self._clock() + self._health_timeout
        host, bound_port = self._await_bound_endpoint(deadline)
        if host is not None and host != LOOPBACK_HOST:
            raise _StartupFailed(
                f"sidecar reported bound address {host!r}:{bound_port}; only "
                f"{LOOPBACK_HOST} loopback is permitted")
        if bound_port == SHARED_DEFAULT_PORT:
            raise _StartupFailed(f"sidecar reported the shared :{SHARED_DEFAULT_PORT} port")
        base_url = f"http://{LOOPBACK_HOST}:{bound_port}"
        self._await_health(base_url, deadline)

        self._base_url = base_url
        with suppress(AdmissionError):  # residency advance is best-effort; tick() re-drives it
            self._grant = store.renew(self._grant, now=self._clock(), residency="resident")
        self._started_at = time.time()
        self._state = "running"
        self._detail = f"sidecar healthy at {base_url}"
        logger.info("media sidecar running pid=%s url=%s instance=%s",
                    self._pid, base_url, self._instance_id)
        return self.status()

    @staticmethod
    def _deliver_password(proc, password: str) -> None:
        """Password goes to the child's STDIN ONLY, then is forgotten here."""
        stdin = getattr(proc, "stdin", None)
        if stdin is None:
            return
        with suppress(OSError, ValueError):  # early exit surfaces via poll() below
            stdin.write(password + "\n")
            stdin.flush()
            stdin.close()

    def _read_log(self) -> str:
        if self._log_path is None or not self._log_path.exists():
            return ""
        return self._log_path.read_text(encoding="utf-8", errors="replace")

    def _await_bound_endpoint(self, deadline: float) -> tuple[str | None, int]:
        proc = self._proc
        while True:
            rc = proc.poll()
            if rc is not None:
                raise _StartupFailed(
                    f"sidecar exited rc={rc} during startup before reporting a bound port "
                    f"(log tail: {sanitize_credentials(self.log_tail(5)) or 'empty'})",
                    exit_code=rc)
            host, port = parse_bound_endpoint(self._read_log())
            if port is not None:
                return host, port
            if self._clock() >= deadline:
                raise _StartupFailed(
                    f"sidecar did not report a bound port within {self._health_timeout}s "
                    f"(log tail: {sanitize_credentials(self.log_tail(5)) or 'empty'})")
            time.sleep(0.1)

    def _await_health(self, base_url: str, deadline: float) -> None:
        proc = self._proc
        url = base_url + HEALTH_PATH
        while True:
            rc = proc.poll()
            if rc is not None:
                raise _StartupFailed(f"sidecar exited rc={rc} during health check", exit_code=rc)
            if _probe_health(url):
                return
            if self._clock() >= deadline:
                raise _StartupFailed(
                    f"sidecar never returned HTTP 200 from GET {url} within "
                    f"{self._health_timeout}s")
            time.sleep(0.15)

    # -- teardown internals (mirrors LlamaServerSupervisor discipline) --------

    def _teardown_tree(self, timeout: float) -> tuple[bool, int | None]:
        """Terminate children then root, escalate to kill; returns (verified, rc)."""
        proc = self._proc
        if proc is None:
            return True, self._exit_code
        rc = proc.poll()
        children: list = []
        with suppress(Exception):
            import psutil

            with suppress(psutil.Error):
                children = psutil.Process(proc.pid).children(recursive=True)
        if rc is None:
            half = max(0.5, timeout / 2.0)
            for child in children:
                _quiet(child.terminate)
            _quiet(proc.terminate)
            try:
                rc = proc.wait(timeout=half)
            except Exception:
                _quiet(proc.kill)
                try:
                    rc = proc.wait(timeout=half)
                except Exception:
                    rc = None
        for child in children:
            with suppress(Exception):
                if child.is_running():
                    child.kill()
        if self._job is not None:
            _quiet(self._job.close)
            self._job = None
        group_empty = True
        if os.name == "posix":
            from hermes_cli import process_identity as pi

            # A crashed leader can leave its children holding the group: stop
            # them, then release the group number only once it is empty, so a
            # tree we failed to kill stays covered by the owner-death supervisor.
            _quiet(lambda: pi.stop_owned_group_survivors(
                proc.pid, proc=proc, grace=min(3.0, max(0.5, timeout / 4.0))))
            with suppress(Exception):
                group_empty = pi.release_process_group_once_empty(
                    proc.pid, proc=proc, timeout=max(1.0, timeout / 3.0))
        if self._log_handle is not None:
            _quiet(self._log_handle.close)
            self._log_handle = None
        verified = self._process_gone(proc.pid, proc) and (os.name != "posix" or group_empty)
        return verified, rc

    def _safe_identity(self, pid: int) -> str | None:
        try:
            return self._identity_fn(pid)
        except Exception:
            return None

    def _process_gone(self, pid: int | None, proc: subprocess.Popen | None = None) -> bool:
        """True only when exit is PROVEN. Anything ambiguous fails closed.

        Proof is either (a) we reaped the Popen handle ourselves — the kernel
        has then released that pid, whatever a later process called ``pid``
        looks like — or (b) the pid is gone / is a zombie, or (c) it now carries
        a DIFFERENT creation identity than the one recorded for our child.

        A live process whose identity cannot be read, or one we never managed
        to identify, is ambiguous (it may be our child or a recycled pid): the
        caller must keep the grant held rather than treat that as permission.
        """
        if pid is None:
            return True
        if proc is not None and getattr(proc, "returncode", None) is not None:
            return True
        try:
            import psutil

            try:
                process = psutil.Process(pid)
            except psutil.NoSuchProcess:
                return True
            with suppress(psutil.Error):
                if process.status() == psutil.STATUS_ZOMBIE:
                    return True
        except ImportError:
            pass
        current = self._safe_identity(pid)
        if current is None or self._child_identity is None:
            return False  # live and not comparable: ambiguous, never "gone"
        return current != self._child_identity

    # -- grant handback internals ---------------------------------------------

    def _release_cleanly(self) -> str:
        """Terminal handback after a clean verified stop. Never on unverified exit."""
        if self._grant is None:
            return "no grant held"
        try:
            self._admission_store().release(self._grant, now=self._clock())
            return "grant released"
        except FencedError as exc:
            return f"grant NOT released (fenced: {exc})"
        except AdmissionError as exc:
            return f"grant NOT released ({exc})"

    def _release_on_verified_exit(self, verified: bool) -> str:
        """Crash/failure handback: attest the exit through ``recover`` itself."""
        if self._grant is None:
            return "no grant held"
        store = self._admission_store()
        if not verified:
            # No proof the recorded process is gone: never attest. The record
            # stays live and quarantines-by-lease (recover with
            # verified_exit=False would need external verification too).
            return "process exit NOT verified: grant intentionally left live (quarantined-by-lease)"
        try:
            outcome = store.recover(self._resource, self._clock(), self._boot_id_fn(),
                                    verified_exit=True, expected_epoch=self._grant.epoch)
            return f"grant recovery: {outcome}"
        except AdmissionError as exc:
            return f"grant recovery failed ({exc})"

    def _set_failed(self, detail: str) -> None:
        self._state = "failed"
        self._base_url = None
        self._detail = sanitize_credentials(detail)


__all__ = [
    "DEFAULT_ENGINE",
    "DEFAULT_RESOURCE",
    "HEALTH_PATH",
    "LOOPBACK_HOST",
    "SHARED_DEFAULT_PORT",
    "MediaRuntimeController",
    "RuntimeStatus",
    "disposable_environment",
    "free_loopback_port",
    "parse_bound_endpoint",
    "sanitize_credentials",
]
