"""Supervision of one llama-server in router mode.

The router process is ours (restart with backoff on crash); router children are its problem — child
failures surface via GET /models exit_code, never auto-retried here. Learned on real hardware:
health-200 is NOT readiness — every readiness claim requires a touch generation (temp-0, expected
token, generous budget, reasoning_content scanned); always dial 127.0.0.1 — resolving localhost adds
~2s per request on Windows via IPv6 fallback.
"""

from __future__ import annotations

from contextlib import suppress
import json
import logging
import secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from hermes_cli.local_runtime.binaries import server_binary, runtimes_root

logger = logging.getLogger(__name__)

TOUCH_PROMPT = "Reply with exactly one word: the capital of France."
TOUCH_EXPECT = "paris"
_RESTART_BACKOFF_S = (1, 5, 15, 60)
_RESIDENT = ("loaded", "ready")

# Chosen once and reused across restarts: sessions persist the resolved base_url, so an ephemeral
# port would strand every resumed session after each restart. Deliberately NOT 8080 so we never
# collide with a user's own llama-server/Ollama-adjacent stack.
_DEFAULT_PORT = 18434


def state_path() -> Path:
    """Endpoint state for other Hermes processes (provider resolution routes llamacpp-alias
    requests at the managed server from this)."""
    return runtimes_root() / "server.json"


def _quiet(fn) -> None:
    """Best-effort call; a child that vanished mid-walk is not an error."""
    with suppress(Exception):
        fn()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stable_port() -> int:
    """The stable default port, or an ephemeral one only when something else already listens
    there (a leftover managed server would have been cleaned up by stop())."""
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", _DEFAULT_PORT))
            return _DEFAULT_PORT
    except OSError:
        logger.warning(
            "port %d busy; managed llama-server falling back to an ephemeral "
            "port — existing sessions may need a model re-pick", _DEFAULT_PORT)
        return _free_port()


def _stable_api_key() -> str:
    """One key for the life of the install, persisted beside the runtimes.

    Endpoint identity must survive restarts as a UNIT — sessions persist base_url + api_key, so a
    per-boot key strands every resumed session on HTTP 401 exactly as a per-boot port would on
    connection errors.
    """
    key_path = runtimes_root() / ".api_key"
    with suppress(OSError):
        existing = key_path.read_text(encoding="utf-8").strip()
        if len(existing) >= 16:
            return existing
    key = secrets.token_urlsafe(24)
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(key, encoding="utf-8")
    except OSError as exc:
        logger.warning("could not persist api key (%s); sessions will need "
                       "a re-pick after restart", exc)
    return key


# Flags the supervisor itself manages — accepting them from user config would desync
# supervised endpoint identity (base_url/api_key) or the models directory the router
# serves, stranding resumed sessions the same way a rotated port/key does.
_MANAGED_SERVER_FLAGS = frozenset({"--host", "--port", "--api-key", "--models-dir"})

_MAX_SERVER_EXTRA_ARGS = 32
_MAX_SERVER_EXTRA_ARG_LEN = 512


def normalize_server_extra_args(value) -> list[str]:
    """Sanitize the ``local_runtime.server_extra_args`` config value into the argv tail
    appended after the supervisor's managed flags (last-wins on duplicates).

    Keeps non-empty string items; drops non-list values, non-string/blank items, and
    managed-identity flags (``--flag`` and ``--flag=value`` forms, plus a following
    bare value such as the ``8080`` in ``["--port", "8080"]``). Bounds keep a typo
    from becoming a fork-bomb-length command line. Never raises — a bad config value
    must degrade to stock flags, never break boot.
    """
    if not isinstance(value, list):
        if value not in (None, [], ""):
            logger.warning("local_runtime.server_extra_args ignored: expected a list of flags")
        return []
    cleaned: list[str] = []
    skip_next = False
    for item in value:
        if skip_next:
            skip_next = False
            if isinstance(item, str) and not item.strip().startswith("-"):
                continue
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text or len(text) > _MAX_SERVER_EXTRA_ARG_LEN:
            continue
        head = text.split("=", 1)[0]
        if head in _MANAGED_SERVER_FLAGS:
            logger.warning("local_runtime.server_extra_args refused managed flag: %s", head)
            if "=" not in text:
                skip_next = True
            continue
        cleaned.append(text)
        if len(cleaned) >= _MAX_SERVER_EXTRA_ARGS:
            break
    return cleaned


class LlamaServerSupervisor:
    """Own one llama-server router process for the life of a Hermes session."""

    # A model that has gone quiet gets its VRAM back after this long. A constant, not a knob:
    # long enough that an active conversation never trips it, short enough that a wandered-off
    # session frees ~20 GiB within the hour. No exemptions: demand reloads anything the user
    # comes back to.
    IDLE_UNLOAD_S = 15 * 60

    def __init__(self, install_dir: Path, models_dir: Path, *,
                 models_max: int = 4, port: int | None = None,
                 extra_args: list[str] | None = None,
                 log_path: Path | None = None,
                 preset_path: Path | None = None):
        self.install_dir = Path(install_dir)
        self.models_dir = Path(models_dir)
        self.models_max = models_max
        self.port = port or _stable_port()
        self.api_key = _stable_api_key()
        self.extra_args = list(extra_args or [])
        self.log_path = log_path or (self.models_dir.parent / "logs" / "llama-server.log")
        self.preset_path = preset_path
        self.proc: subprocess.Popen | None = None
        self.primary_model: str | None = None
        self._restarts = 0
        self._stopping = False
        self._watchdog: threading.Thread | None = None
        self._log_handle = None
        self._idle_since: dict[str, float] = {}

    # ── endpoints ────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _url(self, route: str) -> str:
        return f"http://127.0.0.1:{self.port}{route}"

    def _open(self, route: str, body: dict | None = None, timeout_s: int = 30,
              *, json_type: bool = True):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if json_type:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self._url(route), headers=headers,
                                     data=json.dumps(body).encode() if body is not None else None)
        return urllib.request.urlopen(req, timeout=timeout_s)

    def _request(self, route: str, body: dict | None = None, timeout_s: int = 30) -> dict:
        with self._open(route, body, timeout_s) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}

    # ── lifecycle ────────────────────────────────────────────

    def _spawn(self) -> None:
        exe = server_binary(self.install_dir)
        cmd = [
            str(exe),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--api-key", self.api_key,
            "--models-dir", str(self.models_dir),
            "--models-max", str(self.models_max),
            # Residency contract: a chat request to a staged-but-unloaded model loads it (slow
            # first token) instead of a bare 400/404 after an eject.
            "--models-autoload",
            "--metrics",          # opt-in flag; supervisor telemetry needs it
            "--slots",            # /slots endpoint is also opt-in; is_idle reads it
            "--no-webui",
            "--jinja",
            # Direct I/O on model load bypasses the page cache so a multi-GB load doesn't evict
            # half the OS cache — measured faster on NVMe, and our router bounces reload often.
            "-dio",
        ]
        if self.preset_path and self.preset_path.exists():
            cmd += ["--models-preset", str(self.preset_path)]
        cmd += self.extra_args
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log_handle is not None:
            # The crash-restart loop calls _spawn repeatedly; each restart would leak one fd.
            _quiet(self._log_handle.close)
        self._log_handle = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self._log_handle.write(f"\n# spawn: {cmd}\n")
        self._log_handle.flush()
        # list-args, never a shell: spaced paths (user homes) must survive.
        self.proc = subprocess.Popen(cmd, stdout=self._log_handle,
                                     stderr=subprocess.STDOUT, cwd=str(exe.parent))
        logger.info("llama-server router spawned pid=%s port=%s", self.proc.pid, self.port)
        # State goes down at SPAWN, not after health: endpoint resolution treats a
        # live-pid-but-not-yet-healthy server as "starting" rather than "unconfigured", so a
        # readiness probe racing the boot doesn't throw the app back to onboarding.
        self._write_state()

    def start(self, timeout_s: int = 120) -> None:
        self._stopping = False
        self._spawn()
        self._wait_health(timeout_s)
        self._write_state()
        self._watchdog = threading.Thread(target=self._watch, daemon=True, name="llamacpp-supervisor")
        self._watchdog.start()

    def _write_state(self) -> None:
        path = state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"base_url": self.base_url, "api_key": self.api_key,
                                    "pid": self.proc.pid if self.proc else None}), encoding="utf-8")

    def _wait_health(self, timeout_s: int) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited rc={self.proc.returncode} during startup "
                                   f"(log: {self.log_path})")
            with suppress(urllib.error.URLError, OSError, TimeoutError):
                with urllib.request.urlopen(self._url("/health"), timeout=3) as r:
                    if r.status == 200:
                        return
            time.sleep(1)
        raise TimeoutError(f"llama-server not healthy after {timeout_s}s (log: {self.log_path})")

    def _watch(self) -> None:
        """Restart the router (not its children) on crash, with backoff."""
        while not self._stopping:
            proc = self.proc
            if proc is None:
                return
            rc = proc.poll()
            if rc is None:
                time.sleep(2)
                continue
            if self._stopping:
                return
            backoff = _RESTART_BACKOFF_S[min(self._restarts, len(_RESTART_BACKOFF_S) - 1)]
            logger.warning("llama-server exited rc=%s; restart #%s in %ss", rc, self._restarts + 1, backoff)
            time.sleep(backoff)
            self._restarts += 1
            try:
                self._reap_orphaned_children()
                self._spawn()
                self._wait_health(120)
                if self.primary_model:
                    self.ensure_model_ready(self.primary_model)
            except Exception as exc:  # noqa: BLE001
                logger.error("llama-server restart failed: %s", exc)

    def stop(self) -> None:
        self._stopping = True
        state_path().unlink(missing_ok=True)
        if self.proc and self.proc.poll() is None:
            self._terminate_tree(self.proc)
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    @staticmethod
    def _terminate_tree(proc: subprocess.Popen) -> None:
        """Terminate the router AND its model children.

        Each child holds gigabytes of VRAM; terminating only the router (TerminateProcess on
        Windows does no cleanup) orphans them with the weights still resident. Enumerate children
        FIRST (the parent must be alive to walk them), terminate all, escalate to kill.
        """
        children: list = []
        with suppress(Exception):  # no psutil view; still stop the router
            import psutil

            children = psutil.Process(proc.pid).children(recursive=True)
        proc.terminate()
        for child in children:
            _quiet(child.terminate)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        for child in children:
            _quiet(lambda: child.is_running() and child.kill())

    def _reap_orphaned_children(self) -> None:
        """Kill model children orphaned by a router crash, before respawn.

        A dead parent can't be walked, so match by identity: any process running OUR
        llama-server binary whose parent is gone is an orphan of a previous router. Its VRAM must
        come back before the new router loads models next to the ghosts.
        """
        try:
            import psutil

            exe = str(server_binary(self.install_dir))
        except Exception:  # noqa: BLE001
            return
        own_pid = self.proc.pid if self.proc is not None else None
        for p in psutil.process_iter(["exe", "ppid"]):
            with suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                ppid = p.info.get("ppid") or 0
                if (p.info.get("exe") != exe or p.pid == own_pid
                        or (ppid and psutil.pid_exists(ppid))):
                    continue
                logger.warning("reaping orphaned llama-server child pid=%s", p.pid)
                p.kill()

    # ── model management (router endpoints) ──────────────────

    def models(self) -> dict:
        """{model_id: status_value} from GET /models."""
        return {m["id"]: m.get("status", {}).get("value", "unknown")
                for m in self._request("/models").get("data", [])}

    def load_model(self, model_id: str, timeout_s: int = 600) -> None:
        self._request("/models/load", {"model": model_id}, timeout_s=timeout_s)

    def unload_model(self, model_id: str) -> None:
        """Free the child's VRAM now (POST /models/unload; bogus name -> 400). Momentary: never
        touches primary_model — the declaration is durable, an eject is not."""
        self._request("/models/unload", {"model": model_id}, timeout_s=120)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                if self.models().get(model_id) not in (*_RESIDENT, "unloading"):
                    return
            except Exception:  # noqa: BLE001
                return
            time.sleep(0.3)

    def sweep_idle(self, now: float | None = None) -> list[str]:
        """Unload models idle past IDLE_UNLOAD_S; returns their ids. Idle = no busy slots and no
        queued work, tracked per model across calls; a model seen busy resets its clock."""
        now = time.monotonic() if now is None else now
        unloaded: list[str] = []
        try:
            statuses = self.models()
        except Exception:  # noqa: BLE001
            return unloaded
        for model_id, status in statuses.items():
            if status not in _RESIDENT or not self.is_idle(model_id):
                self._idle_since.pop(model_id, None)
                continue
            first_idle = self._idle_since.setdefault(model_id, now)
            if now - first_idle < self.IDLE_UNLOAD_S:
                continue
            try:
                self.unload_model(model_id)
                self._idle_since.pop(model_id, None)
                unloaded.append(model_id)
                logger.info("idle-unloaded %s (idle %ds)", model_id, int(now - first_idle))
            except Exception as exc:  # noqa: BLE001
                logger.warning("idle unload of %s failed: %s", model_id, exc)
        return unloaded

    def touch_generate(self, model_id: str, timeout_s: int = 300) -> bool:
        """The readiness proof. Generous budget + reasoning_content scan — small token budgets
        false-fail reasoning models, which spend their first tokens thinking."""
        try:
            resp = self._request("/v1/chat/completions", {
                "model": model_id, "messages": [{"role": "user", "content": TOUCH_PROMPT}],
                "max_tokens": 512, "temperature": 0}, timeout_s=timeout_s)
            msg = resp["choices"][0]["message"]
            blob = (msg.get("content") or "") + " " + (msg.get("reasoning_content") or "")
            return TOUCH_EXPECT in blob.lower()
        except Exception as exc:  # noqa: BLE001
            logger.warning("touch generation failed for %s: %s", model_id, exc)
            return False

    def ensure_model_ready(self, model_id: str, timeout_s: int = 600) -> bool:
        """Load if needed, then prove readiness with a touch generation."""
        status = self.models().get(model_id)
        if status is None:
            raise KeyError(f"model {model_id} not present in models dir")
        if status not in _RESIDENT:
            self.load_model(model_id, timeout_s=timeout_s)
        return self.touch_generate(model_id)

    # ── telemetry ────────────────────────────────────────────

    def is_idle(self, model_id: str | None = None) -> bool:
        """No processing requests and no busy slots. Router quirk: /slots and /metrics are
        per-child and require ?model= (bare calls 400). With ``model_id`` checks that one child;
        without, every loaded child."""
        try:
            loaded = ([model_id] if model_id is not None
                      else [m for m, status in self.models().items() if status in _RESIDENT])
            for mid in loaded:
                slots = self._request(f"/slots?model={mid}")
                if any(s.get("is_processing") for s in slots):
                    return False
                with self._open(f"/metrics?model={mid}", timeout_s=10, json_type=False) as r:
                    text = r.read().decode()
                for line in text.splitlines():
                    if (line.startswith("llamacpp:requests_processing")
                            and float(line.split()[-1]) != 0.0):
                        return False
            return True
        except Exception:  # noqa: BLE001
            return False
