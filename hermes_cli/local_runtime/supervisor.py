"""Supervision of one llama-server in router mode.

The router process is ours (restart with backoff on crash); router children
are its problem — child failures surface via GET /models exit_code, never
auto-retried here.

Readiness rules (each learned the hard way on real hardware):
- health-200 is NOT readiness; every readiness claim requires a touch
  generation (temp-0, expected token, generous budget, reasoning_content
  scanned).
- Always dial 127.0.0.1 — resolving localhost adds ~2s per request on
  Windows via IPv6 fallback.
- /metrics is opt-in (--metrics) and carries no KV-usage metric;
  idleness = requests_processing == 0 and no slot is_processing.
- The router's LRU eviction has no pin for the primary model: until an
  upstream pin exists, keep_primary_loaded re-touches the primary after
  any other model load.
"""

from __future__ import annotations

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


def state_path() -> Path:
    """Endpoint state for other Hermes processes (provider resolution reads
    this to route llamacpp-alias requests at the managed server)."""
    return runtimes_root() / "server.json"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Default port for the managed server, chosen once and reused across
# restarts. Sessions persist the resolved base_url; an ephemeral port
# would strand every resumed session on a dead endpoint after each
# restart. Deliberately NOT 8080 so we never collide with a user's own
# llama-server/Ollama-adjacent stack.
_DEFAULT_PORT = 18434


def _stable_port() -> int:
    """The stable default port, falling back to an ephemeral one only when
    something else already listens there (and it isn't a leftover managed
    server, which stop() would have cleaned up)."""
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

    Endpoint identity must survive restarts as a UNIT — sessions persist the
    resolved base_url + api_key, so a per-boot key strands every resumed
    session on HTTP 401 exactly the way a per-boot port would strand them
    on connection errors. Rotating it buys nothing: the key exists to stop
    other loopback processes free-riding, and it lives on the same disk as
    the state file that would leak it. Delete the file to rotate manually.
    """
    key_path = runtimes_root() / ".api_key"
    try:
        existing = key_path.read_text(encoding="utf-8").strip()
        if len(existing) >= 16:
            return existing
    except OSError:
        pass
    key = secrets.token_urlsafe(24)
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_text(key, encoding="utf-8")
    except OSError as exc:
        logger.warning("could not persist api key (%s); sessions will need "
                       "a re-pick after restart", exc)
    return key


class LlamaServerSupervisor:
    """Own one llama-server router process for the life of a Hermes session.

    Usage::

        sup = LlamaServerSupervisor(install_dir, models_dir)
        sup.start()                     # spawn + wait healthy
        sup.ensure_model_ready(name)    # load + touch-generate
        ... sup.base_url is the /v1 endpoint, sup.api_key its key ...
        sup.stop()
    """

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

    def _request(self, route: str, body: dict | None = None, timeout_s: int = 30) -> dict:
        req = urllib.request.Request(
            self._url(route),
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
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
            # The residency contract at the layer that sees every message:
            # a chat request to a staged-but-unloaded model loads it (slow
            # first token) instead of failing with 'model not found' —
            # without this flag, chat after an eject is a bare 400/404.
            "--models-autoload",
            "--metrics",          # opt-in flag; supervisor telemetry needs it
            "--slots",            # /slots endpoint is also opt-in; is_idle reads it
            "--no-webui",
            "--jinja",
            # Direct I/O on model load: bypasses the page cache, so a
            # multi-GB load doesn't evict half the OS cache — measured
            # faster loads on NVMe, and our router bounces (download/
            # delete/activate) reload models often enough to care.
            "-dio",
        ]
        if self.preset_path and self.preset_path.exists():
            cmd += ["--models-preset", str(self.preset_path)]
        cmd += [
            *self.extra_args,
        ]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log_handle is not None:
            # The crash-restart loop calls _spawn repeatedly; without
            # closing the prior handle each restart leaks one fd.
            try:
                self._log_handle.close()
            except Exception:  # noqa: BLE001 — best-effort
                pass
        self._log_handle = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self._log_handle.write(f"\n# spawn: {cmd}\n")
        self._log_handle.flush()
        # list-args, never a shell: spaced paths (user homes) must survive.
        self.proc = subprocess.Popen(cmd, stdout=self._log_handle,
                                     stderr=subprocess.STDOUT, cwd=str(exe.parent))
        logger.info("llama-server router spawned pid=%s port=%s", self.proc.pid, self.port)
        # State goes down at SPAWN, not after health: endpoint resolution
        # treats a live-pid-but-not-yet-healthy server as "starting" rather
        # than "unconfigured", so a readiness probe racing the boot doesn't
        # throw the app back to onboarding (observed on first restart test).
        self._write_state()

    def start(self, timeout_s: int = 120) -> None:
        self._stopping = False
        self._spawn()
        self._wait_health(timeout_s)
        self._write_state()
        self._watchdog = threading.Thread(target=self._watch, daemon=True,
                                          name="llamacpp-supervisor")
        self._watchdog.start()

    def _write_state(self) -> None:
        path = state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "base_url": self.base_url,
            "api_key": self.api_key,
            "pid": self.proc.pid if self.proc else None,
        }), encoding="utf-8")

    def _wait_health(self, timeout_s: int) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited rc={self.proc.returncode} during startup "
                    f"(log: {self.log_path})")
            try:
                with urllib.request.urlopen(self._url("/health"), timeout=3) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
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
            logger.warning("llama-server exited rc=%s; restart #%s in %ss",
                           rc, self._restarts + 1, backoff)
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

        The router spawns one child llama-server per loaded model, each
        holding gigabytes of VRAM. Terminating only the router (on
        Windows, TerminateProcess — no signal handlers, no cleanup pass)
        orphans those children: the port goes quiet but the weights stay
        resident, and the next spawn re-loads models alongside a ghost
        still holding the memory. Enumerate children FIRST (the parent
        must be alive to walk them), then terminate parent and children
        together, escalating to kill for stragglers.
        """
        children: list = []
        try:
            import psutil

            children = psutil.Process(proc.pid).children(recursive=True)
        except Exception:  # noqa: BLE001 — no psutil view; still stop the router
            children = []
        proc.terminate()
        for child in children:
            try:
                child.terminate()
            except Exception:  # noqa: BLE001
                pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        for child in children:
            try:
                if child.is_running():
                    child.kill()
            except Exception:  # noqa: BLE001
                pass

    def _reap_orphaned_children(self) -> None:
        """Kill model children orphaned by a router crash, before respawn.

        A crashed router can't clean up its children, and a dead parent
        can't be walked — so match by identity instead: any process
        running OUR llama-server binary whose parent is gone is an
        orphan of a previous router. Their VRAM must come back before
        the new router loads models next to the ghosts. External
        llama-servers (different binary path) never match.
        """
        try:
            import psutil

            exe = str(server_binary(self.install_dir))
        except Exception:  # noqa: BLE001
            return
        for p in psutil.process_iter(["exe", "ppid"]):
            try:
                if p.info.get("exe") != exe:
                    continue
                if self.proc is not None and p.pid == self.proc.pid:
                    continue
                ppid = p.info.get("ppid") or 0
                if ppid and psutil.pid_exists(ppid):
                    continue
                logger.warning("reaping orphaned llama-server child pid=%s", p.pid)
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    # ── model management (router endpoints) ──────────────────

    def models(self) -> dict:
        """{model_id: status_value} from GET /models."""
        data = self._request("/models")
        return {m["id"]: m.get("status", {}).get("value", "unknown")
                for m in data.get("data", [])}

    def model_failures(self) -> dict:
        """{model_id: exit_code} for children that died — surfaced to the
        UI, never auto-retried (design: router children are its problem)."""
        data = self._request("/models")
        out = {}
        for m in data.get("data", []):
            status = m.get("status", {})
            if status.get("value") == "failed" or status.get("exit_code"):
                out[m["id"]] = status.get("exit_code")
        return out

    def load_model(self, model_id: str, timeout_s: int = 600) -> None:
        self._request("/models/load", {"model": model_id}, timeout_s=timeout_s)

    def unload_model(self, model_id: str) -> None:
        """Free the child's VRAM now. Route existence verified empirically
        on b10290 (POST /models/unload; bogus name -> 400 'model is not
        found'). Momentary action: never touches primary_model — the
        declaration is durable, an eject is not (residency design).

        Settle before returning: for a few seconds after unload returns,
        the router still routes to the dying child and answers chat with
        500 'proxy error: Could not establish connection' (probed on
        b10362). Waiting for the model to report unloaded means the next
        message autoloads cleanly instead of racing the teardown.
        """
        self._request("/models/unload", {"model": model_id}, timeout_s=120)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                if self.models().get(model_id) not in ("loaded", "ready", "unloading"):
                    return
            except Exception:  # noqa: BLE001
                return
            time.sleep(0.3)

    # ── idle residency (non-primary models) ──────────────────

    # A model that has gone quiet gets its VRAM back after this long. A
    # constant, not a knob: long enough that an active conversation never
    # trips it, short enough that a wandered-off session frees ~20 GiB
    # within the hour. No exemptions (residency v2): demand reloads
    # anything the user comes back to.
    IDLE_UNLOAD_S = 15 * 60

    def sweep_idle(self, now: float | None = None) -> list[str]:
        """Unload models idle past IDLE_UNLOAD_S. Returns the model ids
        unloaded. Idle means no busy slots and no queued work, tracked
        per model across calls; a model seen busy resets its clock."""
        now = time.monotonic() if now is None else now
        unloaded: list[str] = []
        try:
            statuses = self.models()
        except Exception:  # noqa: BLE001
            return unloaded
        for model_id, status in statuses.items():
            if status not in ("loaded", "ready"):
                self._idle_since.pop(model_id, None)
                continue
            if not self.is_idle(model_id):
                self._idle_since.pop(model_id, None)
                continue
            first_idle = self._idle_since.setdefault(model_id, now)
            if now - first_idle >= self.IDLE_UNLOAD_S:
                try:
                    self.unload_model(model_id)
                    self._idle_since.pop(model_id, None)
                    unloaded.append(model_id)
                    logger.info("idle-unloaded %s (idle %ds)", model_id,
                                int(now - first_idle))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("idle unload of %s failed: %s", model_id, exc)
        return unloaded

    def touch_generate(self, model_id: str, timeout_s: int = 300) -> bool:
        """The readiness proof. Generous budget + reasoning_content scan —
        small token budgets false-fail reasoning models, which spend their
        first tokens thinking."""
        try:
            resp = self._request("/v1/chat/completions", {
                "model": model_id,
                "messages": [{"role": "user", "content": TOUCH_PROMPT}],
                "max_tokens": 512, "temperature": 0,
            }, timeout_s=timeout_s)
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
        if status not in ("loaded", "ready"):
            self.load_model(model_id, timeout_s=timeout_s)
        return self.touch_generate(model_id)

    def actual_n_ctx(self, model_id: str) -> int | None:
        """/props reconciliation: the granted window as the child reports
        it — the compressor's budget and the picker's 'running at 87K of
        262K' both read THIS value, never the request (design step 4)."""
        try:
            props = self._request(f"/props?model={model_id}")
            return props.get("default_generation_settings", {}).get("n_ctx")
        except Exception:  # noqa: BLE001
            return None

    def keep_primary_loaded(self) -> None:
        """The router's LRU eviction has no pin, so after any other load
        re-touch the primary to keep it most-recently-used. Best-effort
        under bursty multi-model load — replaced when an upstream pin
        exists."""
        if self.primary_model and self.models().get(self.primary_model) in (
                "loaded", "ready"):
            self.touch_generate(self.primary_model, timeout_s=60)

    # ── telemetry ────────────────────────────────────────────

    def is_idle(self, model_id: str | None = None) -> bool:
        """No processing requests and no busy slots. Router quirk: /slots
        and /metrics are per-child and require ?model= (bare calls 400),
        and no KV-usage metric exists. With ``model_id`` checks that one
        child; without, every loaded child."""
        try:
            if model_id is not None:
                loaded = [model_id]
            else:
                loaded = [m for m, status in self.models().items()
                          if status in ("loaded", "ready")]
            for mid in loaded:
                slots = self._request(f"/slots?model={mid}")
                if any(s.get("is_processing") for s in slots):
                    return False
                req = urllib.request.Request(
                    self._url(f"/metrics?model={mid}"),
                    headers={"Authorization": f"Bearer {self.api_key}"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    text = r.read().decode()
                for line in text.splitlines():
                    if line.startswith("llamacpp:requests_processing"):
                        if float(line.split()[-1]) != 0.0:
                            return False
            return True
        except Exception:  # noqa: BLE001
            return False
