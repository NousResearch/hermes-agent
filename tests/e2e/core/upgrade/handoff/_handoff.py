"""Staging for the hand-off suite: a real install at release N-1 (or HEAD), processes running
from it, and the real ``hermes update`` to the next commit, all in one shared PID namespace.

Two columns, both git-mode installs over a local bare ``origin`` (the official URL rewritten to it,
so the updater takes the normal non-fork path and never touches the network):

* ``n1``: ``main`` parked at release N-1 (``git describe --tags --abbrev=0 HEAD~1``, overridable with
  ``HERMES_E2E_UPGRADE_BASE``), its venv built from N-1's own ``uv.lock`` (the installer's tier 0),
  then ``main`` moves to HEAD;
* ``head``: HEAD's ``scripts/install.sh`` into an empty HOME, then ``main`` gets NEXT, a synthetic
  child commit adding a marker file.

Every process (gateway, dashboard, cron ticker, kanban dispatcher, the updater) runs inside ONE
``NamespaceHost`` sandbox, the way they share a user's machine.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import re
import threading
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import hermes_yaml as yaml

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.handoff._nshost import NamespaceHost

TRACEBACK = I.TRACEBACK
UPDATE_TIMEOUT = 1500
API_KEY = "e2e-handoff-api-server-key"
NEXT_MARKER = "docs/e2e-handoff-next-marker.txt"


# -- refs ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Refs:
    head: str
    base_tag: str
    base: str


@functools.cache
def refs() -> Refs:
    """HEAD and release N-1, resolved on first use (collection runs no git)."""
    head = I.head_sha()
    try:
        tag = os.environ.get("HERMES_E2E_UPGRADE_BASE") or I.git(
            "describe", "--tags", "--match", "v20[0-9][0-9].*", "--abbrev=0", "HEAD~1", cwd=H.WORKTREE)
        return Refs(head, tag, I.git("rev-parse", f"{tag}^{{commit}}", cwd=H.WORKTREE))
    except AssertionError:  # shallow checkout without tags
        return Refs(head, "", "")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# -- the install --------------------------------------------------------------------------------


@dataclass
class Install:
    column: str
    root: Path
    origin: Path
    env: dict[str, str]
    host: NamespaceHost | None = None
    target: str = ""
    port: int = 0
    logs: list[Path] = field(default_factory=list)

    @property
    def home(self) -> Path:
        return Path(self.env["HOME"])

    @property
    def hermes_home(self) -> Path:
        return Path(self.env["HERMES_HOME"])

    @property
    def checkout(self) -> Path:
        return self.hermes_home / "hermes-agent"

    @property
    def hermes(self) -> str:
        """The command on the user's PATH (``~/.local/bin/hermes``)."""
        return str(self.home / ".local" / "bin" / "hermes")

    def sha(self) -> str:
        return I.git("rev-parse", "HEAD", cwd=self.checkout)

    def cli(self, *args: str, timeout: float = 600, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        assert self.host is not None
        return self.host.run([self.hermes, *args], timeout=timeout, env=env)

    def spawn(self, name: str, *args: str, env: dict[str, str] | None = None) -> int:
        assert self.host is not None
        log = self.root / f"{name}.log"
        self.logs.append(log)
        return self.host.spawn([self.hermes, *args], log=log, env=env)

    def update(self, *extra: str, timeout: float = UPDATE_TIMEOUT) -> subprocess.CompletedProcess:
        return self.cli("update", "--yes", "--branch", "main", *extra, timeout=timeout)

    def diagnostics(self, cp: subprocess.CompletedProcess | None = None) -> str:
        """Everything a red cell needs: the command, the log tails, receipts, the process table."""
        parts = []
        if cp is not None:
            parts.append(H.describe(cp, 8000))
        for log in self.logs:
            if log.exists():
                parts.append(f"--- {log.name} (tail) ---\n{log.read_text(errors='replace')[-3000:]}")
        logs = self.hermes_home / "logs"
        for name in ("gateway.log", "errors.log", "update.log", "agent.log"):
            p = logs / name
            if p.exists():
                parts.append(f"--- logs/{name} (tail) ---\n{p.read_text(errors='replace')[-3000:]}")
        receipt = logs / "update_receipts" / "latest.json"
        if receipt.exists():
            parts.append(f"--- update receipt ---\n{receipt.read_text(errors='replace')[-4000:]}")
        shims = self.root / "shims" / "shim-calls.log"
        if shims.exists():
            parts.append(f"--- supervisor shim calls ---\n{shims.read_text(errors='replace')[-2000:]}")
        if self.host is not None:
            try:
                parts.append("--- sandbox process table ---\n" + self.host.ps_text())
            except Exception as exc:  # the table is a diagnostic; never mask the real failure
                parts.append(f"--- sandbox process table unavailable: {exc} ---")
            parts.append("--- sandbox transcript ---\n" + "\n".join(self.host.transcript[-40:]))
        return "\n".join(parts)

    def close(self) -> list[str]:
        return self.host.close() if self.host is not None else []


def _user_uv(env: dict[str, str], hermes_home: Path) -> None:
    """Managed uv where the installer provisions it; ``uv self update`` is a no-op (no network)."""
    real = I.real_uv()
    assert real is not None
    bin_dir = hermes_home / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    uv = bin_dir / "uv"
    uv.write_text("#!/bin/sh\n" 'if [ "$1" = self ]; then exit 0; fi\n' f'exec "{real}" "$@"\n', encoding="utf-8")
    uv.chmod(0o755)


def stage_n1(root: Path) -> Install:
    """A git install at release N-1 with its own venv, as the N-1 installer left it."""
    root.mkdir(parents=True, exist_ok=True)
    origin = I.make_origin(root, refs().base)
    sb = I.new_sandbox(root / "sb", origin)
    env = sb.env
    hermes_home = Path(env["HERMES_HOME"])
    hermes_home.mkdir(parents=True, exist_ok=True)
    checkout = hermes_home / "hermes-agent"
    # Local-path clone (objects shared, nothing packed), then the official URL as the remote; the
    # sandbox's ~/.gitconfig rewrites it to the local origin for every fetch the updater makes.
    I.git("clone", "-q", "--shared", "-b", "main", str(origin), str(checkout), cwd=root)
    I.git("remote", "set-url", "origin", I.OFFICIAL_HTTPS, cwd=checkout)
    assert I.git("rev-parse", "HEAD", cwd=checkout) == refs().base
    with (checkout / "pyproject.toml").open("rb") as fh:
        base_python = tomllib.load(fh)["project"]["requires-python"]
    no_cfg = root / "uv-config"
    no_cfg.mkdir(exist_ok=True)
    uv_env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_NO_CONFIG", "UV_CONFIG_FILE")}
    uv_env.update(UV_PROJECT_ENVIRONMENT=str(checkout / "venv"), XDG_CONFIG_HOME=str(no_cfg), XDG_CONFIG_DIRS=str(no_cfg))
    cp = subprocess.run([I.real_uv(), "sync", "-q", "--locked", "--extra", "all", "--managed-python", "--python",
                         base_python], cwd=str(checkout), env=uv_env, capture_output=True, text=True, timeout=1800)
    assert cp.returncode == 0, f"N-1 venv install from its uv.lock failed:\n{cp.stderr[-4000:]}"
    local_bin = sb.home / ".local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    (local_bin / "hermes").symlink_to(checkout / "venv" / "bin" / "hermes")
    _user_uv(env, hermes_home)
    env["PATH"] = os.pathsep.join([str(local_bin), env["PATH"]])
    return Install("n1", root, origin, env)


def stage_head(root: Path) -> Install:
    """HEAD's installer into an empty HOME (the HEAD -> NEXT column)."""
    root.mkdir(parents=True, exist_ok=True)
    origin = I.make_origin(root, refs().head)
    # Serve partial clones as GitHub does: the installer asks for ``--filter=blob:none``.
    I.git("config", "uploadpack.allowFilter", "true", cwd=origin)
    I.git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=origin)
    sb = I.new_sandbox(root / "sb", origin)
    cp = I.run_installer(sb)
    assert cp.returncode == 0 and TRACEBACK not in cp.stdout + cp.stderr, "HEAD install failed:\n" + I.describe(cp)
    sb.env["PATH"] = os.pathsep.join([str(sb.home / ".local" / "bin"), sb.env["PATH"]])
    return Install("head", root, origin, sb.env)


def stage(column: str, root: Path) -> Install:
    return stage_n1(root) if column == "n1" else stage_head(root)


def publish_target(inst: Install) -> str:
    """Upstream moves: N-1 -> HEAD, HEAD -> NEXT (a child commit adding a marker file)."""
    if inst.column == "n1":
        I.git("update-ref", "refs/heads/main", refs().head, cwd=inst.origin)
        inst.target = refs().head
    else:
        inst.target = I.publish_commit(inst.origin, inst.root, "release: e2e handoff NEXT",
                                       {NEXT_MARKER: f"NEXT above {refs().head}\n"})
    return inst.target


def start_host(inst: Install) -> NamespaceHost:
    inst.host = NamespaceHost(inst.root, inst.env)
    return inst.host


# -- user configuration -------------------------------------------------------------------------


def write_config(home: Path, provider_url: str, *, port: int | None = None, extra: dict | None = None) -> None:
    """A user config on the fake provider; ``port`` enables the API server platform."""
    home.mkdir(parents=True, exist_ok=True)
    cfg: dict = {
        "model": {"provider": "custom", "base_url": provider_url, "default": "fake-model", "context_length": 128000},
        "agent": {"api_max_retries": 1},
        "compression": {"enabled": False},
    }
    if port is not None:
        cfg["platforms"] = {"api_server": {"enabled": True, "extra": {"host": "127.0.0.1", "port": port}}}
    for k, v in (extra or {}).items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) else v
    (home / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (home / ".env").write_text(f"OPENAI_API_KEY={I.FAKE_KEY}\nAPI_SERVER_KEY={API_KEY}\n", encoding="utf-8")


# -- talking to the gateway ---------------------------------------------------------------------


def http(method: str, url: str, body: dict | None = None, *, key: str | None = API_KEY,
         timeout: float = 120) -> tuple[int, dict | str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
    except urllib.error.HTTPError as exc:
        raw, status = exc.read().decode("utf-8", errors="replace"), exc.code
    except (urllib.error.URLError, OSError) as exc:
        return 0, f"{type(exc).__name__}: {exc}"
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


def health(port: int) -> dict | None:
    status, body = http("GET", f"http://127.0.0.1:{port}/health/detailed", timeout=5)
    return body if status == 200 and isinstance(body, dict) else None


def chat(port: int, text: str) -> tuple[int, dict | str]:
    return http("POST", f"http://127.0.0.1:{port}/v1/chat/completions",
                {"model": "hermes-agent", "messages": [{"role": "user", "content": text}]}, timeout=180)


def reply_text(body) -> str:
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (TypeError, KeyError, IndexError):
        return ""


def identify(home: Path) -> dict | None:
    """The live gateway's own answer to the control-socket ``identify`` verb: its pid (as the
    sandbox sees it) and the code identity it booted with (per-process, not re-read from disk)."""
    from gateway.control_socket import identify_gateway

    return identify_gateway(home, timeout=5.0)


def gateway_pids(inst: Install) -> list[dict]:
    """Every gateway in the sandbox (the user's view: the process table). A gateway's own children
    (forked helpers carrying the same argv) belong to it and are not counted as a second gateway."""
    out = []
    for p in inst.host.procs():
        argv = p["cmdline"]
        if p["state"] in ("Z", "X") or not argv:
            continue
        joined = " ".join(argv)
        if "gateway" in argv and ("run" in argv or "start" in argv) and "hermes" in joined and "update" not in argv:
            out.append(p)
    pids = {p["pid"] for p in out}
    return [p for p in out if p["ppid"] not in pids]


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def db_rows(db: Path, sql: str, args: tuple = ()) -> list[tuple]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


def wait_for(pred, *, timeout: float, what: str, interval: float = 0.25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = pred()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}")


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, symlinks=True)


# -- a cell: install + provider + sandbox -------------------------------------------------------


@contextlib.contextmanager
def cell(column: str, root: Path, provider_url: str, *, extra: dict | None = None):
    """Stage ``column`` under ``root``, write a user config on the fake provider with the API server
    platform on a free port, and start the shared-PID-namespace sandbox. Closing asserts that the
    sandbox took every process with it."""
    inst = stage(column, root / column)
    inst.port = free_port()
    write_config(inst.hermes_home, provider_url, port=inst.port, extra=extra)
    start_host(inst)
    leaked: list[str] = []
    try:
        yield inst
    finally:
        leaked = inst.close()
    assert not leaked, f"processes outlived the sandbox: {leaked}"


# -- the gateway, from the user's side ----------------------------------------------------------


def start_gateway(inst: Install, *args: str) -> dict:
    """``hermes gateway run`` as a user starts it by hand; returns its ``identify`` answer."""
    inst.spawn("gateway", "gateway", "run", *args)
    try:
        wait_for(lambda: health(inst.port), timeout=240, what="the gateway API server /health")
        ident = wait_for(lambda: identify(inst.hermes_home), timeout=60, what="control-socket identify")
    except AssertionError as exc:
        raise AssertionError(f"premise: the gateway never came up: {exc}\n{inst.diagnostics()}") from None
    assert ident["code_sha"] == inst.sha(), f"premise: the gateway must serve the installed commit: {ident}"
    return ident


def gateway_starts(inst: Install) -> list[int]:
    """PIDs of every gateway boot recorded in ``logs/gateway-exit-diag.log`` (one ``gateway.start`` each)."""
    p = inst.hermes_home / "logs" / "gateway-exit-diag.log"
    pids = []
    for line in (p.read_text(errors="replace").splitlines() if p.exists() else []):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("tag") == "gateway.start":
            pids.append(int(rec.get("pid") or 0))
    return pids


def pid_file_pid(inst: Install) -> int | None:
    """The PID ``$HERMES_HOME/gateway.pid`` names (JSON record or a bare number)."""
    p = inst.hermes_home / "gateway.pid"
    try:
        raw = p.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return int(raw) if raw.isdigit() else None
    if isinstance(data, dict):
        return int(data["pid"]) if str(data.get("pid", "")).isdigit() else None
    return int(data) if isinstance(data, int) else None


def receipt(inst: Install) -> dict:
    return read_json(inst.hermes_home / "logs" / "update_receipts" / "latest.json") or {}


def settle_gateway(inst: Install, old_pid: int, *, timeout: float = 180) -> dict | None:
    """Wait for a gateway serving ``inst.target`` under a new PID; None if none shows up."""
    def serving():
        ident = identify(inst.hermes_home)
        return ident if ident and ident.get("pid") != old_pid and ident.get("code_sha") == inst.target else None
    try:
        return wait_for(serving, timeout=timeout, what="a relaunched gateway", interval=1.0)
    except AssertionError:
        return None


# Which open issue explains a missing relaunch, per column (fix PRs delete their entry). ``n1``: the
# restart watcher runs on the bare store Python and dies importing gateway.status.
RELAUNCH_GATES = {
    "n1": (r"relaunch armed but never started: no new gateway process booted",
           "gated on #124649: the update restart watcher dies before relaunching a manual gateway"),
}

_UNSUPERVISED_STOP = re.compile(r"Stopped \d+ manual gateway process\(es\) that had no supervisor")


def relaunch_verdict(inst: Install, old_pid: int, boots_before: int, ident: dict | None,
                     up: subprocess.CompletedProcess | None = None) -> str:
    """Why no gateway is serving the new commit, in words a gate can key on; '' when one is."""
    if ident is not None:
        return ""
    boots = gateway_starts(inst)[boots_before:]
    alive = [p["pid"] for p in gateway_pids(inst)]
    armed = receipt(inst).get("gateway_restart", {})
    out = (up.stdout + up.stderr) if up is not None else ""
    if not boots and armed.get("relaunched_profiles"):
        return (f"relaunch armed but never started: no new gateway process booted after the update stopped PID "
                f"{old_pid} (receipt relaunched_profiles={armed.get('relaunched_profiles')}, live gateways={alive})")
    if not boots and _UNSUPERVISED_STOP.search(out) and not armed.get("killed_pids"):
        return (f"no relaunch: the update did not recognise gateway PID {old_pid} and stopped it as an unsupervised "
                f"stale survivor (receipt gateway_restart={armed}, live gateways={alive})")
    if not boots:
        return (f"no relaunch: the update left no gateway running after PID {old_pid} and armed none "
                f"(receipt gateway_restart={armed}, live gateways={alive})")
    if not alive:
        return f"relaunched gateway exited: boots {boots} after the update, none alive"
    stale = identify(inst.hermes_home)
    return f"gateway alive ({alive}) but not serving {inst.target[:12]}: identify={stale}"


def pid_file_verdict(inst: Install, pid: int, history: list[tuple[float, str]]) -> str:
    """'' when gateway.pid names ``pid``; otherwise how it went wrong (written then deleted under the
    live gateway, never written, or naming someone else)."""
    if pid_file_pid(inst) == pid:
        return ""
    wrote = any(re.search(rf'\b{pid}\b', v) for _, v in history)
    now = history[-1][1] if history else "<unknown>"
    if wrote and now == "<missing>" and pid in [p["pid"] for p in gateway_pids(inst)]:
        return f"gateway.pid was deleted under the live gateway {pid} (history below)"
    if not wrote:
        return f"the relaunched gateway {pid} never wrote gateway.pid (now: {now[:120]})"
    return f"gateway.pid names someone other than the live gateway {pid}: {now[:120]}"


class FileWatch:
    """Records every change to a file's content (timestamped) while the block runs: the history of
    ``gateway.pid`` across an update is the evidence when it ends up naming nobody."""

    def __init__(self, path: Path, interval: float = 0.25) -> None:
        self.path, self.interval = path, interval
        self.history: list[tuple[float, str]] = []
        self._stop = threading.Event()
        self._t0 = time.time()

    def _snap(self) -> str:
        try:
            return self.path.read_text(errors="replace").strip()[:300]
        except OSError:
            return "<missing>"

    def _run(self) -> None:
        last = None
        while not self._stop.is_set():
            cur = self._snap()
            if cur != last:
                self.history.append((round(time.time() - self._t0, 2), cur))
                last = cur
            self._stop.wait(self.interval)

    def __enter__(self) -> FileWatch:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def render(self) -> str:
        return "\n".join(f"  +{t:>7}s {v}" for t, v in self.history)
