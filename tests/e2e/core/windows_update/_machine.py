"""A fresh Windows machine for the PR-time install.ps1 -> ``hermes update`` suite.

Every file here drives the REAL user entry points on a real Windows runner:

* ``scripts/install.ps1 -NonInteractive`` (this checkout's script, run as a file from a
  directory that is not a project, like a user who downloaded it);
* the ``hermes.exe`` it publishes in ``%LOCALAPPDATA%\\hermes\\bin`` for every later
  command (``--version``, one-shot turns, ``gateway run/status/stop``, ``update``);
* ``hermes update --yes`` from HEAD to NEXT, a synthetic child of HEAD.

Only external edges are replaced (tests/install/README.md, "The isolation trick"):

* git: a bare clone of this checkout (``serve.git``) answers every canonical Hermes URL via
  ``url.<file>.insteadOf`` in a machine-owned ``GIT_CONFIG_GLOBAL``. ``serve.git`` allows
  filtered fetches, so the installer's ``--filter=tree:0`` clone is a real partial clone,
  as it is against GitHub. Every ``git.exe`` directory is removed from PATH, so the
  installer stages its own pinned Git, as it does on a clean Windows box.
* the model provider: the recording loopback server (tests/fakes/fake_llm_provider.py).

Tool and dependency downloads (uv, the managed Python, wheels, Node) use the network,
exactly like the real installer.

Each machine is a fake user profile: ``USERPROFILE``/``HOME``/``LOCALAPPDATA``/``APPDATA``
point inside it and ``HERMES_HOME`` is NOT set, so the installer and every ``hermes``
command resolve the default ``%LOCALAPPDATA%\\hermes`` the way a real user's do. CI
puts the profiles in ``C:\\Users`` itself (``HERMES_E2E_PROFILES_ROOT``): the checkout
carries 159-character paths, so a profile any deeper than a real one would hit MAX_PATH
where no user does. The installer also prepends its bin dir to the user PATH in HKCU;
the machine restores that value on teardown.

Machines run in parallel. Updates and gateway lifetimes take a job-wide lock
(``gateway_phase``) so each journey's gateway hand-off is deterministic; installs, turns and
everything else stay parallel. One install's update sparing another's gateway (#124659) is
its own journey (test_update_spares_other_installs.py).

The suite mutates HKCU and downloads a toolchain per machine, so it only runs where
``HERMES_E2E_WINDOWS_INSTALL=1`` (the CI job sets it).
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pytest

from tests.e2e.core.windows._helpers import (
    _PASSTHROUGH_ENV,
    _SECRET_SUFFIXES,
    Run,
    _decode,
    describe,
    kill_tree,
    wait_until,
)
from tests.fakes.fake_llm_provider import write_hermes_home

REPO_ROOT = Path(__file__).resolve().parents[4]
OPT_IN_ENV = "HERMES_E2E_WINDOWS_INSTALL"
INSTALL_TIMEOUT = 1500.0
UPDATE_TIMEOUT = 1200.0
CMD_TIMEOUT = 300.0
GATEWAY_READY_TIMEOUT = 240.0
CANONICAL_URLS = (
    "https://github.com/intelli-verse-x/IVX-desktop.git",
    "https://github.com/intelli-verse-x/IVX-desktop",
    "git@github.com:intelli-verse-x/IVX-desktop.git",
)
NEXT_MARKER = ".hermes-e2e-next"
# Captured before any machine strips PATH: harness plumbing (serve.git, rev-parse) only.
REAL_GIT = shutil.which("git")

# Every module here also carries platforms("windows") + integration + live_system_guard_bypass
# literally in its own ``pytestmark`` (scripts/ci/list_os_marked_tests.py reads the file text).
REQUIRES_OPT_IN = pytest.mark.skipif(
    os.environ.get(OPT_IN_ENV) != "1",
    reason=f"mutates HKCU PATH and downloads a toolchain; set {OPT_IN_ENV}=1 (CI does)")


def _strip_path(path: str, drop: Iterable[str]) -> str:
    """PATH minus every directory that holds one of ``drop`` (a clean machine has none)."""
    kept = []
    for entry in path.split(os.pathsep):
        if not entry:
            continue
        if any(os.path.isfile(os.path.join(entry, name)) for name in drop):
            continue
        kept.append(entry)
    return os.pathsep.join(kept)


def harness_git(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None,
                timeout: float = 600.0) -> str:
    """The driver's own git (never the product's). Raises on failure: it is plumbing."""
    assert REAL_GIT, "harness needs git on PATH to stage serve.git"
    base = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    base.update(env or {})
    res = subprocess.run([REAL_GIT, "-c", "safe.directory=*", *args], cwd=cwd, env=base,
                         capture_output=True, timeout=timeout)
    out, err = _decode(res.stdout), _decode(res.stderr)
    if res.returncode:
        raise RuntimeError(f"harness git {' '.join(args)} failed rc={res.returncode}: {err[-2000:]}")
    return out.strip()


def _hkcu_path() -> tuple[str, int] | None:
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ) as key:
            value, kind = winreg.QueryValueEx(key, "Path")
            return str(value), int(kind)
    except OSError:
        return None


def _restore_hkcu_path(saved: tuple[str, int] | None) -> None:
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as key:
        if saved is None:
            try:
                winreg.DeleteValue(key, "Path")
            except OSError:
                pass
        else:
            winreg.SetValueEx(key, "Path", 0, saved[1], saved[0])


@dataclass
class Machine:
    """One fresh Windows user with a local git origin and a loopback provider."""

    root: Path
    profile_name: str
    base_url: str
    profiles_root: Path | None = None
    # Git for Windows on PATH (a typical developer box). False: a clean machine, where the
    # installer must stage its own pinned Git.
    system_git: bool = False
    path_prepend: list[str] = field(default_factory=list)
    head: str = ""
    next: str = ""
    started: float = field(default_factory=time.time)
    _hkcu: Any = None
    _seq: int = 0
    _spawned: list[subprocess.Popen] = field(default_factory=list)
    _lock_depth: int = 0
    timings: list[tuple[str, float]] = field(default_factory=list)

    # -- layout ---------------------------------------------------------------

    @property
    def profile(self) -> Path:
        return (self.profiles_root or self.root / "Users") / self.profile_name

    @property
    def local(self) -> Path:
        return self.profile / "AppData" / "Local"

    @property
    def hermes_home(self) -> Path:
        return self.local / "hermes"

    @property
    def install_dir(self) -> Path:
        return self.hermes_home / "hermes-agent"

    @property
    def hermes_exe(self) -> Path:
        return self.hermes_home / "bin" / "hermes.exe"

    @property
    def serve(self) -> Path:
        return self.root / "serve.git"

    @property
    def logs(self) -> Path:
        path = self.root / "transcripts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- environment ----------------------------------------------------------

    def env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items()
               if k in _PASSTHROUGH_ENV and not k.upper().endswith(_SECRET_SUFFIXES)}
        roaming = self.profile / "AppData" / "Roaming"
        self.local.mkdir(parents=True, exist_ok=True)
        roaming.mkdir(parents=True, exist_ok=True)
        drop = ("python.exe", "python3.exe", "uv.exe") + (() if self.system_git else ("git.exe",))
        path = _strip_path(os.environ.get("PATH", ""), drop)
        env.update({
            "PATH": os.pathsep.join([*self.path_prepend, path]),
            "USERPROFILE": str(self.profile),
            "HOME": str(self.profile),
            "LOCALAPPDATA": str(self.local),
            "APPDATA": str(roaming),
            "GIT_CONFIG_GLOBAL": str(self.root / "e2e-gitconfig"),
            "NO_COLOR": "1",
            # state.db lives under tmp; under a pytest ancestor the live-DB guard would refuse it.
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
        })
        env.update(extra or {})
        return env

    # -- staging --------------------------------------------------------------

    def stage(self) -> None:
        """serve.git at HEAD (+ NEXT in its object store), the URL redirect, a seeded home."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.head = harness_git("-C", str(REPO_ROOT), "rev-parse", "HEAD")
        harness_git("clone", "--bare", "--quiet", str(REPO_ROOT), str(self.serve))
        harness_git("-C", str(self.serve), "update-ref", "refs/heads/main", self.head)
        harness_git("-C", str(self.serve), "symbolic-ref", "HEAD", "refs/heads/main")
        for key in ("uploadpack.allowAnySHA1InWant", "uploadpack.allowFilter"):
            harness_git("-C", str(self.serve), "config", key, "true")
        self.next = self._mint_next()
        file_url = "file:///" + str(self.serve).replace("\\", "/")
        rewrites = "".join(f"\tinsteadOf = {url}\n" for url in CANONICAL_URLS)
        (self.root / "e2e-gitconfig").write_text(f'[url "{file_url}"]\n{rewrites}', encoding="utf-8")
        # What the user's config would hold after choosing a provider: the loopback mock.
        # install.ps1 keeps an existing config.yaml/.env.
        write_hermes_home(self.hermes_home, self.base_url)
        with (self.hermes_home / "config.yaml").open("a", encoding="utf-8") as fh:
            fh.write("display:\n  compact: true\n")
        # insteadOf rewrites `remote get-url origin` too, so the updater would see a fork and
        # ask to add the official upstream; this is the product's own headless opt-out.
        (self.hermes_home / ".skip_upstream_prompt").write_text("", encoding="utf-8")
        self._hkcu = _hkcu_path()

    def _mint_next(self) -> str:
        """NEXT: HEAD plus a root marker and a nested one (a new subtree for the treeless
        clone to fetch). Lives only in serve.git until ``advance()`` points main at it."""
        blob_src = self.root / "next-marker.txt"
        blob_src.write_text("synthetic NEXT commit for the Windows install/update E2E\n", encoding="utf-8")
        blob = harness_git("-C", str(self.serve), "hash-object", "-w", "--no-filters", str(blob_src))
        index = self.root / "next.index"
        env = {"GIT_INDEX_FILE": str(index),
               "GIT_AUTHOR_NAME": "Hermes E2E", "GIT_AUTHOR_EMAIL": "e2e@hermes.invalid",
               "GIT_COMMITTER_NAME": "Hermes E2E", "GIT_COMMITTER_EMAIL": "e2e@hermes.invalid"}
        harness_git("-C", str(self.serve), "read-tree", self.head, env=env)
        for path in (NEXT_MARKER, f"tests/e2e/{NEXT_MARKER}"):
            harness_git("-C", str(self.serve), "update-index", "--add", "--cacheinfo",
                        f"100644,{blob},{path}", env=env)
        tree = harness_git("-C", str(self.serve), "write-tree", env=env)
        index.unlink(missing_ok=True)
        return harness_git("-C", str(self.serve), "commit-tree", tree, "-p", self.head,
                           "-m", "e2e: synthetic next commit", env=env)

    def advance(self) -> None:
        """Publish NEXT on main: an update becomes available the way it does for a user."""
        harness_git("-C", str(self.serve), "update-ref", "refs/heads/main", self.next)

    # -- running --------------------------------------------------------------

    def _run_logged(self, argv: list[str], label: str, *, timeout: float, cwd: Path | None = None,
                    env_extra: dict[str, str] | None = None) -> Run:
        """Run to completion with the full transcript on disk (survives a timeout)."""
        self._seq += 1
        log = self.logs / f"{self._seq:02d}-{label}.log"
        started = time.monotonic()
        chunks: list[bytes] = []
        proc = subprocess.Popen(argv, cwd=cwd or self.profile, env=self.env(env_extra),
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        def pump() -> None:
            # Every transcript line carries its offset: install.ps1 prints no timestamps, and
            # a stall is only visible as a gap. The pump outlives the command (daemon) so a
            # detached grandchild that inherited the pipe can never block on a full buffer.
            with log.open("w", encoding="utf-8") as fh:
                for raw in iter(proc.stdout.readline, b""):
                    chunks.append(raw)
                    fh.write(f"[{time.monotonic() - started:7.1f}s] {_decode(raw).rstrip()}\n")
                    fh.flush()

        reader = threading.Thread(target=pump, name=f"pump-{label}", daemon=True)
        reader.start()
        try:
            code = proc.wait(timeout=timeout)
            self.timings.append((label, round(time.monotonic() - started, 1)))
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=60)
            proc.wait(timeout=60)
            reader.join(timeout=10)
            raise AssertionError(
                f"{label} did not finish within {timeout:.0f}s\n{self._tail(log)}\n{self.evidence()}") from None
        reader.join(timeout=20)
        return Run(code, _decode(b"".join(chunks)), "")

    @staticmethod
    def _tail(path: Path, n: int = 6000) -> str:
        try:
            text = _decode(path.read_bytes())
        except OSError:
            return f"<no {path.name}>"
        return f"--- {path.name} (last {n} chars) ---\n{text[-n:]}"

    def install(self) -> Run:
        """``powershell -File install.ps1 -NonInteractive`` from a directory that is no project."""
        script = self.root / "install.ps1"
        shutil.copyfile(REPO_ROOT / "scripts" / "install.ps1", script)
        cwd = self.root / "install-cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        return self._run_logged(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-NonInteractive"],
            "install", timeout=INSTALL_TIMEOUT, cwd=cwd)

    def hermes(self, *args: str, label: str | None = None, timeout: float = CMD_TIMEOUT,
               env_extra: dict[str, str] | None = None) -> Run:
        assert self.hermes_exe.is_file(), f"installer published no {self.hermes_exe}\n{self.evidence()}"
        name = label or "hermes-" + "-".join(a.strip("-") for a in args[:2] if a)
        return self._run_logged([str(self.hermes_exe), *args], name, timeout=timeout, env_extra=env_extra)

    def update(self, *extra: str, label: str = "update") -> Run:
        with self.gateway_phase():
            return self.hermes("update", "--yes", *extra, label=label, timeout=UPDATE_TIMEOUT)

    @contextlib.contextmanager
    def gateway_phase(self):
        """Job-wide mutex for updates and gateway lifetimes (reentrant within a machine), so no
        journey's gateway hand-off races another machine's update."""
        if self._lock_depth:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return
        import msvcrt

        lock_dir = Path(os.environ.get("HERMES_E2E_MACHINE_ROOT") or tempfile.gettempdir())
        lock_dir.mkdir(parents=True, exist_ok=True)
        waited = time.monotonic()
        with (lock_dir / "gateway-phase.lock").open("a+b") as fh:
            while True:
                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() - waited > 1800:
                        raise RuntimeError("harness: gateway-phase lock not acquired within 30 min") from None
                    time.sleep(0.5)
            self.timings.append(("(waited for gateway phase)", round(time.monotonic() - waited, 1)))
            self._lock_depth = 1
            try:
                yield
            finally:
                self._lock_depth = 0
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)

    def installed_head(self) -> str:
        try:
            return harness_git("-C", str(self.install_dir), "rev-parse", "HEAD")
        except RuntimeError as exc:
            return f"<unreadable: {exc}>"

    def git_config(self, key: str) -> str:
        try:
            return harness_git("-C", str(self.install_dir), "config", "--get", key)
        except RuntimeError:
            return ""

    # -- gateway --------------------------------------------------------------

    def gateway_state(self) -> dict:
        try:
            return json.loads((self.hermes_home / "gateway_state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def spawn_gateway(self, env_extra: dict[str, str] | None = None) -> subprocess.Popen:
        """``hermes gateway run`` through the published launcher, windowless and detached from
        this console: the shape the Desktop / login item uses to host a gateway."""
        self._seq += 1
        log = self.logs / f"{self._seq:02d}-gateway-run.log"
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        with log.open("wb") as fh:
            proc = subprocess.Popen([str(self.hermes_exe), "gateway", "run"], cwd=self.profile,
                                    env=self.env(env_extra), stdin=subprocess.DEVNULL, stdout=fh,
                                    stderr=subprocess.STDOUT, creationflags=flags)
        self._spawned.append(proc)
        return proc

    def wait_gateway_running(self, *, not_pid: int | None = None,
                             timeout: float = GATEWAY_READY_TIMEOUT) -> dict:
        """The gateway's own record says ``running`` for a live pid (optionally a new one)."""
        import psutil

        def ready() -> dict | None:
            state = self.gateway_state()
            pid = state.get("pid")
            if state.get("gateway_state") != "running" or not pid or pid == not_pid:
                return None
            return state if psutil.pid_exists(int(pid)) else None

        try:
            return wait_until(ready, timeout, "gateway_state.json to report a live running gateway", interval=0.5)
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n{self.evidence()}") from None

    # -- diagnostics ----------------------------------------------------------

    def owned_processes(self) -> list[Any]:
        """Live processes of this machine: exe/cwd/argv under its root, or its HERMES_HOME."""
        import psutil

        roots = {os.path.normcase(os.path.normpath(str(p))) for p in (self.root, self.profile)}
        home = os.path.normcase(os.path.normpath(str(self.hermes_home)))
        me = os.getpid()
        owned = []
        for proc in psutil.process_iter():
            try:
                if proc.pid == me or proc.create_time() < self.started - 1.0:
                    continue
                env = {k.upper(): v for k, v in proc.environ().items()}
                hh = os.path.normcase(os.path.normpath(env.get("HERMES_HOME", "") or "-"))
                blob = os.path.normcase(" ".join([proc.exe() or "", proc.cwd() or "", *proc.cmdline()]))
            except (psutil.Error, OSError):
                continue
            if hh == home or any(root in blob for root in roots):
                owned.append(proc)
        return owned

    def evidence(self) -> str:
        """Receipts, logs and the process table: what a failure message must carry."""
        parts = [f"machine root: {self.root}", f"profile: {self.profile}", f"HEAD={self.head} NEXT={self.next}",
                 f"installed checkout: {self.installed_head()}", f"timings (s): {self.timings}"]
        receipt = self.hermes_home / "logs" / "update_receipts" / "latest.json"
        if receipt.is_file():
            parts.append(self._tail(receipt, 4000))
        parts.append(f"gateway_state.json: {self.gateway_state()}")
        parts.append(f"gateway.pid present: {(self.hermes_home / 'gateway.pid').exists()}")
        logs = self.hermes_home / "logs"
        for name in ("gateway.log", "gateway-stdio.log", "errors.log", "agent.log", "update.log",
                     "desktop-update-handoff.log"):
            if (logs / name).is_file():
                parts.append(self._tail(logs / name, 3000))
        try:
            parts.append("owned processes:\n  " + "\n  ".join(describe(self.owned_processes()) or ["<none>"]))
        except Exception as exc:  # diagnostics must never mask the real failure
            parts.append(f"process table unavailable: {exc}")
        return "\n".join(parts)

    def kill_owned(self) -> None:
        """Hard-stop every live process of this machine (end of a gateway phase, teardown)."""
        kill_tree(self.owned_processes())
        for proc in self._spawned:
            if proc.poll() is None:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=60)

    def teardown(self) -> None:
        self.kill_owned()
        try:
            _restore_hkcu_path(self._hkcu)
        except OSError:
            pass
        artifacts = os.environ.get("HERMES_E2E_ARTIFACTS")
        if artifacts:
            dest = Path(artifacts) / self.root.name
            shutil.copytree(self.logs, dest / "transcripts", dirs_exist_ok=True)
            for sub in ("logs",):
                src = self.hermes_home / sub
                if src.is_dir():
                    shutil.copytree(src, dest / sub, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns("*.db", "*.db-*"))
            (dest / "evidence.txt").write_text(self.evidence(), encoding="utf-8", errors="replace")
            (dest / "timings.json").write_text(json.dumps(self.timings, indent=1), encoding="utf-8")


def new_machine(tmp_root: Path, base_url: str, *, label: str, person: str = "",
                system_git: bool = False) -> Machine:
    """A staged machine. Its work dir (serve.git, transcripts) lives under
    ``HERMES_E2E_MACHINE_ROOT`` (else ``tmp_root``); its user profile under
    ``HERMES_E2E_PROFILES_ROOT`` (CI: ``C:\\Users``), named ``[<person> ]hermes-e2e-<id>``."""
    sfx = uuid.uuid4().hex[:4]
    base = Path(os.environ.get("HERMES_E2E_MACHINE_ROOT") or tmp_root)
    profiles = os.environ.get("HERMES_E2E_PROFILES_ROOT")
    name = f"{person} hermes-e2e-{sfx}" if person else f"hermes-e2e-{sfx}"
    machine = Machine(root=base / f"{label}-{sfx}", profile_name=name, base_url=base_url,
                      profiles_root=Path(profiles) if profiles else None, system_git=system_git)
    machine.stage()
    return machine


class Journey:
    """One expensive walk (install -> update -> ...) shared by a module's cells.

    Each step's value, or the exception it raised, is recorded; a cell reading a step
    re-raises that step's own failure (never a gate-matching claim), so a broken
    prerequisite fails the cells that need it and nothing else."""

    def __init__(self, machine: Machine) -> None:
        self.machine = machine
        self.results: dict[str, Any] = {}

    def step(self, name: str, fn: Any) -> Any:
        try:
            self.results[name] = fn()
        except Exception as exc:  # recorded, re-raised by the cells that read it
            self.results[name] = exc
        return self.results[name]

    def ok(self, name: str) -> bool:
        return name in self.results and not isinstance(self.results[name], BaseException)

    def __getitem__(self, name: str) -> Any:
        if name not in self.results:
            raise RuntimeError(f"journey step {name!r} never ran\n{self.machine.evidence()}")
        value = self.results[name]
        if isinstance(value, BaseException):
            raise RuntimeError(f"journey step {name!r} failed: {value}") from value
        return value

    def require(self, name: str, ok: bool, message: str, run: Run | None = None) -> None:
        """A prerequisite of later steps: raise (recorded by ``step``) when it does not hold."""
        if not ok:
            raise RuntimeError(fail_with(self.machine, f"{name}: {message}", run))


def failure_line(run: Run) -> str:
    """The first line a ``hermes`` command printed as its failure (``✗ ...``), else ``""``."""
    for line in run.stdout.splitlines():
        if line.strip().startswith("✗"):
            return line.strip()
    return ""


def fail_with(machine: Machine, message: str, run: Run | None = None) -> str:
    """Assertion text: the claim first (gates match on it), then the transcript and evidence."""
    with contextlib.suppress(OSError), (machine.logs / "claims.txt").open("a", encoding="utf-8") as fh:
        fh.write(message.splitlines()[0] + "\n")  # CI prints no reason for a passing file's xfails
    tail = f"\n--- transcript rc={run.returncode} ---\n{run.stdout[-6000:]}" if run is not None else ""
    return f"{message}{tail}\n{machine.evidence()}"


@dataclass
class Turn:
    run: Run
    reply_id: str
    reached_wire: bool

    @property
    def ok(self) -> bool:
        return self.run.returncode == 0 and self.reply_id in self.run.stdout and self.reached_wire


def one_shot_turn(machine: Machine, srv: Any, label: str) -> Turn:
    """``hermes chat -q ... -Q`` through the published launcher against the loopback provider."""
    from tests.e2e.core.windows._helpers import last_user
    from tests.fakes.fake_llm_provider import Text

    prompt_id, reply_id = f"PROMPT-{uuid.uuid4().hex[:8]}", f"REPLY-{uuid.uuid4().hex[:8]}"
    before = len(srv.main_requests())
    srv.push(Text(f"The answer is {reply_id}."))
    res = machine.hermes("chat", "-q", f"Say the code {prompt_id}", "-Q", label=label)
    wired = any(prompt_id in last_user(body) for body in srv.main_requests()[before:])
    return Turn(res, reply_id, wired)


# Printed only when a launch detours through source-update completion instead of running
# the requested command (hermes_cli/venv_sync.py, hermes_cli/source_build.py, update_cmd).
SOURCE_COMPLETION_MARKERS = ("completing source-update", "Preparing Node dependencies", "Update complete")


def source_completion_detour(run: Run) -> str | None:
    return next((m for m in SOURCE_COMPLETION_MARKERS if m in run.stdout), None)
