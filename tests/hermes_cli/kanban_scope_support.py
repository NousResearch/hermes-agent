"""Kanban worker systemd-scope isolation — spawn, registration, re-adoption,
reaping.

Regression tests for the production incident (networkos-agent, 2026-09-01):
workers spawned as plain children of the gateway shared its cgroup, so
build workers (dev servers, browsers, DBs) OOM-throttled the gateway and
each gateway restart orphaned every in-flight run — the new gateway saw
claim_locks owned by the dead gateway pid and marked the runs crashed
("pid <n> not alive"), discarding ~18 runs / ~30h of build time.

The reworked contracts (Gate B findings):

* ``_default_spawn`` wraps the worker argv in ``systemd-run --user
  --scope`` under a RUN-SUFFIXED unit name (a respawn can never collide
  with a lingering scope), with BOTH MemoryMax and MemorySwapMax derived
  from the same bound — or both omitted when no bound is computable;
* the pid recorded at spawn is the systemd-run LAUNCHER's; the WORKER
  self-registers its own pid + start-time fingerprint from its
  heartbeat bridge / the ``kanban_heartbeat`` tool, and liveness is
  scope-cgroup truth first, never bare PID liveness;
* re-adoption after a gateway restart follows the registered worker
  (killing the launcher — exactly what a gateway death does — must not
  crash the run);
* every terminal path stops the whole scope VERIFIED (stop → state
  check → SIGKILL escalation), and a stop that cannot be confirmed
  DEFERS instead of releasing the claim;
* a refused systemd-run launch classifies as spawn_failed: auto mode
  falls back to a plain spawn for that run, systemd-scope mode fails
  loudly with the stderr;
* dispatcher sweeps: never-registered runs past the launch grace become
  spawn failures, orphaned scopes are reaped, and the
  ``worker_isolation_stop_on_shutdown`` policy can stop everything on
  graceful shutdown.

The systemd binaries are faked with PATH shims (``systemd-run`` forks so
the launcher pid genuinely differs from the worker pid; ``systemctl``
models show/stop/kill/list-units over a temp state dir, with knobs for
refused launches, hung stops, and leaked descendants) instead of mocks,
so the failure modes the Gate B review flagged (launcher-vs-worker pid,
stop timeouts, descendant leakage) are exercised end-to-end.
"""

from __future__ import annotations

import hermes_cli.kanban_db as _owner_kanban_db

import hermes_cli.kanban_db_boards as _owner_kanban_db_boards

import hermes_cli.kanban_claims as _owner_kanban_claims
import hermes_cli.kanban_db_models as _owner_kanban_db_models

import json

import os

import shutil

import signal

import subprocess

import sys

import time

from pathlib import Path

import pytest

from gateway import kanban_watchers as _kw  # noqa: F401 — see the fixture below

from hermes_cli import kanban_db as kb


@pytest.fixture(autouse=True)
def _stable_module_identity():
    """K: keep every import in this pytest process agreeing on ONE
    kanban_db object — this file's ``kb``.

    Other test modules (isolated-home fixtures in the kanban CLI tests)
    purge ``hermes_cli.*`` from sys.modules and re-import it. Importing
    ``gateway.kanban_watchers`` at module level (above) pins its ``_kb``
    to the same copy as ``kb``; this autouse guard repairs sys.modules if
    a purge still left a different copy in place, so body-level imports
    inside tests (and any future str-path monkeypatch) cannot bind to a
    second module object whose attributes our patches never reach. It
    also resets the scope-stop service state around every test — the
    pending queue is process-global and must not bleed between tests.
    """
    if sys.modules.get("hermes_cli.kanban_db") is not kb:
        sys.modules["hermes_cli.kanban_db"] = kb
    _kanban_worker_stop.reset_scope_stop_service_for_tests()
    yield
    _kanban_worker_stop.reset_scope_stop_service_for_tests()


_SYSTEMD_RUN_SHIM = f'''#!{sys.executable}
"""Fake systemd-run --user --scope: fork so launcher != worker.

Deliberately dependency-light (no pathlib): this shim must fail or fork
within milliseconds — the spawn path's launch-probe window is what turns
a refused launch into spawn_failed, so a slow-booting shim would race it
and look healthy.
"""
import json, os, sys

STATE = os.environ["HERMES_KANBAN_TEST_SHIM_STATE"]
UNITS = os.path.join(STATE, "units")
os.makedirs(UNITS, exist_ok=True)

argv = sys.argv[1:]
sep = argv.index("--")
flags, cmd = argv[:sep], argv[sep + 1:]
unit = flags[flags.index("--unit") + 1]
unit_path = os.path.join(UNITS, unit + ".json")

fail = os.path.join(STATE, "fail_next")
if os.path.exists(fail):
    with open(fail) as f:
        n = int(f.read().strip() or "1")
    if n > 0:
        with open(fail, "w") as f:
            f.write(str(n - 1))
        sys.stderr.write(
            "Failed to start transient scope unit " + unit + ": "
            "systemd-run-test: user bus connection refused\\n"
        )
        sys.exit(1)

pid = os.fork()
if pid == 0:
    # Child = the worker process. Registers itself in the unit's cgroup
    # (pid survives exec), then becomes the wrapped command.
    try:
        try:
            with open(unit_path) as f:
                data = json.load(f)
        except Exception:
            data = {{"pids": []}}
        data["pids"] = sorted(set(data.get("pids", [])) | {{os.getpid()}})
        data["argv"] = cmd
        tmp = unit_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, unit_path)
    finally:
        os.execvp(cmd[0], cmd)

# Parent = the systemd-run client (the LAUNCHER): stays in the caller's
# process tree, waits for the scoped command, dies with its parent.
_, status = os.waitpid(pid, 0)
rc = os.waitstatus_to_exitcode(status)
try:
    with open(unit_path) as f:
        data = json.load(f)
    data["pids"] = [p for p in data.get("pids", []) if p != pid]
    tmp = unit_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, unit_path)
except Exception:
    pass
# Model transient-unit collection: --collect unloads on ANY completion;
# without it a successful unit still unloads when inactive — only FAILED
# ones stay loaded for inspection.  That asymmetry is what lets the
# spawn probe tell "ran and exited nonzero" from "launch refused".
if "--collect" in flags or rc == 0:
    try:
        os.unlink(unit_path)
    except OSError:
        pass
sys.exit(rc)
'''

_SYSTEMCTL_SHIM = f'''#!{sys.executable}
"""Fake systemctl --user: show/stop/kill/list-units over the state dir.

Dependency-light on purpose (see the systemd-run shim note): every scope
state probe shells out to this, so interpreter boot dominates its cost.
"""
import fnmatch, json, os, signal, sys, time

STATE = os.environ["HERMES_KANBAN_TEST_SHIM_STATE"]
UNITS = os.path.join(STATE, "units")


def log_action(action, unit):
    with open(os.path.join(STATE, "stops.jsonl"), "a") as f:
        f.write(json.dumps({{"action": action, "unit": unit}}) + "\\n")


def unit_file(unit):
    return os.path.join(UNITS, unit + ".json")


def load(unit):
    try:
        with open(unit_file(unit)) as f:
            return json.load(f)
    except Exception:
        return None


def save(unit, data):
    p = unit_file(unit)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, p)


def sticky_pids(unit):
    try:
        with open(os.path.join(STATE, "sticky", unit)) as f:
            return [int(x) for x in f.read().split()]
    except Exception:
        return []


def killproof(unit):
    return os.path.exists(os.path.join(STATE, "killproof", unit))


def slow_secs(unit, op):
    # A wedged helper CLIENT (pass 9, AH): the systemctl invocation
    # itself hangs this many seconds before doing anything — the exact
    # window a blocking subprocess.run(timeout=15) could not cancel
    # out of. Its pid is noted so tests can prove it was killed.
    try:
        with open(os.path.join(STATE, "slowop", unit + "." + op)) as f:
            return float(f.read().strip() or 0)
    except Exception:
        return 0.0


def note_slow_pid(unit, op):
    try:
        os.makedirs(os.path.join(STATE, "slowop"), exist_ok=True)
        with open(
            os.path.join(STATE, "slowop", unit + "." + op + ".pid"), "w"
        ) as f:
            f.write(str(os.getpid()))
    except Exception:
        pass


def deactivating(unit):
    return os.path.exists(os.path.join(STATE, "deactivating", unit))


def badcg(unit):
    # A loaded unit whose reported ControlGroup resolves to a path that
    # cannot be read (mis-derived prefix, custom mount, namespace).
    return os.path.exists(os.path.join(STATE, "badcg", unit))


def unit_pids(unit):
    data = load(unit)
    return (list(data.get("pids", [])) if data else []) + sticky_pids(unit)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def refresh_cgroup(unit):
    # cgroup.procs lists LIVE pids only (the kernel drops dead ones). A
    # unit that was never created (no json) has no cgroup at all.
    if load(unit) is None:
        return
    d = os.path.join(STATE, "cgroup", unit)
    os.makedirs(d, exist_ok=True)
    live = [p for p in unit_pids(unit) if alive(p)]
    with open(os.path.join(d, "cgroup.procs"), "w") as f:
        f.write("".join(str(p) + "\\n" for p in live))


def state_of(unit):
    # A scope is active iff its cgroup holds a live process (or the stop
    # job is wedged — killproof).
    if killproof(unit):
        return "active"
    if deactivating(unit):
        return "deactivating"
    if any(alive(p) for p in unit_pids(unit)):
        return "active"
    return "inactive"


argv = sys.argv[1:]
if argv and argv[0] == "--user":
    argv = argv[1:]
op = argv[0]

if op == "show":
    unit = argv[1]
    if load(unit) is None:
        print("LoadState=not-found")
        print("ActiveState=inactive")
        print("ControlGroup=")
    else:
        refresh_cgroup(unit)
        print("LoadState=loaded")
        print("ActiveState=" + state_of(unit))
        if badcg(unit):
            print("ControlGroup=" + os.path.join(STATE, "nonexistent", unit))
        else:
            print("ControlGroup=" + os.path.join(STATE, "cgroup", unit))
    sys.exit(0)

if op == "stop":
    unit = argv[1]
    log_action("stop", unit)
    slow = slow_secs(unit, "stop")
    if slow:
        note_slow_pid(unit, "stop")
        time.sleep(slow)
    data = load(unit)
    if killproof(unit):
        # Stop job accepted but never completes server-side: unit stays
        # active, systemctl still exits 0 (the "stop timeout" trap).
        sys.exit(0)
    if data is None:
        sys.stderr.write("Unit " + unit + " not loaded.\\n")
        sys.exit(1)
    for p in unit_pids(unit):
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + 0.8
    while time.monotonic() < deadline:
        if not any(alive(p) for p in unit_pids(unit)):
            break
        time.sleep(0.05)
    if not any(alive(p) for p in data.get("pids", [])):
        data["pids"] = []
        save(unit, data)
        # Sticky pids that died still count until kill clears them.
    refresh_cgroup(unit)
    sys.exit(0)

if op == "reset-failed":
    # Unload a dead/failed unit (explicit collection): the unit and its
    # cgroup vanish; a later show reports not-found.
    unit = argv[1]
    log_action("reset-failed", unit)
    try:
        os.unlink(unit_file(unit))
    except OSError:
        pass
    try:
        os.unlink(os.path.join(STATE, "cgroup", unit, "cgroup.procs"))
    except OSError:
        pass
    sys.exit(0)

if op == "kill":
    unit = argv[-1]
    log_action("kill", unit)
    slow = slow_secs(unit, "kill")
    if slow:
        note_slow_pid(unit, "kill")
        time.sleep(slow)
    data = load(unit)
    if killproof(unit):
        sys.exit(0)
    if data is None:
        sys.exit(1)
    for p in unit_pids(unit):
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    time.sleep(0.05)
    data["pids"] = [p for p in data.get("pids", []) if alive(p)]
    save(unit, data)
    refresh_cgroup(unit)
    sys.exit(0)

if op == "list-units":
    pattern = argv[-1]
    try:
        names = sorted(os.listdir(UNITS))
    except OSError:
        names = []
    for name in names:
        if not name.endswith(".scope.json"):
            continue
        unit = name[: -len(".json")]
        if not fnmatch.fnmatch(unit, pattern):
            continue
        st = state_of(unit)
        print(unit, "loaded", st, st, "Hermes kanban worker test scope")
    sys.exit(0)

sys.stderr.write("systemctl-test: unsupported " + repr(argv) + "\\n")
sys.exit(1)
'''

_CHILD_WAIT_PROGRAM = (
    "import os, time\n"
    "deadline = time.monotonic() + 120\n"
    "root = os.environ.get('HERMES_KANBAN_TEST_SHIM_STATE')\n"
    "while time.monotonic() < deadline and (root is None"
    " or os.path.exists(root)):\n"
    "    time.sleep(0.05)\n"
)

_STUBBORN_CHILD_WAIT_PROGRAM = (
    "import os, signal, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "deadline = time.monotonic() + 120\n"
    "root = os.environ.get('HERMES_KANBAN_TEST_SHIM_STATE')\n"
    "while time.monotonic() < deadline and (root is None"
    " or os.path.exists(root)):\n"
    "    time.sleep(0.05)\n"
)


class Shims:
    """Handle on the fake systemd user session."""

    def __init__(self, root: Path, bin_dir: Path):
        self.root = root
        self.bin = bin_dir
        self._extra_pids: list[int] = []

    # -- state -------------------------------------------------------------
    def unit_json(self, unit: str) -> dict | None:
        try:
            return json.loads((self.root / "units" / f"{unit}.json").read_text())
        except OSError:
            return None

    def write_unit(self, unit: str, pids: list[int]) -> None:
        d = self.root / "units"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{unit}.json").write_text(json.dumps({"pids": pids}))
        # Creating the unit creates its cgroup with those pids in it
        # (the shim's show/stop/kill keep the file fresh after this).
        cg = self.root / "cgroup" / unit
        cg.mkdir(parents=True, exist_ok=True)
        (cg / "cgroup.procs").write_text("".join(f"{p}\n" for p in pids))

    def stops(self) -> list[dict]:
        try:
            return [
                json.loads(line)
                for line in (self.root / "stops.jsonl").read_text().splitlines()
                if line.strip()
            ]
        except OSError:
            return []

    def cgroup_pids(self, unit: str) -> list[int]:
        """The unit's cgroup.procs contents (live pids after last refresh)."""
        try:
            return [
                int(x)
                for x in (self.root / "cgroup" / unit / "cgroup.procs")
                .read_text()
                .split()
            ]
        except OSError:
            return []

    # -- knobs ---------------------------------------------------------------
    def arm_fail_next(self, n: int = 1) -> None:
        (self.root / "fail_next").write_text(str(n))

    def arm_killproof(self, unit: str) -> None:
        d = self.root / "killproof"
        d.mkdir(parents=True, exist_ok=True)
        (d / unit).write_text("1")

    def clear_killproof(self, unit: str) -> None:
        (self.root / "killproof" / unit).unlink(missing_ok=True)

    def arm_deactivating(self, unit: str) -> None:
        """Model a stop job mid-flight: ActiveState=deactivating."""
        d = self.root / "deactivating"
        d.mkdir(parents=True, exist_ok=True)
        (d / unit).write_text("1")

    def arm_bad_cgroup_path(self, unit: str) -> None:
        """Loaded unit whose cgroup.procs path cannot be read."""
        d = self.root / "badcg"
        d.mkdir(parents=True, exist_ok=True)
        (d / unit).write_text("1")

    def clear_deactivating(self, unit: str) -> None:
        (self.root / "deactivating" / unit).unlink(missing_ok=True)

    def arm_slow_op(self, unit: str, op: str = "stop", seconds: float = 30.0) -> None:
        """Make the shim's systemctl <op> client hang *seconds* before
        acting (a wedged helper — the uncancellable window of AH)."""
        d = self.root / "slowop"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{unit}.{op}").write_text(str(seconds))

    def slow_op_pid(self, unit: str, op: str = "stop") -> int | None:
        try:
            return int((self.root / "slowop" / f"{unit}.{op}.pid").read_text())
        except (OSError, ValueError):
            return None

    def arm_sticky(self, unit: str, pid: int) -> None:
        d = self.root / "sticky"
        d.mkdir(parents=True, exist_ok=True)
        (d / unit).write_text(str(pid))

    # -- processes -----------------------------------------------------------
    def sleeper(self) -> int:
        """A disposable child standing in for a worker / descendant."""
        p = subprocess.Popen(
            [sys.executable, "-c", _CHILD_WAIT_PROGRAM],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._extra_pids.append(p.pid)
        return p.pid

    def stubborn_sleeper(self) -> int:
        """A child that ignores SIGTERM — a leaked dev server."""
        p = subprocess.Popen(
            [sys.executable, "-c", _STUBBORN_CHILD_WAIT_PROGRAM],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._extra_pids.append(p.pid)
        return p.pid

    def track(self, pid: int) -> None:
        self._extra_pids.append(pid)

    def teardown(self) -> None:
        # Kill scoped workers VIA THE SHIM (a separate process): a worker
        # whose launcher died is reparented outside this test's subtree,
        # and the conftest live-system guard rightly blocks direct
        # os.kill on out-of-subtree pids. The shim is our fake systemd —
        # teardown through it is exactly how production reclaims scopes.
        for unit_json in (self.root / "units").glob("*.json"):
            try:
                subprocess.run(
                    [
                        str(self.bin / "systemctl"),
                        "--user",
                        "kill",
                        "--signal=SIGKILL",
                        unit_json.name[: -len(".json")],
                    ],
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                pass
        for pid in self._extra_pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    def wait_for(self, predicate, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.05)
        return False


@pytest.fixture
def shims(tmp_path, monkeypatch):
    root = tmp_path / "systemd-state"
    (root / "units").mkdir(parents=True)
    bin_dir = tmp_path / "shim-bin"
    bin_dir.mkdir()
    (bin_dir / "systemd-run").write_text(_SYSTEMD_RUN_SHIM)
    (bin_dir / "systemctl").write_text(_SYSTEMCTL_SHIM)
    for f in bin_dir.iterdir():
        os.chmod(f, 0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HERMES_KANBAN_TEST_SHIM_STATE", str(root))
    monkeypatch.setattr(
        "tools.process_registry_scope._systemd_run_user_scope_available",
        lambda: True,
    )
    # A real subprocess worker: python waits on the shim-state file
    # (dies with the test's tmp dir / teardown kill); the trailing worker
    # argv (which follows the -c program) is inert sys.argv baggage.
    monkeypatch.setattr(
        _kanban_worker_spawn,
        "_resolve_hermes_argv",
        lambda: [sys.executable, "-c", _CHILD_WAIT_PROGRAM],
    )
    # Shrink the post-SIGKILL verify loop so wedged-stop tests stay fast.
    monkeypatch.setattr(
        "tools.process_registry_scope._SCOPE_STOP_VERIFY_TIMEOUT",
        0.5,
    )
    # Keep the launch-probe window short until the spawn path learns to
    # exit it early (see test_default_spawn_*): the shims fail/succeed
    # well inside a second.
    monkeypatch.setattr(_kanban_worker_scope, "WORKER_SPAWN_PROBE_SECONDS", 1.0)
    # Run the background verified-stop service inline: same code path,
    # but every tick observes a settled stop outcome — no thread-timing
    # races in single-tick assertions. Service state is per-test because
    # tests reuse unit names.
    monkeypatch.setattr(_kanban_worker_stop, "_scope_stop_inline", True)
    _kanban_worker_stop.reset_scope_stop_service_for_tests()
    handle = Shims(root, bin_dir)
    yield handle
    _kanban_worker_stop.reset_scope_stop_service_for_tests()
    handle.teardown()


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = _owner_kanban_db.kanban_db_path(board="default")
    _kanban_db_connect._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    _kanban_db_connect.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with _kanban_db_connect.connect() as c:
        yield c


def _make_task(task_id="t_scope1", title="build the widget", run_id=7):
    return _owner_kanban_db_models.Task(
        id=task_id,
        title=title,
        body=None,
        assignee="elias",
        status="running",
        priority=0,
        created_by="test",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=None,
        claim_lock="lock",
        claim_expires=None,
        tenant=None,
        current_run_id=run_id,
    )


def _patch_systemd_available(monkeypatch, available: bool):
    """Force the shared cached probe; kanban reads it at call time."""
    monkeypatch.setattr(
        "tools.process_registry_scope._systemd_run_user_scope_available",
        lambda: available,
    )


def _patch_systemd_run_binary(monkeypatch):
    """Pretend ``systemd-run`` exists — the builder re-runs which() itself."""
    real_which = shutil.which

    def fake_which(name, *args, **kwargs):
        if name == "systemd-run":
            return "/usr/bin/systemd-run"
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", fake_which)


def _fake_popen_capture(monkeypatch, captured, pid=4242, rc=None):
    class FakeProc:
        def __init__(self):
            self.pid = pid
            self.returncode = rc

        def poll(self):
            return self.returncode

    def fake_popen(cmd, *args, **kwargs):
        captured.setdefault("cmds", []).append(list(cmd))
        captured["kwargs"] = dict(kwargs)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)


def _write_kanban_config(home: Path, kanban_yaml: str):
    home.joinpath("config.yaml").write_text(f"kanban:\n{kanban_yaml}", encoding="utf-8")


def _capture_worker_argv(
    monkeypatch,
    tmp_path,
    kanban_yaml: str,
    *,
    systemd_available: bool,
    task: _owner_kanban_db_models.Task | None = None,
):
    """Spawn one worker with the given kanban config; returns
    ``(argv, spawned_pid)`` — the pid object carries the scope unit the
    call created (finding F: per-call channel, not a function attribute).

    Writes the config exactly once per call site (load_config caches on
    mtime/size, so rewrites within a test would be unreliable); tests that
    need several captures share one config and flip only the probe.
    """
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    _write_kanban_config(home, kanban_yaml)

    monkeypatch.setattr(
        _kanban_worker_spawn, "_resolve_hermes_argv", lambda: ["hermes"]
    )
    _patch_systemd_available(monkeypatch, systemd_available)
    if systemd_available:
        _patch_systemd_run_binary(monkeypatch)
    captured: dict = {}
    _fake_popen_capture(monkeypatch, captured)

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    result = _kanban_worker_spawn._default_spawn(task or _make_task(), str(workspace))
    return captured["cmds"][0], result


def _assert_plain_argv_shape(cmd: list[str]):
    """Today's (pre-isolation) worker argv shape, independent of which
    toolsets/model flags the config resolves: fixed hermes prefix, fixed
    chat suffix, and not one systemd token."""
    assert cmd[:5] == ["hermes", "-p", "elias", "--cli", "--accept-hooks"]
    assert cmd[-3:] == ["chat", "-q", "work kanban task t_scope1"]
    for token in (
        "systemd-run",
        "--user",
        "--scope",
        "--unit",
        "--collect",
        "--description",
        "--property",
        "MemoryAccounting",
    ):
        assert token not in cmd, f"systemd token {token!r} leaked into plain argv"


def _scoped_task_row(
    conn,
    *,
    scope: str,
    pid: int | None = None,
    started_delta: int = 0,
    claimer: str | None = None,
    registered: bool = False,
):
    """Insert a running task row pinned to a scope (test-scratch state)."""
    tid = kb.create_task(conn, title="scoped row", assignee="w")
    claimer = claimer or kb._claimer_id()
    _owner_kanban_claims.claim_task(conn, tid, claimer=claimer)
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=?, "
        "worker_pid_started_at=?, worker_registered_at=?, "
        "last_heartbeat_at=?, started_at=? WHERE id=?",
        (
            pid,
            scope,
            _kanban_worker_identity._worker_pid_start_time(pid) if pid else None,
            now if registered else None,
            now,
            now + started_delta,
            tid,
        ),
    )
    # The grace sweep measures from the ACTIVE RUN's started_at (it
    # outranks tasks.started_at) — age that too.
    conn.execute(
        "UPDATE task_runs SET started_at=? "
        "WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)",
        (now + started_delta, tid),
    )
    conn.commit()
    return tid


def _patch_managed_gateway(monkeypatch, *, managed: bool):
    """Force the restart-safe wrap's topology gate.

    ``restart_safe_gateway_child_argv`` wraps a gateway child only when the
    host is Linux AND this process owns a supervised gateway AND systemd set
    ``INVOCATION_ID``. These tests run on macOS, so ``_IS_LINUX`` is forced
    True in BOTH directions and the gateway-topology signals alone decide —
    otherwise "unmanaged" would pass for the wrong reason (the platform)
    and prove nothing.
    """
    monkeypatch.setattr("tools.process_registry_scope._IS_LINUX", True)
    monkeypatch.setattr(
        "tools.process_registry_scope._is_supervised_gateway_process",
        lambda: managed,
    )
    if managed:
        monkeypatch.setenv("INVOCATION_ID", "managed-gateway-test")
    else:
        monkeypatch.delenv("INVOCATION_ID", raising=False)


def _fake_refused_launch_popen(monkeypatch, calls, stderr_text: bytes):
    """First Popen models a systemd-run client that refuses the launch
    (writes to the spawn-stderr capture, exits non-zero); any later Popen
    is a live child. Returns nothing — assertions read *calls*."""

    class FakeProc:
        def __init__(self, rc):
            self.pid = 4141 if rc is not None else 4242
            self.returncode = rc

        def poll(self):
            return self.returncode

    def fake_popen(cmd, *args, **kwargs):
        calls.append(list(cmd))
        stderr = kwargs.get("stderr")
        if len(calls) == 1:
            if hasattr(stderr, "write"):
                stderr.write(stderr_text)
                stderr.flush()
            return FakeProc(1)
        return FakeProc(None)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)


def _refused_launch_setup(monkeypatch, tmp_path, *, managed: bool):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    _write_kanban_config(home, "  worker_isolation: auto\n")
    monkeypatch.setattr(
        _kanban_worker_spawn, "_resolve_hermes_argv", lambda: ["hermes"]
    )
    _patch_managed_gateway(monkeypatch, managed=managed)
    _patch_systemd_available(monkeypatch, True)
    _patch_systemd_run_binary(monkeypatch)
    # The launcher exited non-zero and the transient unit never came up:
    # the spawn path's definition of a REFUSED launch.
    monkeypatch.setattr(
        _kanban_worker_scope, "_scope_unit_created", lambda _unit: False
    )
    monkeypatch.setattr(
        _kanban_worker_scope,
        "_stop_kanban_worker_scope",
        lambda _unit, **_kwargs: True,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return workspace


def _spawnable_profile(kanban_home):
    profile = Path(kanban_home) / "profiles" / "elias"
    profile.mkdir(parents=True, exist_ok=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")


def _running_row(conn, tid, *, claimer, pid, pid_started, heartbeat):
    conn.execute(
        "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
        "worker_pid=?, worker_pid_started_at=?, last_heartbeat_at=? "
        "WHERE id=?",
        (claimer, int(time.time()) - 60, pid, pid_started, heartbeat, tid),
    )
    conn.execute(
        "UPDATE task_runs SET status='running', claim_lock=? "
        "WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)",
        (claimer, tid),
    )
    conn.commit()


def _deferred_handoff_row(shims, conn, unit, *, handoff, tid):
    """A running row whose own worker deferred a terminal handoff and
    exited (pass 9, AF scratch state): claim held by this host, worker
    pid already gone, defer grace expired."""
    launcher = subprocess.Popen(["true"])
    launcher.wait()
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute("UPDATE tasks SET worker_scope=? WHERE id=?", (unit, tid))
    conn.commit()
    _kanban_worker_handoff._defer_own_worker_handoff(
        conn,
        tid,
        unit,
        handoff,
        claim_lock=kb._claimer_id(),
    )
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "worker_pid_started_at=?, worker_registered_at=?, worker_scope=?, "
        "claim_expires=?, last_heartbeat_at=? WHERE id=?",
        (
            launcher.pid,
            _kanban_worker_identity._worker_pid_start_time(launcher.pid),
            now,
            unit,
            now - 60,
            now,
            tid,
        ),
    )
    conn.commit()


def _breaker_shaped_row(conn, unit, *, title="breaker shaped", assignee="elias"):
    """A drain-ceiling breaker's output row: blocked (needs_input), claim
    bookkeeping cleared, no 'blocked' event (so not sticky), and the
    scope deliberately retained for the operator."""
    tid = kb.create_task(conn, title=title, assignee=assignee)
    conn.execute(
        "UPDATE tasks SET status='blocked', block_kind='needs_input', "
        "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL, "
        "worker_pid_started_at=NULL, worker_registered_at=NULL, "
        "worker_scope=? WHERE id=?",
        (unit, tid),
    )
    conn.commit()
    return tid


def _max_runtime_row(conn, pid, scope, *, pid_started_at=None, registered=True):
    """A running row past its max_runtime, owned by this host's claimer."""
    tid = kb.create_task(conn, title="max runtime", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "worker_pid_started_at=?, worker_scope=?, worker_registered_at=?, "
        "max_runtime_seconds=1, started_at=? WHERE id=?",
        (
            pid,
            pid_started_at,
            scope,
            now if registered else None,
            now - 100,
            tid,
        ),
    )
    conn.execute(
        "UPDATE task_runs SET started_at = started_at - 9999 "
        "WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)",
        (tid,),
    )
    conn.commit()
    return tid


def _timed_out_payload(conn, tid) -> dict:
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'timed_out'",
        (tid,),
    ).fetchone()
    return json.loads(event["payload"]) if event else {}


def _load_dashboard_plugin():
    """Import plugins/kanban/dashboard/plugin_api.py as a fresh module
    (it is not a package import; the dashboard loads it by path)."""
    pytest.importorskip("fastapi")
    import importlib.util

    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_scope_test",
        plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _untracked_running_row(conn, *, pid: int | None = None, age: int = 0):
    """A running task whose worker scope was never recorded.

    The shape a run has after a pre-fix build spawned it on a managed
    gateway with ``worker_isolation: none``: live worker, live run, a
    recorded launcher pid, no registration yet, and a NULL
    ``worker_scope``. ``age`` puts the launch that far in the past (past
    the registration grace, where the sweep's interaction matters).
    Returns ``(task_id, run_id)``.
    """
    tid = kb.create_task(conn, title="untracked row", assignee="w")
    _owner_kanban_claims.claim_task(conn, tid, claimer=kb._claimer_id())
    started = int(time.time()) - age
    conn.execute(
        "UPDATE tasks SET status='running', worker_scope=NULL, "
        "worker_pid=?, worker_registered_at=NULL, "
        "last_heartbeat_at=?, started_at=? WHERE id=?",
        (pid, started, started, tid),
    )
    conn.execute(
        "UPDATE task_runs SET started_at=? "
        "WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)",
        (started, tid),
    )
    conn.commit()
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()[0]
    return tid, int(run_id)


_PRE_CHANGE_TASKS_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT,
    assignee             TEXT,
    status               TEXT NOT NULL,
    priority             INTEGER DEFAULT 0,
    created_by           TEXT,
    created_at           INTEGER NOT NULL,
    started_at           INTEGER,
    completed_at         INTEGER,
    workspace_kind       TEXT NOT NULL DEFAULT 'scratch',
    workspace_path       TEXT,
    branch_name          TEXT,
    project_id           TEXT,
    claim_lock           TEXT,
    claim_expires        INTEGER,
    tenant               TEXT,
    result               TEXT,
    idempotency_key      TEXT,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    worker_pid           INTEGER,
    last_failure_error   TEXT,
    max_runtime_seconds  INTEGER,
    last_heartbeat_at    INTEGER,
    current_run_id       INTEGER,
    workflow_template_id TEXT,
    current_step_key     TEXT,
    skills               TEXT,
    model_override       TEXT,
    provider_override    TEXT,
    reasoning_effort     TEXT,
    max_retries          INTEGER,
    goal_mode            INTEGER NOT NULL DEFAULT 0,
    goal_max_turns       INTEGER,
    session_id           TEXT,
    block_kind           TEXT,
    block_recurrences    INTEGER NOT NULL DEFAULT 0
);
"""

_PRE_CHANGE_TASK_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS task_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    profile             TEXT,
    step_key            TEXT,
    status              TEXT NOT NULL,
    claim_lock          TEXT,
    claim_expires       INTEGER,
    worker_pid          INTEGER,
    max_runtime_seconds INTEGER,
    last_heartbeat_at   INTEGER,
    started_at          INTEGER NOT NULL,
    ended_at            INTEGER,
    outcome             TEXT,
    summary             TEXT,
    metadata            TEXT,
    error               TEXT
);
"""

from hermes_cli import kanban_boards as _kanban_boards
from hermes_cli import kanban_db_connect as _kanban_db_connect
from hermes_cli import kanban_worker_handoff as _kanban_worker_handoff
from hermes_cli import kanban_worker_identity as _kanban_worker_identity
from hermes_cli import kanban_worker_scope as _kanban_worker_scope
from hermes_cli import kanban_worker_spawn as _kanban_worker_spawn
from hermes_cli import kanban_worker_stop as _kanban_worker_stop
