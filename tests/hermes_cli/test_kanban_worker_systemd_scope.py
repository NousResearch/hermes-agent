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
    kb.reset_scope_stop_service_for_tests()
    yield
    kb.reset_scope_stop_service_for_tests()


# ---------------------------------------------------------------------------
# PATH shims: fake systemd-run / systemctl over a temp state dir
# ---------------------------------------------------------------------------

# Shared state layout (root passed via HERMES_KANBAN_TEST_SHIM_STATE):
#   units/<unit>.json    {"pids": [...], "argv": [...]} — unit EXISTS on
#                        the bus (LoadState=loaded) while this file lives
#   cgroup/<unit>/cgroup.procs  the unit's cgroup LIVE pids, refreshed on
#                        every mutation and every show (the kernel drops
#                        dead pids from the real file automatically)
#   sticky/<unit>        extra pid(s) counted in the unit's liveness
#                        (a "leaked descendant" that ignores SIGTERM)
#   killproof/<unit>     stop/kill refuse to complete for this unit
#                        (a hung stop job: unit stays active)
#   deactivating/<unit>  ActiveState=deactivating (stop job draining)
#   fail_next            integer countdown: systemd-run refuses N launches
#   stops.jsonl          every stop/kill invocation, for assertions
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

# Child "worker" programs (K: no blind sleeps). Each child stands in for
# a worker or a leaked descendant: it stays alive until the test's
# shim-state root disappears (tmp_path teardown) or a 120 s cap expires —
# so a child the teardown kill misses exits by itself instead of idling
# for the full two minutes. Falls back to the plain cap when the env var
# is absent (a bare child spawned outside the shims).
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
            return json.loads(
                (self.root / "units" / f"{unit}.json").read_text()
            )
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
        (cg / "cgroup.procs").write_text(
            "".join(f"{p}\n" for p in pids)
        )

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
                for x in (
                    self.root / "cgroup" / unit / "cgroup.procs"
                ).read_text().split()
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

    def arm_sticky(self, unit: str, pid: int) -> None:
        d = self.root / "sticky"
        d.mkdir(parents=True, exist_ok=True)
        (d / unit).write_text(str(pid))

    # -- processes -----------------------------------------------------------
    def sleeper(self) -> int:
        """A disposable child standing in for a worker / descendant."""
        p = subprocess.Popen(
            [sys.executable, "-c", _CHILD_WAIT_PROGRAM],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._extra_pids.append(p.pid)
        return p.pid

    def stubborn_sleeper(self) -> int:
        """A child that ignores SIGTERM — a leaked dev server."""
        p = subprocess.Popen(
            [sys.executable, "-c", _STUBBORN_CHILD_WAIT_PROGRAM],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
                    [str(self.bin / "systemctl"), "--user", "kill",
                     "--signal=SIGKILL", unit_json.name[: -len(".json")]],
                    capture_output=True, timeout=10,
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
        "tools.process_registry._systemd_run_user_scope_available",
        lambda: True,
    )
    # A real subprocess worker: python waits on the shim-state file
    # (dies with the test's tmp dir / teardown kill); the trailing worker
    # argv (which follows the -c program) is inert sys.argv baggage.
    monkeypatch.setattr(
        kb, "_resolve_hermes_argv",
        lambda: [sys.executable, "-c", _CHILD_WAIT_PROGRAM],
    )
    # Shrink the post-SIGKILL verify loop so wedged-stop tests stay fast.
    monkeypatch.setattr(
        "tools.process_registry._SCOPE_STOP_VERIFY_TIMEOUT", 0.5,
    )
    # Keep the launch-probe window short until the spawn path learns to
    # exit it early (see test_default_spawn_*): the shims fail/succeed
    # well inside a second.
    monkeypatch.setattr(kb, "WORKER_SPAWN_PROBE_SECONDS", 1.0)
    # Run the background verified-stop service inline: same code path,
    # but every tick observes a settled stop outcome — no thread-timing
    # races in single-tick assertions. Service state is per-test because
    # tests reuse unit names.
    monkeypatch.setattr(kb, "_scope_stop_inline", True)
    kb.reset_scope_stop_service_for_tests()
    handle = Shims(root, bin_dir)
    yield handle
    kb.reset_scope_stop_service_for_tests()
    handle.teardown()


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def _make_task(task_id="t_scope1", title="build the widget", run_id=7):
    return kb.Task(
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
        "tools.process_registry._systemd_run_user_scope_available",
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
    home.joinpath("config.yaml").write_text(
        f"kanban:\n{kanban_yaml}", encoding="utf-8"
    )


def _capture_worker_argv(
    monkeypatch, tmp_path, kanban_yaml: str, *, systemd_available: bool,
    task: kb.Task | None = None,
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

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    _patch_systemd_available(monkeypatch, systemd_available)
    if systemd_available:
        _patch_systemd_run_binary(monkeypatch)
    captured: dict = {}
    _fake_popen_capture(monkeypatch, captured)

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    result = kb._default_spawn(task or _make_task(), str(workspace))
    return captured["cmds"][0], result


def _assert_plain_argv_shape(cmd: list[str]):
    """Today's (pre-isolation) worker argv shape, independent of which
    toolsets/model flags the config resolves: fixed hermes prefix, fixed
    chat suffix, and not one systemd token."""
    assert cmd[:5] == ["hermes", "-p", "elias", "--cli", "--accept-hooks"]
    assert cmd[-3:] == ["chat", "-q", "work kanban task t_scope1"]
    for token in ("systemd-run", "--user", "--scope", "--unit", "--collect",
                  "--description", "--property", "MemoryAccounting"):
        assert token not in cmd, f"systemd token {token!r} leaked into plain argv"


def _scoped_task_row(conn, *, scope: str, pid: int | None = None,
                     started_delta: int = 0, claimer: str | None = None,
                     registered: bool = False):
    """Insert a running task row pinned to a scope (test-scratch state)."""
    tid = kb.create_task(conn, title="scoped row", assignee="w")
    claimer = claimer or kb._claimer_id()
    kb.claim_task(conn, tid, claimer=claimer)
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=?, "
        "worker_pid_started_at=?, worker_registered_at=?, "
        "last_heartbeat_at=?, started_at=? WHERE id=?",
        (
            pid, scope,
            kb._worker_pid_start_time(pid) if pid else None,
            now if registered else None,
            now, now + started_delta, tid,
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


# ---------------------------------------------------------------------------
# Spawn wrapping
# ---------------------------------------------------------------------------

def test_default_spawn_wraps_argv_in_systemd_scope(monkeypatch, tmp_path):
    """Probe passes + isolation auto → systemd-run prefix with the
    run-suffixed unit name, description, per-worker memory properties, and
    the legacy argv intact after ``--``. The unwrapped baseline is captured
    from the same code path with the probe disabled, so the comparison is
    exact."""
    config = "  worker_isolation: auto\n  worker_memory_max_mb: 512\n"
    plain, plain_pid = _capture_worker_argv(
        monkeypatch, tmp_path, config, systemd_available=False
    )
    _assert_plain_argv_shape(plain)

    # Same config, probe now passes → same spawn, wrapped.
    wrapped, wrapped_pid = _capture_worker_argv(
        monkeypatch, tmp_path, config, systemd_available=True
    )

    assert wrapped[0] == "/usr/bin/systemd-run"
    # Flags before the command separator, in the builder's canonical order.
    head = wrapped[: wrapped.index("--")]
    assert head[1:5] == ["--user", "--scope", "--quiet", "--unit"]
    unit = kb._kanban_worker_scope_unit("t_scope1", 7)
    assert unit == "hermes-kanban-t_scope1-r7.scope"
    assert unit in head
    # No --collect: a fast nonzero worker exit must leave the failed
    # unit LOADED so the launch probe can tell "ran" from "refused"
    # (collection is explicit once the run is terminal).
    assert "--collect" not in head
    assert head[head.index("--description") + 1] == (
        "Hermes kanban worker t_scope1: build the widget"
    )
    assert head[head.index("--property") + 1] == "MemoryAccounting=yes"
    assert head[head.index("--property") + 3] == f"MemoryMax={512 * 1024 * 1024}"
    assert head[head.index("--property") + 5] == f"MemorySwapMax={512 * 1024 * 1024}"
    assert head[head.index("--property") + 7] == "OOMPolicy=kill"
    # Legacy argv preserved verbatim after the separator.
    assert wrapped[wrapped.index("--") + 1:] == plain
    # The spawn published the unit on the returned pid — a per-call
    # channel, usable verbatim as an int by every existing caller.
    assert isinstance(wrapped_pid, int)
    assert wrapped_pid.scope_unit == unit
    assert plain_pid.scope_unit == ""


def test_default_spawn_memory_default_derives_both_bounds(monkeypatch, tmp_path):
    """No explicit worker_memory_max_mb → BOTH MemoryMax and MemorySwapMax
    come from the shared process-registry helper, at the same value."""
    wrapped = _capture_worker_argv(
        monkeypatch, tmp_path, "  worker_isolation: auto\n",
        systemd_available=True,
    )
    head = wrapped[0][: wrapped[0].index("--")]
    expected = kb._kanban_worker_memory_bytes()
    assert expected, "helper must return a positive bound on a normal host"
    props = [p for p in head if p.startswith("MemoryMax=")]
    assert props == [f"MemoryMax={expected}"], props
    swap = [p for p in head if p.startswith("MemorySwapMax=")]
    assert swap == [f"MemorySwapMax={expected}"], swap


def test_default_spawn_omits_memory_when_helper_falsy(monkeypatch, tmp_path):
    """A helper that cannot compute a bound (returns 0/None) → BOTH memory
    properties omitted. MemoryMax=0 means 'no limit' in systemd — the exact
    opposite of the intended bound — so emitting it is worse than nothing."""
    monkeypatch.setattr(
        "tools.process_registry._worker_memory_max_bytes", lambda: 0
    )
    kb._memory_bound_omitted_warned = False
    try:
        wrapped = _capture_worker_argv(
            monkeypatch, tmp_path, "  worker_isolation: auto\n",
            systemd_available=True,
        )
    finally:
        kb._memory_bound_omitted_warned = False
    head = wrapped[0][: wrapped[0].index("--")]
    assert not [p for p in head if p.startswith("MemoryMax=")]
    assert not [p for p in head if p.startswith("MemorySwapMax=")]


def test_default_spawn_none_keeps_legacy_argv_exactly(monkeypatch, tmp_path):
    """isolation 'none' must produce today's argv byte-for-byte, even when
    systemd is fully available — the rollback contract."""
    none_cmd, none_pid = _capture_worker_argv(
        monkeypatch, tmp_path, "  worker_isolation: none\n",
        systemd_available=True,
    )
    _assert_plain_argv_shape(none_cmd)
    # 'none' ignores availability: a second capture with the probe down
    # (the classic macOS/container host) is byte-identical.
    fallback_cmd, fallback_pid = _capture_worker_argv(
        monkeypatch, tmp_path, "  worker_isolation: none\n",
        systemd_available=False,
    )
    assert fallback_cmd == none_cmd
    assert none_pid.scope_unit == ""
    assert fallback_pid.scope_unit == ""


def test_default_spawn_auto_without_systemd_keeps_legacy_argv(monkeypatch, tmp_path):
    """Unusable systemd (macOS / containers) with the default 'auto' mode
    silently falls back to the plain argv — no behavioural change."""
    cmd, pid = _capture_worker_argv(
        monkeypatch, tmp_path, "  worker_isolation: auto\n",
        systemd_available=False,
    )
    _assert_plain_argv_shape(cmd)
    assert pid.scope_unit == ""


def test_spawn_scope_unit_is_per_call_not_global(monkeypatch, tmp_path):
    """F: the scope unit travels on the spawn's RETURN VALUE, not on a
    process-global function attribute. Per-board dispatches run
    concurrently, so the old global let board B's spawn overwrite the
    unit board A's dispatcher was about to record. Here: spawn A, then
    spawn B, then read A's unit — under the old global that read
    returned B's unit; per-call it is stable and each pid works as a
    plain int for every caller contract."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    _write_kanban_config(home, "  worker_isolation: auto\n")
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    _patch_systemd_available(monkeypatch, True)
    _patch_systemd_run_binary(monkeypatch)
    captured: dict = {}
    _fake_popen_capture(monkeypatch, captured, pid=4242)

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    pid_a = kb._default_spawn(
        _make_task(task_id="t_board_a", run_id=11), str(workspace),
    )
    pid_b = kb._default_spawn(
        _make_task(task_id="t_board_b", run_id=22), str(workspace),
    )

    unit_a = kb._kanban_worker_scope_unit("t_board_a", 11)
    unit_b = kb._kanban_worker_scope_unit("t_board_b", 22)
    assert unit_a != unit_b
    # A's unit survives B's spawn — the cross-wire the global allowed.
    assert pid_a.scope_unit == unit_a
    assert pid_b.scope_unit == unit_b
    # The annotated pid is a real int for every existing caller
    # contract (truthiness, int(), str(), arithmetic).
    assert pid_a == 4242
    assert int(pid_a) == 4242
    assert str(pid_a) == "4242"
    # And the old global channel no longer exists to cross-wire.
    assert not hasattr(kb._default_spawn, "_last_scope_unit")


def test_forced_scope_without_systemd_refuses_spawn(monkeypatch, tmp_path):
    """H: 'systemd-scope' + unusable probe = REFUSED spawn, never a silent
    unisolated fallback. The operator pinned strict — "no worker" beats an
    unisolated worker; only 'auto' may fall back. The refusal raises with
    the operator-facing reason BEFORE any process is launched."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    _write_kanban_config(home, "  worker_isolation: systemd-scope\n")
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    _patch_systemd_available(monkeypatch, False)
    captured: dict = {}
    _fake_popen_capture(monkeypatch, captured)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)

    with pytest.raises(RuntimeError, match="worker_isolation=systemd-scope"):
        kb._default_spawn(_make_task(), str(workspace))

    # Nothing was ever launched — neither isolated nor unisolated.
    assert captured == {}


def test_forced_scope_vanished_binary_refuses_spawn(monkeypatch, tmp_path):
    """H, second gap: the probe passed but systemd-run disappeared from
    PATH before the argv build, so the builder returned the argv
    unwrapped. Strict mode refuses here too — an unwrapped argv must not
    become a silent unisolated spawn one code path away from the probe
    refusal."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    _write_kanban_config(home, "  worker_isolation: systemd-scope\n")
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    _patch_systemd_available(monkeypatch, True)
    real_which = shutil.which

    def gone_which(name, *args, **kwargs):
        if name == "systemd-run":
            return None
        return real_which(name, *args, **kwargs)

    monkeypatch.setattr(shutil, "which", gone_which)
    captured: dict = {}
    _fake_popen_capture(monkeypatch, captured)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)

    with pytest.raises(RuntimeError, match="disappeared between"):
        kb._default_spawn(_make_task(), str(workspace))

    assert captured == {}


# ---------------------------------------------------------------------------
# Dispatcher bookkeeping + launcher-vs-worker pid (real shims)
# ---------------------------------------------------------------------------

def test_set_worker_pid_records_scope_and_start_fingerprint(conn):
    """The pid, its start-time fingerprint (PID-reuse guard), and the scope
    unit land on both the task row and the active run; the ``spawned``
    event carries the scope for operators. A scoped row is NOT registered
    (the pid is the launcher's); a plain spawn is registered immediately —
    its pid IS the worker."""
    tid = kb.create_task(conn, title="record", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())

    live = os.getpid()
    unit = kb._kanban_worker_scope_unit(tid, None)
    kb._set_worker_pid(conn, tid, live, scope_unit=unit)

    row = conn.execute(
        "SELECT worker_pid, worker_pid_started_at, worker_scope, "
        "       worker_registered_at FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_pid"] == live
    assert row["worker_pid_started_at"] == kb._worker_pid_start_time(live)
    assert row["worker_scope"] == unit
    assert row["worker_registered_at"] is None  # launcher pid, not worker

    run = conn.execute(
        "SELECT worker_pid, worker_scope FROM task_runs "
        "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
        (tid,),
    ).fetchone()
    assert run["worker_pid"] == live
    assert run["worker_scope"] == unit

    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'spawned'",
        (tid,),
    ).fetchone()
    assert event and json.loads(event["payload"])["scope"] == unit

    # Plain spawn: no scope → registered at spawn time.
    tid2 = kb.create_task(conn, title="plain", assignee="w")
    kb.claim_task(conn, tid2, claimer=kb._claimer_id())
    kb._set_worker_pid(conn, tid2, live, scope_unit="")
    row2 = conn.execute(
        "SELECT worker_registered_at FROM tasks WHERE id = ?", (tid2,)
    ).fetchone()
    assert row2["worker_registered_at"] is not None


def _spawnable_profile(kanban_home):
    profile = Path(kanban_home) / "profiles" / "elias"
    profile.mkdir(parents=True, exist_ok=True)
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")


def test_dispatch_records_launcher_pid_and_worker_self_registers(
    shims, conn, kanban_home
):
    """End-to-end with the shims: the dispatcher records the LAUNCHER pid
    and the run-suffixed scope; the unit's cgroup holds a DIFFERENT (worker)
    pid; ``register_worker_pid`` (what the heartbeat bridge calls from
    inside the worker) overwrites the launcher pid with the worker's and
    flips ``worker_registered_at`` with a ``worker_registered`` event."""
    _spawnable_profile(kanban_home)
    tid = kb.create_task(conn, title="scoped", assignee="elias")
    result = kb.dispatch_once(conn, dry_run=False)

    assert [s[0] for s in result.spawned] == [tid]
    row = conn.execute(
        "SELECT worker_pid, worker_pid_started_at, worker_scope, "
        "       worker_registered_at, current_run_id FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    launcher_pid = row["worker_pid"]
    unit = row["worker_scope"]
    assert unit == kb._kanban_worker_scope_unit(tid, row["current_run_id"])
    assert "-r" in unit  # run-suffixed: unique per attempt
    assert row["worker_registered_at"] is None  # starting, not registered

    # The launcher survived its probe window (no spawn failure recorded).
    assert result.late_spawn_failed == []

    # The unit's cgroup holds the worker pid — a different process.
    data = shims.unit_json(unit)
    assert data is not None
    worker_pids = [p for p in data["pids"] if p != launcher_pid]
    assert len(worker_pids) == 1
    worker_pid = worker_pids[0]
    assert worker_pid != launcher_pid
    assert kb._pid_alive(worker_pid)

    # Worker-side self-registration (the heartbeat bridge's call).
    assert kb.register_worker_pid(
        conn, tid, expected_run_id=row["current_run_id"], pid=worker_pid,
    )
    after = conn.execute(
        "SELECT worker_pid, worker_registered_at FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert after["worker_pid"] == worker_pid
    assert after["worker_registered_at"] is not None
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? "
        "AND kind = 'worker_registered'", (tid,),
    ).fetchone()
    assert event and json.loads(event["payload"])["pid"] == worker_pid


def test_heartbeat_bridge_registers_worker_pid(shims, conn, kanban_home, monkeypatch):
    """The auto-heartbeat bridge (run from INSIDE the worker process on
    first activity) registers the calling process's own pid."""
    from tools import kanban_tools as kt

    tid = kb.create_task(conn, title="bridge", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["current_run_id"]

    db_path = kb.kanban_db_path(board="default")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kt._auto_heartbeat_last_attempt = 0.0
    assert kt.heartbeat_current_worker_from_env() is True

    row = conn.execute(
        "SELECT worker_pid, worker_pid_started_at, worker_registered_at, "
        "       last_heartbeat_at FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_pid"] == os.getpid()  # THIS process = the worker
    assert row["worker_registered_at"] is not None
    assert row["worker_pid_started_at"] == kb._worker_pid_start_time(os.getpid())
    assert row["last_heartbeat_at"] is not None

    # The explicit tool path registers too (fresh task, direct call).
    tid2 = kb.create_task(conn, title="tool", assignee="w")
    kb.claim_task(conn, tid2, claimer=kb._claimer_id())
    run2 = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid2,)
    ).fetchone()["current_run_id"]
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid2)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run2))
    out = kt._handle_heartbeat({"task_id": tid2})
    assert '"ok"' in out or '"status"' in out
    row2 = conn.execute(
        "SELECT worker_pid, worker_registered_at FROM tasks WHERE id = ?",
        (tid2,),
    ).fetchone()
    assert row2["worker_pid"] == os.getpid()
    assert row2["worker_registered_at"] is not None


# ---------------------------------------------------------------------------
# Re-adoption after gateway restart
# ---------------------------------------------------------------------------

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


def test_adopt_surviving_worker_rewrites_claim_and_run_continues(conn):
    """A live, freshly-heartbeating worker owned by the previous gateway
    pid is re-adopted: claim moves to this claimer, run stays running, no
    failure counted, and crash detection leaves it alone."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="survivor", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:4194304")
    live = os.getpid()
    _running_row(
        conn, tid,
        claimer=f"{host}:4194304",
        pid=live,
        pid_started=kb._worker_pid_start_time(live),
        heartbeat=int(time.time()),
    )

    adopted = kb.adopt_surviving_running_workers(conn)
    assert adopted == [tid]

    row = conn.execute(
        "SELECT status, claim_lock, claim_expires, consecutive_failures "
        "FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    assert row["claim_expires"] > int(time.time())
    assert row["consecutive_failures"] == 0

    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'adopted'",
        (tid,),
    ).fetchone()
    payload = json.loads(event["payload"])
    assert payload["previous_claimer"] == f"{host}:4194304"
    assert payload["claimer"] == kb._claimer_id()

    # The adopted run is not a crash: detection must skip it entirely.
    assert kb.detect_crashed_workers(conn) == []
    # Idempotent: a second pass finds nothing to adopt.
    assert kb.adopt_surviving_running_workers(conn) == []


def test_adoption_skips_stale_heartbeat(conn):
    """Alive pid but no observable progress for > 1h → NOT adopted; the
    existing stale paths own that case."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="wedged", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:4194304")
    live = os.getpid()
    _running_row(
        conn, tid,
        claimer=f"{host}:4194304",
        pid=live,
        pid_started=kb._worker_pid_start_time(live),
        heartbeat=int(time.time()) - 7200,
    )

    assert kb.adopt_surviving_running_workers(conn) == []
    row = conn.execute(
        "SELECT claim_lock FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["claim_lock"] == f"{host}:4194304"


def test_dead_pid_still_crashes_and_counts_failure(conn):
    """Adoption must not rescue a genuinely dead worker: crash detection
    still fires, marks the run crashed, and counts the failure."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="crashed", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:4194304")
    dead = subprocess.Popen(["true"])
    dead.wait()
    _running_row(
        conn, tid,
        claimer=f"{host}:4194304",
        pid=dead.pid,
        pid_started=None,
        heartbeat=int(time.time()),
    )
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.commit()

    assert kb.adopt_surviving_running_workers(conn) == []
    crashed = kb.detect_crashed_workers(conn)
    assert tid in crashed
    row = conn.execute(
        "SELECT status, consecutive_failures FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["consecutive_failures"] >= 1


def test_recycled_pid_is_not_mistaken_for_the_worker(conn):
    """PID-reuse guard: a live but different process at the recorded pid is
    a dead worker, not a survivor — crash detection fires with the reuse
    flag, and adoption never claims it."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="reused", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:4194304")
    # A very much alive process (this test) whose start-time fingerprint
    # differs from the recorded one — exactly what pid reuse looks like.
    _running_row(
        conn, tid,
        claimer=f"{host}:4194304",
        pid=os.getpid(),
        pid_started=12345,
        heartbeat=int(time.time()),
    )
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.commit()

    assert kb.adopt_surviving_running_workers(conn) == []
    crashed = kb.detect_crashed_workers(conn)
    assert tid in crashed

    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'crashed'",
        (tid,),
    ).fetchone()
    payload = json.loads(event["payload"])
    assert payload.get("pid_reused") is True


def test_killing_the_launcher_does_not_kill_the_run(shims, conn, kanban_home):
    """THE launcher-vs-worker contract, end-to-end: a gateway death kills
    the systemd-run launcher (a child of the gateway) while the scoped
    worker survives — re-adoption must follow the registered worker via
    scope truth, and crash detection must NOT fire on the dead launcher."""
    _spawnable_profile(kanban_home)
    tid = kb.create_task(conn, title="restart survivor", assignee="elias")
    kb.dispatch_once(conn, dry_run=False)

    row = conn.execute(
        "SELECT worker_pid, worker_scope, current_run_id FROM tasks "
        "WHERE id = ?", (tid,)
    ).fetchone()
    launcher_pid = row["worker_pid"]
    unit = row["worker_scope"]
    worker_pid = shims.unit_json(unit)["pids"][0]
    assert worker_pid != launcher_pid

    # The worker self-registers (first activity after spawn).
    assert kb.register_worker_pid(
        conn, tid, expected_run_id=row["current_run_id"], pid=worker_pid,
    )
    # Simulate the gateway dying: only the LAUNCHER dies.
    os.kill(launcher_pid, signal.SIGKILL)
    assert shims.wait_for(
        lambda: not kb._pid_alive(launcher_pid), timeout=5.0
    )
    assert kb._pid_alive(worker_pid)  # the scoped worker survives

    # The claim now names a dead gateway; heartbeat is fresh.
    host = kb._claimer_id().split(":", 1)[0]
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET claim_lock=?, claim_expires=?, "
        "last_heartbeat_at=? WHERE id=?",
        (f"{host}:4194304", now - 60, now, tid),
    )
    conn.commit()

    assert kb.detect_crashed_workers(conn) == []  # scope truth: alive
    adopted = kb.adopt_surviving_running_workers(conn)
    assert adopted == [tid]
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'adopted'",
        (tid,),
    ).fetchone()
    payload = json.loads(event["payload"])
    assert payload.get("verified_by") == "scope_active"
    row = conn.execute(
        "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    assert row["worker_pid"] == worker_pid


# ---------------------------------------------------------------------------
# Launch-grace sweep: silent spawn failures
# ---------------------------------------------------------------------------

def test_unregistered_within_grace_is_left_alone(shims, conn):
    """A scoped run inside its launch grace window is 'starting', not dead —
    the sweep must not fail it."""
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_grace", 1)
    shims.write_unit(unit, [pid])
    tid = _scoped_task_row(conn, scope=unit, pid=pid, started_delta=0)
    assert kb.fail_unregistered_workers(conn) == []
    row = conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "running"


def test_unregistered_past_grace_fails_as_spawn_failed(shims, conn):
    """A scoped run that never registered past the grace window is a silent
    launch failure: spawn_failed run, failure counted, scope stopped."""
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_deadspawn", 1)
    shims.write_unit(unit, [pid])
    # The launcher pid is recorded (as every scoped spawn does) but the
    # worker never registered; the run started long before the grace
    # window closed.
    tid = _scoped_task_row(
        conn, scope=unit, pid=os.getpid(),
        started_delta=-kb.WORKER_REGISTRATION_GRACE_SECONDS - 300,
    )
    failed = kb.fail_unregistered_workers(conn)
    assert failed == [tid]
    row = conn.execute(
        "SELECT status, consecutive_failures FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] != "running"
    assert row["consecutive_failures"] >= 1
    run = kb.latest_run(conn, tid)
    assert run is not None and run.outcome == "spawn_failed"


def test_registration_race_mid_stop_keeps_task_running(shims, conn, monkeypatch):
    """F (new a), the race: the worker registers BETWEEN the sweep's
    snapshot and its verified stop. The write-lock re-check plus the CAS
    in the failure record keep the row running — no spawn_failed, no
    failure counted, the registration survives."""
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_race", 1)
    shims.write_unit(unit, [pid])
    tid = _scoped_task_row(
        conn, scope=unit, pid=os.getpid(),
        started_delta=-kb.WORKER_REGISTRATION_GRACE_SECONDS - 300,
    )
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["current_run_id"]

    # The worker's first heartbeat lands while the sweep is stopping the
    # scope — exactly the review's snapshot-vs-stop window.
    real_stop = kb.request_worker_scope_stop

    def register_then_stop(unit_name, *, task_id=None, **kwargs):
        kb.register_worker_pid(
            conn, task_id, expected_run_id=run_id, pid=pid,
        )
        return real_stop(unit_name, task_id=task_id, **kwargs)

    monkeypatch.setattr(kb, "request_worker_scope_stop", register_then_stop)

    assert kb.fail_unregistered_workers(conn) == []
    row = conn.execute(
        "SELECT status, consecutive_failures, worker_pid, "
        "       worker_registered_at FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["consecutive_failures"] == 0
    assert row["worker_pid"] == pid  # the registration was not overwritten
    assert row["worker_registered_at"] is not None
    run = kb.latest_run(conn, tid)
    assert run.outcome is None  # the open run was never closed as failed


def test_queued_stop_skips_when_worker_registers_first(
    shims, conn, monkeypatch,
):
    """J: the worker registers AFTER the sweep's pre-stop check but
    BEFORE the queued verified stop runs on the service thread. The stop
    re-checks registration immediately before acting and stands down —
    the CAS already prevented the spawn_failed record; without this the
    queued stop still killed the legitimate worker."""
    import threading

    monkeypatch.setattr(kb, "_scope_stop_inline", False)
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_qrace", 1)
    shims.write_unit(unit, [pid])
    tid = _scoped_task_row(
        conn, scope=unit, pid=os.getpid(),
        started_delta=-kb.WORKER_REGISTRATION_GRACE_SECONDS - 300,
    )
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["current_run_id"]

    # Hold the service on a decoy unit so the real request sits QUEUED
    # (not in flight) while the worker registers — exactly the finding's
    # window between the pre-stop check and the queued stop running.
    gate = threading.Event()
    stops: list[str] = []

    def gated_stop(unit_name):
        stops.append(unit_name)
        gate.wait(timeout=5.0)
        return True

    monkeypatch.setattr(kb, "_stop_kanban_worker_scope", gated_stop)
    decoy = kb._kanban_worker_scope_unit("t_decoy", 1)
    shims.write_unit(decoy, [shims.sleeper()])  # active, so it queues
    assert not kb.request_worker_scope_stop(decoy)  # queued, now in flight
    assert shims.wait_for(lambda: stops == [decoy])

    # The sweep: pre-stop check sees an unregistered row, queues its
    # stop behind the decoy, and (not confirmed this tick) fails nothing.
    assert kb.fail_unregistered_workers(conn) == []

    # The worker's first heartbeat lands while the stop is still queued.
    kb.register_worker_pid(
        conn, tid, expected_run_id=run_id, pid=pid,
    )

    gate.set()  # release the decoy — the service reaches the real unit
    kb.join_scope_stop_service(timeout=5.0)

    # The queued stop stood down: the decoy was stopped, the real unit's
    # verified stop never ran, and the worker lives on.
    assert stops == [decoy]
    assert kb._pid_alive(pid)
    row = conn.execute(
        "SELECT status, consecutive_failures, worker_registered_at "
        "FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["consecutive_failures"] == 0
    assert row["worker_registered_at"] is not None
    # And the next sweep agrees: a registered row is not its business.
    assert kb.fail_unregistered_workers(conn) == []
    kb.reset_scope_stop_service_for_tests()


def test_queued_stop_cas_wins_registration_self_aborts(
    shims, conn, monkeypatch,
):
    """R, marker wins: the worker's first heartbeat lands AFTER the
    drain's re-check but BEFORE the signal — the exact window a plain
    read cannot close. The stop-pending CAS has already committed by
    then, so the registration self-aborts (no half-registered row) and
    the stop proceeds on the unregistered-launch verdict. The marker
    served its purpose for exactly the signal window: once the stop
    CONFIRMS (pass 8, AC) the service clears it, so a later re-adoption
    of the run can register again."""
    import threading

    monkeypatch.setattr(kb, "_scope_stop_inline", False)
    gate = threading.Event()

    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_casrace", 1)
    shims.write_unit(unit, [pid])
    tid = _scoped_task_row(
        conn, scope=unit, pid=os.getpid(),
        started_delta=-kb.WORKER_REGISTRATION_GRACE_SECONDS - 300,
    )
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["current_run_id"]

    registration: dict = {}
    stops: list[str] = []

    def register_then_confirm(unit_name):
        # The heartbeat lands at the exact instant between the drain's
        # re-check and the signal. It runs on the service thread, so it
        # takes its own connection (sqlite connections are per-thread).
        stops.append(unit_name)
        with kb.connect() as c:
            registration["ok"] = kb.register_worker_pid(
                c, tid, expected_run_id=run_id, pid=pid,
            )
        gate.set()
        return True

    monkeypatch.setattr(kb, "_stop_kanban_worker_scope", register_then_confirm)
    assert not kb.request_worker_scope_stop(
        unit, task_id=tid, skip_if_registered=True,
    )
    kb.join_scope_stop_service(timeout=5.0)

    assert gate.is_set()          # the stop (and thus the race) ran
    assert stops == [unit]
    assert registration["ok"] is False  # self-aborted on the marker
    row = conn.execute(
        "SELECT status, worker_pid, worker_registered_at FROM tasks "
        "WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["worker_pid"] == os.getpid()   # launcher pid kept
    assert row["worker_registered_at"] is None  # no half-registration
    marked = conn.execute(
        "SELECT stop_pending FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()["stop_pending"]
    # Pass 8 (AC): the fake stop reported verified, so the confirmed
    # stop retired the marker it had just set — the signal window is
    # over, and the run row must not carry the marker past it.
    assert marked == 0
    kb.reset_scope_stop_service_for_tests()


def test_queued_stop_stands_down_when_registration_wins_cas(
    shims, conn, monkeypatch,
):
    """R, registration wins: it commits AFTER the drain's re-check read
    (the read missed it by a hair) but BEFORE the stop-pending CAS. The
    CAS excludes registered rows, matches nothing, and the stop stands
    down — the worker lives on despite the stale read."""
    monkeypatch.setattr(kb, "_scope_stop_inline", False)
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_caslost", 1)
    shims.write_unit(unit, [pid])
    tid = _scoped_task_row(
        conn, scope=unit, pid=os.getpid(),
        started_delta=-kb.WORKER_REGISTRATION_GRACE_SECONDS - 300,
    )
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["current_run_id"]

    # Registration commits first (its own connection, as the worker's
    # heartbeat would)…
    with kb.connect() as c:
        assert kb.register_worker_pid(
            c, tid, expected_run_id=run_id, pid=pid,
        )
    # …but the drain's read re-check misses it — the race the CAS closes.
    monkeypatch.setattr(kb, "_task_has_registered_worker", lambda _tid: False)

    stops: list[str] = []

    def never_stop(unit_name):
        stops.append(unit_name)
        return True

    monkeypatch.setattr(kb, "_stop_kanban_worker_scope", never_stop)
    assert not kb.request_worker_scope_stop(
        unit, task_id=tid, skip_if_registered=True,
    )
    kb.join_scope_stop_service(timeout=5.0)

    assert stops == []            # the CAS stood the stop down
    assert kb._pid_alive(pid)
    row = conn.execute(
        "SELECT status, worker_registered_at FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["worker_registered_at"] is not None
    marked = conn.execute(
        "SELECT stop_pending FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()["stop_pending"]
    assert marked is None         # registered rows never get marked
    kb.reset_scope_stop_service_for_tests()


def test_reregistration_same_pid_new_fingerprint_rejected(
    shims, conn, monkeypatch,
):
    """F (new a), the reused pid: a second registration presenting the
    SAME numeric pid but a DIFFERENT start fingerprint is the kernel
    having recycled the number — rejected with a log, and the recorded
    registration is left exactly as the real worker wrote it."""
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_reuse", 1)
    shims.write_unit(unit, [pid])
    tid = _scoped_task_row(conn, scope=unit, pid=os.getpid())
    run_id = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["current_run_id"]

    fingerprints = iter([111, 222, 111])
    monkeypatch.setattr(
        kb, "_worker_pid_start_time", lambda _pid: next(fingerprints),
    )

    assert kb.register_worker_pid(conn, tid, expected_run_id=run_id, pid=pid)
    assert not kb.register_worker_pid(
        conn, tid, expected_run_id=run_id, pid=pid,
    )  # same pid number, new fingerprint — a recycled pid, not our worker
    row = conn.execute(
        "SELECT worker_pid_started_at FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    assert row["worker_pid_started_at"] == 111
    # A true re-registration (same pid, same fingerprint) stays accepted.
    assert kb.register_worker_pid(conn, tid, expected_run_id=run_id, pid=pid)


def test_unscoped_rows_are_normalized_not_failed(shims, conn):
    """Unscoped running rows (plain spawns — and every legacy row from
    before the isolation feature, which are unscoped by definition) get
    ``worker_registered_at`` backfilled: their recorded pid IS the worker.
    Nothing is failed."""
    tid = kb.create_task(conn, title="plain legacy", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "worker_pid_started_at=?, worker_registered_at=NULL, "
        "started_at=? WHERE id=?",
        (os.getpid(), kb._worker_pid_start_time(os.getpid()),
         now - 99999, tid),
    )
    conn.commit()

    assert kb.fail_unregistered_workers(conn) == []
    row = conn.execute(
        "SELECT status, worker_registered_at FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["worker_registered_at"] is not None

    # Idempotent: a second sweep finds nothing to do and fails nothing.
    assert kb.fail_unregistered_workers(conn) == []


# ---------------------------------------------------------------------------
# Verified stops + deferral when the stop cannot be confirmed
# ---------------------------------------------------------------------------

def test_scope_liveness_is_cgroup_procs_truth(shims):
    """Alive/dead comes from the unit's cgroup.procs, not ActiveState
    alone: a live pid → alive; the pid dying (kernel drops it from the
    cgroup) → dead even though the unit is still loaded; a unit that was
    never created (LoadState=not-found) → dead."""
    from tools import process_registry as pr

    never = kb._kanban_worker_scope_unit("t_never", 1)
    assert pr._scope_unit_liveness(never) == "dead"
    assert pr._scope_unit_active_state(never) == "dead"

    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_cg", 1)
    shims.write_unit(unit, [pid])
    assert pid in shims.cgroup_pids(unit)
    assert pr._scope_unit_liveness(unit) == "alive"
    assert pr._scope_unit_active_state(unit) == "active"

    # The pid dies outside any stop: the cgroup empties, the unit stays
    # loaded — still dead, because pids are the truth.
    os.kill(pid, signal.SIGKILL)
    assert shims.wait_for(
        lambda: pr._scope_unit_liveness(unit) == "dead"
    ), "kernel-dropped pid must read as dead via cgroup.procs"


def test_scope_liveness_deactivating_and_query_failure_are_not_dead(shims):
    """ActiveState=deactivating (stop job draining) and an unreachable
    systemctl are 'unknown' — never 'dead': releasing a claim on either
    would let the dispatcher spawn beside a still-draining scope."""
    from tools import process_registry as pr

    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_drain", 1)
    shims.write_unit(unit, [pid])
    shims.arm_deactivating(unit)
    assert pr._scope_unit_liveness(unit) == "unknown"
    assert pr._scope_unit_active_state(unit) == "unknown"

    # systemctl gone entirely (a wedged user bus): unknown, and the
    # verified stop reports failure instead of assuming success.
    real_which = shutil.which

    def no_systemctl(name, *args, **kwargs):
        return None if name == "systemctl" else real_which(name, *args, **kwargs)

    shims.clear_deactivating(unit)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(shutil, "which", no_systemctl)
    try:
        assert pr._scope_unit_liveness(unit) == "unknown"
        assert pr._stop_systemd_unit_verified(unit) is False
    finally:
        monkey.undo()


def test_scope_liveness_unreadable_procs_file_is_not_death(shims):
    """Gate B pass 4 finding C: a LOADED unit whose cgroup.procs path
    cannot be read (mis-derived prefix, custom cgroup mount, container
    or namespace layout) must classify as 'unknown', never 'dead' — the
    old FileNotFoundError->dead verdict released claims beside live
    scopes. Only the unit not being loaded, or a procs file that was
    actually READ and found empty, is verified death."""
    from tools import process_registry as pr

    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_badcg", 1)
    shims.write_unit(unit, [pid])
    assert pr._scope_unit_liveness(unit) == "alive"

    shims.arm_bad_cgroup_path(unit)
    assert pr._scope_unit_liveness(unit) == "unknown"
    assert pr._scope_unit_active_state(unit) == "unknown"
    # An unreadable cgroup cannot CONFIRM a stop either (the stop still
    # fires — signalling a unit we want dead is correct — but the
    # verdict stays False so callers keep their claim and retry).
    assert pr._stop_systemd_unit_verified(unit) is False

    # Contrast: not-loaded units stay verified dead (cgroup gone with
    # the unit), and an EMPTY procs file that was read is real death.
    other = kb._kanban_worker_scope_unit("t_badcg", 2)
    assert pr._scope_unit_liveness(other) == "dead"
    shims.write_unit(other, [])
    assert pr._scope_unit_liveness(other) == "dead"


def test_verified_stop_escalates_to_sigkill_on_stop_timeout(shims):
    """A SIGTERM-immune descendant (the stop "times out" server-side)
    forces the SIGKILL escalation, and the stop is only confirmed once
    the cgroup is actually empty."""
    from tools import process_registry as pr

    stubborn = shims.stubborn_sleeper()
    unit = kb._kanban_worker_scope_unit("t_stubborn", 1)
    shims.write_unit(unit, [stubborn])

    assert pr._stop_systemd_unit_verified(unit) is True
    actions = [s["action"] for s in shims.stops() if s["unit"] == unit]
    assert actions[0] == "stop" and "kill" in actions
    assert shims.wait_for(lambda: not kb._pid_alive(stubborn)), (
        "SIGKILL escalation must reap the SIGTERM-immune descendant"
    )
    assert not shims.cgroup_pids(unit)


def test_verified_stop_cancel_event_aborts_mid_stop(shims, monkeypatch):
    """Y: a cancel event reaches INSIDE an in-flight verified stop —
    after the TERM wait, before the SIGKILL wait, and before the final
    verify — instead of only between units. Every cancelled stop returns
    False ("still stopping") without paying for escalation the caller
    will never read."""
    import threading

    from tools import process_registry as pr

    # --- 1. cancelled after the TERM wait: no SIGKILL, no verify -----
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_cxl1", 1)
    shims.write_unit(unit, [pid])
    cancel = threading.Event()

    def slow_term(u):
        # The stop job is slow; the caller's budget dies mid-wait.
        cancel.set()
        return True

    # A private MonkeyPatch: undoing the shared fixture-level one would
    # take the shim PATH down with it.
    mp1 = pytest.MonkeyPatch()
    mp1.setattr(pr, "_stop_systemd_unit", slow_term)
    assert pr._stop_systemd_unit_verified(unit, cancel_event=cancel) is False
    actions = [s["action"] for s in shims.stops() if s["unit"] == unit]
    assert actions == [], "cancelled stop must not escalate to SIGKILL"
    assert kb._pid_alive(pid), "nothing was signalled — cgroup still live"
    mp1.undo()

    # --- 2. cancelled before the SIGKILL wait ------------------------
    stubborn = shims.stubborn_sleeper()  # ignores SIGTERM
    unit2 = kb._kanban_worker_scope_unit("t_cxl2", 1)
    shims.write_unit(unit2, [stubborn])
    cancel2 = threading.Event()
    real_liveness = pr._scope_unit_liveness
    calls = {"n": 0}

    def liveness_that_cancels(u):
        state = real_liveness(u)
        calls["n"] += 1
        if state == "alive":
            cancel2.set()  # budget dies between TERM and KILL
        return state

    mp2 = pytest.MonkeyPatch()
    mp2.setattr(pr, "_scope_unit_liveness", liveness_that_cancels)
    assert pr._stop_systemd_unit_verified(unit2, cancel_event=cancel2) is False
    actions2 = [s["action"] for s in shims.stops() if s["unit"] == unit2]
    assert actions2 == ["stop"], "no SIGKILL after the pre-KILL cancel"
    assert kb._pid_alive(stubborn)
    mp2.undo()

    # --- 3. cancelled before the final verify ------------------------
    stubborn2 = shims.stubborn_sleeper()
    unit3 = kb._kanban_worker_scope_unit("t_cxl3", 1)
    shims.write_unit(unit3, [stubborn2])
    cancel3 = threading.Event()
    real_run = pr.subprocess.run

    def run_then_cancel(*args, **kwargs):
        out = real_run(*args, **kwargs)
        cmd = args[0] if args else kwargs.get("args")
        if cmd and "kill" in cmd:
            # Budget dies the instant the SIGKILL is on the wire: the
            # next verify probe belongs to a caller that moved on.
            cancel3.set()
        return out

    mp3 = pytest.MonkeyPatch()
    mp3.setattr(pr.subprocess, "run", run_then_cancel)
    assert pr._stop_systemd_unit_verified(unit3, cancel_event=cancel3) is False
    # The escalation DID run (cancel came too late to spare it), but the
    # verdict is still "still stopping": a cancelled caller must re-read
    # state, and the next stop confirms the already-empty cgroup for free.
    actions3 = [s["action"] for s in shims.stops() if s["unit"] == unit3]
    assert "kill" in actions3
    assert shims.wait_for(lambda: not kb._pid_alive(stubborn2))


def test_verified_stop_deadline_aborts_like_cancel_event(shims, monkeypatch):
    """Y: a monotonic deadline is the event's twin — a stop whose
    deadline already passed never even fires the TERM."""
    from tools import process_registry as pr

    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_dx", 1)
    shims.write_unit(unit, [pid])
    fired: list[str] = []

    def no_term(u):
        fired.append("term")
        return True

    monkeypatch.setattr(pr, "_stop_systemd_unit", no_term)
    assert pr._stop_systemd_unit_verified(
        unit, deadline=time.monotonic() - 1.0,
    ) is False
    # The TERM was faked (never reached the shim); the point is the
    # verdict — False even though nothing was wrong with the unit —
    # and that no escalation followed.
    actions = [s["action"] for s in shims.stops() if s["unit"] == unit]
    assert actions == []
    assert kb._pid_alive(pid)




def test_release_stale_claims_stops_worker_scope(shims, conn):
    """TTL-expired reclaim of a scoped worker stops the whole unit before
    the pid kill backstop, and clears the scope bookkeeping."""
    host = kb._claimer_id().split(":", 1)[0]
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_stale", 1)
    shims.write_unit(unit, [pid])
    tid = kb.create_task(conn, title="stale", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:4194304")
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=?, "
        "claim_expires=?, last_heartbeat_at=? WHERE id=?",
        (pid, unit, int(time.time()) - 60, int(time.time()) - 7200, tid),
    )
    conn.commit()

    reclaimed = kb.release_stale_claims(conn)
    assert reclaimed == 1
    assert any(s["unit"] == unit and s["action"] == "stop" for s in shims.stops())
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["worker_scope"] is None


def test_crash_cleanup_defers_until_scope_stop_verified(shims, conn):
    """Crash reclamation of a scoped run waits for the VERIFIED scope
    stop (Gate B review, crash-cleanup ordering): a deactivating unit
    makes the worker look dead (its pid is gone) but the stop cannot yet
    be confirmed — the claim is held, a ``scope_stopping`` event records
    the hold, and the requeue happens only on a later tick once the
    verified stop lands. Nothing is released beside an unconfirmed
    cgroup, so no duplicate worker can spawn."""
    straggler = shims.sleeper()  # live process inside the worker's cgroup
    launcher = subprocess.Popen(["true"])
    launcher.wait()  # the recorded worker pid: already gone
    unit = kb._kanban_worker_scope_unit("t_crashstop", 1)
    shims.write_unit(unit, [straggler])
    shims.arm_deactivating(unit)
    tid = kb.create_task(conn, title="crash stop", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "worker_pid_started_at=?, worker_registered_at=?, worker_scope=?, "
        "claim_expires=?, last_heartbeat_at=? WHERE id=?",
        (launcher.pid, kb._worker_pid_start_time(launcher.pid), now,
         unit, now, now, tid),
    )
    conn.execute(
        "UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,)
    )
    conn.commit()

    # Tick 1: the worker pid is gone and deactivating is NOT "alive", so
    # the run classifies as dead — but the stop job is still draining, so
    # NOTHING is released and the crash is retried next tick.
    assert kb.detect_crashed_workers(conn) == []
    row = conn.execute(
        "SELECT status, claim_lock, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    assert row["worker_scope"] == unit
    stopping = conn.execute(
        "SELECT count(*) AS n FROM task_events WHERE task_id=? "
        "AND kind='scope_stopping'", (tid,),
    ).fetchone()
    assert stopping["n"] == 1  # the hold is auditable, once per run

    # Still exactly one marker if the stop keeps failing: ticks repeat
    # without flooding the timeline.
    assert kb.detect_crashed_workers(conn) == []
    stopping = conn.execute(
        "SELECT count(*) AS n FROM task_events WHERE task_id=? "
        "AND kind='scope_stopping'", (tid,),
    ).fetchone()
    assert stopping["n"] == 1

    # Tick 3: the stop job completes — scope verified dead, so the crash
    # requeue goes through.
    shims.clear_deactivating(unit)
    assert kb.detect_crashed_workers(conn) == [tid]
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["worker_scope"] is None
    # The verified stop killed the straggler the dead worker left behind.
    assert not kb._pid_alive(straggler)


def test_stop_timeout_does_not_release_claim_then_completes(shims, conn):
    """A wedged stop (killproof scope) must NOT release the claim — that
    would spawn a duplicate beside a live worker. The reclaim defers and
    completes on the next tick once the stop can be confirmed."""
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_wedged", 1)
    shims.write_unit(unit, [pid])
    shims.arm_killproof(unit)
    tid = kb.create_task(conn, title="wedged stop", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=?, "
        "worker_registered_at=?, claim_expires=?, last_heartbeat_at=? "
        "WHERE id=?",
        (pid, unit, now, now - 60, now - 7200, tid),
    )
    conn.commit()

    # Tick 1: stop refuses (still stopping) — claim held, task still running.
    assert kb.release_stale_claims(conn) == 0
    row = conn.execute(
        "SELECT status, claim_lock, worker_scope FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"] == kb._claimer_id()
    assert row["worker_scope"] == unit
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? "
        "AND kind = 'reclaim_deferred'", (tid,),
    ).fetchone()
    assert event is not None  # the hold is auditable

    # Tick 2 (after the defer grace expires): the stop completes — the
    # reclaim goes through. The defer extended claim_expires, so age it.
    shims.clear_killproof(unit)
    conn.execute(
        "UPDATE tasks SET claim_expires = ? WHERE id = ?",
        (int(time.time()) - 60, tid),
    )
    conn.commit()
    assert kb.release_stale_claims(conn) == 1
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["worker_scope"] is None


def test_enforce_max_runtime_defers_then_completes(shims, conn):
    """Same deferral contract on the max-runtime path: unverified stop →
    skip this tick; once verifiable → timeout recorded, scope stopped."""
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_maxrt", 1)
    shims.write_unit(unit, [pid])
    shims.arm_killproof(unit)
    tid = kb.create_task(conn, title="timeout", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=?, "
        "worker_registered_at=?, max_runtime_seconds=1, started_at=? "
        "WHERE id=?",
        (pid, unit, now, now - 100, tid),
    )
    conn.execute(
        "UPDATE task_runs SET started_at = started_at - 9999 "
        "WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)", (tid,),
    )
    conn.commit()

    def fake_kill(pid_, sig):
        raise ProcessLookupError()

    assert kb.enforce_max_runtime(conn, signal_fn=fake_kill) == []
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "running"
    assert row["worker_scope"] == unit

    shims.clear_killproof(unit)
    assert kb.enforce_max_runtime(conn, signal_fn=fake_kill) == [tid]
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_scope"] is None


def _max_runtime_row(conn, pid, scope, *, pid_started_at=None,
                     registered=True):
    """A running row past its max_runtime, owned by this host's claimer."""
    tid = kb.create_task(conn, title="max runtime", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "worker_pid_started_at=?, worker_scope=?, worker_registered_at=?, "
        "max_runtime_seconds=1, started_at=? WHERE id=?",
        (
            pid, pid_started_at, scope,
            now if registered else None, now - 100, tid,
        ),
    )
    conn.execute(
        "UPDATE task_runs SET started_at = started_at - 9999 "
        "WHERE id=(SELECT current_run_id FROM tasks WHERE id=?)", (tid,),
    )
    conn.commit()
    return tid


def _timed_out_payload(conn, tid) -> dict:
    event = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'timed_out'",
        (tid,),
    ).fetchone()
    return json.loads(event["payload"]) if event else {}


def test_enforce_max_runtime_scoped_never_signals_the_pid(shims, conn):
    """E (finding 3, scoped half): a scoped max-runtime run is ended by a
    VERIFIED scope stop and the recorded pid is never signalled — once
    the cgroup is confirmed empty nothing of the run survives, and the
    pid number may already have been handed to an unrelated process."""
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_maxrt_scoped", 1)
    shims.write_unit(unit, [pid])
    tid = _max_runtime_row(
        conn, pid, unit, pid_started_at=kb._worker_pid_start_time(pid),
    )
    signals: list[tuple[int, int]] = []

    def recording_kill(p, s):
        signals.append((p, s))

    assert kb.enforce_max_runtime(conn, signal_fn=recording_kill) == [tid]
    assert signals == []  # the unit was stopped; the pid itself untouched
    assert not kb._pid_alive(pid)  # teardown verified, not assumed
    assert _timed_out_payload(conn, tid)["scope_stopped"] == unit


def test_enforce_max_runtime_never_signals_a_recycled_pid(shims, conn):
    """E (finding 3, unscoped half): when the recorded start fingerprint no
    longer matches the live pid (the worker died and the kernel reused its
    number), the run times out WITHOUT signalling — a bare liveness kill
    would murder an unrelated process that inherited the pid."""
    pid = shims.sleeper()  # stands in for the unrelated impostor process
    tid = _max_runtime_row(conn, pid, None, pid_started_at=111111)
    signals: list[tuple[int, int]] = []

    def recording_kill(p, s):
        signals.append((p, s))

    assert kb.enforce_max_runtime(conn, signal_fn=recording_kill) == [tid]
    assert signals == []
    assert kb._pid_alive(pid)  # the impostor was never touched
    payload = _timed_out_payload(conn, tid)
    assert payload.get("pid_reused") is True


def test_enforce_max_runtime_legacy_row_with_predating_pid_not_signalled(
    shims, conn, monkeypatch
):
    """B (finding, legacy half): a row written before the fingerprint
    column existed has no pid identity to check — but a live process
    that was running BEFORE the run started cannot be that run's worker,
    so the pid must not be signalled; the run still times out."""
    pid = shims.sleeper()  # stands in for the unrelated impostor
    tid = _max_runtime_row(conn, pid, None, pid_started_at=None)
    started = conn.execute(
        "SELECT COALESCE(r.started_at, t.started_at) AS s "
        "FROM tasks t LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.id = ?",
        (tid,),
    ).fetchone()["s"]
    # The live pid began long before the run row existed.
    monkeypatch.setattr(
        kb, "_worker_pid_epoch_start",
        lambda _pid: float(started) - 3600.0,
    )
    signals: list[tuple[int, int]] = []

    def recording_kill(p, s):
        signals.append((p, s))

    assert kb.enforce_max_runtime(conn, signal_fn=recording_kill) == [tid]
    assert signals == []
    assert kb._pid_alive(pid)  # the impostor was never touched
    payload = _timed_out_payload(conn, tid)
    assert payload.get("pid_reused") is True


def test_enforce_max_runtime_legacy_row_matching_start_still_killed(
    shims, conn
):
    """B, control case: a legacy row whose live pid started AFTER the run
    began (the normal case — worker spawned at run start) still gets the
    bare-pid timeout; the gate must not regress legacy handling."""
    pid = shims.sleeper()
    tid = _max_runtime_row(conn, pid, None, pid_started_at=None)

    assert kb.enforce_max_runtime(conn) == [tid]
    assert not kb._pid_alive(pid)  # teardown verified, not assumed
    payload = _timed_out_payload(conn, tid)
    assert payload.get("pid") == pid
    assert "pid_reused" not in payload


def test_enforce_max_runtime_legacy_row_unreadable_start_not_signalled(
    shims, conn, monkeypatch
):
    """B, fail-safe half (Gate B pass 4 finding P): when the epoch-start
    check cannot run at all (psutil missing, exotic process), the
    membership of the pid is UNKNOWN — a signal we cannot attribute
    might hit an unrelated process, so the run times out and reclaims
    WITHOUT signalling, exactly like a predating pid."""
    pid = shims.stubborn_sleeper()
    tid = _max_runtime_row(conn, pid, None, pid_started_at=None)
    monkeypatch.setattr(kb, "_worker_pid_epoch_start", lambda _pid: None)
    signals: list[tuple[int, int]] = []

    def recording_kill(p, s):
        signals.append((p, s))
        os.kill(p, s)

    assert kb.enforce_max_runtime(conn, signal_fn=recording_kill) == [tid]
    assert signals == []
    assert kb._pid_alive(pid)  # never touched
    payload = _timed_out_payload(conn, tid)
    assert payload.get("pid_reused") is True
    assert payload.get("signal_skipped") == "pid_membership_unknown"
    # The stand-down is logged once per (task, run, pid, verdict); the
    # key captured the run id before the reclaim cleared it.
    assert any(
        k[0] == tid and k[2] == pid and k[3] == "pid_membership_unknown"
        for k in kb._bare_pid_gate_warned
    )


def test_enforce_max_runtime_identity_unreadable_not_signalled(
    shims, conn, monkeypatch
):
    """P, fingerprint half: a row WITH a start fingerprint whose live
    process start time cannot be read is an unknown identity — no
    bare-pid signal, the run still times out."""
    import gateway.status as gateway_status

    pid = shims.sleeper()
    tid = _max_runtime_row(conn, pid, None, pid_started_at=111111)
    monkeypatch.setattr(
        gateway_status, "get_process_start_time", lambda _pid: None,
    )
    signals: list[tuple[int, int]] = []

    def recording_kill(p, s):
        signals.append((p, s))

    assert kb.enforce_max_runtime(conn, signal_fn=recording_kill) == [tid]
    assert signals == []
    assert kb._pid_alive(pid)
    payload = _timed_out_payload(conn, tid)
    assert payload.get("signal_skipped") == "pid_identity_unknown"


def test_reclaim_task_stops_worker_scope(shims, conn):
    """Operator reclaim of a scoped worker stops its scope too."""
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_manual", 1)
    shims.write_unit(unit, [pid])
    tid = kb.create_task(conn, title="manual", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=? "
        "WHERE id=?",
        (pid, unit, tid),
    )
    conn.commit()

    assert kb.reclaim_task(conn, tid, reason="operator abort") is True
    assert any(s["unit"] == unit for s in shims.stops())
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_scope"] is None


def test_archive_and_schedule_stop_scope(shims, conn):
    """archive_task / schedule_task stop a scoped worker's scope and clear
    the bookkeeping — the finding-5 surfaces."""
    pid = shims.sleeper()
    unit_a = kb._kanban_worker_scope_unit("t_arch", 1)
    shims.write_unit(unit_a, [pid])
    tid_a = kb.create_task(conn, title="to archive", assignee="w")
    kb.claim_task(conn, tid_a, claimer=kb._claimer_id())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=? "
        "WHERE id=?",
        (pid, unit_a, tid_a),
    )
    conn.commit()

    assert kb.archive_task(conn, tid_a) is True
    assert any(s["unit"] == unit_a for s in shims.stops())
    row = conn.execute(
        "SELECT worker_scope FROM tasks WHERE id = ?", (tid_a,)
    ).fetchone()
    assert row["worker_scope"] is None

    pid_s = shims.sleeper()
    unit_s = kb._kanban_worker_scope_unit("t_sched", 1)
    shims.write_unit(unit_s, [pid_s])
    tid_s = kb.create_task(conn, title="to schedule", assignee="w")
    kb.claim_task(conn, tid_s, claimer=kb._claimer_id())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=? "
        "WHERE id=?",
        (pid_s, unit_s, tid_s),
    )
    conn.commit()

    assert kb.schedule_task(conn, tid_s, reason="later") is True
    assert any(s["unit"] == unit_s for s in shims.stops())
    row = conn.execute(
        "SELECT worker_scope FROM tasks WHERE id = ?", (tid_s,)
    ).fetchone()
    assert row["worker_scope"] is None


def test_invalidate_descendants_stops_scope(shims, conn):
    """Ancestor reopen invalidation stops a running scoped descendant."""
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_inval", 1)
    shims.write_unit(unit, [pid])
    parent = kb.create_task(conn, title="ancestor", assignee="planner")
    assert kb.complete_task(conn, parent)
    child = kb.create_task(
        conn, title="running child", assignee="builder", parents=[parent],
    )
    kb.claim_task(conn, child, claimer=kb._claimer_id())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, worker_scope=? "
        "WHERE id=?",
        (pid, unit, child),
    )
    conn.commit()

    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'todo', completed_at = NULL "
            "WHERE id = ?", (parent,),
        )
    result = kb.invalidate_descendants_for_parent_reopen(
        conn, parent, author="operator",
    )
    # Scoped descendants are verified stopped BEFORE the demotion — no
    # post-commit termination tuple exists for an already-empty cgroup
    # (E: a spawnable 'todo' never lands beside a live scope).
    assert result["terminations"] == []
    assert any(s["unit"] == unit for s in shims.stops())
    assert shims.wait_for(lambda: not kb._pid_alive(pid))
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id = ?", (child,)
    ).fetchone()
    assert row["status"] == "todo"
    assert row["worker_scope"] is None


def test_reopen_demotion_checks_identity_for_every_descendant(
    shims, conn, monkeypatch,
):
    """W: the reopen demotion's run-identity guard must not depend on the
    row still having a worker_scope. Phase 0 can confirm the OLD scoped
    run's cgroup empty while a retry replaces it with a fresh UNSCOPED
    run (the spawn-fallback shape) before the demotion transaction — a
    stop verdict about the old run must not demote (or kill) the new
    one. Per-row guards, not a blanket stand-down: an unchanged unscoped
    running descendant still demotes with its kill queued."""
    import threading

    parent = kb.create_task(conn, title="ancestor", assignee="planner")
    assert kb.complete_task(conn, parent)
    child = kb.create_task(
        conn, title="replaced child", assignee="builder", parents=[parent],
    )
    kb.claim_task(conn, child, claimer=kb._claimer_id())
    # The old run: scoped, cgroup already empty (the unit was never
    # written, so the Phase 0 probe confirms dead instantly).
    old_unit = kb._kanban_worker_scope_unit("t_w_old", 1)
    conn.execute(
        "UPDATE tasks SET status='running', worker_scope=? WHERE id=?",
        (old_unit, child),
    )
    # Positive control: an unscoped running descendant whose identity
    # stays stable across the two phases — still demoted, kill queued.
    steady = kb.create_task(
        conn, title="steady child", assignee="builder", parents=[parent],
    )
    kb.claim_task(conn, steady, claimer=kb._claimer_id())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=424242 WHERE id=?",
        (steady,),
    )
    conn.commit()

    # Race the probe->txn window: right after the OLD scope's stop
    # confirms, replace the run with a fresh UNSCOPED worker.
    real_stop = kb.request_worker_scope_stop
    replaced = threading.Event()

    def stop_then_replace(unit, **kwargs):
        result = real_stop(unit, **kwargs)
        if result and unit == old_unit and not replaced.is_set():
            replaced.set()
            new_pid = shims.sleeper()
            with kb.write_txn(conn):
                kb._end_run(conn, child, outcome="crashed", status="ready")
                conn.execute(
                    "UPDATE tasks SET status='ready', claim_lock=NULL, "
                    "claim_expires=NULL, worker_pid=NULL, "
                    "worker_pid_started_at=NULL, worker_registered_at=NULL, "
                    "worker_scope=NULL WHERE id=?",
                    (child,),
                )
            assert kb.claim_task(conn, child, claimer=kb._claimer_id())
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status='running' WHERE id=?", (child,),
                )
            kb._set_worker_pid(conn, child, new_pid)
            stop_then_replace.new_pid = new_pid
        return result

    monkeypatch.setattr(kb, "request_worker_scope_stop", stop_then_replace)

    result = kb.invalidate_descendants_for_parent_reopen(
        conn, parent, author="operator",
    )
    assert replaced.is_set()
    row = conn.execute(
        "SELECT status, worker_scope, worker_pid FROM tasks WHERE id=?",
        (child,),
    ).fetchone()
    # The new run is NOT demoted, never queued for a kill, and its
    # worker survives the retraction untouched.
    assert row["status"] == "running"
    assert row["worker_scope"] is None
    assert row["worker_pid"] == stop_then_replace.new_pid
    assert kb._pid_alive(stop_then_replace.new_pid)
    assert all(entry["id"] != child for entry in result["invalidated"])
    assert all(
        t[0] != stop_then_replace.new_pid for t in result["terminations"]
    )
    # The stable unscoped sibling demotes as before.
    assert any(entry["id"] == steady for entry in result["invalidated"])
    assert any(t[0] == 424242 for t in result["terminations"])


def test_completion_stops_scope_and_reaps_leaked_descendant(shims, conn, kanban_home):
    """Normal completion must not leak the worker's descendants: the
    worker-side stop is detached (it cannot wait on its own teardown), the
    audit sweep does the verified kill — and a stubborn descendant that
    ignores SIGTERM dies to the SIGKILL escalation."""
    leaked = shims.stubborn_sleeper()
    _spawnable_profile(kanban_home)
    tid = kb.create_task(conn, title="done soon", assignee="elias")
    kb.dispatch_once(conn, dry_run=False)

    row = conn.execute(
        "SELECT worker_pid, worker_scope, current_run_id FROM tasks "
        "WHERE id = ?", (tid,)
    ).fetchone()
    unit = row["worker_scope"]
    worker_pid = shims.unit_json(unit)["pids"][0]
    assert kb.register_worker_pid(
        conn, tid, expected_run_id=row["current_run_id"], pid=worker_pid,
    )
    # A descendant the worker "spawned" that outlives it in the cgroup.
    shims.arm_sticky(unit, leaked)

    assert kb.complete_task(conn, tid, result="done") is True

    # The detached stop fired (worker-side terminal path).
    assert shims.wait_for(
        lambda: any(s["unit"] == unit for s in shims.stops())
    ), "complete_task never attempted to stop the scope"
    # The task row no longer claims the unit, so the audit sweep reaps it —
    # and the VERIFIED stop escalates: the SIGTERM-immune descendant dies to
    # the SIGKILL pass instead of outliving the task.
    kb.reap_orphaned_worker_scopes(conn)
    assert shims.wait_for(lambda: not kb._pid_alive(leaked), timeout=8.0), (
        "leaked descendant survived the SIGKILL escalation"
    )


# ---------------------------------------------------------------------------
# Retry uniqueness + spawn-failure classification (real shims)
# ---------------------------------------------------------------------------

def test_retry_spawn_uses_a_new_unique_unit(shims, conn, kanban_home):
    """A respawned attempt gets a DIFFERENT unit name (run-suffixed), so a
    lingering half-dead scope from the previous attempt can never collide —
    and the audit sweep stops the orphaned old unit."""
    _spawnable_profile(kanban_home)
    tid = kb.create_task(conn, title="retry", assignee="elias")
    assert kb.dispatch_once(conn, dry_run=False).spawned
    row = conn.execute(
        "SELECT worker_scope, current_run_id FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    unit_a = row["worker_scope"]
    assert unit_a.endswith(f"-r{row['current_run_id']}.scope")

    # Operator reclaims (verified stop) → task returns to its source lane.
    assert kb.reclaim_task(conn, tid, reason="retry") is True

    # Second attempt: new run id → new unit name.
    assert kb.dispatch_once(conn, dry_run=False).spawned
    row = conn.execute(
        "SELECT worker_scope, current_run_id, status FROM tasks "
        "WHERE id = ?", (tid,)
    ).fetchone()
    unit_b = row["worker_scope"]
    assert unit_b != unit_a
    assert unit_b.endswith(f"-r{row['current_run_id']}.scope")
    assert kb._task_id_from_kanban_scope_unit(unit_b) == tid


def test_spawn_failure_auto_falls_back_to_plain_spawn(shims, conn, kanban_home):
    """A refused systemd-run launch in 'auto' mode falls back to a plain
    spawn for THIS run (with a warning), records the real pid (which IS the
    worker for a plain spawn, so it counts as registered), and keeps the
    board moving."""
    _write_kanban_config(Path(kanban_home), "  worker_isolation: auto\n")
    kb._INITIALIZED_PATHS.clear()
    _spawnable_profile(kanban_home)
    shims.arm_fail_next(1)
    tid = kb.create_task(conn, title="degraded", assignee="elias")

    result = kb.dispatch_once(conn, dry_run=False)
    assert [s[0] for s in result.spawned] == [tid]
    row = conn.execute(
        "SELECT worker_pid, worker_scope, worker_registered_at "
        "FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["worker_scope"] is None
    assert row["worker_pid"] is not None
    shims.track(row["worker_pid"])
    assert row["worker_registered_at"] is not None  # plain pid == worker
    assert result.late_spawn_failed == []


def test_spawn_failure_systemd_scope_mode_fails_loudly(shims, conn, kanban_home):
    """'systemd-scope' never degrades: a refused launch raises into the
    dispatcher's failure recording — spawn_failed run with the systemd-run
    stderr, failure counted, nothing spawned."""
    _write_kanban_config(Path(kanban_home), "  worker_isolation: systemd-scope\n")
    kb._INITIALIZED_PATHS.clear()
    _spawnable_profile(kanban_home)
    shims.arm_fail_next(1)
    tid = kb.create_task(conn, title="strict", assignee="elias")

    result = kb.dispatch_once(conn, dry_run=False)
    assert result.spawned == []
    assert result.auto_blocked == []
    row = conn.execute(
        "SELECT status, consecutive_failures, last_failure_error, "
        "       worker_pid, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["consecutive_failures"] == 1
    assert "systemd-run launch failed" in (row["last_failure_error"] or "")
    assert row["worker_pid"] is None
    assert row["worker_scope"] is None
    run = kb.latest_run(conn, tid)
    assert run is not None and run.outcome == "spawn_failed"
    assert "user bus connection refused" in (run.error or "")


def test_strict_probe_unavailable_surfaces_spawn_failed(
    shims, conn, kanban_home, monkeypatch,
):
    """H: the strict-mode probe refusal must reach the TASK, not just the
    dispatcher log. With the probe itself unusable (macOS / user bus gone)
    the spawn refuses BEFORE launching anything; dispatch_once records
    spawn_failed with the operator-facing reason on the row and the run,
    and nothing is ever spawned."""
    _write_kanban_config(Path(kanban_home), "  worker_isolation: systemd-scope\n")
    kb._INITIALIZED_PATHS.clear()
    _spawnable_profile(kanban_home)
    monkeypatch.setattr(
        "tools.process_registry._systemd_run_user_scope_available",
        lambda: False,
    )
    tid = kb.create_task(conn, title="no bus", assignee="elias")

    result = kb.dispatch_once(conn, dry_run=False)

    assert result.spawned == []
    assert result.auto_blocked == []
    row = conn.execute(
        "SELECT status, consecutive_failures, last_failure_error, "
        "       worker_pid, worker_scope FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] != "running"
    assert row["consecutive_failures"] == 1
    assert "worker_isolation=systemd-scope is configured but" in (
        row["last_failure_error"] or ""
    )
    assert "refusing to spawn task" in (row["last_failure_error"] or "")
    assert row["worker_pid"] is None
    assert row["worker_scope"] is None
    run = kb.latest_run(conn, tid)
    assert run is not None and run.outcome == "spawn_failed"
    assert "refusing to spawn task" in (run.error or "")
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND "
        "kind='spawn_failed' ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    assert event is not None
    assert "refusing to spawn task" in (event["payload"] or "")


def test_fast_worker_exit_is_not_a_launch_failure(
    shims, conn, kanban_home, monkeypatch,
):
    """A worker that legitimately finishes within the launch probe is NOT
    a failed systemd launch: rc=0 from the launcher means the scoped
    command ran and exited. Auto mode must NOT plain-spawn a duplicate
    beside it (the review's critical duplication bug) — the spawn stands,
    exactly one worker ever exists, and exit classification owns the
    outcome on the next tick."""
    _write_kanban_config(Path(kanban_home), "  worker_isolation: auto\n")
    kb._INITIALIZED_PATHS.clear()
    _spawnable_profile(kanban_home)
    monkeypatch.setattr(
        kb, "_resolve_hermes_argv",
        lambda: [sys.executable, "-c", "pass"],
    )
    tid = kb.create_task(conn, title="fast worker", assignee="elias")

    result = kb.dispatch_once(conn, dry_run=False)
    assert [s[0] for s in result.spawned] == [tid]
    row = conn.execute(
        "SELECT worker_pid, worker_scope, status FROM tasks WHERE id = ?",
        (tid,),
    ).fetchone()
    assert row["worker_scope"] is not None  # no fallback re-spawn happened
    spawned_events = conn.execute(
        "SELECT count(*) AS n FROM task_events "
        "WHERE task_id=? AND kind='spawned'", (tid,),
    ).fetchone()
    assert spawned_events["n"] == 1  # exactly one worker, ever
    assert result.late_spawn_failed == []


def test_fast_worker_nonzero_exit_is_not_a_launch_failure(
    shims, conn, kanban_home, monkeypatch,
):
    """Same contract in strict mode with a non-zero rc: the unit WAS
    created, so the worker ran and exited 3 — that is a worker exit, not
    a launch failure. No spawn_failed, no duplicate spawn, no breaker
    tick; the exit registry classifies the run on a later tick."""
    _write_kanban_config(Path(kanban_home), "  worker_isolation: systemd-scope\n")
    kb._INITIALIZED_PATHS.clear()
    _spawnable_profile(kanban_home)
    monkeypatch.setattr(
        kb, "_resolve_hermes_argv",
        lambda: [sys.executable, "-c", "raise SystemExit(3)"],
    )
    tid = kb.create_task(conn, title="fast fail", assignee="elias")

    result = kb.dispatch_once(conn, dry_run=False)
    assert [s[0] for s in result.spawned] == [tid]
    row = conn.execute(
        "SELECT worker_pid, worker_scope, status, consecutive_failures "
        "FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["worker_scope"] is not None
    assert row["consecutive_failures"] == 0  # no spawn_failed recorded
    spawned_events = conn.execute(
        "SELECT count(*) AS n FROM task_events "
        "WHERE task_id=? AND kind='spawned'", (tid,),
    ).fetchone()
    assert spawned_events["n"] == 1
    # The load-bearing fact for this regression: without --collect the
    # FAILED unit stays loaded on the bus (inspectable), which is what
    # lets the probe read "ran and exited" instead of "never created"
    # (the shim models --collect unloading any completion; under the old
    # argv the unit was gone when the probe looked).  The probe SAW it —
    # and once the run is terminal the tick's sweep collects the unit
    # EXPLICITLY, which is observable in the shim's action log:
    unit = kb._kanban_worker_scope_unit(tid, 1)
    assert {"action": "reset-failed", "unit": unit} in shims.stops()


def test_spawn_failure_with_uncleanable_unit_refuses_fallback(
    shims, conn, kanban_home, monkeypatch,
):
    """A failed launch whose scope cleanup CANNOT be verified must not
    plain-spawn a replacement beside the possibly-live half-created unit
    — auto mode included. The dispatcher records spawn_failed with the
    refusal instead."""
    _write_kanban_config(Path(kanban_home), "  worker_isolation: auto\n")
    kb._INITIALIZED_PATHS.clear()
    _spawnable_profile(kanban_home)
    shims.arm_fail_next(1)
    cleanup_calls: list[str] = []

    def unverifiable_stop(unit):
        cleanup_calls.append(unit)
        return False  # cleanup could not be verified (wedged stop job)

    monkeypatch.setattr(kb, "_stop_kanban_worker_scope", unverifiable_stop)
    tid = kb.create_task(conn, title="dirty launch", assignee="elias")

    result = kb.dispatch_once(conn, dry_run=False)
    assert result.spawned == []
    assert cleanup_calls == [kb._kanban_worker_scope_unit(tid, 1)]
    row = conn.execute(
        "SELECT status, worker_pid, worker_scope, consecutive_failures, "
        "last_failure_error FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    assert row["status"] != "running"
    assert row["worker_pid"] is None  # no plain-spawn duplicate beside it
    assert row["consecutive_failures"] == 1
    assert "refusing to spawn a replacement" in (
        row["last_failure_error"] or ""
    )


def test_set_worker_pid_refuses_terminal_task(conn):
    """The status guard: a task that left 'running' before its spawn was
    recorded (fast worker completing mid-spawn, crash reclaim racing the
    spawn loop) must not get worker bookkeeping reattached — the row
    keeps its terminal state and no spawned event claims a live run."""
    tid = kb.create_task(conn, title="done already", assignee="w")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, worker_pid=NULL, "
        "worker_scope=NULL WHERE id=?", (tid,),
    )
    conn.commit()

    kb._set_worker_pid(conn, tid, os.getpid(), scope_unit="u.scope")

    row = conn.execute(
        "SELECT status, worker_pid, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "ready"
    assert row["worker_pid"] is None
    assert row["worker_scope"] is None
    events = conn.execute(
        "SELECT count(*) AS n FROM task_events "
        "WHERE task_id=? AND kind='spawned'", (tid,),
    ).fetchone()
    assert events["n"] == 0


def test_dashboard_direct_running_to_ready_terminates_cleanly(
    shims, conn, kanban_home,
):
    """Dashboard drag running->ready must not crash the termination
    drain (Gate B review, finding 5): the direct path records the same
    four-field termination tuple as every other transition — with and
    without a scope — and a scoped worker's unit is stopped (verified)
    BEFORE the status lands, never beside it."""
    mod = _load_dashboard_plugin()

    # A scoped running row and an unscoped one.
    scoped_pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_dash", 3)
    shims.write_unit(unit, [scoped_pid])
    tid_scoped = _scoped_task_row(
        conn, scope=unit, pid=scoped_pid, registered=True,
    )
    plain_pid = shims.sleeper()
    tid_plain = kb.create_task(conn, title="plain drag", assignee="w")
    kb.claim_task(conn, tid_plain, claimer=kb._claimer_id())
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "worker_pid_started_at=?, worker_registered_at=? WHERE id=?",
        (plain_pid, kb._worker_pid_start_time(plain_pid), now, tid_plain),
    )
    conn.commit()

    # The old bug: ValueError unpacking a two-tuple into four names.
    assert mod._set_status_direct(conn, tid_scoped, "ready") is True
    assert mod._set_status_direct(conn, tid_plain, "ready") is True

    for tid in (tid_scoped, tid_plain):
        row = conn.execute(
            "SELECT status, worker_pid, worker_pid_started_at, "
            "worker_registered_at, worker_scope FROM tasks WHERE id=?",
            (tid,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["worker_pid"] is None
        assert row["worker_pid_started_at"] is None
        assert row["worker_registered_at"] is None
        assert row["worker_scope"] is None
    # The scoped worker was terminated through its scope (the plain one
    # via the pid loop); both are gone.
    assert shims.wait_for(lambda: not kb._pid_alive(scoped_pid))
    assert shims.cgroup_pids(unit) == []
    assert shims.wait_for(lambda: not kb._pid_alive(plain_pid))


def _load_dashboard_plugin():
    """Import plugins/kanban/dashboard/plugin_api.py as a fresh module
    (it is not a package import; the dashboard loads it by path)."""
    pytest.importorskip("fastapi")
    import importlib.util

    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_scope_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dashboard_direct_running_to_ready_refuses_unverified_stop(
    shims, conn, kanban_home,
):
    """E: with the scope stop unconfirmed (stop job wedged), the drag
    does NOT flip the task to spawnable 'ready' — the row stays running
    with its claim held, a scope_stopping marker records why, and the
    refusal names the reason. Once the stop can confirm, the retry
    lands."""
    mod = _load_dashboard_plugin()
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_dash_refuse", 1)
    shims.write_unit(unit, [pid])
    shims.arm_deactivating(unit)
    shims.arm_killproof(unit)
    tid = _scoped_task_row(conn, scope=unit, pid=pid, registered=True)

    with pytest.raises(mod._StatusTransitionRefused):
        mod._set_status_direct(conn, tid, "ready")
    row = conn.execute(
        "SELECT status, claim_lock, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "running"  # non-spawnable: no duplicate window
    assert row["claim_lock"]  # claim held
    assert row["worker_scope"] == unit
    kinds = [
        r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id",
            (tid,),
        ).fetchall()
    ]
    assert "scope_stopping" in kinds
    assert "reclaim_deferred" in kinds
    # The pid-kill backstop may well have killed the worker directly —
    # that is fine; what must NOT happen is confirming the SCOPE or
    # releasing the row while its cgroup state is unknown.
    assert any(
        a["action"] in {"stop", "kill"} and a["unit"] == unit
        for a in shims.stops()
    )

    # Unwedged: the retry verified-stops first, then flips the status.
    shims.clear_deactivating(unit)
    shims.clear_killproof(unit)
    assert mod._set_status_direct(conn, tid, "ready") is True
    row = conn.execute(
        "SELECT status, worker_pid, worker_scope FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert row["status"] == "ready"
    assert row["worker_pid"] is None
    assert row["worker_scope"] is None
    assert shims.wait_for(lambda: not kb._pid_alive(pid))
    assert shims.cgroup_pids(unit) == []


def test_ancestor_reopen_defers_scoped_running_descendant(shims, conn):
    """E, invalidation half: reopening an ancestor demotes a scoped
    running descendant only after its scope is verified dead. An
    unconfirmed stop defers the whole descendant (stays running, claim
    held, marker written) instead of parking a spawnable 'todo' beside a
    draining cgroup; a confirmed one demotes and the worker is dead."""
    # Wedged descendant.
    wedged_pid = shims.sleeper()
    wedged_unit = kb._kanban_worker_scope_unit("t_desc_wedged", 1)
    shims.write_unit(wedged_unit, [wedged_pid])
    shims.arm_deactivating(wedged_unit)
    shims.arm_killproof(wedged_unit)
    # Clean descendant.
    clean_pid = shims.sleeper()
    clean_unit = kb._kanban_worker_scope_unit("t_desc_clean", 1)
    shims.write_unit(clean_unit, [clean_pid])
    parent = kb.create_task(conn, title="ancestor", assignee="planner")
    assert kb.complete_task(conn, parent)
    for scope, pid, title in (
        (wedged_unit, wedged_pid, "wedged child"),
        (clean_unit, clean_pid, "clean child"),
    ):
        child = kb.create_task(
            conn, title=title, assignee="builder", parents=[parent],
        )
        claimed = kb.claim_task(conn, child)
        assert claimed is not None and claimed.status == "running"
        kb._set_worker_pid(conn, child, pid)
        conn.execute(
            "UPDATE tasks SET worker_scope=? WHERE id=?", (scope, child),
        )
        conn.commit()
    wedged_child, clean_child = (
        conn.execute(
            "SELECT id FROM tasks WHERE title=?", (t,),
        ).fetchone()["id"]
        for t in ("wedged child", "clean child")
    )

    result = kb.invalidate_descendants_for_parent_reopen(
        conn, parent, author="operator",
    )

    # Wedged: deferred whole — still running, claim held, marked.
    row = conn.execute(
        "SELECT status, claim_lock, worker_scope FROM tasks WHERE id=?",
        (wedged_child,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["claim_lock"]
    assert row["worker_scope"] == wedged_unit
    kinds = [
        r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id",
            (wedged_child,),
        ).fetchall()
    ]
    assert "scope_stopping" in kinds
    assert "reclaim_deferred" in kinds
    # Clean: verified dead first, then demoted — no termination tuple is
    # left for a post-commit kill of an already-empty cgroup.
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id=?",
        (clean_child,),
    ).fetchone()
    assert row["status"] == "todo"
    assert row["worker_scope"] is None
    assert result["terminations"] == []
    assert shims.wait_for(lambda: not kb._pid_alive(clean_pid))
    assert kb._pid_alive(wedged_pid)  # killproof held the wedge

    # Unwedge + re-run: the deferred descendant demotes on the retry.
    shims.clear_deactivating(wedged_unit)
    shims.clear_killproof(wedged_unit)
    kb.invalidate_descendants_for_parent_reopen(conn, parent, author="op")
    row = conn.execute(
        "SELECT status, worker_scope FROM tasks WHERE id=?",
        (wedged_child,),
    ).fetchone()
    assert row["status"] == "todo"
    assert row["worker_scope"] is None
    assert shims.wait_for(lambda: not kb._pid_alive(wedged_pid))


# ---------------------------------------------------------------------------
# Audit sweep + shutdown policy
# ---------------------------------------------------------------------------

def test_reap_orphaned_scope_sweep(shims, conn):
    """Active scopes with no running task claiming them are stopped; the
    scope a running task still owns is left alone."""
    orphan_pid = shims.sleeper()
    orphan_unit = kb._kanban_worker_scope_unit("t_orphan", 4)
    shims.write_unit(orphan_unit, [orphan_pid])
    # And a scope whose task moved on to a DIFFERENT unit name.
    stale_pid = shims.sleeper()
    stale_unit = kb._kanban_worker_scope_unit("t_moved", 1)
    shims.write_unit(stale_unit, [stale_pid])
    current_unit = kb._kanban_worker_scope_unit("t_moved", 2)
    live_pid = shims.sleeper()
    shims.write_unit(current_unit, [live_pid])
    tid = _scoped_task_row(conn, scope=current_unit, pid=live_pid)

    reaped = kb.reap_orphaned_worker_scopes(conn)
    assert set(reaped) == {orphan_unit, stale_unit}
    assert shims.wait_for(lambda: not kb._pid_alive(orphan_pid))
    assert shims.wait_for(lambda: not kb._pid_alive(stale_pid))
    assert kb._pid_alive(live_pid)  # the running task's worker survives
    row = conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "running"


def test_reap_sweep_escalates_a_wedged_deactivating_orphan(shims, conn):
    """D: a deactivating orphan is not terminal — a stop job draining a
    stubborn process sits in deactivating forever.  The sweep must
    re-request the verified stop (whose SIGKILL escalation drains the
    wedge) instead of quietly collecting around it."""
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_wedged", 1)
    shims.write_unit(unit, [pid])
    shims.arm_deactivating(unit)
    shims.arm_killproof(unit)  # the stop job never completes server-side

    # Wedged: stop re-requested, SIGKILL escalation fired, unit neither
    # confirmed nor collected.
    assert kb.reap_orphaned_worker_scopes(conn) == []
    assert any(
        a["action"] == "stop" and a["unit"] == unit for a in shims.stops()
    )
    assert any(
        a["action"] == "kill" and a["unit"] == unit for a in shims.stops()
    )
    assert not any(
        a["action"] == "reset-failed" and a["unit"] == unit
        for a in shims.stops()
    )
    assert kb._pid_alive(pid)  # killproof: the shim refused, unconfirmed

    # Unwedged: once the stop can complete the next sweep drains,
    # confirms, and collects the orphan like any other.
    shims.clear_deactivating(unit)
    shims.clear_killproof(unit)
    assert kb.reap_orphaned_worker_scopes(conn) == [unit]
    assert shims.wait_for(lambda: not kb._pid_alive(pid))
    assert any(
        a["action"] == "reset-failed" and a["unit"] == unit
        for a in shims.stops()
    )


def test_dispatch_tick_at_concurrency_cap_still_sweeps_orphan_scopes(
    shims, conn,
):
    """T: the cap guards return before the spawn loops, but the orphan
    scope audit must still fire — under sustained caps the tick used to
    return at the first guard and leaked scopes persisted indefinitely."""
    spawns: list[str] = []

    def never_spawn(task, workspace, board):
        spawns.append(task.id)
        return None

    orphan_pid = shims.sleeper()
    orphan_unit = kb._kanban_worker_scope_unit("t_cap_orphan", 1)
    shims.write_unit(orphan_unit, [orphan_pid])
    live_pid = shims.sleeper()
    live_unit = kb._kanban_worker_scope_unit("t_cap_live", 1)
    shims.write_unit(live_unit, [live_pid])
    tid = _scoped_task_row(conn, scope=live_unit, pid=live_pid)

    result = kb._dispatch_once_locked(
        conn, spawn_fn=never_spawn, max_spawn=1,  # running_count == cap
    )

    assert spawns == []                       # the cap held
    assert result.scopes_reaped == [orphan_unit]  # but the audit ran
    assert shims.wait_for(lambda: not kb._pid_alive(orphan_pid))
    assert kb._pid_alive(live_pid)            # the claimed worker survives
    row = conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "running"


def test_dispatch_tick_under_critical_pressure_still_sweeps_orphan_scopes(
    shims, conn, monkeypatch,
):
    """T, the other early return: critical memory pressure stands the
    spawn side down but must not skip the orphan scope audit either."""
    monkeypatch.setattr(kb, "_memory_pressure_level", lambda: "critical")

    spawns: list[str] = []

    def never_spawn(task, workspace, board):
        spawns.append(task.id)
        return None

    orphan_pid = shims.sleeper()
    orphan_unit = kb._kanban_worker_scope_unit("t_mem_orphan", 1)
    shims.write_unit(orphan_unit, [orphan_pid])

    result = kb._dispatch_once_locked(conn, spawn_fn=never_spawn)

    assert spawns == []
    assert result.memory_pressure == "critical"
    assert result.scopes_reaped == [orphan_unit]
    assert shims.wait_for(lambda: not kb._pid_alive(orphan_pid))


def test_stop_all_scoped_workers_is_host_local(shims, conn):
    """The shutdown policy stops every scoped worker THIS host claims and
    leaves other hosts' workers to their own gateways."""
    mine_pid = shims.sleeper()
    mine_unit = kb._kanban_worker_scope_unit("t_mine", 1)
    shims.write_unit(mine_unit, [mine_pid])
    _scoped_task_row(conn, scope=mine_unit, pid=mine_pid)

    theirs_pid = shims.sleeper()
    theirs_unit = kb._kanban_worker_scope_unit("t_theirs", 1)
    shims.write_unit(theirs_unit, [theirs_pid])
    _scoped_task_row(
        conn, scope=theirs_unit, pid=theirs_pid, claimer="otherhost:99",
    )

    stopped = kb.stop_all_scoped_workers(conn)
    assert stopped == [mine_unit]
    assert shims.wait_for(lambda: not kb._pid_alive(mine_pid))
    assert kb._pid_alive(theirs_pid)


def test_shutdown_policy_knob_runs_on_watcher_exit(shims, conn, kanban_home):
    """The gateway dispatcher watcher honours
    ``kanban.worker_isolation_stop_on_shutdown`` on graceful exit: knob true
    → workers stopped; default (unset) → they keep running for re-adoption."""
    import asyncio

    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    kb._INITIALIZED_PATHS.clear()
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_policy", 1)
    shims.write_unit(unit, [pid])
    _scoped_task_row(conn, scope=unit, pid=pid)

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

    asyncio.run(Harness()._kanban_dispatcher_watcher())
    assert shims.wait_for(lambda: not kb._pid_alive(pid), timeout=8.0)


def test_shutdown_waits_for_cleanup_before_releasing_lock(
    shims, conn, kanban_home, monkeypatch,
):
    """H (new g): with the knob on, the dispatcher lock is not released
    while the scoped-worker cleanup is still running inside its budget —
    the watcher waits for the cleanup thread, then releases."""
    import asyncio
    import threading

    import gateway.kanban_watchers as kw
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    kb._INITIALIZED_PATHS.clear()
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_hwait", 1)
    shims.write_unit(unit, [pid])
    _scoped_task_row(conn, scope=unit, pid=pid)

    # Shrink the budget knobs so the test is fast, but keep the scaled
    # math: base 0.3s + 1 unit x (0.3s bound + 2s drain margin).
    monkeypatch.setattr(kw, "_SHUTDOWN_STOP_BASE_SECONDS", 0.3)
    monkeypatch.setattr(
        "tools.process_registry.SCOPE_STOP_VERIFY_BOUND_SECONDS", 0.3,
    )

    cleanup_started = threading.Event()
    gate = threading.Event()
    released = threading.Event()
    released_before_gate = []

    real_stop = kb.stop_all_scoped_workers

    def gated_stop(c, should_abort=None, **kwargs):
        cleanup_started.set()
        gate.wait(timeout=10.0)
        return real_stop(c)

    monkeypatch.setattr(kb, "stop_all_scoped_workers", gated_stop)

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

        def _release_kanban_dispatcher_lock(self) -> None:
            released_before_gate.append(gate.is_set())
            released.set()

    # The watcher on its own thread so this thread can observe the
    # lock NOT being released while cleanup is mid-flight.
    watcher_thread = threading.Thread(
        target=lambda: asyncio.run(Harness()._kanban_dispatcher_watcher()),
        daemon=True,
    )
    watcher_thread.start()

    # The watcher's own startup (lock + setup) takes a few seconds
    # before the graceful-exit path runs — the budget only starts once
    # the cleanup does.
    assert cleanup_started.wait(timeout=15.0)
    # Inside the budget, cleanup still running: bounded negative poll —
    # a correct watcher never releases here, so the poll simply runs
    # out; a buggy one (releasing without waiting) trips it. No blind
    # wall-clock sleep (item K).
    assert not shims.wait_for(released.is_set, timeout=0.5)
    gate.set()
    assert released.wait(timeout=5.0)
    assert released_before_gate == [True]  # released only AFTER cleanup
    watcher_thread.join(timeout=5.0)


def test_shutdown_budget_expiry_logs_leftovers_and_releases(
    shims, conn, kanban_home, monkeypatch, caplog,
):
    """H (new g), the bound: when cleanup exceeds its budget the lock IS
    released (shutdown must not hang) — but only after logging exactly
    which units were left stopping."""
    import asyncio
    import threading

    import gateway.kanban_watchers as kw
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    kb._INITIALIZED_PATHS.clear()
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_hslow", 1)
    shims.write_unit(unit, [pid])
    _scoped_task_row(conn, scope=unit, pid=pid)

    monkeypatch.setattr(kw, "_SHUTDOWN_STOP_BASE_SECONDS", 0.1)
    monkeypatch.setattr(
        "tools.process_registry.SCOPE_STOP_VERIFY_BOUND_SECONDS", 0.1,
    )

    gate = threading.Event()
    released = threading.Event()

    def wedged_stop(c, should_abort=None, **kwargs):
        gate.wait(timeout=10.0)  # "worse than any budget"
        return []

    monkeypatch.setattr(kb, "stop_all_scoped_workers", wedged_stop)

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

        def _release_kanban_dispatcher_lock(self) -> None:
            released.set()

    import logging
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        asyncio.run(Harness()._kanban_dispatcher_watcher())
        gate.set()  # let the daemon thread finish for teardown

    assert released.wait(timeout=1.0) or released.is_set()
    warnings = [r for r in caplog.records if "still stopping" in r.message]
    assert warnings, "expected the leftover-units warning"
    assert unit in warnings[0].getMessage()


def test_join_scope_stop_service_reports_inflight_unit(
    shims, monkeypatch,
):
    """I: a unit mid verified-stop (popped off the queue, still being
    stopped on the service thread) is reported by a draining join — an
    in-flight stop is as "still stopping" as one still queued, and the
    old return (pending only) omitted it."""
    import threading

    gate = threading.Event()
    real_stop = kb._stop_kanban_worker_scope

    def wedged_stop(unit):
        gate.wait(timeout=10.0)
        return real_stop(unit)

    monkeypatch.setattr(kb, "_stop_kanban_worker_scope", wedged_stop)
    monkeypatch.setattr(kb, "_scope_stop_inline", False)
    unit = kb._kanban_worker_scope_unit("t_inflight", 1)
    shims.write_unit(unit, [shims.sleeper()])
    try:
        assert not kb.request_worker_scope_stop(unit)
        leftover = kb.join_scope_stop_service(timeout=0.5)
        # Whichever side of the pop the service thread is on, the unit
        # must be reported: queued OR in flight.
        assert unit in leftover
    finally:
        gate.set()
        kb.join_scope_stop_service(timeout=5.0)
        kb.reset_scope_stop_service_for_tests()


def test_join_scope_stop_service_returns_fast_when_drained(shims):
    """I: "joined" means DRAINED, not thread-exit. The service thread is
    immortal (daemon for the process lifetime), so a plain Thread.join
    always burned the full timeout; an empty queue must return
    immediately."""
    t0 = time.monotonic()
    assert kb.join_scope_stop_service(timeout=5.0) == []
    assert time.monotonic() - t0 < 1.0


def test_scope_stop_intents_are_connection_local_not_thread_local(
    shims, conn, kanban_home, tmp_path, monkeypatch,
):
    """AA: the commit-conditional intent stack is keyed by connection, not
    thread. A shared ``check_same_thread=False`` connection that is mid
    transaction on one thread must COLLECT (not immediately queue) a stop
    requested from another thread — the thread-local stack let that
    request bypass the transaction and fire a kill the rollback could not
    recall. And a commit on one connection must never flush another
    connection's intents."""
    import sqlite3
    import threading

    monkeypatch.setattr(kb, "_scope_stop_inline", False)
    monkeypatch.setattr(kb, "_ensure_scope_stop_thread", lambda: None)
    # A service thread left alive by an EARLIER test is parked on the
    # module wake event; the flush below sets it, and the thread would
    # drain (and pop) the queued entry before the asserts look at it.
    # Parking that thread on the OLD event keeps this test the only
    # observer of the queue.
    monkeypatch.setattr(kb, "_scope_stop_wake", threading.Event())
    monkeypatch.setattr(kb, "_kanban_scope_state", lambda unit: "unknown")
    kb.reset_scope_stop_service_for_tests()

    unit_shared = kb._kanban_worker_scope_unit("t_shared", 1)
    unit_a = kb._kanban_worker_scope_unit("t_a", 1)
    unit_b = kb._kanban_worker_scope_unit("t_b", 1)

    # Shared connection: thread 2 requests a stop while thread 1 holds the
    # transaction open.
    shared = sqlite3.connect(
        kb.kanban_db_path(board="default"), timeout=5.0, check_same_thread=False,
    )
    # A second, independent database file so two write transactions can be
    # open at once on this thread.
    other = kb.connect(db_path=tmp_path / "board2.db")
    try:
        requested = threading.Event()

        def request_from_other_thread():
            kb.request_worker_scope_stop(unit_shared, conn=shared)
            requested.set()

        with kb.write_txn(shared):
            t = threading.Thread(target=request_from_other_thread)
            t.start()
            assert requested.wait(timeout=5.0)
            t.join(timeout=5.0)
            # Collected as an intent of THIS transaction — not queued.
            assert unit_shared not in kb._scope_stop_pending
        # The outermost commit flushed exactly that intent.
        assert unit_shared in kb._scope_stop_pending

        kb.reset_scope_stop_service_for_tests()
        with kb.write_txn(conn):
            with kb.write_txn(other):
                kb.request_worker_scope_stop(unit_a, conn=conn)
                kb.request_worker_scope_stop(unit_b, conn=other)
            # The inner (other) transaction committed first: only its
            # intent reached the queue; conn's stays collected.
            assert unit_b in kb._scope_stop_pending
            assert unit_a not in kb._scope_stop_pending
        # conn's outermost commit flushes its own intent.
        assert unit_a in kb._scope_stop_pending
    finally:
        shared.close()
        other.close()
        kb.reset_scope_stop_service_for_tests()


def test_committed_txn_flushes_stops_when_invariant_raises(conn, monkeypatch):
    """Z: the outermost intent level is popped and flushed BEFORE the
    post-commit file-length invariant check. An invariant exception used
    to leave committed DB state with no queued stop (the check raised
    first); the flush now runs first, so a committed transaction always
    queues its stops."""
    import sqlite3

    def boom(_conn):
        raise sqlite3.DatabaseError("torn-extend detected (test)")

    tid = kb.create_task(conn, title="invariant probe", assignee="w")
    monkeypatch.setattr(kb, "_check_file_length_invariant", boom)
    monkeypatch.setattr(kb, "_scope_stop_inline", False)
    monkeypatch.setattr(kb, "_ensure_scope_stop_thread", lambda: None)
    # Park any lingering service thread on the OLD wake event (see the
    # connection-local intents test) — the flush must not let it drain
    # the queued entry before the assert reads it.
    import threading as _threading

    monkeypatch.setattr(kb, "_scope_stop_wake", _threading.Event())
    monkeypatch.setattr(kb, "_kanban_scope_state", lambda unit: "unknown")
    kb.reset_scope_stop_service_for_tests()
    unit = kb._kanban_worker_scope_unit("t_inv", 3)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET title='flushed anyway' WHERE id=?",
                    (tid,),
                )
                assert not kb.request_worker_scope_stop(unit, conn=conn)
                assert unit not in kb._scope_stop_pending
        # The transaction COMMITTED (the raise came after COMMIT) and its
        # stop reached the queue despite the invariant exception.
        assert conn.execute(
            "SELECT title FROM tasks WHERE id=?", (tid,),
        ).fetchone()["title"] == "flushed anyway"
        assert unit in kb._scope_stop_pending
    finally:
        kb.reset_scope_stop_service_for_tests()


def test_stop_pending_cleared_on_run_end_and_confirmed_stop(shims, conn):
    """AC: the stop-pending marker has clearing paths. ``_end_run`` wipes
    it when the run closes (a respawn's fresh registration must never see
    a stale marker), and the scope-stop service wipes it once a
    registration-sensitive stop CONFIRMS (a verified-empty cgroup means
    nothing is left to signal, so an adopted run must be registrable
    again)."""
    # Half 1 — run end: a marked run closes, the task respawns, the new
    # run registers cleanly.
    tid = kb.create_task(conn, title="ac respawn", assignee="w")
    claimed = kb.claim_task(conn, tid, claimer=kb._claimer_id())
    assert claimed is not None
    run_n = kb._current_run_id(conn, tid)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET stop_pending=1 WHERE id=?", (run_n,),
        )
        kb._end_run(conn, tid, outcome="crashed", status="crashed")
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL, "
            "worker_pid_started_at=NULL, worker_registered_at=NULL, "
            "worker_scope=NULL WHERE id=?",
            (tid,),
        )
    assert conn.execute(
        "SELECT stop_pending FROM task_runs WHERE id=?", (run_n,),
    ).fetchone()["stop_pending"] == 0
    claimed2 = kb.claim_task(conn, tid, claimer=kb._claimer_id())
    assert claimed2 is not None
    run_m = kb._current_run_id(conn, tid)
    assert run_m != run_n
    assert kb.register_worker_pid(conn, tid, expected_run_id=run_m)

    # Half 2 — confirmed stop: the service marks the run right before
    # signalling, the stop verifies, the marker clears.
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_ac", 1)
    shims.write_unit(unit, [pid])
    tid2 = _scoped_task_row(
        conn, scope=unit, pid=pid, started_delta=-3600,
    )
    run2 = kb._current_run_id(conn, tid2)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET stop_pending=1 WHERE id=?", (run2,),
        )
    # The drain re-marks (CAS) then signals; verified True clears again.
    assert kb.request_worker_scope_stop(
        unit, task_id=tid2, skip_if_registered=True,
    )
    assert shims.wait_for(lambda: not kb._pid_alive(pid))
    assert conn.execute(
        "SELECT stop_pending FROM task_runs WHERE id=?", (run2,),
    ).fetchone()["stop_pending"] == 0


def test_generic_reclaim_never_signals_unreadable_identity(
    conn, monkeypatch, caplog,
):
    """X: the generic reclaim path treats an unreadable pid identity as a
    third state and never signals it — a signal we cannot attribute might
    hit an unrelated process. The row is reclaimed WITHOUT a signal (the
    same fail-safe stance enforce_max_runtime took in pass 4) and the
    stand-down warns once per run id."""
    import logging

    killed: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        killed.append((pid, sig))

    # Alive pid whose live start fingerprint cannot be read.
    monkeypatch.setattr(
        "gateway.status.get_process_start_time", lambda pid: None,
    )
    tid = kb.create_task(conn, title="unreadable identity", assignee="w")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    kb._set_worker_pid(conn, tid, os.getpid())
    run_id = kb._current_run_id(conn, tid)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        assert kb.reclaim_task(
            conn, tid, reason="operator reclaim", signal_fn=fake_kill,
        )
    assert killed == []  # unknown identity — no signal, ever
    row = conn.execute(
        "SELECT status, claim_lock FROM tasks WHERE id=?", (tid,),
    ).fetchone()
    assert row["status"] == "ready"
    assert row["claim_lock"] is None
    payload = json.loads(conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? "
        "AND kind='reclaimed'", (tid,),
    ).fetchone()["payload"])
    assert payload["termination_attempted"] is True
    assert payload["terminated"] is True
    assert payload["signal_skipped"] == "pid_identity_unknown"
    warnings = [
        r for r in caplog.records
        if "not signalling; reclaiming" in r.message
    ]
    assert len(warnings) == 1
    assert str(run_id) in warnings[0].message


def test_generic_reclaim_legacy_missing_fingerprint_not_signalled(
    conn, monkeypatch,
):
    """X, legacy half: a row with NO start fingerprint (pre-column spawn)
    is also 'unknown', never 'alive' — the old boolean helper folded
    missing into alive and signalled the bare pid. The row reclaims
    without a signal."""
    killed: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        killed.append((pid, sig))

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    tid = kb.create_task(conn, title="legacy row", assignee="w")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid=4242, worker_pid_started_at=NULL "
            "WHERE id=?",
            (tid,),
        )
    assert kb.reclaim_task(
        conn, tid, reason="operator reclaim", signal_fn=fake_kill,
    )
    assert killed == []
    payload = json.loads(conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? "
        "AND kind='reclaimed'", (tid,),
    ).fetchone()["payload"])
    assert payload["signal_skipped"] == "pid_identity_unknown"
    assert payload["terminated"] is True


def test_queue_coalescing_never_reenables_skipping(monkeypatch):
    """AB: two queued requests for one unit coalesce into one entry and
    ``skip_if_registered`` composes with AND. A terminal request (False —
    the completed task's scope must be reaped regardless of registration)
    makes the coalesced entry False in BOTH arrival orders: a later True
    can never re-enable skipping past a terminal stop."""
    monkeypatch.setattr(kb, "_scope_stop_inline", False)
    monkeypatch.setattr(kb, "_ensure_scope_stop_thread", lambda: None)
    # Park any lingering service thread on the OLD wake event (see the
    # connection-local intents test) so nothing drains the entries the
    # asserts inspect.
    import threading as _threading

    monkeypatch.setattr(kb, "_scope_stop_wake", _threading.Event())
    monkeypatch.setattr(kb, "_kanban_scope_state", lambda unit: "unknown")
    kb.reset_scope_stop_service_for_tests()
    unit = kb._kanban_worker_scope_unit("t_coal", 2)
    try:
        # Registration-sensitive first, terminal second.
        assert not kb.request_worker_scope_stop(unit, skip_if_registered=True)
        assert not kb.request_worker_scope_stop(
            unit, task_id="t_coal", skip_if_registered=False,
        )
        assert kb._scope_stop_pending[unit].skip_if_registered is False
        kb.reset_scope_stop_service_for_tests()

        # Terminal first, registration-sensitive second.
        assert not kb.request_worker_scope_stop(
            unit, task_id="t_coal", skip_if_registered=False,
        )
        assert not kb.request_worker_scope_stop(unit, skip_if_registered=True)
        assert kb._scope_stop_pending[unit].skip_if_registered is False
    finally:
        kb.reset_scope_stop_service_for_tests()


def test_shutdown_single_budget_bounds_both_joins(
    shims, conn, kanban_home, monkeypatch, caplog,
):
    """I: base + N x per-unit is a CEILING on the whole stop. The old
    code joined the wedged worker for the full budget and THEN stacked a
    whole extra per-unit drain timeout for the service join — nearly
    double the stated budget. One deadline now bounds both joins."""
    import asyncio
    import logging
    import threading

    import gateway.kanban_watchers as kw
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    kb._INITIALIZED_PATHS.clear()
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_hbudget", 1)
    shims.write_unit(unit, [pid])
    _scoped_task_row(conn, scope=unit, pid=pid)

    monkeypatch.setattr(kw, "_SHUTDOWN_STOP_BASE_SECONDS", 0.1)
    monkeypatch.setattr(
        "tools.process_registry.SCOPE_STOP_VERIFY_BOUND_SECONDS", 0.1,
    )

    gate = threading.Event()
    timings: dict[str, float] = {}

    def wedged_stop(c, should_abort=None, **kwargs):
        timings["stop_entered"] = time.time()
        gate.wait(timeout=10.0)  # worse than any budget
        return []

    monkeypatch.setattr(kb, "stop_all_scoped_workers", wedged_stop)

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        asyncio.run(Harness()._kanban_dispatcher_watcher())
        gate.set()  # let the daemon thread finish for teardown

    warnings = [r for r in caplog.records if "still stopping" in r.message]
    assert warnings, "expected the leftover-units warning"
    # budget = 0.1 base + 1 unit x (0.1 bound + 2.0 margin) = 2.2 s. The
    # old stacked joins cost ~budget + per-unit + margin = ~4.3 s; the
    # single deadline costs the budget alone.
    assert "stop_entered" in timings
    elapsed = warnings[0].created - timings["stop_entered"]
    assert elapsed <= 3.0, f"shutdown stop took {elapsed:.1f}s (budget 2.2s)"


# ---------------------------------------------------------------------------
# Migration from the pre-isolation schema
# ---------------------------------------------------------------------------

# The tasks table as of v2026.8.31 — BEFORE worker_pid_started_at,
# worker_scope, or worker_registered_at existed. Column-for-column copy of
# the shipped CREATE (comments trimmed), so the migration test opens a
# genuinely old-shaped file.
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


def test_migration_adds_worker_columns_without_data_loss(tmp_path, monkeypatch):
    """A DB created with the pre-change schema opens cleanly: the three
    worker-lifecycle columns appear via the additive migration, and every
    legacy row survives intact."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = kb.kanban_db_path(board="default")
    import sqlite3

    raw = sqlite3.connect(db_path)
    raw.executescript(_PRE_CHANGE_TASKS_SQL)
    now = int(time.time())
    raw.execute(
        "INSERT INTO tasks (id, title, assignee, status, created_by, "
        "created_at, started_at, worker_pid, claim_lock, claim_expires, "
        "current_run_id, consecutive_failures) "
        "VALUES ('t_legacy1', 'legacy run', 'elias', 'running', 'op', "
        "?, ?, 4242, 'oldhost:1', ?, 3, 2)",
        (now - 5000, now - 4000, now - 60),
    )
    raw.execute(
        "INSERT INTO tasks (id, title, assignee, status, created_by, "
        "created_at, completed_at) "
        "VALUES ('t_legacy2', 'legacy done', 'maya', 'done', 'op', ?, ?)",
        (now - 9000, now - 8000),
    )
    raw.commit()
    raw.close()

    conn = kb.connect(db_path)
    try:
        cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(tasks)")
        }
        assert {"worker_pid_started_at", "worker_scope",
                "worker_registered_at"} <= cols
        legacy = conn.execute(
            "SELECT title, assignee, status, worker_pid, claim_lock, "
            "       consecutive_failures, worker_pid_started_at, "
            "       worker_scope, worker_registered_at "
            "FROM tasks WHERE id = 't_legacy1'"
        ).fetchone()
        assert legacy["title"] == "legacy run"
        assert legacy["status"] == "running"
        assert legacy["worker_pid"] == 4242
        assert legacy["claim_lock"] == "oldhost:1"
        assert legacy["consecutive_failures"] == 2
        assert legacy["worker_pid_started_at"] is None
        assert legacy["worker_scope"] is None
        assert legacy["worker_registered_at"] is None
        done = conn.execute(
            "SELECT status, completed_at FROM tasks WHERE id = 't_legacy2'"
        ).fetchone()
        assert done["status"] == "done"
        assert done["completed_at"] == now - 8000
    finally:
        conn.close()


# A task_runs table as of v2026.8.31 — everything except worker_scope
# (the one column the isolation feature adds to runs).
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


def test_migration_completes_a_partial_pre_existing_schema(
    tmp_path, monkeypatch,
):
    """I (new e): a DB caught mid-migration — tasks already carries TWO of
    the three worker-lifecycle columns (with live data in them) while
    task_runs predates the feature entirely — opens cleanly: the missing
    column is added, already-migrated values survive untouched, the run
    table gains its column, and a second open is a no-op."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = kb.kanban_db_path(board="default")
    import sqlite3

    raw = sqlite3.connect(db_path)
    raw.executescript(_PRE_CHANGE_TASKS_SQL)
    raw.executescript(_PRE_CHANGE_TASK_RUNS_SQL)
    # Half-migrated tasks: the fingerprint and registration columns
    # shipped, worker_scope did not (a deploy interrupted mid-rollout).
    raw.execute("ALTER TABLE tasks ADD COLUMN worker_pid_started_at INTEGER")
    raw.execute("ALTER TABLE tasks ADD COLUMN worker_registered_at INTEGER")
    now = int(time.time())
    raw.execute(
        "INSERT INTO tasks (id, title, assignee, status, created_by, "
        "created_at, started_at, worker_pid, worker_pid_started_at, "
        "worker_registered_at, claim_lock, claim_expires, current_run_id) "
        "VALUES ('t_partial', 'half migrated', 'elias', 'running', 'op', "
        "?, ?, 4242, 1712345678, ?, 'oldhost:1', ?, 11)",
        (now - 5000, now - 4000, now - 3000, now - 60),
    )
    raw.execute(
        "INSERT INTO task_runs (id, task_id, profile, status, claim_lock, "
        "worker_pid, started_at) VALUES (11, 't_partial', 'elias', "
        "'running', 'oldhost:1', 4242, ?)",
        (now - 4000,),
    )
    raw.commit()
    raw.close()

    conn = kb.connect(db_path)
    try:
        cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(tasks)")
        }
        run_cols = {
            r["name"] for r in conn.execute("PRAGMA table_info(task_runs)")
        }
        assert {"worker_pid_started_at", "worker_scope",
                "worker_registered_at"} <= cols
        assert "worker_scope" in run_cols
        task = conn.execute(
            "SELECT worker_pid_started_at, worker_registered_at "
            "FROM tasks WHERE id = 't_partial'"
        ).fetchone()
        # The already-migrated half is preserved, not reset to defaults.
        assert task["worker_pid_started_at"] == 1712345678
        assert task["worker_registered_at"] == now - 3000
        run = conn.execute(
            "SELECT worker_pid, status, worker_scope FROM task_runs "
            "WHERE id = 11"
        ).fetchone()
        assert run["worker_pid"] == 4242
        assert run["status"] == "running"
        assert run["worker_scope"] is None
    finally:
        conn.close()

    # Idempotent: a second open changes nothing.
    conn = kb.connect(db_path)
    try:
        task = conn.execute(
            "SELECT worker_pid_started_at, worker_registered_at, "
            "worker_scope FROM tasks WHERE id = 't_partial'"
        ).fetchone()
        assert task["worker_pid_started_at"] == 1712345678
        assert task["worker_registered_at"] == now - 3000
        assert task["worker_scope"] is None
    finally:
        conn.close()


def test_stop_all_scoped_workers_honors_abort_between_units(
    shims, conn, monkeypatch,
):
    """Q, unit half: the per-unit abort check stands the stop loop down
    BETWEEN units — a cancelled shutdown never proceeds to the next
    worker, and the units it did not reach are simply not reported
    stopped."""
    u1 = kb._kanban_worker_scope_unit("t_abort1", 1)
    u2 = kb._kanban_worker_scope_unit("t_abort2", 2)
    p1, p2 = shims.sleeper(), shims.sleeper()
    shims.write_unit(u1, [p1])
    shims.write_unit(u2, [p2])
    _scoped_task_row(conn, scope=u1, pid=p1)
    _scoped_task_row(conn, scope=u2, pid=p2)
    stopped_calls: list[str] = []
    monkeypatch.setattr(
        kb, "_stop_kanban_worker_scope",
        lambda unit, **kw: (stopped_calls.append(unit), True)[1],
    )

    def abort_after_first() -> bool:
        return len(stopped_calls) >= 1

    stopped = kb.stop_all_scoped_workers(conn, should_abort=abort_after_first)

    assert stopped == [u1]
    assert stopped_calls == [u1]  # u2 was never attempted


def test_shutdown_cancels_cleanup_thread_and_reports_unstopped(
    shims, conn, kanban_home, monkeypatch, caplog,
):
    """Q, wiring half: when the shutdown budget expires, the cleanup
    daemon thread is CANCELLED (stop event + same deadline) instead of
    scanning and stopping scopes after the dispatcher lock is released,
    and the caller's warning names what was left un-stopped."""
    import asyncio
    import logging
    import threading

    import gateway.kanban_watchers as kw
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    kb._INITIALIZED_PATHS.clear()
    u1 = kb._kanban_worker_scope_unit("t_slowstop1", 1)
    u2 = kb._kanban_worker_scope_unit("t_slowstop2", 2)
    p1, p2 = shims.sleeper(), shims.sleeper()
    shims.write_unit(u1, [p1])
    shims.write_unit(u2, [p2])
    _scoped_task_row(conn, scope=u1, pid=p1)
    _scoped_task_row(conn, scope=u2, pid=p2)

    monkeypatch.setattr(kw, "_SHUTDOWN_STOP_BASE_SECONDS", 0.1)
    monkeypatch.setattr(
        "tools.process_registry.SCOPE_STOP_VERIFY_BOUND_SECONDS", 0.1,
    )

    attempts: list[str] = []
    gate = threading.Event()

    def slow_verified_stop(unit, **kwargs):
        attempts.append(unit)
        if unit == u1:
            gate.wait(timeout=15.0)  # slow fake stop: outlives the budget
        return True

    monkeypatch.setattr(kb, "_stop_kanban_worker_scope", slow_verified_stop)

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        # budget = 0.1 base + 2 units x (0.1 bound + 2.0 margin) = 4.3 s.
        # u1's stop blocks past it: the caller times out, cancels the
        # cleanup thread, and reports BOTH units (u1 unconfirmed, u2 not
        # reached) while u1's stop is still blocked.
        asyncio.run(Harness()._kanban_dispatcher_watcher())
        gate.set()  # let u1's stop return so the thread hits the check

    warnings = [r for r in caplog.records if "still stopping" in r.message]
    assert warnings, "expected the leftover-units warning"
    assert u1 in warnings[0].message and u2 in warnings[0].message

    # Bounded wait for the daemon thread to observe the cancellation and
    # stand down BEFORE u2.
    deadline = time.monotonic() + 5.0
    stood_down = []
    while time.monotonic() < deadline:
        stood_down = [
            r for r in caplog.records if "stood down before" in r.message
        ]
        if stood_down:
            break
        time.sleep(0.05)
    assert stood_down, "cleanup thread never logged its stand-down"
    assert u2 in stood_down[0].message
    assert attempts == [u1]  # u2 was never signalled after cancellation

def test_shutdown_prescan_timeout_reports_incomplete_not_zero(
    shims, conn, kanban_home, monkeypatch, caplog,
):
    """Y: when the shutdown pre-scan outlives its base budget the summary
    must say the scan is incomplete — never a confident "0 unit(s) still
    stopping" for boards it never enumerated."""
    import asyncio
    import logging
    import threading

    import gateway.kanban_watchers as kw
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    kb._INITIALIZED_PATHS.clear()
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_scanx", 1)
    shims.write_unit(unit, [pid])
    _scoped_task_row(conn, scope=unit, pid=pid)

    monkeypatch.setattr(kw, "_SHUTDOWN_STOP_BASE_SECONDS", 0.1)
    monkeypatch.setattr(
        "tools.process_registry.SCOPE_STOP_VERIFY_BOUND_SECONDS", 0.1,
    )

    gate = threading.Event()
    real_list_boards = kb.list_boards

    def stalled_list_boards(**kwargs):
        gate.wait(timeout=15.0)  # board listing wedged past any budget
        return real_list_boards(**kwargs)

    monkeypatch.setattr(kb, "list_boards", stalled_list_boards)

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        asyncio.run(Harness()._kanban_dispatcher_watcher())
        gate.set()  # let the daemon thread finish for teardown

    warnings = [r for r in caplog.records if "still stopping" in r.message]
    assert warnings, "expected the leftover-units warning"
    msg = warnings[0].message
    assert "scan incomplete" in msg
    assert "not enumerated" in msg
    assert "0 unit(s) still stopping" not in msg, (
        "an incomplete scan must not present an enumerated zero"
    )

def test_shutdown_prescan_partial_names_unscanned_boards(
    shims, conn, kanban_home, monkeypatch, caplog,
):
    """Y, partial-scan branch: the board listing returned (so the board
    count is known) but a board's connect stalls — the summary names the
    unscanned boards instead of a zero."""
    import asyncio
    import logging
    import threading

    import gateway.kanban_watchers as kw
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    _write_kanban_config(
        Path(kanban_home), "  worker_isolation_stop_on_shutdown: true\n"
    )
    kb._INITIALIZED_PATHS.clear()
    pid = shims.sleeper()
    unit = kb._kanban_worker_scope_unit("t_scanp", 1)
    shims.write_unit(unit, [pid])
    _scoped_task_row(conn, scope=unit, pid=pid)

    monkeypatch.setattr(kw, "_SHUTDOWN_STOP_BASE_SECONDS", 0.1)
    monkeypatch.setattr(
        "tools.process_registry.SCOPE_STOP_VERIFY_BOUND_SECONDS", 0.1,
    )

    gate = threading.Event()
    real_connect = kb.connect

    def stalled_connect(*args, **kwargs):
        gate.wait(timeout=15.0)  # per-board scan wedged past any budget
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(kb, "connect", stalled_connect)

    class Harness(GatewayKanbanWatchersMixin):
        def __init__(self):
            self._running = False
            self._kanban_dispatcher_lock_handle = None

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        asyncio.run(Harness()._kanban_dispatcher_watcher())
        gate.set()  # let the daemon thread finish for teardown

    warnings = [r for r in caplog.records if "still stopping" in r.message]
    assert warnings, "expected the leftover-units warning"
    msg = warnings[0].message
    assert "scan incomplete" in msg
    assert "unscanned board(s) not enumerated" in msg
    assert "0 unit(s) still stopping" not in msg
    pre = [r for r in caplog.records if "pre-scan incomplete" in r.message]
    assert pre, "expected the pre-scan timeout warning"
