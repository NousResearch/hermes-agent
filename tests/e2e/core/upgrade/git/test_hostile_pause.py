"""Linux/macOS ``hermes update`` pauses this install's gateways before the first checkout move.

One real HEAD install (``_git_world``: the real ``scripts/install.sh`` over real smart HTTP, every
command in a ``bwrap`` sandbox) is shared by the cells. A credential-free, platform-less (cron-only)
``hermes gateway run`` lives in the same PID namespace as the updater (``handoff._nshost``), as on a
user's machine. What is asserted is what the user sees: which gateway process serves the checkout
(its control-socket ``identify``: pid + ``code_sha``), the update's exit code and transcript, HEAD,
and the durable pause record ``update_pause_record`` keeps in the root hermes home.

(a) a real release: the gateway is gone BEFORE HEAD moves (a 0.1 s timeline of both), and exactly
    one gateway, a new process, serves the new commit once the update returns; the restart line is
    printed after the dependency sync and before the product builds.
(b) no-op update (already current): the same gateway process keeps serving; nothing is stopped.
(c) the fetch fails (origin unreachable): nothing is stopped.
(d) ``--no-gateway-restart``: nothing is stopped before the move (the user manages gateways).
(e) SIGKILL mid-stop: the gateway is SIGSTOPped so the updater blocks in its control-socket pause
    request with that request recorded; the whole update tree is SIGKILLed there. The
    next ``hermes`` launch must leave exactly one gateway serving: none lost, none duplicated.
(f) SIGKILL after the commit, before the restart: the release's dependency preparation holds while
    a toggle exists; the update tree is SIGKILLed once HEAD is the release and the gateway is
    paused. The next launch syncs the dependencies and restarts exactly the paused gateway on the
    new commit.
(g) the dependency sync fails after the commit (contract amendment A6): the release's dependency
    preparation raises while a toggle exists and changes ``uv.lock``'s bytes (the dependency stamp),
    so the venv is stale at the new HEAD. The update exits 0 with a ``dependencies`` follow-up and
    the paused gateway stays stopped (the tree gate holds it: it would boot new code on stale
    dependencies). A ``hermes gateway run`` launched then (the replayed argv) syncs the owed
    dependencies in its own launch before serving; the next ``hermes`` launch adopts that gateway
    instead of starting a twin, and the record is discharged.
(i) the checkout move itself fails before the commit point (the release adds a file under a
    directory the user cannot write): the paused gateway is restarted on the OLD code, and the
    update's exit code and receipt outcome equal those of the same failing update run with no
    gateway at all.
(h) the Desktop hand-off (``scripts/desktop-update/posix.sh``, what the Desktop app spawns before it
    quits) runs the same ``hermes update --gateway``: the gateway is paused and restarted on the new
    commit, and the hand-off's result file reports success.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import tempfile
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest

import hermes_yaml as yaml
from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade.git import _git_world as G
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.handoff._nshost import NamespaceHost

pytestmark = G.PYTESTMARK

PAUSED = "⏸ Paused for the update"
RESTARTED = "▶ Restarted paused gateway(s)"
SLOW_DEPS = ".e2e-slow-deps-sync"
DEPS_STARTED = ".e2e-deps-sync-started"
UPDATE_TIMEOUT = 1500
_DEPS_MARK = "# e2e (test_hostile_pause): slow dependency sync toggle"
BREAK_DEPS = ".e2e-break-deps-sync"
DEPS_OWED = "dependencies not installed yet"  # update_completion._settle_after_commit
_BREAK_MARK = "# e2e (test_hostile_pause): failing dependency sync toggle"
_BREAK_DEPS = f"""    {_BREAK_MARK}
    if __import__("os").path.exists(__import__("os").path.join(__import__("os").path.expanduser("~"), "{BREAK_DEPS}")):
        raise RuntimeError("e2e: dependency sync fails on purpose ({BREAK_DEPS})")
"""
_SLOW_DEPS = f"""    {_DEPS_MARK}
    _e2e_home = __import__("os").path.expanduser("~")
    if __import__("os").path.exists(__import__("os").path.join(_e2e_home, "{SLOW_DEPS}")):
        open(__import__("os").path.join(_e2e_home, "{DEPS_STARTED}"), "w").write("started")
        while __import__("os").path.exists(__import__("os").path.join(_e2e_home, "{SLOW_DEPS}")):
            __import__("time").sleep(0.2)
"""


@pytest.fixture(scope="module")
def w() -> Iterator[G.World]:
    """A SHORT root: the gateway's AF_UNIX control socket lives under ``$HERMES_HOME``."""
    base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    root = Path(tempfile.mkdtemp(prefix="hz", dir=base))
    try:
        with G.world(root, base=I.head_sha()) as world:
            cfg_path = world.sb.hermes_home / "config.yaml"
            cfg = (yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig")) if cfg_path.is_file() else None) or {}
            cfg["updates"] = {**(cfg.get("updates") or {}), "check": False}
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            yield world
    finally:
        if not os.environ.get("HERMES_E2E_KEEP_ROOT"):
            shutil.rmtree(root, ignore_errors=True)


# -- helpers ---------------------------------------------------------------------------------------


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig", errors="replace") if p.is_file() else ""


def _ref_head(w: G.World) -> str:
    """HEAD straight from the ref files (no git process, safe at 10 Hz)."""
    git = w.checkout / ".git"
    head = _read(git / "HEAD").strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    loose = _read(git / ref).strip()
    if loose:
        return loose
    for line in _read(git / "packed-refs").splitlines():
        if line.endswith(" " + ref):
            return line.split()[0]
    return ""


def _identify(w: G.World) -> dict | None:
    from gateway.control_socket import identify_gateway

    with contextlib.suppress(Exception):
        return identify_gateway(w.sb.hermes_home, timeout=5.0)
    return None


def _serving(w: G.World, sha: str) -> dict | None:
    ident = _identify(w)
    return ident if ident and ident.get("code_sha") == sha else None


def _gateways(host: NamespaceHost) -> list[dict]:
    """``gateway run`` processes in the namespace, by the canonical matcher."""
    from gateway.status import looks_like_gateway_command_line

    return [p for p in host.procs() if p["state"] not in ("Z", "X")
            and looks_like_gateway_command_line(" ".join(p["cmdline"]))]


def _record(w: G.World) -> list[dict]:
    """Every paused-gateway record (live, orphaned, claimed) in the root hermes home."""
    found = []
    for p in sorted(w.sb.hermes_home.glob(".hermes-update-paused-gateways*.json")):
        with contextlib.suppress(OSError, ValueError):
            found.append({"path": p.name, **json.loads(p.read_text(encoding="utf-8-sig"))})
    return found


@contextlib.contextmanager
def _namespace(w: G.World) -> Iterator[NamespaceHost]:
    host = NamespaceHost(w.sb.root, w.sb.env)
    try:
        yield host
    finally:
        leaked = host.close()
        (w.sb.hermes_home / "gateway.pid").unlink(missing_ok=True)  # names a pid of the dead namespace
        assert not leaked, f"processes outlived the sandbox: {leaked}"


def _ns_cli(host: NamespaceHost, w: G.World, *args: str, timeout: float = 600):
    cp = host.run([w.sb.hermes, *args], timeout=timeout)
    w.transcripts.append(f"$ hermes {' '.join(args)} -> rc={cp.returncode}\n{(cp.stdout or '')[-6000:]}\n"
                         f"{(cp.stderr or '')[-3000:]}")
    return cp


def _start_gateway(host: NamespaceHost, w: G.World, tag: str) -> dict:
    host.spawn([w.sb.hermes, "gateway", "run"], log=w.sb.root / f"gateway-{tag}.log")
    sha = w.head()
    try:
        return H.wait_for(lambda: _serving(w, sha), timeout=240, interval=1.0, what="the gateway's identify")
    except AssertionError as exc:
        raise AssertionError(f"premise: the gateway never came up: {exc}\n{host.ps_text()}\n"
                             f"{_read(w.sb.root / f'gateway-{tag}.log')[-4000:]}") from None


def _kill_born_after(host: NamespaceHost, before: set[int], keep: set[int] | frozenset[int] = frozenset()) -> list[int]:
    killed: list[int] = []
    for _ in range(10):
        live = [p["pid"] for p in host.procs()
                if p["pid"] not in before and p["pid"] not in keep and p["state"] not in ("Z", "X")]
        if not live:
            break
        for pid in live:
            host.kill(pid, signal.SIGSTOP)
        for pid in live:
            host.kill(pid, signal.SIGKILL)  # windows-footgun: ok - linux-only (bwrap) module
            if pid not in killed:
                killed.append(pid)
        time.sleep(0.5)
    return killed


class _Timeline:
    """10 Hz samples of (gateway pid alive, HEAD) while the update runs."""

    def __init__(self, host: NamespaceHost, w: G.World, pid: int):
        self.host, self.w, self.pid = host, w, pid
        self.rows: list[tuple[float, bool, str]] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        t0 = time.monotonic()
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self.rows.append((time.monotonic() - t0, self.host.alive(self.pid), _ref_head(self.w)))
            time.sleep(0.1)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=10)

    def first(self, pred) -> float | None:
        return next((t for t, alive, head in self.rows if pred(alive, head)), None)

    def text(self) -> str:
        out, last = [], None
        for t, alive, head in self.rows:
            state = (alive, head[:10])
            if state != last:
                out.append(f"  t={t:7.2f}s gateway_pid_alive={alive} HEAD={head[:10]}")
                last = state
        return "\n".join(out)


def _publish_release(w: G.World, tag: str) -> str:
    """A release that changes a module the gateway imports lazily plus a web build input."""
    return w.publish(f"release: e2e hostile pause {tag}", {
        f"docs/e2e-hostile-pause-{tag}.txt": f"release {tag}\n",
        f"web/e2e-hostile-pause-{tag}.txt": f"web input changed by release {tag}\n",
    })


def _publish_slow_deps_release(w: G.World, tag: str) -> str:
    text = w.git("show", "HEAD:pm/client.py") + "\n"
    if _DEPS_MARK not in text:
        m = re.search(r"^def ensure_tools_for_sync\(\) -> None:\n    \"\"\".*?\"\"\"\n", text, re.S | re.M)
        assert m, "premise: pm/client.py has no ensure_tools_for_sync() with a docstring to inject after"
        text = text[:m.end()] + _SLOW_DEPS + text[m.end():]
    return w.publish(f"release: e2e hostile pause slow deps {tag}", {
        "pm/client.py": text, f"docs/e2e-hostile-pause-{tag}.txt": f"release {tag}\n"})


def _publish_broken_deps_release(w: G.World, tag: str) -> str:
    text = w.git("show", "HEAD:pm/client.py") + "\n"
    if _BREAK_MARK not in text:
        m = re.search(r"^def ensure_tools_for_sync\(\) -> None:\n    \"\"\".*?\"\"\"\n", text, re.S | re.M)
        assert m, "premise: pm/client.py has no ensure_tools_for_sync() with a docstring to inject after"
        text = text[:m.end()] + _BREAK_DEPS + text[m.end():]
    # pm's dependency stamp hashes uv.lock's bytes: a trailing TOML comment makes the venv stale
    # at the new HEAD without changing what any dependency resolves to.
    lock = w.git("show", "HEAD:uv.lock") + f"\n# e2e hostile pause {tag}: dependency stamp moves\n"
    return w.publish(f"release: e2e hostile pause broken deps {tag}", {
        "pm/client.py": text, "uv.lock": lock, f"docs/e2e-hostile-pause-{tag}.txt": f"release {tag}\n"})


def _diag(w: G.World, host: NamespaceHost, cp=None, **extra) -> str:
    parts = [f"{k}={v}" for k, v in extra.items()]
    parts.append("pause records: " + json.dumps(_record(w), default=str)[:3000])
    parts.append("--- sandbox ---\n" + host.ps_text())
    text = "\n".join(parts) + "\n" + w.diag(cp)
    print(text[:12000])  # the cell's receipt: ``-rA`` shows it for passing cells too
    return text


# -- cells -----------------------------------------------------------------------------------------


def test_real_release_pauses_before_the_move_and_restarts_after_deps(w):
    w.reset_clean()
    with _namespace(w) as host:
        # A second profile: the one multiplexing gateway serves both and is paused/restarted once.
        prof = _ns_cli(host, w, "profile", "create", "p2probe-a", "--no-alias")
        assert prof.returncode == 0 or "already exists" in G.output(prof), f"premise: {G.output(prof)[-2000:]}"
        before = _start_gateway(host, w, "a")
        old_sha, old_pid = w.head(), int(before["pid"])
        target = _publish_release(w, "a")
        with _Timeline(host, w, old_pid) as tl:
            cp = _ns_cli(host, w, "update", "--yes", "--branch", "main", timeout=UPDATE_TIMEOUT)
        out = G.output(cp)
        after = H.wait_for(lambda: _serving(w, target), timeout=60, interval=1.0, what="a gateway on the release") \
            if cp.returncode == 0 else None
        gws = _gateways(host)
        diag = _diag(w, host, cp, timeline="\n" + tl.text(), before=before, after=after, gateways=gws)

        assert cp.returncode == 0 and w.head() == target, f"premise: the update did not land:\n{diag}"
        gone, moved = tl.first(lambda alive, _h: not alive), tl.first(lambda _a, head: head == target)
        assert gone is not None and moved is not None, f"timeline missed the stop or the move:\n{diag}"
        assert tl.first(lambda alive, head: alive and head == target) is None, \
            f"the OLD gateway process was alive while HEAD named the release:\n{diag}"
        assert gone < moved, f"the gateway stopped only after the checkout moved:\n{diag}"
        assert PAUSED in out and RESTARTED in out, f"the pause/restart lines are missing:\n{diag}"
        assert out.index(PAUSED) < out.index(RESTARTED), f"restart printed before the pause:\n{diag}"
        built = [m.start() for m in re.finditer(r"web UI|Building", out)]
        assert not built or out.index(RESTARTED) < built[0], \
            f"the paused gateway was restarted only after the product builds:\n{diag}"
        assert after and int(after["pid"]) != old_pid, f"no NEW gateway process serves {target[:12]}:\n{diag}"
        assert len(gws) == 1, f"expected exactly one gateway process, found {len(gws)}:\n{diag}"
        paused_line = next(line for line in out.splitlines() if PAUSED in line)
        assert paused_line.count("gateway PID") == 1, f"the multiplexer was paused more than once:\n{diag}"
        assert not [r for r in _record(w) if (r.get("token") or {}).get("resume_needed")], \
            f"a discharged pause left an owed record:\n{diag}"
        assert old_sha != target


def test_noop_update_stops_nothing(w):
    w.reset_clean()
    with _namespace(w) as host:
        before = _start_gateway(host, w, "b")
        cp = _ns_cli(host, w, "update", "--yes", "--branch", "main", timeout=UPDATE_TIMEOUT)
        ident = _identify(w)
        diag = _diag(w, host, cp, before=before, after=ident)
        assert cp.returncode == 0, f"the no-op update failed:\n{diag}"
        assert PAUSED not in G.output(cp), f"an already-current update paused the gateways:\n{diag}"
        assert ident and ident.get("pid") == before.get("pid"), f"the gateway process changed:\n{diag}"


def test_failed_fetch_stops_nothing(w):
    w.reset_clean()
    url = w.config("remote.origin.url")
    with _namespace(w) as host:
        before = _start_gateway(host, w, "c")
        _publish_release(w, "c")
        w.git("remote", "set-url", "origin", "http://127.0.0.1:9/unreachable.git")
        try:
            cp = _ns_cli(host, w, "update", "--yes", "--branch", "main", timeout=UPDATE_TIMEOUT)
        finally:
            w.git("remote", "set-url", "origin", url)
        ident = _identify(w)
        diag = _diag(w, host, cp, before=before, after=ident)
        assert cp.returncode != 0, f"premise: the update with an unreachable origin succeeded:\n{diag}"
        assert PAUSED not in G.output(cp), f"a fetch failure paused the gateways:\n{diag}"
        assert ident and ident.get("pid") == before.get("pid"), f"the gateway process changed:\n{diag}"


def test_no_gateway_restart_stops_nothing_before_the_move(w):
    w.reset_clean()
    with _namespace(w) as host:
        before = _start_gateway(host, w, "d")
        target = _publish_release(w, "d")
        with _Timeline(host, w, int(before["pid"])) as tl:
            cp = _ns_cli(host, w, "update", "--yes", "--branch", "main", "--no-gateway-restart",
                         timeout=UPDATE_TIMEOUT)
        diag = _diag(w, host, cp, timeline="\n" + tl.text(), before=before, after=_identify(w))
        assert cp.returncode == 0 and w.head() == target, f"premise: the update did not land:\n{diag}"
        assert PAUSED not in G.output(cp), f"--no-gateway-restart paused the gateways:\n{diag}"
        assert tl.first(lambda alive, _h: not alive) is None, f"--no-gateway-restart stopped the gateway:\n{diag}"


def test_sigkill_mid_stop_next_launch_restores_exactly_the_set(w):
    w.reset_clean()
    with _namespace(w) as host:
        before = _start_gateway(host, w, "e")
        gw_pid, sha = int(before["pid"]), w.head()
        _publish_release(w, "e")
        procs0 = {p["pid"] for p in host.procs()}
        host.kill(gw_pid, signal.SIGSTOP)  # the socket pause request queues; the updater waits on it
        killed: list[int] = []
        try:
            upd = host.spawn([w.sb.hermes, "update", "--yes", "--branch", "main"], log=w.sb.root / "update-e.log")

            def _request_on_disk():
                # The pause records its request before sending it (no planned-stop marker on the drain path).
                if not host.alive(upd):
                    raise AssertionError("premise: the update exited before its stop request:\n"
                                         + _read(w.sb.root / "update-e.log")[-6000:])
                return any(str(gw_pid) in ((r.get("token") or {}).get("stop_sent") or []) for r in _record(w))

            try:
                H.wait_for(_request_on_disk, timeout=900, interval=0.2, what="the updater's recorded stop request")
            except AssertionError as exc:
                raise AssertionError(f"{exc}\nrecord: {_record(w)}\nupdate log:\n"
                                     + _read(w.sb.root / "update-e.log")[-6000:]) from None
            time.sleep(1.0)  # inside the socket call now (its timeout is seconds, the drain minutes)
            rec_at_kill = _record(w)
            killed = _kill_born_after(host, procs0, keep={gw_pid})
        finally:
            with contextlib.suppress(AssertionError):  # a gateway already gone must not mask the real failure
                host.kill(gw_pid, signal.SIGCONT)
        head_at_kill = w.head()
        # The queued request may still drain it: let that settle before the next launch judges.
        with contextlib.suppress(AssertionError):
            H.wait_for(lambda: not host.alive(gw_pid), timeout=120, interval=1.0, what="the old gateway to drain")
        status = _ns_cli(host, w, "status", timeout=600)
        served = None
        with contextlib.suppress(AssertionError):
            served = H.wait_for(lambda: _identify(w), timeout=120, interval=1.0, what="a serving gateway")
        time.sleep(3.0)
        gws = _gateways(host)
        diag = _diag(w, host, None, killed=killed, record_at_kill=rec_at_kill, head_at_kill=head_at_kill,
                     served=served, gateways=gws, status=G.output(status)[-3000:],
                     update_log=_read(w.sb.root / "update-e.log")[-5000:])
        assert killed, f"premise: nothing was killed:\n{diag}"
        assert rec_at_kill and any((r.get("token") or {}).get("unmapped") for r in rec_at_kill), \
            f"premise: no durable record named the gateway before its stop request:\n{diag}"
        assert head_at_kill == sha, f"premise: the checkout moved before the kill:\n{diag}"
        assert served, f"the next launch left NO gateway serving (lost):\n{diag}"
        assert len(gws) == 1, f"expected exactly one gateway after recovery, found {len(gws)}:\n{diag}"


def test_sigkill_after_commit_before_restart_next_launch_restarts_on_new_code(w):
    w.reset_clean()
    started, toggle = w.sb.home / DEPS_STARTED, w.sb.home / SLOW_DEPS
    with _namespace(w) as host:
        before = _start_gateway(host, w, "f")
        gw_pid = int(before["pid"])
        target = _publish_slow_deps_release(w, "f")
        procs0 = {p["pid"] for p in host.procs()}
        toggle.write_text("on\n", encoding="utf-8")
        killed: list[int] = []
        try:
            upd = host.spawn([w.sb.hermes, "update", "--yes", "--branch", "main"], log=w.sb.root / "update-f.log")

            def _held_after_commit():
                if not host.alive(upd):
                    raise AssertionError("premise: the update exited before the slow dependency sync:\n"
                                         + _read(w.sb.root / "update-f.log")[-6000:])
                return started.is_file() and _ref_head(w) == target

            H.wait_for(_held_after_commit, timeout=1200, interval=0.5, what="the committed update in its deps sync")
            gw_alive_at_kill = host.alive(gw_pid)
            rec_at_kill = _record(w)
            killed = _kill_born_after(host, procs0, keep={gw_pid})
        finally:
            toggle.unlink(missing_ok=True)
            started.unlink(missing_ok=True)
        status = _ns_cli(host, w, "status", timeout=900)
        served = None
        with contextlib.suppress(AssertionError):
            served = H.wait_for(lambda: _serving(w, target), timeout=180, interval=1.0, what="a gateway on the release")
        time.sleep(3.0)
        gws = _gateways(host)
        diag = _diag(w, host, None, killed=killed, gw_alive_at_kill=gw_alive_at_kill, record_at_kill=rec_at_kill,
                     served=served, gateways=gws, status=G.output(status)[-3000:],
                     update_log=_read(w.sb.root / "update-f.log")[-5000:])
        assert killed and w.head() == target, f"premise: the kill did not land after the commit:\n{diag}"
        assert not gw_alive_at_kill, f"the gateway still ran old code while HEAD named the release:\n{diag}"
        assert served, f"the next launch did not restart the paused gateway on the release:\n{diag}"
        assert int(served["pid"]) != gw_pid and len(gws) == 1, f"lost or duplicated gateway:\n{diag}"


def test_deps_failure_after_commit_holds_the_paused_set_until_a_launch_syncs(w):
    w.reset_clean()
    toggle = w.sb.home / BREAK_DEPS
    # As users run: the suite-wide HERMES_DISABLE_LAZY_INSTALLS also turns off each launch's
    # dependency sync (venv_sync.prepare_launch), and with it the tree gate's dependency hold.
    lazy = w.sb.env.pop("HERMES_DISABLE_LAZY_INSTALLS", None)
    try:
        _deps_failure_cell(w, toggle)
    finally:
        if lazy is not None:
            w.sb.env["HERMES_DISABLE_LAZY_INSTALLS"] = lazy


def _deps_failure_cell(w: G.World, toggle: Path) -> None:
    with _namespace(w) as host:
        before = _start_gateway(host, w, "g")
        gw_pid = int(before["pid"])
        target = _publish_broken_deps_release(w, "g")
        toggle.write_text("on\n", encoding="utf-8")
        try:
            cp = _ns_cli(host, w, "update", "--yes", "--branch", "main", timeout=UPDATE_TIMEOUT)
        finally:
            toggle.unlink(missing_ok=True)
        out = G.output(cp)
        held = _gateways(host)
        rec_after_update = _record(w)
        diag = _diag(w, host, cp, before=before, gateways_after_update=held)
        assert w.head() == target, f"premise: the release never committed:\n{diag}"
        assert DEPS_OWED in out, f"premise: the dependency sync did not fail after the commit:\n{diag}"
        assert cp.returncode == 0, f"A6: a dependency failure after the commit failed the update:\n{diag}"
        assert PAUSED in out, f"premise: the gateway was not paused:\n{diag}"
        assert not held, f"a paused gateway was started on the new code with stale dependencies:\n{diag}"
        assert any((r.get("token") or {}).get("resume_needed") for r in rec_after_update), \
            f"the held set is not owed in the durable record:\n{diag}"

        # The replayed argv's own launch: does it sync the owed dependencies before importing?
        log = w.sb.root / "gateway-g2.log"
        host.spawn([w.sb.hermes, "gateway", "run"], log=log)
        served = None
        with contextlib.suppress(AssertionError):
            served = H.wait_for(lambda: _serving(w, target), timeout=600, interval=1.0, what="a gateway on the release")
        launch_log = _read(log)
        status = _ns_cli(host, w, "status", timeout=600)
        time.sleep(3.0)
        gws = _gateways(host)
        diag = _diag(w, host, None, served=served, gateways=gws, launch_log=launch_log[-4000:],
                     status=G.output(status)[-3000:], launch_log_after=_read(log)[len(launch_log):][-4000:],
                     resume_log=_read(w.sb.hermes_home / "logs" / "gateway-update-resume.log")[-4000:],
                     gateway_log=_read(w.sb.hermes_home / "logs" / "gateway.log")[-6000:])
        assert served and int(served["pid"]) != gw_pid, f"the gateway's own launch did not come up on the release:\n{diag}"
        assert len(gws) == 1, f"the next launch started a twin beside the running gateway:\n{diag}"
        assert not [r for r in _record(w) if (r.get("token") or {}).get("resume_needed")], \
            f"the record still owes a gateway that is serving:\n{diag}"


def test_desktop_handoff_update_pauses_and_restarts_through_the_same_path(w):
    w.reset_clean()
    with _namespace(w) as host:
        before = _start_gateway(host, w, "h")
        old_pid = int(before["pid"])
        target = _publish_release(w, "h")
        result = w.sb.hermes_home / ".hermes-update-result.json"
        result.unlink(missing_ok=True)
        log = w.sb.hermes_home / "logs" / "desktop-update-handoff.log"
        log0 = len(_read(log))
        ulog = w.sb.hermes_home / "logs" / "update.log"
        ulog0 = len(_read(ulog))
        with _Timeline(host, w, old_pid) as tl:
            cp = host.run(["bash", str(w.checkout / "scripts" / "desktop-update" / "posix.sh"),
                           "--install-root", str(w.checkout), "--desktop-pid", "0", "--no-ui"],
                          timeout=UPDATE_TIMEOUT, cwd=w.checkout)

            def finished() -> dict | None:  # posix.sh re-execs itself detached (--daemonized)
                with contextlib.suppress(OSError, ValueError):
                    return json.loads(result.read_text(encoding="utf-8-sig"))
                return None
            res = {}
            with contextlib.suppress(AssertionError):
                res = H.wait_for(finished, timeout=UPDATE_TIMEOUT, interval=1.0, what="the hand-off result")
        handoff = _read(log)[log0:] + "\n--- update.log (this run) ---\n" + _read(ulog)[ulog0:]
        after = None
        with contextlib.suppress(AssertionError):
            after = H.wait_for(lambda: _serving(w, target), timeout=60, interval=1.0, what="a gateway on the release")
        gws = _gateways(host)
        diag = _diag(w, host, cp, timeline="\n" + tl.text(), result=res, after=after, gateways=gws,
                     handoff_log=handoff[-6000:])
        assert w.head() == target, f"premise: the hand-off did not land the release:\n{diag}"
        assert res.get("ok") is True, f"the hand-off result is not a success:\n{diag}"
        assert PAUSED in handoff and RESTARTED in handoff, f"the hand-off update did not pause/restart:\n{diag}"
        assert tl.first(lambda alive, head: alive and head == target) is None, \
            f"the OLD gateway process was alive while HEAD named the release:\n{diag}"
        assert after and int(after["pid"]) != old_pid and len(gws) == 1, \
            f"not exactly one NEW gateway on the release after the hand-off:\n{diag}"


def test_failed_move_restarts_exactly_the_paused_set_on_the_old_code(w):
    w.reset_clean()
    blocked = w.checkout / "website"
    assert blocked.is_dir(), "premise: website/ is tracked"
    target = w.publish("release: e2e hostile pause i", {"website/e2e-hostile-pause-i.txt": "release i\n"})

    def failing_update(host):
        blocked.chmod(0o555)
        try:
            return _ns_cli(host, w, "update", "--yes", "--branch", "main", timeout=UPDATE_TIMEOUT)
        finally:
            blocked.chmod(0o755)

    with _namespace(w) as host:
        before = _start_gateway(host, w, "i")
        old_sha, old_pid = w.head(), int(before["pid"])
        cp = failing_update(host)
        out = G.output(cp)
        receipt = w.receipt()
        after = None
        with contextlib.suppress(AssertionError):
            after = H.wait_for(lambda: _serving(w, old_sha), timeout=60, interval=1.0, what="a gateway on the old code")
        gws = _gateways(host)
        diag = _diag(w, host, cp, before=before, after=after, gateways=gws, receipt=receipt)
        assert w.head() == old_sha != target, f"premise: the move did not fail before the commit:\n{diag}"
        assert cp.returncode != 0, f"premise: the failed move reported success:\n{diag}"
        assert PAUSED in out, f"premise: nothing was paused before the move:\n{diag}"
        assert after and int(after["pid"]) != old_pid and len(gws) == 1, \
            f"not exactly one gateway restarted on the OLD code:\n{diag}"
        assert not [r for r in _record(w) if (r.get("token") or {}).get("resume_needed")], \
            f"the restarted set is still owed:\n{diag}"
        rc_with = cp.returncode
        host.kill(int(after["pid"]), signal.SIGTERM)
        H.wait_for(lambda: not _gateways(host), timeout=120, interval=0.5, what="the gateway to exit")
        (w.sb.hermes_home / "gateway.pid").unlink(missing_ok=True)
        ctrl = failing_update(host)
        ctrl_receipt = w.receipt()
        diag = _diag(w, host, ctrl, with_gateway_rc=rc_with, with_gateway_receipt=receipt)
        assert w.head() == old_sha, f"premise: the control run moved the checkout:\n{diag}"
        assert ctrl.returncode == rc_with, f"the paused set's restart changed the exit code:\n{diag}"
        assert ctrl_receipt.get("outcome") == receipt.get("outcome"), \
            f"the paused set's restart changed the receipt outcome:\n{diag}"
        assert PAUSED not in G.output(ctrl), f"the control run paused something:\n{diag}"
