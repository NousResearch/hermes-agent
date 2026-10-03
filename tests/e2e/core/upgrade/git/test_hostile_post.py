"""Once ``hermes update`` has committed the new code, the update never fails (contract C3).

One real HEAD install (``_git_world``: the real ``scripts/install.sh`` over real smart HTTP, every
command in a ``bwrap`` sandbox) is shared by the cells. Each cell publishes a hostile upstream
release and breaks something that runs AFTER the checkout moved, for real:

(a) the web UI build fails: the release's own ``scripts/build/web.mjs`` exits 1 while
    ``$HOME/.e2e-break-web-build`` exists, and the release changes a web build input (``web.mjs``
    itself lives under ``scripts/build``, plus a new file under ``web/``) so ``freshness.mjs`` reports
    the web UI stale and the updater must build it. A config migration is owed too
    (``_config_version`` lowered by one): one failed step must not skip later independent ones.
(b) a gateway that refuses to restart: a credential-free, platform-less (cron-only) ``hermes
    gateway run`` lives in the same PID namespace as the updater (``handoff._nshost``). The release
    makes ``hermes_cli.gateway.run_gateway`` exit 1 before booting while
    ``$HOME/.e2e-break-gateway-boot`` exists. The updater drains the OLD gateway (pre-update code,
    no fault) and its relaunch boots the NEW code, which refuses to start: the restart fails for real,
    no gateway serves the new commit. Removing the toggle and updating again must retry the owed
    restart.
(c) SIGKILL during the post-commit tail: the release's ``web.mjs`` writes
    ``$HOME/.e2e-web-build-started`` and sleeps 120 s while ``$HOME/.e2e-slow-web-build`` exists. The
    update runs in the shared namespace; once the slow build started, every process born after the
    update was spawned is SIGSTOPped then SIGKILLed (that is the whole update tree, including a
    completion child in its own session or reparented to the namespace init).
(d) a sticky profile (``hermes profile use e2epost``) must not move update.log / update receipts off
    the root hermes home, and ``hermes logs update`` under that profile reads the root's update.log;
    pm's own sync receipts stay where pm writes them and ``hermes pm status`` there still shows them.
(e) the dependency sync fails after the tree moved (contract amendment A6): the release's
    ``pm.client.ensure_tools_for_sync`` raises while ``$HOME/.e2e-break-deps-sync`` exists; it runs in
    the completion bootstrap from the NEW tree, after the checkout moved. Exit 0, a ``dependencies``
    follow-up, the tail obligation armed; the next update completes it.
(f) Ctrl-C after the commit point: SIGINT (what a terminal's Ctrl-C delivers to the foreground
    ``hermes update``; the completion child runs in its own session) while the slow web build runs.
    The user must read that the new code is in place and the rest is owed, never a failure.

What is asserted is what the user sees: the exit code, ``⚠`` lines, the receipt at
``<root>/logs/update_receipts/latest.json`` (``outcome``, ``followups``), the
``source-completion-pending`` obligation in the install-state dir, the config version, the
owed-restart warning a later command prints, and which gateway (by its control-socket ``identify``)
serves the checkout.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import signal
import tempfile
import time
from pathlib import Path
from typing import Iterator

import pytest

import hermes_yaml as yaml
from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.git import _git_world as G
from tests.e2e.core.upgrade.handoff._nshost import NamespaceHost

pytestmark = G.PYTESTMARK

WARN = "⚠"
BREAK_WEB = ".e2e-break-web-build"
SLOW_WEB = ".e2e-slow-web-build"
WEB_STARTED = ".e2e-web-build-started"
BREAK_GATEWAY = ".e2e-break-gateway-boot"
BREAK_DEPS = ".e2e-break-deps-sync"
TOGGLES = (BREAK_WEB, SLOW_WEB, WEB_STARTED, BREAK_GATEWAY, BREAK_DEPS)
DEPS_FAULT_TEXT = "e2e: dependency sync fails on purpose"
DEPS_OWED = "dependencies not installed yet"  # update_completion._settle_after_commit
INTERRUPTED_AFTER_COMMIT = "Interrupted after the code was updated"  # update_receipt
WEB_FAULT_TEXT = "e2e: the web build fails on purpose"
GATEWAY_FAULT_TEXT = "e2e: gateway boot refused on purpose"
OWED_RESTART = "did not restart running gateways"  # update_cmd_fleet._warn_pending_fleet_restart
PROFILE = "e2epost"
UPDATE_TIMEOUT = 1500
_WEB_MARK = "// e2e (test_hostile_post): hostile build toggles"
_GATEWAY_MARK = "# e2e (test_hostile_post): hostile gateway boot toggle"
_DEPS_MARK = "# e2e (test_hostile_post): hostile dependency sync toggle"

_HOSTILE_WEB = f"""{_WEB_MARK}, keyed on files in $HOME.
import {{ existsSync as e2eExists, writeFileSync as e2eWrite }} from 'node:fs'
import {{ homedir as e2eHomedir }} from 'node:os'
const e2eHomes = [...new Set([process.env.HOME, e2eHomedir()].filter(Boolean))]
const e2eToggle = name => e2eHomes.some(home => e2eExists(`${{home}}/${{name}}`))
if (e2eToggle('{BREAK_WEB}')) {{
  console.error('{WEB_FAULT_TEXT} ({BREAK_WEB})')
  process.exit(1)
}}
if (e2eToggle('{SLOW_WEB}')) {{
  for (const home of e2eHomes) {{ try {{ e2eWrite(`${{home}}/{WEB_STARTED}`, String(process.pid)) }} catch {{}} }}
  console.log('e2e: slow web build started ({SLOW_WEB})')
  await new Promise(resolve => setTimeout(resolve, 120000))
}}
"""

_HOSTILE_GATEWAY = f"""    {_GATEWAY_MARK}
    if __import__("os").path.exists(__import__("os").path.join(__import__("os").path.expanduser("~"), "{BREAK_GATEWAY}")):
        print("{GATEWAY_FAULT_TEXT} ({BREAK_GATEWAY})", file=__import__("sys").stderr, flush=True)
        raise SystemExit(1)
"""

_HOSTILE_DEPS = f"""    {_DEPS_MARK}
    if __import__("os").path.exists(__import__("os").path.join(__import__("os").path.expanduser("~"), "{BREAK_DEPS}")):
        raise RuntimeError("{DEPS_FAULT_TEXT} ({BREAK_DEPS})")
"""


# -- the shared install --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def w() -> Iterator[G.World]:
    """A SHORT root (the gateway's AF_UNIX control socket lives under ``$HERMES_HOME``; pytest's deep
    ``tmp_path`` overflows ``sun_path``). Set ``HERMES_E2E_KEEP_ROOT=1`` to keep it for a post-mortem."""
    base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    root = Path(tempfile.mkdtemp(prefix="hp", dir=base))
    try:
        with G.world(root, base=I.head_sha()) as world:
            _edit_config(world, lambda cfg: cfg.update(updates={**(cfg.get("updates") or {}), "check": False}))
            yield world
    finally:
        if not os.environ.get("HERMES_E2E_KEEP_ROOT"):
            shutil.rmtree(root, ignore_errors=True)


def _config_path(w: G.World) -> Path:
    return w.sb.hermes_home / "config.yaml"


def _load_config(w: G.World) -> dict:
    p = _config_path(w)
    data = yaml.safe_load(p.read_text(encoding="utf-8-sig")) if p.is_file() else None
    return data if isinstance(data, dict) else {}


def _edit_config(w: G.World, edit) -> None:
    cfg = _load_config(w)
    edit(cfg)
    _config_path(w).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _config_version(w: G.World) -> int | None:
    v = _load_config(w).get("_config_version")
    return int(v) if isinstance(v, int) or (isinstance(v, str) and v.isdigit()) else None


def _toggle(w: G.World, name: str, on: bool) -> None:
    p = w.sb.home / name
    if on:
        p.write_text("on\n", encoding="utf-8")
    else:
        p.unlink(missing_ok=True)


def _clear_toggles(w: G.World) -> None:
    for name in TOGGLES:
        _toggle(w, name, False)


def _pending_markers(w: G.World) -> list[str]:
    """The ``source-completion-pending`` obligation(s) in the install-state dir(s)."""
    installs = w.sb.hermes_home / "installs"
    return sorted(str(p) for p in installs.glob("**/source-completion-pending")) if installs.is_dir() else []


def _followups(rec: dict) -> list[dict]:
    items = rec.get("followups") or []
    return [f for f in items if isinstance(f, dict)]


def _followup_steps(rec: dict) -> set[str]:
    return {str(f.get("step")) for f in _followups(rec)}


def _update_log(w: G.World) -> Path:
    return w.sb.hermes_home / "logs" / "update.log"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig", errors="replace") if p.is_file() else ""


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _warn_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if WARN in line]


def _obligation(w: G.World) -> str:
    """The host fleet-restart obligation record(s) anywhere in the sandbox (diagnostics only)."""
    found = []
    for base, dirs, files in os.walk(w.sb.root):
        dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "venv", ".venv", "hermes-agent")]
        if "host-update-restart.json" in files:
            path = Path(base) / "host-update-restart.json"
            found.append(f"{path}: {_read(path)[:600]}")
    return "; ".join(found) or "none"


def _facts(w: G.World, cp=None, **extra) -> str:
    """A one-glance summary heading every failure message."""
    rec = w.receipt()
    parts = [f"rc={cp.returncode}" if cp is not None else "rc=-",
             f"head={w.head()[:12]}",
             f"receipt.outcome={rec.get('outcome')!r} update_id={str(rec.get('update_id'))[:12]} "
             f"pid={rec.get('pid')} followups={_followups(rec)}",
             f"pending={_pending_markers(w)}",
             f"config_version={_config_version(w)}",
             f"obligation={_obligation(w)}"]
    if cp is not None:
        parts.append(f"⚠ lines={_warn_lines(G.output(cp))[:8]}")
    parts += [f"{k}={v}" for k, v in extra.items()]
    return "\n".join(parts) + "\n"


# -- hostile releases ----------------------------------------------------------------------------


def _head_file(w: G.World, rel: str) -> str:
    return w.git("show", f"HEAD:{rel}") + "\n"


def _hostile_web_mjs(w: G.World) -> str:
    """The checkout's ``scripts/build/web.mjs`` with the toggle block after its imports."""
    text = _head_file(w, "scripts/build/web.mjs")
    if _WEB_MARK in text:
        return text
    lines = text.splitlines(keepends=True)
    last_import = max(i for i, line in enumerate(lines) if line.startswith("import "))
    while " from " not in lines[last_import] and last_import + 1 < len(lines):  # a multi-line import
        last_import += 1
    out = "".join(lines[:last_import + 1]) + _HOSTILE_WEB + "".join(lines[last_import + 1:])
    assert out.count(_WEB_MARK) == 1
    return out


def _publish_web_release(w: G.World, tag: str) -> str:
    """A release whose web build obeys the toggles and whose web inputs changed (web UI stale)."""
    return w.publish(f"release: e2e hostile post web {tag}", {
        "scripts/build/web.mjs": _hostile_web_mjs(w),
        f"web/e2e-hostile-post-{tag}.txt": f"web input changed by release {tag}\n",
    })


def _publish_gateway_fault_release(w: G.World, tag: str) -> str:
    """A release whose ``hermes gateway run`` refuses to boot while the toggle exists."""
    text = _head_file(w, "hermes_cli/gateway.py")
    if _GATEWAY_MARK not in text:
        m = re.search(r"^def run_gateway\(.*?\):\n    \"\"\".*?\"\"\"\n", text, re.S | re.M)
        assert m, "premise: hermes_cli/gateway.py has no run_gateway() with a docstring to inject after"
        text = text[:m.end()] + _HOSTILE_GATEWAY + text[m.end():]
    return w.publish(f"release: e2e hostile post gateway {tag}", {
        "hermes_cli/gateway.py": text,
        f"docs/e2e-hostile-post-{tag}.txt": f"release {tag}\n",
    })


def _publish_deps_fault_release(w: G.World, tag: str) -> str:
    """A release whose dependency preparation (``ensure_tools_for_sync``) fails while the toggle exists."""
    text = _head_file(w, "pm/client.py")
    if _DEPS_MARK not in text:
        m = re.search(r"^def ensure_tools_for_sync\(\) -> None:\n    \"\"\".*?\"\"\"\n", text, re.S | re.M)
        assert m, "premise: pm/client.py has no ensure_tools_for_sync() with a docstring to inject after"
        text = text[:m.end()] + _HOSTILE_DEPS + text[m.end():]
    return w.publish(f"release: e2e hostile post deps {tag}", {
        "pm/client.py": text,
        f"docs/e2e-hostile-post-{tag}.txt": f"release {tag}\n",
    })


# -- one shared PID namespace (gateway / killable update) ----------------------------------------


@contextlib.contextmanager
def _namespace(w: G.World) -> Iterator[NamespaceHost]:
    """Every process of the cell in ONE sandbox, as on a user's machine: the updater's process scans
    see the gateway it restarts. Closing kills the namespace init, which takes everything with it."""
    host = NamespaceHost(w.sb.root, w.sb.env)
    try:
        yield host
    finally:
        leaked = host.close()
        assert not leaked, f"processes outlived the sandbox: {leaked}"


def _ns_cli(host: NamespaceHost, w: G.World, *args: str, timeout: float = 600):
    cp = host.run([w.sb.hermes, *args], timeout=timeout)
    w.transcripts.append(f"$ hermes {' '.join(args)} -> rc={cp.returncode}\n{(cp.stdout or '')[-6000:]}\n"
                         f"{(cp.stderr or '')[-3000:]}")
    return cp


def _ns_update(host: NamespaceHost, w: G.World, *extra: str):
    return _ns_cli(host, w, "update", "--yes", "--branch", "main", *extra, timeout=UPDATE_TIMEOUT)


def _identify(w: G.World) -> dict | None:
    from gateway.control_socket import identify_gateway

    with contextlib.suppress(Exception):
        return identify_gateway(w.sb.hermes_home, timeout=5.0)
    return None


def _serving(w: G.World, sha: str) -> dict | None:
    ident = _identify(w)
    return ident if ident and ident.get("code_sha") == sha else None


def _kill_born_after(host: NamespaceHost, before: set[int]) -> list[int]:
    """SIGSTOP then SIGKILL every process born after ``before`` was taken (by PID, namespace view):
    the update, its re-exec'd interpreter, build children and any detached completion child (own
    session, or reparented to the namespace init). Repeats until none is left."""
    killed: list[int] = []
    for _ in range(10):
        live = [p["pid"] for p in host.procs() if p["pid"] not in before and p["state"] not in ("Z", "X")]
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


def _gateway_logs(w: G.World) -> str:
    logs = w.sb.hermes_home / "logs"
    chunks = [_read(p)[-4000:] for p in sorted(logs.glob("*.log"))] if logs.is_dir() else []
    chunks += [_read(p)[-4000:] for p in sorted(w.sb.root.glob("gateway-*.log"))]
    return "\n".join(chunks)


# -- cells ---------------------------------------------------------------------------------------


def test_failed_web_build_after_commit_is_a_followup_not_a_failed_update(w):
    w.reset_clean()
    _clear_toggles(w)
    latest = _config_version(w)
    if latest is None:
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        latest = int(DEFAULT_CONFIG["_config_version"])
    owed = latest - 1
    _edit_config(w, lambda cfg: cfg.update(_config_version=owed))
    target = _publish_web_release(w, "a")
    log0 = len(_read(_update_log(w)))

    _toggle(w, BREAK_WEB, True)
    try:
        cp = w.update("--no-gateway-restart")
    finally:
        _toggle(w, BREAK_WEB, False)
    out = G.output(cp)
    log = _read(_update_log(w))[log0:]
    rec = w.receipt()
    facts = _facts(w, cp, config_version_lowered_to=owed, latest_config_version=latest)
    diag = facts + w.diag(cp)

    assert w.head() == target, f"premise: the release never committed, so this is no post-commit cell:\n{diag}"
    assert WEB_FAULT_TEXT in out + log, f"premise: the hostile web build never ran (web UI not seen stale?):\n{diag}"
    assert cp.returncode == 0, f"the code committed, yet a failed web build failed `hermes update`:\n{diag}"
    assert _warn_lines(out), f"the failed web build was not surfaced with a ⚠ line:\n{diag}"
    assert rec.get("outcome") == "success", f"receipt outcome is not success after the commit:\n{diag}"
    assert "build" in _followup_steps(rec), f"receipt followups do not name the failed `build` step:\n{diag}"
    assert _pending_markers(w), f"the failed build disarmed the source-completion obligation:\n{diag}"
    assert (_config_version(w) or 0) >= latest, (
        f"the failed web build skipped the independent config migration (_config_version still "
        f"{_config_version(w)}, latest {latest}):\n{diag}")

    # The fault clears; the next plain update (no new release) pays the owed tail.
    cp = w.update()
    rec = w.receipt()
    diag = _facts(w, cp) + w.diag(cp)
    assert cp.returncode == 0 and w.head() == target, f"the healing update failed:\n{diag}"
    assert not _followups(rec), f"the healing update still reports followups:\n{diag}"
    assert not _pending_markers(w), f"the healing update left the completion obligation armed:\n{diag}"
    assert (w.checkout / "hermes_cli" / "web_dist" / "index.html").is_file(), \
        f"the healing update did not build the web UI:\n{diag}"


def test_gateway_that_cannot_reboot_is_an_owed_restart_not_a_failed_update(w):
    """Fault: the RELEASE's gateway cannot boot (``run_gateway`` exits 1 while the toggle exists). The
    old gateway drains/stops normally; the relaunch runs the new code and dies at boot, so no gateway
    serves the new commit. A SIGSTOPped old gateway would not do: the survivor sweep SIGKILLs it and
    the relaunch then succeeds."""
    w.reset_clean()
    _clear_toggles(w)
    with _namespace(w) as host:
        try:
            gw = host.spawn([w.sb.hermes, "gateway", "run"], log=w.sb.root / "gateway-b.log")  # sandbox-writable
            before_sha = w.head()
            try:
                ident = H.wait_for(lambda: _serving(w, before_sha), timeout=240, interval=1.0,
                                   what="the cron-only gateway's control-socket identify")
            except AssertionError as exc:
                raise AssertionError(f"premise: the gateway never came up: {exc}\n{_gateway_logs(w)[-4000:]}\n"
                                     f"{host.ps_text()}") from None
            target = _publish_gateway_fault_release(w, "b")
            _toggle(w, BREAK_GATEWAY, True)
            cp = _ns_update(host, w)
            out = G.output(cp)
            rec = w.receipt()
            obligation_after_update = _obligation(w)
            status = _ns_cli(host, w, "status", timeout=300)
            facts = _facts(w, cp, gateway_before=f"pid {ident.get('pid')} sha {before_sha[:12]} (spawned {gw})",
                           identify_after=_identify(w), obligation_after_update=obligation_after_update)
            diag = (facts + w.diag(cp) + "\n--- gateway logs ---\n" + _gateway_logs(w)[-6000:]
                    + "\n--- sandbox ---\n" + host.ps_text() + "\n--- hermes status ---\n" + G.output(status)[-3000:])

            assert w.head() == target, f"premise: the release never committed:\n{diag}"
            assert not _serving(w, target), f"premise: a gateway serves the new commit; the restart fault never fired:\n{diag}"
            assert cp.returncode == 0, f"the code committed, yet a failed gateway restart failed `hermes update`:\n{diag}"
            assert _warn_lines(out), f"the failed gateway restart was not surfaced with a ⚠ line:\n{diag}"
            assert rec.get("outcome") == "success", f"receipt outcome is not success after the commit:\n{diag}"
            assert "gateway_restart" in _followup_steps(rec), \
                f"receipt followups do not name the failed `gateway_restart` step:\n{diag}"
            assert OWED_RESTART in G.output(status), \
                f"a later command does not warn that a gateway restart is still owed:\n{diag}"

            # The fault clears; the next plain update retries the owed restart. This gateway is a
            # manual `gateway run` that died at boot: no supervisor and no live process, so the
            # retry has nothing it can relaunch. It must still exit 0 and must NOT silently discharge
            # the obligation (no gateway serves HEAD yet).
            _toggle(w, BREAK_GATEWAY, False)
            cp = _ns_update(host, w)
            status = _ns_cli(host, w, "status", timeout=300)
            diag = (_facts(w, cp, identify_after=_identify(w)) + w.diag(cp) + "\n--- gateway logs ---\n"
                    + _gateway_logs(w)[-6000:] + "\n--- sandbox ---\n" + host.ps_text()
                    + "\n--- hermes status ---\n" + G.output(status)[-3000:])
            assert cp.returncode == 0, f"the retrying update failed:\n{diag}"
            assert OWED_RESTART in G.output(status), \
                f"the retry discharged the owed restart while no gateway serves HEAD:\n{diag}"

            # The operator restarts it (the remedy the warning names); once a gateway serves HEAD the
            # obligation discharges and the warning goes away.
            host.spawn([w.sb.hermes, "gateway", "run"], log=w.sb.root / "gateway-b2.log")
            served = None
            with contextlib.suppress(AssertionError):
                served = H.wait_for(lambda: _serving(w, target), timeout=240, interval=1.0, what="a gateway on HEAD")
            status = _ns_cli(host, w, "status", timeout=300)
            diag = (_facts(w, cp, identify_after=_identify(w)) + "\n--- gateway logs ---\n"
                    + _gateway_logs(w)[-6000:] + "\n--- sandbox ---\n" + host.ps_text()
                    + "\n--- hermes status ---\n" + G.output(status)[-3000:])
            assert served, f"premise: the restarted gateway never served {target[:12]}:\n{diag}"
            assert OWED_RESTART not in G.output(status), \
                f"the owed-restart warning survived a gateway serving HEAD:\n{diag}"
        finally:
            _toggle(w, BREAK_GATEWAY, False)
            for p in host.procs():
                if p["pid"] > 2 and "gateway" in p["cmdline"]:
                    host.kill(p["pid"], signal.SIGKILL)  # windows-footgun: ok - linux-only (bwrap) module
    # The namespace (and its gateways) is gone; its PID file names a PID of a dead namespace.
    (w.sb.hermes_home / "gateway.pid").unlink(missing_ok=True)


def test_sigkill_during_post_commit_tail_leaves_a_running_receipt(w):
    w.reset_clean()
    _clear_toggles(w)
    prev = w.receipt()
    target = _publish_web_release(w, "c")
    log0 = len(_read(_update_log(w)))
    started = w.sb.home / WEB_STARTED
    _toggle(w, SLOW_WEB, True)
    killed: list[int] = []
    try:
        with _namespace(w) as host:
            before = {p["pid"] for p in host.procs()}
            spawn_log = w.sb.root / "update-c.log"  # sandbox-writable
            upd = host.spawn([w.sb.hermes, "update", "--yes", "--branch", "main", "--no-gateway-restart"],
                             log=spawn_log)

            def _slow_build_running():
                if started.is_file():
                    return True
                if not host.alive(upd):
                    raise AssertionError(f"premise: the update exited before the slow web build started:\n"
                                         f"{_read(spawn_log)[-6000:]}\n{w.diag()}")
                return False

            H.wait_for(_slow_build_running, timeout=1200, interval=1.0, what="the slow web build to start")
            with contextlib.suppress(AssertionError):
                H.wait_for(lambda: re.search(r"web UI|slow web build", _read(_update_log(w))[log0:]),
                           timeout=30, interval=0.5, what="update.log to show the web build")
            log_shows_build = bool(re.search(r"web UI|slow web build", _read(_update_log(w))[log0:]))
            killed = _kill_born_after(host, before)
            rec = w.receipt()
            facts = _facts(w, killed=killed, update_spawn_pid=upd, log_shows_web_build=log_shows_build,
                           previous_update_id=prev.get("update_id"))
            diag = facts + w.diag() + "\n--- spawned update output ---\n" + _read(spawn_log)[-6000:]
    finally:
        _toggle(w, SLOW_WEB, False)
        started.unlink(missing_ok=True)

    assert killed, f"premise: nothing was killed:\n{diag}"
    assert w.head() == target, f"premise: the kill landed before the release committed:\n{diag}"
    assert rec.get("outcome") == "running" and rec.get("update_id") and \
        rec.get("update_id") != prev.get("update_id"), \
        f"latest.json is not a `running` record of the killed update:\n{json.dumps(rec, default=str)[:3000]}\n{diag}"
    assert rec.get("pid") in killed, f"the running receipt names pid {rec.get('pid')}, not a killed process:\n{diag}"
    killed_id = rec["update_id"]

    cp = w.update("--no-gateway-restart")
    rec = w.receipt()
    diag = _facts(w, cp, killed_update_id=killed_id) + w.diag(cp)
    assert "interrupted" in G.output(cp), f"the next update did not report the interrupted run:\n{diag}"
    assert cp.returncode == 0, f"the update after the interrupted one failed:\n{diag}"
    assert rec.get("outcome") == "success" and rec.get("update_id") != killed_id, \
        f"latest.json is not this run's terminal success:\n{diag}"


def test_sticky_profile_update_logs_and_receipts_land_in_the_root_home(w):
    w.reset_clean()
    _clear_toggles(w)
    root = w.sb.hermes_home
    prof_receipts = root / "profiles" / PROFILE / "logs" / "update_receipts"
    if not (root / "profiles" / PROFILE).is_dir():
        cp = w.sb.cli("profile", "create", PROFILE, "--no-alias", timeout=300)
        assert cp.returncode == 0, f"premise: profile create failed:\n{H.describe(cp)}"
    cp = w.sb.cli("profile", "use", PROFILE, timeout=300)
    assert cp.returncode == 0, f"premise: profile use failed:\n{H.describe(cp)}"
    try:
        assert (root / "active_profile").read_text(encoding="utf-8-sig").strip() == PROFILE, \
            "premise: the sticky profile was not recorded"
        prof_before = {p.name for p in prof_receipts.iterdir()} if prof_receipts.is_dir() else set()
        log0 = _update_log(w).stat().st_size if _update_log(w).is_file() else 0
        prev = w.receipt()
        target = w.publish("release: e2e hostile post profile d", {"docs/e2e-hostile-post-d.txt": "d\n"})

        cp = w.update("--no-gateway-restart")

        prof_after = {p.name for p in prof_receipts.iterdir()} if prof_receipts.is_dir() else set()
        log1 = _update_log(w).stat().st_size if _update_log(w).is_file() else 0
        rec = w.receipt()
        diag = _facts(w, cp, profile_receipts_new=sorted(prof_after - prof_before), root_update_log_bytes=f"{log0}->{log1}",
                      previous_update_id=prev.get("update_id")) + w.diag(cp)
        assert w.head() == target, f"premise: the release never committed:\n{diag}"
        assert cp.returncode == 0, f"the sticky-profile update failed:\n{diag}"
        # The update's receipts never follow the sticky profile: the root dir is what the Desktop
        # and the hand-off scripts read. pm's own sync receipts stay where pm writes them (the
        # active home; pm is not the updater's), and pm's reader under the profile still finds them.
        assert not [n for n in prof_after - prof_before if n.startswith("update_")], \
            f"the update wrote update receipts into the profile home {prof_receipts}:\n{diag}"
        pm_cp = w.sb.cli("pm", "status", timeout=300)
        assert pm_cp.returncode == 0 and json.loads(pm_cp.stdout).get("kind") == "sync", \
            f"`hermes pm status` under the profile does not show pm's own receipt:\n{H.describe(pm_cp)}\n{diag}"
        assert log1 > log0, f"the root update.log did not grow:\n{diag}"
        assert rec.get("update_id") != prev.get("update_id") and rec.get("outcome") == "success", \
            f"root latest.json is not this run's success:\n{diag}"
        # The readers follow the writers: `hermes logs update` under the profile shows the root log.
        logs_cp = w.sb.cli("logs", "update", "-n", "400", timeout=300)
        assert logs_cp.returncode == 0 and "hermes update started" in G.output(logs_cp), \
            f"`hermes logs update` under the sticky profile does not show the root update.log:\n{H.describe(logs_cp)}\n{diag}"
    finally:
        w.sb.cli("profile", "use", "default", timeout=300)


def test_dependency_sync_failure_after_commit_is_a_followup_not_a_failed_update(w):
    w.reset_clean()
    _clear_toggles(w)
    target = _publish_deps_fault_release(w, "e")
    _toggle(w, BREAK_DEPS, True)
    try:
        cp = w.update("--no-gateway-restart")
    finally:
        _toggle(w, BREAK_DEPS, False)
    out = G.output(cp)
    rec = w.receipt()
    diag = _facts(w, cp) + w.diag(cp)

    assert w.head() == target, f"premise: the release never committed, so this is no post-commit cell:\n{diag}"
    assert DEPS_FAULT_TEXT in out, f"premise: the hostile dependency sync never ran:\n{diag}"
    assert cp.returncode == 0, f"the code committed, yet a failed dependency sync failed `hermes update`:\n{diag}"
    assert any(DEPS_OWED in line for line in _warn_lines(out)), \
        f"no ⚠ line says the dependencies are not installed yet:\n{diag}"
    assert rec.get("outcome") == "success", f"receipt outcome is not success after the commit:\n{diag}"
    assert "dependencies" in _followup_steps(rec), f"receipt followups do not name `dependencies`:\n{diag}"
    assert _pending_markers(w), f"the failed dependency sync disarmed the source-completion obligation:\n{diag}"

    # The fault clears; the next plain update completes the owed dependencies and tail.
    cp = w.update("--no-gateway-restart")
    rec = w.receipt()
    diag = _facts(w, cp) + w.diag(cp)
    assert cp.returncode == 0 and w.head() == target, f"the completing update failed:\n{diag}"
    assert not _followups(rec), f"the completing update still reports followups:\n{diag}"
    assert not _pending_markers(w), f"the completing update left the completion obligation armed:\n{diag}"


def _foreground_group(host: NamespaceHost, upd: int) -> list[int]:
    """What a terminal's Ctrl-C reaches: the update and its descendants, minus the completion child's
    subtree (it runs in its own session, so the terminal's SIGINT never reaches it)."""
    procs = {p["pid"]: p for p in host.procs() if p["state"] not in ("Z", "X")}
    group, frontier = [], [upd]
    while frontier:
        pid = frontier.pop()
        if pid not in procs or "update_completion.py" in " ".join(procs[pid]["cmdline"]):
            continue
        group.append(pid)
        frontier += [p for p, row in procs.items() if row["ppid"] == pid]
    return group


def test_ctrl_c_after_commit_reports_the_new_code_not_a_failed_update(w):
    w.reset_clean()
    _clear_toggles(w)
    prev = w.receipt()
    target = _publish_web_release(w, "f")
    started = w.sb.home / WEB_STARTED
    _toggle(w, SLOW_WEB, True)
    try:
        with _namespace(w) as host:
            spawn_log = w.sb.root / "update-f.log"  # sandbox-writable
            upd = host.spawn([w.sb.hermes, "update", "--yes", "--branch", "main", "--no-gateway-restart"],
                             log=spawn_log)

            def _slow_build_running():
                if started.is_file():
                    return True
                if not host.alive(upd):
                    raise AssertionError(f"premise: the update exited before the slow web build started:\n"
                                         f"{_read(spawn_log)[-6000:]}\n{w.diag()}")
                return False

            H.wait_for(_slow_build_running, timeout=1200, interval=1.0, what="the slow web build to start")
            interrupted = _foreground_group(host, upd)
            for pid in interrupted:
                host.kill(pid, signal.SIGINT)
            exited = True
            try:
                H.wait_for(lambda: not host.alive(upd), timeout=120, interval=0.5, what="the interrupted update to exit")
            except AssertionError:
                exited = False
            out = _read(spawn_log)
            rec = w.receipt()
            diag = (_facts(w, sigint=interrupted, update_exited=exited, previous_update_id=prev.get("update_id"))
                    + w.diag() + "\n--- sandbox ---\n" + host.ps_text()
                    + "\n--- interrupted update output ---\n" + out[-6000:])
    finally:
        _toggle(w, SLOW_WEB, False)
        started.unlink(missing_ok=True)

    assert w.head() == target, f"premise: the SIGINT landed before the release committed:\n{diag}"
    assert exited, f"the update did not exit after Ctrl-C:\n{diag}"
    assert INTERRUPTED_AFTER_COMMIT in out, \
        f"after Ctrl-C the user is not told the new code is in place and the rest is owed:\n{diag}"
    assert rec.get("update_id") != prev.get("update_id") and rec.get("outcome") == "interrupted", \
        f"latest.json is not this run's `interrupted` record (a `failed` one reads as the previous version):\n{diag}"
    assert _pending_markers(w), f"the interrupt disarmed the source-completion obligation:\n{diag}"

    cp = w.update("--no-gateway-restart")
    rec = w.receipt()
    diag = _facts(w, cp) + w.diag(cp)
    assert cp.returncode == 0 and rec.get("outcome") == "success", f"the update after Ctrl-C failed:\n{diag}"
    assert not _pending_markers(w), f"the update after Ctrl-C left the completion obligation armed:\n{diag}"
