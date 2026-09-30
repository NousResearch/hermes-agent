"""Process-wide machinery for the standalone CLI (run.py) only: environment retargeting for --home
and friends, and the --all-users pass with its read guard. Both patch process globals (os.environ,
builtins.open, sqlite3.connect, subprocess.Popen), so an in-process caller such as a Hermes backend
must never import this module; it calls runner.run(), which touches none of them."""
from __future__ import annotations

import builtins
import contextlib
import io
import os
import time
import uuid

from .host import OVERRIDABLE, HostAccess, collect_l0, per_user_overrides
from .registry import REGISTRY
from .runner import _truthy

USER_FAMILIES = ("apps", "ai_agents", "dev", "browser", "comms_work", "files", "gaming", "media")


class AggregateHostAccess(HostAccess):
    """HostAccess for an --all-users pass: stat/scandir/registry only. SQLite, file copies and
    subprocesses are refused so a probe cannot pull contents out of another account's home."""

    def sqlite(self, path, query, params=(), timeout_ms=1500):
        return None

    def copy_locked(self, path, dst_name=None, allow_vss=False):
        return None, "blocked"

    def run(self, args, timeout_ms=5000, text=True, env=None):
        return None

    def powershell(self, script, timeout_ms=8000):
        return None


class _ReadGuard:
    def __init__(self, roots):
        self.roots = [os.path.normcase(os.path.realpath(r)).rstrip("\\/") for r in roots if r]
        self.blocked = 0

    def denies(self, file) -> bool:
        if isinstance(file, int):
            return False
        try:
            p = os.path.normcase(os.path.realpath(os.fspath(file)))
        except (TypeError, ValueError, OSError):
            return False
        for r in self.roots:
            if p == r or p.startswith(r + os.sep):
                self.blocked += 1
                return True
        return False


@contextlib.contextmanager
def deny_reads_under(roots):
    """While active: open(), sqlite3.connect() and subprocess spawns fail with PermissionError for any
    path under `roots` (spawns are refused outright). stat/scandir still work, so presence probes run."""
    import sqlite3 as _sq
    import subprocess as _sp
    guard = _ReadGuard(roots)
    real_open, real_io_open, real_connect, real_popen = builtins.open, io.open, _sq.connect, _sp.Popen

    def g_open(file, *a, **kw):
        if guard.denies(file):
            raise PermissionError(f"all-users pass: content read refused: {file}")
        return real_open(file, *a, **kw)

    def g_connect(database, *a, **kw):
        target = str(database)
        if target.startswith("file:"):
            target = target[5:].split("?", 1)[0]
        if guard.denies(target):
            raise PermissionError("all-users pass: sqlite refused")
        return real_connect(database, *a, **kw)

    class g_popen(real_popen):
        def __init__(self, *a, **kw):
            self._child_created = False
            guard.blocked += 1
            raise PermissionError("all-users pass: subprocess refused")

    builtins.open = io.open = g_open
    _sq.connect = g_connect
    _sp.Popen = g_popen
    try:
        yield guard
    finally:
        builtins.open, io.open, _sq.connect, _sp.Popen = real_open, real_io_open, real_connect, real_popen


@contextlib.contextmanager
def override_env(l0: dict):
    """Point HOME/USERPROFILE/LOCALAPPDATA/APPDATA/HERMES_HOME at the l0 values named in
    l0["overridden"] for the duration of a run, so probes that call os.path.expanduser or read
    os.environ directly follow the override. Empty value = variable unset. Restored on exit."""
    saved = {}
    try:
        for key in l0.get("overridden", ()):
            for name in OVERRIDABLE.get(key, ()):
                saved[name] = os.environ.get(name)
                v = l0.get(key) or ""
                if v:
                    os.environ[name] = v
                else:
                    os.environ.pop(name, None)
        yield
    finally:
        for name, v in saved.items():
            if v is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = v


def scan_accounts(os_override: str = None, include_operator: bool = False) -> dict:
    """--all-users: one presence-only pass per login account. Per account it writes the username,
    home_present, process_count and one boolean per user-level L1 detector. Nothing else from the
    pass leaves this function: no probe values, no file names, no contents."""
    t0 = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]
    base = collect_l0(run_id, os_override=os_override)
    h = HostAccess(base)
    users = h.list_users(include_operator=include_operator)
    procs = h.process_counts()
    homes = [u["home"] for u in users if u["home"]]
    me_home = os.path.normcase(os.path.realpath(base["home"]))
    detectors = sorted((p for p in REGISTRY.values()
                        if p.fn and p.level == "L1" and p.family in USER_FAMILIES
                        and p.os in ("any", base["os"])), key=lambda q: q.id)
    accounts, touched = [], []
    for u in users:
        s = time.perf_counter()
        home = u["home"]
        present = bool(home) and os.path.isdir(home)
        readable = present and os.access(home, os.R_OK | os.X_OK)
        is_self = present and os.path.normcase(os.path.realpath(home)) == me_home
        rec = {"user": u["user"], "home_present": present, "home_readable": readable,
               "is_invoking_user": is_self,
               "process_count": (procs.get(u["user"], 0) if procs is not None else None),
               "apps": None, "status": "ok", "detectors_run": 0, "errors": 0, "refused": 0}
        if not present:
            rec["status"] = "home_absent"
        elif not readable:
            rec["status"] = "home_unreadable"
        else:
            touched.append(u["user"])
            l0u = collect_l0(run_id, os_override=os_override, overrides=per_user_overrides(base["os"], home))
            l0u["user"] = u["user"]
            l0u["foreign_user"] = not is_self
            hu = AggregateHostAccess(l0u)
            apps, facts = {}, {}
            with override_env(l0u), deny_reads_under(homes) as guard:
                for p in detectors:
                    if p.gate is not None and not _truthy(facts.get(p.gate)):
                        apps[p.id] = False
                        continue
                    rec["detectors_run"] += 1
                    try:
                        v = p.fn(hu, facts)
                    except Exception:
                        rec["errors"] += 1
                        v = None
                    if _truthy(v):
                        facts[p.id] = v
                    apps[p.id] = _truthy(v)
            hu.cleanup()
            rec["apps"] = apps
            rec["refused"] = guard.blocked
        rec["ms"] = round((time.perf_counter() - s) * 1000, 1)
        accounts.append(rec)
    return {"os": base["os"], "os_detected": base["os_detected"], "families": list(USER_FAMILIES),
            "detectors": len(detectors),
            "process_counts_visible": procs is not None,
            "processes_other_accounts": (sum(n for u, n in procs.items() if u not in {a["user"] for a in users})
                                         if procs is not None else None),
            "accounts": accounts, "touched": touched,
            "ms": round((time.perf_counter() - t0) * 1000, 1)}
