"""The runner: gates -> L1 detectors -> L2 extractors (batched PS + worker threads) -> L3 insights."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid

from . import __version__
from .host import HostAccess, collect_l0
from .registry import REGISTRY, INSIGHTS, TIERS


def _truthy(v):
    if v is None:
        return False
    if isinstance(v, dict):
        return bool(v.get("present", True))
    return bool(v)


def _filter_values(out: dict, max_tier: str) -> None:
    """Drop values above max_tier from the output. A T3 probe is allowed to run (it returns
    presence only); what the filter removes is anything content-shaped."""
    cap = TIERS.index(max_tier)
    # `host` is L0 and carries no tier of its own, but it holds the account name and absolute
    # home paths. Those are identity (contract rule 3) and must not survive at any tier.
    for k in ("user", "home", "localappdata", "appdata", "hermes_home", "run_id"):
        out["host"].pop(k, None)
    for fid, f in out["facts"].items():
        v = f.get("value")
        if v is None:
            continue
        t = TIERS.index(f.get("tier", "T0"))
        if t <= cap:
            continue
        if isinstance(v, dict):
            for k in list(v):
                if k not in ("present", "bytes", "count", "files", "dirs", "mtime", "size"):
                    v.pop(k)
        elif isinstance(v, list):
            f["value"] = v[:5]


def run(max_tier: str = "T1", budget_ms: int = 4000, allow_vss: bool = False,
        only: list = None, skip_family: list = None, include_deep: bool = False,
        os_override: str = None, overrides: dict = None, child_env: dict = None) -> dict:
    """One collection pass. `os_override` pretends to be another OS for probe selection;
    `overrides` retargets home/localappdata/appdata/hermes_home as l0 data only (see host.collect_l0);
    probes that read os.environ directly follow them only under cli.override_env, which the CLI applies;
    `child_env` is the environment every spawned child gets (default: this process's)."""
    run_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    l0 = collect_l0(run_id, allow_vss=allow_vss, os_override=os_override, overrides=overrides)
    return _run(l0, t0, max_tier, budget_ms, allow_vss, only, skip_family, include_deep, child_env)


def _run(l0, t0, max_tier, budget_ms, allow_vss, only, skip_family, include_deep, child_env):
    ps = _Pass(l0, HostAccess(l0, child_env), max_tier, only, skip_family, include_deep)
    ps.run_l1()
    ps.run_l2(budget_ms, allow_vss)
    insights, no_input = ps.run_l3()
    ps.h.cleanup()
    out = {
        "schema": "user-insights/1",
        "run": {"id": l0["run_id"], "max_tier": max_tier, "budget_ms": budget_ms,
                "allow_vss": allow_vss, "collector_version": __version__},
        "host": l0,
        "timing": ps.timing,
        "facts": ps.facts,
        "insights": insights,
        "coverage": {
            "detectors_total": sum(1 for p in REGISTRY.values() if ps.selected(p)),
            "fired": sum(1 for v in ps.facts.values() if v["status"] == "ok"),
            "extractors_run": sum(1 for v in ps.facts.values() if v["level"] == "L2" and v["status"] == "ok"),
            "skipped": ps.skipped,
            "errors": ps.errors,
            "insights_no_input": no_input,
        },
    }
    out["run"]["wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    _filter_values(out, max_tier)
    return out


class _Pass:
    """State of one collection pass; each run_* method is one layer."""

    def __init__(self, l0, h, max_tier, only, skip_family, include_deep):
        self.l0, self.h, self.max_tier, self.include_deep = l0, h, max_tier, include_deep
        self.skip_family = set(skip_family or [])
        self.only = set(only) if only else None
        self.facts: dict = {}
        self.timing = {}
        self.skipped: dict = {}
        self.errors: dict = {}

    def selected(self, p):
        if p.os not in ("any", self.l0["os"]):
            return False
        if p.family in self.skip_family:
            return False
        if p.collect == "deep" and not self.include_deep:
            return False
        if self.only is not None and p.id not in self.only and p.family not in self.only:
            return False
        if p.needs_admin and not self.l0["admin"]:
            return False
        return True

    def record(self, p, status, ms, value=None, via=None):
        self.facts[p.id] = {"level": p.level, "tier": p.tier, "family": p.family,
                            "status": status, "ms": round(ms, 3),
                            "method": (p.ps and "powershell") or (p.fn.__name__ if p.fn else "ps"),
                            "via": via, "value": value if status == "ok" else None}

    def gate_ok(self, p):
        return p.gate is None or _truthy(self.facts.get(p.gate, {}).get("value"))

    def run_l1(self):
        """L1 detectors, serially."""
        t = time.perf_counter()
        for p in sorted((q for q in REGISTRY.values() if self.selected(q) and q.level == "L1"), key=lambda q: q.id):
            if not self.gate_ok(p):
                self.record(p, "skipped_flag", 0)
                continue
            s = time.perf_counter()
            try:
                v = p.fn(self.h, self.facts) if p.fn else None
                self.record(p, "ok" if _truthy(v) else "absent", (time.perf_counter() - s) * 1000,
                            v if _truthy(v) else {"present": False})
            except Exception as e:
                self.errors[p.id] = f"{type(e).__name__}: {e}"
                self.record(p, "error", (time.perf_counter() - s) * 1000)
        self.timing["l1_ms"] = round((time.perf_counter() - t) * 1000, 1)

    def run_l2(self, budget_ms, allow_vss):
        """L2 extractors: batched PowerShell on a sidecar thread in parallel with the python probes."""
        ps_probes = [p for p in REGISTRY.values() if self.selected(p) and p.level == "L2" and p.ps]
        if ps_probes and self.l0.get("os_detected") != "windows":
            for p in ps_probes:
                self.record(p, "skipped_os", 0)
                self.skipped[p.id] = "skipped_os"
            ps_probes = []
        ps_holder = {}
        ps_start = time.perf_counter()

        def _ps_worker():
            ps_holder["results"] = _run_ps_batch(ps_probes, self.h, self.gate_ok, self.max_tier, allow_vss,
                                                 self.skipped)

        ps_thread = threading.Thread(target=_ps_worker, daemon=True) if ps_probes else None
        if ps_thread:
            ps_thread.start()

        t = time.perf_counter()
        deadline = self._run_python_l2(budget_ms)
        if ps_thread:
            ps_thread.join(timeout=max(1.0, deadline - time.perf_counter() + 2.0))
            for p, status, ms, value in ps_holder.get("results", []):
                self.record(p, status, ms, value, via="ps")
            for p in ps_probes:
                if p.id not in self.facts:
                    self.record(p, "timeout", 0)
        self.timing["ps_sidecar_ms"] = round((time.perf_counter() - ps_start) * 1000, 1)
        self.timing["l2_ms"] = round((time.perf_counter() - t) * 1000, 1)

    def _run_python_l2(self, budget_ms):
        """Python L2 probes on worker threads. Returns the pass deadline."""
        h, facts, errors = self.h, self.facts, self.errors
        py_probes = [p for p in REGISTRY.values() if self.selected(p) and p.level == "L2" and not p.ps]
        t_now = time.perf_counter()
        deadline = t_now + budget_ms / 1000.0
        lock = threading.Lock()
        queue = [p for p in py_probes if self.gate_ok(p)]
        order = sorted(queue, key=lambda q: (q.collect != "core", q.collect == "deep", q.id))

        # Each probe gets its own deadline: its own timeout_ms, or the remaining budget, whichever
        # is larger for a deep probe (a 25 s profile walk must not be starved by a 4 s budget) and
        # whichever is smaller otherwise. Nothing is force-killed; a probe past its deadline is
        # recorded as "timeout" and the runner moves on.
        def _probe_deadline(p):
            own = p.timeout_ms / 1000.0
            remaining = max(0.0, deadline - time.perf_counter())
            return time.perf_counter() + (max(own, remaining) if p.collect == "deep" else min(own, remaining))

        join_deadline = max([_probe_deadline(p) for p in order] or [deadline])
        idx = [0]

        def worker():
            while True:
                with lock:
                    i = idx[0]
                    if i >= len(order):
                        return
                    idx[0] = i + 1
                    p = order[i]
                if time.perf_counter() > deadline and p.collect != "deep":
                    self.record(p, "skipped_budget", 0)
                    continue
                s = time.perf_counter()
                try:
                    v = p.fn(h, facts)
                    self.record(p, "ok" if _truthy(v) else "absent", (time.perf_counter() - s) * 1000,
                                v if _truthy(v) else {"present": False},
                                via=h.last_via if hasattr(h, "last_via") else "live")
                except Exception as e:
                    with lock:
                        errors[p.id] = f"{type(e).__name__}: {e}"
                    self.record(p, "error", (time.perf_counter() - s) * 1000)

        workers = [threading.Thread(target=worker, daemon=True) for _ in range(min(8, (os.cpu_count() or 4)))]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=max(0.5, join_deadline - time.perf_counter() + 1.0))
        for p in order:
            if p.id not in facts:
                self.record(p, "timeout", 0)
        return deadline

    def run_l3(self):
        """L3 insights over the facts the reader will see. Returns (insights, ids with no input)."""
        t = time.perf_counter()
        cap = TIERS.index(self.max_tier)
        # Rules see only the facts the reader will see. A rule fed a hidden T2 fact would echo it in
        # its claim, which is exactly how persona.comms_surface leaked T2 detail into a T1 run.
        ok_facts = {k: v["value"] for k, v in self.facts.items()
                    if v["status"] == "ok" and TIERS.index(v.get("tier", "T0")) <= cap}
        # An input is answered when its probe was selected on this OS and finished (ok, absent, or gated off).
        # Rules that return a claim from absence ("no notes app") need at least one answered input; otherwise
        # the claim would rest on a probe that does not exist for this OS or that errored/timed out.
        selected_ids = {p.id for p in REGISTRY.values() if self.selected(p)}
        unanswered = {k for k, v in self.facts.items()
                      if v["status"] in ("error", "timeout", "skipped_budget", "skipped_os")}
        no_input = []
        insights = []
        for ins in INSIGHTS.values():
            if self.only is not None and ins.id not in self.only:
                continue
            if not any(i in selected_ids and i not in unanswered for i in ins.inputs):
                no_input.append(ins.id)
                continue
            try:
                got = ins.fn(ok_facts)
            except Exception as e:
                self.errors[ins.id] = f"{type(e).__name__}: {e}"
                continue
            if not got:
                continue
            missing = [i for i in ins.inputs if i not in ok_facts]
            insights.append({
                "id": ins.id,
                "claim": got.get("claim", ""),
                "strength": got.get("strength", "weak"),
                "confidence": "partial" if missing else "full",
                "evidence": [i for i in ins.inputs if i in ok_facts],
                "value": got.get("value"),
            })
        self.timing["l3_ms"] = round((time.perf_counter() - t) * 1000, 1)
        return insights, no_input


def _run_ps_batch(probes, h, gate_ok, max_tier, allow_vss, skipped):
    """One powershell.exe for every ps probe. Returns [(probe, status, ms, value)]."""
    results = []
    todo = []
    for p in probes:
        if not gate_ok(p):
            results.append((p, "skipped_flag", 0, None))
        else:
            todo.append(p)
    if not todo:
        return results
    blocks = []
    for p in todo:
        blocks.append(
            f"$ErrorActionPreference='Continue'; $sw=[Diagnostics.Stopwatch]::StartNew(); "
            f"try {{ $__v = & {{ {p.ps} }} | ConvertTo-Json -Compress -Depth 4 }} catch {{ $__v = $null }}; "
            f"[PSCustomObject]@{{ id = '{p.id}'; ms = $sw.Elapsed.TotalMilliseconds; v = $__v }}\n")
    # PS 5.1 rejects a line that starts with '|', so the blocks are wrapped in one scriptblock.
    # The script goes through a scratch file: -Command hits the 32 KB command-line limit.
    script = "$ProgressPreference='SilentlyContinue';\n& {\n" + "".join(blocks) + "} | ConvertTo-Json -Compress\n"
    try:
        path = os.path.join(h.scratch(), "ps_batch.ps1")
        with open(path, "w", encoding="utf-8-sig") as fh:
            fh.write(script)
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", path],
                           capture_output=True, timeout=120, stdin=subprocess.DEVNULL, env=h.child_env(),
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        raw = (r.stdout or b"").decode("utf-8", "replace")
        data = json.loads(raw) if raw.strip() else []
    except Exception:
        data = []
    if isinstance(data, dict):
        data = [data]
    by_id = {}
    for d in data:
        if isinstance(d, dict) and "id" in d:
            v = d.get("v")
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except Exception:
                    v = {"raw": v[:2000]}
            by_id[d["id"]] = (d.get("ms", 0.0), v)
    for p in todo:
        if p.id in by_id:
            ms, v = by_id[p.id]
            results.append((p, "ok" if _truthy(v) else "absent", ms, v if _truthy(v) else {"present": False}))
        else:
            results.append((p, "error", 0.0, None))
    return results
