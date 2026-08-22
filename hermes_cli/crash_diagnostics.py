"""Crash-cause diagnostics — parse *why* a Hermes process died, not just that it restarted.

Hermes already notifies on restart ("gateway restarting / back online"). This is the
complementary signal: when a process dies, read the OS's *own* crash record and surface a
normalized, human-readable **cause** + faulting backtrace — so an unclean shutdown becomes
``SIGABRT in libmlx (GPU memory pressure)`` instead of a silent restart.

A native crash (a GPU OOM, a segfault in a C extension, the OOM-killer) never produces a
Python traceback, so the app/Docker logs just show the process vanishing and coming back.
Parsing the OS-level crash record is the only reliable, cross-platform way to recover the
actual cause + faulting frames.

Sources, per platform (best-effort, fully wrapped — never raises into the caller):

* **macOS**   — ``~/Library/Logs/DiagnosticReports/*.ips`` (signal + triggered-thread backtrace)
* **Linux**   — ``systemd-coredump`` via ``coredumpctl`` (signal + stack); ``journalctl`` fallback
* **Windows** — Application event log / Windows Error Reporting via ``Get-WinEvent``

GPU-aware (Metal/MLX on Apple, CUDA/NVIDIA on Linux/DGX). One call::

    from hermes_cli.crash_diagnostics import recent_crashes, summarize
    for c in recent_crashes(name_filter="python", since_hours=24):
        print(summarize(c))

Each record is a plain dict: ``{os, when, process, signal, cause, backtrace, source}``.

Wired into the gateway's restart machinery: ``_send_home_channel_startup_notifications``
appends the cause to the "gateway is back online" message after an *unexpected* restart
(gated by the same ``gateway_restart_notification`` setting). Also surfaced on demand via
the "Recent Crashes" section of ``hermes doctor`` (see :func:`doctor_section`).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def os_kind() -> str:
    """Coarse platform bucket: ``macos`` / ``linux`` / ``windows`` / ``unknown``."""
    p = sys.platform
    if p == "darwin":
        return "macos"
    if p.startswith("linux"):
        return "linux"
    if p.startswith("win") or os.name == "nt":
        return "windows"
    return "unknown"


# ─── Gateway lifecycle (clean-exit tracking) ─────────────────────────────────
# So a crash is only ever reported when the gateway's OWN lifecycle says the prior run ended
# UNCLEANLY — never for a planned restart, and the OS-crash search is bounded to that prior run's
# window. The gateway records its start (status="running"); a clean shutdown flips it to "clean".
# A run that never got flipped = it crashed. (Addresses the false-attribution + lifecycle review.)

def read_run_state(state_path: "os.PathLike | str") -> dict:
    """The persisted lifecycle marker (whatever the most recent run wrote), or ``{}``."""
    try:
        p = Path(state_path)
        return json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        return {}


def record_gateway_start(state_path: "os.PathLike | str") -> dict:
    """Call ONCE at gateway startup. Reads the PRIOR run's marker (returned so the caller can
    decide whether to report a crash), then overwrites it with a fresh ``running`` marker for THIS
    run. Best-effort — never raises."""
    prior = read_run_state(state_path)
    try:
        p = Path(state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"started_at": time.time(), "status": "running", "pid": os.getpid()}))
    except Exception:
        pass
    return prior


def mark_gateway_clean_exit(state_path: "os.PathLike | str") -> None:
    """Call on a PLANNED shutdown/restart — flips this run's marker to ``clean`` so the next
    startup knows the prior run ended on purpose (nothing to report). Best-effort."""
    try:
        data = read_run_state(state_path)
        data["status"] = "clean"
        data["exited_at"] = time.time()
        Path(state_path).write_text(json.dumps(data))
    except Exception:
        pass


def prior_run_crashed(prior: dict) -> tuple[bool, float]:
    """From a prior-run marker → ``(crashed, started_at)``. A run recorded ``running`` that never
    got flipped to ``clean`` = crashed. Unknown/empty/clean → ``(False, 0.0)`` so we never
    misreport a planned restart or a first-ever start."""
    if not prior or prior.get("status") != "running":
        return (False, 0.0)
    try:
        return (True, float(prior.get("started_at") or 0.0))
    except (TypeError, ValueError):
        return (True, 0.0)


def _infer_cause(signal: str, backtrace: list[str]) -> str:
    """Map a signal + faulting frames to a short, human-readable cause."""
    bt = " ".join(backtrace).lower()
    sig = str(signal)
    apple_gpu = any(k in bt for k in ("metal", "libmlx", "mtlcommand", "iogpu"))
    nv_gpu = any(k in bt for k in ("cuda", "libcudart", "libcublas", "nvidia", "nvrtc"))
    if apple_gpu or nv_gpu:
        kind = "Metal/MLX" if apple_gpu else "CUDA/NVIDIA"
        return f"GPU error ({kind}) → {sig or 'abort'} (often GPU memory pressure / OOM)"
    if "out of memory" in bt or "oom" in bt:
        return f"out-of-memory → {sig}"
    if "SIGSEGV" in sig or "SEGV" in sig or "segfault" in bt:
        return f"segmentation fault ({sig}) — bad memory access, usually a native extension"
    if "SIGABRT" in sig or "Abort" in sig or "abort" in bt:
        return f"aborted ({sig}) — uncaught native exception / assertion"
    if "SIGKILL" in sig or sig == "9":
        return f"killed ({sig}) — OOM-killer or manual kill"
    return f"crashed ({sig or 'unknown signal'})"


# ─── macOS (.ips DiagnosticReports) ──────────────────────────────────────────
def _macos(name_filter: str, since_hours: float, limit: int) -> list[dict]:
    reports = Path(os.path.expanduser("~/Library/Logs/DiagnosticReports"))
    if not reports.exists():
        return []
    cutoff = time.time() - since_hours * 3600
    out: list[dict] = []
    files = [
        p for p in reports.iterdir()
        if p.suffix == ".ips" and p.stat().st_mtime >= cutoff
        and (not name_filter or name_filter.lower() in p.name.lower())
    ]
    for p in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        rec = {"os": "macos", "when": p.stat().st_mtime, "process": "?",
               "signal": "?", "backtrace": [], "source": p.name}
        try:
            # .ips is a header line + a JSON body
            body = json.loads(p.read_text(errors="replace").split("\n", 1)[1])
            rec["process"] = body.get("procName", "?")
            exc = body.get("exception", {}) or {}
            term = body.get("termination", {}) or {}
            rec["signal"] = exc.get("signal") or term.get("indicator") or "?"
            images = body.get("usedImages", [])
            for thread in body.get("threads", []) or []:
                if thread.get("triggered"):
                    for fr in (thread.get("frames") or [])[:14]:
                        i = fr.get("imageIndex", -1)
                        name = images[i].get("name", "?") if 0 <= i < len(images) else "?"
                        sym = (fr.get("symbol") or "")[:54]
                        rec["backtrace"].append(f"{name}: {sym}" if sym else name)
                    break
        except Exception:
            pass
        rec["cause"] = _infer_cause(rec["signal"], rec["backtrace"])
        out.append(rec)
    return out


# ─── Linux (systemd-coredump / journald) ─────────────────────────────────────
def _linux(name_filter: str, since_hours: float, limit: int) -> list[dict]:
    out: list[dict] = []
    try:  # preferred: systemd-coredump
        r = subprocess.run(
            ["coredumpctl", "list", "--json=short", "--no-pager"],
            capture_output=True, text=True, timeout=4,   # short — must never block the restart path
        )
        entries = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else []
        cutoff_us = (time.time() - since_hours * 3600) * 1e6
        for e in reversed(entries):
            exe = e.get("exe", "") or ""
            if name_filter and name_filter.lower() not in exe.lower():
                continue
            if e.get("time", 0) and e["time"] < cutoff_us:
                continue
            rec = {"os": "linux", "when": (e.get("time", 0) or 0) / 1e6,
                   "process": Path(exe).name or "?", "signal": str(e.get("sig", "?")),
                   "backtrace": [], "source": "coredumpctl"}
            try:
                info = subprocess.run(
                    ["coredumpctl", "info", str(e.get("pid", ""))],
                    capture_output=True, text=True, timeout=4,   # short — must never block the restart path
                ).stdout
                grab = False
                for line in info.splitlines():
                    if line.strip().startswith("Stack trace"):
                        grab = True
                        continue
                    if grab and line.startswith("    #"):
                        rec["backtrace"].append(line.strip()[:80])
            except Exception:
                pass
            rec["cause"] = _infer_cause(rec["signal"], rec["backtrace"])
            out.append(rec)
            if len(out) >= limit:
                break
    except Exception:
        pass
    if not out:  # fallback: kernel log segfault lines
        try:
            r = subprocess.run(
                ["journalctl", "-k", "--no-pager", "-n", "200"],
                capture_output=True, text=True, timeout=4,   # short — must never block the restart path
            )
            for line in reversed(r.stdout.splitlines()):
                low = line.lower()
                if ("segfault" in low or "core dumped" in low) and (
                    not name_filter or name_filter.lower() in low
                ):
                    out.append({"os": "linux", "when": time.time(), "process": "?",
                                "signal": "SIGSEGV", "backtrace": [line.strip()[:120]],
                                "cause": "segfault (from kernel log)", "source": "journald"})
                    if len(out) >= limit:
                        break
        except Exception:
            pass
    return out


# ─── Windows (Application event log / WER) ───────────────────────────────────
def _windows(name_filter: str, since_hours: float, limit: int) -> list[dict]:
    out: list[dict] = []
    ps = (
        f"$s=(Get-Date).AddHours(-{since_hours}); "
        "Get-WinEvent -FilterHashtable @{LogName='Application';"
        "ProviderName='Application Error';StartTime=$s} "
        f"-MaxEvents {max(limit, 10)} -ErrorAction SilentlyContinue | "
        "ForEach-Object { [pscustomobject]@{ t=$_.TimeCreated.ToString('o'); m=$_.Message } } "
        "| ConvertTo-Json -Compress"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=5,   # short — must never block the restart path
        )
        data = json.loads(r.stdout) if r.stdout.strip() else []
        if isinstance(data, dict):
            data = [data]
        for e in data:
            msg = e.get("m") or ""
            if name_filter and name_filter.lower() not in msg.lower():
                continue
            proc = "?"
            for line in msg.splitlines():
                if "Faulting application name" in line:
                    proc = line.split(":", 1)[1].strip().split(",")[0]
                    break
            out.append({"os": "windows", "when": time.time(), "process": proc,
                        "signal": "WER",
                        "backtrace": [l.strip() for l in msg.splitlines() if "module" in l.lower()][:5],
                        "cause": "application fault (Windows Error Reporting)", "source": "EventLog"})
            if len(out) >= limit:
                break
    except Exception:
        pass
    return out


def recent_crashes(name_filter: str = "", since_hours: float = 24, limit: int = 10) -> list[dict]:
    """Most-recent crashes (newest first), normalized + human-readable, for THIS OS.

    ``name_filter`` matches the crashing process/executable (e.g. ``"python"``).
    Returns a list of ``{os, when, process, signal, cause, backtrace, source}`` dicts.
    Best-effort: returns ``[]`` rather than raising if the platform tools are absent.
    """
    try:
        fn = {"macos": _macos, "linux": _linux, "windows": _windows}.get(os_kind())
        return fn(name_filter, since_hours, limit) if fn else []
    except Exception:
        return []


def summarize(crash: dict) -> str:
    """One-line (+ indented backtrace) human summary of a crash record."""
    when = (
        time.strftime("%Y-%m-%d %H:%M", time.localtime(crash.get("when", 0)))
        if crash.get("when") else "?"
    )
    head = f"[{when}] {crash.get('process', '?')} ({crash.get('signal', '?')}) — {crash.get('cause', '?')}"
    bt = crash.get("backtrace") or []
    return head + (("\n   " + "\n   ".join(bt[:6])) if bt else "")


def doctor_section(name_filter: str = "python", since_hours: float = 168) -> list[dict]:
    """Return recent crashes for a ``hermes doctor`` 'Recent crashes' section
    (default: last 7 days). Empty list = nothing to report (clean)."""
    return recent_crashes(name_filter=name_filter, since_hours=since_hours, limit=5)


def restart_notice(name_filter: str = "python", since_hours: float = 0.5) -> str:
    """One-line crash-cause suffix for the gateway's 'back online' notification.

    When the previous run died unexpectedly, returns a short ``"\\n\\n⚠️ …"`` string naming
    the cause; a clean restart leaves no recent crash record, so this returns ``""``. Returns
    ``""`` on ANY failure too — never raises, so it is safe to append directly to a message
    and can never block the restart path. ``since_hours`` is intentionally small: a crash that
    maps to *this* restart is seconds-to-minutes old, not hours.
    """
    try:
        crashes = recent_crashes(name_filter=name_filter, since_hours=since_hours, limit=1)
        if not crashes:
            return ""
        crash = crashes[0]
        return (
            "\n\n⚠️ The previous run ended unexpectedly: "
            f"{crash.get('process', '?')} ({crash.get('signal', '?')}) — {crash.get('cause', '?')}"
        )
    except Exception:
        return ""
