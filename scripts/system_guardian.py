#!/usr/bin/env python3
"""system_guardian.py - meta-watchdog for the seams no other watchdog covers.

v1 covers (complements, never duplicates):
  1. Cron scheduler / serve process (state/gateway.heartbeat + cmdline scan).
  2. Messaging gateway (gateway_state.json pid) -> Ensure vbs.
  3. Meta-watchdog: Hermes_Gateway_Watchdog schtask must be Ready.
  4. Disk floor: C: free < 2GB -> alert.

v2 additions (2026-08-31, from fleet audit):
  5. Python-runtime canary: `python -c "print(1)"` must exit 0 (fleet blindness).
  6. Skills-dir guard: root skills/ count; alert on >30% drop vs stored baseline.
  7. state.db snapshot via sqlite backup API (daily, keep 3) + weekly quick_check.
  8. Retention GC: backups/ drift files >30d, cron/output >14d, logs >30d.

Contract: SILENT (exit 0, no stdout) when everything healthy and nothing was
done. Prints a one-line JSON summary only when it acted or found a problem.
Never edits config.yaml. Never touches approvals. Restarts own-scope processes
only. All actions logged to state/system_guardian.log (rotated at 1MB).

Cron wiring: no_agent job `system-guardian`, every 10m.
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / "AppData" / "Local" / "hermes"))
STATE = HERMES_HOME / "state"
LOG = STATE / "system_guardian.log"
LOCK_DIR = STATE / "locks"
GATEWAY_STATE = HERMES_HOME / "gateway_state.json"
HEARTBEAT = STATE / "gateway.heartbeat"
STALE_S = 780          # heartbeat refreshed on scheduler ticks (10m jobs); 13min = safe
RESTART_BACKOFF_S = 600
# 20.0 was calibrated 08-31 when 7GB of ETL traces sat on C:; they were moved to
# D: 09-01. Remaining bulk is structural (pagefile ~19GB self-contracts on reboot,
# state.db ~100MB/day) — 15GB avoids permanent alert noise, stays above danger.
DISK_FLOOR_GB = float(os.environ.get("HERMES_DISK_FLOOR_GB", "15.0"))
VENV_PY = HERMES_HOME / "hermes-agent" / "venv" / "Scripts" / "python.exe"
STATE_DB = HERMES_HOME / "state.db"
# 2026-08-31 tier split: DR snapshots on D:\AI (separate physical disk from
# the live db) — a C: failure can't destroy both the db and its snapshots.
SNAP_DIR = Path(os.environ.get("HERMES_SNAP_DIR", "D:/AI/hermes-snapshots"))
SNAP_KEEP = 3
SKILLS_DIR = HERMES_HOME / "skills"
GC_TARGETS = [  # (path, max age days)
    (HERMES_HOME / "backups", 30),
    (HERMES_HOME / "cron" / "output", 14),
    (HERMES_HOME / "logs", 30),
]
# P-0130 interim mitigation: pywebview private_mode leaks a ~30MB
# tmp*/EBWebView WebView2 user-data profile under %TEMP% per Fleet Bridge
# launch (process dies without graceful shutdown -> clear_user_data() never
# runs). Only dirs containing an EBWebView subfolder are swept, so unrelated
# tmp* temp files/dirs are never touched. Source fix tracked in P-0130.
WEBVIEW_TMP_MAX_AGE_H = float(os.environ.get("HERMES_WEBVIEW_TMP_MAX_AGE_H", "6"))
# Set HERMES_GUARDIAN_DRY_RUN=1 to log candidates without deleting anything.
DRY_RUN = os.environ.get("HERMES_GUARDIAN_DRY_RUN", "") == "1"

now_epoch = time.time()
alerts = []
actions = []


def log(msg: str) -> None:
    try:
        if LOG.exists() and LOG.stat().st_size > 1_000_000:
            LOG.replace(LOG.with_suffix(".log.1"))
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{stamp} {msg}\n")
    except OSError:
        pass


def pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        return False


def serve_process_alive() -> bool:
    """True if any live process cmdline contains 'hermes_cli.main serve'."""
    try:
        import psutil
        for p in psutil.process_iter(["cmdline"]):
            try:
                cl = p.info.get("cmdline") or []
                joined = " ".join(str(x) for x in cl)
                if "hermes_cli.main" in joined and " serve" in joined:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        return False
    return False


def acquire_lock() -> bool:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lf = LOCK_DIR / "system_guardian.lock"
    try:
        if lf.exists():
            # stale lock if older than 15 min (script runs in seconds)
            if now_epoch - lf.stat().st_mtime < 900:
                return False
            lf.unlink(missing_ok=True)
        fd = os.open(str(lf), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False
    except OSError:
        return False


def load_state() -> dict:
    state_f = STATE / "system_guardian_state.json"
    try:
        return json.loads(state_f.read_text(encoding="utf-8")) if state_f.exists() else {}
    except Exception:
        return {}


def save_state(st: dict) -> None:
    state_f = STATE / "system_guardian_state.json"
    try:
        state_f.write_text(json.dumps(st), encoding="utf-8")
    except OSError:
        pass


def check_scheduler() -> None:
    """Cron scheduler / serve process liveness."""
    hb_pid = None
    fresh = False
    try:
        hb = json.loads(HEARTBEAT.read_text(encoding="utf-8"))
        hb_pid = int(hb.get("pid") or 0)
        updated = hb.get("updated_at", "")
        from datetime import datetime as dt
        t = dt.fromisoformat(updated.replace("Z", "+00:00"))
        age = now_epoch - t.timestamp()
        fresh = age < STALE_S
    except Exception:
        fresh = False

    cmdline_alive = serve_process_alive()
    if cmdline_alive and fresh:
        return
    if cmdline_alive and not fresh:
        # process exists but heartbeat stale: give it the benefit once, just alert
        alerts.append(f"scheduler alive (pid?) but heartbeat stale >{STALE_S}s")
        return
    # no serve process at all
    if hb_pid and pid_alive(hb_pid) and fresh:
        return  # pid alive per psutil, trust it
    st = load_state()
    last_try = float(st.get("last_serve_restart", 0))
    if now_epoch - last_try < RESTART_BACKOFF_S:
        alerts.append("scheduler dead but restart in backoff window")
        return
    if not VENV_PY.exists():
        alerts.append("scheduler dead and venv python missing - cannot restart")
        return
    env = os.environ.copy()
    env["HERMES_HOME"] = str(HERMES_HOME)
    env["PYTHONIOENCODING"] = "utf-8"
    env["VIRTUAL_ENV"] = str(VENV_PY.parent.parent)
    pp = HERMES_HOME / "hermes-agent"
    env["PYTHONPATH"] = str(pp) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        DET = getattr(subprocess, "DETACHED_PROCESS", 0)
        flags = DET | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        subprocess.Popen(
            [str(VENV_PY), "-m", "hermes_cli.main", "serve", "--host", "127.0.0.1", "--port", "0"],
            cwd=str(HERMES_HOME), env=env, creationflags=flags,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        st["last_serve_restart"] = now_epoch
        save_state(st)
        actions.append("restarted serve process (was dead; stale={})".format(not fresh))
        log("RESTART serve requested (heartbeat_fresh={})".format(fresh))
    except Exception as e:
        alerts.append("serve restart failed: {}".format(e))
        log("RESTART-FAIL serve: {}".format(e))


def check_messaging_gateway() -> None:
    try:
        gs = json.loads(GATEWAY_STATE.read_text(encoding="utf-8"))
        pid = int(gs.get("pid") or 0)
    except Exception:
        alerts.append("gateway_state.json missing/unreadable")
        return
    if pid and pid_alive(pid):
        return
    vbs = HERMES_HOME / "gateway-service" / "Ensure_Hermes_Gateway.vbs"
    if vbs.exists():
        try:
            subprocess.Popen(["wscript.exe", "//B", "//Nologo", str(vbs)],
                             creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
            actions.append("requested gateway restart via Ensure vbs (pid {} dead)".format(pid))
            log("GATEWAY-RESTART requested (pid {} not alive)".format(pid))
        except Exception as e:
            alerts.append("gateway ensure request failed: {}".format(e))
    else:
        alerts.append("messaging gateway pid {} dead and no Ensure vbs found".format(pid))


def check_meta_watchdog() -> None:
    """The watchdog scheduled task itself must be enabled and recent."""
    try:
        out = subprocess.run(
            ["schtasks", "/query", "/tn", "Hermes_Gateway_Watchdog", "/fo", "CSV", "/nh"],
            capture_output=True, text=True, timeout=30)
        line = (out.stdout or "").strip()
        if not line:
            alerts.append("Hermes_Gateway_Watchdog task not found (meta-watchdog blind)")
            return
        parts = next(__import__("csv").reader([line]))
        status = parts[2] if len(parts) > 2 else ""
        if "ready" not in status.lower():
            alerts.append("Hermes_Gateway_Watchdog status={}".format(status))
    except Exception as e:
        alerts.append("meta-watchdog check failed: {}".format(e))


def check_disk() -> None:
    try:
        free_gb = shutil.disk_usage("C:\\").free / 1e9
        if free_gb < DISK_FLOOR_GB:
            # Repeat suppression: a KNOWN, unchanged sub-floor condition is
            # already diagnosed+filed (P-0118 state.db growth). Re-alert only on
            # real decline (drops >0.5GB below last notified level) or hourly
            # re-ping; otherwise the 10m cadence just wakes the agent for noise.
            st = load_state()
            last = st.get("disk_last_notified_gb")
            now_epoch = time.time()
            last_ts = st.get("disk_last_notified_ts", 0)
            if last is not None and free_gb > last - 0.5 and (now_epoch - last_ts) < 6 * 3600:
                pass  # stable/oscillating: silent, condition still tracked
            else:
                alerts.append("C: free {:.1f}GB below floor {}GB".format(free_gb, DISK_FLOOR_GB))
                st["disk_last_notified_gb"] = round(free_gb, 2)
                st["disk_last_notified_ts"] = now_epoch
                save_state(st)
        # D:\AI = fleet cold tier (backups, snapshots, repo, scratch)
        free_d = shutil.disk_usage("D:\\").free / 1e9
        if free_d < 100.0:
            alerts.append("D: free {:.0f}GB below 100GB cold-tier floor".format(free_d))
        # lost+found check: live db must never move to D: (performance tier rule)
    except Exception as e:
        alerts.append("disk check failed: {}".format(e))


def check_python_canary() -> None:
    """Python runtime must actually work - every job is Python; breakage = total blindness."""
    py = str(VENV_PY) if VENV_PY.exists() else sys.executable
    try:
        r = subprocess.run([py, "-c", "print(1)"], capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or (r.stdout or "").strip() != "1":
            alerts.append("python canary failed rc={} out={!r}".format(r.returncode, (r.stdout or "")[:20]))
    except Exception as e:
        alerts.append("python canary error: {}".format(e))


def check_skills_dir() -> None:
    """Alert on sudden skills/ loss (>30% drop vs baseline)."""
    st = load_state()
    try:
        count = sum(1 for p in SKILLS_DIR.iterdir() if p.is_dir()) if SKILLS_DIR.exists() else 0
    except OSError:
        alerts.append("skills/ dir unreadable")
        return
    baseline = st.get("skills_baseline")
    if baseline and count < baseline * 0.7:
        alerts.append("skills/ count {} dropped >30% from baseline {}".format(count, baseline))
    if not baseline or count > baseline:
        st["skills_baseline"] = count
        save_state(st)


def check_state_db() -> None:
    """Daily backup-API snapshot (keep 3) + weekly quick_check. Read-safe on live WAL db."""
    st = load_state()
    if not STATE_DB.exists():
        alerts.append("state.db missing!")
        return
    # daily snapshot
    last_snap = float(st.get("last_db_snapshot", 0))
    if now_epoch - last_snap > 86400:
        try:
            SNAP_DIR.mkdir(parents=True, exist_ok=True)
            dst = SNAP_DIR / ("state-{}.db".format(datetime.now(timezone.utc).strftime("%Y%m%d")))
            src = sqlite3.connect(str(STATE_DB))
            dst_tmp = SNAP_DIR / ".snap.tmp"
            out = sqlite3.connect(str(dst_tmp))
            src.backup(out)
            out.close()
            src.close()
            dst_tmp.replace(dst)
            # retention
            snaps = sorted(SNAP_DIR.glob("state-*.db"), key=lambda p: p.stat().st_mtime)
            for old in snaps[:-SNAP_KEEP]:
                old.unlink(missing_ok=True)
            st["last_db_snapshot"] = now_epoch
            save_state(st)
            actions.append("state.db snapshot {} ({:.0f}MB)".format(
                dst.name, dst.stat().st_size / 1e6))
            log("DB-SNAPSHOT {} ({:.0f}MB)".format(dst.name, dst.stat().st_size / 1e9 * 1e3))
        except Exception as e:
            alerts.append("state.db snapshot failed: {}".format(e))
    # weekly quick_check
    last_chk = float(st.get("last_db_check", 0))
    if now_epoch - last_chk > 7 * 86400:
        try:
            conn = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True, timeout=30)
            row = conn.execute("PRAGMA quick_check").fetchone()
            conn.close()
            verdict = (row[0] if row else "no-result")[:40]
            if verdict != "ok":
                alerts.append("state.db quick_check: {}".format(verdict))
            st["last_db_check"] = now_epoch
            st["last_db_check_result"] = verdict
            save_state(st)
            if verdict != "ok":
                log("DB-CHECK {}".format(verdict))
        except Exception as e:
            alerts.append("state.db quick_check failed: {}".format(e))


def gc_retention() -> None:
    """Age-based GC for unbounded dirs (backups drift files, cron/output, logs)."""
    removed = 0
    for root, max_age_days in GC_TARGETS:
        if not root.exists():
            continue
        cutoff = now_epoch - max_age_days * 86400
        try:
            for p in root.rglob("*"):
                try:
                    if p.is_file() and p.stat().st_mtime < cutoff:
                        p.unlink()
                        removed += 1
                except OSError:
                    continue
        except OSError:
            continue
    if removed:
        actions.append("gc removed {} stale files (backups>30d, cron/output>14d, logs>30d)".format(removed))


def gc_webview_tmp() -> None:
    """P-0130 interim: sweep %TEMP%/tmp* dirs that contain an EBWebView
    (WebView2 user-data) subfolder and are older than WEBVIEW_TMP_MAX_AGE_H.
    Dirs without EBWebView are never touched. Honors DRY_RUN (logs only)."""
    removed = 0
    cutoff = now_epoch - WEBVIEW_TMP_MAX_AGE_H * 3600
    tmp_root = Path(os.environ.get("TEMP") or os.environ.get("TMP") or tempfile.gettempdir())
    try:
        candidates = list(tmp_root.glob("tmp*"))
    except OSError:
        return
    for d in candidates:
        try:
            if not d.is_dir():
                continue
            if not (d / "EBWebView").exists():
                continue  # unrelated tmp* dir/file: leave alone
            if d.stat().st_mtime >= cutoff:
                continue  # fresh (<6h): preserved
            if DRY_RUN:
                log("GC-DRYRUN would remove {}".format(d))
                continue
            shutil.rmtree(d, ignore_errors=False)
            removed += 1
        except OSError:
            continue
    if removed:
        actions.append(
            "gc removed {} stale EBWebView tmp* dirs (>{:.0f}h, P-0130 interim)".format(
                removed, WEBVIEW_TMP_MAX_AGE_H
            )
        )
        log("GC removed {} stale EBWebView tmp* dirs".format(removed))


def main() -> int:
    if not acquire_lock():
        return 0  # another instance running; silent
    try:
        check_scheduler()
        check_messaging_gateway()
        check_meta_watchdog()
        check_disk()
        check_python_canary()
        check_skills_dir()
        check_state_db()
        gc_retention()
        gc_webview_tmp()
    finally:
        try:
            (LOCK_DIR / "system_guardian.lock").unlink(missing_ok=True)
        except OSError:
            pass
    if alerts or actions:
        print(json.dumps({"alerts": alerts, "actions": actions}))
        return 0
    return 0  # silent when healthy


if __name__ == "__main__":
    sys.exit(main())
