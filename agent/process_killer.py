"""Process tree termination for preemptive cancellation.

When a job is cancelled, we need to kill not just the top-level shell
process but the entire process tree (child processes, grandchildren, etc.).
This module uses psutil for cross-platform process tree discovery and killing.
"""
from __future__ import annotations

import os
import signal
import time
from typing import Optional

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def kill_process_tree(pid: int, timeout: float = 5.0) -> dict:
    """Kill a process and all its children.

    Tries graceful termination first (SIGTERM), waits up to timeout seconds,
    then force-kills survivors (SIGKILL).

    Returns a dict with:
        - killed_pids: list of PIDs that were terminated
        - survived: list of PIDs that could not be killed
        - method: 'psutil' or 'fallback'
    """
    if _HAS_PSUTIL:
        return _kill_tree_psutil(pid, timeout)
    return _kill_tree_fallback(pid, timeout)


def _kill_tree_psutil(pid: int, timeout: float) -> dict:
    """Kill process tree using psutil."""
    killed_pids = []
    survived = []

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return {"killed_pids": [], "survived": [], "method": "psutil"}

    # Collect all children (recursive)
    children = parent.children(recursive=True)
    all_procs = children + [parent]

    # Send SIGTERM to all (children first, then parent)
    for proc in reversed(all_procs):
        try:
            proc.terminate()
            killed_pids.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Wait for graceful termination
    deadline = time.time() + timeout
    for proc in all_procs:
        remaining = max(0.1, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except psutil.TimeoutExpired:
            # Force kill
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            except psutil.TimeoutExpired:
                survived.append(proc.pid)
        except psutil.NoSuchProcess:
            pass  # Already gone

    # Verify all dead
    for pid in killed_pids[:]:
        if psutil.pid_exists(pid):
            try:
                p = psutil.Process(pid)
                if p.status() != psutil.STATUS_ZOMBIE:
                    survived.append(pid)
            except psutil.NoSuchProcess:
                pass  # Dead now

    # Deduplicate survived
    survived = list(set(survived))
    killed_pids = [p for p in killed_pids if p not in survived]

    return {
        "killed_pids": killed_pids,
        "survived": survived,
        "method": "psutil",
    }


def _kill_tree_fallback(pid: int, timeout: float) -> dict:
    """Fallback process tree kill without psutil (os-level only).

    On Unix: use os.kill with process group.
    On Windows: use taskkill.
    """
    killed_pids = []
    survived = []

    if os.name == "nt":
        # Windows: taskkill /T /F /PID
        import subprocess
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode == 0:
                killed_pids.append(pid)
            else:
                survived.append(pid)
        except subprocess.TimeoutExpired:
            survived.append(pid)
        except FileNotFoundError:
            survived.append(pid)
    else:
        # Unix: try to kill the process group
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            killed_pids.append(pid)
        except (ProcessLookupError, PermissionError):
            pass
        except OSError:
            pass

        # Wait briefly
        time.sleep(min(1.0, timeout))

        # Force kill if still alive
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass  # Already dead
        except OSError:
            pass

        # Check if still alive
        try:
            os.kill(pid, 0)
            survived.append(pid)
        except (ProcessLookupError, PermissionError):
            pass  # Dead
        except OSError:
            pass

    return {
        "killed_pids": killed_pids,
        "survived": survived,
        "method": "fallback",
    }


def kill_processes(pids: list[int], timeout: float = 5.0) -> list[dict]:
    """Kill multiple process trees. Returns list of kill results."""
    results = []
    for pid in pids:
        result = kill_process_tree(pid, timeout)
        results.append(result)
    return results


def get_remaining_processes(pids: list[int]) -> list[dict]:
    """Check which processes are still alive.

    Returns list of {pid, status, name} for surviving processes.
    """
    remaining = []
    for pid in pids:
        if _HAS_PSUTIL:
            try:
                proc = psutil.Process(pid)
                remaining.append({
                    "pid": pid,
                    "status": proc.status(),
                    "name": proc.name(),
                })
            except psutil.NoSuchProcess:
                pass
        else:
            try:
                os.kill(pid, 0)
                remaining.append({"pid": pid, "status": "alive", "name": "unknown"})
            except (ProcessLookupError, PermissionError):
                pass
            except OSError:
                pass
    return remaining
