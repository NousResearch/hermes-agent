#!/usr/bin/env python3
"""
HERMES RESOURCE GUARD — Pre-launch gate for heavyweight models.

Checks system resources and running processes before allowing
large-model launches (30B+). Can optionally kill non-essential
processes to reclaim memory.

Integrated with model_routing.py — called before any model
selection that would exceed the RESOURCE_GUARD_THRESHOLD.
"""

import os
import sys
import json
import subprocess
import shutil
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Configuration ────────────────────────────────────────────────

# Models that require resource guard check (size in GB)
GUARDED_MODELS = {
    "qwen3-coder:30b-a3b-q4_k_M":  {"ram_gb": 14, "label": "Qwen3 Coder 30B A3B"},  # q4 actual ~14GB
    "qwen2.5-coder:32b-instruct-q4_K_M": {"ram_gb": 20, "label": "Qwen2.5 Coder 32B"},
    "qwen2.5-coder:32b-instruct-q4_K_M-96k": {"ram_gb": 20, "label": "Qwen2.5 Coder 32B 96K"},
}

GUARD_THRESHOLD_GB = 12  # Any model needing >12GB triggers guard (was 16, blocked 30B coder @ 18GB)

# Processes that MUST NOT be killed
ESSENTIAL_PROCESSES = {"Terminal", "iTerm2", "Telegram"}  # never kill

# Processes that SHOULD be killed before heavy model launch
KILL_CANDIDATES = {
    "Google Chrome":    "chrome",
    "Chromium":         "chromium",
    "Brave Browser":    "brave",
    "Firefox":          "firefox",
    "Safari":           "safari",  # macOS only
    "Docker Desktop":   "Docker",
    "IntelliJ IDEA":    "idea",
    "VSCode":           "Code",
    "Slack":            "Slack",
    "Discord":          "Discord",
}

# Telegram and terminal are EXEMPT from killing
EXEMPT_PROCESSES = {"Telegram", "Terminal", "iTerm2", os.path.basename(sys.executable)}

LOG_DIR = Path(os.path.expanduser("~/.hermes/logs"))
LOG_DIR.mkdir(exist_ok=True)
GUARD_LOG = LOG_DIR / "resource_guard.jsonl"


# ── System info ──────────────────────────────────────────────────

def get_platform() -> str:
    """Detect current platform."""
    try:
        with open("/proc/version", "r") as f:
            if "microsoft" in f.read().lower():
                return "wsl"
    except FileNotFoundError:
        pass

    if sys.platform == "darwin":
        return "macos"
    elif sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def get_memory_info() -> Dict:
    """Get available system memory."""
    platform = get_platform()

    if platform == "macos":
        # Use `vm_stat` on macOS — page size varies by hardware
        try:
            result = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split("\n")
            info = {}
            page_size = 4096  # default, overridden below
            for line in lines:
                if line.startswith("Pages") and "page size" not in line:
                    key = line.split(":")[0].strip().replace(" ", "_")
                    val = line.split(":")[1].strip().replace(".", "").replace(",", "")
                    try:
                        info[key] = int(val)
                    except ValueError:
                        info[key] = val
                elif "page size of" in line.lower():
                    # "page size of 16384 bytes"
                    parts = line.split()
                    for p in parts:
                        try:
                            page_size = int(p)
                            break
                        except ValueError:
                            continue

            pages_free = info.get("Pages_free", 0)
            pages_active = info.get("Pages_active", 0)
            pages_inactive = info.get("Pages_inactive", 0)
            pages_speculative = info.get("Pages_speculative", 0)
            pages_wired = info.get("Pages_wired_down", 0)
            # Available = free + inactive + speculative (macOS memory pressure model)
            available = (pages_free + pages_inactive + pages_speculative) * page_size
            used = (pages_active + pages_wired) * page_size
            total = (pages_free + pages_active + pages_inactive + pages_wired + pages_speculative) * page_size

            return {
                "total_gb": round(total / (1024**3), 2),
                "available_gb": round(available / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(pages_free * page_size / (1024**3), 2),
                "pages_free": pages_free,
                "pages_active": pages_active,
                "pages_inactive": pages_inactive,
                "pages_wired": pages_wired,
                "page_size": page_size,
            }
        except Exception as e:
            return {"error": str(e), "total_gb": 32.0, "available_gb": 16.0}

    elif platform == "linux":
        try:
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            info = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1]) * 1024  # kB → bytes

            total = info.get("MemTotal", 0)
            avail = info.get("MemAvailable", info.get("MemFree", 0))
            used = total - avail

            return {
                "total_gb": round(total / (1024**3), 2),
                "available_gb": round(avail / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(info.get("MemFree", 0) / (1024**3), 2),
            }
        except Exception as e:
            return {"error": str(e), "total_gb": 32.0, "available_gb": 16.0}

    return {"error": f"Unknown platform: {platform}", "total_gb": 32.0, "available_gb": 16.0}


def get_ollama_processes() -> List[Dict]:
    """Get info about running Ollama processes."""
    results = []
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            if "ollama" in line.lower() and "grep" not in line.lower():
                parts = line.split()
                if len(parts) >= 11:
                    results.append({
                        "user": parts[0],
                        "pid": parts[1],
                        "cpu": parts[2],
                        "mem": parts[3],
                        "command": " ".join(parts[10:]),
                    })
    except Exception as e:
        results.append({"error": str(e)})
    return results


def _is_real_safari(command: str) -> bool:
    """Filter out Safari helper daemons — only block the real browser."""
    # Real Safari binary contains "Safari.app/Contents/MacOS/Safari"
    # Exclude XPC services, launch agents, helpers, extensions
    excludes = ["XPCServices", "xpc", "Helper", "LaunchAgent",
                "BrowserDataImportingService", "SafeBrowsing.Service",
                "BookmarksSyncAgent", "History", "WidgetExtension",
                "PlatformSupport", "ConfigurationSubscriber",
                "NotificationAgent", "Cryptexes"]
    return "Safari.app" in command and not any(e in command for e in excludes)


def find_processes(process_names: List[str]) -> Dict[str, List[Dict]]:
    """Find running processes by name."""
    found = {}
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().split("\n")

        for name in process_names:
            found[name] = []
            for line in lines[1:]:
                if name.lower() in line.lower() and "grep" not in line.lower():
                    parts = line.split(None, 10)
                    cmd = parts[10] if len(parts) > 10 else line
                    # Special filter for Safari
                    if "Safari" in name and not _is_real_safari(cmd):
                        continue
                    found[name].append({
                        "user": parts[0] if len(parts) > 0 else "?",
                        "pid": parts[1] if len(parts) > 1 else "?",
                        "cpu": parts[2] if len(parts) > 2 else "?",
                        "mem": parts[3] if len(parts) > 3 else "?",
                        "command": cmd,
                    })
    except Exception as e:
        found["_error"] = [{"error": str(e)}]
    return found


# ── Guard checks ─────────────────────────────────────────────────

def check_requirements(model_name: str) -> Dict:
    """
    Full resource guard check for a model launch.
    Returns {allowed: bool, warnings: [], errors: [], actions: []}
    """
    result = {
        "allowed": True,
        "warnings": [],
        "errors": [],
        "actions_needed": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # 1. Check if this model needs guarding
    model_key = model_name.replace("mac-ollama:", "").replace("linux-ollama:", "")
    guard_config = GUARDED_MODELS.get(model_key)

    if not guard_config or model_key not in GUARDED_MODELS:
        # Not a guarded model — allow but log
        if any(k in model_name for k in ["30b", "32b"]):
            result["warnings"].append(
                f"Model {model_name} is 30B+ class but not in GUARDED_MODELS registry — "
                f"treating as guarded anyway"
            )
            guard_config = {"ram_gb": 18, "label": model_name}
        else:
            result["warnings"].append(f"Model {model_name} not in guarded list — skipping guard")
            return result

    required_gb = guard_config["ram_gb"]
    result["model"] = guard_config["label"]
    result["required_gb"] = required_gb

    # 2. Check memory
    mem = get_memory_info()
    result["memory"] = mem

    if "error" in mem:
        result["warnings"].append(f"Could not determine memory: {mem['error']}")
    else:
        available = mem.get("available_gb", 0)
        if available < required_gb:
            result["errors"].append(
                f"INSUFFICIENT MEMORY: {available:.1f}GB available, "
                f"{required_gb}GB required for {guard_config['label']}"
            )
            result["allowed"] = False
        elif available < required_gb * 1.3:
            result["warnings"].append(
                f"TIGHT MEMORY: {available:.1f}GB available vs {required_gb}GB required "
                f"({available/required_gb:.0%} of requirement)"
            )
        else:
            result["warnings"].append(
                f"Memory OK: {available:.1f}GB available for {required_gb}GB requirement"
            )

    # 3. Check for browser processes
    platform = get_platform()
    browser_names = []
    if platform == "macos":
        browser_names = ["Google Chrome", "Chromium", "Brave Browser", "Safari", "Firefox"]
    elif platform == "linux":
        browser_names = ["chrome", "chromium", "brave", "firefox"]

    browsers = find_processes(browser_names)
    running_browsers = {k: v for k, v in browsers.items() if v and "error" not in v}

    if running_browsers:
        browser_strs = []
        for name, procs in running_browsers.items():
            for p in procs:
                browser_strs.append(f"{name} (PID {p['pid']}, {p['mem']} mem)")

        result["errors"].append(
            "BROWSERS RUNNING — MUST CLOSE BEFORE 30B LAUNCH:\n  " +
            "\n  ".join(browser_strs)
        )
        result["allowed"] = False
        result["actions_needed"].append(
            "Close ALL browsers (Chrome, Brave, Safari, Firefox) before launch"
        )

    # 4. Check for other heavy processes
    heavy_check_names = list(KILL_CANDIDATES.keys())
    heavy = find_processes(heavy_check_names)
    running_heavy = {k: v for k, v in heavy.items() if v and "error" not in v}

    for name, procs in running_heavy.items():
        if name in running_browsers:
            continue  # Already reported
        for p in procs:
            result["warnings"].append(
                f"Heavy process running: {name} (PID {p['pid']}, {p['mem']} mem) — "
                f"consider closing to reclaim RAM"
            )
            result["actions_needed"].append(
                f"Close {name} (PID {p['pid']}) to free ~{p['mem']}%"
            )

    # 5. Check Ollama memory usage
    ollama_procs = get_ollama_processes()
    if ollama_procs:
        total_mem_pct = sum(
            float(p.get("mem", "0").replace("%", "")) for p in ollama_procs if "%" in p.get("mem", "")
        )
        result["ollama_procs"] = ollama_procs
        result["warnings"].append(
            f"Ollama has {len(ollama_procs)} process(es) using ~{total_mem_pct:.1f}% memory combined"
        )

    return result


def kill_processes(process_names: List[str], dry_run: bool = True) -> List[str]:
    """Kill specified processes. Returns list of actions taken."""
    actions = []
    for name in process_names:
        procs = find_processes([name])
        if name in procs and procs[name]:
            for p in procs[name]:
                pid = p["pid"]
                if pid in ("?", "1"):
                    continue
                if dry_run:
                    actions.append(f"[DRY RUN] Would kill {name} (PID {pid})")
                else:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        actions.append(f"Killed {name} (PID {pid})")
                    except ProcessLookupError:
                        actions.append(f"{name} (PID {pid}) already gone")
                    except PermissionError:
                        actions.append(f"Cannot kill {name} (PID {pid}) — permission denied")
                    except Exception as e:
                        actions.append(f"Failed to kill {name} (PID {pid}): {e}")
    return actions


def suggest_kills() -> Tuple[List[str], List[str]]:
    """Suggest which processes to kill. Returns (safe_to_kill, do_not_kill)."""
    safe = []
    do_not = []
    platform = get_platform()

    all_check = list(KILL_CANDIDATES.keys())
    found = find_processes(all_check)

    for name in all_check:
        if name in found and found[name]:
            for p in found[name]:
                if name in ESSENTIAL_PROCESSES or name in EXEMPT_PROCESSES:
                    do_not.append(f"{name} (PID {p['pid']}) — EXEMPT")
                else:
                    safe.append(f"{name} (PID {p['pid']})")

    return safe, do_not


def log_guard_result(result: Dict, model_name: str):
    """Append guard check to JSONL log."""
    log_entry = {
        "timestamp": result.get("timestamp"),
        "model": model_name,
        "allowed": result["allowed"],
        "errors": result["errors"],
        "warnings_count": len(result["warnings"]),
        "memory": {
            "total_gb": result.get("memory", {}).get("total_gb"),
            "available_gb": result.get("memory", {}).get("available_gb"),
        },
    }
    try:
        with open(GUARD_LOG, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass


# ── Main interface ───────────────────────────────────────────────

def require_resources(model_name: str) -> bool:
    """
    Called before launching a heavy model. Returns True if launch is safe.
    Prints warnings/errors and returns False if conditions are not met.
    """
    result = check_requirements(model_name)
    log_guard_result(result, model_name)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║           HERMES RESOURCE GUARD — PRE-LAUNCH            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Model:  {result.get('model', model_name)}")
    print(f"  Required: {result.get('required_gb', '?')}GB RAM")
    print()

    if result["memory"].get("total_gb"):
        mem = result["memory"]
        print(f"  System RAM: {mem['total_gb']}GB total")
        print(f"  Available:  {mem['available_gb']}GB")
        print(f"  Used:       {mem['used_gb']}GB")
        print()

    if result["errors"]:
        print("  ❌ ERRORS (launch BLOCKED):")
        for e in result["errors"]:
            for line in e.split("\n"):
                print(f"    • {line}")
        print()

    if result["warnings"]:
        print("  ⚠️  WARNINGS:")
        for w in result["warnings"]:
            print(f"    • {w}")
        print()

    if result["actions_needed"]:
        print("  📋 ACTIONS REQUIRED:")
        for a in result["actions_needed"]:
            print(f"    → {a}")
        print()

    if result["allowed"]:
        print("  ✅ RESOURCE GUARD: PASSED — safe to launch")
    else:
        print("  🚫 RESOURCE GUARD: BLOCKED — resolve errors above first")

    print()

    # Show killable processes
    safe, exempt = suggest_kills()
    if safe:
        print("  Processes safe to kill before launch:")
        for s in safe:
            print(f"    ✗ {s}")
        print()
    if exempt:
        print("  Processes exempt (will not be touched):")
        for e in exempt:
            print(f"    ✓ {e}")
        print()

    return result["allowed"]


def auto_free_resources(dry_run: bool = False) -> List[str]:
    """Kill non-essential processes to free resources for heavy model launch."""
    safe, _ = suggest_kills()
    actions = []
    for entry in safe:
        parts = entry.split(" (PID ")
        name = parts[0]
        pid = parts[1].rstrip(")")
        try:
            if not dry_run:
                os.kill(int(pid), signal.SIGTERM)
            actions.append(f"{'[DRY RUN] ' if dry_run else ''}Killed {name} (PID {pid})")
        except (ProcessLookupError, PermissionError) as e:
            actions.append(f"Could not kill {name} (PID {pid}): {e}")
    return actions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hermes Resource Guard")
    parser.add_argument("--check", "-c", metavar="MODEL",
                        help="Check if MODEL can be launched safely")
    parser.add_argument("--free", "-f", action="store_true",
                        help="Auto-kill non-essential processes")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be killed without actually killing")
    parser.add_argument("--full-report", "-r", action="store_true",
                        help="Full system resource report")

    args = parser.parse_args()

    if args.check:
        ok = require_resources(args.check)
        sys.exit(0 if ok else 1)

    elif args.free:
        print("Auto-freeing resources...")
        actions = auto_free_resources(dry_run=args.dry_run)
        for a in actions:
            print(f"  {a}")
        if not actions:
            print("  Nothing to kill — path is clear.")

    elif args.full_report:
        print("=== FULL SYSTEM RESOURCE REPORT ===\n")
        mem = get_memory_info()
        print(f"Platform:      {get_platform()}")
        print(f"Total RAM:     {mem.get('total_gb', '?')}GB")
        print(f"Available RAM: {mem.get('available_gb', '?')}GB")
        print(f"Used RAM:      {mem.get('used_gb', '?')}GB\n")

        ollama = get_ollama_processes()
        if ollama:
            print("Ollama processes:")
            for p in ollama:
                print(f"  PID {p['pid']} — {p.get('command', '?')} ({p['mem']} mem, {p['cpu']} cpu)")
            print()

        print("Heavy processes (candidates for kill):")
        platform = get_platform()
        names = list(KILL_CANDIDATES.keys())
        found = find_processes(names)
        for name, procs in found.items():
            if procs and "error" not in procs:
                for p in procs:
                    exempt = " [EXEMPT]" if name in ESSENTIAL_PROCESSES else ""
                    print(f"  {name} — PID {p['pid']} ({p['mem']} mem, {p['cpu']} cpu){exempt}")

        print(f"\nExempt processes: {', '.join(EXEMPT_PROCESSES)}")
        print(f"\nGuard threshold: models >{GUARD_THRESHOLD_GB}GB RAM")
        print(f"Guarded models: {list(GUARDED_MODELS.keys())}")

    else:
        # Default: run full check for 30B model
        print("Running default guard check for qwen3-coder:30b-a3b...\n")
        require_resources("qwen3-coder:30b-a3b-q4_k_M")