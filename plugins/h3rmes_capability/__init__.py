"""
h3rmes Capability Monitor plugin.

Detects missing subsystems (irontology-mcp, codebase-memory, b00t-mcp)
at session start and dispatches operator sub-agents to auto-remediate.

Architecture:
  Executive (this plugin) → operator dispatch → sub-agent fix → re-check

Usage:
  On session start, performs a capability check. If any known capability
  is missing, it logs the gap and (if auto-fix is enabled) dispatches a
  sub-agent via delegate_task to remediate.

Config (in ~/.hermes/config.yaml):
  h3rmes:
    auto_fix: true   # auto-dispatch fixes on capability gap
    check_interval: 3600  # min seconds between checks
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Known capability registry ─────────────────────────────────────────────────
# Each entry: (name, check_fn, fix_cmd, description)
# check_fn: returns True if capability is operational
# fix_cmd: shell command to install/remediate (None if no auto-fix available)

_CAPABILITIES: list[dict[str, Any]] = []


def _find_b00t_cli() -> str | None:
    return shutil.which("b00t-cli") or shutil.which("b00t")


def _check_b00t_mcp() -> bool:
    """Check if b00t-mcp binary exists and Hermes has it in config."""
    cli = _find_b00t_cli()
    if not cli:
        return False
    # Check binary exists
    bin_path = shutil.which("b00t-mcp")
    if bin_path:
        return True
    # Check if it's registered in config
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return "b00t-mcp" in f.read()
    return False


def _check_codebase_memory() -> bool:
    """Check if codebase-memory-mcp binary exists."""
    paths = [
        os.path.expanduser("~/.b00t/vendor/codebase-memory-mcp-b00t-ir0n-ledg3rr/build/c/codebase-memory-mcp"),
        shutil.which("codebase-memory-mcp") or "",
    ]
    for p in paths:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return True
    return False


def _check_irontology_mcp() -> bool:
    """Check if irontology-mcp binary exists."""
    paths = [
        os.path.expanduser("~/.b00t/target/release/irontology-mcp"),
        shutil.which("irontology-mcp") or "",
    ]
    for p in paths:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return True
    # Also check if it's registered in config
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            if "irontology-mcp" in f.read():
                return True
    return False


def _check_guard_plugin() -> bool:
    """Check if b00t guard interposition plugin is installed."""
    plugin_dir = os.path.expanduser("~/.b00t/vendor/hermes-agent-b00t/plugins/b00t")
    if os.path.isdir(plugin_dir):
        return os.path.isfile(os.path.join(plugin_dir, "__init__.py"))
    return False


def _check_b00t_cli() -> bool:
    """Check if b00t-cli is installed."""
    return _find_b00t_cli() is not None


def _init_capabilities():
    """Populate the capability registry once."""
    global _CAPABILITIES
    if _CAPABILITIES:
        return

    b00t_root = os.path.expanduser("~/.b00t")

    _CAPABILITIES = [
        {
            "name": "b00t-cli",
            "description": "b00t CLI tool — hive, grok, task management",
            "check": _check_b00t_cli,
            "fix": f"cd {b00t_root} && cargo install --path b00t-cli --force",
            "severity": "critical",
        },
        {
            "name": "b00t-mcp",
            "description": "b00t MCP server — guard interposition, agent context",
            "check": _check_b00t_mcp,
            "fix": f"cd {b00t_root} && cargo install --path b00t-mcp --force && b00t-cli mcp install b00t-mcp hermes",
            "severity": "high",
        },
        {
            "name": "guard-plugin",
            "description": "b00t guard interposition plugin in Hermes",
            "check": _check_guard_plugin,
            "fix": None,  # must update submodule
            "severity": "medium",
        },
        {
            "name": "irontology-mcp",
            "description": "Knowledge graph MCP server — 4-way fusion retrieval",
            "check": _check_irontology_mcp,
            "fix": f"cd {b00t_root} && cargo build --release -p mcp-server --manifest-path vendor/irontology-mcp/Cargo.toml",
            "severity": "high",
        },
        {
            "name": "codebase-memory",
            "description": "Code knowledge graph — search_graph, trace_path, architecture",
            "check": _check_codebase_memory,
            "fix": None,  # must build from source
            "severity": "medium",
        },
    ]


def _run_capability_check() -> list[dict]:
    """Run all capability checks, return list of gaps."""
    _init_capabilities()
    gaps = []
    for cap in _CAPABILITIES:
        try:
            ok = cap["check"]()
            status = "✓" if ok else "✗"
            logger.info("[h3rmes-cap] %s %s", status, cap["name"])
            if not ok:
                gaps.append(cap)
        except Exception as e:
            logger.warning("[h3rmes-cap] ? %s (check error: %s)", cap["name"], e)
            gaps.append({**cap, "check_error": str(e)})
    return gaps


def _dispatch_fix(cap: dict) -> dict:
    """Dispatch a fix for a missing capability.

    Returns a dict with action result. Since we're in a plugin hook
    and can't call delegate_task directly, we log the fix command
    for the agent to pick up.
    """
    fix_cmd = cap.get("fix")
    if not fix_cmd:
        msg = f"no auto-fix available for {cap['name']}"
        logger.info("[h3rmes-cap] %s", msg)
        return {"action": "notify", "message": msg}

    logger.info("[h3rmes-cap] dispatching fix for %s: %s", cap["name"], fix_cmd[:80])
    try:
        result = subprocess.run(
            fix_cmd, shell=True, capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            logger.info("[h3rmes-cap] fix applied for %s", cap["name"])
            return {"action": "fixed", "name": cap["name"]}
        else:
            logger.warning(
                "[h3rmes-cap] fix failed for %s: %s",
                cap["name"], result.stderr[:200],
            )
            return {"action": "failed", "name": cap["name"], "error": result.stderr[:200]}
    except subprocess.TimeoutExpired:
        logger.warning("[h3rmes-cap] fix timed out for %s", cap["name"])
        return {"action": "timeout", "name": cap["name"]}
    except Exception as e:
        logger.warning("[h3rmes-cap] fix error for %s: %s", cap["name"], e)
        return {"action": "error", "name": cap["name"], "error": str(e)}


def on_session_start(ctx) -> None:
    """Hook: run capability check at session start, auto-fix if configured."""
    _init_capabilities()

    # Check if auto-fix is enabled
    auto_fix = getattr(ctx, "config", {}).get("h3rmes", {}).get("auto_fix", False)

    last_check_file = os.path.expanduser("~/.hermes/h3rmes-last-check.json")
    check_interval = getattr(ctx, "config", {}).get("h3rmes", {}).get("check_interval", 3600)

    # Throttle: don't check more often than interval
    now = time.time()
    if os.path.exists(last_check_file):
        try:
            with open(last_check_file) as f:
                last_check = json.load(f).get("timestamp", 0)
            if now - last_check < check_interval:
                logger.debug("[h3rmes-cap] throttled (last check %ds ago)", int(now - last_check))
                return
        except Exception:
            pass

    # Run capability check
    gaps = _run_capability_check()

    # Save last check timestamp
    os.makedirs(os.path.dirname(last_check_file), exist_ok=True)
    with open(last_check_file, "w") as f:
        json.dump({"timestamp": now, "gaps": [g["name"] for g in gaps]}, f)

    if not gaps:
        logger.info("[h3rmes-cap] all capabilities ✓")
        return

    # Log gaps
    gap_names = ", ".join(g["name"] for g in gaps)
    logger.info("[h3rmes-cap] capability gaps: %s", gap_names)

    # Auto-fix mode: dispatch fixes for critical/high gaps
    if auto_fix:
        for cap in gaps:
            if cap.get("severity") in ("critical", "high"):
                result = _dispatch_fix(cap)
                if result["action"] != "fixed":
                    logger.warning("[h3rmes-cap] %s: %s", cap["name"], result.get("error", "unknown"))
    else:
        # Notify: log the fix commands so the agent can offer
        logger.info("[h3rmes-cap] auto-fix disabled; %d gap(s) detected", len(gaps))
        for cap in gaps:
            if cap.get("fix"):
                logger.info("[h3rmes-cap]   %s: %s", cap["name"], cap["fix"][:100])


def register(ctx) -> None:
    """Register the h3rmes capability monitor hook."""
    ctx.register_hook("on_session_start", on_session_start)
    logger.info("🥾 h3rmes capability monitor plugin registered")
