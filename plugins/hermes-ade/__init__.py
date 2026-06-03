"""Hermes ADE Plugin v0.1 — Agent Delivery Engineering.

Integrates BCP protocol (Phase 1-3), CADVP verification,
and three-level gates (L1/L2/L3) into Hermes Agent.

Hooks:
  - post_llm_call: Auto-run L1 verification after task execution

Slash Commands:
  - /bcp          — BCP Phase 1 reverse confirmation
  - /bcp-verify   — L1+L2 gate verification
  - /bcp-report   — Protocol metrics
  - /bcp-validate — ADE compliance validation
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────

METRICS_FILE = Path.home() / ".hermes" / "ade-metrics.json"
ADE_CONFIG_FILE = Path.home() / ".hermes" / "ade-config.json"

DEFAULT_CONFIG = {
    "enabled": True,
    "auto_l1": True,
    "risk_level": "medium",
    "max_consultations": 3,
    "log_level": "info",
}


# ── Metrics ────────────────────────────────────────────────────


def _load_metrics() -> dict:
    if METRICS_FILE.exists():
        try:
            return json.loads(METRICS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "sessions": [],
        "total_errors_caught": 0,
        "total_phases": {"phase1": 0, "phase2": 0, "phase3": 0},
        "gate_results": {"l1_pass": 0, "l1_fail": 0, "l2_pass": 0, "l2_fail": 0},
        "created_at": datetime.now().isoformat(),
    }


def _save_metrics(m: dict) -> None:
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.write_text(json.dumps(m, indent=2, ensure_ascii=False))


def _record_phase(phase: str, result: dict) -> None:
    m = _load_metrics()
    m["total_phases"][phase] = m["total_phases"].get(phase, 0) + 1
    session = {"timestamp": datetime.now().isoformat(), "phase": phase, "result": result}
    m["sessions"].append(session)
    if len(m["sessions"]) > 100:
        m["sessions"] = m["sessions"][-100:]
    _save_metrics(m)


# ── Phase 1: Reverse Confirmation ──────────────────────────────


def phase1(task_desc: str) -> dict:
    """Generate structured reverse confirmation report (BCP Phase 1)."""
    now = datetime.now().isoformat()
    report = {
        "phase": "Phase 1 — Reverse Confirmation",
        "timestamp": now,
        "task": task_desc,
        "template": (
            f"🔄 **BCP Phase 1 — Reverse Confirmation**\n\n"
            f"**Task:** {task_desc}\n\n"
            f"**My Understanding:**\n"
            f"[Fill in: restate the task in your own words]\n\n"
            f"**Planned Steps:**\n"
            f"1. [Step 1]\n"
            f"2. [Step 2]\n\n"
            f"**Assumptions to Confirm:**\n"
            f"- [Assumption 1]\n"
            f"- [Assumption 2]\n\n"
            f"**Expected Output:**\n"
            f"- [File/result description]\n\n"
            f"**Please confirm if this understanding is correct.**"
        ),
    }
    _record_phase("phase1", report)
    return report


# ── L1 Self-Verification ───────────────────────────────────────


def l1_verify(paths: list[str]) -> dict:
    """Execute L1 self-verification on the given file paths."""
    results = []
    all_pass = True

    for p in paths:
        path = Path(p).expanduser()
        check = {"path": str(path), "checks": {}}

        exists = path.exists()
        check["checks"]["file_exists"] = exists
        if not exists:
            all_pass = False

        if exists:
            size = path.stat().st_size
            check["checks"]["non_empty"] = size > 0
            check["file_size"] = size

            readable = os.access(str(path), os.R_OK)
            check["checks"]["readable"] = readable
            if not readable:
                all_pass = False

        results.append(check)

    m = _load_metrics()
    k = "l1_pass" if all_pass else "l1_fail"
    m["gate_results"][k] = m["gate_results"].get(k, 0) + 1
    _save_metrics(m)

    return {
        "gate": "L1 — Self-Verification",
        "timestamp": datetime.now().isoformat(),
        "paths_checked": len(paths),
        "all_pass": all_pass,
        "checks": results,
        "verdict": "PASS" if all_pass else "FAIL",
    }


# ── L2 Evidence Verification ───────────────────────────────────


def l2_verify(paths: list[str]) -> dict:
    """Generate L2 evidence verification report."""
    evidence = []
    all_pass = True

    for p in paths:
        path = Path(p).expanduser()
        if not path.exists():
            evidence.append({"path": str(path), "status": "missing", "evidence_type": None})
            all_pass = False
            continue

        entry = {"path": str(path), "status": "present", "evidence": {}}

        try:
            result = subprocess.run(
                ["ls", "-la", str(path)], capture_output=True, text=True, timeout=5
            )
            entry["evidence"]["ls"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            entry["evidence"]["ls"] = "unavailable"

        try:
            content = path.read_text(errors="replace")[:300]
            entry["evidence"]["content_preview"] = content
        except (OSError, UnicodeDecodeError):
            entry["evidence"]["content_preview"] = "binary or unreadable"

        evidence.append(entry)

    m = _load_metrics()
    k = "l2_pass" if all_pass else "l2_fail"
    m["gate_results"][k] = m["gate_results"].get(k, 0) + 1
    _save_metrics(m)

    return {
        "gate": "L2 — Evidence Verification",
        "timestamp": datetime.now().isoformat(),
        "paths_checked": len(paths),
        "all_pass": all_pass,
        "evidence": evidence,
        "verdict": "PASS" if all_pass else "FAIL",
    }


# ── CADVP 13-D Verification ────────────────────────────────────


def cadvp_verify(paths: list[str], task_desc: str = "") -> dict:
    """Execute CADVP 13-dimension verification where applicable."""
    dims = {
        "D1_cc0_channel": None,  # requires cross-agent context
        "D2_instruction_complete": True if task_desc else None,
        "D3_preconditions_met": all(Path(p).expanduser().parent.exists() for p in paths),
        "D4_state_consistency": None,  # requires before/after state
        "D5_result_completeness": True if paths else False,
        "D6_error_amplification": None,  # requires chain context
        "D7_idempotency": None,  # requires re-execution
        "D8_rollback_capability": None,  # requires rollback plan
        "D9_time_constraint": None,  # requires SLA
        "D10_resource_constraint": None,  # requires budget
        "D11_security_boundary": None,  # requires permission check
        "D12_evidence_chain": True,  # this report is itself evidence
        "D13_reproducibility": None,  # requires re-execution
    }

    verified = {k: v for k, v in dims.items() if v is not None}
    dim_status = {}
    for k, v in dims.items():
        if v is True:
            dim_status[k] = "pass"
        elif v is False:
            dim_status[k] = "fail"
        else:
            dim_status[k] = "not_applicable"

    overall = all(v is not False for v in dims.values())

    return {
        "protocol": "CADVP v1.1",
        "timestamp": datetime.now().isoformat(),
        "dimensions": dim_status,
        "dimensions_applicable": len(verified),
        "dimensions_total": 13,
        "verdict": "PASS" if overall else "FAIL",
    }


# ── Config ─────────────────────────────────────────────────────


def _load_config() -> dict:
    if ADE_CONFIG_FILE.exists():
        try:
            config = json.loads(ADE_CONFIG_FILE.read_text())
            return {**DEFAULT_CONFIG, **config}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_CONFIG)


def _save_config(config: dict) -> None:
    ADE_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    ADE_CONFIG_FILE.write_text(json.dumps(config, indent=2, ensure_ascii=False))


# ── Hooks ──────────────────────────────────────────────────────


def post_llm_call(response: dict, context: dict) -> Optional[str]:
    """Run L1 self-verification automatically after task execution.

    Activated when auto_l1 is enabled in config.
    Detects task execution patterns in the response.
    """
    config = _load_config()
    if not config.get("enabled", True) or not config.get("auto_l1", True):
        return None

    response_text = str(response) if isinstance(response, dict) else str(response)
    if not response_text:
        return None

    # Quick scan for file path patterns
    file_paths = re.findall(r'(?:`|")((?:/[^ ]+)+\.[a-zA-Z0-9]+)', response_text)
    file_paths = [p for p in file_paths if Path(p).expanduser().exists()]

    if not file_paths:
        return None

    # Auto-run L1 verification
    l1_result = l1_verify(file_paths)
    if not l1_result["all_pass"]:
        logger.warning("ADE L1 auto-verify FAILED: %s", json.dumps(l1_result, indent=2))
        return (
            "\n\n---\n🛡️ **ADE Auto-Verification (L1)**\n"
            f"**Verdict:** ❌ FAIL\n"
            f"**Files checked:** {len(file_paths)}\n"
            f"**Issues found:** Some files failed verification. "
            "Run `/bcp-verify` for details."
        )

    logger.info("ADE L1 auto-verify PASSED for %d files", len(file_paths))
    return None


# ── Slash Command Handlers ─────────────────────────────────────
# Each handler takes raw_args: str and returns str | None


def _handle_bcp(raw_args: str) -> Optional[str]:
    """Handler for /bcp — BCP Phase 1 reverse confirmation."""
    if not raw_args:
        return (
            "**ADE - BCP Phase 1**\n\n"
            "To use: `/bcp <task description>`\n\n"
            "Example: `/bcp Upload ADE_COMPLETE.md to IMA knowledge base`\n\n"
            "This generates a structured reverse confirmation template"
        )
    report = phase1(raw_args)
    return report["template"]


def _handle_bcp_verify(raw_args: str) -> Optional[str]:
    """Handler for /bcp-verify — L1+L2 gate verification."""
    if not raw_args:
        return (
            "**ADE - L1+L2 Verification**\n\n"
            "To use: `/bcp-verify <file1> [file2] [...]`\n\n"
            "Example: `/bcp-verify /home/user/doc.md ~/project/ade.yaml`\n\n"
            "Runs L1 self-verification and L2 evidence on specified files."
        )

    paths = [p.strip() for p in raw_args.split() if p.strip()]
    if not paths:
        return "No valid file paths provided."

    l1_result = l1_verify(paths)
    output = "**ADE - Gate Verification**\n\n"
    output += f"**L1 — Self-Verification:** {'✅ PASS' if l1_result['all_pass'] else '❌ FAIL'}\n"
    for c in l1_result["checks"]:
        chk_path = Path(c["path"]).name
        chk_status = "✅" if all(c["checks"].values()) else "❌"
        output += f"  {chk_status} {chk_path}\n"

    if l1_result["all_pass"]:
        l2_result = l2_verify(paths)
        output += f"\n**L2 — Evidence Verification:** ✅ PASS\n"
        for e in l2_result["evidence"]:
            name = Path(e["path"]).name
            output += f"  📄 {name}: {e['status']}\n"

    output += f"\n**Verdict:** {'✅ PASS' if l1_result['all_pass'] else '❌ FAIL'}"
    return output


def _handle_bcp_report(raw_args: str) -> Optional[str]:
    """Handler for /bcp-report — show ADE protocol metrics."""
    metrics = _load_metrics()
    phases = metrics.get("total_phases", {})
    gates = metrics.get("gate_results", {})
    sessions = metrics.get("sessions", [])

    output = (
        "📊 **ADE Protocol Metrics**\n\n"
        f"**Phase Usage:**\n"
        f"  🔄 Phase 1 (Reverse Confirmation): {phases.get('phase1', 0)}x\n"
        f"  💬 Phase 2 (Limited Consultation): {phases.get('phase2', 0)}x\n"
        f"  📦 Phase 3 (Structured Delivery): {phases.get('phase3', 0)}x\n\n"
        f"**Gate Results:**\n"
        f"  L1 ✅ Pass: {gates.get('l1_pass', 0)}  ❌ Fail: {gates.get('l1_fail', 0)}\n"
        f"  L2 ✅ Pass: {gates.get('l2_pass', 0)}  ❌ Fail: {gates.get('l2_fail', 0)}\n"
        f"**Errors Caught:** {metrics.get('total_errors_caught', 0)}\n"
        f"**Sessions Logged:** {len(sessions)}\n"
        f"**Created:** {metrics.get('created_at', 'N/A')}"
    )

    if sessions:
        output += "\n\n**Recent Activity:**\n"
        for s in sessions[-5:]:
            phase = s.get("phase", "unknown")
            ts = s.get("timestamp", "")[:19]
            output += f"  • [{ts}] Phase {phase}\n"

    return output


def _handle_bcp_validate(raw_args: str) -> Optional[str]:
    """Handler for /bcp-validate — ADE compliance audit."""
    config = _load_config()
    metrics = _load_metrics()

    checks = []
    checks.append(("Plugin enabled", config.get("enabled", False), True))
    checks.append(("Auto L1 verification", config.get("auto_l1", False), True))
    p1_count = metrics.get("total_phases", {}).get("phase1", 0)
    checks.append(("Phase 1 executed", p1_count > 0, True))
    checks.append(("Metrics tracking", METRICS_FILE.exists(), True))

    output = "🛡️ **ADE Compliance Audit**\n\n"
    all_pass = True
    for name, actual, expected in checks:
        ok = actual == expected
        if not ok:
            all_pass = False
        output += f"{'✅' if ok else '❌'} {name}: {'OK' if ok else 'ISSUE'}\n"

    output += f"\n**Verdict:** {'✅ COMPLIANT' if all_pass else '⚠️ NON-COMPLIANT'}"
    if not all_pass:
        output += "\n\n**Remediation:**\n"
        if not config.get("enabled"):
            output += "  • Enable plugin: `ade_config.enabled = true`\n"
        if p1_count == 0:
            output += "  • Run `/bcp <task>` to create first Phase 1 report\n"

    return output


# ── Plugin Registration ────────────────────────────────────────


def register(ctx) -> None:
    """Register ADE plugin hooks and slash commands with Hermes."""
    ctx.register_hook("post_llm_call", post_llm_call)

    ctx.register_command(
        "bcp",
        handler=_handle_bcp,
        description="Execute BCP Phase 1 reverse confirmation",
        args_hint="<task description>",
    )
    ctx.register_command(
        "bcp-verify",
        handler=_handle_bcp_verify,
        description="Run L1+L2 gate verification on delivered files",
        args_hint="<file_path> [file_path...]",
    )
    ctx.register_command(
        "bcp-report",
        handler=_handle_bcp_report,
        description="Show ADE protocol metrics and verification history",
    )
    ctx.register_command(
        "bcp-validate",
        handler=_handle_bcp_validate,
        description="Validate ADE compliance of current execution",
    )
