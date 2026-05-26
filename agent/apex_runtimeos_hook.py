"""APEX RuntimeOS integration hook for Hermes conversation completion.

This module is intentionally defensive: if the external APEX RuntimeOS workspace
is absent or broken, Hermes must still complete the user turn. The hook attaches
machine-readable metadata to the run_conversation result and writes a best-effort
report artifact when enabled.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from agent.apex_runtimeos_organs import run_pre_completion_organs
from agent.apex_runtimeos_audit import persist_checkpoint
from agent.apex_runtimeos_autonomy import persist_autowrite_candidate

logger = None

APEX_RUNTIMEOS_DIR = Path(
    os.environ.get(
        "APEX_RUNTIMEOS_DIR",
        "/Users/appleoppa/.hermes/workspace/github/z-dashen/apex/apex-spiral/apex_runtimeos",
    )
)


def _env_flag(name: str, default: str = "1") -> bool:
    return str(os.environ.get(name, default)).strip().lower() not in {"0", "false", "no", "off", ""}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses and some libraries require module registration during exec.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_text(value: Any, max_len: int = 500) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def run_apex_runtimeos_completion_hook(
    *,
    result: Dict[str, Any],
    final_response: Optional[str],
    completed: bool,
    interrupted: bool,
    agent: Any,
    messages: list,
    turn_exit_reason: str,
) -> Dict[str, Any]:
    """Attach APEX RuntimeOS completion-gate metadata to a conversation result.

    Default behavior is enabled in report/warn mode but non-blocking. Set
    ``APEX_RUNTIMEOS_GATE_ENABLED=0`` to disable. Set
    ``APEX_RUNTIMEOS_GATE_MODE=enforce`` to make gate failure mark the result
    incomplete; enforce is deliberately opt-in.
    """
    if not _env_flag("APEX_RUNTIMEOS_GATE_ENABLED", "1"):
        result["apex_runtimeos"] = {"enabled": False, "status": "SKIPPED"}
        return result

    mode = os.environ.get("APEX_RUNTIMEOS_GATE_MODE", "warn").strip().lower() or "warn"
    if mode not in {"dry_run", "warn", "enforce"}:
        mode = "warn"

    metadata: Dict[str, Any] = {
        "enabled": True,
        "mode": mode,
        "status": "UNKNOWN",
        "runtime_hook_enabled": True,
        "blocking": False,
        "apex_runtimeos_dir": str(APEX_RUNTIMEOS_DIR),
    }
    try:
        if not APEX_RUNTIMEOS_DIR.exists():
            raise FileNotFoundError(str(APEX_RUNTIMEOS_DIR))
        dry = _load_module(
            "apex_runtimeos_completion_report_dry_run",
            APEX_RUNTIMEOS_DIR / "completion_report_dry_run.py",
        )
        runtime_context = {
            "session_id": getattr(agent, "session_id", None),
            "model": getattr(agent, "model", None),
            "provider": getattr(agent, "provider", None),
            "completed": completed,
            "interrupted": interrupted,
            "turn_exit_reason": turn_exit_reason,
            "message_count": len(messages or []),
            "final_response_present": bool(final_response),
        }
        organ_results = run_pre_completion_organs(runtime_context)
        audit_result = persist_checkpoint(
            "pre_completion",
            organ_results,
            session_id=str(getattr(agent, "session_id", "") or ""),
        )
        autowrite_result = persist_autowrite_candidate(
            stage="pre_completion",
            session_id=str(getattr(agent, "session_id", "") or ""),
            recommendations=organ_results.get("recommendations", {}),
        )
        report = dry.build_report(
            dry_run=False,
            runtime_hook_enabled=True,
            mode=mode,
            context=runtime_context,
        )
        status = report.get("status", "UNKNOWN")
        metadata.update(
            {
                "status": status,
                "report_type": report.get("report_type"),
                "dry_run": report.get("dry_run"),
                "runtime_context_recorded": report.get("runtime_context", {}).get("provided"),
                "organs": organ_results,
                "organ_audit": audit_result,
                "evm_defect_count": report.get("evm_gate", {}).get("defect_count"),
                "evm_gate_score": report.get("evm_gate", {}).get("evm_gate_score"),
                "full_agi_claimed": report.get("completion_boundary", {}).get("full_agi_claimed"),
                "autonomous_core_rewrite_enabled": report.get("completion_boundary", {}).get("autonomous_core_rewrite_enabled"),
                "production_runtime_governance_claimed": report.get("completion_boundary", {}).get("production_runtime_governance_claimed"),
                "recommendation_gate": organ_results.get("recommendations", {}).get("status"),
                "recommendation_control": organ_results.get("recommendations", {}),
                "autowrite": autowrite_result,
                "errors": report.get("gate_summary", {}).get("errors", []),
            }
        )

        if organ_results.get("blocking") and mode == "enforce":
            metadata["blocking"] = True
            result["completed"] = False
            result["failed"] = True
            result["error"] = "APEX RuntimeOS organ checkpoint blocked completion in enforce mode"

        output_path = os.environ.get("APEX_RUNTIMEOS_REPORT_OUTPUT", "").strip()
        if output_path:
            out = Path(output_path).expanduser()
        else:
            base = Path(os.environ.get("HERMES_APEX_RUNTIMEOS_REPORT_DIR", str(Path.home() / ".hermes" / "apex_runtimeos_reports")))
            out = base / f"{getattr(agent, 'session_id', 'no-session') or 'no-session'}_{int(__import__('time').time())}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        metadata["report_path"] = str(out)

        if status != "PASS" and mode == "enforce":
            metadata["blocking"] = True
            result["completed"] = False
            result["failed"] = True
            result["error"] = "APEX RuntimeOS gate failed in enforce mode"
        result["apex_runtimeos"] = metadata
        return result
    except Exception as exc:  # never crash the main conversation in warn/dry_run
        metadata.update({"status": "ERROR", "error": _safe_text(exc)})
        if mode == "enforce":
            metadata["blocking"] = True
            result["completed"] = False
            result["failed"] = True
            result["error"] = f"APEX RuntimeOS hook error in enforce mode: {_safe_text(exc)}"
        result["apex_runtimeos"] = metadata
        return result


__all__ = ["run_apex_runtimeos_completion_hook"]
