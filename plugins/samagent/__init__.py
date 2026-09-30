"""SamAgent Hermes Plugin (plugins/samagent/__init__.py).

Wires the SamAgent library (samagent/*) into the unmodified Hermes runtime via hooks,
slash commands, and plugin tools:
- pre_tool_call: Layer-1 module ownership guard + weak-model patch argument validator
- pre_llm_call: Ephemeral <=2K-token Ledger memory injection into user turn (prompt-cache safe)
- pre_verify: L0–L4 verification & security gate before turn completion
- /samagent slash command & samagent_pipeline tool
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from samagent.conductor.engine import SamAgentConductor, build_plan_card
from samagent.conductor.verify import VerificationRunner
from samagent.contract.freeze import check_path_ownership, load_ownership_map
from samagent.ledger.store import ProjectLedger
from samagent.router.policy import DEFAULT_MODEL_PROFILES, TaskBoundaryRouter
from samagent.spec.interview import generate_interview_questions
from samagent.spec.models import SpecDocument

logger = logging.getLogger(__name__)

# Active task -> module mapping for in-process delegated workers
_TASK_MODULE_MAP: Dict[str, str] = {}


def set_task_module_owner(task_id: str, module_name: str) -> None:
    """Bind a delegated worker's task_id to its owned module name for pre_tool_call enforcement."""
    _TASK_MODULE_MAP[task_id] = module_name


def clear_task_module_owner(task_id: str) -> None:
    _TASK_MODULE_MAP.pop(task_id, None)


def _active_project_dir() -> Path:
    cwd = os.environ.get("SAMAGENT_PROJECT_DIR") or os.environ.get("TERMINAL_CWD") or os.getcwd()
    return Path(cwd)


def _is_cloud_model(model_name: str) -> bool:
    if not model_name:
        return False
    prof = DEFAULT_MODEL_PROFILES.get(model_name)
    if prof is not None:
        return not prof.is_local
    lower = model_name.lower()
    return not any(k in lower for k in ("local", "llama", "gguf", "ollama", "qwen3.8", "deepseek-v4-flash"))


def _on_pre_tool_call(
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    task_id: str = "",
    session_id: str = "",
    **_: Any,
) -> Optional[Dict[str, Any]]:
    """1. Validate patch arguments for weaker local models.
    2. Enforce Layer-1 module ownership guard when .samagent/contract/ownership.yaml is active.
    """
    if not isinstance(args, dict):
        return None

    proj_dir = _active_project_dir()

    # Guard A: Weak-model patch validator
    if tool_name == "patch":
        target = args.get("path")
        old_text = args.get("old_text")
        if isinstance(target, str) and target.strip() and isinstance(old_text, str) and old_text:
            candidate = Path(target) if Path(target).is_absolute() else (proj_dir / target)
            if candidate.exists() and candidate.is_file():
                try:
                    content = candidate.read_text(encoding="utf-8")
                    if old_text not in content:
                        return {
                            "action": "block",
                            "message": (
                                f"SamAgent patch guard: `old_text` was not found in '{target}'. "
                                "Call `read_file` on the target lines before retrying `patch`."
                            ),
                        }
                except Exception:
                    pass

    # Guard B: Layer-1 Ownership Guard on write_file / patch
    if tool_name in ("write_file", "patch"):
        target_path = args.get("path")
        if not isinstance(target_path, str) or not target_path.strip():
            return None
        module_name = _TASK_MODULE_MAP.get(task_id) or os.environ.get("SAMAGENT_WORKER_MODULE")
        if not module_name:
            return None
        ownership_map = load_ownership_map(proj_dir)
        if not ownership_map:
            return None
        verdict = check_path_ownership(
            target_path,
            module_name=module_name,
            ownership_map=ownership_map,
            project_dir=proj_dir,
        )
        if not verdict.allowed:
            return {"action": "block", "message": f"SamAgent ownership guard: {verdict.reason}"}

    return None


def _on_pre_llm_call(
    session_id: str = "",
    task_id: str = "",
    user_message: Any = "",
    model: str = "",
    **_: Any,
) -> Optional[Dict[str, str]]:
    """Inject <=2K-token Ledger context into the user message (cache-safe via turn_context.py)."""
    proj_dir = _active_project_dir()
    db_path = proj_dir / ".samagent" / "ledger.db"
    if not db_path.exists():
        return None
    try:
        ledger = ProjectLedger(proj_dir)
        query_str = user_message if isinstance(user_message, str) else str(user_message or "")
        block = ledger.build_turn_context_block(
            query_str,
            is_cloud_route=_is_cloud_model(model),
            max_tokens=2000,
        )
        if block:
            return {"context": block}
    except Exception as exc:
        logger.debug("SamAgent pre_llm_call ledger injection skipped: %s", exc)
    return None


def _on_pre_verify(
    session_id: str = "",
    model: str = "",
    coding: bool = False,
    attempt: int = 0,
    final_response: str = "",
    changed_paths: Optional[List[str]] = None,
    **_: Any,
) -> Optional[Dict[str, str]]:
    """Run L0–L4 verification when .samagent/spec.yaml is present and code was edited."""
    proj_dir = _active_project_dir()
    spec_path = proj_dir / ".samagent" / "spec.yaml"
    if not spec_path.exists() or not changed_paths:
        return None
    try:
        spec = SpecDocument.load(proj_dir)
        runner = VerificationRunner(proj_dir, spec)
        report = runner.run_all()
        return report.to_pre_verify_hook_response()
    except Exception as exc:
        logger.debug("SamAgent pre_verify check skipped: %s", exc)
        return None


SAMAGENT_PIPELINE_SCHEMA = {
    "name": "samagent_pipeline",
    "description": (
        "Run SamAgent spec-first pipeline actions: 'interview' (generate <=5 adaptive questions), "
        "'plan' (freeze spec, contract, red-first acceptance tests, and return Plan Card), "
        "'build' (execute deterministic scaffold + module implementation + L0–L4 verification), "
        "'verify' (run L0–L4 verification & security probes), or 'ledger' (query/record project facts)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["interview", "plan", "build", "verify", "ledger"],
                "description": "Pipeline action to execute.",
            },
            "brief": {
                "type": "string",
                "description": "User project brief or change request (for interview/plan/build).",
            },
            "answers": {
                "type": "object",
                "description": "Optional answers to interview questions (omitted questions use defaults).",
            },
            "project_dir": {
                "type": "string",
                "description": "Target project directory (defaults to current working directory).",
            },
        },
        "required": ["action"],
    },
}


def handle_samagent_pipeline(args: Dict[str, Any], **_: Any) -> str:
    action = str(args.get("action", "plan")).strip().lower()
    proj_dir = Path(args.get("project_dir") or _active_project_dir())
    proj_dir.mkdir(parents=True, exist_ok=True)
    brief = str(args.get("brief") or "Web application").strip()
    answers = args.get("answers") if isinstance(args.get("answers"), dict) else None

    if action == "interview":
        qs = [q.to_dict() for q in generate_interview_questions(brief)]
        return json.dumps({"action": "interview", "questions": qs}, indent=2)

    conductor = SamAgentConductor(proj_dir)
    if action == "plan":
        res = conductor.prepare_spec_and_contract(brief, answers=answers)
        return json.dumps(res, indent=2)

    if action == "build":
        if not (proj_dir / ".samagent" / "spec.yaml").exists():
            conductor.prepare_spec_and_contract(brief, answers=answers)
        res = conductor.execute_and_verify()
        return json.dumps(res, indent=2)

    if action == "verify":
        spec = SpecDocument.load(proj_dir)
        rep = VerificationRunner(proj_dir, spec).run_all()
        return json.dumps(rep.to_dict(), indent=2)

    if action == "ledger":
        ledger = ProjectLedger(proj_dir)
        facts = [f.to_dict() for f in ledger.list_facts(active_only=True)]
        return json.dumps({"facts": facts}, indent=2)

    return json.dumps({"error": f"Unknown action: {action}"})


def _handle_slash(raw_args: str) -> str:
    parts = (raw_args or "").strip().split(maxsplit=1)
    sub = parts[0].lower() if parts else "help"
    arg = parts[1] if len(parts) > 1 else ""
    proj_dir = _active_project_dir()

    if sub in ("help", "-h", "--help"):
        return (
            "/samagent commands:\n"
            "  /samagent interview <brief>  — Generate <=5 adaptive questions with defaults\n"
            "  /samagent plan <brief>       — Freeze spec, contract, red-first tests & Plan Card\n"
            "  /samagent build [brief]      — Run scaffold, workers, and L0–L4 verification\n"
            "  /samagent verify             — Run L0–L4 verification & security probes\n"
            "  /samagent ledger             — Show active project ledger facts\n"
        )
    if sub == "interview":
        qs = generate_interview_questions(arg or "Web application")
        return "\n".join([f"- {q.id}: {q.question} (default: {q.recommended_id})" for q in qs])
    if sub == "plan":
        res = SamAgentConductor(proj_dir).prepare_spec_and_contract(arg or "Web application")
        pc = res["plan_card"]
        return (
            f"Spec & Contract Frozen for: {pc['goal']}\n"
            f"Red-first check: {res['red_first_check']['is_red_for_right_reason']}\n"
            f"Estimated cost: ${pc['estimated_cost_usd_range'][0]:.2f}–${pc['estimated_cost_usd_range'][1]:.2f}\n"
            f"Local task share: {pc['local_task_share_pct']}%"
        )
    if sub == "build":
        cond = SamAgentConductor(proj_dir)
        if not (proj_dir / ".samagent" / "spec.yaml").exists():
            cond.prepare_spec_and_contract(arg or "Web application")
        out = cond.execute_and_verify()
        return f"Build status: {out['status']} | Verification L0–L4 passed: {out['verification']['passed']}"
    if sub == "verify":
        spec = SpecDocument.load(proj_dir)
        rep = VerificationRunner(proj_dir, spec).run_all()
        return f"Verification passed={rep.passed} | Layers: " + ", ".join(f"{l.level}={l.passed}" for l in rep.layers)
    if sub == "ledger":
        facts = ProjectLedger(proj_dir).list_facts(active_only=True)
        return "\n".join(f"- [{f.scope}/{f.kind}] {f.id}: {f.text}" for f in facts) or "No active facts."
    return f"Unknown /samagent subcommand: {sub}. Run `/samagent help`."


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("pre_verify", _on_pre_verify)
    ctx.register_command(
        "samagent",
        handler=_handle_slash,
        description="SamAgent spec-first, contract-gated pipeline and verification.",
    )
    ctx.register_tool(
        name="samagent_pipeline",
        toolset="samagent",
        schema=SAMAGENT_PIPELINE_SCHEMA,
        handler=handle_samagent_pipeline,
        description="Execute SamAgent interview, contract freeze, build, verification, and ledger actions.",
    )
