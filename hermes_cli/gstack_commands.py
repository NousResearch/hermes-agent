"""gstack Persona Command Handlers

Handles all slash commands for the 7 gstack personas:
- /reviewer — Code quality and production safety
- /ceo-review — Strategic product fit
- /design-review — UX and visual design
- /eng-review — Architecture and tech debt
- /qa-audit — Testing and edge cases
- /cso — Security and compliance
- /release-check — Deployment safety

Each handler spawns a subagent with a curated system prompt and toolset.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from tools.gstack_personas import PersonaRole, PERSONA_DEFINITIONS, get_persona_system_prompt, get_persona_toolsets, get_persona_max_iterations
from tools.delegate_tool import delegate_task


def _ensure_review_dir() -> Path:
    """Ensure ~/.hermes/reviews directory exists."""
    review_dir = Path.home() / ".hermes" / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    return review_dir


def _save_review_report(persona_name: str, content: str) -> Path:
    """Save review report to ~/.hermes/reviews/<persona>_<timestamp>.md"""
    review_dir = _ensure_review_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{persona_name}_{timestamp}.md"
    filepath = review_dir / filename
    
    with open(filepath, "w") as f:
        f.write(content)
    
    return filepath


def _build_persona_task(persona_role: PersonaRole, target: str, context: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Build task description and context for a persona subagent."""
    persona_info = PERSONA_DEFINITIONS[persona_role]
    persona_name = persona_info["name"]
    persona_title = persona_info["title"]
    
    task = f"""Review the target: {target}

Use your designated tools to thoroughly examine the code/design/product.
Provide a detailed assessment following your standard review format."""
    
    return task, context


def handle_reviewer_command(target: str, context: Optional[str] = None, cli_obj=None) -> str:
    """Handle /reviewer <target> — Code quality and production safety review."""
    persona_role = PersonaRole.REVIEWER
    task, ctx = _build_persona_task(persona_role, target, context)
    
    # Delegate the review task to a subagent with the persona's system prompt and toolset
    parent_agent = cli_obj._agent if hasattr(cli_obj, '_agent') else None
    
    if not parent_agent:
        return "ERROR: No active agent context. Start a session first."
    
    # Call delegate_task with the persona's system prompt injected into context
    full_context = f"SYSTEM PROMPT:\n{get_persona_system_prompt(persona_role)}"
    if ctx:
        full_context += f"\n\nADDITIONAL CONTEXT:\n{ctx}"
    
    result_json = delegate_task(
        goal=task,
        context=full_context,
        toolsets=get_persona_toolsets(persona_role),
        max_iterations=get_persona_max_iterations(persona_role),
        parent_agent=parent_agent,
    )
    
    try:
        result = json.loads(result_json)
        if result.get("results"):
            report = result["results"][0].get("summary", str(result))
        else:
            report = str(result)
    except (json.JSONDecodeError, KeyError, IndexError):
        report = result_json
    
    report_path = _save_review_report("reviewer", report)
    
    return f"Code review complete. Report saved to {report_path}"


def _delegate_persona_review(persona_role: PersonaRole, target: str, context: Optional[str], cli_obj=None, report_name: str = "review") -> str:
    """Common implementation for all persona review handlers."""
    task, ctx = _build_persona_task(persona_role, target, context)
    
    # Delegate the review task to a subagent with the persona's system prompt and toolset
    parent_agent = cli_obj._agent if hasattr(cli_obj, '_agent') else None
    
    if not parent_agent:
        return "ERROR: No active agent context. Start a session first."
    
    # Call delegate_task with the persona's system prompt injected into context
    full_context = f"SYSTEM PROMPT:\n{get_persona_system_prompt(persona_role)}"
    if ctx:
        full_context += f"\n\nADDITIONAL CONTEXT:\n{ctx}"
    
    result_json = delegate_task(
        goal=task,
        context=full_context,
        toolsets=get_persona_toolsets(persona_role),
        max_iterations=get_persona_max_iterations(persona_role),
        parent_agent=parent_agent,
    )
    
    try:
        result = json.loads(result_json)
        if result.get("results"):
            report = result["results"][0].get("summary", str(result))
        else:
            report = str(result)
    except (json.JSONDecodeError, KeyError, IndexError):
        report = result_json
    
    report_path = _save_review_report(report_name, report)
    persona_info = PERSONA_DEFINITIONS[persona_role]
    persona_name = persona_info["name"]
    
    return f"{persona_name} review complete. Report saved to {report_path}"


def handle_ceo_review_command(target: str, context: Optional[str] = None, cli_obj=None) -> str:
    """Handle /ceo-review <target> — Strategic product fit and user value."""
    return _delegate_persona_review(PersonaRole.CEO, target, context, cli_obj, "ceo")


def handle_design_review_command(target: str, context: Optional[str] = None, cli_obj=None) -> str:
    """Handle /design-review <target> — UX, visual design, and accessibility."""
    return _delegate_persona_review(PersonaRole.DESIGNER, target, context, cli_obj, "designer")


def handle_eng_review_command(target: str, context: Optional[str] = None, cli_obj=None) -> str:
    """Handle /eng-review <target> — Architecture, tech debt, and patterns."""
    return _delegate_persona_review(PersonaRole.ENG_MANAGER, target, context, cli_obj, "eng_manager")


def handle_qa_audit_command(target: str, context: Optional[str] = None, cli_obj=None) -> str:
    """Handle /qa-audit <target> — Testing, edge cases, and user flows."""
    return _delegate_persona_review(PersonaRole.QA_LEAD, target, context, cli_obj, "qa_lead")


def handle_cso_command(target: str, context: Optional[str] = None, cli_obj=None) -> str:
    """Handle /cso <target> — Security, OWASP, and compliance."""
    return _delegate_persona_review(PersonaRole.CSO, target, context, cli_obj, "cso")


def handle_release_check_command(target: str, context: Optional[str] = None, cli_obj=None) -> str:
    """Handle /release-check <target> — Deployment safety and rollback plan."""
    return _delegate_persona_review(PersonaRole.RELEASE_ENGINEER, target, context, cli_obj, "release_engineer")


def handle_gstack_list_command(cli_obj=None) -> str:
    """Handle /gstack list — Show all available personas."""
    from tools.gstack_personas import list_personas
    
    personas = list_personas()
    output = ["Available gstack personas:\n"]
    
    for role_value, description in personas.items():
        output.append(f"  {description}")
    
    output.append("\nUsage: /{command} <target> [context]")
    output.append("\nCommands:")
    output.append("  /reviewer — Code quality and production safety")
    output.append("  /ceo-review — Strategic product fit and user value")
    output.append("  /design-review — UX, visual design, and accessibility")
    output.append("  /eng-review — Architecture, tech debt, and patterns")
    output.append("  /qa-audit — Testing, edge cases, and user flows")
    output.append("  /cso — Security, OWASP, and compliance")
    output.append("  /release-check — Deployment safety and rollback")
    
    return "\n".join(output)
