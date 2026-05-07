"""Registers the ``run_code`` tool with the Hermes tool registry.

``run_code`` is the single tool the model sees in CodeAct mode.  Its schema
embeds a compact catalogue of all other available tools so the model always
has a reference of what's callable from Python.

The actual handler is a lightweight shim — real execution is routed through
the HermesKernel instance owned by the AIAgent.  Because the kernel is
session-scoped (not module-scoped), the handler is injected at session start
via ``set_kernel_dispatcher()`` rather than at import time.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Callable, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# Session-scoped dispatchers set by AIAgent when it initialises CodeAct mode.
# Signature: (raw_tool_args: dict) -> str
_session_dispatchers: dict[str, Callable[[dict], str]] = {}
_dispatcher_lock = threading.RLock()


def set_kernel_dispatcher(
    session_id: str,
    fn: Optional[Callable[[dict], str]],
) -> None:
    """Register or clear the CodeAct dispatcher for one session.

    ``session_id`` is a stable agent-owned routing key, not necessarily the
    user-visible Hermes conversation id. This avoids the old process-global
    dispatcher where one active CodeAct session could overwrite another.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required when registering a CodeAct dispatcher")

    with _dispatcher_lock:
        if fn is None:
            _session_dispatchers.pop(session_id, None)
        else:
            _session_dispatchers[session_id] = fn


def _run_code_handler(args: dict, **_kwargs) -> str:
    """Tool handler: route to the current session's kernel dispatcher."""
    session_id = str(_kwargs.get("session_id", "") or "").strip()
    if not session_id:
        return tool_error(
            "CodeAct session routing missing. "
            "run_code requires a session-scoped dispatcher key."
        )

    with _dispatcher_lock:
        dispatcher = _session_dispatchers.get(session_id)

    if dispatcher is None:
        return tool_error(
            f"CodeAct kernel not initialised for session '{session_id}'. "
            "This tool requires codeact_mode to be enabled for the current model profile."
        )

    try:
        return dispatcher(args)
    except Exception as exc:
        logger.exception("run_code handler error: %s", exc)
        return tool_error(str(exc))


def _check_codeact_available() -> bool:
    """Availability check: True when at least one kernel dispatcher is registered."""
    with _dispatcher_lock:
        return bool(_session_dispatchers)


# Placeholder schema — replaced at session start by build_codeact_tool_schema().
# This keeps the tool registered in the registry (for toolset filtering,
# hermes doctor, etc.) even before a session begins.
_PLACEHOLDER_SCHEMA = {
    "name": "run_code",
    "description": (
        "Execute Python in a persistent interpreter.  "
        "Available when CodeAct mode is active for the current model profile."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "thoughts": {
                "type": "string",
                "description": "Your reasoning before writing the code (required)",
            },
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
        },
        "required": ["thoughts", "code"],
    },
}

registry.register(
    name="run_code",
    toolset="codeact",
    schema=_PLACEHOLDER_SCHEMA,
    handler=_run_code_handler,
    check_fn=_check_codeact_available,
    description="Execute Python in a persistent CodeAct kernel",
    emoji="🐍",
    max_result_size_chars=50_000,
)


# ---------------------------------------------------------------------------
# Schema builder (called by model_tools.build_codeact_tool_schema)
# ---------------------------------------------------------------------------

_ENVELOPE_INSTRUCTION = """\

FORMAT REQUIREMENT: Your response to this tool MUST be a JSON object with exactly \
two string fields:
  "thoughts" — your reasoning before writing the code (required)
  "code"     — the Python code to execute (required)

Example:
  {"thoughts": "I need source-grounded current evidence.",
   "code": "result = research_web(question='X', freshness='latest')"}

Do NOT output markdown fences, prose, or any text outside this JSON object.\
"""


def build_run_code_schema(
    compact_catalogue: str,
    envelope_mode: bool = False,
    workflow_guidance: str = "",
    recipe_catalogue: str = "",
) -> dict:
    """Return a full ``run_code`` tool schema with the compact catalogue embedded.

    Parameters
    ----------
    compact_catalogue:
        Output of ``codeact_namespace.build_compact_tool_catalogue()``.
    envelope_mode:
        When True, append an explicit format instruction telling the model it
        must output the JSON envelope.  Used for prompt-only enforcement (Phase 2).
        Grammar enforcement (Phase 3) makes this redundant but harmless.
    workflow_guidance:
        Compact task recipes generated from the enabled tool namespace.
    recipe_catalogue:
        Compact list of high-level CodeAct recipes available in the namespace.
    """
    workflow_block = (
        "Fast workflow hints:\n" + workflow_guidance.strip() + "\n\n"
        if workflow_guidance and workflow_guidance.strip()
        else ""
    )
    recipe_block = (
        "Core recipes:\n" + recipe_catalogue.strip() + "\n\n"
        if recipe_catalogue and recipe_catalogue.strip()
        else ""
    )
    description = (
        "Execute Python code in a persistent interpreter.\n"
        "• Variables defined here survive across turns — no need to recompute.\n"
        "• Call any Hermes tool as a plain Python function (no imports needed).\n"
        "• Call print(help()) for the full tool list, "
        "print(help('name')) for full parameter docs.\n"
        "• Prefer existing Hermes tools before environment probing or package installs.\n\n"
        "For search/research/report/latest/current/as-of-date tasks, the first "
        "code call should usually be:\n"
        "  result = research_web(question=USER_REQUEST, freshness='latest', "
        "depth='thorough', max_sources=8)\n"
        "For drug/clinical-trial/pharma tasks, use:\n"
        "  result = medical_pharma_research(question=USER_REQUEST)\n"
        "Do not start by debugging web_search, importing tools, inspecting "
        "sys.modules, or curling search engines. Final research reports must "
        "include citation metadata or a source table from the evidence bundle.\n\n"
        + workflow_block
        + recipe_block
        + "Available tools:\n"
        + compact_catalogue
        + "\n\n"
        "Call print(help('tool_name')) inside code to get full parameter documentation."
        + (_ENVELOPE_INSTRUCTION if envelope_mode else "")
    )
    return {
        "name": "run_code",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "thoughts": {
                    "type": "string",
                    "description": "Your reasoning before writing the code",
                },
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
            },
            "required": ["thoughts", "code"],
        },
    }
