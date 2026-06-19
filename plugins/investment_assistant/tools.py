"""Hermes tool adapter for the investment assistant workflow."""

from __future__ import annotations

import os
from typing import Any

from tools.registry import tool_error, tool_result

from .output_guard import remember_output_guard, transform_llm_output
from .workflow import InvestmentAssistantWorkflow


IA_PORTFOLIO_WORKFLOW_SCHEMA = {
    "name": "ia_portfolio_workflow",
    "description": (
        "Run a stateful investment-assistant portfolio-map research workflow. Use this "
        "when the user asks to build or explore a theme portfolio layout such as "
        "'establish an AI/storage/power portfolio map'. start runs AI discovery v1, "
        "Futu lightweight enrichment, and candidate-triage strategy planning, then "
        "waits for user confirmation before final candidate triage. After candidate "
        "triage completes, build_portfolio_maps runs the PydanticAI portfolio "
        "architect over saved triage/Futu/offline filing artifacts and returns "
        "target portfolio-map drafts. revise_portfolio_map revises a selected or "
        "specified target map from user feedback, using a PydanticAI revision "
        "agent and waiting for user confirmation. discover is a diagnostic action that stops "
        "after theme discovery. The workflow never reads current holdings and does "
        "not generate orders or options."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "start",
                    "discover",
                    "answer_human_input",
                    "select_option",
                    "build_portfolio_maps",
                    "revise_portfolio_map",
                    "continue",
                    "status",
                    "cancel",
                ],
                "description": "Workflow action to perform.",
            },
            "session_id": {
                "type": "string",
                "description": "Existing workflow session id. Omit to resume the latest session for this tenant.",
            },
            "payload": {
                "type": "object",
                "description": (
                    "Action payload. start accepts free-form themes in "
                    "theme/theme_key/topic and runs through candidate-triage strategy "
                    "planning. discover runs only AI discovery v1. Use "
                    "theme_description for extra user intent. Use required_symbols "
                    "or base_symbols for mandatory symbols such as US.QQQ, US.SOXX, "
                    "US.NVDA. For answer_human_input/select_option, pass option_id "
                    "or answer/modifications/must_include_symbols/exclude_symbols. "
                    "For build_portfolio_maps, optional policy overrides include "
                    "target_portfolio_weight, cash_reserve, single_name_limit, "
                    "objective, risk_level, required_symbols, and allow_options. "
                    "For revise_portfolio_map, pass request/answer/message and "
                    "optionally base_map_id; if omitted, the selected map or latest "
                    "confirmed revision is used. "
                    "This workflow does not read holdings, submit orders, or create "
                    "option strategies."
                ),
                "additionalProperties": True,
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}


def handle_ia_portfolio_workflow(args: dict[str, Any], **_: Any) -> str:
    action = str(args.get("action") or "").strip()
    session_id = args.get("session_id")
    payload = args.get("payload") or {}
    if not isinstance(payload, dict):
        return tool_error("payload must be an object")
    try:
        workflow = InvestmentAssistantWorkflow()
        result = workflow.run(
            tenant=current_hermes_tenant(),
            action=action,
            session_id=str(session_id) if session_id else None,
            payload=payload,
        )
        if not result.get("success", False):
            return tool_error(result.get("error", "investment assistant workflow failed"), **result)
        remember_output_guard(result)
        return tool_result(result)
    except Exception as exc:
        return tool_error(f"investment assistant workflow failed: {exc}")


def current_hermes_tenant() -> str:
    """Derive a stable tenant key from Hermes gateway/CLI environment."""
    platform = os.getenv("HERMES_SESSION_PLATFORM") or os.getenv("HERMES_PLATFORM") or "cli"
    if platform == "cli":
        return "cli:local"
    identifiers = [
        os.getenv("HERMES_SESSION_CHAT_ID"),
        os.getenv("HERMES_SESSION_THREAD_ID"),
        os.getenv("HERMES_SESSION_USER_ID"),
        os.getenv("HERMES_SESSION_ID"),
    ]
    identifier = next((value for value in identifiers if value), "local")
    return f"{platform}:{identifier}"
