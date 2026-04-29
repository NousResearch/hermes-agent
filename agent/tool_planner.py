from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.auxiliary_client import call_llm


VALID_STAGES = ("exact_path", "known_subtree", "session_search", "broad_search")
DEFAULT_SEQUENCE = ["exact_path", "known_subtree", "session_search"]


@dataclass
class RetrievalPlan:
    needs_retrieval: bool = True
    goal: str = ""
    recommended_sequence: list[str] = field(default_factory=lambda: list(DEFAULT_SEQUENCE))
    max_retrieval_calls: int = 4
    allow_broad_search: bool = False
    stop_if: list[str] = field(default_factory=list)
    source: str = "fallback"


def should_skip_retrieval_planner(function_name: str, function_args: dict[str, Any]) -> bool:
    if function_name != "read_file":
        return False
    path = str((function_args or {}).get("path") or "").strip()
    return bool(path)


def build_default_plan(
    *,
    user_message: str,
    function_name: str,
    function_args: dict[str, Any],
    max_retrieval_calls: int,
    allow_broad_search: bool,
) -> RetrievalPlan:
    function_args = function_args or {}
    goal = user_message.strip() or f"Handle retrieval for {function_name}"
    if function_name == "session_search":
        sequence = ["session_search"]
    elif function_name == "read_file":
        sequence = ["exact_path", "known_subtree", "session_search"]
    else:
        sequence = list(DEFAULT_SEQUENCE)
    if allow_broad_search and "broad_search" not in sequence:
        sequence.append("broad_search")
    return RetrievalPlan(
        needs_retrieval=True,
        goal=goal,
        recommended_sequence=sequence,
        max_retrieval_calls=max(1, int(max_retrieval_calls or 4)),
        allow_broad_search=bool(allow_broad_search),
        stop_if=["found_exact_path", "two_empty_results"],
        source="fallback",
    )


def parse_planner_response(
    content: str,
    *,
    user_message: str,
    function_name: str,
    function_args: dict[str, Any],
    max_retrieval_calls: int,
    allow_broad_search: bool,
) -> RetrievalPlan:
    default = build_default_plan(
        user_message=user_message,
        function_name=function_name,
        function_args=function_args,
        max_retrieval_calls=max_retrieval_calls,
        allow_broad_search=allow_broad_search,
    )
    try:
        payload = json.loads(content)
    except Exception:
        return default
    if not isinstance(payload, dict):
        return default

    sequence = payload.get("recommended_sequence") or default.recommended_sequence
    if not isinstance(sequence, list):
        sequence = default.recommended_sequence
    filtered_sequence = [stage for stage in sequence if stage in VALID_STAGES]
    if not filtered_sequence:
        filtered_sequence = list(default.recommended_sequence)

    raw_calls = payload.get("max_retrieval_calls", default.max_retrieval_calls)
    try:
        parsed_calls = max(1, int(raw_calls))
    except (TypeError, ValueError):
        parsed_calls = default.max_retrieval_calls

    stop_if = payload.get("stop_if")
    if not isinstance(stop_if, list):
        stop_if = list(default.stop_if)

    return RetrievalPlan(
        needs_retrieval=bool(payload.get("needs_retrieval", True)),
        goal=str(payload.get("goal") or default.goal),
        recommended_sequence=filtered_sequence,
        max_retrieval_calls=parsed_calls,
        allow_broad_search=bool(payload.get("allow_broad_search", default.allow_broad_search)),
        stop_if=[str(item) for item in stop_if],
        source="planner",
    )


def plan_retrieval_tool_use(
    *,
    user_message: str,
    function_name: str,
    function_args: dict[str, Any],
    recent_messages: Optional[list[dict[str, Any]]] = None,
    max_retrieval_calls: int,
    allow_broad_search: bool,
) -> RetrievalPlan:
    if should_skip_retrieval_planner(function_name, function_args):
        return build_default_plan(
            user_message=user_message,
            function_name=function_name,
            function_args=function_args,
            max_retrieval_calls=max_retrieval_calls,
            allow_broad_search=allow_broad_search,
        )

    recent_messages = recent_messages or []
    compact_context = []
    for msg in recent_messages[-4:]:
        role = str(msg.get("role") or "unknown")
        if role == "tool":
            continue
        compact_context.append({"role": role, "content": str(msg.get("content") or "")[:600]})

    system = (
        "You are a cheap retrieval planner for Hermes. Return strict JSON only. "
        "Choose the narrowest retrieval ladder that fits the user's current request. "
        "Prefer session_search for transcript recall, read_file for a known exact path, "
        "search_files within a known subtree before any broad search. "
        "Allow broad_search only when clearly necessary."
    )
    user = {
        "user_message": user_message,
        "candidate_tool": function_name,
        "candidate_args": function_args,
        "recent_messages": compact_context,
        "default_max_retrieval_calls": max_retrieval_calls,
        "default_allow_broad_search": allow_broad_search,
        "valid_stages": list(VALID_STAGES),
        "required_json_schema": {
            "needs_retrieval": True,
            "goal": "string",
            "recommended_sequence": ["exact_path", "known_subtree", "session_search"],
            "max_retrieval_calls": max_retrieval_calls,
            "allow_broad_search": allow_broad_search,
            "stop_if": ["found_exact_path"],
        },
    }

    try:
        response = call_llm(
            task="planner",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=0,
            max_tokens=300,
        )
        content = response.choices[0].message.content
    except Exception:
        return build_default_plan(
            user_message=user_message,
            function_name=function_name,
            function_args=function_args,
            max_retrieval_calls=max_retrieval_calls,
            allow_broad_search=allow_broad_search,
        )

    return parse_planner_response(
        content,
        user_message=user_message,
        function_name=function_name,
        function_args=function_args,
        max_retrieval_calls=max_retrieval_calls,
        allow_broad_search=allow_broad_search,
    )
