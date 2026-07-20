"""LangGraph AI Company router.

Reads tasks from Obsidian markdown, dispatches via node4 coordinator to mac_hermes
and windows_hermes in parallel, mocks remote RPC and DB, and runs a SkillOpt-style
reflection loop over failure logs.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from ai_company_router.obsidian_io import read_task, write_status


def merge_results(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {**(a or {}), **(b or {})}


def merge_errors(a: Optional[List[str]], b: Optional[List[str]]) -> List[str]:
    return list(a or []) + list(b or [])


DEFAULT_PROTOCOL_VERSION = "2025-06-18"
MOCK_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "obsidian_vault", "output", "mock_db.json")
FAILURE_LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "obsidian_vault", "output", "failure_log.md")
SKILL_DOC_FILE = os.path.join(os.path.dirname(__file__), "..", "obsidian_vault", "skills", "distributed_router.md")


class Task(TypedDict):
    id: str
    description: str
    target_os: List[Literal["mac", "windows", "any"]]
    payload: Dict[str, Any]
    priority: Literal["low", "medium", "high", "critical"]


class RouterState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    task: Optional[Task]
    plan: List[Dict[str, Any]]
    results: Annotated[Dict[str, Any], merge_results]
    status: Literal["pending", "dispatched", "running", "done", "failed"]
    errors: Annotated[List[str], merge_errors]
    skill_patch: Optional[str]


def mock_db_read() -> Dict[str, Any]:
    if not os.path.exists(MOCK_DB_PATH):
        return {"tasks": [], "agents": {}}
    with open(MOCK_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def mock_db_write(record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(MOCK_DB_PATH), exist_ok=True)
    db = mock_db_read()
    db["tasks"].append(record)
    with open(MOCK_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def mock_remote_execute(agent_id: str, sub_task: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[MOCK-RPC] {agent_id}: {sub_task['description'][:60]}...")
    return {
        "agent_id": agent_id,
        "status": "success",
        "output": f"Completed by {agent_id}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def node4_coordinator(state: RouterState) -> RouterState:
    task_raw = read_task()
    if not task_raw:
        return {**state, "status": "failed", "errors": ["No task in Obsidian input.md"]}

    task: Task = {
        "id": f"task-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        "description": task_raw.strip(),
        "target_os": ["any"],
        "payload": {},
        "priority": "medium",
    }

    plan = []
    targets = set()
    for os_target in task["target_os"]:
        if os_target in ("mac", "any"):
            targets.add("mac_hermes")
        if os_target in ("windows", "any"):
            targets.add("windows_hermes")

    for agent in sorted(targets):
        plan.append({
            "agent": agent,
            "description": task["description"],
            "payload": task["payload"],
            "status": "pending",
        })

    new_state = {**state, "task": task, "plan": plan, "results": {}, "status": "dispatched", "errors": []}
    write_status(new_state)
    return new_state


def mac_hermes(state: RouterState) -> Dict[str, Any]:
    task = state.get("task")
    if task is None or not any(p["agent"] == "mac_hermes" for p in state["plan"]):
        return {}
    result = mock_remote_execute("mac_hermes", {"description": task["description"], "payload": task["payload"]})
    return {
        "results": {"mac_hermes": result},
        "messages": [AIMessage(content=f"mac_hermes: {result['output']}")],
    }


def windows_hermes(state: RouterState) -> Dict[str, Any]:
    task = state.get("task")
    if task is None or not any(p["agent"] == "windows_hermes" for p in state["plan"]):
        return {}
    result = mock_remote_execute("windows_hermes", {"description": task["description"], "payload": task["payload"]})
    return {
        "results": {"windows_hermes": result},
        "messages": [AIMessage(content=f"windows_hermes: {result['output']}")],
    }


def db_logger(state: RouterState) -> RouterState:
    if state["task"] is None:
        return state
    status = "done" if not state["errors"] else "failed"
    mock_db_write({
        "task_id": state["task"]["id"],
        "status": status,
        "results": state["results"],
        "errors": state["errors"],
        "logged_at": datetime.now(timezone.utc).isoformat(),
    })
    return {**state, "status": status}


def skillopt_reflection(state: RouterState) -> RouterState:
    if not state["errors"]:
        return {**state, "skill_patch": None}

    os.makedirs(os.path.dirname(FAILURE_LOG_FILE), exist_ok=True)
    for error in state["errors"]:
        with open(FAILURE_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n## {datetime.now(timezone.utc).isoformat()}Z — {state['task']['id'] if state['task'] else 'unknown'}\n- {error}\n")

    patch = (
        f"\n## Patch {datetime.now(timezone.utc).isoformat()}Z\n"
        f"- Detected recurring failure: {state['errors'][0]}\n"
        "- Added guard in `node4_coordinator` to validate Obsidian input before dispatch.\n"
    )
    os.makedirs(os.path.dirname(SKILL_DOC_FILE), exist_ok=True)
    with open(SKILL_DOC_FILE, "a", encoding="utf-8") as f:
        f.write(patch)
    return {**state, "skill_patch": patch}


def finalize_status(state: RouterState) -> RouterState:
    write_status(state)
    return state


def dispatch_router(state: RouterState) -> List[str]:
    if state["status"] == "failed":
        return ["skillopt_reflection"]
    agents = sorted({p["agent"] for p in state["plan"]})
    return agents if agents else ["skillopt_reflection"]


def aggregate_router(state: RouterState) -> str:
    status = "done" if not state["errors"] else "failed"
    return "db_logger" if status == "done" else "skillopt_reflection"


def build_graph():
    builder = StateGraph(RouterState)

    builder.add_node("node4_coordinator", node4_coordinator)
    builder.add_node("mac_hermes", mac_hermes)
    builder.add_node("windows_hermes", windows_hermes)
    builder.add_node("finalize_status", finalize_status)
    builder.add_node("db_logger", db_logger)
    builder.add_node("skillopt_reflection", skillopt_reflection)

    builder.set_entry_point("node4_coordinator")

    builder.add_conditional_edges(
        "node4_coordinator",
        dispatch_router,
        {
            "mac_hermes": "mac_hermes",
            "windows_hermes": "windows_hermes",
            "skillopt_reflection": "skillopt_reflection",
        },
    )

    for node in ("mac_hermes", "windows_hermes"):
        builder.add_conditional_edges(node, aggregate_router, {"db_logger": "db_logger"})

    builder.add_edge("db_logger", "finalize_status")
    builder.add_edge("finalize_status", "skillopt_reflection")
    builder.add_edge("skillopt_reflection", END)

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph()
    initial_state: RouterState = {
        "messages": [HumanMessage(content="Start AI Company router")],
        "task": None,
        "plan": [],
        "results": {},
        "status": "pending",
        "errors": [],
        "skill_patch": None,
    }
    final_state = graph.invoke(initial_state)
    print(json.dumps(final_state, indent=2, default=str))
