"""A handoff is an assistant-authored checkpoint, never a new policy instruction."""

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HandoffIntent:
    session_id: str
    tool_call_id: str
    summary: str
    compression_count: int


SCHEMA = {
    "name": "fresh_session",
    "description": (
        "Request a fresh session in this Telegram group topic after this turn completes. "
        "Omit handoff to inspect available usage telemetry without requesting a handoff. "
        "Choose a task boundary when carrying this long conversation is inefficient; consider "
        "context size and cache/usage evidence when available, not a fixed threshold. "
        "Supply a compact factual checkpoint: objective, user constraints and authorization, "
        "decisions, completed work, files/CWD, checks/failures, outstanding work and next action. "
        "Do not invent permissions or copy secrets. Call as your last tool, then finish normally. "
        "Do not announce the handoff; the runtime adds one short note. "
        "Runtime may decline if interrupted, queued input or active work makes settlement unsafe. "
        "This requests a new session; it does not compact or rewrite this transcript."
    ),
    "parameters": {"type": "object", "properties": {
        "handoff": {"type": "string", "minLength": 1, "maxLength": 12000},
    }, "additionalProperties": False},
}


def authorized(agent):
    return bool(getattr(agent, "_fresh_topic_handoff_allowed", False)) and getattr(agent, "api_mode", None) != "codex_app_server"


def inject(agent, tools_list=None, name_set=None):
    # Session surface is fixed at construction, as with message_agent. No core tools[] change.
    tools_list = agent.tools if tools_list is None else tools_list
    name_set = agent.valid_tool_names if name_set is None else name_set
    if not authorized(agent):
        tools_list[:] = [tool for tool in tools_list if tool.get("function", {}).get("name") != "fresh_session"]
        name_set.discard("fresh_session")
    elif tools_list and "fresh_session" not in name_set:
        tools_list.append({"type": "function", "function": SCHEMA})
        name_set.add("fresh_session")


def request(agent, args, ctx):
    if not isinstance(args, dict) or set(args) - {"handoff"}:
        return json.dumps({"error": "Only the handoff checkpoint is accepted"})
    summary = args.get("handoff")
    if not authorized(agent):
        return json.dumps({"error": "Fresh-session handoff requires an owning Telegram topic with the Hermes tool-loop runtime; native Codex is unsupported"})
    if "handoff" not in args:
        compressor = getattr(agent, "context_compressor", None)
        return json.dumps({"status": "telemetry", "usage": {
            "last_prompt_tokens": getattr(compressor, "last_prompt_tokens", None),
            "context_length": getattr(compressor, "context_length", None),
            "input_tokens": getattr(agent, "session_prompt_tokens", None),
            "output_tokens": getattr(agent, "session_completion_tokens", None),
            "cached_tokens": getattr(agent, "session_cache_read_tokens", None),
        }})
    if not isinstance(summary, str) or not 1 <= len(summary.strip()) <= 12000:
        return json.dumps({"error": "handoff must contain 1–12000 characters"})
    intent = HandoffIntent(agent.session_id, ctx.tool_call_id, summary.strip(),
                           getattr(getattr(agent, "context_compressor", None), "compression_count", 0))
    previous = getattr(agent, "_fresh_topic_handoff", None)
    if previous is None:
        agent._fresh_topic_handoff = intent
    return json.dumps({
        "status": "already_requested" if previous is not None else "requested_after_turn",
        "final_reply_guidance": (
            "Finish with only the task result or pending next action. Do not announce or describe "
            "a session change, even when explaining efficiency: Hermes adds the transition note."
        ),
    })


def eligible_intent(agent, result):
    intent = getattr(agent, "_fresh_topic_handoff", None)
    if intent is None or any(result.get(k) for k in ("failed", "error", "interrupted", "partial")):
        return None
    if result.get("completed") is not True or agent.session_id != intent.session_id:
        return None
    if intent.compression_count != getattr(getattr(agent, "context_compressor", None), "compression_count", 0):
        return None
    # Steering or more work after the checkpoint invalidates it. Repeated requests are idempotent.
    seen = False
    for message in result.get("messages", []):
        if seen and message.get("role") == "user":
            return None
        calls = message.get("tool_calls") or []
        for call in calls:
            if call.get("id") == intent.tool_call_id:
                seen = True
            elif seen and call.get("function", {}).get("name") != "fresh_session":
                return None
        if seen and calls and any(call.get("function", {}).get("name") != "fresh_session" for call in calls):
            return None
    return asdict(intent) if seen else None
