"""Connector calls re-enter the normal dispatcher under their composed names."""

import json
from dataclasses import asdict

from tools.registry import tool_error
from tools.tool_gateway.config import MAX_CALLS_PER_DISPATCH
from tools.tool_gateway.merge import assemble_results, fill_remote_failure, partition_calls


def dispatch_connector_call(name, arguments, tool_call_id):
    """Transport leg only; the caller owns the normal tool policy pipeline.

    Execution middleware wraps the actual I/O, so connector entries execute
    individually rather than queuing side effects after a policy callback returns.
    """
    from tools.tool_gateway.bridge import run_remote

    partition = partition_calls([{"name": name, "arguments": arguments}])
    entries = run_remote(partition.remote, tool_call_id, availability=None, client_factory=None)
    entry = entries[0]
    return json.dumps({key: value for key, value in entry.items() if key in {"response", "error"}},
                      ensure_ascii=False)


def dispatch_connector_batch(calls, ids, *, user_task, enabled_tools,
                             middleware_trace, enabled_toolsets, disabled_toolsets):
    from model_tools import handle_function_call
    from tools.interrupt import is_interrupted

    if len(calls) > MAX_CALLS_PER_DISPATCH:
        return tool_error(f"too many calls: {len(calls)} > max {MAX_CALLS_PER_DISPATCH}. "
                          "Retry with fewer calls per batch.")
    partition = partition_calls(calls)
    if partition.local:
        return tool_error("Local tools require one entry per tool_call; mixed and multi-local batches are not supported.")
    entries = list(partition.errors)
    for offset, plan in enumerate(partition.remote):
        if is_interrupted():
            # The executor only checks for /stop between tools, and this whole batch
            # is one tool to it: unstarted entries stay unsent, or a stop landing on
            # entry 1 of 20 would still fire 19 remote side effects.
            entries.extend(fill_remote_failure(
                partition.remote[offset:], "Stopped by the user before this call was made.",
                code="INTERRUPTED"))
            break
        # Wrapper-level skip flags describe only the wrapper, never its entries.
        payload = handle_function_call(
            plan.name, plan.arguments, **asdict(ids), user_task=user_task,
            enabled_tools=enabled_tools, tool_request_middleware_trace=list(middleware_trace),
            skip_pre_tool_call_hook=False, skip_tool_request_middleware=False,
            skip_tool_execution_middleware=False,
            enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets,
        )
        try:
            value = json.loads(payload) if isinstance(payload, str) else payload
        except ValueError:
            value = payload
        entry = {"index": plan.position, "name": plan.name}
        if isinstance(value, dict) and "error" in value:
            error = value["error"]
            entry["error"] = error if isinstance(error, dict) else {"code": "TOOL_ERROR", "message": str(error)}
        else:
            entry["response"] = value.get("response", value) if isinstance(value, dict) else value
        entries.append(entry)
    return json.dumps(assemble_results(len(calls), entries), ensure_ascii=False)
