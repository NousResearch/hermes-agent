"""Tool control resolver (Phase A2).

Parses hook results into control decisions using a priority-based protocol.

Pre-tool actions:  allow | deny | modify | ask | short_circuit
Post-tool actions: allow | modify_result

Priority (highest wins): short_circuit > deny > ask > modify > allow
"""

from typing import Any

_VALID_PRE_ACTIONS = {"allow", "deny", "modify", "ask", "short_circuit"}
_VALID_POST_ACTIONS = {"allow", "modify_result"}


def is_control_protocol(result: Any) -> bool:
    """Check if a hook result follows the control protocol."""
    return (
        isinstance(result, dict)
        and result.get("action") in (_VALID_PRE_ACTIONS | _VALID_POST_ACTIONS)
    )


def resolve_pre_tool_control(hook_results: list, original_args: dict) -> dict:
    """Resolve multiple pre_tool_call hook results into a single control decision.

    Returns a dict with at least an "action" key.
    Priority: short_circuit > deny > ask > modify(pipeline) > allow
    Non-protocol results are silently ignored.
    """
    if not hook_results:
        return {"action": "allow"}

    controls = [r for r in hook_results if is_control_protocol(r)]
    if not controls:
        return {"action": "allow"}

    # Priority ordering
    for action in ("short_circuit", "deny", "ask"):
        for c in controls:
            if c.get("action") == action:
                return c

    # Pipeline modify: apply each modify in order
    args = dict(original_args)
    found_modify = False
    for c in controls:
        if c.get("action") == "modify" and "args" in c:
            args.update(c["args"])
            found_modify = True

    if found_modify:
        return {"action": "modify", "args": args}

    return {"action": "allow"}


def resolve_post_tool_control(hook_results: list, original_result: str) -> str:
    """Resolve post_tool_call hook results — apply modify_result chain.

    Returns the (possibly modified) result string.
    """
    if not hook_results:
        return original_result

    result = original_result
    for r in hook_results:
        if is_control_protocol(r) and r.get("action") == "modify_result":
            if "result" in r:
                result = r["result"]

    return result
