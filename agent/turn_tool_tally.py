"""Thread-safe positive refusal evidence bound to the attempt's original turn."""
from functools import wraps
import json
import threading

_lock = threading.Lock()


def reset(agent):
    with _lock:
        agent._turn_tool_tally = {"attempted": 0, "refused": 0, "refused_names": []}


def export(agent, result):
    with _lock:
        tally = getattr(agent, "_turn_tool_tally", None)
        if isinstance(result, dict) and tally is not None and tally["attempted"] > 0:
            result.update(tool_calls_attempted=tally["attempted"], tool_calls_refused=tally["refused"],
                          tool_calls_refused_names=list(tally["refused_names"]))


def _guard_refused(name, result):
    if name != "terminal" or not isinstance(result, str) or len(result) >= 65536 or not result.lstrip().startswith("{"):
        return False
    try:
        body = json.loads(result)
    except ValueError:
        return False
    return (isinstance(body, dict) and body.get("output") == "" and body.get("exit_code") == -1
            and "error" in body and (body.get("status") in ("blocked", "pending_approval")
                                     or body.get("approval_pending") is True))


def track_attempt(run):
    """Exceptions or unfinished calls are attempts, never positive refusals."""
    @wraps(run)
    def tracked(agent, **kwargs):
        with _lock:
            tally = getattr(agent, "_turn_tool_tally", None)
            if tally is not None:
                tally["attempted"] += 1
        state = run(agent, **kwargs)
        if tally is not None and (state.blocked or not state.dispatched or _guard_refused(kwargs["function_name"], state.result)):
            with _lock:
                tally["refused"] += 1
                name = kwargs["function_name"]
                if name not in tally["refused_names"] and len(tally["refused_names"]) < 5:
                    tally["refused_names"].append(name)
        return state
    return tracked
