"""Authenticated continuation keeps durable failure notices out of model context."""
from agent.turn_failure_copy import FAILED_TURN_DISPLAY_KIND, failed_turn_notice


def retry_failure_row_ids(history: list) -> tuple[int, ...] | None:
    """Prove the Hermes failure boundaries in this user turn; None refuses continuation.

    A continuation keeps these display rows in storage, but omits them on every model
    request, including after another rate limit. Earlier user turns remain untouched.
    Its caller authenticates the complete transcript before using these row addresses.
    """
    user = next((i for i in range(len(history) - 1, -1, -1) if history[i].get("role") == "user"), None)
    ids = []
    for index, row in enumerate(history):
        if user is None or index <= user or row.get("display_kind") != FAILED_TURN_DISPLAY_KIND:
            continue
        surface = (row.get("display_metadata") or {}).get("error_surface")
        if (type(row.get("_row_id")) is not int or row.get("role") != "assistant" or row.get("tool_calls")
                or row.get("content") != failed_turn_notice(history[user:index])
                or not isinstance(surface, dict) or surface.get("layer") != "provider"
                or surface.get("code") != "rate_limit"):
            return None
        ids.append(row["_row_id"])
    return tuple(ids)


def repair_preserving_failure_rows(agent, messages, repair):
    """Repair each model segment without merging a proven display boundary into new output.

    The outer cursor-aware repair still accounts for any merges inside the segments.
    None delegates the ordinary path to the caller.
    """
    if getattr(agent, "_resume_user_row_id", None) is None:
        return None
    ids = getattr(agent, "_resume_failure_row_ids", ())
    if not ids or not any(row.get("_row_id") in ids for row in messages):
        return None
    rebuilt, segment, count = [], [], 0
    for row in messages:
        if row.get("_row_id") in ids:
            count += repair(agent, segment)
            rebuilt.extend([*segment, row])
            segment = []
        else:
            segment.append(row)
    count += repair(agent, segment)
    if count:
        messages[:] = [*rebuilt, *segment]
    return count
