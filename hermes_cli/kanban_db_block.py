"""Block transition routing, split from the Kanban database facade."""


def route_block(
    kind: str | None, reason: str | None, source_status: str, *,
    prev_kind: str | None, prev_recurrences: int, recurrence_limit: int,
) -> tuple[str, str, str, tuple, dict]:
    """Return the destination status and event data for a block transition."""
    payload = {"reason": reason, "kind": kind, "source_status": source_status}
    if kind == "dependency":
        return "todo", "dependency_wait", "block_kind    = ?", (kind,), payload
    recurrences = prev_recurrences + 1 if prev_kind == kind else 1
    set_sql = "block_kind    = ?,\n                       block_recurrences = ?"
    payload = {"reason": reason, "kind": kind, "recurrences": recurrences, "source_status": source_status}
    if recurrences >= recurrence_limit:
        payload["limit"] = recurrence_limit
        return "triage", "block_loop_detected", set_sql, (kind, recurrences), payload
    return "blocked", "blocked", set_sql, (kind, recurrences), payload
