"""Per-call deadline floors: a tool that declares how long one call may block
(``ToolEntry.deadline_floor``) has the executor's generic deadline raised to clear it."""

# A tool that declares a per-call budget (ToolEntry.deadline_floor) gets the generic deadline
# raised to it plus this grace, so the tool's OWN timeout is the one that fires: it returns the
# partial output and exit code it collected, where the generic guard discards both and reports
# only ``tool_timeout``.
_DECLARED_DEADLINE_GRACE_S = 5.0


def declared_deadline_floor(name: str, args: dict) -> float | None:
    """Lower bound this call's deadline must clear, from the tool's own declaration."""
    from tools.registry import registry

    floor = registry.get_deadline_floor(name, args)
    return None if floor is None else floor + _DECLARED_DEADLINE_GRACE_S


def deadline_with_floor(generic: float | None, floor: float | None) -> float | None:
    """Raise *generic* to *floor*. A disabled generic deadline (``None``) stays unbounded —
    a floor only ever lengthens a deadline, it never introduces one."""
    if generic is None or floor is None:
        return generic
    return max(generic, floor)


def batch_deadline(generic: float | None, parsed_calls) -> float | None:
    """One deadline covers a whole concurrent batch, so it has to clear the longest budget any
    member declared — otherwise one slow-but-valid call takes every sibling down with it."""
    for call in parsed_calls:
        if call.parse_error is None:
            generic = deadline_with_floor(generic, declared_deadline_floor(call.name, call.args))
    return generic
