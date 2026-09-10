"""Keep parent checkout leases while follow-up workers still need them."""

from collections.abc import Callable, Mapping


def descendants_finished(
    lookup: Callable[[str, str], Mapping[str, object] | None],
    board: str,
    task_id: str,
    terminal_statuses: frozenset[str],
) -> bool:
    pending = [task_id]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        if len(seen) >= 256:
            return False
        seen.add(current)
        task = lookup(board, current)
        if task is None:
            # Preserve the existing orphan-root policy, but an unresolved
            # dependent is not proof that its parent's checkout is free.
            return current == task_id
        if task.get("status") not in terminal_statuses:
            return False
        children = task.get("_children", [])
        if not isinstance(children, list) or any(
            not isinstance(child, str) or not child for child in children
        ):
            return False
        pending.extend(children)
    return True
