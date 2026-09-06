"""Oldest-first PR intake with canonical branch dependencies and fair read windows."""

def order_pull_requests(pulls, *, priority=None):
    """Prefer older PR numbers, except when an open parent must come first.

    Branch identity includes its repository: a same-named fork branch is not
    the branch a child PR targets. Cyclic or ambiguous relations never invent
    merge authority; the existing exact-base merge gates still decide eligibility.
    """
    remaining = {pull.number: pull for pull in pulls}
    branches = {}
    for pull in remaining.values():
        branches.setdefault((pull.head_repository.casefold(), pull.head_ref_name), set()).add(pull.number)
    parents = {
        pull.number: branches.get((pull.base_repository.casefold(), pull.base_branch), set()) - {pull.number}
        for pull in remaining.values()
    }
    ordered = []
    while remaining:
        ready = [number for number in remaining if not (parents[number] & remaining.keys())]
        if not ready:
            # Preserve catalogue coverage for diagnostics; merge gates remain closed.
            ordered.extend(remaining[number] for number in sorted(remaining))
            break
        number = min(ready, key=lambda number: ((priority or {}).get(number, 0), number))
        ordered.append(remaining.pop(number))
    return tuple(ordered)


def repair_window(ledger, repository, pulls, limit):
    """Persist the next inspection position independently of PR update activity.

    Rotation is for inspection only, not permission to merge a child. Sorting
    the selected window restores parent-before-child whenever both are present.
    """
    ordered = order_pull_requests(pulls)
    if len(ordered) <= limit:
        return ordered
    with ledger._transaction():
        ledger._connection.execute(
            "CREATE TABLE IF NOT EXISTS repair_selection_cursors "
            "(repository TEXT PRIMARY KEY, next_pr INTEGER NOT NULL)"
        )
        row = ledger._connection.execute(
            "SELECT next_pr FROM repair_selection_cursors WHERE repository = ?", (repository,)
        ).fetchone()
        start = next((i for i, pull in enumerate(ordered) if row and pull.number == row[0]), 0)
        rotated = ordered[start:] + ordered[:start]
        selected = rotated[:limit]
        ledger._connection.execute(
            "INSERT INTO repair_selection_cursors VALUES (?, ?) "
            "ON CONFLICT(repository) DO UPDATE SET next_pr = excluded.next_pr",
            (repository, rotated[limit].number),
        )
    return order_pull_requests(selected)
