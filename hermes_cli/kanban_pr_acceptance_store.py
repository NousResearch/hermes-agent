"""Persist acceptance with the same ownership snapshot as the terminal write."""
from __future__ import annotations

from typing import Any, Optional

from hermes_cli.kanban_completion_attempt import stamp_attempt
from hermes_cli.kanban_db_connect import write_txn
from hermes_cli.kanban_pr_acceptance import _PR, collect_acceptance

#: The ONE key the gate reads. Handoffs that name the PR under a neighbouring
#: key are rejected loudly instead of being silently re-keyed: the card's PR is
#: pinned permanently from this value, so guessing which field meant "the PR"
#: is how a sibling PR gets pinned to the wrong card.
PUBLISHED_PR_KEY = "published_pr"
_NEAR_MISS_KEYS = (
    "pr_url", "pull_request_url", "pr", "pull_request", "pr_link",
    "published_pull_request", "publishedPr", "published_pr_url", "github_pr",
)


class PublishedPrBindingError(ValueError):
    """The handoff carries PR evidence the contract gate cannot bind.

    Either the URL sits under a key other than ``metadata.published_pr``, or it
    names a different repository than the card's persisted contract. A
    ``ValueError`` so tool error handlers treat it as recoverable: nothing was
    mutated and the same handoff can be re-sent with the right key.
    """


def _snapshot(conn, task_id):
    row = conn.execute("SELECT current_run_id, status, completion_contract FROM tasks WHERE id=?", (task_id,)).fetchone()
    return tuple(row) if row else None


def published_pr_value(metadata: Any) -> Optional[str]:
    """The exact PR URL under ``metadata.published_pr``, or None."""
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(PUBLISHED_PR_KEY)
    return value if isinstance(value, str) and _PR.fullmatch(value) else None


def require_published_pr_key(contract: str, metadata: Any) -> None:
    """Raise when a repository contract still needs its PR and the handoff put
    one under the wrong key. Silent on handoffs that name no PR at all — that
    case stays a receipt ("supply metadata.published_pr"), not an exception,
    so a card can be completed by a human with evidence off the board."""
    if not isinstance(metadata, dict):
        return
    for key in _NEAR_MISS_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and _PR.search(value):
            raise PublishedPrBindingError(
                f"completion contract {contract} binds the card's exact PR from "
                f"metadata.{PUBLISHED_PR_KEY}, but this handoff put the PR URL under "
                f"metadata.{key!r}. Nothing changed. Re-send the same handoff with "
                f'metadata={{"{PUBLISHED_PR_KEY}": "{_PR.search(value)[0]}"}} — the key is '
                f"exact and {key!r} is never read as the published PR."
            )


def bind_published_pr(conn, task_id: str, metadata: Any, *, run_id: Optional[int] = None) -> Optional[str]:
    """Pin the card's exact PR inside the CALLER's write transaction.

    Called by ``request_review`` so the reviewer's completion does not have to
    repeat ``metadata.published_pr``: the implementer is the only actor that
    knows the URL, and the reviewer completing the card is the actor the gate
    runs for. Returns the newly pinned URL, else None.

    An already-pinned contract is immutable — a second URL never replaces the
    first, so a failed retry cannot substitute a green sibling PR.
    """
    row = conn.execute("SELECT completion_contract FROM tasks WHERE id=?", (task_id,)).fetchone()
    contract = row["completion_contract"] if row else None
    if not contract or contract == "local-only":
        return None
    published = published_pr_value(metadata)
    if _PR.fullmatch(contract):
        # Already pinned: re-handing off the SAME PR is a no-op, naming a
        # different one is refused loudly rather than ignored in silence.
        if published and published != contract:
            raise PublishedPrBindingError(
                f"this card is already pinned to {contract}; metadata.{PUBLISHED_PR_KEY}="
                f"{published} cannot replace it. Nothing changed. Push to the pinned PR, or "
                f"have an operator create a new card for a different pull request."
            )
        return None
    if published is None:
        require_published_pr_key(contract, metadata)
        return None
    if _PR.fullmatch(published)[1] != contract:
        raise PublishedPrBindingError(
            f"metadata.{PUBLISHED_PR_KEY}={published} is not a pull request of {contract}, the "
            f"repository this card's completion contract declares. Nothing changed. Pass the PR "
            f"you published for THIS card, or have an operator correct the card's contract."
        )
    from hermes_cli.kanban_db import _append_event

    if conn.execute(
        "UPDATE tasks SET completion_contract=? WHERE id=? AND completion_contract=?",
        (published, task_id, contract),
    ).rowcount != 1:
        return None
    _append_event(conn, task_id, "pr_pinned",
                  {"pr_url": published, "previous_contract": contract}, run_id=run_id)
    return published


def prepare_acceptance(conn, task_id, expected_run_id, metadata, *, attempt_id=None):
    """Collect acceptance evidence for ONE completion attempt.

    ``attempt_id`` is stamped onto the receipt so the attempt that persisted it
    can be identified later (``kanban_completion_attempt``); the network work
    itself happens with no transaction open.
    """
    snapshot = _snapshot(conn, task_id)
    if snapshot is None:
        return False
    run_id, status, contract = snapshot
    if not contract or contract == "local-only":
        return None
    if status not in {"running", "ready", "blocked", "review"} or (expected_run_id is not None and run_id != expected_run_id):
        return False
    published_pr = published_pr_value(metadata)
    # A pinned contract carries its own PR (bound at request_review); only a
    # repository contract still needs the key, so only that case may error on
    # a near-miss key.
    if published_pr is None and not _PR.fullmatch(contract):
        require_published_pr_key(contract, metadata)
    # Publication binds once. Retrying cannot replace the task's PR with a green sibling.
    if published_pr and contract == _PR.fullmatch(published_pr)[1]:
        with write_txn(conn):
            if _snapshot(conn, task_id) != snapshot:
                return False
            conn.execute("UPDATE tasks SET completion_contract=? WHERE id=?", (published_pr, task_id))
        snapshot = (run_id, status, published_pr)
        contract = published_pr
    # The assignee profile's gh login owns the repo: acceptance must not run as
    # the ambient login of whichever process completes the card (#122689).
    assignee = conn.execute("SELECT assignee FROM tasks WHERE id=?", (task_id,)).fetchone()["assignee"]
    return snapshot, stamp_attempt(
        collect_acceptance(contract, published_pr, assignee=assignee), attempt_id)


def record_acceptance(conn, task_id, acceptance):
    """Called under complete_task's write_txn, before its terminal UPDATE."""
    from hermes_cli.kanban_db import _append_event
    snapshot, receipt = acceptance
    if _snapshot(conn, task_id) != snapshot:
        return False
    _append_event(conn, task_id, "pr_acceptance", receipt, run_id=snapshot[0])
    if not receipt["ok"]:
        detail = f"PR acceptance {receipt['classification']}: {receipt.get('detail', '')} {receipt['recovery']}"
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (detail, task_id))
    return receipt["ok"]
