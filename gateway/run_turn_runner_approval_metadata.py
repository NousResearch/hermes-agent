"""Pure helpers for the approval-flow metadata pipeline.

Holds the small bits that were inlined in ``gateway/run_turn_runner.py``'s
``_approval_notify_sync`` and were hard to test without the runner module's
import side effects (it pulls ``agent.replay_cleanup`` → ``hermes_cli.config``
→ reads ``~/.hermes/manifest.json`` under ``scripts/run_tests.sh``'s pm
activation, tripping the conftest's home-guard).

By extracting them here, we get:

  * A module that imports cleanly under the test sandbox — no hermes-core
    imports — so the helper-level tests run in CI without the home-guard
    trip.
  * One tested function per shape, callable directly from the runner.
    The runner still owns the side effects (adapter.send_exec_approval,
    timeout registration); this file owns only the deterministic
    data-stamping helpers.

This file replaces the inline block at the original ``run_turn_runner.py``
v1 base location (lines around 1479-1483).
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def stamp_request_id_into_metadata(
    metadata: Optional[Dict[str, Any]],
    approval_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a copy of ``metadata`` with ``approval_data['request_id']`` stamped.

    The bug this fixes: the gateway runner used to pass
    ``metadata=ctx._status_thread_metadata`` straight to ``adapter.send_exec_approval``
    without copying in ``approval_data["request_id"]``. The mattermost adapter's
    entry-registration read ``request_id`` from ``prompt.metadata`` (then from
    ``prompt.request_id``); with no ``request_id`` stamped, every pending card
    landed in the registry with ``request_id == ""`` and the tap handler
    fell through to ``queue.pop(0)`` (FIFO head). Two pending approvals in
    one session meant tapping ✅ on card B resolved card A's command.

    AI review follow-up #5863117294 (Finding 3 part B).

    Rules (intentionally narrow — anything else belongs in the runner):

      * If ``metadata`` is ``None``, the result is a fresh dict.
      * If ``approval_data["request_id"]`` is missing / falsy (``""`` /
        ``None``), the returned metadata is unchanged. This keeps legacy
        and pre-PR callers working without forcing every approval to mint
        a UUID — the mattermost adapter treats ``""`` as "no id".
      * The returned dict is a new copy; the input ``metadata`` is not
        mutated. Callers should pass the return value through, not retain
        the input.
    """
    merged: Dict[str, Any] = dict(metadata) if metadata else {}
    request_id = approval_data.get("request_id") or ""
    if request_id:
        merged["request_id"] = request_id
    return merged
