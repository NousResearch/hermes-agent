"""Exit-code contract for non-interactive single-query runs.

Both non-interactive entry points — the fully-quiet machine path
(``hermes chat -q … -Q``) and the human-facing single-query path
(``hermes chat -q …``) — must report the turn's outcome through the
process exit code, because their only consumers are automation wrappers:
shell scripts, cron jobs, and the kanban dispatcher.

The logic lives here rather than inline in ``cli.py`` so the two call
sites cannot drift apart again. They did drift: the quiet path grew the
0/1/75 contract while the human-facing path never inspected the turn
result at all and returned 0 unconditionally. Kanban workers are spawned
with ``-q`` and no ``-Q`` (see ``_default_spawn`` in
``hermes_cli/kanban_db.py``), so every provider-side failure reached the
dispatcher as a clean exit — which its reap classifier can only read as
"the worker finished without calling ``kanban_complete``", i.e. a
protocol violation by the agent. Incident 2026-09-02: a Copilot session
limit produced eight consecutive ``crashed`` runs on card ``t_d16778b1``,
each labelled a protocol violation, each burning one of the card's
failure-counter lives, none of them the agent's doing.

The contract:

* success / interrupted → ``0``
* failed turn → ``1``
* failed turn **inside a kanban worker** because the provider walled the
  quota (``failure_reason`` of ``rate_limit`` or ``billing``) →
  ``KANBAN_RATE_LIMIT_EXIT_CODE`` (75, ``EX_TEMPFAIL``). The dispatcher's
  reap classifier maps that code to a ``rate_limited`` exit and releases
  the card back to ``ready`` WITHOUT incrementing the failure counter, so
  a multi-hour quota window cannot trip the circuit breaker and block the
  card permanently.

Non-kanban runs keep the plain 0/1 contract automation wrappers expect.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

# ``failure_reason`` values that mean "quota wall, not a task error".
QUOTA_FAILURE_REASONS = ("rate_limit", "billing")


def single_query_exit_code(
    result: Optional[Any],
    *,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Return the process exit code for a finished single-query turn.

    ``result`` is the dict ``Agent.run_conversation()`` returns (or
    ``None`` when the turn never got that far, e.g. a credential refresh
    failure — treated as a failure, since no answer was produced).
    ``env`` defaults to ``os.environ`` and is injectable for tests.
    """
    _env = os.environ if env is None else env

    if result is None:
        # The turn never reached the agent (credentials, agent init) — no
        # answer was produced, so this is a failure. Matches the quiet
        # path's ``sys.exit(1)`` for a failed init.
        return 1
    if not isinstance(result, dict):
        # The quiet path already tolerates a non-dict result by printing
        # ``str(result)``; something was produced, so don't call it a
        # failure.
        return 0
    if not result.get("failed"):
        return 0

    if _env.get("HERMES_KANBAN_TASK") and result.get(
        "failure_reason"
    ) in QUOTA_FAILURE_REASONS:
        try:
            from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE
        except Exception:
            return 1
        return int(KANBAN_RATE_LIMIT_EXIT_CODE)

    return 1
