"""ownership-gate plugin — #47 Phase 2 C1/C2 DIRECT-dispatch ownership gate.

Bundled ``kind: backend`` (auto-loads), but a NO-OP until
``HERMES_OWNERSHIP_DIRECT_ENFORCE=1``:

* ``pre_tool_call`` claims every scope in the dispatch batch (top-level
  ``goal``/``context`` plus every ``tasks[].goal``/``tasks[].context``) in ONE
  all-or-nothing ledger transaction, or blocks the call. Fail-closed: a missing /
  broken ledger blocks rather than dispatching unowned.
* ``post_tool_call`` releases the scopes that call claimed when the call
  reported an error. Fail-open: a release failure never breaks the tool result.

The logic lives in :mod:`gateway.ownership_gate`; this module only wires the
two hook callbacks into the plugin context.
"""

from __future__ import annotations

import logging

from gateway.ownership_gate import post_tool_call, pre_tool_call

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("post_tool_call", post_tool_call)
