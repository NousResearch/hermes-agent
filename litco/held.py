"""A command held for approval, as the model and the app see it.

A turn that arrives through ``litco_turn`` has no approval channel (deploy/host/README.md,
"Approvals"). When Hermes's approval gate holds a command, the terminal tool answers at once with
``{"status": "pending_approval", "approval_pending": true, "exit_code": -1, "error": ""}``. Read
as it stands, that looks like a failed command, and on 2026-10-01 Ana took one such hold (a script
printing her environment) as proof that her LitKit credentials were missing.

``held_tool_result`` is a ``transform_tool_result`` hook (registered by ``plugins/litkit``). During a
LitCo turn it replaces a held result with one that says the command did not run and that nothing
follows from that. The relay (:func:`litco.hermes_runner._summarize_result`) reports either shape to
the app as ``status: "ok"`` with ``held: true``, because the app's tool pill knows only ``ok`` and
``error`` and a held command is not an error.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from litco.litkit.context import current_turn

HELD_STATUS = "held_for_approval"
# What the app's tool pill shows.
HELD_SUMMARY = "held for approval; not run"
# What the model reads in place of the gate's result.
HELD_MESSAGE = (
    "Held for approval. The command did not run, and nobody in this conversation can approve it, so it "
    "will not run in this turn. This is not a failure. It says nothing about your tools, LitKit, any "
    "credentials, or the documents, so infer nothing from it. Do not run it again or rephrase it. If one "
    "of your tools does the job, use that tool. Otherwise tell the person the command was held for "
    "approval."
)

_GATE_STATUSES = frozenset({"pending_approval", "approval_required"})


def _as_dict(result: Any) -> Optional[dict]:
    if isinstance(result, dict):
        return result
    if not isinstance(result, str) or result[:1] != "{":
        return None
    try:
        parsed, _ = json.JSONDecoder().raw_decode(result)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def is_held(result: Any) -> bool:
    """True for the approval gate's hold and for the result :func:`held_tool_result` puts in its place."""
    parsed = _as_dict(result)
    if parsed is None:
        return False
    if parsed.get("status") == HELD_STATUS:
        return True
    # Every shape Hermes's gate returns names the rule that matched (``pattern_key``); a LitKit
    # response that happens to carry a similar status does not.
    return "pattern_key" in parsed and (parsed.get("status") in _GATE_STATUSES
                                        or parsed.get("approval_pending") is True)


def held_tool_result(tool_name: str = "", args: Any = None, result: Any = None, **_: Any) -> Optional[str]:
    """``transform_tool_result`` hook: the held result the model reads during a LitCo turn (None = unchanged)."""
    if current_turn() is None or not is_held(result):
        return None
    parsed = _as_dict(result) or {}
    if parsed.get("status") == HELD_STATUS:
        return None
    held = {"status": HELD_STATUS, "ran": False, "message": HELD_MESSAGE}
    command = parsed.get("command")
    if isinstance(command, str) and command:
        held["command"] = command
    return json.dumps(held, ensure_ascii=False)
