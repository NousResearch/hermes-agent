"""send-approval plugin — effect-class approval gate for send / spend / hard-to-undo calls.

Instinct safety parity: draft → show recipient + final content → send only after
approval; never spend without confirmation; never make a hard-to-undo change from a
guess. This plugin classifies every dispatched tool call into an effect class
(``send-to-person`` / ``spend`` / ``hard-to-undo``) and — for flagged calls — returns
the documented ``pre_tool_call`` directive ``{"action": "approve", "message": ...,
"rule_key": ...}`` (``plugins/AGENTS.md``). Core (``hermes_cli/plugins.py``) then
escalates the call through ``tools/approval.py:request_tool_approval`` — the SAME
human gate as Tier-2 dangerous shell patterns: session cache, the ``[a]lways``
allowlist (grained by ``rule_key``), the CLI prompt, the blocking gateway
round-trip (``tools/approval_gateway_wait.py``), cron ``approvals.cron_mode``, and
fail-closed when no human is present.

The classifier NEVER blocks by itself: a refusal is the user's deny verdict (or the
gate's fail-closed default), never a classifier decision. Destructive shell commands
keep their own core gate (``tools/approval.py`` dangerous-command detection +
``tools/write_approval.py``); this plugin adds the missing policy layer for the
piece/MCP families below:

* ``mcp__activepieces__ap_run_action`` — the ActivePieces piece runner, classified
  from the piece/action name in its args (send/create/delete/pay/charge/post verbs).
* Piece actions surfacing as ``<piece>_<action>`` MCP names: ``gmail_send*``,
  ``slack_send*``, ``*_send_message``, ``stripe_*`` (spend), calendar
  create/delete, file/repo destructive ops (``*_delete_*`` and friends).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

SEND = "send-to-person"
SPEND = "spend"
UNDO = "hard-to-undo"

# Explicit families from the parity contract: (needle, effect, human label).
# Substring matches against the FULL lowercased tool name, so they hit every wire
# form (bare ``gmail_send_email``, ``mcp__activepieces__gmail_send_email``, legacy
# ``mcp_activepieces_gmail_send_email``). Checked before the generic verb scan so
# calendar create classifies as send-to-person (it invites attendees), not create.
_NAME_RULES: Tuple[Tuple[str, str, str], ...] = (
    ("gmail_send", SEND, "Gmail send"),
    ("slack_send", SEND, "Slack send"),
    ("stripe_", SPEND, "Stripe"),  # every stripe_* action moves money or card data
    ("send_message", SEND, "outbound message"),  # *_send_message and bare send_message
    ("calendar_create", SEND, "calendar create"),
    ("calendar_delete", UNDO, "calendar delete"),
)

# Generic verb scan, applied to MCP-scoped names only (native toolsets — terminal,
# file, memory — already have their own core gates). First group with a token-boundary
# hit wins, so "create_charge" classifies as spend, not create.
_VERB_RULES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("send", "post", "reply", "forward", "invite"), SEND),
    (("pay", "charge", "purchase", "checkout"), SPEND),
    (("delete", "remove", "trash", "purge", "revoke"), UNDO),
    (("create",), UNDO),  # creating external state from a guess is the hard-to-undo risk
)

_EFFECT_REASON = {
    SEND: ("send-to-person: this action sends content to a person ({detail}). Confirm "
           "the recipient and the final content before anything goes out."),
    SPEND: ("spend: this action moves money ({detail}). Confirm the amount and the "
            "recipient before any payment goes through."),
    UNDO: ("hard-to-undo: this action is hard to undo ({detail}). Confirm before it "
           "runs — never make a hard-to-undo change from a guess."),
}

# The ActivePieces action runner: classification comes from its args, not its name.
_RUNNER_SERVER = "activepieces"
_RUNNER_BASES = ("ap_run_action", "run_action")
_RUNNER_KEY_SUBSTRINGS = ("piece", "action")


def _split_mcp_name(tool_name: str) -> Tuple[str, str]:
    """``(server, base)`` from ``mcp__<server>__<tool>`` / ``mcp_<server>_<tool>``;
    bare names get the leading component as server (harmless — no rule needs it)."""
    name = (tool_name or "").strip().lower()
    if name.startswith("mcp__"):
        server, _, base = name[5:].partition("__")
        return server, base
    server, _, base = name.removeprefix("mcp_").partition("_")
    return server, base


def _verb_hit(text: str) -> Optional[str]:
    """First flagged verb at a token boundary in *text* (``resend_`` does not match
    ``send``; the underscore of snake_case does not hide a verb)."""
    for verbs, effect in _VERB_RULES:
        for verb in verbs:
            if re.search(rf"(?<![A-Za-z]){re.escape(verb)}", text):
                return effect
    return None


def _action_strings(args: Any, depth: int = 2) -> List[str]:
    """Piece/action identifier strings from the runner args. Only values under keys
    naming the piece or action are scanned — never user content like an email body."""
    if not isinstance(args, dict):
        return []
    found: List[str] = []
    for key, value in args.items():
        key_l = str(key).lower()
        if not any(marker in key_l for marker in _RUNNER_KEY_SUBSTRINGS):
            continue
        found.extend(_string_values(value, depth))
    return found


def _string_values(value: Any, depth: int) -> List[str]:
    if isinstance(value, str):
        return [value]
    if depth <= 0:
        return []
    if isinstance(value, dict):
        return [s for v in value.values() for s in _string_values(v, depth - 1)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _string_values(v, depth - 1)]
    return []


def _classify(tool_name: str, args: Any) -> Optional[Dict[str, str]]:
    """Classify one tool call; ``None`` when it is not flagged, else
    ``{"effect", "message", "rule_key"}`` for the approval gate."""
    name = (tool_name or "").strip().lower()
    if not name:
        return None
    for needle, effect, label in _NAME_RULES:
        if needle in name:
            return _verdict(effect, label, name)
    server, base = _split_mcp_name(name)
    if not server:
        return None
    effect = _verb_hit(base)
    if effect is not None:
        return _verdict(effect, base, name)
    if _RUNNER_SERVER in server and base in _RUNNER_BASES:
        for action_name in _action_strings(args):
            effect = _verb_hit(action_name)
            if effect is not None:
                return _verdict(effect, action_name, name)
    return None


def _verdict(effect: str, label: str, tool_name: str) -> Dict[str, str]:
    rule_key = f"send-approval:{re.sub(r'[^a-z0-9]+', '-', label.lower()).strip('-')}"
    message = _EFFECT_REASON[effect].format(detail=f"{label} via {tool_name}")
    return {"effect": effect, "message": message, "rule_key": rule_key}


def _on_pre_tool_call(tool_name: str = "", args: Any = None, **_: Any) -> Optional[Dict[str, str]]:
    """pre_tool_call hook: escalate flagged calls to the human gate (None = let it through)."""
    verdict = _classify(tool_name, args)
    if verdict is None:
        return None
    return {"action": "approve", "message": verdict["message"], "rule_key": verdict["rule_key"]}


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)