"""The policy enforcement plugin, as installed into the runtime.

This file is **copied verbatim** into each agent's profile as the plugin entry point. It
runs inside a worker process, not inside NOVA, so it imports nothing from NOVA and
nothing from the runtime — only the standard library and its sibling ``_decide`` module,
which is a verbatim copy of :mod:`nova.policy.decide`.

It implements ``pre_tool_call``, the runtime's documented policy hook: returning
``{"action": "block", "message": ...}`` vetoes a call, and ``{"action": "approve", ...}``
escalates it to the same human gate that guards dangerous shell commands — which fails
closed when no human is present.

**Every failure path here blocks.** A policy that cannot be read, a decision that raises,
a document with an unknown schema: all produce a refusal. A governance control that fails
open is not a control, and a worker that is too restricted fails loudly and visibly,
while one that is silently unrestricted does not.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:  # installed layout: the copied decision module sits beside this file
    from ._decide import ALLOW, DENY, REQUIRE_APPROVAL, decide
except ImportError:  # in-tree layout, for tests that import this module directly
    from nova.policy.decide import ALLOW, DENY, REQUIRE_APPROVAL, decide

#: Written beside the profile's configuration by the runtime adapter.
POLICY_FILENAME = "nova-policy.json"

_CACHE: Dict[str, Any] = {}

#: Budget-consuming tool calls made by this process. A worker handles one task per
#: process, so this is a per-run counter; in a long-lived interactive session it is
#: per-session. Named accordingly in the spec (``max_tool_calls_per_run``) so it is never
#: mistaken for a per-day or per-tenant budget, which this runtime cannot enforce.
_CALLS_USED = 0


def _policy_path() -> Path:
    """The compiled policy for the agent this worker is running as.

    Resolved from this file's location — ``<profile>/plugins/nova-policy/__init__.py`` —
    rather than from the environment, so it cannot be pointed elsewhere by a variable and
    is correct even when several agents run concurrently on one host.
    """
    return Path(__file__).resolve().parents[2] / POLICY_FILENAME


def _load_policy() -> Optional[Dict[str, Any]]:
    """Read and cache the compiled policy. None when it is absent or unreadable.

    Cached per process: a worker handles one task and the policy cannot change underneath
    it, so re-reading on every tool call would buy nothing and cost a syscall per call.
    """
    if "policy" in _CACHE:
        return _CACHE["policy"]
    policy: Optional[Dict[str, Any]] = None
    try:
        path = _policy_path()
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                policy = loaded
    except (OSError, ValueError):
        policy = None
    _CACHE["policy"] = policy
    return policy


def _record(policy: Dict[str, Any], decision, tool_name: str) -> None:
    """Append a governance record for a refusal or an escalation.

    Permitted calls are not recorded: they are the overwhelming majority and recording
    them would bury the events a reviewer is actually looking for. What was refused, and
    what needed a human, is the governance question.
    """
    target = (policy or {}).get("audit_log")
    if not target:
        return
    event = {
        "event_id": uuid.uuid4().hex,
        "correlation_id": os.environ.get("HERMES_KANBAN_TASK", "") or "runtime",
        "ts": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "kind": "policy.decision",
        "phase": "record",
        "actor": "nova-policy-plugin",
        "tenant_id": (policy or {}).get("tenant_id", ""),
        "model_visible": False,
        "subject": (policy or {}).get("agent_id", ""),
        "digest": "",
        "detail": {
            "tool": tool_name,
            "effect": decision.effect,
            "reason": decision.reason,
            "rule": decision.rule,
            "action": decision.action,
            "calls_used": _CALLS_USED,
        },
        "error": "",
    }
    try:
        line = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        handle = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(handle, line)
        finally:
            os.close(handle)
    except OSError:
        # A governance record we cannot write must not stop the decision being enforced.
        pass


def pre_tool_call(tool_name: str = "", args: Optional[Dict[str, Any]] = None, **_: Any):
    """The runtime's policy hook. Returns a directive, or None to proceed.

    Unrecognised returns are ignored by the runtime, so returning ``None`` for a permitted
    call is the correct way to stay out of the way.
    """
    global _CALLS_USED
    try:
        policy = _load_policy()
        decision = decide(policy, tool_name, calls_used=_CALLS_USED)
    except Exception as exc:  # noqa: BLE001 — a policy bug must never permit a call
        return {
            "action": "block",
            "message": (
                f"BLOCKED: the NOVA policy check failed for {tool_name!r} "
                f"({type(exc).__name__}). Refusing rather than proceeding unchecked."
            ),
        }

    # Count only calls that will actually run, and only those the ceiling applies to.
    # Baseline calls are exempt (an agent out of budget must still close its task), and a
    # refused call costs nothing, so neither consumes the budget.
    if decision.effect in (ALLOW, REQUIRE_APPROVAL) and decision.rule != "baseline":
        _CALLS_USED += 1

    if decision.effect == ALLOW:
        return None

    _record(policy or {}, decision, tool_name)

    if decision.effect == REQUIRE_APPROVAL:
        return {
            "action": "approve",
            "message": decision.reason,
            # One allowlist grain per action, so approving "refund" once does not also
            # approve every other escalated action on the same agent.
            "rule_key": f"nova:{decision.action or tool_name}",
        }

    if decision.effect == DENY:
        return {"action": "block", "message": f"BLOCKED by NOVA policy: {decision.reason}"}

    return None


# -- registration -----------------------------------------------------------


def register(ctx: Any) -> None:
    """Wire the policy hook. Called once by the runtime's plugin loader.

    Declaring ``hooks: [pre_tool_call]`` in the manifest is documentation, not wiring: the
    loader calls ``register(ctx)`` and the plugin registers its own callbacks. Without this
    function the loader logs "no register() function", the plugin loads, registers nothing,
    and every tool call proceeds unchecked — a governance control that is installed,
    correct, and never consulted.

    That was the state of this plugin until a live worker run proved it. Unit tests called
    :func:`pre_tool_call` directly, which tested the decision and never tested that anything
    would ever call it; ``tests/platform/test_policy_enforcement.py`` now asserts the
    registration itself.
    """
    ctx.register_hook("pre_tool_call", pre_tool_call)
