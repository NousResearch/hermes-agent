"""Thin qwen-ops routed-client helpers.

qwen-ops is a Telegram/Hermes gateway command surface.  It does not own
routing, budget accounting, escalation execution, or authority policy.
"""

from __future__ import annotations

from datetime import datetime, timezone

QWEN_OPS_COMMAND = "qwen-ops"
QWEN_OPS_ALIASES = ("qwen", "qwenops", "qwen_ops")
QWEN_OPS_PROVIDER = "qwen-oauth"
QWEN_OPS_MODEL = "qwen3-coder"
QWEN_OPS_MODEL_COMMAND_TEXT = f"/model {QWEN_OPS_MODEL} --provider {QWEN_OPS_PROVIDER}"

QWEN_OPS_GRANTED_AUTHORITIES = frozenset()

QWEN_OPS_SCOPE_DECLARATION = (
    "qwen-ops is a routed Hermes client for local/advisory/local-ops work. "
    "Hermes remains the routing, budget, and escalation control layer. "
    "qwen-ops does not add merge, approve, comment, label, write, or runner "
    "authority."
)

QWEN_OPS_PROMPT_PREFIX = (
    "[qwen-ops routed client scope]\n"
    "- Stay advisory/local-ops oriented.\n"
    "- Use Hermes shared routing, budget, and tool policy.\n"
    "- Escalation is explicit and packet-only unless shared policy permits execution.\n"
    "- Do not claim production, launch, customer-data, OAuth/banking, money-movement, "
    "tax/accounting, finality, protected approval, GitHub green, or merge readiness."
)

_DEFAULT_ESCALATION_FIELDS = (
    "frontier_ticket_id",
    "target_model_lane",
    "org",
    "repo",
    "pr_numbers",
    "pr_head_sha",
    "blocker_type",
    "why_frontier_is_needed",
    "why_controller_cannot_drain_directly",
    "requested_output_type",
    "expected_next_action_after_frontier_response",
    "state_fingerprint",
    "created_at",
    "status",
)


def qwen_ops_help_text() -> str:
    """Return the qwen-ops scope and usage text."""
    return "\n".join(
        [
            QWEN_OPS_SCOPE_DECLARATION,
            "",
            "Usage:",
            f"- /{QWEN_OPS_COMMAND} or /qwen on: switch this session through shared /model routing.",
            f"- /{QWEN_OPS_COMMAND} <prompt>: switch to qwen-ops, then send the prompt through the normal Hermes turn path.",
            f"- /{QWEN_OPS_COMMAND} escalate [codex|gpt-5.5|opus] <blocker>: render a shared packet only.",
        ]
    )


def qwen_ops_prompt_text(prompt: str) -> str:
    """Wrap a qwen-ops one-shot prompt with scope boundaries."""
    return f"{QWEN_OPS_PROMPT_PREFIX}\n\n{prompt.strip()}"


def is_qwen_ops_escalation(raw_args: str) -> bool:
    """Return True when qwen-ops should render a packet-only escalation."""
    first = (raw_args or "").strip().split(maxsplit=1)[0].lower()
    return first == "escalate"


def _shared_escalation_fields() -> tuple[str, ...]:
    """Load the shared local frontier-ticket field list when present."""
    try:
        from neoengine_local.pr_ci_traffic_controller.controller_contract import (
            FRONTIER_ESCALATION_PACKET_FIELDS,
        )

        fields = tuple(str(field) for field in FRONTIER_ESCALATION_PACKET_FIELDS)
        if fields:
            return fields
    except Exception:
        pass
    return _DEFAULT_ESCALATION_FIELDS


def _split_escalation_request(raw_args: str) -> tuple[str, str]:
    body = (raw_args or "").strip()
    if body.lower().startswith("escalate"):
        body = body[len("escalate") :].strip()
    if not body:
        return "CODEX", "<fill: exact blocker>"
    first, _, rest = body.partition(" ")
    target_key = first.strip().lower()
    target_map = {
        "codex": "CODEX",
        "gpt-5.5": "GPT-5.5",
        "gpt55": "GPT-5.5",
        "gpt-55": "GPT-5.5",
        "opus": "OPUS",
    }
    if target_key in target_map:
        return target_map[target_key], (rest.strip() or "<fill: exact blocker>")
    return "CODEX", body


def _requested_output_type(target: str) -> str:
    if target == "OPUS":
        return "REVIEW_VERDICT"
    if target == "GPT-5.5":
        return "DECISION"
    return "PATCH_INSTRUCTION"


def build_qwen_ops_escalation_packet(raw_args: str) -> str:
    """Render an escalation packet using the shared field names.

    This intentionally does not queue, invoke, or execute anything.  The
    existing shared routing policy must do that work if it is allowed.
    """
    target, blocker = _split_escalation_request(raw_args)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    values = {
        "frontier_ticket_id": "<packet-only-not-queued>",
        "target_model_lane": target,
        "org": "<fill>",
        "repo": "<fill>",
        "pr_numbers": "<fill>",
        "pr_head_sha": "<fill>",
        "blocker_type": "<fill>",
        "why_frontier_is_needed": blocker,
        "why_controller_cannot_drain_directly": (
            "qwen-ops is advisory/local-ops only and has no execution authority."
        ),
        "requested_output_type": _requested_output_type(target),
        "expected_next_action_after_frontier_response": (
            "Route through shared Hermes policy; qwen-ops does not execute."
        ),
        "state_fingerprint": "<fill>",
        "created_at": now,
        "status": "blocked_no_consumer",
    }

    lines = [
        "Shared escalation packet (packet only; no qwen-ops execution)",
        QWEN_OPS_SCOPE_DECLARATION,
        "",
    ]
    for field in _shared_escalation_fields():
        lines.append(f"{field}: {values.get(field, '<fill>')}")
    return "\n".join(lines)
