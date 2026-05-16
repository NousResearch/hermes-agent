"""ACGS constitutional governance backend.

Treats every tool call as an *action proposal* to be evaluated against a
constitutional ruleset. Each evaluation produces a typed receipt with a
deterministic content hash so the audit trail is verifiable: given the
same rules + the same call, two evaluations produce the same receipt
hash, and a third party can recompute the hash to confirm the run was
not tampered with.

The actual rule evaluation is delegated to an ``ACGSClient`` — anything
satisfying the protocol works. A no-network ``LocalACGSClient`` is
provided for offline development and tests; production deployments would
inject a client that talks to the ACGS service over HTTP/gRPC.

Design choices:

  * Fail-closed. Any client error, schema mismatch, or unmapped verdict
    becomes ``deny`` with the exception surfaced in ``reason``.
  * Deterministic receipts. The hash inputs are sorted JSON; the same
    inputs always produce the same hash regardless of dict ordering.
  * Wire shape stable. ``ACGSDecisionReceipt`` is a frozen dataclass and
    the JSON wire shape is documented — receipts can be emitted to a
    governance store without coupling to runtime internals.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable

from .interfaces import GovernanceContext, GovernanceDecision
from .steps import ToolCall


# ----- rule + receipt types --------------------------------------------------


@dataclass(frozen=True, slots=True)
class ACGSRule:
    """A single constitutional rule.

    ``rule_id`` is a stable identifier (e.g. ``"R-002-no-shell-rm-rf"``).
    ``severity`` is informational — actual blocking is driven by
    ``effect``: ``"deny"`` blocks, ``"require_approval"`` pauses,
    ``"allow"`` is permissive and only meaningful for explicit allowlists.
    """

    rule_id: str
    description: str
    effect: str  # "allow" | "deny" | "require_approval"
    severity: str = "info"  # "info" | "warn" | "block"
    applies_to_tools: tuple[str, ...] = ()  # empty tuple → applies to all tools


@dataclass(frozen=True, slots=True)
class ACGSDecisionReceipt:
    """Verifiable evaluation result for a single proposed action.

    ``receipt_hash`` is sha256 over the canonical JSON of
    ``{rule_set_id, rule_set_hash, tool_name, arguments_hash, verdict, matched_rules}``.
    A consumer can recompute the hash to confirm the receipt was not
    altered after the fact.
    """

    receipt_hash: str
    rule_set_id: str
    rule_set_hash: str
    tool_name: str
    arguments_hash: str
    verdict: str  # "allow" | "deny" | "require_approval"
    reason: str
    matched_rules: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ACGSClient(Protocol):
    """Pluggable evaluator. The local client below is for dev/tests; the
    production client should talk to the ACGS service."""

    rule_set_id: str
    rule_set_hash: str

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: GovernanceContext,
    ) -> ACGSDecisionReceipt: ...


# ----- local (offline) client -------------------------------------------------


class LocalACGSClient:
    """In-process rule evaluator with no network dependency.

    Rules are applied in order. The first ``deny`` rule wins; otherwise
    the first ``require_approval`` rule wins; otherwise ``allow``. Empty
    rule list is fail-closed (``deny``) — this matches the kernel's
    overall stance.

    Suitable for tests, CI, and air-gapped deployments. Production
    callers should replace it with a client that talks to the central
    governance service.
    """

    def __init__(self, rules: Iterable[ACGSRule], *, rule_set_id: str = "local") -> None:
        self._rules = tuple(rules)
        self.rule_set_id = rule_set_id
        self.rule_set_hash = _hash_rules(self._rules)

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: GovernanceContext,
    ) -> ACGSDecisionReceipt:
        applicable = [
            r for r in self._rules
            if not r.applies_to_tools or tool_name in r.applies_to_tools
        ]
        deny = next((r for r in applicable if r.effect == "deny"), None)
        approval = next((r for r in applicable if r.effect == "require_approval"), None)
        allow_rules = [r for r in applicable if r.effect == "allow"]

        if deny is not None:
            verdict = "deny"
            reason = f"rule {deny.rule_id}: {deny.description}"
            matched = (deny.rule_id,)
        elif approval is not None:
            verdict = "require_approval"
            reason = f"rule {approval.rule_id}: {approval.description}"
            matched = (approval.rule_id,)
        elif allow_rules:
            verdict = "allow"
            reason = f"rule {allow_rules[0].rule_id}: {allow_rules[0].description}"
            matched = tuple(r.rule_id for r in allow_rules)
        else:
            # No rule matched. Fail-closed.
            verdict = "deny"
            reason = "no constitutional rule matched (fail-closed default)"
            matched = ()

        args_hash = _hash_json({"tool": tool_name, "arguments": arguments})
        receipt_hash = _hash_json({
            "rule_set_id": self.rule_set_id,
            "rule_set_hash": self.rule_set_hash,
            "tool_name": tool_name,
            "arguments_hash": args_hash,
            "verdict": verdict,
            "matched_rules": list(matched),
        })
        return ACGSDecisionReceipt(
            receipt_hash=receipt_hash,
            rule_set_id=self.rule_set_id,
            rule_set_hash=self.rule_set_hash,
            tool_name=tool_name,
            arguments_hash=args_hash,
            verdict=verdict,
            reason=reason,
            matched_rules=matched,
        )


# ----- governance policy wrapping ACGS ---------------------------------------


class ACGSGovernance:
    """``GovernanceProtocol`` adapter for an ``ACGSClient``.

    Translates ACGS receipts into ``GovernanceDecision`` records and tucks
    the full receipt into the decision's ``policy`` field so the audit
    trail keeps every verifiable receipt hash. Fail-closed: any client
    exception becomes a ``deny`` decision with the exception details on
    ``reason``.
    """

    policy_id = "acgs"

    def __init__(self, client: ACGSClient) -> None:
        self._client = client

    def decide(self, call: ToolCall, context: GovernanceContext) -> GovernanceDecision:
        try:
            receipt = self._client.evaluate(call.name, call.arguments, context)
        except Exception as exc:
            return GovernanceDecision(
                call_id=call.id,
                tool_name=call.name,
                verdict="deny",
                reason=f"acgs_client_error: {type(exc).__name__}: {exc}",
                policy=f"{self.policy_id}:error",
            )

        verdict = receipt.verdict
        if verdict not in ("allow", "deny", "require_approval"):
            # Unmapped verdict — refuse to guess, fail closed.
            return GovernanceDecision(
                call_id=call.id,
                tool_name=call.name,
                verdict="deny",
                reason=f"acgs_unmapped_verdict: {verdict!r}",
                policy=f"{self.policy_id}:{receipt.rule_set_id}",
            )

        return GovernanceDecision(
            call_id=call.id,
            tool_name=call.name,
            verdict=verdict,  # type: ignore[arg-type]
            reason=f"{receipt.reason} [receipt={receipt.receipt_hash[:12]}…]",
            policy=f"{self.policy_id}:{receipt.rule_set_id}@{receipt.rule_set_hash[:8]}",
        )


# ----- config wiring ---------------------------------------------------------


def build_acgs_governance_from_config(config: dict[str, Any]) -> ACGSGovernance:
    """Build an ``ACGSGovernance`` from ``config['agent']['acgs']``.

    Expected shape::

        agent:
          governance: acgs
          acgs:
            rule_set_id: hermes-prod
            rules:
              - rule_id: R-001-no-rm
                description: refuse destructive shell calls
                effect: deny
                applies_to_tools: [shell, exec]
              - rule_id: R-002-default-allow-reads
                description: read-only tools are allowed
                effect: allow
                applies_to_tools: [lookup, search, read_file]

    Production deployments should replace this with one that constructs
    a network-backed ``ACGSClient`` from the same config block.
    """
    block = (config.get("agent", {}) or {}).get("acgs", {}) or {}
    rule_set_id = block.get("rule_set_id", "local")
    raw_rules = block.get("rules", []) or []
    rules = tuple(
        ACGSRule(
            rule_id=r["rule_id"],
            description=r.get("description", ""),
            effect=r["effect"],
            severity=r.get("severity", "info"),
            applies_to_tools=tuple(r.get("applies_to_tools", []) or ()),
        )
        for r in raw_rules
    )
    return ACGSGovernance(LocalACGSClient(rules, rule_set_id=rule_set_id))


# ----- helpers ---------------------------------------------------------------


def _hash_json(payload: Any) -> str:
    """Stable sha256 over canonical JSON (sorted keys, no whitespace)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _hash_rules(rules: tuple[ACGSRule, ...]) -> str:
    return _hash_json([
        {
            "rule_id": r.rule_id,
            "effect": r.effect,
            "severity": r.severity,
            "applies_to_tools": list(r.applies_to_tools),
            "description": r.description,
        }
        for r in rules
    ])
