"""ACGS-Lite constitutional governance backend.

Adapter for the `acgs-lite <https://github.com/dislovelhl/acgs-lite>`_
fail-closed legitimacy layer. Every tool call the model proposes is
evaluated against a typed ``Constitution`` (an identified, hash-pinned
rule set); the evaluator returns one of eight ACGS-Lite verdicts plus a
replayable ``ACGSDecisionReceipt``. The kernel's three-verdict gate
(``allow`` / ``deny`` / ``require_approval``) is the downstream
projection, so the kernel keeps its narrow protocol while the audit
trail keeps every ACGS verdict and receipt hash.

Design choices:

  * Fail-closed by construction. Any client exception, schema mismatch,
    unmapped verdict, or missing receipt becomes ``deny`` with the
    cause on ``reason`` — matches ACGS-Lite's "fail-closed on any
    missing/unverifiable input" stance.
  * Deterministic receipts. ``receipt_hash`` is sha256 over canonical
    JSON of the inputs, so two evaluations with the same constitution
    + tool + arguments produce the same hash regardless of dict key
    ordering.
  * Pluggable client. ``ACGSClient`` is a Protocol; production
    deployments inject a client that calls the ACGS service (HTTP /
    MCP / in-process), and the kernel never reaches into either
    network code or the upstream ``acgs-lite`` package.
  * Decision taxonomy mirrors ACGS-Lite v2.x:

        ALLOW
        ALLOW_WITH_CONTROLS
        TRANSFORM_REQUIRED
        REPLAN_REQUIRED
        STRUCTURED_REVIEW_REQUIRED
        DENY_OPERATION_WITH_ALTERNATIVE
        DENY_GOAL
        HARD_DENY

    These are projected onto the kernel's ``GovernanceVerdict``:

        ALLOW, ALLOW_WITH_CONTROLS                              → "allow"
        STRUCTURED_REVIEW_REQUIRED                              → "require_approval"
        TRANSFORM_REQUIRED, REPLAN_REQUIRED,
        DENY_OPERATION_WITH_ALTERNATIVE, DENY_GOAL, HARD_DENY   → "deny"

    The original ACGS verdict + receipt hash + matched rule ids ride
    along on the ``GovernanceDecision.reason`` / ``policy`` fields so
    nothing is lost.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol, runtime_checkable

from .interfaces import GovernanceContext, GovernanceDecision
from .steps import ToolCall


# ----- decision taxonomy -----------------------------------------------------


ACGSVerdict = Literal[
    "ALLOW",
    "ALLOW_WITH_CONTROLS",
    "TRANSFORM_REQUIRED",
    "REPLAN_REQUIRED",
    "STRUCTURED_REVIEW_REQUIRED",
    "DENY_OPERATION_WITH_ALTERNATIVE",
    "DENY_GOAL",
    "HARD_DENY",
]

Severity = Literal["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


# Map ACGS-Lite verdicts onto the kernel's narrow gate verdicts.
_KERNEL_VERDICT: dict[str, str] = {
    "ALLOW": "allow",
    "ALLOW_WITH_CONTROLS": "allow",
    "STRUCTURED_REVIEW_REQUIRED": "require_approval",
    "TRANSFORM_REQUIRED": "deny",
    "REPLAN_REQUIRED": "deny",
    "DENY_OPERATION_WITH_ALTERNATIVE": "deny",
    "DENY_GOAL": "deny",
    "HARD_DENY": "deny",
}


# ----- constitution + rule types ---------------------------------------------


@dataclass(frozen=True, slots=True)
class ACGSRule:
    """A single constitutional rule.

    ``id`` is the stable identifier surfaced on receipts (e.g. ``no-pii``).
    ``pattern`` is a regex matched against the canonical text composed
    from a tool call (``"<tool_name> <canonical-json-args>"``).
    ``effect`` is the ACGS verdict to return when ``pattern`` matches.
    ``severity`` is informational metadata for downstream telemetry.
    """

    id: str
    pattern: str
    description: str = ""
    effect: ACGSVerdict = "HARD_DENY"
    severity: Severity = "MEDIUM"
    applies_to_tools: tuple[str, ...] = ()  # empty → applies to all tools


@dataclass(frozen=True, slots=True)
class Constitution:
    """A versioned, hash-pinned constitutional rule set.

    Mirrors ACGS-Lite's ``Constitution`` shape — the ``constitutional_hash``
    is what auditors and replay verifiers anchor on. If a caller supplies
    an explicit hash that disagrees with the computed hash, the
    constitution raises at construction time (fail-closed on tamper).
    """

    constitutional_hash: str
    rules: tuple[ACGSRule, ...] = ()

    @staticmethod
    def from_yaml_dict(data: dict[str, Any]) -> "Constitution":
        """Build from the ACGS-Lite YAML shape (already parsed to dict).

        Expected shape::

            constitutional_hash: "608508a9bd224290"
            rules:
              - id: no-pii
                pattern: "SSN|passport|social security"
                severity: CRITICAL
                description: Prevent PII leakage
                effect: HARD_DENY            # optional, defaults HARD_DENY
                applies_to_tools: []         # optional, empty → all tools
        """
        raw_rules = data.get("rules", []) or []
        rules = tuple(
            ACGSRule(
                id=r["id"],
                pattern=r["pattern"],
                description=r.get("description", ""),
                effect=r.get("effect", "HARD_DENY"),
                severity=r.get("severity", "MEDIUM"),
                applies_to_tools=tuple(r.get("applies_to_tools", []) or ()),
            )
            for r in raw_rules
        )
        declared = data.get("constitutional_hash") or ""
        computed = _hash_rules(rules)
        if declared and declared != computed:
            # Fail-closed on declared/computed mismatch. The caller is
            # claiming a constitution hash that doesn't match the rules
            # they shipped — refuse rather than silently re-anchoring.
            raise ValueError(
                f"constitutional_hash mismatch: declared={declared!r}, "
                f"computed={computed!r}"
            )
        return Constitution(constitutional_hash=declared or computed, rules=rules)


# ----- receipts --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ACGSDecisionReceipt:
    """Verifiable evaluation result for one proposed action.

    The receipt is **emitted before execution**, per ACGS-Lite's
    "replayable receipt before execution" guarantee. ``receipt_hash`` is
    sha256 over the canonical JSON of:

        {constitutional_hash, tool_name, arguments_hash, verdict, matched_rules}

    A third party can recompute the hash with the same inputs and detect
    any post-hoc edit to the receipt.
    """

    receipt_hash: str
    constitutional_hash: str
    tool_name: str
    arguments_hash: str
    verdict: ACGSVerdict
    reason: str
    matched_rules: tuple[str, ...] = ()
    severities: tuple[Severity, ...] = ()
    controls: tuple[str, ...] = ()  # set when verdict == ALLOW_WITH_CONTROLS
    extra: dict[str, Any] = field(default_factory=dict)


# ----- client protocol -------------------------------------------------------


@runtime_checkable
class ACGSClient(Protocol):
    """Pluggable evaluator. ``LocalACGSClient`` below is for dev/tests; a
    production deployment should inject a client that delegates to the
    central ACGS-Lite service (or runs the upstream ``acgs-lite``
    package in-process)."""

    constitutional_hash: str

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: GovernanceContext,
    ) -> ACGSDecisionReceipt: ...


# ----- local (offline) client -----------------------------------------------


class LocalACGSClient:
    """In-process evaluator with no network dependency.

    Composes the matchable text as ``"<tool_name> <canonical-json-args>"``
    and runs each rule's regex pattern against it. The first matching
    rule wins by precedence: HARD_DENY > DENY_* > REPLAN_REQUIRED >
    TRANSFORM_REQUIRED > STRUCTURED_REVIEW_REQUIRED > ALLOW_WITH_CONTROLS
    > ALLOW. With no matching rule the evaluator emits ``HARD_DENY``
    (fail-closed default — same stance ACGS-Lite takes on missing
    legitimacy inputs).

    Suitable for tests, CI, and air-gapped use. Production callers
    should swap in a service-backed ``ACGSClient``.
    """

    # Higher index = stronger refusal. Used to pick a winner when
    # several rules match the same call.
    _PRECEDENCE: tuple[ACGSVerdict, ...] = (
        "ALLOW",
        "ALLOW_WITH_CONTROLS",
        "STRUCTURED_REVIEW_REQUIRED",
        "TRANSFORM_REQUIRED",
        "REPLAN_REQUIRED",
        "DENY_OPERATION_WITH_ALTERNATIVE",
        "DENY_GOAL",
        "HARD_DENY",
    )

    def __init__(self, constitution: Constitution) -> None:
        self.constitution = constitution
        self.constitutional_hash = constitution.constitutional_hash
        # Pre-compile patterns so a malformed regex surfaces at init,
        # not deep inside an action step.
        self._compiled: list[tuple[ACGSRule, re.Pattern[str]]] = [
            (r, re.compile(r.pattern, re.IGNORECASE)) for r in constitution.rules
        ]

    def evaluate(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: GovernanceContext,
    ) -> ACGSDecisionReceipt:
        haystack = _compose_haystack(tool_name, arguments)
        matched: list[tuple[ACGSRule, ACGSVerdict]] = []
        for rule, pattern in self._compiled:
            if rule.applies_to_tools and tool_name not in rule.applies_to_tools:
                continue
            if pattern.search(haystack):
                matched.append((rule, rule.effect))

        if not matched:
            verdict: ACGSVerdict = "HARD_DENY"
            reason = "no constitutional rule matched (fail-closed default)"
            matched_ids: tuple[str, ...] = ()
            severities: tuple[Severity, ...] = ()
            controls: tuple[str, ...] = ()
        else:
            # Winner is the matched rule with the strongest refusal.
            ranked = sorted(matched, key=lambda m: self._PRECEDENCE.index(m[1]))
            winning_rule, winning_verdict = ranked[-1]
            verdict = winning_verdict
            reason = f"rule {winning_rule.id}: {winning_rule.description or winning_rule.pattern}"
            matched_ids = tuple(r.id for r, _ in matched)
            severities = tuple(r.severity for r, _ in matched)
            controls = ()
            if verdict == "ALLOW_WITH_CONTROLS":
                # Concrete controls would be defined per-rule in a fuller
                # client. The local evaluator surfaces the matched rule
                # ids as the controls that should be enforced downstream.
                controls = matched_ids

        arguments_hash = _hash_json({"tool": tool_name, "arguments": arguments})
        receipt_hash = _hash_json({
            "constitutional_hash": self.constitutional_hash,
            "tool_name": tool_name,
            "arguments_hash": arguments_hash,
            "verdict": verdict,
            "matched_rules": list(matched_ids),
        })
        return ACGSDecisionReceipt(
            receipt_hash=receipt_hash,
            constitutional_hash=self.constitutional_hash,
            tool_name=tool_name,
            arguments_hash=arguments_hash,
            verdict=verdict,
            reason=reason,
            matched_rules=matched_ids,
            severities=severities,
            controls=controls,
        )


# ----- governance policy wrapping ACGS ---------------------------------------


class ACGSGovernance:
    """``GovernanceProtocol`` adapter for an ``ACGSClient``.

    Translates ACGS-Lite receipts into the kernel's 3-verdict
    ``GovernanceDecision`` shape. The original ACGS verdict, receipt
    hash prefix, and matched rule ids ride along on ``reason`` /
    ``policy`` so the audit trail keeps full fidelity. Fail-closed on
    every error path:

      * client exception          → kernel verdict ``deny``
      * unmapped ACGS verdict     → kernel verdict ``deny``
      * receipt with empty hash   → kernel verdict ``deny``
    """

    policy_id = "acgs-lite"

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

        if not receipt.receipt_hash:
            return GovernanceDecision(
                call_id=call.id,
                tool_name=call.name,
                verdict="deny",
                reason="acgs_missing_receipt_hash (fail-closed)",
                policy=f"{self.policy_id}:{receipt.constitutional_hash[:8]}",
            )

        kernel_verdict = _KERNEL_VERDICT.get(receipt.verdict)
        if kernel_verdict is None:
            return GovernanceDecision(
                call_id=call.id,
                tool_name=call.name,
                verdict="deny",
                reason=f"acgs_unmapped_verdict: {receipt.verdict!r}",
                policy=f"{self.policy_id}:{receipt.constitutional_hash[:8]}",
            )

        suffix_parts = [f"acgs={receipt.verdict}", f"receipt={receipt.receipt_hash[:12]}"]
        if receipt.matched_rules:
            suffix_parts.append(f"rules={','.join(receipt.matched_rules)}")
        if receipt.controls:
            suffix_parts.append(f"controls={','.join(receipt.controls)}")
        return GovernanceDecision(
            call_id=call.id,
            tool_name=call.name,
            verdict=kernel_verdict,  # type: ignore[arg-type]
            reason=f"{receipt.reason} [{' '.join(suffix_parts)}]",
            policy=f"{self.policy_id}:{receipt.constitutional_hash[:16]}",
        )


# ----- config wiring ---------------------------------------------------------


def build_acgs_governance_from_config(config: dict[str, Any]) -> ACGSGovernance:
    """Build an ``ACGSGovernance`` from an ACGS-Lite-shaped config block.

    Expected shape in hermes config::

        agent:
          governance: acgs
          acgs:
            constitutional_hash: "608508a9bd224290"   # optional; computed if absent
            rules:
              - id: no-pii
                pattern: "SSN|passport|social security"
                severity: CRITICAL
                description: Prevent PII leakage
                effect: HARD_DENY
              - id: require-approval-transfer
                pattern: "transfer|payment|wire"
                severity: HIGH
                effect: STRUCTURED_REVIEW_REQUIRED
                applies_to_tools: [bank_transfer]

    Production deployments should replace this with a builder that
    constructs a service-backed ``ACGSClient`` from the same block —
    e.g. ``HTTPACGSClient(endpoint=config['acgs']['endpoint'])``.
    """
    block = (config.get("agent", {}) or {}).get("acgs", {}) or {}
    constitution = Constitution.from_yaml_dict(block)
    return ACGSGovernance(LocalACGSClient(constitution))


# ----- helpers ---------------------------------------------------------------


def _compose_haystack(tool_name: str, arguments: dict[str, Any]) -> str:
    args_blob = json.dumps(arguments, sort_keys=True, default=str, ensure_ascii=False)
    return f"{tool_name} {args_blob}"


def _hash_json(payload: Any) -> str:
    """Stable sha256 over canonical JSON (sorted keys, no whitespace)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _hash_rules(rules: tuple[ACGSRule, ...]) -> str:
    return _hash_json([
        {
            "id": r.id,
            "pattern": r.pattern,
            "effect": r.effect,
            "severity": r.severity,
            "description": r.description,
            "applies_to_tools": list(r.applies_to_tools),
        }
        for r in rules
    ])
