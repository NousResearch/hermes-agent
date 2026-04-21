from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover - optional dependency at integration time
    Draft202012Validator = None


class GovernanceError(RuntimeError):
    pass


class GovernanceBlocked(GovernanceError):
    pass


class GovernanceEscalationRequired(GovernanceError):
    pass


def _resolve_var(context: dict[str, Any], path: str) -> Any:
    value: Any = context
    for segment in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(segment)
    return value


def _evaluate_logic(expr: Any, context: dict[str, Any]) -> Any:
    if not isinstance(expr, dict):
        return expr

    if "var" in expr:
        return _resolve_var(context, expr["var"])

    if "==" in expr:
        left, right = expr["=="]
        return _evaluate_logic(left, context) == _evaluate_logic(right, context)

    if ">" in expr:
        left, right = expr[">"]
        return _evaluate_logic(left, context) > _evaluate_logic(right, context)

    if "<=" in expr:
        left, right = expr["<="]
        return _evaluate_logic(left, context) <= _evaluate_logic(right, context)

    if "and" in expr:
        return all(_evaluate_logic(item, context) for item in expr["and"])

    if "or" in expr:
        return any(_evaluate_logic(item, context) for item in expr["or"])

    raise GovernanceError(f"Unsupported policy operator: {list(expr.keys())}")


def _render_template(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {"var"}:
            return _resolve_var(context, value["var"])
        return {key: _render_template(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_render_template(item, context) for item in value]
    return value


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


@dataclass
class GovernanceState:
    domain: str = "mixed"
    risk_level: str = "medium"
    contains_reserved_act: bool = False
    requires_personalized_recommendation: bool = False
    materiality_eur: float = 0.0
    deadline_days_remaining: int = 999
    facts_summary: list[str] = field(default_factory=list)
    required_tools_called: bool = True
    primary_sources_verified: bool = True
    conflicts_detected: bool = False
    tool_errors: list[str] = field(default_factory=list)
    source_refs: list[str] = field(default_factory=list)
    contains_sensitive_data: bool = False
    detected_categories: list[str] = field(default_factory=list)
    client_file_execution_requested: bool = False
    supervisor_execution_approved: bool = False
    ongoing_dispute: bool = False
    requested_optimization_scheme: bool = False
    suspected_fraud: bool = False
    fact_treatment_inconsistency: bool = False
    status: str = "ANALYSE_PREPARATOIRE"
    certainty: str = "MOYENNE"
    tools_called: list[str] = field(default_factory=list)
    tool_trace_ids: list[str] = field(default_factory=list)
    tool_call_fingerprints: list[str] = field(default_factory=list)
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    tool_repeat_violations: list[str] = field(default_factory=list)
    policy_rule_hits: list[str] = field(default_factory=list)
    audit_event_ids: list[str] = field(default_factory=list)
    escalation_id: str | None = None
    final_response_repair_count: int = 0

    def build_policy_context(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "classification": {
                "domain": self.domain,
                "risk_level": self.risk_level,
                "contains_reserved_act": self.contains_reserved_act,
                "requires_personalized_recommendation": self.requires_personalized_recommendation,
            },
            "facts": {
                "materiality_eur": self.materiality_eur,
                "deadline_days_remaining": self.deadline_days_remaining,
                "summary": self.facts_summary,
            },
            "verification": {
                "required_tools_called": self.required_tools_called,
                "primary_sources_verified": self.primary_sources_verified,
                "conflicts_detected": self.conflicts_detected,
                "tool_errors": self.tool_errors,
                "tool_errors_count": len(self.tool_errors),
                "source_refs": self.source_refs,
                "tool_call_fingerprints": self.tool_call_fingerprints,
                "repeated_tool_call_violations": self.tool_repeat_violations,
                "repeated_tool_call_violations_count": len(self.tool_repeat_violations),
                "total_tool_calls": len(self.tools_called),
            },
            "privacy": {
                "contains_sensitive_data": self.contains_sensitive_data,
                "detected_categories": self.detected_categories,
            },
            "execution": {
                "client_file_execution_requested": self.client_file_execution_requested,
                "supervisor_execution_approved": self.supervisor_execution_approved,
            },
            "case": {
                "ongoing_dispute": self.ongoing_dispute,
                "requested_optimization_scheme": self.requested_optimization_scheme,
            },
            "signals": {
                "suspected_fraud": self.suspected_fraud,
                "fact_treatment_inconsistency": self.fact_treatment_inconsistency,
            },
            "config": {
                "escalation_materiality_eur": config["escalation_materiality_eur"],
                "deadline_escalation_days": config["deadline_escalation_days"],
                "max_identical_tool_calls": config["max_identical_tool_calls"],
                "max_final_repair_attempts": config["max_final_repair_attempts"],
            },
        }


class GovernanceRuntime:
    _JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*(\{.*\})\s*```$", re.DOTALL)

    def __init__(
        self,
        pack_root: str | Path,
        escalation_materiality_eur: float = 10_000.0,
        deadline_escalation_days: int = 7,
    ) -> None:
        self.pack_root = Path(pack_root)

        self.system_prompt = self._read_text("prompts/system_prompt.production.md")
        self.tool_contracts = self._read_json("contracts/tool_contracts.v1.json")
        self.policy = self._read_json("policies/escalation_policy.v1.json")
        self.final_response_schema = self._read_json("schemas/final_response.schema.json")
        self.policy_context_schema = self._read_json("schemas/policy_context.schema.json")

        self.runtime_contract = self.tool_contracts.get("runtime_contract", {})
        self.config = {
            "escalation_materiality_eur": escalation_materiality_eur,
            "deadline_escalation_days": deadline_escalation_days,
            "max_identical_tool_calls": int(self.runtime_contract.get("max_identical_tool_calls", 2)),
            "max_final_repair_attempts": int(self.runtime_contract.get("max_final_repair_attempts", 2)),
        }
        self.expose_only_registered_tools = bool(
            self.runtime_contract.get("expose_only_registered_tools", False)
        )
        self.allowed_passthrough_tools = set(
            self.runtime_contract.get("allowed_passthrough_tools") or []
        )
        self.strip_unknown_arguments = bool(
            self.runtime_contract.get("strip_unknown_arguments", True)
        )

        self.tool_contract_map = {
            tool["name"]: tool for tool in self.tool_contracts.get("tools", [])
        }

        self._final_validator = (
            Draft202012Validator(self.final_response_schema)
            if Draft202012Validator is not None
            else None
        )
        self._policy_context_validator = (
            Draft202012Validator(self.policy_context_schema)
            if Draft202012Validator is not None
            else None
        )

    def _read_text(self, relative_path: str) -> str:
        return (self.pack_root / relative_path).read_text(encoding="utf-8")

    def _read_json(self, relative_path: str) -> dict[str, Any]:
        return json.loads(self._read_text(relative_path))

    def prepare_tool_definitions(self, tool_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for item in tool_definitions:
            function = (item or {}).get("function") or {}
            tool_name = function.get("name")
            if not tool_name:
                continue

            if (
                self.expose_only_registered_tools
                and tool_name not in self.tool_contract_map
                and tool_name not in self.allowed_passthrough_tools
            ):
                continue

            contract = self.tool_contract_map.get(tool_name)
            if not contract:
                prepared.append(item)
                continue

            description = function.get("description", "").strip()
            must_call_when = contract.get("must_call_when") or []
            usage_rules = contract.get("usage_rules") or []
            alias_rules = contract.get("argument_aliases") or {}
            canonical_args = list(
                ((contract.get("arguments_schema") or {}).get("properties") or {}).keys()
            )

            contract_suffix = [
                "Governance contract:",
                f"- blocks_if_unavailable={str(contract.get('blocks_if_unavailable', False)).lower()}",
            ]
            if must_call_when:
                contract_suffix.append("- must_call_when=" + " | ".join(must_call_when))
            if canonical_args:
                contract_suffix.append("- canonical_arguments=" + ", ".join(canonical_args))
            for rule in usage_rules:
                contract_suffix.append(f"- usage_rule={rule}")
            for canonical_name, aliases in alias_rules.items():
                if aliases:
                    contract_suffix.append(
                        "- compatibility_aliases "
                        + canonical_name
                        + " <= "
                        + ", ".join(aliases)
                    )

            prepared.append({
                **item,
                "function": {
                    **function,
                    "description": (description + "\n\n" + "\n".join(contract_suffix)).strip(),
                    "parameters": contract["arguments_schema"],
                },
            })
        return prepared

    def build_effective_system_prompt(self, base_prompt: str) -> str:
        if not base_prompt:
            return self.system_prompt
        return f"{self.system_prompt}\n\n{base_prompt}".strip()

    def normalize_tool_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raw_arguments = dict(arguments or {})
        contract = self.tool_contract_map.get(tool_name)
        if not contract:
            return raw_arguments

        properties = ((contract.get("arguments_schema") or {}).get("properties") or {})
        aliases = contract.get("argument_aliases") or {}
        normalized: dict[str, Any] = {}

        for canonical_name in properties:
            if canonical_name in raw_arguments:
                normalized[canonical_name] = raw_arguments[canonical_name]
                continue
            for alias in aliases.get(canonical_name, []):
                if alias in raw_arguments:
                    normalized[canonical_name] = raw_arguments[alias]
                    break

        if not self.strip_unknown_arguments:
            for key, value in raw_arguments.items():
                normalized.setdefault(key, value)

        return normalized

    def build_tool_call_fingerprint(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> str:
        normalized_arguments = self.normalize_tool_arguments(tool_name, arguments)
        payload = {
            "tool_name": tool_name,
            "arguments": normalized_arguments,
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]
        return f"{tool_name}:{digest}"

    def _record_tool_call_fingerprint(
        self,
        state: GovernanceState,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], str, int]:
        normalized_arguments = self.normalize_tool_arguments(tool_name, arguments)
        fingerprint = self.build_tool_call_fingerprint(tool_name, normalized_arguments)
        count = state.tool_call_counts.get(fingerprint, 0) + 1
        state.tool_call_counts[fingerprint] = count
        state.tool_call_fingerprints.append(fingerprint)
        return normalized_arguments, fingerprint, count

    def _register_repeat_violation(
        self,
        state: GovernanceState,
        tool_name: str,
        fingerprint: str,
        reason_code: str,
    ) -> None:
        violation = f"{fingerprint}:{reason_code}"
        if violation not in state.tool_repeat_violations:
            state.tool_repeat_violations.append(violation)

    def update_state_from_tool_result(
        self,
        state: GovernanceState,
        tool_name: str,
        arguments: dict[str, Any] | None,
        result_raw: str,
    ) -> None:
        normalized_arguments, fingerprint, count = self._record_tool_call_fingerprint(
            state,
            tool_name,
            arguments,
        )
        state.tools_called.append(tool_name)

        parsed: dict[str, Any] | None = None
        try:
            parsed = json.loads(result_raw)
        except Exception:
            state.tool_errors.append(f"{tool_name}: invalid_json_result")
            if count > self.config["max_identical_tool_calls"]:
                self._register_repeat_violation(
                    state,
                    tool_name,
                    fingerprint,
                    "identical_call_budget_exceeded",
                )
            return

        trace_id = parsed.get("trace_id")
        if isinstance(trace_id, str) and trace_id:
            state.tool_trace_ids.append(trace_id)

        if parsed.get("ok") is False or parsed.get("error_code") or parsed.get("error"):
            error_msg = parsed.get("error_message") or parsed.get("error") or "tool_error"
            state.tool_errors.append(f"{tool_name}: {error_msg}")
            if count > self.config["max_identical_tool_calls"]:
                self._register_repeat_violation(
                    state,
                    tool_name,
                    fingerprint,
                    "identical_call_budget_exceeded",
                )
            return

        result = parsed.get("result") or {}
        sources = result.get("sources") or []
        for source in sources:
            if isinstance(source, dict):
                ref = source.get("reference") or source.get("id") or source.get("url")
                if isinstance(ref, str):
                    state.source_refs.append(ref)

        if result.get("primary_sources_verified") is False:
            state.primary_sources_verified = False
        if result.get("coverage_status") in {"partial", "not_verified"}:
            state.primary_sources_verified = False
        if result.get("conflicts_detected") is True:
            state.conflicts_detected = True

        if tool_name == "escalate_to_human_supervisor":
            escalation_id = result.get("escalation_id")
            if isinstance(escalation_id, str):
                state.escalation_id = escalation_id

        if tool_name == "log_audit_event":
            audit_event_id = result.get("audit_event_id")
            if isinstance(audit_event_id, str):
                state.audit_event_ids.append(audit_event_id)

        useful_result = bool(result) and (
            bool(sources)
            or result.get("coverage_status") in {"verified", "partial"}
            or result.get("amount") is not None
            or result.get("records") is not None
        )
        if useful_result and count > 1:
            self._register_repeat_violation(
                state,
                tool_name,
                fingerprint,
                "useful_result_repeated",
            )
        elif count > self.config["max_identical_tool_calls"]:
            self._register_repeat_violation(
                state,
                tool_name,
                fingerprint,
                "identical_call_budget_exceeded",
            )

        _ = normalized_arguments

    def evaluate_policy(self, state: GovernanceState) -> dict[str, Any]:
        context = state.build_policy_context(self.config)
        if self._policy_context_validator is not None:
            self._policy_context_validator.validate(context)

        for rule in self.policy.get("rules", []):
            if _evaluate_logic(rule["when"], context):
                state.policy_rule_hits.append(rule["id"])
                actions = []
                for action in rule.get("actions", []):
                    rendered = dict(action)
                    if "args_template" in action:
                        rendered["args"] = _render_template(action["args_template"], context)
                    actions.append(rendered)
                return {
                    "matched": True,
                    "rule_id": rule["id"],
                    "severity": rule.get("severity"),
                    "terminal": bool(rule.get("terminal")),
                    "actions": actions,
                }

        return {
            "matched": False,
            "rule_id": None,
            "severity": None,
            "terminal": False,
            "actions": [],
            **self.policy.get("default_outcome", {}),
        }

    def apply_policy_outcome_to_state(
        self,
        state: GovernanceState,
        outcome: dict[str, Any],
    ) -> dict[str, Any]:
        tool_calls: list[dict[str, Any]] = []
        stop_generation = False
        for action in outcome.get("actions", []):
            action_type = action["type"]
            if action_type == "set_status":
                state.status = action["value"]
            elif action_type == "set_certainty":
                state.certainty = action["value"]
            elif action_type == "call_tool":
                tool_calls.append({
                    "name": action["tool"],
                    "arguments": action.get("args", {}),
                })
            elif action_type == "stop_generation":
                stop_generation = bool(action["value"])

        if not outcome.get("matched"):
            if "status" in outcome:
                state.status = outcome["status"]
            if "certainty" in outcome:
                state.certainty = outcome["certainty"]

        return {
            "tool_calls": tool_calls,
            "stop_generation": stop_generation,
        }

    def _extract_json_payload(self, text: str) -> dict[str, Any]:
        candidate = (text or "").strip()
        if not candidate:
            raise GovernanceBlocked("final_response_empty")

        fenced_match = self._JSON_FENCE_RE.fullmatch(candidate)
        if fenced_match:
            candidate = fenced_match.group(1).strip()

        decoder = json.JSONDecoder()
        try:
            payload, end_index = decoder.raw_decode(candidate)
        except Exception as exc:
            raise GovernanceBlocked(f"final_response_invalid_json: {exc}") from exc

        if not isinstance(payload, dict):
            raise GovernanceBlocked("final_response_root_not_object")
        if candidate[end_index:].strip():
            raise GovernanceBlocked("final_response_mixed_content")
        return payload

    def validate_final_response_text(self, text: str) -> dict[str, Any]:
        payload = self._extract_json_payload(text)

        if self._final_validator is not None:
            errors = sorted(self._final_validator.iter_errors(payload), key=lambda e: list(e.path))
            if errors:
                first = errors[0]
                path = ".".join(str(part) for part in first.path) or "<root>"
                raise GovernanceBlocked(f"final_response_schema_error at {path}: {first.message}")

        return payload

    def register_final_response_repair_attempt(self, state: GovernanceState) -> None:
        state.final_response_repair_count += 1
        if state.final_response_repair_count > self.config["max_final_repair_attempts"]:
            raise GovernanceBlocked("final_response_repair_budget_exhausted")

    def build_repair_message(self, error_message: str) -> str:
        required_keys = [
            "status",
            "certainty",
            "scope",
            "facts",
            "sources",
            "analysis",
            "risks",
            "next_action",
            "audit_trail",
        ]
        return (
            "Previous final response is invalid.\n"
            "Return exactly one JSON object and nothing else.\n"
            "Do not use markdown, prose, headings, bullets or code fences.\n"
            "Do not call another tool unless a mandatory source is still missing.\n"
            "Required root keys: "
            + ", ".join(required_keys)
            + ".\n"
            f"Validation error: {error_message}"
        )

    def build_blocked_final_response(
        self,
        state: GovernanceState,
        reason_message: str,
        *,
        jurisdiction: str = "FR",
        fact_date: str | None = None,
    ) -> str:
        now = _utc_now_iso()
        payload = {
            "status": state.status,
            "certainty": state.certainty,
            "scope": {
                "domain": state.domain,
                "jurisdiction": jurisdiction,
                "fact_date": fact_date,
                "source_checked_at": now,
                "risk_level": state.risk_level,
            },
            "facts": {
                "verified_facts": state.facts_summary or ["Blocage avant consolidation complète des faits."],
                "assumptions": [],
                "missing_facts": [],
            },
            "sources": [
                {
                    "source_type": "other",
                    "reference": "governance://runtime/policy-block",
                    "effective_date": None,
                    "checked_at": now,
                    "verified": True,
                    "url": None,
                }
            ],
            "analysis": {
                "mode": "blocked",
                "major": "Blocage imposé par la gouvernance Hermes.",
                "minor": reason_message,
                "conclusion": "La génération finale est suspendue jusqu'à correction ou revue humaine.",
            },
            "calculations": [],
            "entries": [],
            "risks": [],
            "next_action": {
                "type": "escalate" if state.status == "ESCALADE_REQUISE" else "blocked",
                "message": reason_message,
                "escalation_id": state.escalation_id,
            },
            "audit_trail": {
                "tools_called": state.tools_called,
                "tool_trace_ids": state.tool_trace_ids,
                "tool_call_fingerprints": state.tool_call_fingerprints,
                "repetition_violations": state.tool_repeat_violations,
                "policy_rule_hits": state.policy_rule_hits,
                "audit_event_ids": state.audit_event_ids,
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def ensure_contract_tools_exist(self, available_tool_names: Iterable[str]) -> list[str]:
        available = set(available_tool_names)
        missing = [
            name
            for name, contract in self.tool_contract_map.items()
            if name not in available and contract.get("blocks_if_unavailable")
        ]
        return missing
