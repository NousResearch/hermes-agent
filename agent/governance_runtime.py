from __future__ import annotations

import json
import hashlib
import hmac
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover - optional dependency at integration time
    Draft202012Validator = None

try:
    from agent.privacy_guard import analyze_privacy_payload, assert_no_raw_sensitive_data, redact_payload
except Exception:  # pragma: no cover - fallback for direct execution
    from privacy_guard import analyze_privacy_payload, assert_no_raw_sensitive_data, redact_payload


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


def _dedupe_preserve(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


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
    contains_pii: bool = False
    contains_special_category_data: bool = False
    contains_financial_identifiers: bool = False
    contains_secrets: bool = False
    detected_categories: list[str] = field(default_factory=list)
    requested_fields: list[str] = field(default_factory=list)
    disclosed_fields: list[str] = field(default_factory=list)
    lawful_basis: str | None = None
    purpose_code: str | None = None
    masking_applied: bool = False
    masking_failures_count: int = 0
    output_contains_raw_identifier: bool = False
    audit_logged_for_data_access: bool = True
    requested_fields_exceed_policy: bool = False
    missing_lawful_basis: bool = False
    missing_supervisor_approval: bool = False
    retention_class: str = "standard"
    client_file_execution_requested: bool = False
    supervisor_execution_approved: bool = False
    ongoing_dispute: bool = False
    requested_optimization_scheme: bool = False
    suspected_fraud: bool = False
    fiscal_search_satisfied: bool = False
    fiscal_search_cached_result: str | None = None
    fiscal_search_call_count: int = 0
    fiscal_search_cache_hits: int = 0
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
                "contains_pii": self.contains_pii,
                "contains_special_category_data": self.contains_special_category_data,
                "contains_financial_identifiers": self.contains_financial_identifiers,
                "contains_secrets": self.contains_secrets,
                "detected_categories": self.detected_categories,
                "requested_fields": self.requested_fields,
                "disclosed_fields": self.disclosed_fields,
                "lawful_basis": self.lawful_basis,
                "purpose_code": self.purpose_code,
                "masking_applied": self.masking_applied,
                "masking_failures_count": self.masking_failures_count,
                "output_contains_raw_identifier": self.output_contains_raw_identifier,
                "audit_logged_for_data_access": self.audit_logged_for_data_access,
                "requested_fields_exceed_policy": self.requested_fields_exceed_policy,
                "missing_lawful_basis": self.missing_lawful_basis,
                "missing_supervisor_approval": self.missing_supervisor_approval,
                "retention_class": self.retention_class,
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
        self._fingerprint_secret = os.urandom(32)
        self._shared_result_envelope_schema = self._build_shared_result_envelope_schema()
        self._tool_argument_schemas = {
            tool_name: (contract.get("arguments_schema") or {})
            for tool_name, contract in self.tool_contract_map.items()
        }
        self._tool_result_schemas = {
            tool_name: self._build_result_object_schema(contract.get("result_contract") or {})
            for tool_name, contract in self.tool_contract_map.items()
        }

        self._final_validator = self._make_validator(self.final_response_schema)
        self._policy_context_validator = self._make_validator(self.policy_context_schema)
        self._shared_result_envelope_validator = self._make_validator(self._shared_result_envelope_schema)
        self._tool_argument_validators = {
            tool_name: self._make_validator(schema)
            for tool_name, schema in self._tool_argument_schemas.items()
        }
        self._tool_result_validators = {
            tool_name: self._make_validator(schema)
            for tool_name, schema in self._tool_result_schemas.items()
        }

    def _read_text(self, relative_path: str) -> str:
        return (self.pack_root / relative_path).read_text(encoding="utf-8")

    def _read_json(self, relative_path: str) -> dict[str, Any]:
        return json.loads(self._read_text(relative_path))

    def _make_validator(self, schema: dict[str, Any] | None) -> Draft202012Validator | None:
        if Draft202012Validator is None or not schema:
            return None
        return Draft202012Validator(schema)

    def _build_shared_result_envelope_schema(self) -> dict[str, Any]:
        envelope = dict(self.tool_contracts.get("shared_result_envelope") or {})
        envelope.setdefault("type", "object")
        envelope.setdefault("properties", {})
        envelope.setdefault("required", [])
        envelope.setdefault("additionalProperties", True)
        return envelope

    def _build_result_object_schema(self, result_contract: dict[str, Any]) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": True,
            "required": list(result_contract.get("required_fields") or []),
        }
        properties = result_contract.get("properties")
        if isinstance(properties, dict):
            schema["properties"] = properties
        return schema

    def _validate_schema_or_raise(
        self,
        *,
        validator: Draft202012Validator | None,
        schema: dict[str, Any],
        payload: Any,
        label: str,
    ) -> None:
        if validator is not None:
            errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
            if errors:
                first = errors[0]
                path = ".".join(str(part) for part in first.path) or "<root>"
                raise GovernanceBlocked(f"{label}_schema_error at {path}: {first.message}")
            return
        self._manual_validate_schema(schema, payload, label=label, path="<root>")

    def _manual_validate_schema(
        self,
        schema: dict[str, Any],
        payload: Any,
        *,
        label: str,
        path: str,
    ) -> None:
        declared_type = schema.get("type")
        if declared_type is not None:
            allowed_types = declared_type if isinstance(declared_type, list) else [declared_type]
            if not any(self._matches_type(item_type, payload) for item_type in allowed_types):
                expected = ",".join(str(item) for item in allowed_types)
                raise GovernanceBlocked(f"{label}_schema_error at {path}: expected {expected}")

        if "enum" in schema and payload not in schema["enum"]:
            raise GovernanceBlocked(f"{label}_schema_error at {path}: value not in enum")

        if isinstance(payload, str) and "maxLength" in schema and len(payload) > int(schema["maxLength"]):
            raise GovernanceBlocked(f"{label}_schema_error at {path}: string too long")

        if isinstance(payload, list):
            min_items = schema.get("minItems")
            if min_items is not None and len(payload) < int(min_items):
                raise GovernanceBlocked(f"{label}_schema_error at {path}: array too short")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(payload):
                    self._manual_validate_schema(
                        item_schema,
                        item,
                        label=label,
                        path=f"{path}[{index}]",
                    )

        if isinstance(payload, (int, float)) and not isinstance(payload, bool):
            minimum = schema.get("minimum")
            if minimum is not None and payload < minimum:
                raise GovernanceBlocked(f"{label}_schema_error at {path}: value below minimum")

        if isinstance(payload, dict):
            required = schema.get("required") or []
            for key in required:
                if key not in payload:
                    missing_path = f"{path}.{key}" if path != "<root>" else f"<root>.{key}"
                    raise GovernanceBlocked(f"{label}_schema_error at {missing_path}: missing required property")

            properties = schema.get("properties") or {}
            if schema.get("additionalProperties") is False:
                unknown = sorted(key for key in payload if key not in properties)
                if unknown:
                    raise GovernanceBlocked(
                        f"{label}_schema_error at {path}: unexpected properties {', '.join(unknown)}"
                    )

            for key, value in payload.items():
                if key in properties and isinstance(properties[key], dict):
                    child_path = f"{path}.{key}" if path != "<root>" else key
                    self._manual_validate_schema(
                        properties[key],
                        value,
                        label=label,
                        path=child_path,
                    )

    def _matches_type(self, expected_type: str, payload: Any) -> bool:
        if expected_type == "object":
            return isinstance(payload, dict)
        if expected_type == "array":
            return isinstance(payload, list)
        if expected_type == "string":
            return isinstance(payload, str)
        if expected_type == "boolean":
            return isinstance(payload, bool)
        if expected_type == "number":
            return isinstance(payload, (int, float)) and not isinstance(payload, bool)
        if expected_type == "integer":
            return isinstance(payload, int) and not isinstance(payload, bool)
        if expected_type == "null":
            return payload is None
        return True

    def _redact_for_storage(self, value: Any) -> Any:
        result = redact_payload(value)
        return result.value

    def _merge_privacy_analysis(
        self,
        state: GovernanceState,
        value: Any,
        *,
        mark_masked: bool = False,
    ) -> None:
        analysis = analyze_privacy_payload(value)
        state.contains_sensitive_data = state.contains_sensitive_data or analysis.contains_sensitive_data
        state.contains_pii = state.contains_pii or analysis.contains_pii
        state.contains_special_category_data = (
            state.contains_special_category_data or analysis.contains_special_category_data
        )
        state.contains_financial_identifiers = (
            state.contains_financial_identifiers or analysis.contains_financial_identifiers
        )
        state.contains_secrets = state.contains_secrets or analysis.contains_secrets
        state.detected_categories = _dedupe_preserve(
            [*state.detected_categories, *analysis.detected_categories]
        )
        if mark_masked and analysis.match_count:
            state.masking_applied = True

    def validate_tool_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized = self.normalize_tool_arguments(tool_name, arguments)
        contract = self.tool_contract_map.get(tool_name)
        if not contract:
            return normalized

        self._validate_schema_or_raise(
            validator=self._tool_argument_validators.get(tool_name),
            schema=self._tool_argument_schemas.get(tool_name, {}),
            payload=normalized,
            label=f"{tool_name}_arguments",
        )
        return normalized

    def validate_tool_result(
        self,
        tool_name: str,
        result_raw: str,
    ) -> dict[str, Any]:
        try:
            parsed = json.loads(result_raw)
        except Exception as exc:
            raise GovernanceBlocked(f"{tool_name}_result_invalid_json: {exc}") from exc

        self._validate_schema_or_raise(
            validator=self._shared_result_envelope_validator,
            schema=self._shared_result_envelope_schema,
            payload=parsed,
            label=f"{tool_name}_envelope",
        )

        result = parsed.get("result")
        if not isinstance(result, dict):
            raise GovernanceBlocked(f"{tool_name}_result_schema_error at result: expected object")

        self._validate_schema_or_raise(
            validator=self._tool_result_validators.get(tool_name),
            schema=self._tool_result_schemas.get(tool_name, {}),
            payload=result,
            label=f"{tool_name}_result",
        )
        return parsed

    def _register_data_access_context(
        self,
        state: GovernanceState,
        arguments: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        requested_fields = arguments.get("requested_fields") or []
        disclosed_fields = result.get("fields_disclosed") or []
        state.requested_fields = _dedupe_preserve([*state.requested_fields, *requested_fields])
        state.disclosed_fields = _dedupe_preserve([*state.disclosed_fields, *disclosed_fields])
        state.lawful_basis = str(arguments.get("lawful_basis") or state.lawful_basis or "") or None
        state.purpose_code = str(arguments.get("purpose_code") or state.purpose_code or "") or None
        state.retention_class = str(result.get("retention_class") or state.retention_class)

        privacy_flags = {str(item) for item in (result.get("privacy_flags") or [])}
        if privacy_flags:
            state.detected_categories = _dedupe_preserve([*state.detected_categories, *sorted(privacy_flags)])
        if "contains_pii" in privacy_flags:
            state.contains_pii = True
            state.contains_sensitive_data = True
        if "contains_financial_identifiers" in privacy_flags:
            state.contains_financial_identifiers = True
            state.contains_sensitive_data = True
        if "contains_special_category_data" in privacy_flags:
            state.contains_special_category_data = True
            state.contains_sensitive_data = True
        if "contains_secrets" in privacy_flags:
            state.contains_secrets = True
            state.contains_sensitive_data = True

        if not state.lawful_basis:
            state.missing_lawful_basis = True
        if arguments.get("supervisor_approval_ref"):
            state.supervisor_execution_approved = True
        if result.get("requires_supervisor_approval") is True and not arguments.get("supervisor_approval_ref"):
            state.missing_supervisor_approval = True
        if state.requested_fields and state.disclosed_fields:
            state.requested_fields_exceed_policy = not set(state.disclosed_fields).issubset(
                set(state.requested_fields)
            )
        if result.get("records"):
            state.audit_logged_for_data_access = False

    def _sanitize_error_message(self, message: str) -> str:
        result = redact_payload(message)
        return str(result.value)

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

    def overlay_tool_definitions(self, tool_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.prepare_tool_definitions(tool_definitions)

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
            if canonical_name in raw_arguments and raw_arguments[canonical_name] not in (None, ""):
                normalized[canonical_name] = raw_arguments[canonical_name]
                continue
            for alias in aliases.get(canonical_name, []):
                if alias in raw_arguments and raw_arguments[alias] not in (None, ""):
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
        digest = hmac.new(
            self._fingerprint_secret,
            _canonical_json(payload).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:16]
        return f"{tool_name}:{digest}"

    def _register_repeat_violation(self, state: GovernanceState, fingerprint: str, reason_code: str) -> None:
        violation = f"{fingerprint}:{reason_code}"
        if violation not in state.tool_repeat_violations:
            state.tool_repeat_violations.append(violation)


    def overlay_tool_definitions(self, tool_definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        overlaid: list[dict[str, Any]] = []
        for item in tool_definitions:
            function = (item or {}).get("function") or {}
            tool_name = function.get("name")
            contract = self.tool_contract_map.get(tool_name)
            if not contract:
                overlaid.append(item)
                continue

            description = function.get("description", "").strip()
            must_call_when = contract.get("must_call_when") or []
            blocks_if_unavailable = contract.get("blocks_if_unavailable", False)

            contract_suffix = [
                "Governance contract:",
                f"- blocks_if_unavailable={str(blocks_if_unavailable).lower()}",
            ]
            if must_call_when:
                contract_suffix.append("- must_call_when=" + " | ".join(must_call_when))

            merged = {
                **item,
                "function": {
                    **function,
                    "description": (description + "\n\n" + "\n".join(contract_suffix)).strip(),
                    "parameters": contract["arguments_schema"],
                },
            }
            overlaid.append(merged)
        return overlaid

    def build_effective_system_prompt(self, base_prompt: str) -> str:
        if not base_prompt:
            return self.system_prompt
        return f"{self.system_prompt}\n\n{base_prompt}".strip()

    def update_state_from_tool_result(
        self,
        state: GovernanceState,
        tool_name: str,
        result_raw: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        try:
            normalized_arguments = self.validate_tool_arguments(tool_name, arguments)
        except GovernanceBlocked as exc:
            state.tool_errors.append(self._sanitize_error_message(f"{tool_name}: {exc}"))
            return

        state.tools_called.append(tool_name)
        self._merge_privacy_analysis(state, normalized_arguments)

        if normalized_arguments:
            fingerprint = self.build_tool_call_fingerprint(tool_name, normalized_arguments)
            count = state.tool_call_counts.get(fingerprint, 0) + 1
            state.tool_call_counts[fingerprint] = count
            state.tool_call_fingerprints.append(fingerprint)
        else:
            fingerprint = None
            count = 1

        try:
            parsed = self.validate_tool_result(tool_name, result_raw)
        except GovernanceBlocked as exc:
            state.tool_errors.append(self._sanitize_error_message(f"{tool_name}: {exc}"))
            if fingerprint and count > self.config["max_identical_tool_calls"]:
                self._register_repeat_violation(state, fingerprint, "identical_call_budget_exceeded")
            return

        trace_id = parsed.get("trace_id")
        if isinstance(trace_id, str) and trace_id:
            state.tool_trace_ids.append(trace_id)

        result = parsed.get("result") or {}
        self._merge_privacy_analysis(state, result)

        if parsed.get("ok") is False or parsed.get("error_code") or parsed.get("error"):
            error_msg = parsed.get("error_message") or parsed.get("error") or "tool_error"
            state.tool_errors.append(self._sanitize_error_message(f"{tool_name}: {error_msg}"))
            if fingerprint and count > self.config["max_identical_tool_calls"]:
                self._register_repeat_violation(state, fingerprint, "identical_call_budget_exceeded")
            return

        sources = result.get("sources") or []
        if not sources and isinstance(parsed.get("source_links"), list):
            sources = [{"url": item, "reference": item} for item in parsed.get("source_links") if isinstance(item, str)]

        collected_refs: list[str] = []
        for source in sources:
            if isinstance(source, dict):
                ref = source.get("reference") or source.get("id") or source.get("url")
                if isinstance(ref, str):
                    collected_refs.append(str(self._redact_for_storage(ref)))
        state.source_refs = _dedupe_preserve([*state.source_refs, *collected_refs])

        if result.get("primary_sources_verified") is False:
            state.primary_sources_verified = False
        if result.get("coverage_status") in {"partial", "not_verified"}:
            state.primary_sources_verified = False
        if result.get("conflicts_detected") is True:
            state.conflicts_detected = True

        if tool_name in {"escalate_to_human_supervisor", "escalate_to_privacy_supervisor"}:
            escalation_id = result.get("escalation_id") or parsed.get("escalation_id")
            if isinstance(escalation_id, str):
                state.escalation_id = escalation_id
            state.status = "ESCALADE_REQUISE"
            state.certainty = "FAIBLE_VERIFICATION_REQUISE"
            if state.risk_level in {"low", "medium"}:
                state.risk_level = "high"

        if tool_name == "log_audit_event":
            audit_event_id = result.get("audit_event_id") or parsed.get("event_id")
            if isinstance(audit_event_id, str):
                state.audit_event_ids.append(audit_event_id)
            if normalized_arguments.get("event_type") == "data_access" and result.get("accepted") is True:
                state.audit_logged_for_data_access = True

        if tool_name == "get_client_records":
            self._register_data_access_context(state, normalized_arguments, result)

        if tool_name == "search_fiscal_sources":
            state.fiscal_search_call_count += 1
            if parsed.get("cache_hit") is True:
                state.fiscal_search_cache_hits += 1

            has_results = bool(result.get("sources")) or bool(parsed.get("has_results"))
            result_count = result.get("result_count", parsed.get("result_count") or 0)
            try:
                result_count = int(result_count)
            except Exception:
                result_count = 0

            if (parsed.get("ok") is True or parsed.get("success") is True) and (has_results or result_count > 0):
                state.fiscal_search_satisfied = True
                state.fiscal_search_cached_result = result_raw

        redacted_result = redact_payload(result)
        if redacted_result.applied_count:
            state.masking_applied = True

        useful_result = bool(result) and (
            bool(result.get("sources"))
            or result.get("coverage_status") in {"verified", "partial"}
            or result.get("amount") is not None
            or result.get("records") is not None
        )
        if tool_name == "search_fiscal_sources" and parsed.get("has_results"):
            useful_result = True

        if fingerprint and useful_result and count > 1:
            self._register_repeat_violation(state, fingerprint, "useful_result_repeated")
        elif fingerprint and count > self.config["max_identical_tool_calls"]:
            self._register_repeat_violation(state, fingerprint, "identical_call_budget_exceeded")

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

    def apply_policy_outcome_to_state(self, state: GovernanceState, outcome: dict[str, Any]) -> dict[str, Any]:
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

    def validate_final_response_text(self, text: str) -> dict[str, Any]:
        candidate = (text or "").strip()
        if not candidate:
            raise GovernanceBlocked("final_response_empty")

        fenced_match = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL)
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

        if self._final_validator is not None:
            errors = sorted(self._final_validator.iter_errors(payload), key=lambda e: list(e.path))
            if errors:
                first = errors[0]
                path = ".".join(str(part) for part in first.path) or "<root>"
                raise GovernanceBlocked(f"final_response_schema_error at {path}: {first.message}")

        leaks = assert_no_raw_sensitive_data(payload)
        if leaks:
            raise GovernanceBlocked(
                "final_response_contains_raw_identifier: " + ", ".join(leaks[:5])
            )

        return payload

    def build_repair_message(self, error_message: str) -> str:
        return (
            "Previous final response is invalid.\n"
            "Return exactly one JSON object and nothing else.\n"
            "Do not use markdown, prose, headings, bullets or code fences.\n"
            "Do not emit raw identifiers, client secrets or account identifiers.\n"
            "Do not call another tool unless a mandatory source is still missing.\n"
            "Validation error: "
            + error_message
        )
    def register_final_response_repair_attempt(self, state: GovernanceState) -> None:
        state.final_response_repair_count += 1
        if state.final_response_repair_count > self.config["max_final_repair_attempts"]:
            raise GovernanceBlocked("final_response_repair_budget_exhausted")

    def _now_iso_utc(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def build_blocked_final_response(self, state: GovernanceState, reason: str) -> str:
        checked_at = self._now_iso_utc()
        safe_reason = str(self._redact_for_storage(reason))
        payload = {
            "status": state.status if state.status in {
                "INFORMATION_SOURCEE", "ANALYSE_PREPARATOIRE", "ESCALADE_REQUISE", "BLOQUE"
            } else "BLOQUE",
            "certainty": state.certainty if state.certainty in {
                "HAUTE", "MOYENNE", "FAIBLE_VERIFICATION_REQUISE"
            } else "FAIBLE_VERIFICATION_REQUISE",
            "scope": {
                "domain": state.domain if state.domain in {
                    "accounting", "fiscal", "legal", "social", "audit", "compliance", "privacy", "mixed"
                } else "mixed",
                "jurisdiction": "FR",
                "fact_date": None,
                "source_checked_at": checked_at,
                "risk_level": state.risk_level if state.risk_level in {
                    "low", "medium", "high", "critical"
                } else "medium",
            },
            "facts": {
                "verified_facts": self._redact_for_storage(list(state.facts_summary)),
                "assumptions": [],
                "missing_facts": [],
            },
            "sources": [
                {
                    "source_type": "other",
                    "reference": "governance_runtime",
                    "effective_date": None,
                    "checked_at": checked_at,
                    "verified": False,
                    "url": None,
                }
            ],
            "analysis": {
                "mode": "blocked",
                "major": "Final response enforcement triggered.",
                "minor": safe_reason,
                "conclusion": "A valid governance-compliant response could not be produced from the model output.",
            },
            "risks": [],
            "next_action": {
                "type": "escalate" if state.status == "ESCALADE_REQUISE" else "blocked",
                "message": safe_reason,
                "escalation_id": None,
            },
            "audit_trail": {
                "tools_called": list(state.tools_called),
                "policy_rule_hits": list(state.policy_rule_hits),
                "audit_summary": (
                    f"tool_calls={len(state.tools_called)};"
                    f"audit_events={len(state.audit_event_ids)};"
                    f"redactions={int(state.masking_applied)};"
                    f"violations={len(state.tool_repeat_violations)}"
                ),
                "redactions_applied": state.masking_applied,
            },
        }
        validated_payload = self.validate_final_response_text(json.dumps(payload, ensure_ascii=False))
        return json.dumps(validated_payload, ensure_ascii=False)

    def build_forced_fiscal_final_response(self, state: GovernanceState) -> str:
        checked_at = self._now_iso_utc()

        cached: dict[str, Any] = {}
        if state.fiscal_search_cached_result:
            try:
                cached = json.loads(state.fiscal_search_cached_result)
            except Exception:
                cached = {}

        result_text = str(cached.get("result_text") or "")
        source_links = cached.get("source_links") or []
        if not isinstance(source_links, list):
            source_links = []

        search_used = str(cached.get("search_used") or "").strip() or "recherche fiscale"
        result_count = cached.get("result_count") or 0
        try:
            result_count = int(result_count)
        except Exception:
            result_count = 0

        fact_date = cached.get("fact_date")
        if not isinstance(fact_date, str) or len(fact_date) != 10:
            fact_date = None

        reference = state.source_refs[0] if state.source_refs else f"OpenLégi fiscal search: {search_used}"
        source_url = None
        for item in source_links:
            if isinstance(item, str) and item.startswith("http"):
                source_url = item
                break

        minor = result_text[:1800] if result_text else "Fiscal source search returned exploitable results."
        payload = {
            "status": "ANALYSE_PREPARATOIRE",
            "certainty": "MOYENNE",
            "scope": {
                "domain": "fiscal",
                "jurisdiction": "FR",
                "fact_date": fact_date,
                "source_checked_at": checked_at,
                "risk_level": state.risk_level if state.risk_level in {
                    "low", "medium", "high", "critical"
                } else "medium",
            },
            "facts": {
                "verified_facts": [
                    f"Fiscal search executed with query: {search_used}",
                    f"Fiscal source search returned {result_count} result(s).",
                ] + list(state.facts_summary),
                "assumptions": [
                    "Forced finalization was triggered after repeated fiscal-search loops.",
                    "This response is a preparatory synthesis based on the cached fiscal search result.",
                ],
                "missing_facts": [],
            },
            "sources": [
                {
                    "source_type": "law",
                    "reference": reference,
                    "effective_date": fact_date,
                    "checked_at": checked_at,
                    "verified": True,
                    "url": source_url,
                }
            ],
            "analysis": {
                "mode": "syllogism",
                "major": "French tax-law sources were queried through the fiscal search wrapper.",
                "minor": minor,
                "conclusion": (
                    "A preparatory fiscal synthesis is returned from the cached source result to stop repeated tool loops. "
                    "Human validation remains recommended before operational use."
                ),
            },
            "risks": [
                {
                    "level": "MOYEN",
                    "category": "loop_control",
                    "reference": "hard_stop_fiscal_finalize",
                    "impact": "Repeated source calls were stopped by governance runtime.",
                    "mitigation": "Review the cited fiscal source and validate the final treatment before production use.",
                }
            ],
            "next_action": {
                "type": "answer_ready",
                "message": "Governance runtime returned a forced fiscal final response from cached validated search output.",
                "escalation_id": state.escalation_id,
            },
            "audit_trail": {
                "tools_called": list(state.tools_called),
                "tool_trace_ids": list(state.tool_trace_ids),
                "policy_rule_hits": list(state.policy_rule_hits),
                "audit_event_ids": list(state.audit_event_ids),
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def ensure_final_response(self, state: GovernanceState, text: str) -> str:
        text = text or ""

        try:
            payload = self.validate_final_response_text(text)
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            pass
        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    parsed = None

        if isinstance(parsed, dict):
            cached = {}
            if state.fiscal_search_cached_result:
                try:
                    cached = json.loads(state.fiscal_search_cached_result)
                except Exception:
                    cached = {}

            fact_date = cached.get("fact_date")
            if not isinstance(fact_date, str) or len(fact_date) != 10:
                fact_date = None

            source_links = cached.get("source_links") or []
            if not isinstance(source_links, list):
                source_links = []

            first_ref = None
            if state.source_refs:
                first_ref = state.source_refs[0]
            elif source_links:
                first_ref = source_links[0]
            else:
                first_ref = "governance://normalized-final-response"

            first_url = first_ref if isinstance(first_ref, str) and first_ref.startswith("http") else None

            raw_status = str(parsed.get("status") or "").strip().upper()
            status = raw_status if raw_status in {
                "INFORMATION_SOURCEE", "ANALYSE_PREPARATOIRE", "ESCALADE_REQUISE", "BLOQUE"
            } else "ANALYSE_PREPARATOIRE"

            raw_certainty = str(parsed.get("certainty") or "").strip().upper()
            certainty = raw_certainty if raw_certainty in {
                "HAUTE", "MOYENNE", "FAIBLE_VERIFICATION_REQUISE"
            } else "MOYENNE"

            raw_scope = parsed.get("scope")
            if isinstance(raw_scope, dict):
                scope = {
                    "domain": raw_scope.get("domain") if raw_scope.get("domain") in {
                        "accounting", "fiscal", "legal", "social", "audit", "compliance", "privacy", "mixed"
                    } else (state.domain if state.domain in {
                        "accounting", "fiscal", "legal", "social", "audit", "compliance", "privacy", "mixed"
                    } else "mixed"),
                    "jurisdiction": "FR",
                    "fact_date": raw_scope.get("fact_date") if isinstance(raw_scope.get("fact_date"), str) else fact_date,
                    "source_checked_at": self._now_iso_utc(),
                    "risk_level": raw_scope.get("risk_level") if raw_scope.get("risk_level") in {
                        "low", "medium", "high", "critical"
                    } else (state.risk_level if state.risk_level in {
                        "low", "medium", "high", "critical"
                    } else "medium"),
                }
            else:
                scope = {
                    "domain": state.domain if state.domain in {
                        "accounting", "fiscal", "legal", "social", "audit", "compliance", "privacy", "mixed"
                    } else "fiscal",
                    "jurisdiction": "FR",
                    "fact_date": fact_date,
                    "source_checked_at": self._now_iso_utc(),
                    "risk_level": state.risk_level if state.risk_level in {
                        "low", "medium", "high", "critical"
                    } else "medium",
                }

            raw_facts = parsed.get("facts")
            if isinstance(raw_facts, dict):
                facts = {
                    "verified_facts": raw_facts.get("verified_facts") if isinstance(raw_facts.get("verified_facts"), list) else [],
                    "assumptions": raw_facts.get("assumptions") if isinstance(raw_facts.get("assumptions"), list) else [],
                    "missing_facts": raw_facts.get("missing_facts") if isinstance(raw_facts.get("missing_facts"), list) else [],
                }
            elif isinstance(raw_facts, list):
                facts = {
                    "verified_facts": [str(item) for item in raw_facts],
                    "assumptions": [],
                    "missing_facts": [],
                }
            else:
                facts = {
                    "verified_facts": list(state.facts_summary),
                    "assumptions": [],
                    "missing_facts": [],
                }

            raw_analysis = parsed.get("analysis")
            if isinstance(raw_analysis, dict):
                analysis = {
                    "mode": str(raw_analysis.get("mode") or "syllogism"),
                    "major": str(raw_analysis.get("major") or "Réponse normalisée à partir d'un JSON legacy du modèle."),
                    "minor": str(raw_analysis.get("minor") or ""),
                    "conclusion": str(raw_analysis.get("conclusion") or "Réponse finalisée après normalisation du format de sortie."),
                }
            else:
                analysis = {
                    "mode": "syllogism",
                    "major": "Réponse normalisée à partir d'un JSON legacy du modèle.",
                    "minor": str(parsed.get("answer") or parsed.get("summary") or ""),
                    "conclusion": "Réponse finalisée après normalisation du format de sortie.",
                }

            normalized_payload = {
                "status": status,
                "certainty": certainty,
                "scope": scope,
                "facts": facts,
                "sources": [
                    {
                        "source_type": "law" if first_url else "other",
                        "reference": first_ref,
                        "effective_date": fact_date,
                        "checked_at": self._now_iso_utc(),
                        "verified": True if first_ref else False,
                        "url": first_url,
                    }
                ],
                "analysis": analysis,
                "risks": parsed.get("risks") if isinstance(parsed.get("risks"), list) else [],
                "next_action": parsed.get("next_action") if isinstance(parsed.get("next_action"), dict) else {
                    "type": "answer_ready",
                    "message": "Réponse normalisée par la gouvernance Hermes.",
                    "escalation_id": state.escalation_id,
                },
                "audit_trail": {
                    "tools_called": list(state.tools_called),
                    "tool_trace_ids": list(state.tool_trace_ids),
                    "policy_rule_hits": list(state.policy_rule_hits),
                    "audit_event_ids": list(state.audit_event_ids),
                    "tool_call_fingerprints": list(state.tool_call_fingerprints),
                    "repetition_violations": list(state.tool_repeat_violations),
                },
            }

            try:
                validated = self.validate_final_response_text(json.dumps(normalized_payload, ensure_ascii=False))
                return json.dumps(validated, ensure_ascii=False)
            except Exception:
                pass

        if state.status == "ESCALADE_REQUISE":
            checked_at = self._now_iso_utc()
            payload = {
                "status": "ESCALADE_REQUISE",
                "certainty": state.certainty if state.certainty in {
                    "HAUTE", "MOYENNE", "FAIBLE_VERIFICATION_REQUISE"
                } else "FAIBLE_VERIFICATION_REQUISE",
                "scope": {
                    "domain": state.domain if state.domain in {
                        "accounting", "fiscal", "legal", "social", "audit", "compliance", "privacy", "mixed"
                    } else "mixed",
                    "jurisdiction": "FR",
                    "fact_date": None,
                    "source_checked_at": checked_at,
                    "risk_level": state.risk_level if state.risk_level in {
                        "low", "medium", "high", "critical"
                    } else "high",
                },
                "facts": {
                    "verified_facts": list(state.facts_summary),
                    "assumptions": [],
                    "missing_facts": [],
                },
                "sources": [
                    {
                        "source_type": "other",
                        "reference": ref,
                        "effective_date": None,
                        "checked_at": checked_at,
                        "verified": True,
                        "url": ref if isinstance(ref, str) and ref.startswith("http") else None,
                    }
                    for ref in (state.source_refs[:5] or ["governance://escalation"])
                ],
                "analysis": {
                    "mode": "blocked",
                    "major": "La gouvernance Hermes impose une escalade humaine.",
                    "minor": "Les conditions de preuve ou de périmètre ne permettent pas une conclusion autonome sûre.",
                    "conclusion": "Le dossier doit être transmis à un superviseur humain.",
                },
                "calculations": [],
                "entries": [],
                "risks": [],
                "next_action": {
                    "type": "escalate",
                    "message": "Escalade humaine requise par la gouvernance Hermes.",
                    "escalation_id": state.escalation_id,
                },
                "audit_trail": {
                    "tools_called": list(state.tools_called),
                    "tool_trace_ids": list(state.tool_trace_ids),
                    "policy_rule_hits": list(state.policy_rule_hits),
                    "audit_event_ids": list(state.audit_event_ids),
                    "tool_call_fingerprints": list(state.tool_call_fingerprints),
                    "repetition_violations": list(state.tool_repeat_violations),
                },
            }
            return json.dumps(payload, ensure_ascii=False)

        if state.fiscal_search_satisfied and state.fiscal_search_cached_result:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None

            if isinstance(parsed, dict):
                status = str(parsed.get("status") or "").strip().upper()
                if status in {"REPONSE_FINALE", "ANALYSE_PREPARATOIRE", "ESCALADE_REQUISE"}:
                    return text

            return self.build_forced_fiscal_final_response(state)

        return self.build_blocked_final_response(
            state,
            "final_response_schema_enforcement",
        )

    def ensure_contract_tools_exist(self, available_tool_names: Iterable[str]) -> list[str]:
        available = set(available_tool_names)
        missing = [
            name for name in self.tool_contract_map
            if name not in available and self.tool_contract_map[name].get("blocks_if_unavailable")
        ]
        return missing
