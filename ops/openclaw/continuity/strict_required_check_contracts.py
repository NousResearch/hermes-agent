#!/usr/bin/env python3
"""Shared strict required-check contracts for autonomy regression gating.

This module centralizes contract metadata used by:
- harness emitters (`required_check_provenance` payloads)
- cluster check registry command mapping for required checks
- `verify_then_resume` strict required-check enforcement
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_CHECK_PROVENANCE_SCHEMA_VERSION = "openclaw.continuity.required_check_summary_provenance.v1"
STRICT_REQUIRED_CHECK_COMMAND_PREFIX = "check_"


def _is_safe_local_python_command_suffix(value: str) -> bool:
    text = str(value or "")
    if not text or text != text.strip():
        return False
    if text.startswith(("/", "\\")):
        return False
    if "/" in text or "\\" in text:
        return False
    if text in {".", ".."}:
        return False
    if not text.endswith(".py"):
        return False
    return True


def resolve_contract_command_path(*, continuity_dir: Path, command_suffix: str) -> Path:
    if not _is_safe_local_python_command_suffix(command_suffix):
        raise RuntimeError(
            "invalid strict required-check command_suffix: "
            "must be a local .py filename without path separators"
        )

    base_dir = continuity_dir.resolve()
    command_path = (base_dir / command_suffix).resolve()
    try:
        command_path.relative_to(base_dir)
    except ValueError as exc:
        raise RuntimeError(
            "invalid strict required-check command_suffix: path escapes continuity_dir"
        ) from exc

    return command_path


@dataclass(frozen=True)
class StrictRequiredCheckContract:
    check_id: str
    harness: str
    summary_source: str
    summary_schema_version: str
    command_suffix: str
    scenario_names: tuple[str, ...]
    minimum_result_count: int
    require_provenance_contract_inputs: bool = True
    expected_summary_fields: dict[str, Any] | None = None

    def contract_inputs(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "harness": self.harness,
            "source": self.summary_source,
            "minimum_result_count": self.minimum_result_count,
            "scenario_names": list(self.scenario_names),
        }

    def contract_fingerprint(self) -> str:
        return compute_contract_fingerprint(self.contract_inputs())

    def required_check_provenance(self) -> dict[str, Any]:
        contract_inputs = self.contract_inputs()
        return {
            "schema_version": REQUIRED_CHECK_PROVENANCE_SCHEMA_VERSION,
            "check_id": self.check_id,
            "contract_fingerprint": compute_contract_fingerprint(contract_inputs),
            "contract_inputs": contract_inputs,
        }

    def verify_required_check_contract(self) -> dict[str, Any]:
        contract_inputs = self.contract_inputs()
        return {
            "id": self.check_id,
            "command_suffix": self.command_suffix,
            "expected_harness": self.harness,
            "expected_summary_source": self.summary_source,
            "expected_summary_schema_version": self.summary_schema_version,
            "expected_summary_fields": dict(self.expected_summary_fields or {}),
            "expected_provenance_schema_version": REQUIRED_CHECK_PROVENANCE_SCHEMA_VERSION,
            "expected_contract_fingerprint_inputs": contract_inputs,
            "require_provenance_contract_inputs": self.require_provenance_contract_inputs,
            "minimum_result_count": self.minimum_result_count,
            "required_scenario_names": list(self.scenario_names),
        }

    def cluster_command(self, *, python_bin: str, continuity_dir: Path) -> list[str]:
        command_path = resolve_contract_command_path(
            continuity_dir=continuity_dir,
            command_suffix=self.command_suffix,
        )
        return [str(python_bin), str(command_path)]


STRICT_REQUIRED_CHECK_CONTRACTS: tuple[StrictRequiredCheckContract, ...] = (
    StrictRequiredCheckContract(
        check_id="gtc_latest_schema_failclose",
        harness="gtc_latest_schema_failclose_regressions",
        summary_source="check_gtc_latest_schema_regressions.py",
        summary_schema_version="openclaw.continuity.gtc_latest_schema_regression_summary.v1",
        command_suffix="check_gtc_latest_schema_failclose_regressions.py",
        scenario_names=(
            "seed_schema_fixture_contract",
            "baseline_valid_fixture",
            "missing_gateboard_surface",
            "publish_manifest_paths_mismatch",
            "publish_manifest_digest_tamper",
            "publish_manifest_auth_signature_tamper",
            "generation_mismatch",
        ),
        minimum_result_count=7,
    ),
    StrictRequiredCheckContract(
        check_id="gtc_publish_manifest_auth_dual_mode",
        harness="publish_manifest_authenticity_dual_mode",
        summary_source="check_gtc_latest_schema_regressions.py",
        summary_schema_version="openclaw.continuity.publish_manifest_auth_regression_summary.v1",
        command_suffix="check_gtc_publish_manifest_auth_regressions.py",
        scenario_names=(
            "seed_schema_fixture_contract",
            "publish_manifest_auth_mode_compat_hmac_valid",
            "publish_manifest_auth_mode_default_ed25519_valid",
            "publish_manifest_auth_signature_tamper",
            "publish_manifest_auth_signature_tamper_ed25519",
        ),
        minimum_result_count=5,
    ),
    StrictRequiredCheckContract(
        check_id="gtc_incident_replay_verify_gate_posture",
        harness="gtc_incident_replay_verify_gate_posture",
        summary_source="check_gtc_incident_replay_regressions.py",
        summary_schema_version="openclaw.continuity.gtc_incident_replay_regression_summary.v1",
        command_suffix="check_gtc_incident_replay_regressions.py",
        scenario_names=(
            "route_selection_open_incident",
            "verify_gate_preflight_posture_projection",
            "incident_scope_checkpoint_filter",
            "full_scope_break_glass_includes_neighbors",
            "typed_artifact_roles_manifested",
            "bundle_written",
        ),
        minimum_result_count=6,
        expected_summary_fields={
            "check_id": "gtc_incident_replay_verify_gate_posture",
            "schema_version": "gtc.incident_replay.regressions.v1",
        },
    ),
    StrictRequiredCheckContract(
        check_id="gtc_publish_transaction_regressions",
        harness="gtc_publish_transaction_regressions",
        summary_source="check_gtc_publish_transaction_regressions.py",
        summary_schema_version="openclaw.continuity.gtc_publish_transaction_regression_summary.v1",
        command_suffix="check_gtc_publish_transaction_regressions.py",
        scenario_names=(
            "lock_busy_failclose_preserves_live_latest",
            "crash_window_then_recovery",
            "mid_promotion_crash_then_recovery_semantics",
            "fully_promoted_crash_recovery_discards_backups",
        ),
        minimum_result_count=4,
        expected_summary_fields={
            "check_id": "gtc_publish_transaction_regressions",
            "schema_version": "gtc.publish.transaction.regressions.v1",
        },
    ),
    StrictRequiredCheckContract(
        check_id="queue_cooldown_authority_regressions",
        harness="queue_cooldown_authority_regressions",
        summary_source="check_queue_cooldown_authority_regressions.py",
        summary_schema_version="openclaw.continuity.queue_cooldown_authority_regression_summary.v1",
        command_suffix="check_queue_cooldown_authority_regressions.py",
        scenario_names=(
            "cooldown_projection_and_claim_gate",
            "fixed_now_clock_authority",
        ),
        minimum_result_count=2,
        expected_summary_fields={
            "check_id": "queue_cooldown_authority_regressions",
            "schema_version": "queue.cooldown.authority.regressions.v1",
        },
    ),
    StrictRequiredCheckContract(
        check_id="no_nudge_reminder_runtime_hardening",
        harness="no_nudge_reminder_runtime_regressions",
        summary_source="check_no_nudge_reminder_runtime_regressions.py",
        summary_schema_version="openclaw.continuity.no_nudge_reminder_runtime_regression_summary.v1",
        command_suffix="check_no_nudge_reminder_runtime_regressions.py",
        scenario_names=(
            "ready_no_reply",
            "progress_no_reply",
            "blocker_forward_only",
        ),
        minimum_result_count=3,
        expected_summary_fields={
            "check_id": "no_nudge_reminder_runtime_hardening",
            "summary_schema_version": "openclaw.continuity.no_nudge_reminder_runtime_regression_summary.v1",
            "schema_version": "no_nudge.reminder.runtime.regressions.v1",
        },
    ),
    StrictRequiredCheckContract(
        check_id="swarm_operability_contract_regressions",
        harness="swarm_operability_contract_regressions",
        summary_source="check_swarm_operability_regressions.py",
        summary_schema_version="openclaw.continuity.swarm_operability_regression_summary.v1",
        command_suffix="check_swarm_operability_regressions.py",
        scenario_names=(
            "healthy_fixture_ok",
            "missing_required_role_failclose",
            "malformed_role_shape_failclose",
            "runbook_snippet_drift_warn_only",
        ),
        minimum_result_count=4,
        expected_summary_fields={
            "check_id": "swarm_operability_contract_regressions",
            "schema_version": "swarm.operability.regressions.v1",
        },
    ),
    StrictRequiredCheckContract(
        check_id="slot_fill_protocol_contract_regressions",
        harness="slot_fill_protocol_contract_regressions",
        summary_source="check_slot_fill_protocol_regressions.py",
        summary_schema_version="openclaw.continuity.slot_fill_protocol_regression_summary.v1",
        command_suffix="check_slot_fill_protocol_regressions.py",
        scenario_names=(
            "healthy_fixture_ok",
            "missing_required_protocol_snippet_failclose",
            "workflow_execution_tuple_drift_failclose",
            "workflow_reference_drift_warn_only",
            "dispatcher_slot_fill_route_ok",
        ),
        minimum_result_count=5,
        expected_summary_fields={
            "check_id": "slot_fill_protocol_contract_regressions",
            "schema_version": "slot_fill_protocol.regressions.v1",
        },
    ),
)

def validate_strict_required_check_contracts(contracts: tuple[StrictRequiredCheckContract, ...]) -> None:
    errors: list[str] = []
    if not contracts:
        errors.append("no strict required-check contracts configured")

    seen_ids: set[str] = set()
    seen_command_suffixes: set[str] = set()

    for idx, contract in enumerate(contracts):
        label = f"contract[{idx}]"

        check_id = str(contract.check_id or "").strip()
        if not check_id:
            errors.append(f"{label}: missing check_id")
        elif check_id in seen_ids:
            errors.append(f"{label}: duplicate check_id={check_id}")
        else:
            seen_ids.add(check_id)

        harness = str(contract.harness or "").strip()
        if not harness:
            errors.append(f"{label}: check_id={check_id or '<missing>'} missing harness")

        summary_source = str(contract.summary_source or "").strip()
        if not summary_source:
            errors.append(f"{label}: check_id={check_id or '<missing>'} missing summary_source")

        summary_schema_version = str(contract.summary_schema_version or "").strip()
        if not summary_schema_version:
            errors.append(f"{label}: check_id={check_id or '<missing>'} missing summary_schema_version")

        command_suffix = str(contract.command_suffix or "").strip()
        if not command_suffix:
            errors.append(f"{label}: check_id={check_id or '<missing>'} missing command_suffix")
        elif command_suffix in seen_command_suffixes:
            errors.append(f"{label}: check_id={check_id or '<missing>'} duplicate command_suffix={command_suffix}")
        else:
            seen_command_suffixes.add(command_suffix)

        if command_suffix and not _is_safe_local_python_command_suffix(command_suffix):
            errors.append(
                f"{label}: check_id={check_id or '<missing>'} command_suffix must be a local .py filename without path separators"
            )
        elif command_suffix and not command_suffix.startswith(STRICT_REQUIRED_CHECK_COMMAND_PREFIX):
            errors.append(
                f"{label}: check_id={check_id or '<missing>'} command_suffix must use dedicated {STRICT_REQUIRED_CHECK_COMMAND_PREFIX}*.py wrapper"
            )

        scenario_names = [str(item or "").strip() for item in contract.scenario_names]
        if not scenario_names:
            errors.append(f"{label}: check_id={check_id or '<missing>'} has no required scenario_names")
        else:
            if any(not item for item in scenario_names):
                errors.append(f"{label}: check_id={check_id or '<missing>'} has blank scenario_name")
            seen_scenarios: set[str] = set()
            duplicate_scenarios: list[str] = []
            for scenario_name in scenario_names:
                if scenario_name in seen_scenarios:
                    duplicate_scenarios.append(scenario_name)
                else:
                    seen_scenarios.add(scenario_name)
            if duplicate_scenarios:
                unique_dupes = sorted(set(duplicate_scenarios))
                errors.append(
                    f"{label}: check_id={check_id or '<missing>'} duplicate scenario_names={','.join(unique_dupes)}"
                )

        if not isinstance(contract.minimum_result_count, int) or contract.minimum_result_count <= 0:
            errors.append(
                f"{label}: check_id={check_id or '<missing>'} invalid minimum_result_count={contract.minimum_result_count!r}"
            )
        elif contract.minimum_result_count < len(scenario_names):
            errors.append(
                f"{label}: check_id={check_id or '<missing>'} minimum_result_count={contract.minimum_result_count} "
                f"< required_scenarios_count={len(scenario_names)}"
            )

        if not isinstance(contract.require_provenance_contract_inputs, bool):
            errors.append(
                f"{label}: check_id={check_id or '<missing>'} require_provenance_contract_inputs must be bool"
            )

        expected_summary_fields = contract.expected_summary_fields
        if expected_summary_fields is not None and not isinstance(expected_summary_fields, dict):
            errors.append(
                f"{label}: check_id={check_id or '<missing>'} expected_summary_fields must be dict|None"
            )
        elif isinstance(expected_summary_fields, dict):
            blank_summary_fields = sorted(
                {str(field_name) for field_name in expected_summary_fields if not str(field_name).strip()}
            )
            if blank_summary_fields:
                errors.append(
                    f"{label}: check_id={check_id or '<missing>'} expected_summary_fields contains blank keys"
                )

    if errors:
        joined = "\n- ".join(errors)
        raise RuntimeError(f"invalid strict required-check contracts:\n- {joined}")


validate_strict_required_check_contracts(STRICT_REQUIRED_CHECK_CONTRACTS)

_CONTRACTS_BY_ID = {item.check_id: item for item in STRICT_REQUIRED_CHECK_CONTRACTS}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_contract_fingerprint(contract_inputs: dict[str, Any]) -> str:
    canonical = canonical_json(contract_inputs)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def strict_required_check_contract(check_id: str) -> StrictRequiredCheckContract:
    key = str(check_id or "").strip()
    if not key:
        raise KeyError("missing check_id")
    return _CONTRACTS_BY_ID[key]


def required_check_contract_inputs(check_id: str) -> dict[str, Any]:
    return strict_required_check_contract(check_id).contract_inputs()


def required_check_provenance(check_id: str) -> dict[str, Any]:
    return strict_required_check_contract(check_id).required_check_provenance()


def strict_required_cluster_command_map(*, python_bin: str, continuity_dir: str | Path) -> dict[str, list[str]]:
    base_dir = Path(continuity_dir)
    return {
        item.check_id: item.cluster_command(python_bin=python_bin, continuity_dir=base_dir)
        for item in STRICT_REQUIRED_CHECK_CONTRACTS
    }


def strict_required_contracts_for_verify_then_resume() -> tuple[dict[str, Any], ...]:
    return tuple(item.verify_required_check_contract() for item in STRICT_REQUIRED_CHECK_CONTRACTS)
