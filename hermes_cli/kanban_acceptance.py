"""Acceptance Authority — immutable contracts, deterministic receipts, and derived completion.

Guarantees:
1. P0_SPEC is immutable during project execution; only human-authorized amendments can version it.
2. Acceptance is never written directly by agents; `PASS` is a purely derived status.
3. Every criterion verification emits a cryptographically bound `VerificationReceipt`.
4. Evidence Freshness: receipts are invalidated immediately if source tree, spec, or runner changes.
5. All tasks marked done does NOT imply project acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, Optional

from hermes_cli.kanban_governance import (
    TrustedPrincipalContext, GovernancePolicyMissing,
    dump_governance_yaml, load_governance_yaml,
)

logger = logging.getLogger(__name__)


def _compute_runner_hash() -> str:
    try:
        content = Path(__file__).read_bytes()
        return hashlib.sha256(content).hexdigest()
    except Exception:
        return hashlib.sha256(b"hermes-acceptance-runner-v2").hexdigest()


RUNNER_HASH = _compute_runner_hash()


def _record_approved_spec(project_id: str, spec_id: str, spec_hash: str) -> None:
    try:
        from hermes_cli.kanban_db import kanban_home
        base = kanban_home()
    except Exception:
        home_dir = os.environ.get("HERMES_KANBAN_HOME") or os.environ.get("HERMES_HOME")
        base = Path(home_dir).expanduser() if home_dir else Path.home() / ".hermes"
    auth_dir = base / ".governance_authority"
    auth_dir.mkdir(parents=True, exist_ok=True)
    registry_file = auth_dir / "approved_specs.json"
    data = {}
    if registry_file.is_file():
        try:
            data = json.loads(registry_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    data[project_id] = {
        "spec_id": spec_id,
        "spec_hash": spec_hash,
        "updated_at": int(time.time()),
    }
    tmp = auth_dir / f"tmp_{secrets.token_hex(4)}"
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except Exception:
        pass
    tmp.replace(registry_file)


def get_approved_spec_record(project_id: str) -> tuple[str, Optional[dict[str, Any]]]:
    """Retrieve the control-plane approved specification record for a project.

    Returns (status, record_dict):
    - ("APPROVED", {"spec_id": ..., "spec_hash": ..., "updated_at": ...})
    - ("MISSING_REGISTRY", None)
    - ("CORRUPT_REGISTRY", None)
    - ("UNKNOWN_PROJECT", None)
    """
    try:
        from hermes_cli.kanban_db import kanban_home
        base = kanban_home()
    except Exception:
        home_dir = os.environ.get("HERMES_KANBAN_HOME") or os.environ.get("HERMES_HOME")
        base = Path(home_dir).expanduser() if home_dir else Path.home() / ".hermes"

    registry_file = base / ".governance_authority" / "approved_specs.json"
    if not registry_file.is_file():
        return "MISSING_REGISTRY", None

    try:
        data = json.loads(registry_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return "CORRUPT_REGISTRY", None
    except Exception:
        return "CORRUPT_REGISTRY", None

    if project_id not in data or not isinstance(data.get(project_id), dict):
        return "UNKNOWN_PROJECT", None

    rec = data[project_id]
    if "spec_hash" not in rec:
        return "CORRUPT_REGISTRY", None

    return "APPROVED", rec


def _get_approved_spec_hash(project_id: str) -> Optional[str]:
    status, rec = get_approved_spec_record(project_id)
    if status == "APPROVED" and rec:
        return rec.get("spec_hash")
    return None



@dataclass
class CriterionSpec:
    id: str
    statement: str
    required: bool = True
    verification: dict[str, Any] = field(default_factory=dict)
    evidence_policy: dict[str, Any] = field(default_factory=dict)


@dataclass
class P0Spec:
    p0_spec_version: int | str
    project_id: str
    spec_id: str
    spec_hash: str
    source_manifest: list[str]
    acceptance_criteria: list[CriterionSpec]
    completion_policy: dict[str, Any]
    immutable_after_start: bool = True
    created_at: int = field(default_factory=lambda: int(time.time()))


@dataclass
class VerificationReceipt:
    receipt_id: str
    criterion_id: str
    project_id: str
    spec_hash: str
    source_manifest_hash: str
    source_tree_hash: str
    runner_hash: str
    argv: list[str]
    started_at: int
    finished_at: int
    exit_code: int
    stdout_sha256: str
    stderr_sha256: str
    predicate_satisfied: bool
    stdout_snippet: str = ""
    stderr_snippet: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    signature_proof: str = ""


def _compute_receipt_signature(
    receipt_id: str,
    criterion_id: str,
    project_id: str,
    spec_hash: str,
    source_tree_hash: str,
    runner_hash: str,
    argv: list[str],
    exit_code: int,
    stdout_sha256: str,
    stderr_sha256: str,
    expected_exit_code: int,
    stdout_contains_expected: list[str],
    stderr_contains_expected: list[str],
    predicate_satisfied: bool,
) -> str:
    from hermes_cli.kanban_governance import get_governance_secret
    msg_parts = [
        receipt_id,
        criterion_id,
        project_id,
        spec_hash,
        source_tree_hash,
        runner_hash,
        json.dumps(argv),
        str(exit_code),
        stdout_sha256,
        stderr_sha256,
        str(expected_exit_code),
        json.dumps(sorted(stdout_contains_expected)),
        json.dumps(sorted(stderr_contains_expected)),
        str(int(predicate_satisfied)),
    ]
    msg = "|".join(msg_parts).encode("utf-8")
    return hmac.new(get_governance_secret(), msg, hashlib.sha256).hexdigest()




@dataclass
class ProjectAcceptanceResult:
    status: str  # "PASS", "NOT_COMPLETE", "STALE_EVIDENCE", "FAILED_ACCEPTANCE", "BLOCKED"
    project_id: str
    spec_id: str
    spec_hash: str
    criteria_results: dict[str, dict[str, Any]]
    required_passed: int
    required_total: int
    unresolved_material_gaps: int
    governance_violations: int
    evaluated_at: int
    evidence_set_hash: str


def compute_source_tree_hash(workspace_dir: Path, source_manifest: list[str]) -> tuple[str, str]:
    """Compute deterministic SHA-256 for the declared source manifest and files."""
    manifest_bytes = json.dumps(sorted(source_manifest)).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    file_hashes: list[str] = []
    for rel_path in sorted(source_manifest):
        file_path = workspace_dir / rel_path
        if not file_path.is_file():
            file_hashes.append(f"{rel_path}:MISSING")
        else:
            raw = file_path.read_bytes()
            h = hashlib.sha256(raw).hexdigest()
            file_hashes.append(f"{rel_path}:{h}")

    tree_bytes = "\n".join(file_hashes).encode("utf-8")
    tree_hash = hashlib.sha256(tree_bytes).hexdigest()
    return manifest_hash, tree_hash


def freeze_acceptance_spec(
    workspace_dir: Path,
    spec_dict: dict[str, Any],
    *,
    principal_context: Optional[TrustedPrincipalContext] = None,
) -> P0Spec:
    """Freeze acceptance spec into workspace_dir/.governance/P0_SPEC.yaml.

    Once written, immutable_after_start is enforced.
    Overwriting an existing frozen spec without authenticated human authority is strictly DENIED.
    """
    gov_dir = workspace_dir / ".governance"
    gov_dir.mkdir(parents=True, exist_ok=True)
    spec_path = gov_dir / "P0_SPEC.yaml"

    if spec_path.is_file():
        if principal_context is None:
            raise PermissionError(
                f"GOVERNANCE DENIAL: P0_SPEC already frozen at {spec_path}. "
                "Unauthorized attempt to overwrite frozen spec without human principal proof."
            )
        principal_context.verify_human_authority(action="amend_spec", target=str(workspace_dir))

    normalized = dict(spec_dict)
    normalized["p0_spec_version"] = normalized.get("p0_spec_version", 1)
    normalized["immutable_after_start"] = True

    criteria_raw = normalized.get("acceptance_criteria") or []
    criteria_objs = []
    for c in criteria_raw:
        criteria_objs.append(CriterionSpec(
            id=str(c.get("id")),
            statement=str(c.get("statement", "")),
            required=bool(c.get("required", True)),
            verification=dict(c.get("verification") or {}),
            evidence_policy=dict(c.get("evidence_policy") or {}),
        ))

    raw_yaml = dump_governance_yaml(normalized)
    spec_path.write_text(raw_yaml, encoding="utf-8")
    spec_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()

    (gov_dir / "P0_SPEC.sha256").write_text(spec_hash, encoding="utf-8")

    spec_obj = P0Spec(
        p0_spec_version=normalized["p0_spec_version"],
        project_id=str(normalized.get("project", {}).get("project_id", "project")),
        spec_id=str(normalized.get("project", {}).get("spec_id", "spec")),
        spec_hash=spec_hash,
        source_manifest=list(normalized.get("scope", {}).get("source_manifest") or []),
        acceptance_criteria=criteria_objs,
        completion_policy=dict(normalized.get("completion_policy") or {}),
        immutable_after_start=True,
    )
    _record_approved_spec(spec_obj.project_id, spec_obj.spec_id, spec_hash)
    return spec_obj


def amend_acceptance_spec(
    workspace_dir: Path,
    new_spec_dict: dict[str, Any],
    principal_context: TrustedPrincipalContext,
) -> P0Spec:
    """Amend acceptance spec. Strictly restricted to human authority."""
    principal_context.verify_human_authority(action="amend_spec", target=str(workspace_dir))
    return freeze_acceptance_spec(workspace_dir, new_spec_dict, principal_context=principal_context)


def load_frozen_spec(workspace_dir: Path) -> tuple[dict[str, Any], str, list[CriterionSpec], list[str], str]:
    """Load authoritative frozen specification from workspace_dir/.governance/P0_SPEC.yaml.

    Returns (spec_data, spec_hash, criteria_list, source_manifest, project_id).
    Fails closed if the specification file is missing or unreadable,
    or if the YAML content does not match the frozen sidecar anchor or approved control plane hash.
    """
    gov_dir = workspace_dir / ".governance"
    spec_path = gov_dir / "P0_SPEC.yaml"
    sha_path = gov_dir / "P0_SPEC.sha256"
    if not spec_path.is_file():
        raise GovernancePolicyMissing(f"No frozen P0_SPEC.yaml found at {spec_path}")
    if not sha_path.is_file():
        raise GovernancePolicyMissing(f"Missing P0_SPEC.sha256 sidecar anchor at {sha_path}")

    expected_anchor_hash = sha_path.read_text(encoding="utf-8").strip()
    raw_bytes = spec_path.read_bytes()
    actual_yaml_hash = hashlib.sha256(raw_bytes).hexdigest()

    # 1. Local sidecar anchor check: did someone edit P0_SPEC.yaml directly on disk?
    if actual_yaml_hash != expected_anchor_hash:
        raise PermissionError(
            f"GOVERNANCE DENIAL: Tampered P0_SPEC.yaml detected! Actual hash {actual_yaml_hash[:12]} "
            f"does not match frozen anchor {expected_anchor_hash[:12]}. Modifications require amend_acceptance_spec()."
        )

    spec_data = load_governance_yaml(raw_bytes) or {}
    project_id = str(spec_data.get("project", {}).get("project_id", "project"))

    # 2. Control plane approved spec check (fail-closed contract)
    status, rec = get_approved_spec_record(project_id)
    if status == "MISSING_REGISTRY":
        raise PermissionError(
            "GOVERNANCE DENIAL: Control plane approved specification registry is missing. "
            "Fail-closed: cannot verify spec approval."
        )
    elif status == "CORRUPT_REGISTRY":
        raise PermissionError(
            "GOVERNANCE DENIAL: Control plane approved specification registry is corrupt. "
            "Fail-closed: cannot verify spec approval."
        )
    elif status == "UNKNOWN_PROJECT":
        raise PermissionError(
            f"GOVERNANCE DENIAL: Project '{project_id}' has no approved specification in control plane registry. "
            "Fail-closed: unregistered project identity."
        )
    elif status == "APPROVED":
        approved_hash = rec.get("spec_hash") if rec else None
        if not approved_hash or actual_yaml_hash != approved_hash:
            raise PermissionError(
                f"GOVERNANCE DENIAL: P0_SPEC hash {actual_yaml_hash[:12]} does not match control plane "
                f"approved specification hash {str(approved_hash)[:12]}. Unauthorized spec replacement detected."
            )
    else:
        raise PermissionError(f"GOVERNANCE DENIAL: Invalid approved spec status '{status}'.")

    criteria_objs: list[CriterionSpec] = []
    for c in spec_data.get("acceptance_criteria") or []:
        criteria_objs.append(CriterionSpec(
            id=str(c.get("id")),
            statement=str(c.get("statement", "")),
            required=bool(c.get("required", True)),
            verification=dict(c.get("verification") or {}),
            evidence_policy=dict(c.get("evidence_policy") or {}),
        ))
    return spec_data, actual_yaml_hash, criteria_objs, list(spec_data.get("scope", {}).get("source_manifest") or []), project_id


def execute_acceptance_run(
    workspace_dir: Path,
    spec: Optional[P0Spec] = None,
    *,
    criterion_id: Optional[str] = None,
    runner_env: Optional[dict[str, str]] = None,
) -> list[VerificationReceipt]:
    """Execute verification commands for criteria and emit immutable receipts.

    Security Guarantee:
    Always loads and executes the authoritative frozen specification from disk (.governance/P0_SPEC.yaml).
    Caller-supplied in-memory spec objects are never trusted for execution commands or expected predicates.
    """
    spec_data, frozen_spec_hash, criteria_objs, source_manifest, project_id = load_frozen_spec(workspace_dir)

    receipts_dir = workspace_dir / ".governance" / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)

    manifest_hash, tree_hash = compute_source_tree_hash(workspace_dir, source_manifest)
    target_criteria = [c for c in criteria_objs if criterion_id is None or c.id == criterion_id]

    receipts: list[VerificationReceipt] = []

    for crit in target_criteria:
        verif = crit.verification
        v_type = verif.get("type", "command")

        if v_type == "command":
            argv = list(verif.get("argv") or [])
            if not argv:
                continue

            expected = dict(verif.get("expected") or {})
            exp_exit = int(expected.get("exit_code", 0))
            stdout_contains = expected.get("stdout_contains") or expected.get("stdout", {}).get("contains") or []
            stderr_contains = expected.get("stderr_contains") or expected.get("stderr", {}).get("contains") or []

            # Scrub governance authority secrets so verification subprocesses NEVER inherit control plane keys
            # Merge all environment sources FIRST, then filter out any sensitive governance keys
            merged_env = {**os.environ, **(runner_env or {})}
            clean_env = {
                k: v for k, v in merged_env.items()
                if not (
                    k.startswith("HERMES_KANBAN_GOVERNANCE_SECRET")
                    or k.startswith("HERMES_GOVERNANCE_SECRET")
                    or "GOVERNANCE_SECRET" in k.upper()
                    or "GOVERNANCE_KEY" in k.upper()
                )
            }

            started_at = int(time.time())
            try:
                proc = subprocess.run(
                    argv,
                    cwd=workspace_dir,
                    capture_output=True,
                    timeout=verif.get("timeout", 30),
                    env=clean_env,
                )
                exit_code = proc.returncode
                raw_stdout = proc.stdout
                raw_stderr = proc.stderr
            except Exception as exc:
                exit_code = 127
                raw_stdout = b""
                raw_stderr = str(exc).encode("utf-8")
            finished_at = int(time.time())

            out_sha = hashlib.sha256(raw_stdout).hexdigest()
            err_sha = hashlib.sha256(raw_stderr).hexdigest()
            out_str = raw_stdout.decode("utf-8", errors="replace")
            err_str = raw_stderr.decode("utf-8", errors="replace")

            # Check predicates
            exit_code_ok = (exit_code == exp_exit)
            stdout_ok = all(pattern in out_str for pattern in stdout_contains)
            stderr_ok = all(pattern in err_str for pattern in stderr_contains)
            predicate_satisfied = bool(exit_code_ok and stdout_ok and stderr_ok)

            receipt_id = f"vr_{secrets.token_hex(6)}"
            sig_proof = _compute_receipt_signature(
                receipt_id=receipt_id,
                criterion_id=crit.id,
                project_id=project_id,
                spec_hash=frozen_spec_hash,
                source_tree_hash=tree_hash,
                runner_hash=RUNNER_HASH,
                argv=argv,
                exit_code=exit_code,
                stdout_sha256=out_sha,
                stderr_sha256=err_sha,
                expected_exit_code=exp_exit,
                stdout_contains_expected=list(stdout_contains),
                stderr_contains_expected=list(stderr_contains),
                predicate_satisfied=predicate_satisfied,
            )
            receipt = VerificationReceipt(
                receipt_id=receipt_id,
                criterion_id=crit.id,
                project_id=project_id,
                spec_hash=frozen_spec_hash,
                source_manifest_hash=manifest_hash,
                source_tree_hash=tree_hash,
                runner_hash=RUNNER_HASH,
                argv=argv,
                started_at=started_at,
                finished_at=finished_at,
                exit_code=exit_code,
                stdout_sha256=out_sha,
                stderr_sha256=err_sha,
                predicate_satisfied=predicate_satisfied,
                stdout_snippet=out_str[:500],
                stderr_snippet=err_str[:500],
                details={
                    "expected_exit_code": exp_exit,
                    "actual_exit_code": exit_code,
                    "stdout_contains_expected": list(stdout_contains),
                    "stderr_contains_expected": list(stderr_contains),
                    "exit_code_ok": exit_code_ok,
                    "stdout_ok": stdout_ok,
                    "stderr_ok": stderr_ok,
                },
                signature_proof=sig_proof,
            )

            # Persist receipt
            receipt_file = receipts_dir / f"{receipt.criterion_id}.json"
            receipt_file.write_text(json.dumps(asdict(receipt), ensure_ascii=False, indent=2), encoding="utf-8")
            receipts.append(receipt)

    return receipts


def derive_project_acceptance(
    workspace_dir: Path,
    *,
    board_slug: Optional[str] = None,
    unresolved_material_gaps: int = 0,
    governance_violations: int = 0,
) -> ProjectAcceptanceResult:
    """Pure derived status computation. There is NO API to set status to PASS.

    Evaluates:
    - Spec integrity & hash
    - Current source tree hash vs receipt source tree hash (Freshness)
    - Criterion satisfaction
    - Completion policy
    """
    gov_dir = workspace_dir / ".governance"
    # Load and verify authoritative on-disk spec against anchor and control plane registry
    try:
        spec_data, current_spec_hash, criteria_objs, source_manifest, project_id = load_frozen_spec(workspace_dir)
    except PermissionError as exc:
        return ProjectAcceptanceResult(
            status="FAILED_ACCEPTANCE",
            project_id="tampered",
            spec_id="tampered",
            spec_hash="",
            criteria_results={"SPEC_TAMPERED": {"status": "TAMPERED_SPEC", "error": str(exc)}},
            required_passed=0,
            required_total=1,
            unresolved_material_gaps=unresolved_material_gaps,
            governance_violations=governance_violations + 1,
            evaluated_at=int(time.time()),
            evidence_set_hash="",
        )
    except GovernancePolicyMissing:
        return ProjectAcceptanceResult(
            status="NOT_COMPLETE",
            project_id="unknown",
            spec_id="unknown",
            spec_hash="",
            criteria_results={},
            required_passed=0,
            required_total=0,
            unresolved_material_gaps=unresolved_material_gaps,
            governance_violations=governance_violations,
            evaluated_at=int(time.time()),
            evidence_set_hash="",
        )

    curr_manifest_hash, curr_tree_hash = compute_source_tree_hash(workspace_dir, source_manifest)

    criteria_raw = spec_data.get("acceptance_criteria") or []
    criteria_results: dict[str, dict[str, Any]] = {}
    required_total = 0
    required_passed = 0
    has_failed_required = False
    has_stale_required = False
    has_missing_required = False

    receipts_dir = gov_dir / "receipts"
    receipt_hashes: list[str] = []

    for c in criteria_raw:
        cid = str(c.get("id"))
        is_required = bool(c.get("required", True))
        if is_required:
            required_total += 1

        receipt_file = receipts_dir / f"{cid}.json"
        if not receipt_file.is_file():
            criteria_results[cid] = {
                "status": "NO_EVIDENCE",
                "required": is_required,
                "fresh": False,
                "predicate_satisfied": False,
            }
            if is_required:
                has_missing_required = True
            continue

        try:
            r_data = json.loads(receipt_file.read_text(encoding="utf-8"))
        except Exception:
            criteria_results[cid] = {
                "status": "CORRUPT_EVIDENCE",
                "required": is_required,
                "fresh": False,
                "predicate_satisfied": False,
            }
            if is_required:
                has_failed_required = True
            continue

        receipt_hashes.append(hashlib.sha256(receipt_file.read_bytes()).hexdigest())

        # 1. Freshness checks
        is_fresh = (
            r_data.get("source_tree_hash") == curr_tree_hash
            and r_data.get("spec_hash") == current_spec_hash
            and r_data.get("runner_hash") == RUNNER_HASH
        )

        # 2. Cryptographic signature check
        actual_sig = str(r_data.get("signature_proof", ""))
        r_argv = list(r_data.get("argv") or [])
        r_exit_code = int(r_data.get("exit_code", -1))
        r_stdout_sha = str(r_data.get("stdout_sha256", ""))
        r_stderr_sha = str(r_data.get("stderr_sha256", ""))
        r_details = dict(r_data.get("details") or {})
        r_exp_exit = int(r_details.get("expected_exit_code", 0))
        r_stdout_contains = list(r_details.get("stdout_contains_expected") or [])
        r_stderr_contains = list(r_details.get("stderr_contains_expected") or [])
        r_pred_sat = bool(r_data.get("predicate_satisfied", False))

        expected_sig = _compute_receipt_signature(
            receipt_id=str(r_data.get("receipt_id", "")),
            criterion_id=cid,
            project_id=str(r_data.get("project_id", "")),
            spec_hash=str(r_data.get("spec_hash", "")),
            source_tree_hash=str(r_data.get("source_tree_hash", "")),
            runner_hash=str(r_data.get("runner_hash", "")),
            argv=r_argv,
            exit_code=r_exit_code,
            stdout_sha256=r_stdout_sha,
            stderr_sha256=r_stderr_sha,
            expected_exit_code=r_exp_exit,
            stdout_contains_expected=r_stdout_contains,
            stderr_contains_expected=r_stderr_contains,
            predicate_satisfied=r_pred_sat,
        )
        sig_valid = bool(actual_sig and hmac.compare_digest(expected_sig, actual_sig))

        # 3. Independent contract and predicate re-computation against frozen on-disk spec
        crit_verification = c.get("verification", {})
        frozen_argv = list(crit_verification.get("argv") or [])
        crit_expected = crit_verification.get("expected", {})
        frozen_exp_exit = int(crit_expected.get("exit_code", 0))
        frozen_stdout_contains = list(crit_expected.get("stdout_contains") or crit_expected.get("stdout", {}).get("contains") or [])
        frozen_stderr_contains = list(crit_expected.get("stderr_contains") or crit_expected.get("stderr", {}).get("contains") or [])

        contract_matches = (
            r_argv == frozen_argv
            and r_exp_exit == frozen_exp_exit
            and sorted(r_stdout_contains) == sorted(frozen_stdout_contains)
            and sorted(r_stderr_contains) == sorted(frozen_stderr_contains)
        )

        predicates_satisfied_real = (
            r_exit_code == frozen_exp_exit
            and r_pred_sat is True
            and contract_matches
            and sig_valid
        )

        is_tampered = (not sig_valid) or (not contract_matches) or (r_pred_sat and r_exit_code != frozen_exp_exit)

        if is_tampered:
            c_status = "TAMPERED_EVIDENCE"
            if is_required:
                has_failed_required = True
        elif not is_fresh:
            c_status = "STALE_EVIDENCE"
            if is_required:
                has_stale_required = True
        elif not predicates_satisfied_real:
            c_status = "FAILED"
            if is_required:
                has_failed_required = True
        else:
            c_status = "PASS"
            if is_required:
                required_passed += 1

        criteria_results[cid] = {
            "status": c_status,
            "required": is_required,
            "fresh": is_fresh,
            "predicate_satisfied": predicates_satisfied_real,
            "receipt_id": r_data.get("receipt_id"),
            "exit_code": r_data.get("exit_code"),
        }

    # Derive overall status
    if has_failed_required:
        overall_status = "FAILED_ACCEPTANCE"
    elif has_stale_required:
        overall_status = "STALE_EVIDENCE"
    elif has_missing_required:
        overall_status = "NOT_COMPLETE"
    elif required_passed == required_total and required_total > 0:
        if unresolved_material_gaps > 0 or governance_violations > 0:
            overall_status = "BLOCKED"
        else:
            overall_status = "PASS"
    else:
        overall_status = "NOT_COMPLETE"

    evidence_set_hash = hashlib.sha256("".join(sorted(receipt_hashes)).encode("utf-8")).hexdigest()

    return ProjectAcceptanceResult(
        status=overall_status,
        project_id=str(spec_data.get("project", {}).get("project_id", "project")),
        spec_id=str(spec_data.get("project", {}).get("spec_id", "spec")),
        spec_hash=current_spec_hash,
        criteria_results=criteria_results,
        required_passed=required_passed,
        required_total=required_total,
        unresolved_material_gaps=unresolved_material_gaps,
        governance_violations=governance_violations,
        evaluated_at=int(time.time()),
        evidence_set_hash=evidence_set_hash,
    )
