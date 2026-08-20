"""Permanent ``hermes.builder_dispatch.v1`` capability implementation."""

from __future__ import annotations

import json
from pathlib import Path
from pydantic import ValidationError

from .canonical import canonical_sha256
from .errors import AdapterError
from .completion import CompletionAttestor
from .gitops import GitVerifier
from .models import DispatchRequest, ResolvedDispatchRequest, result_record
from .native import KanbanBackend, TaskSnapshot
from .schemas import SchemaRegistry
from .store import DispatchStore
from .validation import ValidationRunner


STATUS_MAP = {
    "triage": ("BLOCKED", False),
    "todo": ("QUEUED", False),
    "scheduled": ("QUEUED", False),
    "ready": ("QUEUED", False),
    "running": ("RUNNING", False),
    "completion_pending": ("RUNNING", False),
    "blocked": ("BLOCKED", False),
    "review": ("BLOCKED", False),
    "done": ("SUCCEEDED", True),
    "archived": ("CANCELLED", True),
}


class BuilderDispatchAdapter:
    def __init__(
        self,
        *,
        store: DispatchStore,
        schemas: SchemaRegistry,
        git: GitVerifier,
        kanban: KanbanBackend,
        validation: ValidationRunner,
        governance_repo: str | Path,
        governance_attestor,
        profile_resolver,
        cycle_registry: dict[str, dict] | None = None,
    ):
        self.store = store
        self.schemas = schemas
        self.git = git
        self.kanban = kanban
        self.validation = validation
        self.governance_repo = Path(governance_repo)
        self.governance_attestor = governance_attestor
        self.profile_resolver = profile_resolver
        self.cycle_registry = cycle_registry or {}

    def _snapshot_for_cycle(self, state: dict):
        commit = state.get("governance_commit")
        path = state.get("contract_path")
        if commit is None and path is None:
            return self.governance_attestor
        if not isinstance(commit, str) or not isinstance(path, str):
            raise AdapterError("CONTRACT_MISMATCH", "cycle governance binding is incomplete")
        from .attestation import GovernanceSnapshot

        return GovernanceSnapshot(
            self.governance_repo,
            commit,
            registered_contract_path=path,
        )

    def _resolve_request(self, intent: DispatchRequest, snapshot) -> ResolvedDispatchRequest:
        """Expand caller IDs exclusively from owner-controlled runtime state."""
        state = self.cycle_registry.get(intent.cycle_id)
        if not isinstance(state, dict):
            raise AdapterError("CONTRACT_MISMATCH", "cycle is not registered")
        if (
            state.get("revision") != intent.expected_cycle_revision
            or state.get("contract_id") != intent.contract_id
            or state.get("repository_id") != intent.repository_id
        ):
            raise AdapterError(
                "CONTRACT_MISMATCH", "cycle identity or revision is not current"
            )
        if not hasattr(snapshot, "bindings"):
            raise AdapterError(
                "CONTRACT_MISMATCH", "trusted governance registry is unavailable"
            )
        manifest_path, _, _ = snapshot.bindings["allowed_path_manifest"]
        manifest_binding = snapshot.contract["artifact_bindings"][
            "allowed_path_manifest"
        ]
        return ResolvedDispatchRequest.model_validate(
            {
                "schema_version": intent.schema_version,
                "dispatch_id": intent.dispatch_id,
                "idempotency_key": intent.idempotency_key,
                "cycle_id": intent.cycle_id,
                "contract": {
                    "contract_id": intent.contract_id,
                    "repository_id": state["governance_repository_id"],
                    "path": getattr(
                        snapshot,
                        "registered_contract_path",
                        snapshot.REGISTERED_CONTRACT_PATH,
                    ),
                    "commit": snapshot.commit,
                    "sha256": snapshot.contract_sha256,
                },
                "repository": {
                    "repository_id": intent.repository_id,
                    "canonical_remote": state["canonical_remote"],
                },
                "worktree_path": state["worktree_path"],
                "branch": state["branch"],
                "expected_head_sha": state["expected_head_sha"],
                "allowed_path_manifest": {
                    "repository_id": state["governance_repository_id"],
                    "path": manifest_path,
                    "commit": snapshot.commit,
                    "sha256": manifest_binding["sha256"],
                },
                "validation_profile": state["validation_profile_id"],
                "builder_role": intent.builder_role,
                "timeout_policy": state["timeout_policy"],
                "retry_policy": state["retry_policy"],
                "completion_schema_version": intent.completion_schema_version,
            }
        )

    def _attest_profile(self, snapshot):
        policy, interface = snapshot.load()
        if (
            policy.get("profile") != "deepseek-builder"
            or policy.get("provider") != "deepseek"
            or policy.get("model") != "deepseek-v4-pro"
            or policy.get("fallback_chain") != []
            or interface.get("capability_id") != "hermes.builder_dispatch.v1"
        ):
            raise AdapterError(
                "PROFILE_POLICY_MISMATCH", "effective builder route is not canonical"
            )
        return policy, self.profile_resolver.resolve(policy)

    def _verify_runtime_governance_selection(self, request, snapshot) -> None:
        """Reject every caller-selected governance coordinate except the snapshot."""
        if not hasattr(snapshot, "bindings"):
            return
        contract = request.contract
        if (
            contract.path
            != getattr(
                snapshot,
                "registered_contract_path",
                snapshot.REGISTERED_CONTRACT_PATH,
            )
            or contract.commit != snapshot.commit
            or contract.sha256 != snapshot.contract_sha256
            or contract.contract_id != snapshot.contract.get("contract_id")
        ):
            raise AdapterError(
                "CONTRACT_MISMATCH", "request does not name the approved governance root"
            )
        manifest_path, _, _ = snapshot.bindings["allowed_path_manifest"]
        manifest = request.allowed_path_manifest
        registered_manifest = snapshot.contract["artifact_bindings"][
            "allowed_path_manifest"
        ]
        if (
            manifest.path != manifest_path
            or manifest.commit != snapshot.commit
            or manifest.sha256 != registered_manifest["sha256"]
        ):
            raise AdapterError(
                "MANIFEST_MISMATCH", "request does not name the approved manifest"
            )
        registered_profile = snapshot.value("validation_profile").get("profile_id")
        if request.validation_profile != registered_profile:
            raise AdapterError(
                "MANIFEST_MISMATCH", "validation profile ID is not registered"
            )

    def _execution_packet(self, request, manifest, policy: dict, snapshot) -> dict:
        contract = getattr(snapshot, "contract", {})
        objective = contract.get("objective", {})
        packet = {
            "schema_version": "1.0.0",
            "dispatch_id": str(request.dispatch_id),
            "cycle_id": request.cycle_id,
            "governance_commit": getattr(
                snapshot, "commit", request.contract.commit
            ),
            "contract_id": request.contract.contract_id,
            "objective": objective.get("summary"),
            "acceptance_criteria": objective.get("success_criteria", []),
            "permitted_paths": list(manifest.patterns),
            "repository": {
                "repository_id": request.repository.repository_id,
                "canonical_remote": request.repository.canonical_remote,
                "worktree": request.worktree_path,
                "branch": request.branch,
                "base_sha": request.expected_head_sha,
            },
            "routing_policy": {
                "profile": policy.get("profile"),
                "provider": policy.get("provider"),
                "model": policy.get("model"),
                "fallback_chain": policy.get("fallback_chain"),
                "allowed_tools": policy.get("allowed_tools"),
            },
            "validation_profile_id": request.validation_profile,
            "limits": {
                "max_runtime_seconds": request.timeout_policy.max_runtime_seconds,
                "heartbeat_timeout_seconds": request.timeout_policy.heartbeat_timeout_seconds,
                "max_attempts": request.retry_policy.max_attempts,
            },
            "prohibitions": policy.get("authority", {}).get("forbidden", []),
        }
        return {"packet": packet, "sha256": canonical_sha256(packet)}

    def _reject(self, operation: str, principal: str, payload: dict, error: AdapterError):
        dispatch_id = str(payload.get("dispatch_id", "00000000-0000-0000-0000-000000000000"))
        cycle_id = str(payload.get("cycle_id", "UNKNOWN"))
        request_hash = canonical_sha256(payload)
        existing = self.store.get(dispatch_id)
        reserved_match = bool(
            existing
            and existing["principal"] == principal
            and existing["request_sha256"] == request_hash
            and payload.get("idempotency_key") == existing["idempotency_key"]
        )
        audit = (
            self.store.new_event_id()
            if reserved_match
            else self.store.audit(
                "PREFLIGHT_REJECTED",
                dispatch_id,
                {"code": error.code, "principal": principal},
            )
        )
        result = result_record(
            operation=operation,
            dispatch_id=dispatch_id,
            cycle_id=cycle_id,
            principal=principal,
            request_sha256=request_hash,
            status="REJECTED",
            side_effects_state="NONE",
            terminal=True,
            errors=[error.as_dict()],
            audit_refs=[audit],
        )
        if reserved_match:
            self.store.transition_with_audit(
                dispatch_id,
                phase="REJECTED",
                result=result,
                event_id=audit,
                kind="PREFLIGHT_REJECTED",
                payload={"code": error.code, "principal": principal},
                expected_principal=principal,
                expected_idempotency_key=existing["idempotency_key"],
                expected_request_sha256=request_hash,
                expected_phase=existing["phase"],
            )
        return result

    def dispatch(self, principal: str, payload: dict) -> dict:
        request_hash = canonical_sha256(payload)
        side_effect_task_id = None
        try:
            self.schemas.validate("dispatch_request", payload)
            intent = DispatchRequest.model_validate(payload)
            cycle_state = self.cycle_registry.get(intent.cycle_id)
            if not isinstance(cycle_state, dict):
                raise AdapterError("CONTRACT_MISMATCH", "cycle is not registered")
            snapshot = self._snapshot_for_cycle(cycle_state)
            request = self._resolve_request(intent, snapshot)
            self._verify_runtime_governance_selection(request, snapshot)
            policy, _ = self._attest_profile(snapshot)
            dispatch_id = str(request.dispatch_id)
            if hasattr(snapshot, "contract_raw"):
                contract_raw = snapshot.contract_raw
            else:
                contract_raw = self.git.verify_artifact(
                    self.governance_repo, request.contract, "CONTRACT_MISMATCH"
                )
            contract = json.loads(contract_raw)
            if contract.get("contract_id") != request.contract.contract_id:
                raise AdapterError("CONTRACT_MISMATCH", "contract identity mismatch")
            if hasattr(snapshot, "raw"):
                manifest_raw = snapshot.raw("allowed_path_manifest")
            else:
                manifest_raw = self.git.verify_artifact(
                    self.governance_repo,
                    request.allowed_path_manifest,
                    "MANIFEST_MISMATCH",
                )
            self.schemas.validate("allowed_manifest", json.loads(manifest_raw))
            worktree = self.git.verify_worktree(request)
            manifest = self.git.manifest_from_artifact(manifest_raw)
            if manifest.base_sha != request.expected_head_sha:
                raise AdapterError("MANIFEST_MISMATCH", "manifest base SHA mismatch")
            if request.validation_profile not in self.validation._profiles:
                raise AdapterError("MANIFEST_MISMATCH", "validation profile unregistered")
            packet = self._execution_packet(request, manifest, policy, snapshot)
            # Reserve only after every side-effect-free preflight has passed.
            # A preflight rejection therefore cannot poison an idempotency key.
            record, created = self.store.reserve(
                dispatch_id,
                request.idempotency_key,
                request_hash,
                request.cycle_id,
                principal,
                request.model_dump(mode="json"),
                packet,
            )
            reservation_audit = record.get("reservation_event_id")
            if not created:
                if record.get("result_json"):
                    prior = json.loads(record["result_json"])
                    if prior.get("terminal"):
                        return prior
                if record.get("task_id"):
                    return self._status_for_record(record, operation="dispatch")
                # Recovery across the reservation/task-creation boundary is
                # delegated to Kanban's durable idempotency key. Reissuing the
                # same canonical create operation must return the same task.
            task_id = self.kanban.create_task(request_hash, request)
            side_effect_task_id = task_id
            audit = self.store.new_event_id()
            audit_payload = {
                "task_id": task_id,
                "request_sha256": request_hash,
                "principal": principal,
                "worktree": str(worktree),
            }
            result = result_record(
                operation="dispatch",
                dispatch_id=dispatch_id,
                cycle_id=request.cycle_id,
                principal=principal,
                request_sha256=request_hash,
                status="ACCEPTED",
                side_effects_state="STARTED",
                terminal=False,
                task_id=task_id,
                audit_refs=[ref for ref in (reservation_audit, audit) if ref],
            )
            self.store.transition_with_audit(
                dispatch_id,
                phase="TASK_CREATED",
                task_id=task_id,
                result=result,
                event_id=audit,
                kind="KANBAN_TASK_CREATED",
                payload=audit_payload,
            )
            return result
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return self._reject(
                "dispatch",
                principal,
                payload,
                AdapterError("INVALID_REQUEST", "request validation failed"),
            )
        except AdapterError as error:
            if side_effect_task_id:
                unknown = AdapterError(
                    "DISPATCH_STATE_UNKNOWN",
                    "task exists but journal transition requires reconciliation",
                )
                audit = self.store.audit(
                    "TASK_CORRELATION_PENDING",
                    str(payload.get("dispatch_id", "")),
                    {"task_id": side_effect_task_id, "principal": principal},
                )
                return result_record(
                    operation="dispatch",
                    dispatch_id=str(payload.get("dispatch_id", "")),
                    cycle_id=str(payload.get("cycle_id", "UNKNOWN")),
                    principal=principal,
                    request_sha256=request_hash,
                    status="UNKNOWN",
                    side_effects_state="STARTED",
                    terminal=False,
                    task_id=side_effect_task_id,
                    errors=[unknown.as_dict()],
                    audit_refs=[audit],
                )
            if "dispatch_id" in payload:
                existing = self.store.get(str(payload["dispatch_id"]))
                if existing and existing.get("task_id"):
                    unknown = AdapterError(
                        "DISPATCH_STATE_UNKNOWN",
                        "side effect may have started; deterministic reconciliation required",
                    )
                    return result_record(
                        operation="dispatch",
                        dispatch_id=str(payload["dispatch_id"]),
                        cycle_id=str(payload.get("cycle_id", "UNKNOWN")),
                        principal=principal,
                        request_sha256=request_hash,
                        status="UNKNOWN",
                        side_effects_state="UNKNOWN",
                        terminal=False,
                        task_id=existing["task_id"],
                        errors=[unknown.as_dict()],
                    )
            return self._reject("dispatch", principal, payload, error)
        except Exception:
            if side_effect_task_id:
                error = AdapterError(
                    "DISPATCH_STATE_UNKNOWN",
                    "unexpected failure after native task creation",
                )
                audit = self.store.audit(
                    "TASK_CORRELATION_PENDING",
                    str(payload.get("dispatch_id", "")),
                    {"task_id": side_effect_task_id, "principal": principal},
                )
                return result_record(
                    operation="dispatch",
                    dispatch_id=str(payload.get("dispatch_id", "")),
                    cycle_id=str(payload.get("cycle_id", "UNKNOWN")),
                    principal=principal,
                    request_sha256=request_hash,
                    status="UNKNOWN",
                    side_effects_state="STARTED",
                    terminal=False,
                    task_id=side_effect_task_id,
                    errors=[error.as_dict()],
                    audit_refs=[audit],
                )
            error = AdapterError("INTERNAL_ERROR", "dispatch failed closed")
            existing = self.store.get(str(payload.get("dispatch_id", "")))
            if existing and existing.get("task_id"):
                error = AdapterError(
                    "DISPATCH_STATE_UNKNOWN",
                    "unexpected failure after task correlation",
                )
                audit = self.store.audit(
                    "RECONCILIATION_REQUIRED",
                    str(payload.get("dispatch_id", "")),
                    {"task_id": existing["task_id"], "principal": principal},
                )
                return result_record(
                    operation="dispatch",
                    dispatch_id=str(payload.get("dispatch_id", "")),
                    cycle_id=str(payload.get("cycle_id", "UNKNOWN")),
                    principal=principal,
                    request_sha256=request_hash,
                    status="UNKNOWN",
                    side_effects_state="STARTED",
                    terminal=False,
                    task_id=existing["task_id"],
                    errors=[error.as_dict()],
                    audit_refs=[audit],
                )
            return self._reject("dispatch", principal, payload, error)

    def _status_for_record(self, record: dict, *, operation: str) -> dict:
        snapshot = self.kanban.snapshot(record["task_id"])
        if snapshot.status == "done":
            if record.get("result_json") and record.get("phase") == "COMPLETED":
                return json.loads(record["result_json"])
            if (
                record.get("result_json")
                and record.get("phase") == "COMPLETION_ATTESTED"
            ):
                release = getattr(self.kanban, "release_completion_lease", None)
                if release is None or not release(record["task_id"]):
                    raise AdapterError(
                        "DISPATCH_STATE_UNKNOWN",
                        "completion lease release remains unconfirmed",
                    )
                result = json.loads(record["result_json"])
                self.store.update(
                    record["dispatch_id"],
                    phase="COMPLETED",
                    result=result,
                )
                return result
            if not record.get("request_json"):
                raise AdapterError(
                    "DISPATCH_STATE_UNKNOWN",
                    "terminal task lacks an independently verified completion path",
                )
            request = ResolvedDispatchRequest.model_validate(
                json.loads(record["request_json"])
            )
            cycle_state = self.cycle_registry.get(request.cycle_id)
            if not isinstance(cycle_state, dict):
                raise AdapterError("CONTRACT_MISMATCH", "cycle is not registered")
            governance_snapshot = self._snapshot_for_cycle(cycle_state)
            if hasattr(governance_snapshot, "raw"):
                manifest_raw = governance_snapshot.raw("allowed_path_manifest")
            else:
                manifest_raw = self.git.verify_artifact(
                    self.governance_repo,
                    request.allowed_path_manifest,
                    "MANIFEST_MISMATCH",
                )
            manifest = self.git.manifest_from_artifact(manifest_raw)
            if not self.kanban.completion_exclusive(record["task_id"]):
                raise AdapterError(
                    "DISPATCH_STATE_UNKNOWN",
                    "exclusive worker termination cannot be proven",
                )
            _, effective_profile = self._attest_profile(governance_snapshot)
            completion = CompletionAttestor(
                self.git,
                self.validation,
                self.schemas,
                effective_profile,
            )
            evidence = completion.complete(
                request,
                governance_snapshot,
                record["principal"],
                record["request_sha256"],
                manifest,
            )
            audit = self.store.new_event_id()
            audit_payload = {"resulting_sha": evidence["git"]["resulting_sha"]}
            evidence["audit_event_refs"] = [audit]
            self.schemas.validate("completion_evidence", evidence)
            result = result_record(
                operation=operation,
                dispatch_id=record["dispatch_id"],
                cycle_id=record["cycle_id"],
                principal=record["principal"],
                request_sha256=record["request_sha256"],
                status="SUCCEEDED",
                side_effects_state="STARTED",
                terminal=True,
                attempt_count=snapshot.attempt_count,
                task_id=snapshot.task_id,
                run_ids=snapshot.run_ids,
                evidence=evidence,
                audit_refs=[audit],
            )
            self.schemas.validate("dispatch_result", result)
            self.store.transition_with_audit(
                record["dispatch_id"],
                phase="COMPLETION_ATTESTED",
                result=result,
                event_id=audit,
                kind="COMPLETION_ATTESTED",
                payload=audit_payload,
            )
            release = getattr(self.kanban, "release_completion_lease", None)
            if release is None or not release(record["task_id"]):
                raise AdapterError(
                    "DISPATCH_STATE_UNKNOWN",
                    "completion evidence persisted but lease release was not confirmed",
                )
            self.store.update(
                record["dispatch_id"],
                phase="COMPLETED",
                result=result,
            )
            return result
        status, terminal = STATUS_MAP.get(snapshot.status, ("UNKNOWN", False))
        errors = []
        side_effects = "STARTED"
        if status == "UNKNOWN":
            side_effects = "UNKNOWN"
            errors = [
                AdapterError(
                    "DISPATCH_STATE_UNKNOWN", "native task state is unknown"
                ).as_dict()
            ]
        return result_record(
            operation=operation,
            dispatch_id=record["dispatch_id"],
            cycle_id=record["cycle_id"],
            principal=record["principal"],
            request_sha256=record["request_sha256"],
            status=status,
            side_effects_state=side_effects,
            terminal=terminal,
            attempt_count=snapshot.attempt_count,
            task_id=snapshot.task_id,
            run_ids=snapshot.run_ids,
            errors=errors,
        )

    def get_status(self, principal: str, dispatch_id: str, cycle_id: str) -> dict:
        record = self.store.get(dispatch_id)
        if not record or record["cycle_id"] != cycle_id:
            raise AdapterError("INVALID_REQUEST", "dispatch identity not found")
        if record["principal"] != principal:
            raise AdapterError("AUTHORIZATION_FAILED", "dispatch principal mismatch")
        return self._status_for_record(record, operation="get_status")

    def cancel(
        self, principal: str, dispatch_id: str, cycle_id: str, reason_code: str
    ) -> dict:
        allowed = {
            "HUMAN_CANCELLED",
            "CONTRACT_SUPERSEDED",
            "TIMEOUT",
            "GOVERNANCE_REJECTED",
        }
        if reason_code not in allowed:
            raise AdapterError("INVALID_REQUEST", "invalid cancellation reason")
        record = self.store.get(dispatch_id)
        if not record or record["cycle_id"] != cycle_id:
            raise AdapterError("INVALID_REQUEST", "dispatch identity not found")
        if record["principal"] != principal:
            raise AdapterError("AUTHORIZATION_FAILED", "dispatch principal mismatch")
        proof = self.kanban.cancel(record["task_id"], reason_code)
        if (
            not proof.confirmed
            or not proof.process_tree_terminated
            or not proof.task_archived
        ):
            error = AdapterError(
                "CANCELLATION_UNCONFIRMED",
                "worker process-tree termination could not be proven",
            )
            audit = self.store.audit(
                "CANCELLATION_UNCONFIRMED",
                dispatch_id,
                {"reason_code": reason_code, "principal": principal},
            )
            return result_record(
                operation="cancel",
                dispatch_id=dispatch_id,
                cycle_id=cycle_id,
                principal=principal,
                request_sha256=record["request_sha256"],
                status="BLOCKED",
                side_effects_state="UNKNOWN",
                terminal=False,
                task_id=record["task_id"],
                errors=[error.as_dict()],
                audit_refs=[audit],
            )
        audit = self.store.new_event_id()
        result = result_record(
            operation="cancel",
            dispatch_id=dispatch_id,
            cycle_id=cycle_id,
            principal=principal,
            request_sha256=record["request_sha256"],
            status="CANCELLED",
            side_effects_state="STARTED",
            terminal=True,
            task_id=record["task_id"],
            audit_refs=[audit],
            errors=[
                AdapterError("CANCELLED", "dispatch cancelled by policy").as_dict()
            ],
        )
        self.store.transition_with_audit(
            dispatch_id,
            phase="CANCELLED",
            result=result,
            event_id=audit,
            kind="TASK_ARCHIVED",
            payload={"reason_code": reason_code},
        )
        return result
