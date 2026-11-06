"""Task-scoped local GPU admission and exact JSON completion oracle.

This candidate controller supports only registry-pinned, fixed-facts JSON cards with a
task-specific expected fixture. It coordinates cooperating Resource Balancer jobs; it does
not make direct Ollama or CUDA clients participate in that queue.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from hermes_cli.kanban_local_model_scope import (
    CONTRACT_SCHEMA,
    capability_id_from_body,
    task_run_contract_sha256,
)

_ADMISSION_FIELDS = (
    "task_id", "kanban_task_id", "task_run_id", "kanban_board", "kanban_db", "kanban_workspace",
    "kanban_claim_lock", "capability_id", "task_contract_sha256", "api_request_id",
    "context_length", "ollama_num_ctx", "session_id", "model", "provider", "base_url",
    "streaming", "has_tools", "api_call_count", "existing_response_format",
)
_STABLE_FIELDS = tuple(field for field in _ADMISSION_FIELDS
                       if field not in {"api_request_id", "api_call_count", "existing_response_format"})
_VERIFY_FIELDS = (
    "task_id", "kanban_task_id", "task_run_id", "kanban_claim_lock", "kanban_board", "kanban_db",
    "kanban_workspace", "capability_id", "task_contract_sha256", "workspace_path", "assignee",
    "model_override", "provider_override", "completion_contract", "artifacts",
)
_MAX_ARTIFACT_BYTES = 1024 * 1024
_LOGGER = logging.getLogger(__name__)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _json_object(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object)
    if type(value) is not dict:
        raise ValueError("expected a JSON object")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _schema_for_value(value: Any) -> dict[str, Any]:
    """Build a JSON Schema from JSON types and object keys, never from fixture values."""
    if value is None:
        return {"type": "null"}
    if type(value) is bool:
        return {"type": "boolean"}
    if type(value) is int:
        return {"type": "integer"}
    if type(value) is float:
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        if not value:
            return {"type": "array"}
        choices = []
        for item in value:
            schema = _schema_for_value(item)
            if schema not in choices:
                choices.append(schema)
        return {"type": "array", "items": choices[0] if len(choices) == 1 else {"anyOf": choices}}
    if isinstance(value, dict):
        if not value:
            return {"type": "object"}
        return {
            "type": "object",
            "properties": {key: _schema_for_value(item) for key, item in value.items()},
            "required": list(value),
            "additionalProperties": False,
        }
    raise ValueError("expected JSON fixture contains an unsupported type")


def _response_format_for(expected: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"schema": _schema_for_value(expected)},
    }


def _safe_error_detail(exc: Exception) -> str:
    """Return a bounded single-line controller reason for the operator-facing denial."""
    detail = " ".join(str(exc).split())
    return detail[:240] if detail else type(exc).__name__


def _normalize_url(value: str) -> tuple[str, str, str]:
    parsed = urlsplit(value.strip())
    if not parsed.scheme or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError("local model endpoint is malformed")
    return parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/")


def _run_pinned_verifier(verifier_path: Path, artifact: Path) -> subprocess.CompletedProcess:
    """Run the registry-pinned verifier with a clean environment and a short bound."""
    result = subprocess.run(
        [sys.executable, str(verifier_path), str(artifact)],
        cwd=str(verifier_path.parent),
        env={"PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, timeout=5, check=False,
    )
    if result.returncode != 0:
        raise ValueError("pinned independent verifier rejected the artifact")
    return result


class _ActiveLease:
    def __init__(self, manager, job_id, admission_scope, task, registry_hash,
                 verifier_hash, verifier_path, fixture_hash, runtime_evidence, expected, response_format):
        self.manager = manager
        self.job_id = job_id
        self.admission_scope = admission_scope
        self.task = task
        self.registry_hash = registry_hash
        self.verifier_hash = verifier_hash
        self.verifier_path = verifier_path
        self.fixture_hash = fixture_hash
        self.runtime_evidence = runtime_evidence
        self.expected = expected
        self.response_format = response_format
        evidence_payload = (
            f"{task.id}:{task.current_run_id}:{admission_scope['task_contract_sha256']}:{registry_hash}:"
            f"{verifier_hash}:{fixture_hash}:{json.dumps(runtime_evidence, sort_keys=True)}:"
            f"{json.dumps(response_format, sort_keys=True)}"
        ).encode()
        self.evidence_id = _sha256(evidence_payload)


class ResourceBalancerController:
    """Single-profile trusted callback for the Hermes local-model admission contract."""

    def __init__(self, *, profile: str, config: dict[str, Any]):
        self.profile = str(profile or "")
        self.rb_python_path = Path(str(config.get("resource_balancer_python_path") or "")).expanduser()
        self.state_dir = Path(str(config.get("resource_balancer_state_dir") or "")).expanduser()
        self.registry_path = Path(str(config.get("capability_registry_path") or "")).expanduser()
        self.registry_sha256 = str(config.get("capability_registry_sha256") or "").lower()
        self._leases: dict[tuple[str, str, str], _ActiveLease] = {}
        self._lock = threading.RLock()

    def __call__(self, *, action: str, lease_ids: tuple[str, ...] = (), **context: Any) -> dict[str, Any]:
        try:
            if action == "acquire":
                return self._acquire(context)
            if action == "validate":
                return self._validate(context, tuple(lease_ids))
            if action == "release":
                return self._release(context, tuple(lease_ids))
            if action == "verify_completion":
                return self._verify_completion(context)
            return self._block("unsupported_action", "Unsupported admission action")
        except Exception as exc:
            # Never let a callback exception turn into provider fallthrough.
            _LOGGER.exception("Local GPU admission callback failed (action=%s)", action)
            return self._block("controller_error", f"Local GPU admission refused: {_safe_error_detail(exc)}")

    @staticmethod
    def _block(policy_id: str, reason: str) -> dict[str, Any]:
        return {"decision": "block", "policy_id": policy_id, "reason": reason}

    @staticmethod
    def _scope(context: dict[str, Any], fields=_ADMISSION_FIELDS) -> dict[str, str]:
        def render(value: Any) -> str:
            if isinstance(value, bool):
                return str(value).lower()
            return "" if value is None else str(value)

        return {field: render(context.get(field)) for field in fields}

    @staticmethod
    def _key(scope: dict[str, str]) -> tuple[str, str, str]:
        return scope["kanban_task_id"], scope["task_run_id"], scope["kanban_claim_lock"]

    def _configuration(self) -> tuple[Path, Path, Path]:
        rb_root = self.rb_python_path.resolve(strict=True)
        registry = self.registry_path.resolve(strict=True)
        state_dir = self.state_dir.resolve(strict=True)
        if not (rb_root / "resource_balancer" / "admission.py").is_file():
            raise ValueError("Resource Balancer package path is invalid")
        if not (state_dir / "queue.sqlite3").is_file():
            raise ValueError("Resource Balancer queue database is unavailable")
        if not registry.is_file() or len(self.registry_sha256) != 64:
            raise ValueError("Capability registry configuration is incomplete")
        if str(rb_root) not in sys.path:
            sys.path.insert(0, str(rb_root))
        return rb_root, state_dir, registry

    def _read_task(self, context: dict[str, Any]):
        db_path = Path(str(context.get("kanban_db") or "")).expanduser().resolve(strict=True)
        if db_path != Path(os.environ.get("HERMES_KANBAN_DB", "")).expanduser().resolve(strict=True):
            raise ValueError("pinned Kanban database mismatch")
        task_id = str(context.get("kanban_task_id") or "")
        if not task_id:
            raise ValueError("Kanban task identity is missing")
        run_id = int(str(context.get("task_run_id") or ""))
        uri = db_path.as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=1)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                raise ValueError("Kanban task is missing")
            from hermes_cli.kanban_db import Task
            task = Task.from_row(row)
            run = conn.execute(
                "SELECT task_id,status,claim_lock,worker_pid FROM task_runs WHERE id=?", (run_id,),
            ).fetchone()
        finally:
            conn.close()
        if (task.status != "running" or task.current_run_id != run_id
                or task.claim_lock != context.get("kanban_claim_lock")
                or task.workspace_path != context.get("kanban_workspace")
                or task.assignee != self.profile
                or task.completion_contract != "local-only"
                or task.workspace_kind != "scratch"
                or task.max_runtime_seconds != 300
                or task.max_retries != 1
                or task.goal_mode
                or task.skills != ["local-gpu-model-task-gate"]
                or capability_id_from_body(task.body) != context.get("capability_id")
                or task_run_contract_sha256(task) != context.get("task_contract_sha256")):
            raise ValueError("task/run/claim/capability contract no longer matches")
        workspace = Path(task.workspace_path or "").resolve(strict=True)
        from hermes_cli.kanban_db import _is_managed_scratch_path
        if not workspace.is_dir() or not _is_managed_scratch_path(workspace):
            raise ValueError("task workspace is not a managed scratch directory")
        if (run is None or run["task_id"] != task.id or run["status"] != "running"
                or run["claim_lock"] != task.claim_lock
                or run["worker_pid"] != os.getpid()):
            raise ValueError("current worker does not own the claimed run")
        return task

    def _load_oracle(self, task, scope):
        registry_hash, registry_path, capability = self._pinned_capability(scope)
        self._validate_capability_envelope(capability, scope)
        live_runtime = self._validate_live_route(capability, task, scope)
        verifier_hash, verifier_path, fixture_hash, expected = self._load_verifier_oracle(
            registry_path, capability, task,
        )
        return (registry_hash, verifier_hash, verifier_path, fixture_hash,
                live_runtime, expected, _response_format_for(expected))

    def _pinned_capability(self, scope):
        _rb_root, _state_dir, registry_path = self._configuration()
        registry_bytes = registry_path.read_bytes()
        registry_hash = _sha256(registry_bytes)
        if registry_hash != self.registry_sha256:
            raise ValueError("capability registry digest mismatch")
        registry = json.loads(registry_bytes.decode("utf-8"))
        binding = registry.get("task_run_binding") or {}
        if binding.get("schema") != CONTRACT_SCHEMA:
            raise ValueError("task/run binding schema mismatch")
        capabilities = [item for item in registry.get("capabilities", [])
                        if item.get("id") == scope["capability_id"]]
        if len(capabilities) != 1:
            raise ValueError("capability record is missing or ambiguous")
        return registry_hash, registry_path, capabilities[0]

    @staticmethod
    def _validate_capability_envelope(capability, scope):
        route_evidence = capability.get("profile_route_evidence") or {}
        trial_binding = capability.get("trial_binding") or {}
        mapping_review = trial_binding.get("mapping_review") or {}
        if (trial_binding.get("admission_capability_id") != capability.get("id")
                or not trial_binding.get("trial_task_class")
                or trial_binding.get("trial_task_class") != route_evidence.get("task_class")
                or mapping_review.get("status") != "approved"
                or not str(mapping_review.get("reviewer") or "").strip()
                or not str(mapping_review.get("reviewed_at_utc") or "").strip()):
            raise ValueError("trial task-class mapping is missing or not independently reviewed")
        if (capability.get("allow") is not True
                or not str(capability.get("status", "")).startswith("narrowly_supported")
                or capability.get("model_tag") != scope["model"]):
            raise ValueError("model/capability pair is outside the proven envelope")
        if capability.get("request_schema_policy") != "json_schema_from_fixture_types_v1":
            raise ValueError("capability does not authorize the pinned source-type JSON Schema")
        if (scope.get("streaming") != "false" or scope.get("has_tools") != "false"
                or scope.get("existing_response_format") != "false"
                or scope.get("api_call_count") != "1"):
            raise ValueError("fixed-JSON capability requires its first non-streaming, tool-free request")

    def _validate_live_route(self, capability, task, scope):
        runtime = capability.get("observed_runtime") or {}
        expected_digest = str(capability.get("model_digest") or "").lower()
        if len(expected_digest) != 64:
            raise ValueError("historical capability evidence has no pinned model digest")
        expected_quantization = str(runtime.get("quantization") or "").strip().upper()
        expected_runtime_version = str(runtime.get("runtime_version") or "").strip()
        if not expected_quantization or not expected_runtime_version:
            raise ValueError("capability evidence is missing exact quantization or runtime version")
        expected_backend = str(runtime.get("backend") or "").strip().lower()
        actual_provider = str(scope.get("provider") or "").strip().lower()
        endpoint = urlsplit(scope["base_url"])
        if endpoint.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("pinned Ollama endpoint is not loopback")
        live_runtime = self._live_model_evidence(scope)
        if live_runtime["digest"] != expected_digest:
            raise ValueError("installed model digest differs from the digest used in capability trials")
        if live_runtime["quantization"] != expected_quantization:
            raise ValueError("installed model quantization differs from capability trials")
        if live_runtime["runtime_version"] != expected_runtime_version:
            raise ValueError("Ollama runtime version differs from capability trials")
        if int(scope["ollama_num_ctx"]) != int(runtime.get("ollama_num_ctx") or 0):
            raise ValueError("Ollama num_ctx differs from capability trials")
        allowed_profiles = set(capability.get("observed_profiles") or [])
        if runtime.get("profile"):
            allowed_profiles.add(runtime["profile"])
        if self.profile not in allowed_profiles:
            raise ValueError("profile is outside the proven capability envelope")
        if (int(scope["context_length"]) != int(runtime.get("context_length") or 0)
                or expected_backend != actual_provider
                or _normalize_url(str(runtime.get("base_url") or "")) != _normalize_url(scope["base_url"])):
            raise ValueError("runtime/provider/context differs from the measured route")
        if task.model_override and task.model_override != scope["model"]:
            raise ValueError("task model override differs from the admitted model")
        if (task.provider_override
                and task.provider_override.strip().lower() != actual_provider):
            raise ValueError("task provider override differs from the admitted provider")
        return live_runtime

    @staticmethod
    def _load_verifier_oracle(registry_path, capability, task):
        verifier_records = []
        one = capability.get("prepared_external_verifier")
        if isinstance(one, dict):
            verifier_records.append(one)
        more = capability.get("additional_prepared_external_verifiers")
        if isinstance(more, list):
            verifier_records.extend(item for item in more if isinstance(item, dict))
        records = [item for item in verifier_records if item.get("task_id") == task.id]
        if len(records) != 1:
            raise ValueError("no task-specific independent verifier is pinned")
        verifier = records[0]
        root = registry_path.parent.resolve(strict=True)
        verifier_path = (root / verifier["verifier_path"]).resolve(strict=True)
        fixture_path = (root / verifier["expected_fixture_path"]).resolve(strict=True)
        if not verifier_path.is_relative_to(root) or not fixture_path.is_relative_to(root):
            raise ValueError("verifier assets escape the registry directory")
        verifier_bytes = verifier_path.read_bytes()
        fixture_bytes = fixture_path.read_bytes()
        verifier_hash, fixture_hash = _sha256(verifier_bytes), _sha256(fixture_bytes)
        if (verifier_hash != verifier.get("verifier_sha256")
                or fixture_hash != verifier.get("expected_fixture_sha256")):
            raise ValueError("independent verifier asset digest mismatch")
        expected = _json_object(fixture_bytes)
        # The expected object is an oracle for completion verification only. Never require or
        # inject it into the task body: doing so leaks the answer to the model and makes the
        # capability canary measure copying rather than task performance.
        return verifier_hash, verifier_path, fixture_hash, expected

    @staticmethod
    def _live_model_evidence(scope: dict[str, str]) -> dict[str, str]:
        endpoint = urlsplit(scope["base_url"])
        tags_url = urlunsplit((endpoint.scheme, endpoint.netloc, "/api/tags", "", ""))
        with urlopen(tags_url, timeout=2) as response:
            tags_bytes = response.read(1024 * 1024 + 1)
        if len(tags_bytes) > 1024 * 1024:
            raise ValueError("Ollama model inventory exceeds the admission response limit")
        installed = json.loads(tags_bytes.decode("utf-8"))
        matches = [model for model in installed.get("models", [])
                   if model.get("name") == scope["model"]]
        if len(matches) != 1:
            raise ValueError("exact model tag is not uniquely installed in Ollama")
        details = matches[0].get("details") or {}
        quantization = str(details.get("quantization_level") or "").strip().upper()
        digest = str(matches[0].get("digest") or "").strip().lower()
        if not digest or not quantization:
            raise ValueError("Ollama did not report exact model digest and quantization")
        version_url = urlunsplit((endpoint.scheme, endpoint.netloc, "/api/version", "", ""))
        with urlopen(version_url, timeout=2) as response:
            version_bytes = response.read(64 * 1024 + 1)
        if len(version_bytes) > 64 * 1024:
            raise ValueError("Ollama version response exceeds the admission response limit")
        version_data = json.loads(version_bytes.decode("utf-8"))
        runtime_version = str(version_data.get("version") or "").strip()
        if not runtime_version:
            raise ValueError("Ollama did not report its runtime version")
        return {"digest": digest, "quantization": quantization, "runtime_version": runtime_version}

    def _resource_balancer_status(self, state_dir: Path, job_id: str = "") -> dict[str, Any]:
        db_path = state_dir / "queue.sqlite3"
        conn = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True, timeout=1)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT data FROM controller WHERE id=1").fetchone()
            if row is None:
                raise ValueError("Resource Balancer controller has no status")
            status = json.loads(row["data"])
            if job_id:
                job = conn.execute("SELECT class,state FROM jobs WHERE id=?", (job_id,)).fetchone()
                if job is None or job["class"] not in ("gpu", "mixed") or job["state"] != "running":
                    raise ValueError("Resource Balancer GPU lease is not active")
                other = conn.execute(
                    "SELECT id FROM jobs WHERE class IN ('gpu','mixed') "
                    "AND state IN ('starting','running') AND id<>? LIMIT 1", (job_id,),
                ).fetchone()
                if other is not None:
                    raise ValueError("another cooperating GPU lease is active")
        finally:
            conn.close()
        if (status.get("mode") != "manage" or status.get("blocked") is not False
                or time.time() - float(status.get("updated") or 0) > 30
                or (status.get("resource_gates") or {}).get("gpu", {}).get("blocked") is not False):
            raise ValueError("Resource Balancer is stale, blocked, or GPU-gated")
        return status

    @staticmethod
    def _validate_external_reservation(state_dir: Path, job_id: str,
                                       scope: dict[str, str]) -> None:
        """Validate a pre-spawn RB reservation carried through an exec handoff.

        The dispatcher-side RB exec adapter owns the lease for the lifetime of
        this worker. Environment metadata alone is never sufficient: the live
        RB row must contain the same task/run/claim/contract/capability tuple.
        """
        conn = sqlite3.connect((state_dir / "queue.sqlite3").as_uri() + "?mode=ro",
                               uri=True, timeout=1)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT j.class,j.state,r.context FROM jobs AS j "
                "LEFT JOIN hermes_local_gpu_reservations AS r ON r.job_id=j.id WHERE j.id=?",
                (job_id,),
            ).fetchone()
        except sqlite3.OperationalError as exc:
            raise ValueError("Resource Balancer has no task-bound reservation table") from exc
        finally:
            conn.close()
        if row is None or row["class"] != "gpu" or row["state"] != "running":
            raise ValueError("pre-spawn Resource Balancer reservation is not active")
        try:
            reservation = json.loads(row["context"] or "")
        except (TypeError, ValueError) as exc:
            raise ValueError("pre-spawn reservation context is missing or malformed") from exc
        expected = {
            "task_id": scope["kanban_task_id"],
            "run_id": scope["task_run_id"],
            "claim_lock": scope["kanban_claim_lock"],
            "task_contract_sha256": scope["task_contract_sha256"],
            "capability_id": scope["capability_id"],
        }
        if reservation != expected:
            raise ValueError("pre-spawn reservation does not match the current task/run contract")

    def _acquire(self, context: dict[str, Any]) -> dict[str, Any]:
        scope = self._scope(context)
        if not all(scope[field] for field in _ADMISSION_FIELDS):
            return self._block("missing_scope", "Task/run/provider scope is incomplete")
        key = self._key(scope)
        with self._lock:
            if key in self._leases:
                return self._block("duplicate_acquire", "This task/run already owns an admission lease")
        try:
            task = self._read_task(context)
            registry_hash, verifier_hash, verifier_path, fixture_hash, runtime_evidence, expected, response_format = self._load_oracle(task, scope)
            _rb_root, state_dir, _registry_path = self._configuration()
            self._resource_balancer_status(state_dir)
            external_job_id = os.environ.get("RESOURCE_BALANCER_ADMISSION_JOB_ID", "").strip()
            manager = None
            if external_job_id:
                self._validate_external_reservation(state_dir, external_job_id, scope)
                job_id = external_job_id
            else:
                from resource_balancer.admission import admission
                from resource_balancer.integration import process_identity

                manager = admission(
                    "gpu", state_dir=state_dir, origin="hermes-kanban-local-model",
                    owner={"pid": os.getpid(), "start": process_identity(os.getpid())},
                )
                job_id = manager.__enter__()
            try:
                # Queue wait may outlive the original claim; re-read all identities after it obtains a slot.
                task_after_wait = self._read_task(context)
                if manager is None:
                    self._validate_external_reservation(state_dir, str(job_id), scope)
                self._resource_balancer_status(state_dir, str(job_id))
                if task_after_wait.current_run_id != task.current_run_id:
                    raise ValueError("Kanban run changed while waiting for GPU admission")
            except Exception:
                if manager is not None:
                    manager.__exit__(None, None, None)
                raise
            lease = _ActiveLease(manager, str(job_id), scope, task_after_wait,
                                 registry_hash, verifier_hash, verifier_path, fixture_hash,
                                 runtime_evidence, expected, response_format)
            with self._lock:
                self._leases[key] = lease
            return {
                "decision": "allow", "policy_id": "local-gpu-task-capability-v1",
                "reason": "Exact task/run contract, route, verifier, and Resource Balancer GPU lease passed",
                "evidence_id": lease.evidence_id, "lease_ids": [str(job_id)], "scope": scope,
                "response_format": response_format,
            }
        except Exception as exc:
            _LOGGER.exception("Local GPU admission acquire refused")
            return self._block("acquire_refused", f"Admission refused: {_safe_error_detail(exc)}")

    def _validate(self, context: dict[str, Any], lease_ids: tuple[str, ...]) -> dict[str, Any]:
        scope = self._scope(context)
        key = self._key(scope)
        with self._lock:
            lease = self._leases.get(key)
        if lease is None or lease_ids != (lease.job_id,):
            return self._block("lease_missing", "No matching task/run GPU lease exists")
        stable = {field: scope[field] for field in _STABLE_FIELDS}
        expected_stable = {field: lease.admission_scope[field] for field in _STABLE_FIELDS}
        if stable != expected_stable:
            return self._block("lease_scope_changed", "Current request differs from the admitted route or task")
        if scope.get("api_call_count") != "1":
            return self._block(
                "capability_request_limit",
                "This fixed-JSON capability allows one model request; retries and continuations are refused",
            )
        try:
            task = self._read_task(context)
            if task_run_contract_sha256(task) != lease.admission_scope["task_contract_sha256"]:
                raise ValueError("task contract changed during run")
            _rb_root, state_dir, registry_path = self._configuration()
            if lease.manager is None:
                external_job_id = os.environ.get("RESOURCE_BALANCER_ADMISSION_JOB_ID", "").strip()
                if external_job_id != lease.job_id:
                    raise ValueError("pre-spawn Resource Balancer reservation identity changed")
                self._validate_external_reservation(state_dir, lease.job_id, scope)
            if _sha256(registry_path.read_bytes()) != lease.registry_hash:
                raise ValueError("capability registry changed during run")
            if self._live_model_evidence(scope) != lease.runtime_evidence:
                raise ValueError("model digest, quantization, or runtime version changed during run")
            self._resource_balancer_status(state_dir, lease.job_id)
        except Exception as exc:
            _LOGGER.exception("Local GPU admission lease validation refused")
            return self._block("lease_validation_refused", f"Lease validation refused: {_safe_error_detail(exc)}")
        return {
            "decision": "allow", "policy_id": "local-gpu-task-capability-v1",
            "reason": "Existing task/run GPU lease and current request were revalidated",
            "evidence_id": lease.evidence_id, "lease_ids": [lease.job_id], "scope": scope,
            "response_format": lease.response_format,
        }

    def _release(self, context: dict[str, Any], lease_ids: tuple[str, ...]) -> dict[str, Any]:
        scope = {key: str(value or "") for key, value in context.items()
                 if key not in {"telemetry_schema_version", "middleware_schema_version"}}
        key = self._key(scope)
        with self._lock:
            lease = self._leases.get(key)
            if lease is None or lease_ids != (lease.job_id,):
                return self._block("release_refused", "No matching task/run lease to release")
            del self._leases[key]
        try:
            if lease.manager is not None:
                lease.manager.__exit__(None, None, None)
        except Exception:
            _LOGGER.exception("Local GPU admission lease release failed")
            return self._block("release_failed", "Resource Balancer lease release failed")
        return {"decision": "released", "lease_ids": [lease.job_id], "scope": scope}

    def _verify_completion(self, context: dict[str, Any]) -> dict[str, Any]:
        scope = {field: context.get(field) for field in _VERIFY_FIELDS}
        key = (str(scope["kanban_task_id"] or ""), str(scope["task_run_id"] or ""),
               str(scope["kanban_claim_lock"] or ""))
        with self._lock:
            lease = self._leases.get(key)
        if lease is None:
            return self._block("completion_lease_missing", "No active lease for this task/run")
        expected_scope = {field: str(value or "") for field, value in scope.items() if field != "artifacts"}
        task_scope = lease.admission_scope
        for source, target in (
            ("task_id", "kanban_task_id"), ("kanban_task_id", "kanban_task_id"),
            ("task_run_id", "task_run_id"),
            ("kanban_claim_lock", "kanban_claim_lock"), ("kanban_board", "kanban_board"),
            ("kanban_db", "kanban_db"), ("kanban_workspace", "kanban_workspace"),
            ("capability_id", "capability_id"),
            ("task_contract_sha256", "task_contract_sha256"),
        ):
            if expected_scope.get(source) != task_scope.get(target):
                return self._block("completion_scope_mismatch", "Completion scope differs from the admitted task")
        try:
            _rb_root, state_dir, _registry_path = self._configuration()
            if lease.manager is None:
                external_job_id = os.environ.get("RESOURCE_BALANCER_ADMISSION_JOB_ID", "").strip()
                if external_job_id != lease.job_id:
                    raise ValueError("pre-spawn Resource Balancer reservation identity changed")
                self._validate_external_reservation(state_dir, lease.job_id, task_scope)
            task = self._read_task({**task_scope, "task_id": key[0], "task_run_id": key[1],
                                    "kanban_claim_lock": key[2]})
            if (task.workspace_path != scope["workspace_path"]
                    or task.assignee != scope["assignee"]
                    or (task.model_override or "") != scope["model_override"]
                    or (task.provider_override or "") != scope["provider_override"]
                    or (task.completion_contract or "") != scope["completion_contract"]):
                raise ValueError("task workspace or route changed before completion")
            artifacts = scope["artifacts"]
            if not isinstance(artifacts, (list, tuple)) or len(artifacts) != 1:
                raise ValueError("exactly one declared JSON artifact is required")
            workspace = Path(task.workspace_path or "").resolve(strict=True)
            artifact_arg = Path(str(artifacts[0])).expanduser()
            candidate = artifact_arg if artifact_arg.is_absolute() else workspace / artifact_arg
            artifact = candidate.resolve(strict=True)
            if not artifact.is_relative_to(workspace) or not artifact.is_file():
                raise ValueError("artifact must be a regular file inside the task workspace")
            if artifact.stat().st_size > _MAX_ARTIFACT_BYTES:
                raise ValueError("artifact exceeds the fixed JSON task size limit")
            data = artifact.read_bytes()
            actual = _json_object(data)
            if actual != lease.expected:
                raise ValueError("artifact differs from the pinned expected JSON object")
            _rb_root, state_dir, registry_path = self._configuration()
            if _sha256(registry_path.read_bytes()) != lease.registry_hash:
                raise ValueError("capability registry changed before completion")
            if self._live_model_evidence(task_scope) != lease.runtime_evidence:
                raise ValueError("model runtime evidence changed before completion")
            if _sha256(lease.verifier_path.read_bytes()) != lease.verifier_hash:
                raise ValueError("independent verifier changed before completion")
            _run_pinned_verifier(lease.verifier_path, artifact)
            self._resource_balancer_status(state_dir, lease.job_id)
        except Exception as exc:
            _LOGGER.exception("Local GPU admission completion verification failed")
            return self._block(
                "completion_verification_failed",
                f"Independent verification failed: {_safe_error_detail(exc)}",
            )
        artifact_hash = _sha256(data)
        evidence_id = _sha256(f"{lease.evidence_id}:{artifact_hash}".encode())
        return {
            "decision": "verified", "policy_id": "local-gpu-json-oracle-v1",
            "reason": "Artifact matches the registry-pinned exact JSON oracle",
            "evidence_id": evidence_id, "verifier_sha256": lease.verifier_hash,
            "artifact_sha256": artifact_hash,
            "scope": {**expected_scope, "artifacts": [str(artifacts[0])]},
        }


def register(ctx):
    config = {
        key: ctx.get_config(key, "")
        for key in (
            "resource_balancer_python_path", "resource_balancer_state_dir",
            "capability_registry_path", "capability_registry_sha256",
        )
    }
    controller = ResourceBalancerController(profile=getattr(ctx, "profile_name", ""), config=config)
    ctx.register_middleware("llm_admission", controller)
