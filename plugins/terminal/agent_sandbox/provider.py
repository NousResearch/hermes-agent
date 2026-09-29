"""Kubernetes Agent Sandbox terminal backend.

The backend owns only the task Sandbox lifecycle. It uses a configured kubectl
binary with fixed argv and a namespace-scoped operator identity. The task Pod
never receives a Kubernetes token.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import posixpath
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from agent.terminal_env_provider import TerminalEnvironmentProvider
from tools.environments.base import BaseEnvironment, EnvironmentConnectionError
from tools.environments.base_output import _BoundedOutputCollector, _pipe_stdin

logger = logging.getLogger(__name__)

_BACKEND_NAME = "agent_sandbox"
_NAMESPACE = "agent-sandbox-tasks"
_SANDBOX_API = "agents.x-k8s.io/v1beta1"
_SANDBOX_KIND = "Sandbox"
_CONTAINER_NAME = "task"
_TASK_LABEL = "agent-sandbox.rbtr.dev/task-id"
_ROLE_LABEL = "agent-sandbox.rbtr.dev/role"
_ROLE_VALUE = "coding-task"
_FULL_TASK_ANNOTATION = "agent-sandbox.rbtr.dev/full-task-id"
_DEFAULT_DEADLINE = 600
_DEFAULT_CREATE_TIMEOUT = 30
_DEFAULT_READY_TIMEOUT = 120
_DEFAULT_CLEANUP_TIMEOUT = 60
_MAX_COMMAND_TIMEOUT = 600
_DEFAULT_MAX_OUTPUT_BYTES = 1 << 20
_MAX_OUTPUT_BYTES_CEILING = 8 << 20
_TASK_ID_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
_LABEL_RE = re.compile(r"[^a-z0-9-]+")
_DIGEST_RE = re.compile(r"^(?:[^@\s]+)@sha256:[0-9a-f]{64}$")
_SECRET_RE = re.compile(r"(?i)(bearer\s+|token\s*[=:]\s*|password\s*[=:]\s*)[^\s]+")


class AgentSandboxError(RuntimeError):
    """A bounded Agent Sandbox lifecycle failure."""

    def __init__(self, step: str, message: str):
        self.step = step
        super().__init__(f"Agent Sandbox {step} failed: {message}")


def _redact(text: str) -> str:
    return _SECRET_RE.sub(lambda match: f"{match.group(1)}[redacted]", text)[:400]


def _safe_task_label(task_id: str) -> str:
    """Return a deterministic Kubernetes label value for a terminal task key."""
    raw = str(task_id or "").strip()
    if not _TASK_ID_RE.fullmatch(raw):
        raise ValueError("task_id must be 1-256 characters with no control characters")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    cleaned = _LABEL_RE.sub("-", raw.lower()).strip("-") or "task"
    return f"{cleaned[:52].rstrip('-')}-{digest}"[:63].rstrip("-")


def _sandbox_name(task_id: str) -> str:
    digest = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()[:20]
    return f"hermes-task-{digest}"


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 1 or parsed > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return parsed


def _config_values(raw: Any) -> dict[str, Any]:
    values = raw if isinstance(raw, dict) else {}
    kubectl = str(values.get("kubectl_path", "")).strip()
    namespace = str(values.get("namespace", _NAMESPACE)).strip()
    image = str(values.get("image", "")).strip()
    if not kubectl or not os.path.isabs(kubectl):
        raise ValueError("kubectl_path must be an absolute path")
    if namespace != _NAMESPACE:
        raise ValueError(f"namespace must be {_NAMESPACE}")
    if not _DIGEST_RE.fullmatch(image):
        raise ValueError("image must use an immutable @sha256 digest")
    return {
        "kubectl_path": kubectl,
        "namespace": namespace,
        "image": image,
        "deadline": _positive_int(values.get("deadline", _DEFAULT_DEADLINE), "deadline", maximum=_DEFAULT_DEADLINE),
        "create_timeout": _positive_int(values.get("create_timeout", _DEFAULT_CREATE_TIMEOUT), "create_timeout", maximum=600),
        "ready_timeout": _positive_int(values.get("ready_timeout", _DEFAULT_READY_TIMEOUT), "ready_timeout", maximum=600),
        "cleanup_timeout": _positive_int(values.get("cleanup_timeout", _DEFAULT_CLEANUP_TIMEOUT), "cleanup_timeout", maximum=600),
        "command_timeout": _positive_int(values.get("command_timeout", _DEFAULT_DEADLINE), "command_timeout", maximum=_MAX_COMMAND_TIMEOUT),
        "max_output_bytes": _positive_int(
            values.get("max_output_bytes", _DEFAULT_MAX_OUTPUT_BYTES),
            "max_output_bytes", maximum=_MAX_OUTPUT_BYTES_CEILING),
    }


def _condition_true(status: dict[str, Any], condition_type: str) -> bool:
    conditions = status.get("conditions", [])
    if not isinstance(conditions, list):
        return False
    return any(
        isinstance(condition, dict)
        and condition.get("type") == condition_type
        and condition.get("status") == "True"
        for condition in conditions
    )


class AgentSandboxEnvironment(BaseEnvironment):
    """Run the BaseEnvironment shell protocol inside one Agent Sandbox Pod."""

    _profile_scoped_passthrough = False
    is_local = False

    def __init__(self, config: dict[str, Any], task_id: str, cwd: str, timeout: int):
        self.config = config
        self.task_id = str(task_id or "default").strip()
        self.task_label = _safe_task_label(self.task_id)
        self.sandbox_name = _sandbox_name(self.task_id)
        self.kubectl = config["kubectl_path"]
        self.namespace = config["namespace"]
        self.container = _CONTAINER_NAME
        self._deleted = False
        self._owned = False
        self.sandbox_uid = None
        self.pod_uid = None
        self.host_cwd = None
        if not _TASK_ID_RE.fullmatch(self.task_id):
            raise ValueError("task_id must be 1-256 characters with no control characters")
        super().__init__(cwd="/workspace", timeout=min(timeout, config["command_timeout"]))
        try:
            self._ensure_sandbox()
            self._wait_ready()
            self._check_shell()
            self.init_session()
        except Exception as exc:
            try:
                self.cleanup()
            except Exception as cleanup_exc:
                exc.add_note(f"cleanup also failed: {cleanup_exc}")
            raise

    def _kubectl(self, args: list[str], *, stdin: Optional[str] = None, timeout: int) -> tuple[int, str]:
        command = [self.kubectl, *args]
        max_output = self.config.get("max_output_bytes", _DEFAULT_MAX_OUTPUT_BYTES)
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=(os.name != "nt"),
            )
        except FileNotFoundError as exc:
            raise AgentSandboxError("kubectl", "configured executable was not found") from exc
        except OSError as exc:
            raise AgentSandboxError("kubectl", "configured executable could not be started") from exc

        if stdin is not None and proc.stdin is not None:
            try:
                proc.stdin.write(stdin.encode("utf-8"))
                proc.stdin.close()
            except OSError:
                self._kill_process(proc)
                proc.wait()
                raise AgentSandboxError("kubectl", "could not write request input")

        streams = (proc.stdout, proc.stderr)
        buffers = [bytearray(), bytearray()]
        exceeded = threading.Event()

        def drain(stream, buffer: bytearray) -> None:
            if stream is None:
                return
            while not exceeded.is_set() and len(buffer) <= max_output:
                chunk = stream.read(min(65536, max_output + 1 - len(buffer)))
                if not chunk:
                    return
                buffer.extend(chunk)
                if len(buffer) > max_output:
                    exceeded.set()
                    self._kill_process(proc)
                    return

        readers = [threading.Thread(target=drain, args=(stream, buffer), daemon=True)
                   for stream, buffer in zip(streams, buffers)]
        for reader in readers:
            reader.start()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self._kill_process(proc)
            proc.wait()
            for reader in readers:
                reader.join(timeout=1)
            raise AgentSandboxError("kubectl", "operation timed out") from exc
        for reader in readers:
            reader.join(timeout=1)
        if exceeded.is_set():
            raise AgentSandboxError("kubectl", "response exceeded the output limit")
        stdout, stderr = (bytes(buffer) for buffer in buffers)
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            detail = _redact(stderr_text.strip()) or "kubectl returned a non-zero status"
            raise AgentSandboxError("kubectl", detail)
        return proc.returncode, stdout_text

    def _get_json(self, resource: str, name: str) -> dict[str, Any] | None:
        command = ["get", resource, name, "-n", self.namespace, "-o", "json"]
        try:
            _, output = self._kubectl(command, timeout=min(self.config["ready_timeout"], 30))
        except AgentSandboxError as exc:
            if "not found" in str(exc).lower() or "status=404" in str(exc).lower():
                return None
            raise
        try:
            value = json.loads(output)
        except json.JSONDecodeError as exc:
            raise AgentSandboxError("inspect", "kubectl returned invalid JSON") from exc
        return value if isinstance(value, dict) else None

    def _manifest(self) -> dict[str, Any]:
        labels = {_ROLE_LABEL: _ROLE_VALUE, _TASK_LABEL: self.task_label}
        return {
            "apiVersion": _SANDBOX_API,
            "kind": _SANDBOX_KIND,
            "metadata": {
                "name": self.sandbox_name,
                "namespace": self.namespace,
                "labels": labels,
                "annotations": {_FULL_TASK_ANNOTATION: self.task_id},
            },
            "spec": {
                "operatingMode": "Running",
                "shutdownPolicy": "Delete",
                "podTemplate": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "activeDeadlineSeconds": self.config["deadline"],
                        "automountServiceAccountToken": False,
                        "terminationGracePeriodSeconds": 10,
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 65532,
                            "runAsGroup": 65532,
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                        "containers": [{
                            "name": self.container,
                            "image": self.config["image"],
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["sh", "-ec"],
                            "args": ["exec sleep 600"],
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "capabilities": {"drop": ["ALL"]},
                            },
                            "resources": {
                                "requests": {"cpu": "10m", "memory": "16Mi", "ephemeral-storage": "64Mi"},
                                "limits": {"cpu": "500m", "memory": "256Mi", "ephemeral-storage": "512Mi"},
                            },
                            "volumeMounts": [
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "tmp", "mountPath": "/tmp"},
                            ],
                        }],
                        "volumes": [
                            {"name": "workspace", "emptyDir": {"sizeLimit": "256Mi"}},
                            {"name": "tmp", "emptyDir": {"sizeLimit": "16Mi"}},
                        ],
                    },
                },
            },
        }

    def _ensure_sandbox(self) -> None:
        existing = self._get_json("sandbox", self.sandbox_name)
        if existing is not None:
            raise AgentSandboxError("create", "Sandbox already exists; refusing to adopt it")
        try:
            _, output = self._kubectl(
                ["create", "-f", "-", "-n", self.namespace, "-o", "json"],
                stdin=json.dumps(self._manifest()), timeout=self.config["create_timeout"])
            self._owned = True
            try:
                existing = json.loads(output)
            except json.JSONDecodeError:
                existing = self._get_json("sandbox", self.sandbox_name)
        except AgentSandboxError as exc:
            raise AgentSandboxError("create", str(exc)) from exc
        if existing is None:
            raise AgentSandboxError("create", "Sandbox was not returned after creation")
        self._validate_sandbox(existing)
        metadata = existing.get("metadata")
        uid = metadata.get("uid") if isinstance(metadata, dict) else None
        if not isinstance(uid, str) or not uid:
            raise AgentSandboxError("create", "Sandbox has no UID")
        self.sandbox_uid = uid

    def _validate_sandbox(self, sandbox: dict[str, Any]) -> None:
        metadata = sandbox.get("metadata") if isinstance(sandbox.get("metadata"), dict) else {}
        if metadata.get("namespace") != self.namespace or metadata.get("name") != self.sandbox_name:
            raise AgentSandboxError("validate", "unexpected Sandbox identity")
        labels = metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}
        annotations = metadata.get("annotations") if isinstance(metadata.get("annotations"), dict) else {}
        if labels.get(_TASK_LABEL) != self.task_label or labels.get(_ROLE_LABEL) != _ROLE_VALUE:
            raise AgentSandboxError("validate", "unexpected Sandbox task labels")
        if annotations.get(_FULL_TASK_ANNOTATION) != self.task_id:
            raise AgentSandboxError("validate", "unexpected Sandbox task annotation")
        spec = sandbox.get("spec") if isinstance(sandbox.get("spec"), dict) else {}
        expected = self._manifest()["spec"]
        if spec.get("operatingMode") != expected["operatingMode"] or spec.get("shutdownPolicy") != expected["shutdownPolicy"]:
            raise AgentSandboxError("validate", "unexpected Sandbox lifecycle policy")
        pod_template = spec.get("podTemplate") if isinstance(spec.get("podTemplate"), dict) else {}
        pod_metadata = pod_template.get("metadata") if isinstance(pod_template.get("metadata"), dict) else {}
        if pod_metadata.get("labels") != expected["podTemplate"]["metadata"]["labels"]:
            raise AgentSandboxError("validate", "unexpected task Pod labels")
        pod_spec = pod_template.get("spec") if isinstance(pod_template.get("spec"), dict) else {}
        expected_pod_spec = expected["podTemplate"]["spec"]
        if any(pod_spec.get(key) is True for key in ("hostNetwork", "hostPID", "hostIPC", "shareProcessNamespace")):
            raise AgentSandboxError("validate", "host namespace is enabled")
        for key in ("automountServiceAccountToken", "activeDeadlineSeconds", "securityContext", "volumes"):
            if pod_spec.get(key) != expected_pod_spec[key]:
                raise AgentSandboxError("validate", f"unexpected Sandbox {key}")
        for key in ("initContainers", "ephemeralContainers"):
            if pod_spec.get(key):
                raise AgentSandboxError("validate", f"unexpected Sandbox {key}")
        volumes = pod_spec.get("volumes") if isinstance(pod_spec.get("volumes"), list) else []
        if any(isinstance(volume, dict) and volume.get("hostPath") is not None for volume in volumes):
            raise AgentSandboxError("validate", "host volume is mounted")
        containers = pod_spec.get("containers") if isinstance(pod_spec.get("containers"), list) else []
        expected_container = expected_pod_spec["containers"][0]
        if len(containers) != 1:
            raise AgentSandboxError("validate", "unexpected task container count")
        matching = [item for item in containers if isinstance(item, dict) and item.get("name") == self.container]
        if len(matching) != 1:
            raise AgentSandboxError("validate", "unexpected task container")
        container = matching[0]
        for key in ("image", "securityContext", "resources", "volumeMounts"):
            if container.get(key) != expected_container[key]:
                raise AgentSandboxError("validate", f"unexpected task container {key}")
        mounts = container.get("volumeMounts") if isinstance(container.get("volumeMounts"), list) else []
        if mounts != expected_container["volumeMounts"]:
            raise AgentSandboxError("validate", "unexpected task volume mounts")

    def _pod(self) -> dict[str, Any] | None:
        selector = f"{_ROLE_LABEL}={_ROLE_VALUE},{_TASK_LABEL}={self.task_label}"
        try:
            _, output = self._kubectl(
                ["get", "pods", "-n", self.namespace, "-l", selector, "-o", "json"],
                timeout=min(self.config["ready_timeout"], 30))
            data = json.loads(output)
        except json.JSONDecodeError as exc:
            raise AgentSandboxError("readiness", "kubectl returned invalid Pod JSON") from exc
        except AgentSandboxError:
            raise
        items = data.get("items", []) if isinstance(data, dict) else []
        if not isinstance(items, list):
            raise AgentSandboxError("readiness", "Pod list was invalid")
        matching = []
        for pod in items:
            if not isinstance(pod, dict):
                continue
            metadata = pod.get("metadata") if isinstance(pod.get("metadata"), dict) else {}
            labels = metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}
            owners = metadata.get("ownerReferences") if isinstance(metadata.get("ownerReferences"), list) else []
            owned_by_sandbox = any(
                isinstance(owner, dict)
                and owner.get("apiVersion") == _SANDBOX_API
                and owner.get("kind") == _SANDBOX_KIND
                and owner.get("name") == self.sandbox_name
                and owner.get("uid") == self.sandbox_uid
                for owner in owners
            )
            if (
                metadata.get("namespace") == self.namespace
                and labels.get(_TASK_LABEL) == self.task_label
                and labels.get(_ROLE_LABEL) == _ROLE_VALUE
                and owned_by_sandbox
            ):
                matching.append(pod)
        if len(matching) > 1:
            raise AgentSandboxError("readiness", "more than one task Pod matched")
        return matching[0] if matching else None

    def _configmaps(self) -> list[dict[str, Any]]:
        selector = f"{_ROLE_LABEL}={_ROLE_VALUE},{_TASK_LABEL}={self.task_label}"
        _, output = self._kubectl(
            ["get", "configmaps", "-n", self.namespace, "-l", selector, "-o", "json"],
            timeout=min(self.config["cleanup_timeout"], 30))
        try:
            data = json.loads(output)
        except json.JSONDecodeError as exc:
            raise AgentSandboxError("cleanup", "kubectl returned invalid ConfigMap JSON") from exc
        items = data.get("items", []) if isinstance(data, dict) else []
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.config["ready_timeout"]
        last_phase = "pending"
        while time.monotonic() < deadline:
            sandbox = self._get_json("sandbox", self.sandbox_name)
            if sandbox is None:
                raise AgentSandboxError("readiness", "Sandbox disappeared")
            self._validate_sandbox(sandbox)
            status = sandbox.get("status") if isinstance(sandbox.get("status"), dict) else {}
            pod = self._pod()
            if pod is not None:
                pod_status = pod.get("status") if isinstance(pod.get("status"), dict) else {}
                last_phase = str(pod_status.get("phase", last_phase))
                pod_ready = _condition_true(pod_status, "Ready")
                container_statuses = pod_status.get("containerStatuses", [])
                container_ready = any(
                    isinstance(item, dict) and item.get("name") == self.container and item.get("ready") is True
                    for item in container_statuses if isinstance(container_statuses, list)
                )
                if _condition_true(status, "Ready") and last_phase == "Running" and pod_ready and container_ready:
                    metadata = pod.get("metadata") if isinstance(pod.get("metadata"), dict) else {}
                    name = metadata.get("name")
                    uid = metadata.get("uid")
                    if not isinstance(name, str) or not name:
                        raise AgentSandboxError("readiness", "task Pod has no name")
                    if not isinstance(uid, str) or not uid:
                        raise AgentSandboxError("readiness", "task Pod has no UID")
                    self.pod_name = name
                    self.pod_uid = uid
                    return
            time.sleep(1)
        raise AgentSandboxError("readiness", f"timed out waiting for task Pod (phase={last_phase})")

    def _kill_process(self, proc: subprocess.Popen) -> None:
        if os.name != "nt":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        super()._kill_process(proc)

    def _force_kill_process(self, proc: subprocess.Popen) -> None:
        if os.name != "nt":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        super()._force_kill_process(proc)

    def _workspace_cwd(self, cwd: str) -> str:
        candidate = posixpath.normpath(str(cwd or ""))
        return candidate if candidate == "/workspace" or candidate.startswith("/workspace/") else "/workspace"

    def _new_output_collector(self, proc, bounded_capture: bool):
        if bounded_capture:
            return _BoundedOutputCollector(self.config["max_output_bytes"])
        return super()._new_output_collector(proc, bounded_capture)

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None,
                stdin_data: str | None = None, rewrite_compound_background: bool = True,
                bounded_capture: bool = False, yield_handler=None) -> dict:
        effective_timeout = min(timeout or self.timeout, self.config["command_timeout"])
        result = super().execute(
            command,
            cwd=self._workspace_cwd(cwd),
            timeout=effective_timeout,
            stdin_data=stdin_data,
            rewrite_compound_background=rewrite_compound_background,
            bounded_capture=bounded_capture,
            yield_handler=yield_handler,
        )
        return result

    def _check_shell(self) -> None:
        """Verify the configured image provides the bash protocol BaseEnvironment uses."""
        argv = self._exec_argv("command -v bash >/dev/null", login=False)
        try:
            completed = subprocess.run(
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AgentSandboxError("shell", "could not verify bash in the task image") from exc
        if completed.returncode != 0:
            raise AgentSandboxError("shell", "configured coding image must contain bash")

    def _validate_pod_identity(self) -> None:
        pod = self._pod()
        if pod is None:
            raise AgentSandboxError("exec", "task Pod is no longer present")
        metadata = pod.get("metadata") if isinstance(pod.get("metadata"), dict) else {}
        labels = metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}
        if (
            metadata.get("namespace") != self.namespace
            or metadata.get("name") != self.pod_name
            or metadata.get("uid") != self.pod_uid
        ):
            raise AgentSandboxError("exec", "unexpected task Pod identity")
        if labels.get(_TASK_LABEL) != self.task_label or labels.get(_ROLE_LABEL) != _ROLE_VALUE:
            raise AgentSandboxError("exec", "unexpected task Pod labels")
        owners = metadata.get("ownerReferences") if isinstance(metadata.get("ownerReferences"), list) else []
        if not any(
            isinstance(owner, dict)
            and owner.get("apiVersion") == _SANDBOX_API
            and owner.get("kind") == _SANDBOX_KIND
            and owner.get("name") == self.sandbox_name
            and owner.get("uid") == self.sandbox_uid
            for owner in owners
        ):
            raise AgentSandboxError("exec", "task Pod is not owned by this Sandbox")
        spec = pod.get("spec") if isinstance(pod.get("spec"), dict) else {}
        if any(spec.get(key) is True for key in ("hostNetwork", "hostPID", "hostIPC", "shareProcessNamespace")):
            raise AgentSandboxError("exec", "host namespace is enabled")
        if spec.get("automountServiceAccountToken") is not False:
            raise AgentSandboxError("exec", "ServiceAccount token mounting is enabled")
        for key in ("initContainers", "ephemeralContainers"):
            if spec.get(key):
                raise AgentSandboxError("exec", f"unexpected task {key}")
        expected_spec = self._manifest()["spec"]["podTemplate"]["spec"]
        expected_pod_security = expected_spec["securityContext"]
        pod_security = spec.get("securityContext")
        if pod_security != expected_pod_security:
            raise AgentSandboxError("exec", "unexpected task Pod security context")
        volumes = spec.get("volumes") if isinstance(spec.get("volumes"), list) else []
        expected_volumes = expected_spec["volumes"]
        if volumes != expected_volumes:
            raise AgentSandboxError("exec", "unexpected task volumes")
        containers = spec.get("containers") if isinstance(spec.get("containers"), list) else []
        if len(containers) != 1:
            raise AgentSandboxError("exec", "unexpected task container count")
        container = containers[0] if containers else {}
        if not isinstance(container, dict) or container.get("name") != self.container:
            raise AgentSandboxError("exec", "task container is not present")
        expected_container = expected_spec["containers"][0]
        security_context = container.get("securityContext")
        expected_security = expected_container["securityContext"]
        if security_context != expected_security:
            raise AgentSandboxError("exec", "unsafe task container security context")
        if container.get("image") != expected_container["image"]:
            raise AgentSandboxError("exec", "task image does not match configuration")
        for key in ("imagePullPolicy", "command", "args", "resources", "volumeMounts"):
            if container.get(key) != expected_container[key]:
                raise AgentSandboxError("exec", f"unexpected task container {key}")
        if container.get("env") or container.get("envFrom"):
            raise AgentSandboxError("exec", "unexpected task container environment")

    def _exec_argv(self, cmd_string: str, *, login: bool) -> list[str]:
        if not isinstance(getattr(self, "pod_name", None), str):
            raise AgentSandboxError("exec", "task Pod is not ready")
        self._validate_pod_identity()
        return [
            self.kubectl, "exec", self.pod_name, "-n", self.namespace, "-c", self.container,
            "--", "bash", *( ["-l"] if login else [] ), "-c", cmd_string,
        ]

    def _run_bash(self, cmd_string: str, *, login: bool = False, timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        try:
            proc = subprocess.Popen(
                self._exec_argv(cmd_string, login=login),
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if stdin_data is not None else subprocess.DEVNULL,
                start_new_session=(os.name != "nt"),
            )
        except OSError as exc:
            raise EnvironmentConnectionError("kubectl exec could not start", retry_hint="Check kubectl and the task Pod") from exc
        if stdin_data is not None:
            _pipe_stdin(proc, stdin_data)
        return proc

    def cleanup(self):
        if self._deleted or not self._owned:
            return
        self._deleted = True
        try:
            self._kubectl(
                ["delete", "sandbox", self.sandbox_name, "-n", self.namespace,
                 "--ignore-not-found=true", "--wait=false"],
                timeout=30)
            deadline = time.monotonic() + self.config["cleanup_timeout"]
            while time.monotonic() < deadline:
                sandbox = self._get_json("sandbox", self.sandbox_name)
                pod = self._pod()
                configmaps = self._configmaps()
                if sandbox is None and pod is None and not configmaps:
                    return
                time.sleep(1)
            raise AgentSandboxError("cleanup", "Sandbox or task Pod is still present")
        except AgentSandboxError as exc:
            logger.warning("Agent Sandbox cleanup failed for %s: %s", self.task_label, exc)
            self._deleted = False
            raise


class AgentSandboxProvider(TerminalEnvironmentProvider):
    """Hermes terminal provider for one disposable Agent Sandbox task."""

    is_remote = True
    is_container = True
    session_isolated_when_nonpersistent = True

    def __init__(self, context: Any):
        self._context = context

    @property
    def name(self) -> str:
        return _BACKEND_NAME

    @property
    def display_name(self) -> str:
        return "Agent Sandbox"

    @property
    def cache_path_base(self) -> Optional[str]:
        return "/workspace/.hermes"

    def _settings(self) -> dict[str, Any]:
        return _config_values(self._context.get_config("backend", {}))

    def is_available(self) -> bool:
        try:
            config = self._settings()
        except ValueError:
            return False
        return Path(config["kubectl_path"]).is_file() and os.access(config["kubectl_path"], os.X_OK)

    def check_requirements(self, config: Dict[str, Any]) -> bool:
        try:
            values = self._settings()
        except ValueError as exc:
            logger.error("Agent Sandbox configuration is invalid: %s", exc)
            return False
        try:
            completed = subprocess.run(
                [values["kubectl_path"], "version", "--client=true", "-o", "json"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        if completed.returncode != 0:
            return False
        permissions = (
            ("create", "sandboxes.agents.x-k8s.io"),
            ("get", "sandboxes.agents.x-k8s.io"),
            ("delete", "sandboxes.agents.x-k8s.io"),
            ("get", "pods"),
            ("list", "pods"),
            ("create", "pods/exec"),
            ("list", "configmaps"),
        )
        for verb, resource in permissions:
            try:
                probe = subprocess.run(
                    [values["kubectl_path"], "auth", "can-i", verb, resource, "-n", values["namespace"]],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return False
            if probe.returncode != 0 or probe.stdout.strip().lower() != "yes":
                return False
        return True

    def setup_instructions(self) -> list[str]:
        return [
            "Set plugins.entries.terminal/agent_sandbox.settings.backend.kubectl_path to an absolute kubectl path.",
            "Set an immutable coding image digest that contains bash and use the reviewed agent-sandbox-tasks namespace.",
            "Provision only namespace-scoped Agent Sandbox permissions for the adapter identity.",
        ]

    def create_environment(self, *, cwd: str, timeout: int, task_id: str = "default", image: Optional[str] = None,
                           container_config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        del container_config, kwargs
        config = self._settings()
        if image and image != config["image"]:
            raise ValueError("task image overrides are not allowed")
        effective_timeout = min(_positive_int(timeout, "timeout", maximum=_MAX_COMMAND_TIMEOUT), config["command_timeout"])
        return AgentSandboxEnvironment(config, task_id=task_id, cwd=cwd, timeout=effective_timeout)
