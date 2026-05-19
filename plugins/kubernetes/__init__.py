"""Read-only Kubernetes diagnostics plugin.

The plugin intentionally shells out to ``kubectl`` instead of depending on the
Kubernetes Python client. That keeps install weight low and reuses the user's
existing kubeconfig, auth plugins, and context setup.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Any


_KUBECTL_TIMEOUT = 30
_MAX_TEXT_CHARS = 24_000
_MAX_ITEMS = 200
_NAME_RE = re.compile(r"^[A-Za-z0-9_.:/=-]+$")
_SELECTOR_RE = re.compile(r"^[A-Za-z0-9_.:/=,!() -]+$")
_NAMESPACE_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")

_RESOURCE_ALIASES = {
    "pod": "pods",
    "pods": "pods",
    "po": "pods",
    "deployment": "deployments",
    "deployments": "deployments",
    "deploy": "deployments",
    "service": "services",
    "services": "services",
    "svc": "services",
    "ingress": "ingresses",
    "ingresses": "ingresses",
    "ing": "ingresses",
    "node": "nodes",
    "nodes": "nodes",
    "namespace": "namespaces",
    "namespaces": "namespaces",
    "ns": "namespaces",
    "event": "events",
    "events": "events",
    "job": "jobs",
    "jobs": "jobs",
    "cronjob": "cronjobs",
    "cronjobs": "cronjobs",
    "statefulset": "statefulsets",
    "statefulsets": "statefulsets",
    "sts": "statefulsets",
    "daemonset": "daemonsets",
    "daemonsets": "daemonsets",
    "ds": "daemonsets",
    "replicaset": "replicasets",
    "replicasets": "replicasets",
    "rs": "replicasets",
    "persistentvolumeclaim": "persistentvolumeclaims",
    "persistentvolumeclaims": "persistentvolumeclaims",
    "pvc": "persistentvolumeclaims",
    "persistentvolume": "persistentvolumes",
    "persistentvolumes": "persistentvolumes",
    "pv": "persistentvolumes",
}

_CLUSTER_SCOPED = {"nodes", "namespaces", "persistentvolumes"}
_DENIED_RESOURCES = {"secret", "secrets"}

_SECRET_PATTERNS = [
    re.compile(r"(?i)(token|password|passwd|secret|api[_-]?key)(\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(authorization:\s*bearer\s+)([A-Za-z0-9._~+/=-]+)"),
]


def _kubectl_binary() -> str | None:
    override = os.environ.get("KUBECTL")
    if override:
        return override if shutil.which(override) or os.path.exists(override) else None
    return shutil.which("kubectl")


def _check_kubectl_available() -> bool:
    kubectl = _kubectl_binary()
    if not kubectl:
        return False
    try:
        proc = subprocess.run(
            [kubectl, "version", "--client"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _error(message: str, **extra: Any) -> str:
    payload = {"ok": False, "error": message}
    payload.update(extra)
    return _json_response(payload)


def _redact(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 3:
            redacted = pattern.sub(r"\1\2[REDACTED]", redacted)
        else:
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
    return redacted


def _truncate(text: str, limit: int = _MAX_TEXT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...[truncated]...", True


def _validate_token(value: str | None, label: str, pattern: re.Pattern[str] = _NAME_RE) -> str | None:
    if value is None or value == "":
        return None
    value = str(value)
    if not pattern.match(value):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _resource_type(value: str) -> str:
    raw = (value or "").strip().lower()
    if raw in _DENIED_RESOURCES:
        raise ValueError("Reading Kubernetes secrets is not allowed by this read-only tool")
    resolved = _RESOURCE_ALIASES.get(raw)
    if not resolved:
        allowed = ", ".join(sorted(set(_RESOURCE_ALIASES.values())))
        raise ValueError(f"Unsupported resource_type {value!r}. Allowed: {allowed}")
    return resolved


def _base_cmd(context: str | None = None) -> list[str]:
    kubectl = _kubectl_binary()
    if not kubectl:
        raise RuntimeError("kubectl was not found. Install kubectl or set KUBECTL.")
    cmd = [kubectl]
    context = _validate_token(context, "context")
    if context:
        cmd.extend(["--context", context])
    return cmd


def _run_kubectl(args: list[str], timeout: int = _KUBECTL_TIMEOUT) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"kubectl timed out after {timeout}s: {' '.join(args[:4])}") from exc


def _load_json(cmd: list[str]) -> dict[str, Any]:
    proc = _run_kubectl(cmd)
    if proc.returncode != 0:
        stderr = _redact((proc.stderr or proc.stdout or "").strip())
        raise RuntimeError(stderr or f"kubectl exited with status {proc.returncode}")
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"kubectl returned invalid JSON: {exc}") from exc


def _meta(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    return {
        "name": metadata.get("name"),
        "namespace": metadata.get("namespace"),
        "created": metadata.get("creationTimestamp"),
        "labels": metadata.get("labels") or {},
    }


def _condition_status(status: dict[str, Any], condition_type: str) -> str | None:
    for condition in status.get("conditions") or []:
        if condition.get("type") == condition_type:
            return condition.get("status")
    return None


def _summarize_pod(item: dict[str, Any]) -> dict[str, Any]:
    status = item.get("status") or {}
    spec = item.get("spec") or {}
    containers = []
    restarts = 0
    for cs in status.get("containerStatuses") or []:
        restarts += int(cs.get("restartCount") or 0)
        containers.append({
            "name": cs.get("name"),
            "ready": cs.get("ready"),
            "restartCount": cs.get("restartCount"),
            "image": cs.get("image"),
            "state": cs.get("state"),
        })
    return {
        **_meta(item),
        "phase": status.get("phase"),
        "ready": _condition_status(status, "Ready"),
        "node": spec.get("nodeName"),
        "podIP": status.get("podIP"),
        "restarts": restarts,
        "containers": containers,
    }


def _summarize_workload(item: dict[str, Any]) -> dict[str, Any]:
    status = item.get("status") or {}
    spec = item.get("spec") or {}
    return {
        **_meta(item),
        "replicas": spec.get("replicas"),
        "readyReplicas": status.get("readyReplicas"),
        "availableReplicas": status.get("availableReplicas"),
        "updatedReplicas": status.get("updatedReplicas"),
        "selector": (spec.get("selector") or {}).get("matchLabels") or {},
    }


def _summarize_service(item: dict[str, Any]) -> dict[str, Any]:
    spec = item.get("spec") or {}
    return {
        **_meta(item),
        "type": spec.get("type"),
        "clusterIP": spec.get("clusterIP"),
        "externalIPs": spec.get("externalIPs") or [],
        "ports": spec.get("ports") or [],
        "selector": spec.get("selector") or {},
    }


def _summarize_ingress(item: dict[str, Any]) -> dict[str, Any]:
    spec = item.get("spec") or {}
    rules = []
    for rule in spec.get("rules") or []:
        rules.append({
            "host": rule.get("host"),
            "paths": [
                (path.get("path") or "/")
                for path in ((rule.get("http") or {}).get("paths") or [])
            ],
        })
    return {
        **_meta(item),
        "className": spec.get("ingressClassName"),
        "rules": rules,
        "loadBalancer": (item.get("status") or {}).get("loadBalancer") or {},
    }


def _summarize_node(item: dict[str, Any]) -> dict[str, Any]:
    status = item.get("status") or {}
    return {
        **_meta(item),
        "ready": _condition_status(status, "Ready"),
        "addresses": status.get("addresses") or [],
        "capacity": status.get("capacity") or {},
        "allocatable": status.get("allocatable") or {},
    }


def _summarize_event(item: dict[str, Any]) -> dict[str, Any]:
    involved = item.get("involvedObject") or item.get("regarding") or {}
    return {
        **_meta(item),
        "type": item.get("type"),
        "reason": item.get("reason"),
        "message": item.get("message"),
        "count": item.get("count") or item.get("series", {}).get("count"),
        "firstTimestamp": item.get("firstTimestamp") or item.get("eventTime"),
        "lastTimestamp": item.get("lastTimestamp") or item.get("eventTime"),
        "object": {
            "kind": involved.get("kind"),
            "name": involved.get("name"),
            "namespace": involved.get("namespace"),
        },
    }


def _summarize_generic(item: dict[str, Any]) -> dict[str, Any]:
    return {
        **_meta(item),
        "kind": item.get("kind"),
        "status": item.get("status") or {},
    }


_SUMMARY_BY_RESOURCE = {
    "pods": _summarize_pod,
    "deployments": _summarize_workload,
    "statefulsets": _summarize_workload,
    "daemonsets": _summarize_workload,
    "replicasets": _summarize_workload,
    "jobs": _summarize_workload,
    "cronjobs": _summarize_workload,
    "services": _summarize_service,
    "ingresses": _summarize_ingress,
    "nodes": _summarize_node,
    "events": _summarize_event,
}


def _summarize(resource: str, data: dict[str, Any]) -> Any:
    summarizer = _SUMMARY_BY_RESOURCE.get(resource, _summarize_generic)
    if "items" in data:
        return [summarizer(item) for item in (data.get("items") or [])[:_MAX_ITEMS]]
    return summarizer(data)


def _namespace_args(resource: str, namespace: str | None, all_namespaces: bool) -> list[str]:
    if resource in _CLUSTER_SCOPED:
        return []
    if all_namespaces:
        return ["--all-namespaces"]
    namespace = _validate_token(namespace, "namespace", _NAMESPACE_RE)
    return ["-n", namespace] if namespace else []


def _handle_contexts(args: dict, **kwargs) -> str:
    try:
        cmd = _base_cmd()
        cmd.extend(["config", "get-contexts", "-o", "name"])
        proc = _run_kubectl(cmd)
        if proc.returncode != 0:
            return _error(_redact((proc.stderr or proc.stdout or "").strip()))

        current_proc = _run_kubectl(_base_cmd() + ["config", "current-context"])
        current = current_proc.stdout.strip() if current_proc.returncode == 0 else None
        contexts = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        return _json_response({"ok": True, "current": current, "contexts": contexts})
    except Exception as exc:
        return _error(str(exc))


def _handle_get(args: dict, **kwargs) -> str:
    try:
        resource = _resource_type(args.get("resource_type", "pods"))
        name = _validate_token(args.get("name"), "name")
        namespace = args.get("namespace")
        all_namespaces = bool(args.get("all_namespaces", False))
        selector = _validate_token(args.get("selector"), "selector", _SELECTOR_RE)
        field_selector = _validate_token(args.get("field_selector"), "field_selector", _SELECTOR_RE)
        context = args.get("context")
        raw = bool(args.get("raw", False))

        cmd = _base_cmd(context) + ["get", resource]
        if name:
            cmd.append(name)
        cmd.extend(_namespace_args(resource, namespace, all_namespaces))
        if selector:
            cmd.extend(["-l", selector])
        if field_selector:
            cmd.extend(["--field-selector", field_selector])
        cmd.extend(["-o", "json"])

        data = _load_json(cmd)
        payload = {
            "ok": True,
            "resource_type": resource,
            "count": len(data.get("items") or []) if "items" in data else 1,
            "result": data if raw else _summarize(resource, data),
        }
        if "items" in data and len(data.get("items") or []) > _MAX_ITEMS and not raw:
            payload["truncated"] = True
            payload["limit"] = _MAX_ITEMS
        return _json_response(payload)
    except Exception as exc:
        return _error(str(exc))


def _handle_describe(args: dict, **kwargs) -> str:
    try:
        resource = _resource_type(args.get("resource_type", "pods"))
        if resource in {"events", "namespaces"}:
            return _error(f"describe is not supported for {resource}; use k8s_get instead")
        name = _validate_token(args.get("name"), "name")
        if not name:
            return _error("name is required")
        namespace = args.get("namespace")
        context = args.get("context")

        cmd = _base_cmd(context) + ["describe", resource, name]
        cmd.extend(_namespace_args(resource, namespace, False))
        proc = _run_kubectl(cmd)
        if proc.returncode != 0:
            return _error(_redact((proc.stderr or proc.stdout or "").strip()))
        text, truncated = _truncate(_redact(proc.stdout or ""))
        return _json_response({
            "ok": True,
            "resource_type": resource,
            "name": name,
            "namespace": namespace,
            "truncated": truncated,
            "output": text,
        })
    except Exception as exc:
        return _error(str(exc))


def _handle_logs(args: dict, **kwargs) -> str:
    try:
        pod = _validate_token(args.get("pod"), "pod")
        if not pod:
            return _error("pod is required")
        namespace = args.get("namespace")
        container = _validate_token(args.get("container"), "container")
        context = args.get("context")
        previous = bool(args.get("previous", False))
        tail = int(args.get("tail", 200))
        tail = max(1, min(tail, 2000))
        since = _validate_token(args.get("since"), "since")

        cmd = _base_cmd(context) + ["logs", pod, "--tail", str(tail)]
        ns_args = _namespace_args("pods", namespace, False)
        cmd.extend(ns_args)
        if container:
            cmd.extend(["-c", container])
        if previous:
            cmd.append("--previous")
        if since:
            cmd.extend(["--since", since])

        proc = _run_kubectl(cmd, timeout=45)
        if proc.returncode != 0:
            return _error(_redact((proc.stderr or proc.stdout or "").strip()))
        text, truncated = _truncate(_redact(proc.stdout or ""))
        return _json_response({
            "ok": True,
            "pod": pod,
            "namespace": namespace,
            "container": container,
            "previous": previous,
            "tail": tail,
            "truncated": truncated,
            "logs": text,
        })
    except Exception as exc:
        return _error(str(exc))


def _handle_events(args: dict, **kwargs) -> str:
    try:
        namespace = args.get("namespace")
        all_namespaces = bool(args.get("all_namespaces", False))
        context = args.get("context")
        limit = max(1, min(int(args.get("limit", 100)), 500))

        cmd = _base_cmd(context) + ["get", "events"]
        cmd.extend(_namespace_args("events", namespace, all_namespaces))
        cmd.extend(["-o", "json"])
        data = _load_json(cmd)
        events = [_summarize_event(item) for item in data.get("items") or []]
        events.sort(key=lambda e: e.get("lastTimestamp") or e.get("created") or "")
        events = events[-limit:]
        return _json_response({
            "ok": True,
            "count": len(events),
            "events": events,
        })
    except Exception as exc:
        return _error(str(exc))


K8S_CONTEXTS_SCHEMA = {
    "name": "k8s_contexts",
    "description": "List available Kubernetes kubeconfig contexts and the current context.",
    "parameters": {"type": "object", "properties": {}},
}

K8S_GET_SCHEMA = {
    "name": "k8s_get",
    "description": (
        "Read Kubernetes resources with kubectl and return a compact JSON summary. "
        "Read-only; Kubernetes secrets are intentionally blocked."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "resource_type": {
                "type": "string",
                "description": "Resource type: pods, deployments, services, ingresses, nodes, namespaces, events, jobs, cronjobs, statefulsets, daemonsets, replicasets, pvc, pv.",
            },
            "name": {"type": "string", "description": "Optional resource name."},
            "namespace": {"type": "string", "description": "Namespace for namespaced resources."},
            "all_namespaces": {"type": "boolean", "description": "List namespaced resources across all namespaces."},
            "selector": {"type": "string", "description": "Label selector, for example app=api,tier=backend."},
            "field_selector": {"type": "string", "description": "Field selector, for example status.phase=Running."},
            "context": {"type": "string", "description": "Optional kubeconfig context."},
            "raw": {"type": "boolean", "description": "Return raw kubectl JSON instead of compact summary."},
        },
        "required": ["resource_type"],
    },
}

K8S_DESCRIBE_SCHEMA = {
    "name": "k8s_describe",
    "description": "Run kubectl describe for a single non-secret Kubernetes resource and return redacted text.",
    "parameters": {
        "type": "object",
        "properties": {
            "resource_type": {"type": "string", "description": "Resource type such as pods, deployments, services, ingresses, nodes, jobs, pvc, or pv."},
            "name": {"type": "string", "description": "Resource name."},
            "namespace": {"type": "string", "description": "Namespace for namespaced resources."},
            "context": {"type": "string", "description": "Optional kubeconfig context."},
        },
        "required": ["resource_type", "name"],
    },
}

K8S_LOGS_SCHEMA = {
    "name": "k8s_logs",
    "description": "Fetch recent logs for a Kubernetes pod. Read-only and size-limited.",
    "parameters": {
        "type": "object",
        "properties": {
            "pod": {"type": "string", "description": "Pod name."},
            "namespace": {"type": "string", "description": "Pod namespace."},
            "container": {"type": "string", "description": "Optional container name."},
            "previous": {"type": "boolean", "description": "Read logs from the previous terminated container instance."},
            "tail": {"type": "integer", "description": "Number of log lines to return, clamped to 1..2000."},
            "since": {"type": "string", "description": "Optional duration such as 10m, 1h, or 30s."},
            "context": {"type": "string", "description": "Optional kubeconfig context."},
        },
        "required": ["pod"],
    },
}

K8S_EVENTS_SCHEMA = {
    "name": "k8s_events",
    "description": "List recent Kubernetes events as compact JSON.",
    "parameters": {
        "type": "object",
        "properties": {
            "namespace": {"type": "string", "description": "Namespace to inspect."},
            "all_namespaces": {"type": "boolean", "description": "List events across all namespaces."},
            "limit": {"type": "integer", "description": "Maximum events to return, clamped to 1..500."},
            "context": {"type": "string", "description": "Optional kubeconfig context."},
        },
    },
}


def register(ctx) -> None:
    tools = (
        ("k8s_contexts", K8S_CONTEXTS_SCHEMA, _handle_contexts),
        ("k8s_get", K8S_GET_SCHEMA, _handle_get),
        ("k8s_describe", K8S_DESCRIBE_SCHEMA, _handle_describe),
        ("k8s_logs", K8S_LOGS_SCHEMA, _handle_logs),
        ("k8s_events", K8S_EVENTS_SCHEMA, _handle_events),
    )
    for name, schema, handler in tools:
        ctx.register_tool(
            name=name,
            toolset="kubernetes",
            schema=schema,
            handler=handler,
            description=schema["description"],
            emoji="☸",
        )
