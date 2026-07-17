"""Fail-closed runtime admission policy for Hermes tool execution.

The policy is deliberately small and dependency-free: the canonical registry is
JSON, is loaded once when an agent is constructed, and the resulting contract is
immutable.  Tool executors call :func:`authorize_tool_call` immediately before
dispatch so registry-only validation cannot be bypassed.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


class AdmissionPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    reason: str
    approval_class: str = "none"


@dataclass(frozen=True)
class RuntimeAgentPolicy:
    canonical_id: str
    version: str
    policy_hash: str
    contract: Mapping[str, Any]


_PATH_KEYS = {
    "path", "file", "file_path", "source", "destination", "dest",
    "cwd", "workdir", "directory", "root", "output_path", "target_path",
}
_SENSITIVE_PATH_PARTS = {
    ".env", "credentials", "secrets", "cookies", "sessions",
}
_SENSITIVE_PATH_NAMES = {
    "agent-policies.json", "nous_auth.json", "known_hosts", "id_rsa",
    "id_ed25519",
}
_HOST_KEYS = {"host", "hostname", "remote_host", "machine"}
_DELEGATION_TOOLS = {"delegate_task", "delegate", "spawn_agent", "subagent"}
_EXTERNAL_TOOLS = {
    "send_email", "send_message", "publish", "purchase", "place_order",
    "financial_transaction", "post_publicly", "submit_asset",
}


def _frozen(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(k): _frozen(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(v) for v in value)
    return value


def load_runtime_policy(path: str | os.PathLike[str], identity: str) -> RuntimeAgentPolicy:
    try:
        raw = Path(path).read_bytes()
        document = json.loads(raw)
    except Exception as exc:
        raise AdmissionPolicyError(f"agent policy unavailable or malformed: {exc}") from exc
    agents = document.get("agents")
    if not isinstance(agents, list):
        raise AdmissionPolicyError("agent policy must contain an agents list")
    ids = [str(row.get("canonical_id", "")).strip().lower() for row in agents if isinstance(row, dict)]
    if not ids or any(not item for item in ids) or len(set(ids)) != len(ids):
        raise AdmissionPolicyError("agent policy contains missing or duplicate canonical IDs")
    contracts = {str(row["canonical_id"]).lower(): row for row in agents}
    for cid, row in contracts.items():
        if cid != "saint" and not str(row.get("supervisor", "")).strip():
            raise AdmissionPolicyError(f"agent '{cid}' has no supervisor")
    canonical = str(identity or "").strip().lower()
    if canonical not in contracts:
        raise AdmissionPolicyError(f"unknown agent identity '{canonical or '<empty>'}'")
    return RuntimeAgentPolicy(
        canonical_id=canonical,
        version=str(document.get("version", "unversioned")),
        policy_hash=hashlib.sha256(raw).hexdigest(),
        contract=_frozen(contracts[canonical]),
    )


def attach_runtime_policy(agent: Any, config: Mapping[str, Any]) -> None:
    """Attach the immutable policy selected by the active profile.

    Admission remains opt-in for upstream installations.  Once ``enabled`` is
    true it is fail-closed: missing paths, malformed registries, and unknown
    profiles abort agent construction.
    """
    admission = config.get("admission", {}) if isinstance(config, Mapping) else {}
    enabled = bool(admission.get("enabled", False)) if isinstance(admission, Mapping) else False
    agent._admission_required = enabled
    agent._admission_policy = None
    if not enabled:
        return
    path = str(admission.get("policy_path") or os.environ.get("HERMES_AGENT_POLICY", "")).strip()
    if not path:
        raise AdmissionPolicyError("admission is enabled but policy_path is missing")
    identity = str(admission.get("identity") or "").strip()
    if not identity:
        try:
            from hermes_cli.profiles import get_active_profile_name
            identity = get_active_profile_name() or ""
        except Exception:
            identity = ""
    aliases = admission.get("identity_aliases", {})
    if isinstance(aliases, Mapping):
        identity = str(aliases.get(identity, identity))
    policy = load_runtime_policy(path, identity)
    runtime_hosts = {str(item).strip().lower() for item in policy.contract.get("allowed_runtime_hostnames", ())}
    current_host = socket.gethostname().strip().lower()
    if runtime_hosts and current_host not in runtime_hosts:
        raise AdmissionPolicyError(
            f"agent '{policy.canonical_id}' is not admitted on runtime host '{current_host}'"
        )
    allowed_providers = set(policy.contract.get("allowed_providers", ()))
    allowed_models = set(policy.contract.get("allowed_models", ()))
    if agent.provider not in allowed_providers:
        raise AdmissionPolicyError(f"provider '{agent.provider}' is not declared for {policy.canonical_id}")
    if agent.model not in allowed_models:
        raise AdmissionPolicyError(f"model '{agent.model}' is not declared for {policy.canonical_id}")
    agent._canonical_agent_id = policy.canonical_id
    agent._admission_policy = policy


def _inside(path: str, roots: tuple[str, ...]) -> bool:
    candidate = Path(path).expanduser().resolve(strict=False)
    for root in roots:
        base = Path(root).expanduser().resolve(strict=False)
        if candidate == base or base in candidate.parents:
            return True
    return False


def _iter_values(args: Mapping[str, Any], keys: set[str]):
    for key, value in args.items():
        if key.lower() in keys and isinstance(value, (str, os.PathLike)):
            yield str(value)


def authorize_tool_call(agent: Any, tool: str, args: Mapping[str, Any]) -> AdmissionDecision:
    required = bool(getattr(agent, "_admission_required", False))
    policy = getattr(agent, "_admission_policy", None)
    if not required and policy is None:
        return AdmissionDecision(True, "admission not activated")
    if policy is None:
        return AdmissionDecision(False, "runtime admission policy is missing")
    contract = policy.contract
    allowed_tools = set(contract.get("allowed_tools", ()))
    if tool not in allowed_tools:
        return AdmissionDecision(False, f"tool '{tool}' is not declared for {policy.canonical_id}")
    read_roots = tuple(contract.get("allowed_read_roots", ()))
    write_roots = tuple(contract.get("allowed_write_roots", ()))
    write_like = tool in set(contract.get("write_tools", ()))
    for value in _iter_values(args, _PATH_KEYS):
        candidate = Path(value).expanduser().resolve(strict=False)
        if candidate.name.lower() in _SENSITIVE_PATH_NAMES or any(
            part.lower() in _SENSITIVE_PATH_PARTS for part in candidate.parts
        ):
            return AdmissionDecision(False, "path is protected by secret/policy boundary")
        roots = write_roots if write_like else read_roots
        if not roots or not _inside(value, roots):
            return AdmissionDecision(False, "requested path is outside declared scope")
    allowed_hosts = set(contract.get("allowed_remote_hosts", ()))
    for host in _iter_values(args, _HOST_KEYS):
        if host not in allowed_hosts:
            return AdmissionDecision(False, f"remote host '{host}' is not declared")
    if tool in _DELEGATION_TOOLS:
        child = str(args.get("agent") or args.get("profile") or args.get("target") or "").lower()
        if child not in set(contract.get("allowed_delegation_targets", ())):
            return AdmissionDecision(False, f"delegation to '{child or '<empty>'}' is forbidden")
    approval_class = str(args.get("approval_class") or "none")
    if tool in _EXTERNAL_TOOLS or approval_class != "none":
        approved = bool(args.get("approved", False))
        required_classes = set(contract.get("approval_classes", ()))
        if approval_class not in required_classes or not approved:
            return AdmissionDecision(False, "protected external action requires explicit approval", approval_class)
    return AdmissionDecision(True, "allowed", approval_class)


def admission_receipt(agent: Any, tool: str, decision: AdmissionDecision) -> dict[str, str]:
    policy = getattr(agent, "_admission_policy", None)
    return {
        "canonical_agent_id": getattr(policy, "canonical_id", "unmanaged"),
        "policy_version": getattr(policy, "version", "none"),
        "policy_hash": getattr(policy, "policy_hash", "none"),
        "tool": tool,
        "decision": "allow" if decision.allowed else "deny",
        "approval_class": decision.approval_class,
        "execution_status": "authorized" if decision.allowed else "blocked",
    }
