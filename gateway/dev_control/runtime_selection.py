"""Runtime selection policy for Dev execution plans."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

AUTO_RUNTIME = "auto"
DEFAULT_RUNTIME = "ao"


def select_worker_runtime(
    *,
    goal: Optional[str] = None,
    prompt: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
    requested_runtime: Optional[str] = None,
    permissions: Optional[str] = None,
    project_id: Optional[str] = None,
    runtimes: Optional[Iterable[Dict[str, Any]]] = None,
    runtime_policy_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    profile = dict(profile or {})
    explicit_runtime = _normalize_requested_runtime(requested_runtime)
    profile_runtime = _normalize_requested_runtime(profile.get("runtime"))
    candidates = list(runtimes or [])
    candidate_ids = [str(runtime.get("id") or "") for runtime in candidates if runtime.get("id")]
    required = _required_capabilities(goal=goal, prompt=prompt, permissions=permissions or profile.get("permissions"))

    if explicit_runtime and explicit_runtime != AUTO_RUNTIME:
        return _decision(
            selected_runtime=explicit_runtime,
            selection_mode="explicit",
            reason=f"Task requested runtime {explicit_runtime}.",
            candidate_runtimes=candidate_ids,
            required_capabilities=required,
            runtime_policy_status="not_applicable",
            runtime_policy_reason="Explicit runtime requests bypass auto runtime policy.",
        )

    if profile_runtime and profile_runtime != AUTO_RUNTIME:
        return _decision(
            selected_runtime=profile_runtime,
            selection_mode="profile",
            reason=f"Launch profile selected runtime {profile_runtime}.",
            candidate_runtimes=candidate_ids,
            required_capabilities=required,
            runtime_policy_status="not_applicable",
            runtime_policy_reason="Explicit launch profile runtime bypasses auto runtime policy.",
        )

    task_kind = _infer_task_kind(goal=goal, prompt=prompt, permissions=permissions or profile.get("permissions"))
    if task_kind == "inspect":
        openhands = _runtime_by_id(candidates, "openhands")
        policy = _runtime_policy_status(runtime_policy_evidence)
        if policy["runtime_policy_status"] == "degraded":
            fallback_reason = f"OpenHands benchmark evidence degraded: {policy['runtime_policy_reason']}"
            return _decision(
                selected_runtime=DEFAULT_RUNTIME,
                selection_mode="fallback",
                reason="Read-only inspection fell back to AO because benchmark evidence degraded OpenHands.",
                candidate_runtimes=candidate_ids,
                fallback_runtime=DEFAULT_RUNTIME,
                required_capabilities=required,
                warnings=[fallback_reason],
                runtime_fallback_reason=fallback_reason,
                **policy,
            )
        if _runtime_satisfies(openhands, required):
            warnings = []
            if policy["runtime_policy_status"] == "insufficient_evidence":
                warnings.append(policy["runtime_policy_reason"])
            return _decision(
                selected_runtime="openhands",
                selection_mode="auto",
                reason="Read-only inspection can run on healthy OpenHands with output capture.",
                candidate_runtimes=candidate_ids,
                fallback_runtime=DEFAULT_RUNTIME,
                required_capabilities=required,
                warnings=warnings,
                **policy,
            )
        warning = _runtime_unavailable_reason(openhands, "OpenHands is unavailable or missing required capabilities.")
        return _decision(
            selected_runtime=DEFAULT_RUNTIME,
            selection_mode="fallback",
            reason="Read-only inspection fell back to AO.",
            candidate_runtimes=candidate_ids,
            fallback_runtime=DEFAULT_RUNTIME,
            required_capabilities=required,
            warnings=[warning],
            runtime_fallback_reason=warning,
            runtime_policy_status="fallback",
            runtime_policy_reason="OpenHands failed runtime availability or capability checks.",
            runtime_policy_evidence=runtime_policy_evidence or {},
        )

    reason = {
        "implement": "Implementation and write-capable tasks require AO worktree/terminal support.",
        "test": "Verification tasks use AO for consistent worktree and terminal support.",
        "recovery": "Recovery and retry-family tasks use AO in this phase.",
    }.get(task_kind, "AO is the default production runtime for this task.")
    return _decision(
        selected_runtime=DEFAULT_RUNTIME,
        selection_mode="auto",
        reason=reason,
        candidate_runtimes=candidate_ids,
        fallback_runtime=None,
        required_capabilities=required,
        runtime_policy_status="not_applicable",
        runtime_policy_reason="Benchmark evidence only gates read-only OpenHands auto-selection.",
        runtime_policy_evidence=runtime_policy_evidence or {},
    )


def _normalize_requested_runtime(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return None
    return AUTO_RUNTIME if text == AUTO_RUNTIME else text


def _infer_task_kind(*, goal: Optional[str], prompt: Optional[str], permissions: Optional[str]) -> str:
    normalized_permissions = str(permissions or "").strip().lower().replace("-", "_")
    if normalized_permissions in {"read_only", "readonly", "read"}:
        return "inspect"
    if normalized_permissions in {"edit", "write", "writable", "writeable"}:
        return "implement"

    text = f"{goal or ''}\n{prompt or ''}\n{permissions or ''}".lower()
    if any(token in text for token in ("repair", "retry", "reassign", "recover", "resume")):
        return "recovery"
    if any(token in text for token in ("implement", "edit", "write", "modify", "fix", "patch")):
        return "implement"
    if any(token in text for token in ("test", "verify", "verification", "build", "lint")):
        return "test"
    if "read_only" in text or any(token in text for token in ("inspect", "read-only", "read only", "analyze", "review")):
        return "inspect"
    return "inspect"


def _required_capabilities(*, goal: Optional[str], prompt: Optional[str], permissions: Optional[str]) -> list[str]:
    task_kind = _infer_task_kind(goal=goal, prompt=prompt, permissions=permissions)
    if task_kind == "inspect":
        return ["can_spawn", "can_capture_output"]
    if task_kind in {"implement", "test", "recovery"}:
        return ["can_spawn", "supports_worktree", "supports_terminal", "can_capture_output"]
    return ["can_spawn", "can_capture_output"]


def _runtime_by_id(runtimes: list[Dict[str, Any]], runtime_id: str) -> Optional[Dict[str, Any]]:
    for runtime in runtimes:
        if runtime.get("id") == runtime_id:
            return runtime
    return None


def _runtime_satisfies(runtime: Optional[Dict[str, Any]], required: list[str]) -> bool:
    if not runtime:
        return False
    if not runtime.get("available") or not runtime.get("launch_supported") or runtime.get("test_only"):
        return False
    capabilities = runtime.get("capabilities") if isinstance(runtime.get("capabilities"), dict) else runtime
    return all(bool(capabilities.get(capability)) for capability in required)


def _runtime_unavailable_reason(runtime: Optional[Dict[str, Any]], default: str) -> str:
    if not runtime:
        return "OpenHands runtime is not registered."
    return str(runtime.get("setup_warning") or default)


def _decision(
    *,
    selected_runtime: str,
    selection_mode: str,
    reason: str,
    candidate_runtimes: list[str],
    required_capabilities: list[str],
    fallback_runtime: Optional[str] = None,
    warnings: Optional[list[str]] = None,
    runtime_fallback_reason: Optional[str] = None,
    runtime_policy_status: str = "not_applicable",
    runtime_policy_reason: Optional[str] = None,
    runtime_policy_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "selected_runtime": selected_runtime,
        "selection_mode": selection_mode,
        "reason": reason,
        "candidate_runtimes": candidate_runtimes,
        "fallback_runtime": fallback_runtime,
        "required_capabilities": required_capabilities,
        "warnings": warnings or [],
        "runtime_fallback_reason": runtime_fallback_reason,
        "runtime_policy_status": runtime_policy_status,
        "runtime_policy_reason": runtime_policy_reason,
        "runtime_policy_evidence": runtime_policy_evidence or {},
    }


def _runtime_policy_status(evidence: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not evidence:
        return {
            "runtime_policy_status": "insufficient_evidence",
            "runtime_policy_reason": "No benchmark evidence was available for runtime policy gating.",
            "runtime_policy_evidence": {},
        }
    status = str(evidence.get("status") or "insufficient_evidence")
    if status not in {"insufficient_evidence", "healthy", "degraded"}:
        status = "insufficient_evidence"
    return {
        "runtime_policy_status": status,
        "runtime_policy_reason": str(evidence.get("reason") or "Runtime policy evidence was inconclusive."),
        "runtime_policy_evidence": evidence,
    }
