"""Fail-closed, run-keyed policy leases for trusted scheduled adapters."""
from __future__ import annotations
import contextlib
from typing import Any, Dict, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from hermes_cli.plugins import PluginManager
AUTHORITATIVE_SCHEDULED_HOOKS = frozenset({
    "on_session_start", "mcp_request_metadata", "mcp_tool_result", "on_session_finalize",
})
def _delivery_manager():
    from hermes_cli.plugins import _delivery_manager
    return _delivery_manager()

def _invoke_authoritative_policy(
    manager: PluginManager, policy_id: str, hook_name: str, payload: Dict[str, Any],
) -> Any:
    callback = manager._authoritative_policies.get(policy_id, {}).get(hook_name)
    if callback is None:
        raise RuntimeError(
            f"required authoritative policy {policy_id!r} has no {hook_name!r} callback"
        )
    return manager._invoke_hook_callback(callback, payload)


def _lease_identity(run_id: str, lease: Dict[str, Any]) -> Dict[str, Any]:
    """Identity every authoritative callback receives for a bound run."""
    return {
        "run_id": run_id,
        "session_id": lease.get("session_id", ""),
        "root_session_id": lease.get("root_session_id", ""),
    }


def _rebind_run_session_locked(
    manager: PluginManager, run_id: str, session_id: str,
) -> None:
    """Record the run's current transcript session under the caller's lock."""
    if not session_id:
        return
    lease = manager._authoritative_runs.get(run_id)
    if lease is None:
        return
    lease["session_id"] = session_id
    manager._authoritative_run_by_session[session_id] = run_id


def activate_authoritative_run(
    policy_id: str, run_id: str, session_id: str, **kwargs: Any,
) -> Dict[str, Any]:
    """Admit and bind one scheduled RUN through a required fail-closed policy.

    The lease is keyed by the run's immutable fire identity, never by the
    transcript session id. Compression rotates ``agent.session_id`` mid-run,
    so a lease keyed by that id silently disappears at the rotation boundary
    and every later authority decision falls back to the best-effort observer
    bus — the exact fail-open the authoritative path exists to remove.

    Re-entry for the same run rebinds the current session and replays the
    original admission decision: admission is decided once per fire.
    """
    manager = _delivery_manager()
    policy_id = str(policy_id)
    run_id = str(run_id or "")
    session_id = str(session_id or "")
    if not run_id:
        raise RuntimeError("authoritative run id is absent")
    with manager._authoritative_lock:
        lease = manager._authoritative_runs.get(run_id)
        if lease is not None:
            if lease.get("policy_id") != policy_id:
                raise RuntimeError(
                    "authoritative run is already bound to a different policy"
                )
            _rebind_run_session_locked(manager, run_id, session_id)
            return dict(lease["decision"])
        lease = {
            "policy_id": policy_id,
            "root_session_id": session_id,
            "session_id": session_id,
        }
        decision = _invoke_authoritative_policy(
            manager, policy_id, "on_session_start",
            {**_lease_identity(run_id, lease), **kwargs},
        )
        if not isinstance(decision, dict) or decision.get("action") != "allow":
            reason = decision.get("reason") if isinstance(decision, dict) else None
            raise RuntimeError(str(reason or "authoritative session admission denied"))
        lease["decision"] = dict(decision)
        manager._authoritative_runs[run_id] = lease
        _rebind_run_session_locked(manager, run_id, session_id)
        return decision


def bind_authoritative_run_session(run_id: str, session_id: str) -> None:
    """Follow a mid-run transcript rotation (a compression continuation).

    Keeping the reverse index current lets a call that only knows the rotated
    session id still resolve its run lease instead of silently downgrading.
    """
    manager = _delivery_manager()
    with manager._authoritative_lock:
        _rebind_run_session_locked(
            manager, str(run_id or ""), str(session_id or ""),
        )


def resolve_authoritative_run(
    run_id: Optional[str] = None, session_id: Optional[str] = None,
) -> Optional[str]:
    """Return the run id whose policy governs this call, else ``None``.

    Prefers the immutable run identity and falls back to any session id the
    lease has been bound to (root or compression child).
    """
    manager = _delivery_manager()
    lock = getattr(manager, "_authoritative_lock", contextlib.nullcontext())
    with lock:
        runs = getattr(manager, "_authoritative_runs", {})
        resolved_run = str(run_id or "")
        if resolved_run and resolved_run in runs:
            return resolved_run
        resolved_session = str(session_id or "")
        if not resolved_session:
            return None
        mapped = getattr(
            manager, "_authoritative_run_by_session", {},
        ).get(resolved_session)
        return mapped if mapped in runs else None


def authoritative_run_policy(run_id: str) -> Optional[str]:
    """Return the policy id bound to ``run_id``, or ``None``."""
    manager = _delivery_manager()
    lock = getattr(manager, "_authoritative_lock", contextlib.nullcontext())
    with lock:
        lease = getattr(manager, "_authoritative_runs", {}).get(str(run_id or ""))
        return lease.get("policy_id") if isinstance(lease, dict) else None


def invoke_authoritative_run_hook(
    run_id: str, hook_name: str, *, session_id: Optional[str] = None, **kwargs: Any,
) -> Any:
    """Invoke one active run's policy without observer-style isolation."""
    manager = _delivery_manager()
    with manager._authoritative_lock:
        resolved_run = str(run_id or "")
        lease = manager._authoritative_runs.get(resolved_run)
        if lease is None:
            raise RuntimeError("run has no active authoritative policy")
        if session_id:
            _rebind_run_session_locked(manager, resolved_run, str(session_id))
        return _invoke_authoritative_policy(
            manager, lease["policy_id"], hook_name,
            {**_lease_identity(resolved_run, lease), **kwargs},
        )


def finalize_authoritative_run(
    run_id: str, *, session_id: Optional[str] = None, **kwargs: Any,
) -> Dict[str, Any]:
    """Require a durable policy receipt before forgetting a run lease.

    A failed settlement deliberately KEEPS the lease: the run is not settled,
    so forgetting the binding would let the next lookup report "no policy
    required" for a fire that never produced a receipt.
    """
    manager = _delivery_manager()
    with manager._authoritative_lock:
        resolved_run = str(run_id or "")
        lease = manager._authoritative_runs.get(resolved_run)
        if lease is None:
            raise RuntimeError("run has no active authoritative policy")
        if session_id:
            _rebind_run_session_locked(manager, resolved_run, str(session_id))
        receipt = _invoke_authoritative_policy(
            manager, lease["policy_id"], "on_session_finalize",
            {**_lease_identity(resolved_run, lease), **kwargs},
        )
        if not isinstance(receipt, dict) or receipt.get("status") != "finalized":
            raise RuntimeError("authoritative session finalizer returned no receipt")
        del manager._authoritative_runs[resolved_run]
        for bound_session in [
            sid
            for sid, rid in manager._authoritative_run_by_session.items()
            if rid == resolved_run
        ]:
            del manager._authoritative_run_by_session[bound_session]
        return {**receipt, "policy": lease["policy_id"], **_lease_identity(resolved_run, lease)}
