"""Host-wide execution permits, independent of profile-local conversation writer leases."""

from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass, field
import threading
from pathlib import Path

from hermes_cli import active_sessions as registry
from hermes_cli.config_backend import ConfigBackendUnavailable
from hermes_constants import get_default_hermes_root, get_hermes_home

_current_permit = contextvars.ContextVar("execution_permit", default=None)


@dataclass
class ExecutionPermit:
    lease: registry.ActiveSessionLease
    budget_id: str
    profile_home: str
    identity: dict = field(repr=False)
    conversation_root: str | None = None
    _root_lock: object = field(default_factory=threading.Lock, repr=False)

    def bind_root(self, root_id):
        with self._root_lock:
            if self.conversation_root is None:
                self.conversation_root = root_id
            return self.conversation_root == root_id

    def release(self):
        from hermes_cli.execution_retirement import record_settlement, reconcile_locked

        if self.lease.released:
            return
        record_settlement(self)
        with registry._FileLock(self.lease.lock_path, timeout=2.0):
            reconcile_locked(self.lease.state_path, registry._read_entries(self.lease.state_path, strict=True))
            self.lease.released = True

    def vouch(self):
        return {"root": str(self.lease.state_path.parent.parent), "lease_id": self.lease.lease_id,
                "budget_id": self.budget_id, "profile_home": self.profile_home,
                "execution_id": self.lease.session_id, "conversation_root": self.conversation_root}


def _unavailable():
    return None, registry.ActiveSessionRefusal(
        "Hermes could not verify the execution budget. No work started; try again after coordination is restored.",
        registry.SESSION_COORDINATION_UNAVAILABLE)


def _paths(root):
    return root / "runtime" / "execution_permits.json", root / "runtime" / "execution_permits.lock"


def _live_entries(path):
    from hermes_cli.execution_retirement import reconcile_locked

    entries, _ = reconcile_locked(path, registry._read_entries(path, strict=True))
    entries = registry._prune_dead(entries, strict=True)
    for entry in entries:
        metadata = entry.get("metadata") or {}
        if (entry.get("process_start_time") is None
                or not all(isinstance(metadata.get(key), str) and metadata[key]
                           for key in ("budget_id", "profile_home"))):
            raise registry.ActiveSessionRegistryError("invalid execution permit ownership")
    return entries


def _holders(entries):
    return list({entry["metadata"]["budget_id"]: entry for entry in entries}.values())


def _capacity_refusal(holders, cap):
    if cap is None or len(holders) < cap:
        return None
    return registry.ActiveSessionRefusal(
        f"Hermes is at the execution limit ({len(holders)}/{cap}). "
        f"Held by: {registry.summarize_holders(holders)}. Try again when work finishes.",
        registry.MAX_CONCURRENT_SESSIONS)


def _new_permit(path, lock, entries, *, execution_id, surface, profile_home, budget_id):
    lease_id = uuid.uuid4().hex
    entry = registry._lease_entry(
        lease_id=lease_id, session_id=execution_id, surface=surface, track_liveness=True,
        metadata={"budget_id": budget_id, "profile_home": profile_home})
    if entry["process_start_time"] is None:
        raise registry.ActiveSessionRegistryError("cannot verify execution owner birth")
    entries.append(entry)
    registry._write_entries(path, entries)
    lease = registry.ActiveSessionLease(lease_id, execution_id, surface, state_path=path, lock_path=lock,
                                        track_liveness=True)
    return ExecutionPermit(lease, budget_id, profile_home, entry), None


def _host_limit(root):
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import build_profile_secret_scope, set_secret_scope, reset_secret_scope

    home_token = set_hermes_home_override(str(root))
    try:
        secret_token = set_secret_scope(build_profile_secret_scope(root), profile_home=str(root))
        try:
            config = load_user_config_effective(root / "config.yaml", fail_closed=True)
        finally:
            reset_secret_scope(secret_token)
    finally:
        reset_hermes_home_override(home_token)
    raw = config.get("max_concurrent_sessions", (config.get("gateway") or {}).get("max_concurrent_sessions"))
    cap = registry.resolve_max_concurrent_sessions(config)
    if isinstance(raw, bool) or (raw is not None and cap is None and raw not in (0, "0")):
        raise ValueError("invalid host execution limit")
    return cap


def try_acquire_execution_permit(*, profile_home=None, execution_id: str, surface: str):
    """Reserve one running conversation from the owning host root's effective configured limit."""
    from agent.safe_worker_policy import worker_config_snapshot

    home = Path(profile_home or get_hermes_home()).resolve()
    root = get_default_hermes_root(home=home).resolve()
    path, lock = _paths(root)
    try:
        # Frozen managed children must borrow their owner's reservation, never substitute
        # their profile snapshot for the host's budget configuration.
        if worker_config_snapshot() is not None:
            return _unavailable()
        cap = _host_limit(root)
        with registry._FileLock(lock, timeout=2.0):
            entries = _live_entries(path)
            holders = _holders(entries)
            if (refusal := _capacity_refusal(holders, cap)) is not None:
                return None, refusal
            return _new_permit(path, lock, entries, execution_id=execution_id, surface=surface,
                               profile_home=str(home), budget_id=uuid.uuid4().hex)
    # health: allow BLE001 -- pluggable config failures refuse all new work; never disclose backend exception content
    except (Exception, ConfigBackendUnavailable):
        return _unavailable()


def borrow_execution_permit(vouch):
    """Add this process's exact reference to a live owner's private reservation.

    Each reference has its own PID/birth and release identity. A child can outlive
    its parent without freeing the conversation's slot or releasing a successor.
    """
    try:
        required = {"root", "lease_id", "budget_id", "profile_home", "execution_id", "conversation_root"}
        if (not isinstance(vouch, dict) or set(vouch) != required
                or any(not isinstance(vouch[key], str) or not vouch[key].strip()
                       for key in required - {"conversation_root"})
                or (vouch["conversation_root"] is not None and not isinstance(vouch["conversation_root"], str))
                or not Path(vouch["root"]).is_absolute() or not Path(vouch["profile_home"]).is_absolute()):
            return _unavailable()
        root = Path(vouch["root"]).resolve()
        path, lock = _paths(root)
        with registry._FileLock(lock, timeout=2.0):
            entries = _live_entries(path)
            parent = next((entry for entry in entries if entry["lease_id"] == vouch["lease_id"]), None)
            if parent is None or parent["session_id"] != vouch["execution_id"]:
                return _unavailable()
            metadata = parent["metadata"]
            if (metadata["budget_id"] != vouch["budget_id"]
                    or metadata["profile_home"] != vouch["profile_home"]
                    or get_default_hermes_root(home=vouch["profile_home"]).resolve() != root):
                return _unavailable()
            permit, refusal = _new_permit(path, lock, entries, execution_id=parent["session_id"], surface=parent["surface"],
                                         profile_home=metadata["profile_home"], budget_id=metadata["budget_id"])
            permit.conversation_root = vouch.get("conversation_root")
            return permit, refusal
    except (OSError, RuntimeError, ValueError):
        return _unavailable()


def bind_execution_permit(permit):
    return _current_permit.set(permit)


def reset_execution_permit(token):
    _current_permit.reset(token)


def current_execution_permit():
    return _current_permit.get()


def execution_budget_snapshot(*, profile_home=None):
    """Root budget and one metadata row per executing conversation; uncertainty raises."""
    home = Path(profile_home or get_hermes_home()).resolve()
    root = get_default_hermes_root(home=home).resolve()
    path, lock = _paths(root)
    cap = _host_limit(root)
    with registry._FileLock(lock, timeout=2.0):
        return cap, _holders(_live_entries(path))


def reconcile_execution_permits(*, profile_home=None):
    """Recover durable completed-reference receipts; never evict live work by age."""
    from hermes_cli.execution_retirement import reconcile_locked

    home = Path(profile_home or get_hermes_home()).resolve()
    path, lock = _paths(get_default_hermes_root(home=home).resolve())
    with registry._FileLock(lock, timeout=2.0):
        _, count = reconcile_locked(path, registry._read_entries(path, strict=True))
        return count


def execution_capacity_refusal(*, profile_home=None):
    """Best-effort preflight of the same budget; only acquiring a permit reserves it."""
    try:
        cap, holders = execution_budget_snapshot(profile_home=profile_home)
        return _capacity_refusal(holders, cap)
    # health: allow BLE001 -- unknown config or ownership must return refusal, never free capacity or expose exception content
    except (Exception, ConfigBackendUnavailable):
        return _unavailable()[1]
