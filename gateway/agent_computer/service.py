"""Agent Computer control plane.

Binds durable computers to permanent Hermes profiles, attaches a separate
BrowserIdentity with exclusive lock, and fences input with a ControlLease.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .adapter import ComputerRuntime, InMemoryRuntime, RuntimeHandle, new_identity_profile_dir
from .errors import (
    CheckpointRequiredError,
    ConflictError,
    ForbiddenError,
    IdentityBusyError,
    InvalidTokenError,
    NotFoundError,
    ObserveRequiredError,
    RevokedError,
    StaleControllerError,
)
from .models import (
    OWNER_PRINCIPAL,
    SENSITIVE_ACTION_CLASSES,
    AgentComputer,
    AuditEvent,
    BrowserIdentity,
    Checkpoint,
    CheckpointStatus,
    ControlAuthority,
    ControlLease,
    Controller,
    InputReceipt,
    LeaseStatus,
    Lifecycle,
    Observation,
    TakeoverToken,
    agent_principal,
    is_agent_principal,
    is_owner_principal,
)
from .store import AgentComputerStore

BACKEND_NAME = "hermes_chromium"
OWNER_TAKEOVER_TTL_S = 30 * 60


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None = None) -> str:
    return (dt or _now()).isoformat()


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AgentComputerService:
    def __init__(
        self,
        store: AgentComputerStore,
        runtime: ComputerRuntime | None = None,
        *,
        data_root: str | Path | None = None,
        clock: Callable[[], datetime] | None = None,
        takeover_ttl_s: int = OWNER_TAKEOVER_TTL_S,
    ):
        self.store = store
        self.runtime = runtime or InMemoryRuntime()
        self.data_root = Path(data_root or ".")
        self.clock = clock or _now
        self.takeover_ttl_s = takeover_ttl_s
        self._lock = threading.RLock()
        self._handles: dict[str, RuntimeHandle] = {}
        self._owner_transports: dict[str, int] = {}

    # ── computers ────────────────────────────────────────────────────
    def ensure_computer(self, profile_id: str) -> AgentComputer:
        if not profile_id or profile_id.startswith("session:") or profile_id.startswith("run:"):
            raise ConflictError("computer ownership must bind to a permanent profile")
        existing = self.store.get_computer_by_profile(profile_id)
        if existing:
            return existing
        cid = f"ac_{uuid.uuid4().hex}"
        persistence = str(self.data_root / "computers" / profile_id / cid)
        Path(persistence).mkdir(parents=True, exist_ok=True)
        computer = AgentComputer(
            id=cid,
            agent_profile_id=profile_id,
            backend=BACKEND_NAME,
            persistence_ref=persistence,
            lifecycle=Lifecycle.IDLE,
            created_at=_iso(self.clock()),
            updated_at=_iso(self.clock()),
        )
        self.store.upsert_computer(computer)
        self._audit(cid, "computer_provision", OWNER_PRINCIPAL, {"profile_id": profile_id})
        return computer

    def get_computer(self, computer_id: str) -> AgentComputer:
        computer = self.store.get_computer(computer_id)
        if not computer:
            raise NotFoundError(f"computer not found: {computer_id}")
        return computer

    def list_computers(self) -> list[AgentComputer]:
        return self.store.list_computers()

    def list_identities(self) -> list[BrowserIdentity]:
        return self.store.list_identities()

    def list_audit(self, computer_id: str | None = None) -> list[AuditEvent]:
        return self.store.list_audit(computer_id)

    def authorize_read(self, computer: AgentComputer, principal: str) -> None:
        if is_owner_principal(principal):
            return
        if is_agent_principal(principal, computer.agent_profile_id):
            return
        raise ForbiddenError("not authorized for this computer")

    def authorize_agent_input(self, computer: AgentComputer, principal: str) -> None:
        if not is_agent_principal(principal, computer.agent_profile_id):
            raise ForbiddenError("only the owning agent may use agent input")

    def authorize_owner(self, principal: str) -> None:
        if not is_owner_principal(principal):
            raise ForbiddenError("owner-only")

    # ── identities ───────────────────────────────────────────────────
    def create_identity(
        self,
        *,
        ownership: list[str],
        metadata: dict[str, Any] | None = None,
        identity_id: str | None = None,
    ) -> BrowserIdentity:
        if not ownership:
            raise ConflictError("BrowserIdentity ownership must name at least one profile")
        iid = identity_id or f"bi_{uuid.uuid4().hex}"
        profile_ref = new_identity_profile_dir(self.data_root, iid)
        identity = BrowserIdentity(
            id=iid,
            profile_ref=profile_ref,
            ownership=list(ownership),
            metadata=dict(metadata or {}),
            created_at=_iso(self.clock()),
        )
        self.store.upsert_identity(identity)
        return identity

    def get_identity(self, identity_id: str) -> BrowserIdentity:
        identity = self.store.get_identity(identity_id)
        if not identity:
            raise NotFoundError(f"browser identity not found: {identity_id}")
        return identity

    def attach_identity(self, computer_id: str, identity_id: str, principal: str) -> AgentComputer:
        with self._lock:
            computer = self.get_computer(computer_id)
            self.authorize_read(computer, principal)
            identity = self.get_identity(identity_id)
            if identity.revoked:
                raise RevokedError("browser identity revoked")
            if not identity.allows(computer.agent_profile_id):
                raise ForbiddenError("profile is not authorized for this BrowserIdentity")
            locked = self.store.try_lock_identity(identity.id, computer.id, principal)
            if locked is None:
                identity = self.get_identity(identity_id)
                raise IdentityBusyError(
                    "browser identity is exclusively locked",
                    details={
                        "identity_id": identity.id,
                        "locked_by_computer_id": identity.lock_computer_id,
                    },
                )
            identity = locked
            computer.active_browser_identity_id = identity.id
            computer.updated_at = _iso(self.clock())
            self.store.upsert_computer(computer)
            self._audit(
                computer.id,
                "browser_identity_attach",
                principal,
                {"identity_id": identity.id},
            )
            return computer

    def detach_identity(self, computer_id: str, principal: str) -> AgentComputer:
        with self._lock:
            computer = self.get_computer(computer_id)
            self.authorize_read(computer, principal)
            iid = computer.active_browser_identity_id
            if iid:
                identity = self.get_identity(iid)
                if identity.lock_computer_id == computer.id:
                    identity.lock_computer_id = None
                    identity.lock_holder = None
                    self.store.upsert_identity(identity)
                computer.active_browser_identity_id = None
                computer.updated_at = _iso(self.clock())
                self.store.upsert_computer(computer)
                self._audit(computer.id, "browser_identity_detach", principal, {"identity_id": iid})
            return computer

    def revoke_identity(self, identity_id: str, principal: str) -> BrowserIdentity:
        self.authorize_owner(principal)
        identity = self.get_identity(identity_id)
        identity.revoked = True
        identity.lock_computer_id = None
        identity.lock_holder = None
        self.store.upsert_identity(identity)
        for computer in self.store.list_computers():
            if computer.active_browser_identity_id == identity_id:
                computer.active_browser_identity_id = None
                self.store.upsert_computer(computer)
        self._audit(None, "identity_revoked", principal, {"identity_id": identity_id})
        return identity

    # ── lifecycle ────────────────────────────────────────────────────
    def wake(self, computer_id: str, principal: str) -> tuple[AgentComputer, ControlLease]:
        with self._lock:
            self.expire_owner_if_needed(computer_id)
            computer = self.get_computer(computer_id)
            self.authorize_read(computer, principal)
            if (
                computer.control_authority == ControlAuthority.OWNER_CONTROLLED
                and is_agent_principal(principal, computer.agent_profile_id)
            ):
                raise ConflictError("owner currently controls this computer")
            existing = self._handles.get(computer.id)
            if existing and not self._runtime_alive(existing):
                try:
                    self.runtime.sleep(existing)
                except Exception:
                    pass
                self._handles.pop(computer.id, None)
                existing = None
            if (
                existing
                and self._runtime_alive(existing)
                and computer.lifecycle in (Lifecycle.READY, Lifecycle.BUSY, Lifecycle.WAKING)
            ):
                lease = self.store.active_lease_for_computer(computer.id)
                if lease is None:
                    if computer.control_authority == ControlAuthority.OWNER_CONTROLLED:
                        lease = self._issue_owner_lease(computer)
                    else:
                        lease = self._issue_agent_lease(computer)
                computer.lifecycle = Lifecycle.READY
                computer.updated_at = _iso(self.clock())
                self.store.upsert_computer(computer)
                return computer, lease
            computer.lifecycle = Lifecycle.WAKING
            self.store.upsert_computer(computer)
            identity = None
            if computer.active_browser_identity_id:
                identity = self.get_identity(computer.active_browser_identity_id)
                if identity.revoked:
                    raise RevokedError("attached browser identity is revoked")
            handle = self.runtime.wake(computer, identity)
            self._handles[computer.id] = handle
            computer.lifecycle = Lifecycle.READY
            computer.updated_at = _iso(self.clock())
            self.store.upsert_computer(computer)
            lease = self.store.active_lease_for_computer(computer.id)
            if lease is None:
                if computer.control_authority in (
                    ControlAuthority.OWNER_CONTROLLED,
                    ControlAuthority.TAKEOVER_PENDING,
                    ControlAuthority.YIELDING,
                ):
                    lease = self._issue_owner_lease(computer)
                else:
                    lease = self._issue_agent_lease(computer)
            self._audit(computer.id, "runtime_start", principal, {"backend": handle.backend})
            return computer, lease

    def sleep(self, computer_id: str, principal: str) -> AgentComputer:
        with self._lock:
            computer = self.get_computer(computer_id)
            self.authorize_read(computer, principal)
            handle = self._handles.pop(computer.id, None)
            if handle:
                self.runtime.sleep(handle)
            computer.lifecycle = Lifecycle.SLEEPING
            computer.updated_at = _iso(self.clock())
            self.store.upsert_computer(computer)
            self._audit(computer.id, "runtime_sleep", principal, {})
            return computer

    def _runtime_alive(self, handle: RuntimeHandle) -> bool:
        alive = getattr(self.runtime, "alive", None)
        if not callable(alive):
            return True
        try:
            return bool(alive(handle))
        except Exception:
            return False

    def _handle(self, computer: AgentComputer) -> RuntimeHandle:
        handle = self._handles.get(computer.id)
        if handle and self._runtime_alive(handle):
            return handle
        if handle:
            try:
                self.runtime.sleep(handle)
            except Exception:
                pass
            self._handles.pop(computer.id, None)
        identity = None
        if computer.active_browser_identity_id:
            identity = self.get_identity(computer.active_browser_identity_id)
        handle = self.runtime.wake(computer, identity)
        self._handles[computer.id] = handle
        self._audit(computer.id, "recovery", "system", {"reason": "recreate_runtime"})
        return handle

    # ── observe / act ────────────────────────────────────────────────
    def observe(self, computer_id: str, principal: str, *, lease_id: str, fencing_epoch: int) -> Observation:
        with self._lock:
            computer = self.get_computer(computer_id)
            self.authorize_read(computer, principal)
            self._require_lease(computer, principal, lease_id, fencing_epoch, allow_owner=True)
            computer = self.get_computer(computer_id)
            obs = self.runtime.observe(self._handle(computer))
            computer.resume_observe_required = False
            computer.workspace_url = obs.url
            computer.workspace_title = obs.title
            computer.updated_at = _iso(self.clock())
            self.store.upsert_computer(computer)
            obs.fencing_epoch = computer.fencing_epoch
            obs.controller = computer.control_authority.value
            return obs

    def act(
        self,
        computer_id: str,
        principal: str,
        *,
        lease_id: str,
        fencing_epoch: int,
        kind: str,
        target: str = "",
        text: str = "",
        action_class: str = "",
        x: float | None = None,
        y: float | None = None,
        key: str = "",
        code: str = "",
        delta_x: float = 0,
        delta_y: float = 0,
    ) -> InputReceipt:
        with self._lock:
            computer = self.get_computer(computer_id)
            self.authorize_read(computer, principal)
            controller = self._require_lease(computer, principal, lease_id, fencing_epoch, allow_owner=True)
            computer = self.get_computer(computer_id)
            if computer.resume_observe_required and controller == Controller.AGENT:
                raise ObserveRequiredError("agent must re-observe after control returns")
            consumed_checkpoint_id = None
            if action_class in SENSITIVE_ACTION_CLASSES:
                open_cp = self.store.open_checkpoint(computer.id, action_class)
                if open_cp is None:
                    cp = Checkpoint(
                        id=f"cp_{uuid.uuid4().hex}",
                        computer_id=computer.id,
                        action_class=action_class,
                        status=CheckpointStatus.BLOCKED,
                        created_at=_iso(self.clock()),
                    )
                    self.store.upsert_checkpoint(cp)
                    self._audit(
                        computer.id,
                        "checkpoint_blocked",
                        principal,
                        {"action_class": action_class, "checkpoint_id": cp.id},
                    )
                    raise CheckpointRequiredError(
                        "sensitive action requires owner checkpoint",
                        details={"checkpoint_id": cp.id, "action_class": action_class},
                    )
                consumed_checkpoint_id = open_cp.id
            computer.lifecycle = Lifecycle.BUSY
            obs = self.runtime.act(
                self._handle(computer),
                kind=kind,
                target=target,
                text=text,
                action_class=action_class,
                x=x,
                y=y,
                key=key,
                code=code,
                delta_x=delta_x,
                delta_y=delta_y,
            )
            computer.workspace_url = obs.url
            computer.workspace_title = obs.title
            computer.lifecycle = Lifecycle.READY
            computer.updated_at = _iso(self.clock())
            self.store.upsert_computer(computer)
            if consumed_checkpoint_id:
                self.store.consume_checkpoint(consumed_checkpoint_id)
            self._audit(
                computer.id,
                "input_accepted",
                principal,
                {
                    "kind": kind,
                    "controller": controller.value,
                    "epoch": computer.fencing_epoch,
                    "lease_id": lease_id,
                },
            )
            return InputReceipt(
                accepted=True,
                fencing_epoch=computer.fencing_epoch,
                url=obs.url,
                title=obs.title,
                text=obs.text,
            )

    # ── takeover ─────────────────────────────────────────────────────
    def request_takeover(self, computer_id: str, principal: str, *, reason: str = "") -> dict[str, Any]:
        self.authorize_owner(principal)
        with self._lock:
            computer = self.get_computer(computer_id)
            if computer.control_authority == ControlAuthority.OWNER_CONTROLLED:
                token = self._mint_takeover_token(computer, principal)
                return {
                    "status": computer.control_authority.value,
                    "fencing_epoch": computer.fencing_epoch,
                    "takeover_token": token,
                    "duplicate": True,
                }
            computer.control_authority = ControlAuthority.TAKEOVER_PENDING
            self.store.revoke_leases(computer.id)
            computer.updated_at = _iso(self.clock())
            self.store.upsert_computer(computer)
            self._audit(computer.id, "takeover_requested", principal, {"reason": reason})
            self._audit(computer.id, "agent_yielded", agent_principal(computer.agent_profile_id), {})
            computer.control_authority = ControlAuthority.OWNER_CONTROLLED
            lease = self._issue_owner_lease(computer)
            token = self._mint_takeover_token(computer, principal)
            self.store.upsert_computer(computer)
            self._audit(
                computer.id,
                "owner_takeover_granted",
                principal,
                {"fencing_epoch": computer.fencing_epoch, "lease_id": lease.lease_id},
            )
            return {
                "status": computer.control_authority.value,
                "fencing_epoch": computer.fencing_epoch,
                "lease_id": lease.lease_id,
                "takeover_token": token,
                "duplicate": False,
            }

    def bind_owner_transport(self, computer_id: str, transport: object | None) -> None:
        if transport is None:
            return
        self._owner_transports[computer_id] = id(transport)

    def release_owner_for_transport(self, transport: object) -> int:
        key = id(transport)
        released = 0
        for computer_id, tok in list(self._owner_transports.items()):
            if tok != key:
                continue
            self.owner_disconnect(computer_id, OWNER_PRINCIPAL)
            self._owner_transports.pop(computer_id, None)
            released += 1
        return released

    def connect_takeover(
        self, computer_id: str, principal: str, *, takeover_token: str
    ) -> ControlLease:
        self.authorize_owner(principal)
        with self._lock:
            computer = self.get_computer(computer_id)
            record = self.store.get_token_by_hash(_hash_token(takeover_token))
            if not record or record.consumed:
                raise InvalidTokenError("takeover token invalid or consumed")
            if record.computer_id != computer.id:
                raise ForbiddenError("takeover token is bound to another computer")
            if record.owner_principal != principal:
                raise ForbiddenError("takeover token is bound to another principal")
            if record.fencing_epoch != computer.fencing_epoch:
                raise StaleControllerError("takeover token epoch is stale")
            expires = datetime.fromisoformat(record.expires_at)
            if self.clock() >= expires:
                self._audit(computer.id, "takeover_expired", principal, {"token_id": record.token_id})
                raise InvalidTokenError("takeover token expired")
            if computer.control_authority != ControlAuthority.OWNER_CONTROLLED:
                raise ConflictError("computer is not owner-controlled")
            self.store.mark_token_consumed(record.token_id)
            lease = self.store.active_lease_for_computer(computer.id)
            if lease is None:
                lease = self._issue_owner_lease(computer)
            self._audit(computer.id, "takeover_connected", principal, {"lease_id": lease.lease_id})
            return lease

    def give_back(self, computer_id: str, principal: str, *, lease_id: str, fencing_epoch: int) -> ControlLease:
        self.authorize_owner(principal)
        with self._lock:
            computer = self.get_computer(computer_id)
            if computer.control_authority != ControlAuthority.OWNER_CONTROLLED:
                # Duplicate give-back: return current agent lease if already returned.
                active = self.store.active_lease_for_computer(computer.id)
                if active and active.controller == Controller.AGENT:
                    return active
                raise ConflictError("computer is not owner-controlled")
            self._require_lease(computer, principal, lease_id, fencing_epoch, allow_owner=True)
            computer.control_authority = ControlAuthority.RETURNING
            self.store.revoke_leases(computer.id)
            self.store.expire_tokens_for_computer(computer.id)
            computer.resume_observe_required = True
            lease = self._issue_agent_lease(computer)
            computer.control_authority = ControlAuthority.AGENT_CONTROLLED
            computer.updated_at = _iso(self.clock())
            self.store.upsert_computer(computer)
            self._owner_transports.pop(computer.id, None)
            self._audit(computer.id, "control_returned", principal, {"new_epoch": computer.fencing_epoch})
            return lease

    def owner_disconnect(self, computer_id: str, principal: str) -> Optional[ControlLease]:
        """Transport loss: return exclusive control to the agent exactly once."""
        self.authorize_owner(principal)
        with self._lock:
            computer = self.get_computer(computer_id)
            if computer.control_authority != ControlAuthority.OWNER_CONTROLLED:
                return self.store.active_lease_for_computer(computer.id)
            self.store.revoke_leases(computer.id)
            self.store.expire_tokens_for_computer(computer.id)
            computer.resume_observe_required = True
            lease = self._issue_agent_lease(computer)
            computer.control_authority = ControlAuthority.AGENT_CONTROLLED
            computer.updated_at = _iso(self.clock())
            self.store.upsert_computer(computer)
            self._owner_transports.pop(computer.id, None)
            self._audit(computer.id, "owner_disconnect", principal, {"new_epoch": computer.fencing_epoch})
            self._audit(computer.id, "fencing_recovery", "system", {"new_epoch": computer.fencing_epoch})
            return lease

    def expire_owner_if_needed(self, computer_id: str) -> Optional[ControlLease]:
        with self._lock:
            computer = self.get_computer(computer_id)
            lease = self.store.active_lease_for_computer(computer.id)
            if not lease or lease.controller != Controller.OWNER:
                return None
            if not lease.expires_at:
                return None
            if self.clock() < datetime.fromisoformat(lease.expires_at):
                return None
            self.store.revoke_leases(computer.id)
            self.store.expire_tokens_for_computer(computer.id)
            computer.control_authority = ControlAuthority.AGENT_CONTROLLED
            computer.resume_observe_required = True
            new_lease = self._issue_agent_lease(computer)
            self.store.upsert_computer(computer)
            self._owner_transports.pop(computer.id, None)
            self._audit(computer.id, "takeover_expired", "system", {"lease_id": lease.lease_id})
            self._audit(computer.id, "fencing_recovery", "system", {"new_epoch": computer.fencing_epoch})
            return new_lease

    # ── checkpoints ──────────────────────────────────────────────────
    def approve_checkpoint(self, checkpoint_id: str, principal: str) -> Checkpoint:
        self.authorize_owner(principal)
        cp = self.store.get_checkpoint(checkpoint_id)
        if not cp:
            raise NotFoundError("checkpoint not found")
        cp.status = CheckpointStatus.APPROVED
        self.store.upsert_checkpoint(cp)
        self._audit(cp.computer_id, "checkpoint_approved", principal, {"checkpoint_id": cp.id})
        return cp

    # ── public status ────────────────────────────────────────────────
    def public_status(self, computer: AgentComputer) -> dict[str, Any]:
        lease = self.store.active_lease_for_computer(computer.id)
        identity = None
        if computer.active_browser_identity_id:
            identity = self.store.get_identity(computer.active_browser_identity_id)
        return {
            "computer_id": computer.id,
            "agent_profile_id": computer.agent_profile_id,
            "lifecycle": computer.lifecycle.value,
            "control": computer.control_authority.value,
            "fencing_epoch": computer.fencing_epoch,
            "resume_observe_required": computer.resume_observe_required,
            "workspace": {"url": computer.workspace_url, "title": computer.workspace_title},
            "browser_identity": (
                {
                    "id": identity.id,
                    "revoked": identity.revoked,
                    "locked": bool(identity.lock_computer_id),
                    "metadata": identity.metadata,
                }
                if identity
                else None
            ),
            "lease": (
                {
                    "lease_id": lease.lease_id,
                    "controller": lease.controller.value,
                    "fencing_epoch": lease.fencing_epoch,
                    "status": lease.status.value,
                    "expires_at": lease.expires_at,
                }
                if lease
                else None
            ),
        }

    def public_identity(self, identity: BrowserIdentity) -> dict[str, Any]:
        return {
            "id": identity.id,
            "ownership": identity.ownership,
            "revoked": identity.revoked,
            "locked": bool(identity.lock_computer_id),
            "lock_computer_id": identity.lock_computer_id,
            "metadata": identity.metadata,
        }

    # ── internals ────────────────────────────────────────────────────
    def _bump_epoch(self, computer: AgentComputer) -> int:
        computer.fencing_epoch += 1
        return computer.fencing_epoch

    def _issue_agent_lease(self, computer: AgentComputer) -> ControlLease:
        self.store.revoke_leases(computer.id)
        epoch = self._bump_epoch(computer)
        lease = ControlLease(
            lease_id=f"ls_{uuid.uuid4().hex}",
            computer_id=computer.id,
            controller=Controller.AGENT,
            fencing_epoch=epoch,
            acquired_at=_iso(self.clock()),
            expires_at=None,
            status=LeaseStatus.ACTIVE,
        )
        self.store.upsert_lease(lease)
        computer.control_authority = ControlAuthority.AGENT_CONTROLLED
        self.store.upsert_computer(computer)
        return lease

    def _issue_owner_lease(self, computer: AgentComputer) -> ControlLease:
        self.store.revoke_leases(computer.id)
        epoch = self._bump_epoch(computer)
        exp = self.clock() + timedelta(seconds=self.takeover_ttl_s)
        lease = ControlLease(
            lease_id=f"ls_{uuid.uuid4().hex}",
            computer_id=computer.id,
            controller=Controller.OWNER,
            fencing_epoch=epoch,
            acquired_at=_iso(self.clock()),
            expires_at=_iso(exp),
            status=LeaseStatus.ACTIVE,
        )
        self.store.upsert_lease(lease)
        computer.control_authority = ControlAuthority.OWNER_CONTROLLED
        self.store.upsert_computer(computer)
        return lease

    def _mint_takeover_token(self, computer: AgentComputer, principal: str) -> str:
        raw = secrets.token_urlsafe(32)
        exp = self.clock() + timedelta(seconds=self.takeover_ttl_s)
        rec = TakeoverToken(
            token_id=f"tt_{uuid.uuid4().hex}",
            token_hash=_hash_token(raw),
            computer_id=computer.id,
            owner_principal=principal,
            fencing_epoch=computer.fencing_epoch,
            expires_at=_iso(exp),
        )
        self.store.insert_token(rec)
        return raw

    def _require_lease(
        self,
        computer: AgentComputer,
        principal: str,
        lease_id: str,
        fencing_epoch: int,
        *,
        allow_owner: bool,
    ) -> Controller:
        self.expire_owner_if_needed(computer.id)
        computer = self.get_computer(computer.id)
        lease = self.store.get_lease(lease_id)
        if not lease or lease.computer_id != computer.id:
            self._audit(computer.id, "stale_controller_rejected", principal, {"reason": "unknown_lease"})
            raise StaleControllerError("unknown lease")
        if lease.status != LeaseStatus.ACTIVE:
            self._audit(computer.id, "stale_controller_rejected", principal, {"reason": "revoked_lease"})
            raise StaleControllerError("lease is not active")
        if lease.fencing_epoch != computer.fencing_epoch or lease.fencing_epoch != fencing_epoch:
            self._audit(
                computer.id,
                "stale_controller_rejected",
                principal,
                {"presented": fencing_epoch, "current": computer.fencing_epoch},
            )
            raise StaleControllerError("stale fencing epoch")
        if lease.controller == Controller.OWNER:
            if not allow_owner or not is_owner_principal(principal):
                raise ForbiddenError("owner lease required")
            if computer.control_authority != ControlAuthority.OWNER_CONTROLLED:
                raise StaleControllerError("owner is not the authority")
            return Controller.OWNER
        if not is_agent_principal(principal, computer.agent_profile_id):
            raise ForbiddenError("agent lease required")
        if computer.control_authority != ControlAuthority.AGENT_CONTROLLED:
            self._audit(computer.id, "stale_controller_rejected", principal, {"reason": "agent_during_owner"})
            raise StaleControllerError("agent is not the authority")
        return Controller.AGENT

    def _audit(self, computer_id: str | None, event_type: str, actor: str, detail: dict[str, Any]) -> None:
        redact = {
            "cookie",
            "cookies",
            "token",
            "takeover_token",
            "password",
            "secret",
            "text",
            "value",
            "typed",
            "input",
            "content",
            "key",
            "keys",
            "code",
            "payload",
            "message_text",
        }
        safe = {k: v for k, v in detail.items() if str(k).lower() not in redact}
        self.store.append_audit(
            AuditEvent(
                event_type=event_type,
                computer_id=computer_id,
                actor=actor,
                detail=safe,
                created_at=_iso(self.clock()),
            )
        )
