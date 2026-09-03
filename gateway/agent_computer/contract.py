"""Public client contract for Agent Computers.

Future clients (mobile / Mac / dashboard) consume this shape only.
Raw CDP URLs, container/VM IDs, cookie jars, user-data-dir paths,
process IDs, and auth blobs are never part of the public payload.
"""

from __future__ import annotations

from typing import Any

from .errors import AgentComputerError
from .models import OWNER_PRINCIPAL, agent_principal
from .pointer import mapping_kind
from .service import AgentComputerService


def live_view_public(*, headed_same_host: bool = False) -> dict[str, Any]:
    return {
        "kind": "screenshot_on_demand",
        "same_environment": True,
        "remote_stream": False,
        "headed_same_host": bool(headed_same_host),
    }

FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "cdp",
        "cdp_url",
        "cdp_loopback",
        "devtools",
        "cookie",
        "cookies",
        "auth_blob",
        "user_data_dir",
        "profile_ref",
        "persistence_ref",
        "process_id",
        "pid",
        "token_hash",
        "password",
        "secret",
        "executable",
        "binary",
        "devtoolsactiveport",
    }
)


def sanitize_public(value: Any) -> Any:
    """Recursively drop host-private fields from a payload."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_PUBLIC_KEYS:
                continue
            out[key] = sanitize_public(item)
        return out
    if isinstance(value, list):
        return [sanitize_public(item) for item in value]
    return value


def error_payload(exc: AgentComputerError) -> dict[str, Any]:
    return sanitize_public(
        {
            "ok": False,
            "error": exc.code,
            "message": str(exc),
            "details": exc.details,
        }
    )


def owner_principal() -> str:
    return OWNER_PRINCIPAL


def agent_from_profile(profile_id: str) -> str:
    return agent_principal(profile_id)


class AgentComputerContract:
    """Single public facade. Adapters must not bypass this."""

    def __init__(self, service: AgentComputerService):
        self.service = service

    def ensure(self, profile_id: str, principal: str) -> dict[str, Any]:
        computer = self.service.ensure_computer(profile_id)
        self.service.authorize_read(computer, principal)
        return sanitize_public(self.service.public_status(computer))

    def status(self, computer_id: str, principal: str) -> dict[str, Any]:
        computer = self.service.get_computer(computer_id)
        self.service.authorize_read(computer, principal)
        return sanitize_public(self.service.public_status(computer))

    def list_computers(self, principal: str) -> dict[str, Any]:
        items = []
        for computer in self.service.list_computers():
            try:
                self.service.authorize_read(computer, principal)
            except Exception:
                continue
            items.append(self.service.public_status(computer))
        return sanitize_public({"computers": items})

    def create_identity(
        self,
        principal: str,
        *,
        ownership: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.service.authorize_owner(principal)
        identity = self.service.create_identity(ownership=ownership, metadata=metadata)
        return sanitize_public({"identity": self.service.public_identity(identity)})

    def attach_identity(
        self, computer_id: str, identity_id: str, principal: str
    ) -> dict[str, Any]:
        computer = self.service.attach_identity(computer_id, identity_id, principal)
        return sanitize_public(self.service.public_status(computer))

    def detach_identity(self, computer_id: str, principal: str) -> dict[str, Any]:
        computer = self.service.detach_identity(computer_id, principal)
        return sanitize_public(self.service.public_status(computer))

    def revoke_identity(self, identity_id: str, principal: str) -> dict[str, Any]:
        identity = self.service.revoke_identity(identity_id, principal)
        return sanitize_public({"identity": self.service.public_identity(identity)})

    def wake(self, computer_id: str, principal: str) -> dict[str, Any]:
        computer, lease = self.service.wake(computer_id, principal)
        payload = self.service.public_status(computer)
        payload["lease_id"] = lease.lease_id
        payload["fencing_epoch"] = lease.fencing_epoch
        return sanitize_public(payload)

    def sleep(self, computer_id: str, principal: str) -> dict[str, Any]:
        computer = self.service.sleep(computer_id, principal)
        return sanitize_public(self.service.public_status(computer))

    def observe(
        self, computer_id: str, principal: str, *, lease_id: str, fencing_epoch: int
    ) -> dict[str, Any]:
        obs = self.service.observe(
            computer_id, principal, lease_id=lease_id, fencing_epoch=fencing_epoch
        )
        screenshot = None
        if obs.screenshot_b64:
            screenshot = {
                "mime": obs.screenshot_mime or "image/jpeg",
                "data": obs.screenshot_b64,
                "width": obs.screenshot_width,
                "height": obs.screenshot_height,
            }
        viewport = {
            "width": obs.viewport_width,
            "height": obs.viewport_height,
        }
        view = live_view_public(headed_same_host=False)
        view["mapping"] = mapping_kind(
            obs.screenshot_width,
            obs.screenshot_height,
            obs.viewport_width,
            obs.viewport_height,
        )
        return sanitize_public(
            {
                "url": obs.url,
                "title": obs.title,
                "text": obs.text,
                "fencing_epoch": obs.fencing_epoch,
                "controller": obs.controller,
                "observed_at": obs.observed_at,
                "screenshot": screenshot,
                "viewport": viewport,
                "live_view": view,
            }
        )

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
    ) -> dict[str, Any]:
        if kind == "set_cookie":
            raise AgentComputerError("cookie mutation is not a public action")
        receipt = self.service.act(
            computer_id,
            principal,
            lease_id=lease_id,
            fencing_epoch=fencing_epoch,
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
        return sanitize_public(
            {
                "accepted": receipt.accepted,
                "fencing_epoch": receipt.fencing_epoch,
                "url": receipt.url,
                "title": receipt.title,
                "text": receipt.text,
            }
        )

    def request_takeover(
        self, computer_id: str, principal: str, *, reason: str = ""
    ) -> dict[str, Any]:
        raw = self.service.request_takeover(computer_id, principal, reason=reason)
        # Token is owner-only and returned once. Still strip every other secret.
        token = raw.get("takeover_token")
        payload = sanitize_public({k: v for k, v in raw.items() if k != "takeover_token"})
        payload["takeover_token"] = token
        payload["live_view"] = live_view_public(headed_same_host=False)
        return payload

    def connect_takeover(
        self, computer_id: str, principal: str, *, takeover_token: str
    ) -> dict[str, Any]:
        lease = self.service.connect_takeover(
            computer_id, principal, takeover_token=takeover_token
        )
        return sanitize_public(
            {
                "lease_id": lease.lease_id,
                "fencing_epoch": lease.fencing_epoch,
                "controller": lease.controller.value,
                "expires_at": lease.expires_at,
            }
        )

    def give_back(
        self, computer_id: str, principal: str, *, lease_id: str, fencing_epoch: int
    ) -> dict[str, Any]:
        lease = self.service.give_back(
            computer_id, principal, lease_id=lease_id, fencing_epoch=fencing_epoch
        )
        computer = self.service.get_computer(computer_id)
        payload = self.service.public_status(computer)
        payload["agent_lease_id"] = lease.lease_id
        return sanitize_public(payload)

    def owner_disconnect(self, computer_id: str, principal: str) -> dict[str, Any]:
        lease = self.service.owner_disconnect(computer_id, principal)
        computer = self.service.get_computer(computer_id)
        payload = self.service.public_status(computer)
        if lease:
            payload["agent_lease_id"] = lease.lease_id
        return sanitize_public(payload)

    def approve_checkpoint(self, checkpoint_id: str, principal: str) -> dict[str, Any]:
        cp = self.service.approve_checkpoint(checkpoint_id, principal)
        return sanitize_public(
            {
                "checkpoint_id": cp.id,
                "status": cp.status.value,
                "action_class": cp.action_class,
            }
        )
