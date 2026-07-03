"""Router — the glue: inbound message -> tenant lookup -> worker -> reply.

This is where the single bot's traffic fans out to per-employee isolated agents.
Unknown senders are refused, unless `security.auto_provision` is on — then the
first message from a new platform user auto-creates their isolated profile.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re

from .config import Settings
from .ingress.base import Ingress
from .models import Employee, InboundMessage
from .provisioner import scaffold_home
from .registry import Registry
from .supervisor import CapacityFull, Supervisor

log = logging.getLogger("orchard.router")


class Router:
    def __init__(self, settings: Settings, registry: Registry,
                 supervisor: Supervisor, ingress: Ingress):
        self.settings = settings
        self.registry = registry
        self.supervisor = supervisor
        self.ingress = ingress
        self._prov_lock = asyncio.Lock()  # serialize auto-provisioning

    async def on_message(self, msg: InboundMessage) -> None:
        emp = self.registry.by_mm_user(msg.sender_id)
        if emp is None:
            if self.settings.security.auto_provision:
                emp = await self._auto_provision(msg)
            elif self.settings.security.require_provisioned:
                log.warning("rejecting unprovisioned sender %s", msg.sender_id)
                await self.ingress.post(
                    msg.channel_id,
                    "⛔ You don't have an assistant workspace yet. Contact IT to get provisioned.",
                    msg.thread_id,
                )
                return
            else:
                log.warning("sender %s not provisioned; ignoring", msg.sender_id)
                return

        await self.ingress.typing(msg.channel_id)
        try:
            reply = await self.supervisor.handle(emp, msg.session_name(), msg.text)
        except CapacityFull:
            reply = "⏳ The assistant is at capacity right now — please resend in a moment."
        except Exception as e:  # keep the bot alive; surface a safe message
            log.exception("handling message for %s failed", emp.id)
            reply = f"⚠️ Sorry, something went wrong ({type(e).__name__})."
        await self.ingress.post(msg.channel_id, reply, msg.thread_id)

    # --- auto-onboarding -----------------------------------------------------
    async def _auto_provision(self, msg: InboundMessage) -> Employee:
        """Create an isolated profile for a first-time sender. Serialized so two
        concurrent first messages don't double-provision."""
        async with self._prov_lock:
            existing = self.registry.by_mm_user(msg.sender_id)  # re-check under lock
            if existing:
                return existing
            eid = self._employee_id_for(msg.sender_id)
            name = (msg.sender_name or msg.sender_id).lstrip("@") or eid
            emp = self.registry.add(eid, name, msg.sender_id)
            scaffold_home(self.settings, emp)
            log.info("auto-provisioned '%s' for platform user %s", eid, msg.sender_id)
            return emp

    def _employee_id_for(self, sender_id: str) -> str:
        """Derive a valid, unique profile id from a platform user id."""
        base = re.sub(r"[^a-z0-9_-]", "-", sender_id.lower()).strip("-")[:48]
        if not base or not re.match(r"[a-z0-9]", base[0]):
            base = "u" + base
        if self.registry.get(base) is None:
            return base
        # collision — append a short stable hash of the raw sender id
        return f"{base[:40]}-{hashlib.sha1(sender_id.encode()).hexdigest()[:6]}"

    async def serve(self) -> None:
        await self.supervisor.start()
        log.info("router serving via %s ingress", type(self.ingress).__name__)
        try:
            await self.ingress.run(self.on_message)
        finally:
            await self.supervisor.stop()
