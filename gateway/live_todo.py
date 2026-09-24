"""Process-local source and admission contract for native live-todo consumers.

No presentation or task state authority lives here. Runs never resume a projection.
An ambiguous dispatched attempt quarantines its surface for this process lifetime.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import logging
import threading
from uuid import uuid4


class DeliveryStatus(str, Enum):
    DELIVERED = "delivered"
    SKIPPED = "skipped"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DeliveryOutcome:
    status: DeliveryStatus
    message_id: str | None = None
    reason: str = ""
    retry_after: float | None = None


def rendered_payload_hash(text: str, rows=()) -> str:
    """Digest the exact bounded text and normalized inline-button payload.

    A confirmed message may satisfy a newer source revision only when this
    complete wire-visible payload is byte-equivalent.  Resource, route and
    destination identity remain the owning receipt/source's responsibility.
    """
    payload = {"text": text, "rows": rows}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TodoBinding:
    profile: str
    transport_profile: str
    profile_home: str
    session_id: str
    run_generation: int
    chat_id: str
    thread_id: str | None
    store_incarnation: str = ""
    platform: str = "telegram"

    @property
    def surface(self):
        return (self.profile_home, self.transport_profile, self.platform,
                self.chat_id, self.thread_id)


@dataclass(frozen=True)
class TodoEvent:
    binding: TodoBinding
    revision: int
    todos: tuple


# Only settlement/fencing metadata, not a second task engine or durable bus.
# Shared across registrations and adapter replacements, preventing successor overlap.
_surfaces = {}
_surfaces_lock = threading.RLock()
# Active/unsettled sources reserve a slot too: uncertainty can never overflow
# the bound at settlement. Never evict an unknown to admit a fresh create.
# Saturation disables new surfaces process-wide until known slots are released;
# if all slots are unknown, admission stays closed for this process lifetime.
_MAX_SURFACES = 4096
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoppedRegistration:
    token: str
    active: bool = False
    sources: frozenset = frozenset()


class TodoRegistration:
    def __init__(self, factory, profile_home, scope):
        self.factory = factory
        self.plugin_id = ""
        self.profile_home = str(profile_home)
        self.scope = scope
        self.token = uuid4().hex
        self.active = True
        self.sources = set()
        self.lock = threading.RLock()

    def close(self):
        """Effective stop: revoke admission synchronously, then wake consumers."""
        with self.lock:
            self.active = False
            pending = []
            for source in tuple(self.sources):
                with source.lock:
                    source.active = False
                    pending.append((source, source.loop))
        # Fence every source before optional code. Never block settlement on an
        # extension close waiting for its task's done callback on another thread.
        for source, loop in pending:
            try:
                source.close()
            finally:
                if loop is not None and not loop.is_closed():
                    def drain(s=source, owner_loop=loop):
                        owner_loop.create_task(s.finish())
                    try:
                        loop.call_soon_threadsafe(drain)
                    except RuntimeError:
                        # Shutdown may close the loop after the check. Admission
                        # is already fenced; no coroutine was created or leaked.
                        if not loop.is_closed():
                            logger.exception("Unable to schedule live todo drain")

    def open(self, adapter, binding, is_current, store):
        with self.lock:
            if not self.active or not is_current() or binding.profile_home != self.profile_home:
                return None
            if getattr(adapter, "live_todo_transport", None) != 1:
                return None
            if adapter._bot is None or getattr(adapter, "_send_path_degraded", False):
                return None
            if getattr(adapter, "_live_todo_client", None) is not adapter._bot:
                # A rebuilt client is a new transport epoch even on the same adapter.
                adapter._live_todo_epoch = uuid4().hex
                adapter._live_todo_client = adapter._bot
            with _surfaces_lock:
                previous = _surfaces.get(binding.surface)
                if previous is not None:
                    same_run = (previous.binding.session_id == binding.session_id
                                and previous.binding.run_generation == binding.run_generation)
                    if (previous.active and previous.registration is self and same_run
                            and previous.adapter is adapter and previous.client is adapter._bot):
                        return previous
                    if (previous.binding.session_id == binding.session_id
                            and previous.binding.run_generation > binding.run_generation):
                        return None
                    previous.close()
                    if previous.inflight or previous.unknown:
                        return None
                if previous is None and len(_surfaces) >= _MAX_SURFACES:
                    logger.warning("Live todo surface capacity exhausted; refusing new surface")
                    return None
                try:
                    source = TodoSource(self, adapter, binding, is_current, store)
                except Exception:
                    logger.exception("Optional live todo factory failed; continuing without presentation")
                    return None
                _surfaces[binding.surface] = source
                self.sources.add(source)
                return source


class TodoDeliveryHandle:
    """Supported plugin surface: admission and route-bound delivery, no SDK client."""
    __slots__ = ("__source",)

    def __init__(self, source):
        self.__source = source

    def admitted(self):
        return self.__source.admitted()

    async def deliver(self, text):
        return await self.__source.deliver(text)


class TodoSource:
    def __init__(self, registration, adapter, binding, is_current, store):
        self.registration = registration
        self.adapter = adapter
        self.binding = binding
        self.is_current = is_current
        self.store = store
        self.client = adapter._bot
        self.epoch = adapter._live_todo_epoch
        self.active = True
        self.inflight = False
        self.unknown = False
        self.revision = -1
        self.message_id = None
        self.last_outcome = None
        self.lock = threading.RLock()
        self.loop = asyncio.get_running_loop()
        self.task = None
        self.consumer = None
        try:
            self.consumer = registration.factory(TodoDeliveryHandle(self))
        except BaseException:
            # Even a factory that retained the handle cannot keep an admitted writer.
            self.active = False
            self.adapter = self.client = self.store = self.is_current = None
            raise
        adapter._live_todo_sources.add(self)

    def admitted(self):
        if not self.active:
            return False
        with self.registration.lock, self.lock:
            store = self.store()
            return bool(self.active and self.registration.active and not self.unknown
                        and self.registration.scope.allows_route(
                            self.binding.profile, self.binding.platform, self.binding.chat_id,
                            self.binding.thread_id)
                        and self.is_current() and self.adapter._bot is self.client
                        and self.adapter._live_todo_epoch == self.epoch
                        and store is not None and self.binding.store_incarnation
                        and store.incarnation == self.binding.store_incarnation)

    def publish(self, snapshot, incarnation):
        """Called only from post-commit executor data; never parses result strings."""
        if not self.active:
            return False
        with self.registration.lock, self.lock:
            if (not self.active or not self.registration.active or not self.is_current()
                    or not isinstance(snapshot, dict) or not incarnation):
                return False
            revision, todos = snapshot.get("revision"), snapshot.get("todos")
            if (type(revision) is not int or revision < 0 or revision <= self.revision
                    or not isinstance(todos, list) or len(todos) > 256
                    or any(not isinstance(item, dict) for item in todos)):
                return False
            store = self.store()
            if store is None or incarnation != store.incarnation:
                return False
            if self.binding.store_incarnation and self.binding.store_incarnation != incarnation:
                return False
            if not self.binding.store_incarnation:
                self.binding = replace(self.binding, store_incarnation=incarnation)
            if not self.admitted():
                return False
            self.revision = revision
            event = TodoEvent(self.binding, revision, tuple(dict(item) for item in todos))
            self.consumer.publish(event)
            return True

    async def deliver(self, text):
        # Message identity belongs to this bound surface, never to plugin-supplied routes.
        if not self.admitted():
            return DeliveryOutcome(DeliveryStatus.REJECTED, reason="stopped source")
        try:
            return await self.adapter.deliver_live_todo(self, text)
        finally:
            self._settled()

    async def run(self):
        self.task = asyncio.current_task()
        self.task.add_done_callback(self._settled)
        try:
            await self.consumer.run()
        except Exception:
            logger.exception("Optional live todo consumer failed")
        finally:
            self.close()

    def close(self):
        with self.lock:
            self.active = False
            consumer = self.consumer
        if consumer is not None:
            try:
                consumer.close()
            except Exception:
                logger.exception("Optional live todo close failed after host fencing")

    def _settled(self, task=None):
        """Detach only after the exact task has terminated, never on cancel request.

        Keep this source's identity for receipt observers, but unknown quarantine
        roots only immutable binding/outcome/token plus primitive fencing fields.
        No consumer, turn closure, old SDK client or completed Task is retained.
        """
        registration = self.registration
        if isinstance(registration, StoppedRegistration):
            return
        # Same order as open (registration -> surfaces -> source), never the
        # reverse. No optional extension code runs under these settlement locks.
        with registration.lock, _surfaces_lock, self.lock:
            if self.active or self.inflight or (self.task is not None and not self.task.done()):
                return
            if self.adapter is None:
                return
            registration.sources.discard(self)
            self.adapter._live_todo_sources.discard(self)
            if not self.unknown and _surfaces.get(self.binding.surface) is self:
                _surfaces.pop(self.binding.surface, None)
            if self.unknown:
                self.registration = StoppedRegistration(self.registration.token)
                self.adapter = self.client = self.consumer = None
                self.store = self.is_current = self.task = self.loop = None

    async def finish(self):
        self.close()  # stop admission BEFORE awaiting anything
        task = self.task
        if task is not None and task is not asyncio.current_task() and not task.done():
            done, _ = await asyncio.wait({task}, timeout=2.0)
            if not done:
                if self.inflight:
                    self.unknown = True
                    self.last_outcome = DeliveryOutcome(DeliveryStatus.UNKNOWN, reason="stop during dispatch")
                task.cancel()
                # Cancellation is not remote retraction. Quarantine above survives this task.
                await asyncio.wait({task}, timeout=0.1)
        self._settled()


def register_live_todo(ctx, factory, *, scope):
    if not callable(factory):
        raise ValueError("live todo factory must be callable")
    from gateway.surface_scope import parse_surface_scope
    parsed_scope = parse_surface_scope(scope, require_tasks=False)
    manager = ctx._manager
    existing = getattr(manager, "_live_todo_registration", None)
    if existing is not None and existing.active:
        # One presenter per profile, idempotent re-registration by the same plugin.
        if existing.plugin_id == ctx.plugin_id:
            if existing.scope != parsed_scope:
                raise ValueError("live todo scope is already bound for this profile")
            return existing
        raise ValueError("a live todo consumer is already registered for this profile")
    registration = TodoRegistration(factory, manager.home_path, parsed_scope)
    registration.plugin_id = ctx.plugin_id
    manager._live_todo_registration = registration
    ctx.on_unload(registration.close)
    return registration


def route_bound_live_source(*, profile_home, profile, chat_id, thread_id, bot_id):
    """Return the one admitted live Telegram source for an exact route.

    This is a process-local transport bridge for another host-owned surface;
    callers receive no credential or adapter lookup by caller-supplied route.
    Ambiguity and stale sources fail closed.
    """
    matches = []
    with _surfaces_lock:
        for source in _surfaces.values():
            binding = source.binding
            bot = getattr(source.client, "id", None)
            if (binding.profile_home == str(profile_home)
                    and binding.profile == profile and binding.platform == "telegram"
                    and binding.chat_id == str(chat_id)
                    and binding.thread_id == (str(thread_id) if thread_id else None)
                    and bot == bot_id and source.admitted()):
                matches.append(source)
    return matches[0] if len(matches) == 1 else None


def open_live_todo(runner, disp, ctx):
    from hermes_cli.plugins import get_plugin_manager
    from gateway.session_identity import identity_of

    if (getattr(ctx.source.platform, "value", ctx.source.platform) != "telegram"
            or ctx.scheduled_heartbeat or ctx.mute_notification_reply
            or not ctx.session_id or ctx.run_generation is None
            or not ctx.source.chat_id
            or (getattr(disp, "_tool_progress_explicit", False) and disp.progress_mode == "off")):
        return None
    manager = get_plugin_manager()
    registration = getattr(manager, "_live_todo_registration", None)
    if registration is None or not registration.active:
        return None
    source = ctx.source
    identity = identity_of(source)
    profile = identity.runtime_profile if identity else str(source.profile or "default")
    if not registration.scope.allows_route(
        profile, source.platform, source.chat_id, source.thread_id,
    ):
        return None
    adapter = runner._delivery_adapter_for(source)
    if adapter is None or getattr(adapter, "live_todo_transport", None) != 1:
        return None
    transport_profile = identity.transport_profile if identity else str(adapter._owner_profile or "default")
    binding = TodoBinding(profile, transport_profile, str(manager.home_path), str(ctx.session_id),
                          int(ctx.run_generation), str(source.chat_id),
                          str(source.thread_id) if source.thread_id is not None else None)

    def store():
        return getattr(ctx.agent_holder[0], "_todo_store", None) if ctx.agent_holder else None

    owner = registration.open(adapter, binding, ctx._run_still_current, store)
    ctx._todo_progress_owner = owner
    return owner


async def finish_live_todo(ctx):
    owner = getattr(ctx, "_todo_progress_owner", None)
    try:
        if owner is not None:
            owner.close()  # host fence is synchronous, before optional draining
            await owner.finish()
    finally:
        ctx._todo_progress_task = None
