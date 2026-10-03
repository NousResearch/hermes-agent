"""Ephemeral, authorized gateway requests and their bound interim-delivery capability.

Policy lives in profile plugins. These helpers are called only AFTER gateway authorization;
no payload metadata can supply a request handle or delivery target. See plugin hooks docs.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from agent.async_utils import consume_detached_task_result
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)
HOOKS = ("gateway_request_lifecycle", "gateway_request_control", "gateway_request_final", "gateway_request_tool")
TERMINAL = frozenset({"final_ready", "completed", "failed", "cancelled", "superseded", "merged", "rejected"})


@dataclass(frozen=True)
class RequestFacts:
    request_id: str
    session_key: str
    profile_home: str
    runtime_profile: str
    transport_profile: str
    platform: str
    chat_id: str
    thread_id: str | None
    requester_id: str
    message_id: str | None
    text: str
    admitted_at: float


@dataclass(frozen=True)
class RequestControl:
    text: str
    message_id: str | None
    reply_to_message_id: str | None
    requester_id: str


@dataclass(frozen=True)
class InterimReceipt:
    initiated_at: float | None
    delivered_at: float | None
    success: bool


class RequestContext:
    """A runtime-owned request, with plugin-owned ephemeral ``state``.

    ``send_interim`` returns transport success, not merely admission. It cannot start a send
    after final readiness/cancellation. A send already accepted by transport cannot be recalled.
    ``run_generation`` is None while queued; admission time includes that wait.
    """
    def __init__(self, runner, event, facts, adapter, loop):
        self.facts = facts
        self.state: dict[str, Any] = {}
        self.run_generation: int | None = None
        self.stage = "admitted"
        self._runner = runner
        self._event = event
        from gateway.session_identity import replace_source
        self._source = replace_source(event.source)
        self.interim_receipts: list[InterimReceipt] = []
        self.final_ready_at: float | None = None
        self.final_receipts: list[InterimReceipt] = []
        self._adapter = adapter
        self._loop = loop
        self._lock = threading.RLock()
        self._sends: set[asyncio.Task] = set()
        self._registry = _requests(runner)
        self._reply_to = runner._reply_anchor_for_event(event)
        self._metadata = dict(runner._thread_metadata_for_source(event.source, self._reply_to) or {})
        self._metadata["_interim_send"] = True

    async def _send(self, content):
        success = False
        delivered = None
        initiated = None
        try:
            with _scope(self._runner, self._source):
                # No loop yield between this grant and the adapter invocation. Final readiness
                # arbitrates under the same lock; it can cancel pending tasks, not recall a grant.
                with self._lock:
                    if not self.active or (self.run_generation is not None and not self._runner._is_session_run_current(
                        self.facts.session_key, self.run_generation
                    )):
                        return None
                    initiated = time.monotonic()
                    delivery = self._adapter.send(
                        self.facts.chat_id, content, reply_to=self._reply_to, metadata=dict(self._metadata))
                result = await delivery
            success = getattr(result, "success", False) is True
            if success:
                delivered = time.monotonic()
            return result
        finally:
            if initiated is not None:
                self.interim_receipts.append(InterimReceipt(initiated, delivered, success))

    @property
    def active(self):
        return self.stage not in TERMINAL

    async def send_interim(self, content: str) -> bool:
        # Execute the guard in the originating loop. Plugins can schedule timers there at admission.
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("request interim delivery must run on its admission loop")
        with self._lock:
            if not self.active or not content:
                return False
            if self.run_generation is not None and not self._runner._is_session_run_current(
                self.facts.session_key, self.run_generation
            ):
                return False
            task = self._loop.create_task(self._send(content))
            self._sends.add(task)
        try:
            result = await task
            return getattr(result, "success", False) is True
        except asyncio.CancelledError:
            if not self.active:
                return False
            raise
        except Exception:
            logger.debug("Request interim transport failed", exc_info=True)
            return False
        finally:
            self._sends.discard(task)


class _RequestRegistry(dict):
    """Per-runner live handles only; terminal events keep their own settled handle."""


def _requests(runner):
    registry = getattr(runner, "_request_lifecycle", None)
    if registry is None:
        registry = _RequestRegistry()
        runner._request_lifecycle = registry
    return registry


def _request(event):
    value = getattr(event, "_gateway_request_context", None)
    return value if isinstance(value, RequestContext) else None


def _scope(runner, source):
    return runner._profile_scope_for_source(source)


async def _notify(request, stage):
    from hermes_cli.plugins import ainvoke_hook
    with _scope(request._runner, request._source):
        await ainvoke_hook("gateway_request_lifecycle", request=request, stage=stage, turn_id=uuid.uuid4().hex)


def _schedule_notify(request, stage):
    def schedule():
        task = request._loop.create_task(_notify(request, stage))
        # Observe plugin dispatcher failures; individual callback exceptions are already isolated.
        task.add_done_callback(consume_detached_task_result)
    if not request._loop.is_closed():
        request._loop.call_soon_threadsafe(schedule)


def capability_enabled(runner, source):
    from hermes_cli.plugins import has_hook
    with _scope(runner, source):
        return any(has_hook(hook) for hook in HOOKS)


async def admit_request(runner, event, session_key):
    """Reuse an event's admission; return None for internal events or profiles without a consumer."""
    existing = _request(event)
    if existing is not None:
        return existing
    source = event.source
    if event.internal or getattr(source, "profile_route_rejected", False) is True or not source.user_id:
        return None
    admitted_at = time.monotonic()
    with _scope(runner, source):
        from hermes_cli.plugins import has_hook, ainvoke_hook
        if not any(has_hook(hook) for hook in HOOKS):
            return None
        from gateway.session_identity import identity_of
        identity = identity_of(source)
        home = str(identity.runtime_home if identity else get_hermes_home())
        runtime_profile = identity.runtime_profile if identity else (source.profile or "default")
        transport_profile = identity.transport_profile if identity else runtime_profile
        adapter = runner._delivery_adapter_for(source)
        if adapter is None:
            return None
        facts = RequestFacts(
            request_id=uuid.uuid4().hex, session_key=session_key, profile_home=home,
            runtime_profile=runtime_profile, transport_profile=transport_profile,
            platform=source.platform.value, chat_id=source.chat_id, thread_id=source.thread_id,
            requester_id=source.user_id, message_id=event.message_id,
            text=event.text or "", admitted_at=admitted_at)
        request = RequestContext(runner, event, facts, adapter, asyncio.get_running_loop())
        event._gateway_request_context = request
        _requests(runner)[facts.request_id] = request
        await ainvoke_hook("gateway_request_lifecycle", request=request, stage="admitted", turn_id=facts.request_id)
        return request


async def consume_control(runner, event, session_key):
    """Pure plugin controls, AFTER correlated approval/clarification, BEFORE either busy guard.

    Plugins may return ``{request_id, state}``; runtime rechecks the live matching owner before
    updating state. Text classification is entirely plugin policy. No new request is admitted.
    """
    if event.internal or not event.allow_gateway_control or event.get_command() or event.media_urls:
        return False
    source = event.source
    registry = getattr(runner, "_request_lifecycle", {})
    if not registry:
        return False
    from hermes_cli.plugins import has_hook, ainvoke_hook
    from gateway.session_identity import identity_of
    with _scope(runner, source):
        if not has_hook("gateway_request_control"):
            return False
        identity = identity_of(source)
        home = str(identity.runtime_home if identity else get_hermes_home())
        profile = identity.runtime_profile if identity else (source.profile or "default")
        adapter = runner._delivery_adapter_for(source)
        candidates = tuple(request for request in registry.copy().values() if request.active
            and request.facts.session_key == session_key and request.facts.profile_home == home
            and request.facts.runtime_profile == profile and request._adapter is adapter
            and request.facts.platform == source.platform.value
            and request.facts.chat_id == source.chat_id and request.facts.thread_id == source.thread_id
            and request.facts.requester_id == source.user_id)
        if not candidates:
            return False
        control = RequestControl(event.text or "", event.message_id, event.reply_to_message_id, source.user_id)
        results = await ainvoke_hook("gateway_request_control", control=control, requests=candidates,
                                    turn_id=uuid.uuid4().hex)
        for result in results:
            if not isinstance(result, dict):
                continue
            target = next((r for r in candidates if r.facts.request_id == result.get("request_id")), None)
            if target is None or not isinstance(result.get("state", {}), dict):
                continue
            with target._lock:
                if not target.active:
                    continue
                target.state.update(result.get("state", {}))
            await _notify(target, "control")
            event._gateway_accepted = True
            return True
    return False


def begin_request(runner, event, generation):
    request = _request(event)
    if request is None or not request.active:
        return request
    request.run_generation = generation
    request.stage = "running"
    _schedule_notify(request, "running")
    return request


def request_for_run(runner, session_key, generation):
    return next((r for r in getattr(runner, "_request_lifecycle", {}).copy().values()
        if r.active and r.facts.session_key == session_key and r.run_generation == generation), None)


def finish_request(request, stage="completed"):
    if request is None:
        return
    with request._lock:
        if not request.active:
            return
        request.stage = stage
        if stage == "final_ready":
            request.final_ready_at = time.monotonic()
        request._registry.pop(request.facts.request_id, None)
        # Cancellation is marshalled to the loop; the synchronous guard is already closed.
        for task in tuple(request._sends):
            if not request._loop.is_closed():
                request._loop.call_soon_threadsafe(task.cancel)
    _schedule_notify(request, stage)


def finish_event(event, stage="completed"):
    finish_request(_request(event), stage)


def finish_session(runner, session_key, *, stage="cancelled", generation=None, include_queued=False):
    for request in tuple(getattr(runner, "_request_lifecycle", {}).copy().values()):
        if request.facts.session_key == session_key and (
            include_queued or request.run_generation is not None
        ) and (generation is None or request.run_generation == generation):
            finish_request(request, stage)


def fold_request(runner, event, session_key):
    """Successful steer/redirect supersedes previous narration and owns the current run."""
    request = _request(event)
    if request is None:
        return
    generation = runner._current_session_run_generation(session_key)
    finish_session(runner, session_key, stage="superseded", generation=generation)
    begin_request(runner, event, generation)


def final_response(runner, session_key, generation, response, *, failure=None):
    """Latest plugin state at the output boundary; no model/tools/prompt/history mutation."""
    request = request_for_run(runner, session_key, generation)
    if request is None:
        return response
    if failure is not None:
        request.state["failure"] = failure
    from hermes_cli.plugins import invoke_hook
    with _scope(runner, request._source):
        for value in invoke_hook("gateway_request_final", request=request, response=response,
                                 turn_id=request.facts.request_id):
            if isinstance(value, str):
                response = value
    # The adapter receives the opening event even when the recursive queue drain answered a
    # later one. Carry the final's owning handle separately from the admission handle.
    state = runner._peek_session_state(session_key)
    delivery_event = getattr(getattr(state, "turn", None), "event", None)
    if delivery_event is not None:
        delivery_event._gateway_final_request_context = request
    request._event._gateway_final_request_context = request
    finish_request(request, "final_ready")
    return response


def reconcile_merge(existing, incoming, *, merged):
    """Only the retained queue event owns narration after an absorption or replacement."""
    if existing is incoming:
        return
    if not merged:
        finish_event(existing, "superseded")
        return
    old, new = _request(existing), _request(incoming)
    finish_request(old, "superseded")
    if new is not None and new.active:
        from dataclasses import replace
        # The retained event has changed; the newer request owns the revised task. Preserve the
        # earliest admission clock only when both events belong to the same trusted requester.
        same_owner = old is not None and (
            old.facts.profile_home, old.facts.runtime_profile, old._adapter, old.facts.chat_id,
            old.facts.thread_id, old.facts.requester_id
        ) == (
            new.facts.profile_home, new.facts.runtime_profile, new._adapter, new.facts.chat_id,
            new.facts.thread_id, new.facts.requester_id
        )
        if old is not None and not same_owner:
            finish_request(new, "merged")
            return
        new.facts = replace(new.facts, text=existing.text or "", admitted_at=(
            min(old.facts.admitted_at, new.facts.admitted_at) if same_owner else new.facts.admitted_at))
        existing._gateway_request_context = new
        _schedule_notify(new, "revised")


def final_delivery_context(event):
    value = getattr(event, "_gateway_final_request_context", None)
    return value if isinstance(value, RequestContext) else _request(event)


def record_final_delivery(request, *, success, initiated_at=None):
    """Actual transport receipt; initiation is unknown for a previously confirmed stream."""
    if request is None:
        return
    delivered = time.monotonic() if success is True else None
    request.final_receipts.append(InterimReceipt(initiated_at, delivered, success is True))
    _schedule_notify(request, "delivered" if success is True else "delivery_failed")


def settle_unqueued_event(runner, event):
    request = _request(event)
    if request is None or not request.active or request.run_generation is not None:
        return
    pending = tuple(getattr(request._adapter, "_pending_messages", {}).values())
    overflow = runner._overflow_queue(request.facts.session_key) or ()
    if not any(_request(item) is request for item in (*pending, *overflow)):
        finish_request(request, "completed")


def transfer_rewritten_event(original, rewritten):
    request = _request(original)
    if request is None:
        return
    from dataclasses import replace
    rewritten._gateway_request_context = request
    request._event = rewritten
    request.facts = replace(request.facts, text=rewritten.text or "")
    _schedule_notify(request, "revised")


def separate_queue_owners(pending_messages, session_key, incoming):
    """Keep subscribed, distinct trusted owners in the existing FIFO before any merge."""
    existing = pending_messages.get(session_key)
    old, new = _request(existing), _request(incoming)
    if old is None or new is None or not old.active or not new.active:
        return False
    old_owner = (old.facts.profile_home, old.facts.runtime_profile, old._adapter,
                 old.facts.chat_id, old.facts.thread_id, old.facts.requester_id)
    new_owner = (new.facts.profile_home, new.facts.runtime_profile, new._adapter,
                 new.facts.chat_id, new.facts.thread_id, new.facts.requester_id)
    if old_owner == new_owner:
        return False
    if new._runner._queue_depth(session_key, adapter=new._adapter) >= new._runner._BUSY_QUEUE_MAX_PENDING:
        finish_request(new, "rejected")
        return True
    new._runner._enqueue_fifo(session_key, incoming, new._adapter)
    return True


def tool_observer_for_run(runner, session_key, generation):
    """Resolve ownership at handler start; the returned completion callback keeps that handle."""
    request = request_for_run(runner, session_key, generation)
    if request is None:
        return None

    def observe(**event):
        from hermes_cli.plugins import invoke_hook
        with request._lock:
            if not request.active or not runner._is_session_run_current(session_key, generation):
                return
        with _scope(runner, request._source):
            invoke_hook("gateway_request_tool", request=request, **event)
    return observe
