"""Bounded synchronous catalog dispatch, with no queue or pretend cancellation.

A reservation is reaped only after its actual thread exits, including scope
cleanup. Caller timeouts never release either the profile or provider slot.
"""

import hashlib
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from pathlib import Path

from agent.memory_provider import spawn_context_thread
from hermes_constants import hermes_home_key
from hermes_cli.plugins_loader import _plugin_home_scope
from hermes_cli.skill_catalog import registry
from hermes_cli.skill_catalog.validation import CatalogError, validate_request, validate_result

_BUDGETS = {"availability": 3, "installed": 3, "reviewed": 3, "operation": 3,
            "search": 5, "inspect": 15, "check": 15, "scan": 30, "review": 60, "confirm": 60}
_lock = threading.RLock()


@dataclass
class _Call:
    entry: registry.Entry
    method: str
    deadline: float
    request: dict
    thread: threading.Thread | None = None
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | CatalogError | None = None
    dispatched: bool = False

    @property
    def effect(self):
        return self.method == "confirm"


@dataclass
class _Profile:
    active: list[_Call] = field(default_factory=list)
    generations: set[tuple[str, str]] = field(default_factory=set)
    fenced: set[tuple[str, str]] = field(default_factory=set)
    entries: weakref.WeakValueDictionary = field(default_factory=weakref.WeakValueDictionary)
    stopped: weakref.WeakValueDictionary = field(default_factory=weakref.WeakValueDictionary)

    def reap(self):
        self.active[:] = [call for call in self.active if call.thread is None or call.thread.is_alive()]


_profiles: dict[str, _Profile] = {}


def _identity(entry):
    return entry.source_id, entry.generation


def _available(entry, profile):
    if (entry.stopped or not entry.ready or _identity(entry) in profile.fenced
            or profile.stopped.get(id(entry)) is entry):
        raise CatalogError("provider_unavailable", "Catalog provider is unavailable.")


def _timeout(call):
    if call.effect and call.dispatched:
        return CatalogError("outcome_unknown", "Catalog operation outcome is unknown.")
    return CatalogError("timeout", "Catalog request timed out.")


def _invoke(call, profile):
    try:
        with _plugin_home_scope(Path(call.entry.scope)):
            if not call.entry.authorized():
                raise CatalogError("consent_required", "Catalog consent is required.")
            with _lock:
                _available(call.entry, profile)
                if time.monotonic() >= call.deadline:
                    raise _timeout(call)
                call.dispatched = True
            try:
                result = getattr(call.entry.provider, call.method)(call.request)
            except BaseException:  # health: allow BLE001 -- plugin exceptions may contain secrets; return only core-authored text
                raise CatalogError("provider_error", "Catalog provider failed.") from None
            call.result = validate_result(call.method, result, call.entry)
    except CatalogError as exc:
        call.result = exc
    except BaseException:  # health: allow BLE001 -- scope/validation failures must not leak exception text or traceback
        call.result = CatalogError("provider_error", "Catalog provider failed.")
    finally:
        call.done.set()


def call(entry, method, payload, *, deadline=None) -> dict:
    """Return sanitized data or raise CatalogError; deadline is absolute monotonic.

    The core's Unix-millisecond wire deadline is derived from the lesser of the
    method budget and caller's remaining monotonic budget. Scope hydration and
    live authorization are part of that budget, entirely inside the worker.
    """
    started = time.monotonic()
    request = validate_request(method, payload)
    end = started + _BUDGETS[method]
    if deadline is not None:
        end = min(end, deadline)
    if time.monotonic() >= end:
        raise CatalogError("timeout", "Catalog request timed out.")
    request["context"] = {
        "profile_key": hashlib.sha256(entry.scope.encode("utf-8")).hexdigest(),
        "plugin_key": entry.plugin_key, "generation": entry.generation,
        "request_id": uuid.uuid4().hex,
        "deadline_ms": int((time.time() + max(0, end - time.monotonic())) * 1000),
    }
    invocation = _Call(entry, method, end, request)
    with _lock:
        profile = _profiles.setdefault(entry.scope, _Profile())
        _available(entry, profile)
        profile.reap()
        if len(profile.active) >= 4 or any(
            active.entry.source_id == entry.source_id and active.effect == invocation.effect
            for active in profile.active
        ):
            raise CatalogError("busy", "Catalog provider is busy.")
        profile.generations.add(_identity(entry))
        profile.entries[id(entry)] = entry
        profile.active.append(invocation)
        try:
            invocation.thread = spawn_context_thread(
                _invoke, name="skill-catalog-call", args=(invocation, profile))
            invocation.thread.start()
        except Exception:  # health: allow BLE001 -- spawn failure is a safe error boundary; release its unused reservation
            profile.active.remove(invocation)
            raise CatalogError("provider_error", "Catalog worker could not start.") from None
    invocation.done.wait(max(0, end - time.monotonic()))
    invocation.thread.join(max(0, end - time.monotonic()))
    if invocation.thread.is_alive() or time.monotonic() >= end:
        raise _timeout(invocation)
    if isinstance(invocation.result, CatalogError):
        raise invocation.result
    assert invocation.result is not None
    return invocation.result


def shutdown(scope, timeout=5):
    """Fence known generations, drain at most five seconds, report live effects.

    Returns a tuple of nonsecret uncertainty records for effects still alive.
    Old generations stay fenced permanently; a newly registered Entry generation
    can use free capacity without reviving any stopped worker generation.
    """
    scope = hermes_home_key(scope)
    end = time.monotonic() + min(5, max(0, timeout))
    with _lock:
        profile = _profiles.setdefault(scope, _Profile())
        profile.fenced.update(profile.generations)
        for entry in registry.snapshot(scope=scope):
            profile.entries[id(entry)] = entry
            profile.fenced.add(_identity(entry))
        profile.stopped.update(profile.entries)
        profile.reap()
        active = tuple(profile.active)
    for invocation in active:
        assert invocation.thread is not None
        invocation.thread.join(max(0, end - time.monotonic()))
    with _lock:
        profile.reap()
        return tuple({"code": "outcome_unknown", "source_id": invocation.entry.source_id,
                      "registration_id": invocation.entry.registration_id,
                      "generation": invocation.entry.generation,
                      "request_id": invocation.request["context"]["request_id"]}
                     for invocation in profile.active if invocation.effect and invocation.dispatched)
