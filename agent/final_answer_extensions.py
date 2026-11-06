"""Optional final-answer extensions: a plugin may silently continue a turn before its final answer.

A provider registered with ``PluginContext.register_final_answer_provider(provider_id, provider)``
sees the would-be final answer and returns either ``None`` (let it through) or a ``FinalNudge``: a
private instruction the loop appends as an internal user row before making one more model call.
The draft and the nudge rows carry ``EXTENSION_SYNTHETIC_FLAG`` (plus the provider's own flag), so
they are never persisted to the session transcript or replayed on resume.

Providers receive copies only (a few identity fields, their own per-session state dict, the message
list). They never receive the agent, a send handle or tool access. The hook is optional and
fail-open: a missing, slow, raising or invalid provider lets the answer through unchanged. Bounding
the number of continuations per turn is the provider's job (it owns its state); the host commits a
provider's state only after a valid return.
"""
from __future__ import annotations

import copy
import logging
import queue
import threading
from dataclasses import dataclass
from types import MappingProxyType

logger = logging.getLogger(__name__)

EXTENSION_SYNTHETIC_FLAG = "_final_answer_extension_synthetic"
CALL_TIMEOUT_SECONDS = 0.25
_IDENTITY_FIELDS = ("platform", "_user_id", "_chat_type", "_interrupt_requested", "_current_turn_id", "session_id")
_CAPACITY = threading.BoundedSemaphore(2)


@dataclass(frozen=True)
class FinalNudge:
    text: str
    reason: str
    synthetic_flag: str


@dataclass(frozen=True)
class FinalContext:
    identity: object
    state: dict
    messages: tuple

def _registered(slot):
    """``(provider_id, provider)`` pairs whose owning plugin is currently loaded without error."""
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    out = []
    for key, (owner, provider) in sorted(getattr(manager, slot, {}).items()):
        info = manager._plugins.get(owner)
        if info is not None and getattr(info, "enabled", True) and not getattr(info, "error", None):
            out.append((key, provider))
    return out


def _bounded(callback, *args):
    if not _CAPACITY.acquire(blocking=False):
        raise TimeoutError("final-answer provider capacity exhausted")
    box = queue.Queue(maxsize=1)

    def run():
        try:
            box.put((True, callback(*args)))
        except BaseException:
            box.put((False, None))
        finally:
            _CAPACITY.release()
    threading.Thread(target=run, daemon=True, name="final-answer-extension").start()
    try:
        okay, value = box.get(timeout=CALL_TIMEOUT_SECONDS)
    except queue.Empty:
        raise TimeoutError("final-answer provider timed out") from None
    if not okay:
        raise RuntimeError("final-answer provider raised")
    return value


def _valid(result) -> bool:
    return (type(result) is FinalNudge and isinstance(result.text, str) and bool(result.text)
            and isinstance(result.reason, str) and isinstance(result.synthetic_flag, str)
            and result.synthetic_flag.startswith("_") and result.synthetic_flag.endswith("_synthetic"))


def before_final(agent, text, messages):
    """First valid ``FinalNudge`` from a loaded provider, else ``None``. Never raises."""
    try:
        identity = {key: getattr(agent, key, None) for key in _IDENTITY_FIELDS}
        identity["_current_turn_id"] = identity["_current_turn_id"] or id(messages)
        states = getattr(agent, "_final_answer_extension_state", None)
        if not isinstance(states, dict):
            states = {}
        for key, provider in _registered("_final_answer_providers"):
            state = copy.deepcopy(states.get(key, {}))
            context = FinalContext(MappingProxyType(dict(identity)), state, tuple(copy.deepcopy(messages)))
            try:
                result = _bounded(provider.before_final, context, text)
            except Exception:
                logger.debug("final-answer provider %s failed open", key)
                continue
            if result is not None and not _valid(result):
                continue
            states[key] = state
            agent._final_answer_extension_state = states
            if result is not None:
                return result
    except Exception:
        logger.debug("final-answer extensions failed open", exc_info=True)
    return None
