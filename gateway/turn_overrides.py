"""Optional, load-owned post-admission turn model-override seam. No domain classifiers or identities.

A plugin registers ``provider.decide(request) -> TurnOverride | None``. The host consults it once per
admitted, claimed human turn and, when a valid override comes back, arms it through the SAME canonical
one-turn owner as ``/moa`` and ``/model --once``: the earliest pending snapshot is kept, the cached agent
is evicted, and the generation-guarded ``finally`` in ``_handle_message`` restores it on every exit.

Providers receive an immutable request (text, source coordinates, flags and a frozen config copy). They
never receive the conversation, runner, agent or a send handle, so they cannot persist a model switch,
restore anything themselves or act on the turn. This is an optional extension: a missing, disabled,
raising, slow or invalid provider means "no override" and the turn runs on its normal route.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import copy
import logging
import queue
import threading

logger = logging.getLogger(__name__)

# Shape of a one-turn model override, identical to the ``/moa`` one-shot sugar and ``/model --once``.
OVERRIDE_KEYS = frozenset({"provider", "model", "base_url", "api_key", "api_mode"})
DECIDE_TIMEOUT_SECONDS = 0.25
# Own capacity: a hung optional provider must never consume the required egress policy's slots.
_CAPACITY = threading.BoundedSemaphore(2)


class _Unavailable(Exception):
    pass


def _bounded(callback, *args):
    if not _CAPACITY.acquire(blocking=False):
        raise _Unavailable("capacity")
    box = queue.Queue(maxsize=1)

    def run():
        try:
            box.put((True, callback(*args)))
        except BaseException:
            box.put((False, None))
        finally:
            _CAPACITY.release()
    threading.Thread(target=run, daemon=True, name="turn-override-decide").start()
    try:
        okay, value = box.get(timeout=DECIDE_TIMEOUT_SECONDS)
    except queue.Empty:
        raise _Unavailable("timeout") from None
    if not okay:
        raise _Unavailable("raised")
    return value


@dataclass(frozen=True)
class TurnOverrideRequest:
    text: str
    platform: str
    chat_type: str
    chat_id: str
    user_id: str
    is_bot: bool
    is_internal: bool
    is_command: bool
    session_key: str
    config: Mapping = field(default_factory=dict)  # detached copy; never the host's object


@dataclass(frozen=True)
class TurnOverride:
    override: Mapping
    reason: str


def _plain_copy(value):
    """Detached plain-container copy: provider edits can never reach host configuration."""
    if isinstance(value, Mapping):
        return {k: _plain_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_copy(v) for v in value]
    return copy.deepcopy(value)


def _platform(source) -> str:
    value = getattr(getattr(source, "platform", None), "value", getattr(source, "platform", None))
    return str(value or "").strip().lower()


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


def _valid(result) -> bool:
    if type(result) is not TurnOverride or not isinstance(result.reason, str) or not result.reason:
        return False
    o = result.override
    return (isinstance(o, Mapping) and set(o) == OVERRIDE_KEYS
            and all(isinstance(v, str) for v in o.values()) and bool(o.get("provider")) and bool(o.get("model")))


def apply_turn_override(*, event, source, session_key, conversation, snapshot_override, evict_cached_agent,
                        full_config=None, is_internal=False, is_command=False) -> bool:
    """Consult registered providers once; arm the first valid override through the canonical owner."""
    try:
        if getattr(event, "_moa_disable_after_turn", False):
            return False  # an explicit one-turn override (/moa, /model --once) already owns this turn
        providers = _registered("_turn_override_providers")
        if not providers:
            return False
        request = TurnOverrideRequest(
            text=str(getattr(event, "text", None) or ""), platform=_platform(source),
            chat_type=str(getattr(source, "chat_type", "") or "").strip().lower(),
            chat_id=str(getattr(source, "chat_id", "") or "").strip(),
            user_id=str(getattr(source, "user_id", "") or "").strip(),
            is_bot=bool(getattr(source, "is_bot", False)), is_internal=bool(is_internal),
            is_command=bool(is_command), session_key=str(session_key),
            config=_plain_copy(full_config) if isinstance(full_config, Mapping) else {})
        for key, provider in providers:
            try:
                result = _bounded(provider.decide, request)
            except Exception:
                logger.debug("turn override provider %s failed open", key)
                continue
            if result is None:
                continue
            if not _valid(result):
                logger.debug("turn override provider %s returned an invalid decision; ignored", key)
                continue
            if not conversation.one_turn_restore:
                conversation.one_turn_restore = snapshot_override(session_key)
            conversation.model_override = dict(result.override)
            try:
                if callable(evict_cached_agent):
                    evict_cached_agent(session_key)
            except Exception:
                logger.debug("turn override evict_cached_agent failed", exc_info=True)
            event._moa_disable_after_turn = True
            event._turn_override_provider = key
            event._turn_override_reason = result.reason
            return True
    except Exception:
        logger.debug("turn override seam failed open", exc_info=True)
    return False
