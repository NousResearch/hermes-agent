"""Session relay: start a turn in a Discord thread from a trusted local caller.

``POST /api/relay`` (API server, ``API_SERVER_KEY``) delivers *text* into a Discord thread as an
internal event, the same way handoff and Kanban wakes enter a session:

- without ``thread_id``: open a thread named ``title`` under ``parent_chat_id``, add ``user_id`` to
  it, and run the first turn there, so the thread's session starts with the relayed text;
- with ``thread_id``: run a turn in that thread's session. Under a session authority the turn is
  committed to the session FIFO behind any running turn (``admit_route_wake``); otherwise the
  adapter queues the internal event, which never interrupts or steers a running turn.

The source is built by the Discord adapter (``build_source``), so ``gateway.profile_routes`` pick the
profile exactly as for a human reply in that thread: a thread under a channel routed to a
route-only profile (no bot of its own) runs in that profile.

Off by default. ``gateway.session_relay.enabled`` and ``gateway.session_relay.allowed_parents`` (parent channel
ids) are read from the launch profile's config. The relayed text is framed so the model can tell
it was not typed by the user; approvals and the emergency stop still apply.
"""
from __future__ import annotations

import contextlib
import logging
import weakref
from typing import Any, Optional

logger = logging.getLogger(__name__)

RELAY_USER_ID = "system:relay"
RELAY_USER_NAME = "Relay"
RELAY_PRODUCER = "relay"
MAX_TEXT_CHARS = 32000
MAX_TITLE_CHARS = 100
MAX_FROM_SESSION_CHARS = 200


class RelayError(Exception):
    """A refused relay request; ``status``/``code`` become the HTTP error."""

    def __init__(self, message: str, status: int = 400, code: str = "relay_invalid"):
        super().__init__(message)
        self.status = status
        self.code = code


def relay_settings(config: Optional[dict]) -> tuple[bool, set[str]]:
    """``(enabled, allowed parent channel ids)`` from a raw config mapping."""
    # Not ``gateway.relay``: gateway.<platform> blocks merge into platform config, and ``relay`` is a
    # platform (Platform.RELAY), so that key would start a relay adapter.
    relay = ((config or {}).get("gateway") or {}).get("session_relay") or {}
    enabled = relay.get("enabled")
    enabled = enabled is True or str(enabled).strip().lower() in {"true", "1", "yes"}
    raw = relay.get("allowed_parents") or []
    if isinstance(raw, (str, int)):
        raw = str(raw).split(",")
    allowed = {str(p).strip() for p in raw if str(p).strip()}
    return enabled, allowed


def frame_text(text: str, from_session: Optional[str]) -> str:
    """Mark relayed text so the model never reads it as the user's own words."""
    origin = f"session {from_session}" if from_session else "a local process"
    return f"[Relayed from {origin}; not typed by the user]\n{text}"


def _snowflake(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text.isdigit():
        raise RelayError(f"{name} must be a numeric Discord id")
    return text


async def _resolve_channel(client: Any, channel_id: str) -> Any:
    channel = client.get_channel(int(channel_id))
    if channel is None:
        channel = await client.fetch_channel(int(channel_id))
    return channel


# Gateway entry points, imported lazily: ``gateway.run`` bootstraps the whole runtime on import.
def _load_config() -> dict:
    from gateway.run import _load_gateway_config
    return _load_gateway_config()


def _profile_scope(profile_home: Any):
    from gateway.run import _async_profile_runtime_scope
    return _async_profile_runtime_scope(profile_home)


def _paused() -> bool:
    try:
        from agent.estop import paused_reply
    except ImportError:
        return False
    return paused_reply() is not None


async def _route_wake(runner: Any, adapter: Any, source: Any, text: str) -> bool:
    """Commit the turn to the session FIFO; False when the thread has no committed human turn."""
    from gateway.session_automation import admit_route_wake
    from hermes_state_runtime import RuntimeStoreError
    try:
        return bool(await admit_route_wake(runner, adapter, source, text, RELAY_PRODUCER))
    except RuntimeStoreError as exc:
        raise RelayError(f"session refused the relay: {exc}", 409, "relay_refused") from exc


async def _admit_internal(adapter: Any, event: Any) -> None:
    from gateway.wake import WakeNotAccepted, admit_internal_event
    try:
        await admit_internal_event(adapter, event)
    except WakeNotAccepted as exc:
        raise RelayError(f"the gateway did not accept the relayed turn: {exc}",
                         409, "relay_not_accepted") from exc


def _routed_home(runner: Any, source: Any) -> Any:
    """Pin *source*'s routing identity the way the primary message handler does for a real
    inbound message (``_admit_primary_source``) and return its runtime home. Raises when the
    route targets a profile this gateway does not serve."""
    admit = getattr(runner, "_admit_primary_source", None)
    if callable(admit):
        from pathlib import Path

        from hermes_constants import get_hermes_home
        home = admit(source, Path(get_hermes_home()))
        if home is None:
            raise RelayError("the thread routes to a profile this gateway does not serve",
                             409, "relay_route_rejected")
        return home
    return runner._resolve_profile_home_for_source(source)


async def _admit(runner: Any, adapter: Any, source: Any, text: str) -> str:
    """Admit *text* as an internal turn for *source*: ``queued`` behind the session FIFO, or
    ``direct`` through the adapter (a thread with no committed human message yet)."""
    from gateway.platforms.event import MessageEvent, MessageType

    async with _profile_scope(_routed_home(runner, source)):
        if await _route_wake(runner, adapter, source, text):
            return "queued"
        event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source, internal=True)
        await _admit_internal(adapter, event)
        return "direct"


async def relay(runner: Any, body: dict) -> dict:
    """Validate *body*, open or resolve the thread, and admit the turn. Raises ``RelayError``."""
    from gateway.config import Platform

    enabled, allowed = relay_settings(_load_config())
    if not enabled:
        raise RelayError("session relay is disabled (gateway.session_relay.enabled)", 403, "relay_disabled")
    if _paused():
        raise RelayError("Hermes is paused; relay refused", 409, "relay_paused")

    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RelayError("text is required")
    if len(text) > MAX_TEXT_CHARS:
        raise RelayError(f"text is longer than {MAX_TEXT_CHARS} characters")
    platform = str(body.get("platform") or "discord").strip().lower()
    if platform != "discord":
        raise RelayError("only the discord platform is supported", 400, "relay_platform")
    from_session = body.get("from_session")
    if from_session is not None:
        from_session = str(from_session).strip()[:MAX_FROM_SESSION_CHARS] or None

    adapter = (getattr(runner, "adapters", None) or {}).get(Platform.DISCORD)
    client = getattr(adapter, "_client", None)
    if adapter is None or client is None:
        raise RelayError("the discord adapter is not connected", 503, "relay_unavailable")
    import discord

    thread_id = body.get("thread_id")
    title = None
    thread: Any = None
    if thread_id:
        thread_id = _snowflake(thread_id, "thread_id")
        thread = await _resolve_channel(client, thread_id)
        if not isinstance(thread, discord.Thread):
            raise RelayError(f"{thread_id} is not a thread")
        parent_id = str(thread.parent_id)
    else:
        parent_id = _snowflake(body.get("parent_chat_id"), "parent_chat_id")
        title = str(body.get("title") or "").strip()[:MAX_TITLE_CHARS]
        if not title:
            raise RelayError("title is required when opening a new thread")
    if parent_id not in allowed:
        raise RelayError(f"parent channel {parent_id} is not in gateway.session_relay.allowed_parents",
                         403, "relay_forbidden")

    created = False
    if not thread_id:
        new_id = await adapter.create_handoff_thread(parent_id, title)
        if not new_id:
            raise RelayError("could not create the thread", 502, "relay_thread_failed")
        thread_id, created = str(new_id), True
        thread = await _resolve_channel(client, thread_id)

    user_added = None
    if body.get("user_id"):
        uid = _snowflake(body.get("user_id"), "user_id")
        try:
            await thread.add_user(discord.Object(id=int(uid)))
            user_added = True
        except Exception:
            logger.warning("Relay: could not add user %s to thread %s", uid, thread_id, exc_info=True)
            user_added = False
    # Discord only answers un-mentioned follow-ups in threads the bot has participated in.
    threads = getattr(adapter, "_threads", None)
    if threads is not None:
        await threads.mark_async(thread_id)

    fmt = getattr(adapter, "_format_thread_chat_name", None)
    chat_name = fmt(thread) if callable(fmt) else getattr(thread, "name", None)
    guild = getattr(thread, "guild", None)
    source = adapter.build_source(
        chat_id=thread_id, chat_name=chat_name, chat_type="thread", user_id=RELAY_USER_ID,
        user_name=RELAY_USER_NAME, thread_id=thread_id,
        guild_id=str(guild.id) if guild is not None else None, parent_chat_id=parent_id,
    )
    if getattr(source, "profile_route_rejected", False) is True:
        raise RelayError("the thread routes to a profile this gateway does not serve",
                         409, "relay_route_rejected")
    with contextlib.suppress(Exception):
        source._transport_adapter_ref = weakref.ref(adapter)

    mode = await _admit(runner, adapter, source, frame_text(text, from_session))
    result = {
        "ok": True, "thread_id": thread_id, "parent_chat_id": parent_id, "created": created,
        "profile": getattr(source, "profile", None) or "default",
        "session_key": runner._session_key_for_source(source), "mode": mode,
    }
    if user_added is not None:
        result["user_added"] = user_added
    logger.info("Relay admitted: thread=%s parent=%s profile=%s mode=%s created=%s from=%s",
                thread_id, parent_id, result["profile"], mode, created, from_session)
    return result
