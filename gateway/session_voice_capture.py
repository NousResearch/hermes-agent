"""Project durable GPT-Live deltas without admitting or executing an agent turn."""

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Callable, Container
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from itertools import groupby
import logging
from uuid import uuid4

from tools.gpt_live_sideband import ClientDelegation

from persistence.conversation.live_voice import VoiceFragment, VoiceTurn
from persistence.core.identity import SessionId
from persistence.conversation.messages import MessageRepository
from persistence.core.errors import Conflict, NotFound, TurnLeaseLost

log = logging.getLogger(__name__)


def assemble_voice_turns(
    fragments: tuple[VoiceFragment, ...], *, finish: bool = False
) -> tuple[VoiceTurn, ...]:
    """Only the trailing speaker run can grow; timestamps never order words."""
    turns = tuple(VoiceTurn(tuple(f.event_id for f in run)) for _, run in groupby(fragments, key=lambda f: f.role))
    return turns if finish else turns[:-1]


async def _publish(
    messages: MessageRepository, call_id: str, *, finish: bool = False,
    write=asyncio.to_thread, published: Callable[[], None] | None = None,
) -> None:
    fragments = await asyncio.to_thread(messages.pending_voice_fragments, call_id)
    while turns := assemble_voice_turns(fragments, finish=finish):
        try:
            count = await write(messages.publish_voice_turns, call_id, turns)
        except TurnLeaseLost:
            # The in-flight agent owns its history snapshot. The pending text is
            # already durable and a subsequent tick can publish it after settlement.
            return
        except Conflict as exc:
            # A claim can consume a marker-bounded prefix of this snapshot
            # between the read and write. Regroup what's still pending.
            if exc.entity != 'live voice projection coverage':
                raise
            remaining = await asyncio.to_thread(messages.pending_voice_fragments, call_id)
            if remaining == fragments:
                raise
            fragments = remaining
        else:
            if count and published is not None:
                published()
            return


async def record_voice_events(
    messages: MessageRepository, call_id: str, events: AsyncIterable[VoiceFragment | ClientDelegation | None], *,
    write=asyncio.to_thread, published: Callable[[], None] | None = None,
    marked: Callable[[str], None] | None = None,
) -> bool:
    """Record a stream from an already-authorized sideband. None is the vendor's clean close."""
    closed = False
    try:
        async for fragment in events:
            if fragment is None:
                closed = True
                break
            if isinstance(fragment, ClientDelegation):
                if fragment.call_id == call_id and fragment.event_id:
                    await write(messages.record_voice_marker, call_id, fragment.delegation_id,
                                fragment.event_id, fragment.offset_ms)
                    if marked is not None:
                        marked(fragment.delegation_id)
                continue
            if await write(messages.ingest_voice_fragment, call_id, fragment):
                await _publish(messages, call_id, write=write, published=published)
    finally:
        incomplete = await write(messages.finish_voice_capture, call_id, incomplete=not closed)
        await _publish(messages, call_id, finish=True, write=write, published=published)
    return not incomplete


async def recover_voice_calls(
    messages: MessageRepository, *, active: Container[str] = (), write=asyncio.to_thread,
    session_id: SessionId | None = None, published: Callable[[str], None] | None = None,
) -> None:
    """Recover captured text after restart, or after a turn that blocked final publication.

    Only the serving profile authority may call this with its own live call registry.
    """
    pending = await asyncio.to_thread(messages.pending_voice_calls, session_id=session_id)
    for call_id, closed in pending:
        if not closed and (call_id in active or session_id is not None):
            # Only the startup sweep can declare an unclosed call abandoned.
            # Once close commits, either publisher can drain the remaining tail.
            continue
        try:
            if not closed:
                await write(messages.finish_voice_capture, call_id, incomplete=True)
            await _publish(messages, call_id, finish=True, write=write,
                           published=partial(published, call_id) if published else None)
        except NotFound:
            # The user deleted this conversation between enumeration and projection.
            continue
        except Exception as exc:  # health: allow BLE001 -- one call's recovery must not stop the sweep; provider errors can carry credentials, so only the type is logged
            log.warning('GPT-Live pending transcript recovery deferred (%s)', type(exc).__name__)


@asynccontextmanager
async def voice_capture_socket(call_id: str, *, api_key: str, base_url: str) -> AsyncIterator[AsyncIterable[VoiceFragment | ClientDelegation | None]]:
    from tools.gpt_live_sideband import SessionClosed, TranscriptDelta, open_sideband

    async with open_sideband(call_id=call_id, api_key=api_key, base_url=base_url) as source:
        async def fragments():
            async for event in source:
                if isinstance(event, SessionClosed):
                    yield None
                    return
                if isinstance(event, ClientDelegation):
                    yield event
                if isinstance(event, TranscriptDelta) and event.text:
                    # We attach once, without reconnect on disconnect. For a provider
                    # event with no ID, preserve the words rather than guessing a match.
                    start, end = event.start_ms, event.end_ms
                    if start is None or end is None:
                        start = end = None
                    yield VoiceFragment(
                        event.event_id or f'local:{uuid4().hex}', event.speaker, event.text, start, end,
                    )

        yield fragments()


async def start_live_voice(authority, actor, ref, sdp: str, history: list | None) -> dict:
    """Negotiate and attach in the serving profile; return no project credentials."""
    from gateway.config import Platform
    from gateway.session_authorities import async_owner_scope
    from gateway.session_local_recovery import reopen_local_session
    from gateway.session_runtime_workers import tracked_write
    from persistence.core.errors import AuthorityRefusal
    from tools.voice_live import _live_section, _resolve_credentials, create_webrtc_session, voice_offer_answer

    if not isinstance(sdp, str) or not sdp.strip() or (history is not None and not isinstance(history, list)):
        raise AuthorityRefusal('invalid_params')
    authority.authorize(actor, ref, 'session:control')

    def negotiate():
        key, base = _resolve_credentials(_live_section())
        return voice_offer_answer(create_webrtc_session(sdp, history, credentials=(key, base))), key, base

    async with async_owner_scope(authority):
        try:
            result, key, base = await asyncio.to_thread(negotiate)
        except (ValueError, RuntimeError) as exc:
            raise AuthorityRefusal('voice_provider_unavailable') from exc
    authority.authorize(actor, ref, 'session:control')
    call_id = (result.get('session') or {}).get('id')
    live = authority.sessions.get(ref.session_id)
    if live is not None and live.source is not None and live.source.platform == Platform.LOCAL:
        # Resume alone stays read-only. Starting voice is new input, like the
        # first typed turn, but does not need a prompt to reopen the same tip.
        async with live.mutation_lock:
            authority.authorize(actor, ref, 'session:control')
            await tracked_write(authority, partial(reopen_local_session, authority, ref))
    recording = isinstance(call_id, str) and bool(call_id) and await authority.voice_captures.start(
        SessionId(ref.session_id), call_id, voice_capture_socket(call_id, api_key=key, base_url=base),
    )
    if not recording:
        raise AuthorityRefusal('voice_capture_unavailable')
    return {**result, 'capture_status': 'recording'}


@dataclass
class _MarkerProgress:
    session_id: str
    received: set[str] = field(default_factory=set)
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    def mark(self, delegation_id: str) -> None:
        self.received.add(delegation_id)
        self.changed.set()
        self.changed = asyncio.Event()


class VoiceCaptures:
    """Own the provider tasks on one serving profile's asyncio loop."""

    def __init__(self, messages: MessageRepository, *, authority=None) -> None:
        self.messages = messages
        self._authority = authority
        self.tasks: dict[str, asyncio.Task] = {}
        self._markers: dict[str, _MarkerProgress] = {}
        self._closing = False
        self._recovery = asyncio.Lock()

    async def _write(self, fn, *args, **kwargs):
        if self._authority is None:
            return await asyncio.to_thread(fn, *args, **kwargs)
        from gateway.session_runtime_workers import tracked_write
        return await tracked_write(self._authority, partial(fn, *args, **kwargs), ordered=True)

    async def recover(self, *, session_id: str | None = None, event_session: str | None = None) -> None:
        async with self._recovery:
            if not self._closing:
                def published(call_id: str) -> None:
                    if event_session is not None:
                        self._publish_capture(event_session, call_id, 'updated')

                await recover_voice_calls(self.messages, active=self.tasks, write=self._write,
                                          session_id=SessionId(session_id) if session_id is not None else None,
                                          published=published if event_session is not None else None)

    async def wait_for_marker(self, session_id: str, call_id: str, delegation_id: str, *,
                              timeout: float = 2.0) -> bool:
        """Allow bounded WebRTC/sideband skew, without replacing the SQLite proof."""
        progress = self._markers.get(call_id)
        if progress is None or progress.session_id != session_id:
            return False
        try:
            async with asyncio.timeout(timeout):
                while delegation_id not in progress.received:
                    if self._markers.get(call_id) is not progress:
                        return False
                    await progress.changed.wait()
                return True
        except TimeoutError:
            return False

    def _publish_capture(self, session_id, call_id: str, state: str) -> None:
        if self._authority is None:
            return
        live = self._authority.sessions.get(session_id)
        if live is not None:
            live.event_stream.publish(session_id, {'call_id': call_id, 'state': state},
                                      event_type='voice.live.capture', execution={})

    async def start(self, session_id, call_id: str, socket: AbstractAsyncContextManager[AsyncIterable[VoiceFragment | ClientDelegation | None]]) -> bool:
        if self._closing or call_id in self.tasks:
            return False
        ready = asyncio.get_running_loop().create_future()
        progress = self._markers[call_id] = _MarkerProgress(session_id)

        async def run():
            state = 'incomplete'
            try:
                async with socket as events:
                    await self._write(self.messages.start_voice_capture, session_id, call_id)
                    ready.set_result(True)
                    self._publish_capture(session_id, call_id, 'recording')
                    closed = await record_voice_events(
                        self.messages, call_id, events, write=self._write,
                        published=partial(self._publish_capture, session_id, call_id, 'updated'),
                        marked=progress.mark,
                    )
                if closed:
                    remaining = await asyncio.to_thread(self.messages.pending_voice_fragments, call_id)
                    state = 'pending' if remaining else 'saved'
            except NotFound:
                pass  # Conversation deleted during capture.
            except Exception as exc:  # health: allow BLE001 -- a capture failure ends only this recording, never the gateway; provider errors can carry credentials, so only the type is logged
                log.warning('GPT-Live capture interrupted (%s)', type(exc).__name__)
            finally:
                if not ready.done():
                    ready.set_result(False)
                elif ready.result():
                    self._publish_capture(session_id, call_id, state)

        def finished(task):
            self.tasks.pop(call_id, None)
            self._markers.pop(call_id, None)
            progress.changed.set()
            if not ready.done():
                ready.set_result(False)  # cancelled before the coroutine entered
            if not task.cancelled():
                task.exception()

        task = self.tasks[call_id] = asyncio.create_task(run(), name='gpt-live-capture')
        task.add_done_callback(finished)
        try:
            return await asyncio.shield(ready)
        except asyncio.CancelledError:
            task.cancel()
            raise

    def stop(self) -> None:
        if self._closing:
            return
        self._closing = True
        for task in self.tasks.values():
            task.cancel()

    def quiescent(self) -> bool:
        return not self.tasks and not self._recovery.locked()

    async def close(self) -> None:
        self.stop()
        tasks = tuple(self.tasks.values())
        if tasks:
            await asyncio.wait(tasks)
        async with self._recovery:
            pass
