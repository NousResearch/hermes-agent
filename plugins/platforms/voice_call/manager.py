"""Call state manager for the voice_call platform.

Owns every active :class:`CallRecord`: state transitions, transcript
accumulation, timers (ring / max-duration / silence / notify hangup),
transcript waiters for ``continue_call``-style turns, and persistence
through :class:`CallStore`. Consumes only ``NormalizedEvent``s — never
provider wire formats.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from .config import VoiceCallConfig, is_e164, normalize_e164
from .events import (
    CallRecord,
    CallState,
    EventType,
    NormalizedEvent,
    TranscriptEntry,
    is_valid_transition,
    new_call_id,
)
from .providers.base import VoiceCallProvider
from .store import CallStore

logger = logging.getLogger(__name__)

# A transcript callback receives the call and the final user utterance.
TranscriptCallback = Callable[[CallRecord, str], Awaitable[None]]
CallEndedCallback = Callable[[CallRecord], Awaitable[None]]

_DEDUPE_CAP = 2048
_ORPHAN_CAP = 256
_ORPHAN_TTL_S = 300.0


class CallNotFoundError(KeyError):
    pass


class CallManager:
    def __init__(
        self,
        config: VoiceCallConfig,
        provider: VoiceCallProvider,
        store: CallStore,
        on_final_transcript: Optional[TranscriptCallback] = None,
        on_call_ended: Optional[CallEndedCallback] = None,
    ):
        self.config = config
        self.provider = provider
        self.store = store
        self.on_final_transcript = on_final_transcript
        self.on_call_ended = on_call_ended
        # Optional hook invoked before the provider dials/answers — the
        # realtime bridge uses it to attach stream metadata to the record.
        self.prepare_call: Optional[Callable[[CallRecord], None]] = None
        # Optional hook for speaking on realtime-bridged calls: the model
        # owns the audio there, so carrier TTS would talk over the stream.
        # Set by the runtime; returns True when the bridge delivered it.
        self.realtime_speaker: Optional[
            Callable[[CallRecord, str], Awaitable[bool]]
        ] = None
        # Optional hook to attach a media stream to a LIVE call (Telnyx
        # streaming_start) — lets a notify-born call upgrade all the way to
        # the realtime voice instead of carrier TTS. Returns True on success.
        self.upgrade_realtime: Optional[
            Callable[[CallRecord], Awaitable[bool]]
        ] = None

        self.active: Dict[str, CallRecord] = {}
        self._by_provider_id: Dict[str, str] = {}
        self._by_chat: Dict[str, str] = {}  # peer E.164 → most recent call_id
        self._processed: "OrderedDict[str, float]" = OrderedDict()
        self._waiters: Dict[str, asyncio.Future] = {}
        # Speak-playback tracking (carrier call.speak.ended events): lets
        # continue_call wait until its question finished playing before
        # listening for the reply.
        self._speak_ended_at: Dict[str, float] = {}
        self._speak_done_waiters: Dict[str, asyncio.Future] = {}
        # Calls whose caller speech should be discarded right now (e.g. the
        # caller reacting to the previous message while the continue_call
        # question is still playing — that's not the answer).
        self._drop_speech: set = set()
        self._timers: Dict[Tuple[str, str], asyncio.Task] = {}
        # Events that arrived for an outbound call before initiate_call()
        # returned the provider call id (webhook/initiate race).
        self._orphans: "OrderedDict[str, List[NormalizedEvent]]" = OrderedDict()
        self._orphan_seen: Dict[str, float] = {}

    # -- lifecycle ----------------------------------------------------------

    async def initialize(self, restore: bool = True) -> None:
        """Restore persisted non-terminal calls, verifying each with the
        provider: terminal → finalize, expired → hang up, unknown → keep."""
        if not restore:
            return
        for record in self.store.load_active().values():
            try:
                status = await self.provider.get_call_status(record)
            except Exception as e:  # noqa: BLE001 — transient carrier errors
                logger.warning(
                    "voice_call restore: status check failed for %s: %s",
                    record.call_id, e,
                )
                status = None
            if status is not None and status.is_terminal:
                record.state = CallState.COMPLETED
                record.ended_at = record.ended_at or time.time()
                record.end_reason = record.end_reason or "completed-while-down"
                self.store.append(record)
                continue
            elapsed = time.time() - record.started_at
            if elapsed > self.config.timeouts.max_call_s:
                try:
                    await self.provider.hangup_call(record)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "voice_call restore: hangup failed for %s", record.call_id,
                        exc_info=True,
                    )
                record.state = CallState.TIMEOUT
                record.ended_at = time.time()
                record.end_reason = "max-duration"
                self.store.append(record)
                continue
            self._register(record)
            remaining = max(1.0, self.config.timeouts.max_call_s - elapsed)
            self._arm_timer(record.call_id, "max", remaining, self._on_max_duration)
            logger.info("voice_call restore: resumed call %s (%s)",
                        record.call_id, record.state.value)

    async def shutdown(self) -> None:
        """Cancel timers and waiters. Calls stay alive at the carrier and are
        re-verified by the next boot's restore pass."""
        for task in list(self._timers.values()):
            task.cancel()
        self._timers.clear()
        for fut in list(self._waiters.values()):
            if not fut.done():
                fut.cancel()
        self._waiters.clear()
        for fut in list(self._speak_done_waiters.values()):
            if not fut.done():
                fut.cancel()
        self._speak_done_waiters.clear()
        self._drop_speech.clear()

    # -- lookups ------------------------------------------------------------

    def get_call(self, call_id: str) -> Optional[CallRecord]:
        return self.active.get(call_id)

    def get_active_calls(self) -> List[CallRecord]:
        return list(self.active.values())

    def call_for_chat(self, chat_id: str) -> Optional[CallRecord]:
        call_id = self._by_chat.get(normalize_e164(chat_id))
        return self.active.get(call_id) if call_id else None

    def append_transcript(self, call_id: str, speaker: str, text: str) -> None:
        """Record a transcript line from an external source (realtime bridge)."""
        record = self.active.get(call_id)
        if record is None:
            return
        record.transcript.append(
            TranscriptEntry(
                timestamp=time.time(),
                speaker="bot" if speaker == "bot" else "user",
                text=text,
            )
        )
        self._persist(record)
        # On realtime calls the carrier transcription is off, so this is the
        # only place caller utterances surface — resolve continue_call
        # waiters here just like _on_speech does for carrier transcripts.
        if speaker != "bot":
            waiter = self._waiters.pop(call_id, None)
            if waiter is not None and not waiter.done():
                waiter.set_result(text)

    def queue_initial_message(self, call_id: str, text: str) -> bool:
        """Queue ``text`` to be spoken when a still-ringing outbound call is
        answered. Returns False when the call already has an opening message
        (or isn't in a queueable state)."""
        record = self.active.get(call_id)
        if (
            record is None
            or record.is_terminal
            or record.answered_at is not None
            or record.direction != "outbound"
            or record.metadata.get("initial_message")
        ):
            return False
        record.metadata["initial_message"] = text
        self._persist(record)
        return True

    def _realtime_owns_audio(self, record: CallRecord) -> bool:
        """True when a realtime bridge handles this call's audio — carrier
        TTS/transcription would talk over the model."""
        return bool(record.metadata.get("realtime"))

    def _register(self, record: CallRecord) -> None:
        self.active[record.call_id] = record
        if record.provider_call_id:
            self._by_provider_id[record.provider_call_id] = record.call_id
        peer = record.peer_number
        if peer:
            self._by_chat[normalize_e164(peer)] = record.call_id

    def _unregister(self, record: CallRecord) -> None:
        self.active.pop(record.call_id, None)
        if record.provider_call_id:
            self._by_provider_id.pop(record.provider_call_id, None)
        peer = normalize_e164(record.peer_number or "")
        if peer and self._by_chat.get(peer) == record.call_id:
            self._by_chat.pop(peer, None)

    # -- persistence / transitions -------------------------------------------

    def _persist(self, record: CallRecord) -> None:
        try:
            self.store.append(record)
        except OSError:
            logger.warning("voice_call: failed to persist call %s",
                           record.call_id, exc_info=True)

    def _transition(self, record: CallRecord, new_state: CallState) -> bool:
        if not is_valid_transition(record.state, new_state):
            logger.debug(
                "voice_call: dropping invalid transition %s → %s for %s",
                record.state.value, new_state.value, record.call_id,
            )
            return False
        record.state = new_state
        self._persist(record)
        return True

    def _session_key(self, record: CallRecord) -> str:
        peer = normalize_e164(record.peer_number or "unknown")
        if self.config.session_scope == "per-call":
            return f"{peer}:{record.call_id}"
        return peer

    # -- timers ---------------------------------------------------------------

    def _arm_timer(
        self,
        call_id: str,
        kind: str,
        delay: float,
        callback: Callable[[str], Awaitable[None]],
    ) -> None:
        self._cancel_timer(call_id, kind)

        async def _fire():
            try:
                await asyncio.sleep(delay)
                self._timers.pop((call_id, kind), None)
                await callback(call_id)
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 — timers must never kill the loop
                logger.exception("voice_call: %s timer failed for %s", kind, call_id)

        self._timers[(call_id, kind)] = asyncio.get_running_loop().create_task(_fire())

    def _cancel_timer(self, call_id: str, kind: str) -> None:
        task = self._timers.pop((call_id, kind), None)
        if task:
            task.cancel()

    def _cancel_all_timers(self, call_id: str) -> None:
        for key in [k for k in self._timers if k[0] == call_id]:
            self._timers.pop(key).cancel()

    async def _on_ring_timeout(self, call_id: str) -> None:
        record = self.active.get(call_id)
        if record is None or record.state not in (CallState.INITIATED, CallState.RINGING):
            return
        logger.info("voice_call: ring timeout for %s", call_id)
        try:
            await self.provider.hangup_call(record)
        except Exception:  # noqa: BLE001
            logger.debug("voice_call: hangup after ring timeout failed", exc_info=True)
        await self._finalize(record, CallState.NO_ANSWER, "ring-timeout")

    async def _on_max_duration(self, call_id: str) -> None:
        record = self.active.get(call_id)
        if record is None:
            return
        logger.info("voice_call: max duration reached for %s", call_id)
        try:
            await self.provider.hangup_call(record)
        except Exception:  # noqa: BLE001
            logger.debug("voice_call: hangup at max duration failed", exc_info=True)
        await self._finalize(record, CallState.TIMEOUT, "max-duration")

    async def _on_silence_timeout(self, call_id: str) -> None:
        record = self.active.get(call_id)
        if record is None or record.is_terminal:
            return
        logger.info("voice_call: silence timeout for %s", call_id)
        try:
            await self.provider.hangup_call(record)
        except Exception:  # noqa: BLE001
            logger.debug("voice_call: hangup after silence failed", exc_info=True)
        await self._finalize(record, CallState.TIMEOUT, "silence-timeout")

    async def _on_notify_hangup(self, call_id: str) -> None:
        record = self.active.get(call_id)
        if record is None or record.is_terminal:
            return
        try:
            await self.provider.hangup_call(record)
        except Exception:  # noqa: BLE001
            logger.debug("voice_call: notify hangup failed", exc_info=True)
        await self._finalize(record, CallState.HANGUP_BOT, "notify-complete")

    def _arm_silence_timer(self, record: CallRecord) -> None:
        silence_s = self.config.timeouts.silence_s
        if silence_s and record.mode == "conversation":
            self._arm_timer(record.call_id, "silence", silence_s, self._on_silence_timeout)

    # -- finalization ----------------------------------------------------------

    async def _finalize(
        self, record: CallRecord, state: CallState, reason: Optional[str]
    ) -> None:
        if record.is_terminal:
            return
        self._cancel_all_timers(record.call_id)
        record.ended_at = time.time()
        record.end_reason = reason
        record.state = state
        self._persist(record)
        self._unregister(record)
        waiter = self._waiters.pop(record.call_id, None)
        if waiter and not waiter.done():
            waiter.set_exception(
                RuntimeError(f"call ended ({state.value}) while waiting for transcript")
            )
        self._drop_speech.discard(record.call_id)
        self._speak_ended_at.pop(record.call_id, None)
        speak_waiter = self._speak_done_waiters.pop(record.call_id, None)
        if speak_waiter is not None and not speak_waiter.done():
            speak_waiter.cancel()
        if self.on_call_ended is not None:
            try:
                await self.on_call_ended(record)
            except Exception:  # noqa: BLE001 — callbacks must not break teardown
                logger.exception("voice_call: on_call_ended callback failed")

    @staticmethod
    def _terminal_state_for_reason(
        reason: Optional[str], record: CallRecord
    ) -> CallState:
        lowered = (reason or "").lower()
        answered = record.answered_at is not None
        if "busy" in lowered:
            return CallState.BUSY
        if "voicemail" in lowered or "machine" in lowered:
            return CallState.VOICEMAIL
        if "no-answer" in lowered or "noanswer" in lowered:
            return CallState.NO_ANSWER
        if "timeout" in lowered:
            return CallState.TIMEOUT if answered else CallState.NO_ANSWER
        if "error" in lowered or "fail" in lowered:
            return CallState.FAILED
        if "hangup-bot" in lowered or "bot" in lowered:
            return CallState.HANGUP_BOT
        if "hangup" in lowered or "user" in lowered or "normal_clearing" in lowered:
            return CallState.HANGUP_USER if answered else CallState.NO_ANSWER
        return CallState.COMPLETED if answered else CallState.NO_ANSWER

    # -- event processing -------------------------------------------------------

    def _is_duplicate(self, event: NormalizedEvent) -> bool:
        key = event.dedupe_key
        if not key:
            return False
        now = time.time()
        # Evict expired / overflow entries.
        while self._processed and (
            len(self._processed) > _DEDUPE_CAP
            or next(iter(self._processed.values())) < now - 600
        ):
            self._processed.popitem(last=False)
        if key in self._processed:
            return True
        self._processed[key] = now
        return False

    def _find_call(self, event: NormalizedEvent) -> Optional[CallRecord]:
        if event.call_id and event.call_id in self.active:
            return self.active[event.call_id]
        if event.provider_call_id:
            call_id = self._by_provider_id.get(event.provider_call_id)
            if call_id:
                return self.active.get(call_id)
        return None

    def _stash_orphan(self, event: NormalizedEvent) -> None:
        pid = event.provider_call_id
        if not pid:
            return
        now = time.time()
        for key in [k for k, ts in self._orphan_seen.items() if ts < now - _ORPHAN_TTL_S]:
            self._orphans.pop(key, None)
            self._orphan_seen.pop(key, None)
        while len(self._orphans) > _ORPHAN_CAP:
            evicted, _ = self._orphans.popitem(last=False)
            self._orphan_seen.pop(evicted, None)
        self._orphans.setdefault(pid, []).append(event)
        self._orphan_seen[pid] = now

    async def _drain_orphans(self, provider_call_id: str) -> None:
        events = self._orphans.pop(provider_call_id, None)
        self._orphan_seen.pop(provider_call_id, None)
        for event in events or []:
            await self.process_event(event)

    async def process_event(self, event: NormalizedEvent) -> None:
        """Apply one normalized provider event to call state."""
        if self._is_duplicate(event):
            logger.debug("voice_call: duplicate event %s dropped", event.dedupe_key)
            return

        record = self._find_call(event)
        if record is None:
            # Carriers differ in the first inbound webhook they send
            # (Telnyx: call.initiated, Twilio: ringing, Plivo: answered via
            # StartApp) — auto-register on any of them.
            if event.direction == "inbound" and event.type in (
                EventType.CALL_INITIATED,
                EventType.CALL_RINGING,
                EventType.CALL_ANSWERED,
            ):
                record = await self._create_inbound(event)
            else:
                # Probably an outbound webhook racing initiate_call().
                self._stash_orphan(event)
                return
        if record is None or record.is_terminal:
            return

        # Some carriers swap identifiers mid-call (Plivo: dial returns a
        # request_uuid, webhooks then carry the real CallUUID). When the
        # event matched via our call_id, adopt the newer carrier id so call
        # control (speak/hangup) targets the right resource.
        if (
            event.provider_call_id
            and event.provider_call_id != record.provider_call_id
        ):
            if record.provider_call_id:
                self._by_provider_id.pop(record.provider_call_id, None)
            record.provider_call_id = event.provider_call_id
            self._by_provider_id[event.provider_call_id] = record.call_id
            self._persist(record)

        if event.type == EventType.CALL_INITIATED:
            pass  # creation handled above; outbound initiated in initiate_call()
        elif event.type == EventType.CALL_RINGING:
            self._transition(record, CallState.RINGING)
        elif event.type == EventType.CALL_ANSWERED:
            await self._on_answered(record)
        elif event.type == EventType.CALL_ACTIVE:
            self._transition(record, CallState.ACTIVE)
        elif event.type == EventType.CALL_SPEECH:
            await self._on_speech(record, event)
        elif event.type == EventType.CALL_DTMF:
            if event.digits:
                record.metadata.setdefault("dtmf_received", []).append(event.digits)
                self._persist(record)
        elif event.type == EventType.CALL_SILENCE:
            pass  # the silence timer is authoritative
        elif event.type == EventType.CALL_SPEAKING:
            self._transition(record, CallState.SPEAKING)
        elif event.type == EventType.CALL_SPEAK_ENDED:
            self._speak_ended_at[record.call_id] = time.time()
            waiter = self._speak_done_waiters.pop(record.call_id, None)
            if waiter is not None and not waiter.done():
                waiter.set_result(True)
        elif event.type == EventType.CALL_ENDED:
            await self._finalize(
                record,
                self._terminal_state_for_reason(event.reason, record),
                event.reason or "ended",
            )
        elif event.type == EventType.CALL_ERROR:
            if event.retryable:
                logger.warning(
                    "voice_call: retryable provider error on %s: %s",
                    record.call_id, event.reason,
                )
            else:
                await self._finalize(record, CallState.ERROR, event.reason or "error")

    async def _create_inbound(self, event: NormalizedEvent) -> Optional[CallRecord]:
        record = CallRecord(
            call_id=event.call_id or new_call_id(),
            provider=self.provider.name,
            direction="inbound",
            provider_call_id=event.provider_call_id,
            from_number=event.from_number,
            to_number=event.to_number,
            mode="conversation",
        )
        record.session_key = self._session_key(record)
        self._register(record)
        if self.prepare_call is not None:
            try:
                self.prepare_call(record)
            except Exception:  # noqa: BLE001
                logger.exception("voice_call: prepare_call hook failed")
        self._persist(record)
        self._arm_timer(
            record.call_id, "max", self.config.timeouts.max_call_s, self._on_max_duration
        )
        try:
            await self.provider.answer_call(record)
        except Exception as e:  # noqa: BLE001
            logger.warning("voice_call: answer failed for inbound %s: %s",
                           record.call_id, e)
            await self._finalize(record, CallState.FAILED, "answer-failed")
            return None
        logger.info(
            "voice_call: inbound call %s from %s", record.call_id, record.from_number
        )
        return record

    async def _on_answered(self, record: CallRecord) -> None:
        self._cancel_timer(record.call_id, "ring")
        record.answered_at = record.answered_at or time.time()
        if not self._transition(record, CallState.ANSWERED):
            return
        self._arm_timer(
            record.call_id, "max", self.config.timeouts.max_call_s, self._on_max_duration
        )

        if self._realtime_owns_audio(record):
            # The realtime bridge speaks the greeting/initial message and
            # consumes the caller's audio directly; carrier TTS or
            # transcription here would talk over the model.
            self._transition(record, CallState.LISTENING)
            return

        if record.direction == "outbound":
            initial = record.metadata.pop("initial_message", None)
            if initial:
                await self.speak(record.call_id, str(initial))
            if record.mode == "notify":
                delay = max(0, self.config.outbound.notify_hangup_delay_s)
                self._arm_timer(record.call_id, "notify", delay, self._on_notify_hangup)
            else:
                await self._start_listening(record)
        else:
            if self.config.inbound_greeting:
                await self.speak(record.call_id, self.config.inbound_greeting)
            await self._start_listening(record)

    async def _start_listening(self, record: CallRecord) -> None:
        try:
            await self.provider.start_listening(record)
        except Exception as e:  # noqa: BLE001
            logger.warning("voice_call: start_listening failed for %s: %s",
                           record.call_id, e)
        self._transition(record, CallState.LISTENING)
        self._arm_silence_timer(record)

    async def _on_speech(self, record: CallRecord, event: NormalizedEvent) -> None:
        text = (event.text or "").strip()
        if not text:
            return
        if not event.is_final:
            return  # partial transcripts are realtime-phase territory
        if record.call_id in self._drop_speech:
            # continue_call's question is still playing — the caller is
            # reacting to the PREVIOUS message ("thanks", "okay"), not
            # answering the question that hasn't finished yet.
            logger.info(
                "voice_call: discarding caller speech during question "
                "playback on %s: %.80r", record.call_id, text,
            )
            return
        record.transcript.append(
            TranscriptEntry(timestamp=event.timestamp, speaker="user", text=text)
        )
        self._persist(record)
        self._arm_silence_timer(record)

        waiter = self._waiters.pop(record.call_id, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(text)
            return
        if self.on_final_transcript is not None and record.mode == "conversation":
            asyncio.get_running_loop().create_task(
                self._dispatch_transcript(record, text)
            )

    async def _dispatch_transcript(self, record: CallRecord, text: str) -> None:
        try:
            await self.on_final_transcript(record, text)
        except Exception:  # noqa: BLE001 — agent failures must not kill the call
            logger.exception("voice_call: transcript callback failed for %s",
                             record.call_id)

    # -- actions ------------------------------------------------------------------

    async def initiate_call(
        self,
        to_number: Optional[str] = None,
        message: Optional[str] = None,
        mode: Optional[str] = None,
        from_number: Optional[str] = None,
    ) -> CallRecord:
        to_number = normalize_e164(to_number or self.config.to_number or "")
        if not to_number:
            raise ValueError(
                "no destination number (pass to_number or set to_number / "
                "VOICE_CALL_TO_NUMBER in config)"
            )
        if not is_e164(to_number):
            raise ValueError(f"to_number must be E.164, got {to_number!r}")
        from_number = normalize_e164(from_number or self.config.from_number or "")
        if not is_e164(from_number):
            raise ValueError(
                "no valid from_number (set from_number or VOICE_CALL_FROM_NUMBER)"
            )
        mode = mode or self.config.outbound.default_mode
        if mode not in ("notify", "conversation"):
            raise ValueError(f"mode must be notify|conversation, got {mode!r}")

        record = CallRecord(
            call_id=new_call_id(),
            provider=self.provider.name,
            direction="outbound",
            from_number=from_number,
            to_number=to_number,
            mode=mode,  # type: ignore[arg-type]
        )
        record.session_key = self._session_key(record)
        if message:
            record.metadata["initial_message"] = message
        self._register(record)
        if self.prepare_call is not None:
            try:
                self.prepare_call(record)
            except Exception:  # noqa: BLE001
                logger.exception("voice_call: prepare_call hook failed")
        self._persist(record)
        self._arm_timer(
            record.call_id, "ring", self.config.timeouts.ring_s, self._on_ring_timeout
        )
        try:
            provider_call_id = await self.provider.initiate_call(record)
        except Exception as e:
            logger.warning("voice_call: initiate failed: %s", e)
            await self._finalize(record, CallState.FAILED, f"initiate-failed: {e}")
            raise
        record.provider_call_id = provider_call_id
        self._by_provider_id[provider_call_id] = record.call_id
        self._persist(record)
        await self._drain_orphans(provider_call_id)
        return record

    def _require_call(self, call_id: str) -> CallRecord:
        record = self.active.get(call_id)
        if record is None:
            raise CallNotFoundError(call_id)
        return record

    async def speak(self, call_id: str, text: str) -> None:
        record = self._require_call(call_id)
        # Realtime calls: route through the bridge (the model speaks it);
        # carrier TTS would talk over the media stream. The bridge mirrors
        # the model's actual words into the transcript, so skip the append.
        if self._realtime_owns_audio(record) and self.realtime_speaker is not None:
            if await self.realtime_speaker(record, text):
                return
            logger.warning(
                "voice_call: realtime delivery failed for %s; falling back "
                "to carrier TTS", call_id,
            )
        # Speaking again on a notify call: hold the auto-hangup while the
        # new message plays, then re-arm it (notify semantics: hang up a
        # few seconds after the LAST message).
        is_notify = record.mode == "notify"
        if is_notify:
            self._cancel_timer(call_id, "notify")
        was_listening = record.state == CallState.LISTENING
        self._transition(record, CallState.SPEAKING)
        await self.provider.speak(record, text)
        record.transcript.append(
            TranscriptEntry(timestamp=time.time(), speaker="bot", text=text)
        )
        self._persist(record)
        if was_listening or record.mode == "conversation":
            self._transition(record, CallState.LISTENING)
            self._arm_silence_timer(record)
        elif is_notify and record.answered_at:
            self._arm_timer(
                call_id, "notify",
                max(0, self.config.outbound.notify_hangup_delay_s),
                self._on_notify_hangup,
            )

    async def continue_call(self, call_id: str, text: str) -> str:
        """Speak ``text`` and wait for the caller's next final utterance.

        On a notify call this upgrades the call to a conversation: the
        auto-hangup is cancelled and carrier transcription starts, so the
        caller's reply can come back.
        """
        record = self._require_call(call_id)
        existing = self._waiters.get(call_id)
        if existing is not None and not existing.done():
            raise RuntimeError(f"already waiting for a reply on {call_id}")
        if record.mode == "notify":
            logger.info(
                "voice_call: continue_call upgrades notify call %s to "
                "conversation", call_id,
            )
            self._cancel_timer(call_id, "notify")
            record.mode = "conversation"
            self._persist(record)
            # Prefer the full realtime upgrade (mid-call media stream →
            # speech-to-speech voice); fall back to carrier transcription.
            if not self._realtime_owns_audio(record):
                upgraded = False
                if self.upgrade_realtime is not None:
                    try:
                        upgraded = await self.upgrade_realtime(record)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "voice_call: realtime upgrade hook failed"
                        )
                if not upgraded:
                    try:
                        await self.provider.start_listening(record)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "voice_call: start_listening failed during notify "
                            "upgrade of %s: %s", call_id, e,
                        )

        realtime = self._realtime_owns_audio(record)
        spoke_at = time.time()
        if not realtime:
            # Anything the caller says while our question is still playing
            # is a reaction to the PREVIOUS message, not the answer —
            # discard it until playback completes.
            self._drop_speech.add(call_id)
        try:
            await self.speak(call_id, text)
            if not realtime:
                await self._wait_speak_done(call_id, spoke_at, text)
        finally:
            self._drop_speech.discard(call_id)

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters[call_id] = fut
        try:
            return await asyncio.wait_for(
                fut, timeout=self.config.timeouts.transcript_wait_s
            )
        finally:
            if self._waiters.get(call_id) is fut:
                self._waiters.pop(call_id, None)

    async def _wait_speak_done(
        self, call_id: str, spoke_at: float, text: str
    ) -> None:
        """Block until the carrier reports our TTS finished playing
        (call.speak.ended), or a duration estimate elapses for carriers
        that don't send one."""
        if self._speak_ended_at.get(call_id, 0.0) >= spoke_at:
            return  # already finished (short text / fast event)
        estimate = min(30.0, 2.0 + 0.08 * len(text))
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._speak_done_waiters[call_id] = fut
        try:
            await asyncio.wait_for(fut, timeout=estimate)
        except asyncio.TimeoutError:
            logger.debug(
                "voice_call: no speak.ended within %.1fs for %s — proceeding",
                estimate, call_id,
            )
        finally:
            if self._speak_done_waiters.get(call_id) is fut:
                self._speak_done_waiters.pop(call_id, None)

    async def send_dtmf(self, call_id: str, digits: str) -> None:
        record = self._require_call(call_id)
        await self.provider.send_dtmf(record, digits)
        record.metadata.setdefault("dtmf_sent", []).append(digits)
        self._persist(record)

    async def end_call(self, call_id: str, reason: str = "hangup-bot") -> None:
        record = self._require_call(call_id)
        try:
            await self.provider.hangup_call(record)
        except Exception as e:  # noqa: BLE001
            logger.warning("voice_call: provider hangup failed for %s: %s", call_id, e)
        await self._finalize(
            record, self._terminal_state_for_reason(reason, record), reason
        )
