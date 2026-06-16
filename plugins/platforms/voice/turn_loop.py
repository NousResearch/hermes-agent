"""Voice turn orchestrator.

One VoiceTurnLoop per live call. Turn-taking is driven by Deepgram Flux —
a single fused model that does transcription AND conversation-native turn
detection — so the loop reacts to events rather than running its own VAD /
endpoint model:

  start_of_turn     -> the user began speaking. If the agent is thinking,
                       speaking, or still has audio playing out, BARGE IN
                       (cancel the in-flight agent turn + mute TTS + drain
                       the transport). Otherwise pre-warm the next agent.
  eager_end_of_turn -> medium-confidence end (~185ms early) -> start the
                       agent speculatively on the transcript so far.
  turn_resumed      -> the user kept talking past the eager signal -> cancel
                       the speculative turn.
  end_of_turn       -> high-confidence end -> start the turn (or, if an eager
                       turn is already running, let it stand).

Per turn, once started:
  -> fresh AIAgent with stream_delta_callback (api_server pattern,
     gateway/platforms/api_server.py:1604-1657 + 3510-3554)
  -> deltas marshaled via loop.call_soon_threadsafe into an asyncio queue
  -> sentence buffer streams words to the per-turn Cartesia TTS socket, which
     synthesizes on sentence-final punctuation or an explicit <flush>
  -> audio out through the transport.

In-process barge-in is the differentiator: a barge-in calls agent.interrupt()
(the same call as api_server.py:4091) and STOPS the LLM mid-generation, then
mutes TTS and drains the transport. An external orchestrator can only mute the
audio while the agent keeps generating server-side.

The agent speaks first: run() executes a greeting turn before listening.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentence-final punctuation: tracks how much UNSYNTHESIZED text the TTS server
# is holding (it only synthesizes on [.!?] or an explicit <flush>); a long
# punctuation-free stretch gets a forced flush on idle.
_PUNCT_RE = re.compile(r"[.!?]")
LONG_FLUSH_LEN = 100
# If the first sentence hasn't reached sentence-final punctuation this long
# after its first words went to TTS, force a <flush> once so audio starts
# (prosody costs less than dead air on the first clause).
FIRST_SENTENCE_FLUSH_S = 0.6
TTS_END_TIMEOUT_S = 30.0
# Bound on awaiting the executor-run agent after a turn ends (normally the
# future is already done; after an interrupt a well-behaved agent returns
# promptly). The thread itself cannot be killed, but the event loop must never
# block on a generation that ignores its interrupt.
RUN_FUTURE_TIMEOUT_S = 60.0

DEFAULT_ALLOW_INTERRUPTIONS = True
DEFAULT_GREETING_PROMPT = (
    "(The user just joined a voice call with you. Greet them by voice in "
    "one short, warm sentence and ask how you can help.)"
)
DEFAULT_FILLER_TEXT = "One moment."

LISTENING, THINKING, SPEAKING = "listening", "thinking", "speaking"

# Durable telemetry sink: the voice/telemetry JSON records are the per-call
# audit trail (latency legs, billable ASR seconds), but log lines vanish at the
# container's default log level and on rotation. Every record ALSO appends to a
# JSONL file on the agent volume (/opt/data persists across restarts):
#   docker exec <agent> tail /opt/data/voice-telemetry.jsonl
# Fail-soft: an unwritable volume logs ONE warning and never raises into the
# call path.
TELEMETRY_SINK_PATH = "/opt/data/voice-telemetry.jsonl"
_sink_warned = False


def emit_telemetry(record: Dict[str, Any]) -> None:
    """Emit one voice/telemetry record: log it at WARNING (visible at the
    default container log level) and append it to the durable JSONL sink.
    Open-append-close per record — a handful per call, and the record must be
    on disk the moment the line is logged."""
    global _sink_warned
    line = json.dumps(record)
    logger.warning("voice/telemetry %s", line)
    try:
        with open(TELEMETRY_SINK_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        if not _sink_warned:
            _sink_warned = True
            logger.warning(
                "voice/telemetry sink unwritable (%s): %s — records stay "
                "in the process logs only", TELEMETRY_SINK_PATH, exc)


def _voice_model_name() -> Optional[str]:
    """Model name from the top-level ``voice_model:`` slot, for telemetry.
    None means voice turns ran on the main model."""
    try:
        from gateway.run import _voice_model_name as _core_voice_model_name
        return _core_voice_model_name()
    except Exception:
        return None


def _resolve_reasoning_override(extra: Dict[str, Any]) -> Optional[dict]:
    """platforms.voice.extra.reasoning_effort overrides the gateway reasoning
    effort for voice turns only: thinking tokens run before the first text
    delta, straight on top of first-audio latency. None = gateway default."""
    raw = (extra or {}).get("reasoning_effort")
    if raw is None or not str(raw).strip():
        return None
    from hermes_constants import parse_reasoning_effort

    parsed = parse_reasoning_effort(str(raw))
    if parsed is None:
        logger.warning("voice/turn: invalid reasoning_effort %r ignored", raw)
    return parsed


def _resolve_bool_extra(extra: Dict[str, Any], key: str, default: bool) -> bool:
    raw = (extra or {}).get(key)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off"):
            return False
    logger.warning("voice/turn: invalid %s %r ignored", key, raw)
    return default


def _resolve_max_iterations(extra: Dict[str, Any]) -> int:
    """platforms.voice.extra.max_turns caps the per-utterance agent loop;
    falls back to the api_server default (HERMES_MAX_ITERATIONS env, 90)."""
    raw = extra.get("max_turns")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning("voice/turn: invalid max_turns %r ignored", raw)
    return int(os.getenv("HERMES_MAX_ITERATIONS", "90"))


def _create_voice_agent(
    session_id: str,
    stream_delta_callback,
    tool_progress_callback,
    *,
    extra: Optional[Dict[str, Any]] = None,
):
    """Fresh agent per turn — mirrors api_server._create_agent
    (gateway/platforms/api_server.py:1029-1065) with platform='voice'.

    The model comes from the dedicated ``voice_model:`` slot (a top-level
    sibling to ``model:`` and ``fallback_model:``) when configured, otherwise
    the main model. Voice is a full agentic turn delivered in a latency-critical
    spoken modality, so it gets its own first-class model slot — not a
    per-platform override.
    """
    from run_agent import AIAgent
    from gateway.run import (
        GatewayRunner,
        _load_gateway_config,
        _resolve_gateway_model,
        _resolve_runtime_agent_kwargs,
        _resolve_voice_runtime_kwargs,
    )
    from hermes_cli.tools_config import _get_platform_tools

    user_config = _load_gateway_config()
    runtime_kwargs = _resolve_runtime_agent_kwargs()
    model = _resolve_gateway_model(user_config)
    # The voice_model: slot overrides endpoint/key/provider/api_mode + model for
    # voice turns. None = slot unset or `auto` -> use the main model.
    voice_override = _resolve_voice_runtime_kwargs()
    if voice_override is not None:
        model = voice_override.pop("model")
        runtime_kwargs = {**runtime_kwargs, **voice_override}
    return AIAgent(
        model=model,
        **runtime_kwargs,
        max_iterations=_resolve_max_iterations(extra or {}),
        quiet_mode=True,
        verbose_logging=False,
        enabled_toolsets=sorted(_get_platform_tools(user_config, "voice")),
        session_id=session_id,
        platform="voice",
        stream_delta_callback=stream_delta_callback,
        tool_progress_callback=tool_progress_callback,
        fallback_model=GatewayRunner._load_fallback_model(),
        reasoning_config=(_resolve_reasoning_override(extra or {})
                          or GatewayRunner._load_reasoning_config()),
    )


class VoiceTurnLoop:
    def __init__(self, stt, tts_factory, transport, *, extra: Dict[str, Any]):
        """tts_factory(on_audio) -> opened TTS turn (one per turn)."""
        self._stt = stt
        self._tts_factory = tts_factory
        self._transport = transport
        self._extra = extra or {}
        self._greeting = self._extra.get("greeting_prompt") or DEFAULT_GREETING_PROMPT
        self._filler = self._extra.get("filler_text") or DEFAULT_FILLER_TEXT
        self._session_id = f"voice-{uuid.uuid4().hex[:12]}"
        self._history: List[Dict[str, str]] = []
        # Holds a barged-in user_message that was never heard, re-queued for
        # the next turn so the caller's words are not lost.
        self._pending_text: List[str] = []
        self._state = LISTENING
        self._allow_interruptions = _resolve_bool_extra(
            self._extra, "allow_interruptions", DEFAULT_ALLOW_INTERRUPTIONS)
        # Telemetry: one structured record per turn, emitted as a JSON log line
        # when the turn ends; per-call summary on stop.
        self._tel: Dict[str, Any] = {}
        self._call_stats: List[Dict[str, Any]] = []
        # Per-turn delta/tool sinks behind stable trampolines, so an agent can
        # be CONSTRUCTED before its turn starts and still stream into the right
        # turn's queue.
        self._delta_sink = None
        self._tool_sink = None
        # Pre-constructed agent for the next turn (concurrent.futures.Future
        # from run_in_executor), made while the user is still speaking.
        self._spare_agent_future = None
        # One-element list so the executor thread can publish the live agent for
        # cross-thread interrupt (api_server agent_ref pattern, 3505-3508).
        self._agent_ref: List[Optional[Any]] = [None]
        # Per-turn interrupt latch: a barge-in landing while the executor thread
        # is still CONSTRUCTING the agent (agent_ref[0] unset) must not be lost —
        # _run checks it right after construction.
        self._interrupt_latch: Optional[threading.Event] = None
        self._tts = None
        self._turn_task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()
        # Timing instrumentation: per-turn sequence + reference timestamp. All
        # voice/timing logs report ms since this reference.
        self._turn_seq = 0
        self._t_ref: Optional[float] = None

    def _mark(self, leg: str, **fields: Any) -> None:
        """INFO timing log: ms since the current turn's reference point."""
        now = time.monotonic()
        ref = self._t_ref if self._t_ref is not None else now
        extra = "".join(f" {k}={v}" for k, v in fields.items())
        logger.info("voice/timing turn=%d %s t=+%.0fms%s",
                    self._turn_seq, leg, (now - ref) * 1000.0, extra)

    # -- telemetry ----------------------------------------------------------

    def _tel_open(self) -> None:
        if not self._tel:
            self._tel = {"opened_at": time.monotonic()}
            reset = getattr(self._transport, "reset_write_mark", None)
            if reset is not None:
                reset()

    def _tel_set(self, key: str, value: Any = None) -> None:
        """Record a telemetry timestamp (default: now) or value. First write
        wins — retries must not overwrite the first leg."""
        self._tel_open()
        self._tel.setdefault(key, time.monotonic() if value is None else value)

    def _tel_emit(self, status: str) -> None:
        """Emit the per-turn structured JSON log line and bank the record for
        the per-call summary. All offsets are ms since speech_end (end of the
        user's utterance)."""
        tel, self._tel = self._tel, {}
        if not tel:
            return
        speech_end = tel.get("speech_end") or tel.get("vad_end") \
            or tel.get("opened_at")
        first_written = getattr(self._transport, "first_write_t", None)

        def off(t: Optional[float]) -> Optional[int]:
            return None if t is None else int(round((t - speech_end) * 1000))

        perceived = off(first_written)
        substantive = off(tel.get("tts_first_audio"))
        record = {
            "event": "voice_turn",
            "turn": self._turn_seq,
            "status": status,
            "eager_start": bool(tel.get("eager_start")),
            "turn_detector": tel.get("turn_detector", self._stt.provider),
            "stt_provider": self._stt.provider,
            "eot_reason": tel.get("eot_reason"),
            "vad_end_ms": off(tel.get("vad_end")),
            "agent_start_ms": off(tel.get("agent_start")),
            "first_delta_ms": off(tel.get("first_delta")),
            "first_sentence_ms": off(tel.get("first_sentence")),
            "tts_first_audio_ms": substantive,
            "first_frame_written_ms": perceived,
            "totals": {
                "perceived_first_audio_ms": perceived,
                "substantive_first_audio_ms": substantive,
                "turn_total_ms": off(time.monotonic()),
            },
        }
        emit_telemetry(record)
        self._call_stats.append(record)

    def _emit_call_summary(self) -> None:
        turns = [r for r in self._call_stats if r["vad_end_ms"] is not None]

        def stats(key: str) -> Optional[Dict[str, int]]:
            vals = sorted(r["totals"][key] for r in turns
                          if r["totals"][key] is not None)
            if not vals:
                return None
            return {"median": vals[len(vals) // 2],
                    "p90": vals[min(len(vals) - 1, int(len(vals) * 0.9))],
                    "n": len(vals)}

        raw_effort = self._extra.get("reasoning_effort")
        record = {
            "event": "voice_call_summary",
            "session": self._session_id,
            "turns": len(turns),
            "stt_provider": self._stt.provider,
            "model_override": _voice_model_name(),
            "reasoning_effort": (str(raw_effort).strip()
                                 if raw_effort is not None
                                 and str(raw_effort).strip()
                                 else "gateway-default"),
            "eager_starts": sum(1 for r in turns if r["eager_start"]),
            "barge_ins": sum(1 for r in self._call_stats
                             if r["status"] == "interrupted"),
            "perceived_first_audio_ms": stats("perceived_first_audio_ms"),
            "substantive_first_audio_ms": stats("substantive_first_audio_ms"),
        }
        emit_telemetry(record)

    # -- agent plumbing -----------------------------------------------------

    def _delta_tramp(self, delta: str) -> None:
        sink = self._delta_sink
        if sink is not None:
            sink(delta)

    def _tool_tramp(self, event_type, tool_name=None, preview=None,
                    args=None, **kwargs) -> None:
        sink = self._tool_sink
        if sink is not None:
            sink(event_type, tool_name=tool_name, preview=preview,
                 args=args, **kwargs)

    def _make_agent(self):
        """Construct a turn agent (executor thread). Construction does not need
        the user message, so it can run before the turn starts."""
        return _create_voice_agent(
            self._session_id, self._delta_tramp, self._tool_tramp,
            extra=self._extra)

    def _preconstruct_agent(self) -> "concurrent.futures.Future":
        """Build the next turn's agent on a worker thread. Returns a concurrent
        Future (thread-safe .result() from the turn executor)."""
        fut: "concurrent.futures.Future" = concurrent.futures.Future()

        def _build() -> None:
            try:
                fut.set_result(self._make_agent())
            except BaseException as e:    # surface in the consuming turn
                fut.set_exception(e)

        threading.Thread(
            target=_build, name="voice-agent-prewarm", daemon=True).start()
        return fut

    # -- main ---------------------------------------------------------------

    async def run(self) -> None:
        consumer = asyncio.create_task(self._consume_flux())
        canned = self._extra.get("greeting_text")
        if canned:
            # Canned greeting: instant TTS, no LLM round-trip. The first real
            # turn's agent is pre-warmed while the greeting plays.
            self._spare_agent_future = self._preconstruct_agent()
            self._turn_task = asyncio.create_task(self._speak_canned(canned))
        else:
            # Agent speaks first: greeting turn before listening.
            self._start_turn(self._greeting, record_user=False)
        await self._stopped.wait()
        consumer.cancel()
        try:
            await consumer
        except (asyncio.CancelledError, Exception):
            pass

    async def stop(self) -> None:
        await self._barge_in()           # kill any in-flight turn
        self._tel_emit("call-ended")
        self._emit_call_summary()
        self._stopped.set()

    async def _speak_canned(self, text: str) -> None:
        """Speak fixed text through TTS without an agent turn (greeting)."""
        self._t_ref = time.monotonic()
        self._mark("canned-greeting-start", chars=len(text))
        self._state = SPEAKING
        try:
            self._tts = await self._tts_factory(self._transport.send_audio)
            await self._tts.send_text(text)
            await asyncio.wait_for(self._tts.end(), timeout=TTS_END_TIMEOUT_S)
            self._history.append({"role": "assistant", "content": text})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("voice/turn: canned greeting failed")
        finally:
            self._tts = None
            if self._state != LISTENING:
                self._state = LISTENING

    # -- Flux turn pump -----------------------------------------------------

    async def _consume_flux(self) -> None:
        """Deepgram Flux turn path: the STT emits conversation-native events
        (normalized to start_of_turn / eager_end_of_turn / turn_resumed /
        end_of_turn) and we drive the turn directly. Our in-process barge-in
        EXECUTION (cancel LLM + mute TTS + clear transport, via _barge_in) is
        the differentiator; Flux just supplies the trigger. See
        deepgram_flux_stt.py for the event names."""
        eager_live = False
        async for ev in self._stt.events():
            kind = ev.get("event")
            text = (ev.get("transcript") or "").strip()
            if kind == "start_of_turn":
                # Barge-in if the agent is thinking/speaking OR still has audio
                # PLAYING OUT — state flips to LISTENING the instant the audio
                # is queued, but the transport plays the tail for seconds after,
                # during which the caller is still hearing the agent.
                speaking = (self._state in (THINKING, SPEAKING)
                            or self._transport.is_playing())
                if speaking:
                    if self._allow_interruptions:
                        self._mark("flux-barge-in", state=self._state)
                        await self._barge_in()
                        eager_live = False
                elif self._spare_agent_future is None:
                    # Pre-warm the agent while the user speaks so construction
                    # (~0.5s) is off the critical path by eager/end-of-turn.
                    self._spare_agent_future = self._preconstruct_agent()
            elif kind == "eager_end_of_turn":
                if text and self._state == LISTENING:
                    eager_live = True
                    self._begin_flux_turn(text, eager=True, reason="eager")
            elif kind == "turn_resumed":
                if eager_live:
                    self._mark("flux-turn-resumed")
                    await self._barge_in()   # cancel the speculative turn
                    eager_live = False
            elif kind == "end_of_turn":
                if eager_live:
                    # The speculative turn is already running; Flux guarantees
                    # the EndOfTurn transcript matches the eager one, so it
                    # stands as-is.
                    eager_live = False
                elif text and self._state == LISTENING:
                    self._begin_flux_turn(text, eager=False, reason="endpoint")
            # "update": incremental partial transcripts — not needed here.

    def _begin_flux_turn(self, text: str, *, eager: bool, reason: str) -> None:
        """Open the per-turn telemetry record and start the agent for a
        Flux-detected turn."""
        self._turn_seq += 1
        self._t_ref = time.monotonic()
        self._tel_set("vad_end", self._t_ref)
        self._tel["speech_end"] = self._t_ref
        self._tel_set("turn_detector", self._stt.provider)
        self._tel_set("eot_reason", reason)
        if eager:
            self._tel_set("eager_start", True)
        self._mark("flux-turn-end", reason=reason, eager=eager)
        agent_future = self._spare_agent_future
        self._spare_agent_future = None
        self._start_turn(text, record_user=True, agent_future=agent_future)

    # -- agent turn ---------------------------------------------------------

    def _start_turn(self, user_message: str, *, record_user: bool,
                    agent_future=None) -> None:
        self._state = THINKING
        if self._t_ref is None:          # greeting turn has no vad-end
            self._t_ref = time.monotonic()
        self._mark("turn-start", chars=len(user_message))
        # Fresh latch BEFORE the task exists so a barge-in can never land
        # between turn creation and the executor publishing the agent.
        self._interrupt_latch = threading.Event()
        self._turn_task = asyncio.create_task(
            self._execute_turn(user_message, record_user=record_user,
                               agent_future=agent_future))

    async def _execute_turn(self, user_message: str, *, record_user: bool,
                            agent_future=None) -> None:
        loop = asyncio.get_running_loop()
        q: "asyncio.Queue[tuple]" = asyncio.Queue()
        agent_ref = self._agent_ref
        agent_ref[0] = None
        interrupt_latch = self._interrupt_latch

        # Threadsafe marshal — mirrors api_server._enqueue (1619-1631).
        def _enqueue(item: tuple) -> None:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            try:
                if running_loop is loop:
                    q.put_nowait(item)
                else:
                    loop.call_soon_threadsafe(q.put_nowait, item)
            except RuntimeError:
                pass

        # Mirrors api_server._delta (1633-1635).
        first_delta_seen = threading.Event()

        def _delta(delta: str) -> None:
            if delta:
                if not first_delta_seen.is_set():
                    first_delta_seen.set()
                    self._mark("first-delta")
                    self._tel_set("first_delta")
                _enqueue(("delta", delta))

        # Signature mirrors api_server._tool_progress (1637).
        def _tool_progress(event_type, tool_name=None, preview=None,
                           args=None, **kwargs) -> None:
            if event_type == "tool.started":
                _enqueue(("tool", tool_name))

        # Route the stable trampolines into THIS turn's queue. One live turn at
        # a time, so plain assignment is safe.
        self._delta_sink = _delta
        self._tool_sink = _tool_progress

        def _run():
            # Executor body mirrors api_server._run_agent (3510-3554).
            if agent_future is not None:
                # Pre-constructed; bounded so a hung construction can never
                # wedge the turn executor forever.
                agent = agent_future.result(timeout=RUN_FUTURE_TIMEOUT_S)
            else:
                self._mark("agent-constructing")
                agent = self._make_agent()
            agent_ref[0] = agent
            self._mark("agent-start")
            self._tel_set("agent_start")
            if interrupt_latch.is_set():
                # A barge-in/stop landed while the agent was still being
                # constructed: interrupt before the first iteration runs.
                try:
                    agent.interrupt("user barge-in (voice)")
                except Exception:
                    pass
            try:
                return agent.run_conversation(
                    user_message=user_message,
                    conversation_history=list(self._history),
                    task_id=self._session_id,
                )
            finally:
                _enqueue(("done", None))

        run_future = loop.run_in_executor(None, _run)
        # Pre-warm the NEXT turn's agent while this one runs: construction
        # (~0.5s of config/toolset loading) drops off the critical path of every
        # subsequent turn.
        if self._spare_agent_future is None:
            self._spare_agent_future = self._preconstruct_agent()

        interrupted = False
        first_audio_seen = False

        async def _on_audio(pcm: bytes) -> None:
            nonlocal first_audio_seen
            if not first_audio_seen:
                first_audio_seen = True
                self._mark("first-tts-audio-chunk", bytes=len(pcm))
                self._tel_set("tts_first_audio")
            await self._transport.send_audio(pcm)

        try:
            self._tts = await self._tts_factory(_on_audio)
            self._mark("tts-socket-open")
            self._state = SPEAKING
            await self._pump_deltas_to_tts(q)
            try:
                # end() blocks until the server's final audio; bounded wait.
                await asyncio.wait_for(self._tts.end(), timeout=TTS_END_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.warning("voice/turn: tts end timed out; aborting socket")
                await self._tts.abort()
        except asyncio.CancelledError:
            interrupted = True
            raise
        finally:
            self._tts = None
            if interrupted and agent_ref[0] is not None:
                # The latch may have raced agent construction; now that the
                # agent surely exists, re-issue the direct interrupt so the
                # bounded await below resolves promptly.
                try:
                    agent_ref[0].interrupt("user barge-in (voice)")
                except Exception:
                    pass
            try:
                result = await asyncio.wait_for(
                    run_future, timeout=RUN_FUTURE_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.error(
                    "voice/turn: agent run did not finish within %.0fs after "
                    "the turn ended; abandoning executor thread (it may still "
                    "be running and burning tokens)", RUN_FUTURE_TIMEOUT_S)
                result = None
            except Exception:
                logger.exception("voice/turn: agent run failed")
                result = None
            if result and isinstance(result, dict) and not interrupted:
                final = result.get("final_response", "")
                if record_user:
                    self._history.append({"role": "user", "content": user_message})
                if final:
                    self._history.append({"role": "assistant", "content": final})
            elif interrupted and record_user and not first_audio_seen:
                # Barged in before the user heard ANY of the reply: their words
                # must not vanish — feed them into the next turn.
                self._pending_text.insert(0, user_message)
            agent_ref[0] = None
            self._delta_sink = None
            self._tool_sink = None
            if self._state != LISTENING:
                self._state = LISTENING
            self._tel_emit("interrupted" if interrupted else "ok")

    async def _pump_deltas_to_tts(self, q: "asyncio.Queue[tuple]") -> None:
        """Stream deltas to TTS at word granularity.

        Cartesia synthesizes on sentence-final punctuation (or an explicit
        <flush>), so forwarding words as they arrive lets audio start the moment
        the model emits the first '.', instead of after the whole reply. Words
        are never split across messages; ``buf`` holds at most one partial word.
        ``unsynthesized`` tracks text the server is holding without a sentence
        boundary — a long punctuation-free stretch is force-flushed on idle."""
        buf = ""
        unsynthesized = 0
        filler_spoken = False
        first_text_sent = False
        first_send_t: Optional[float] = None
        punct_ever = False
        first_flush_done = False

        def _mark_first_send(kind_: str, text: str) -> None:
            nonlocal first_text_sent, first_send_t
            if not first_text_sent:
                first_text_sent = True
                first_send_t = time.monotonic()
                self._mark("first-text-to-tts", kind=kind_, chars=len(text))

        async def _send(fragment: str, kind_: str) -> None:
            nonlocal unsynthesized, punct_ever
            if not fragment:
                return
            _mark_first_send(kind_, fragment)
            await self._tts.send_text(fragment)
            m = None
            for m in _PUNCT_RE.finditer(fragment):
                pass
            if m is not None:
                if not punct_ever:
                    self._tel_set("first_sentence")
                punct_ever = True
                unsynthesized = len(fragment) - m.end()
            else:
                unsynthesized += len(fragment)

        async def _maybe_first_flush() -> None:
            # First-audio guard: if the model's first sentence is dragging on
            # with no sentence-final punctuation, force one <flush> so the user
            # hears SOMETHING (~350ms later) instead of dead air.
            nonlocal first_flush_done, unsynthesized
            if (first_send_t is not None and not punct_ever
                    and not first_flush_done and unsynthesized > 0
                    and time.monotonic() - first_send_t
                    > FIRST_SENTENCE_FLUSH_S):
                first_flush_done = True
                self._mark("first-sentence-forced-flush", held_chars=unsynthesized)
                self._tel_set("first_sentence")
                await self._tts.send_text("<flush>")
                unsynthesized = 0

        while True:
            try:
                kind, payload = await asyncio.wait_for(q.get(), timeout=0.2)
            except asyncio.TimeoutError:
                await _maybe_first_flush()
                if buf and len(buf) + unsynthesized > LONG_FLUSH_LEN:
                    await _send(buf, "long-flush")
                    buf = ""
                if unsynthesized > LONG_FLUSH_LEN:
                    await self._tts.send_text("<flush>")
                    unsynthesized = 0
                continue
            if kind == "tool":
                if not filler_spoken:
                    filler_spoken = True
                    _mark_first_send("filler", self._filler)
                    await self._tts.send_filler(self._filler)
            elif kind == "delta":
                buf += payload
                # Send everything up to the last whitespace (complete words);
                # keep the trailing partial word.
                cut = max(buf.rfind(" "), buf.rfind("\n"), buf.rfind("\t"))
                if cut >= 0:
                    await _send(buf[:cut + 1], "words")
                    buf = buf[cut + 1:]
                await _maybe_first_flush()
            elif kind == "done":
                if buf.strip():
                    await _send(buf, "done-tail")
                return

    # -- barge-in -----------------------------------------------------------

    async def _barge_in(self) -> None:
        t0 = time.monotonic()
        # Latch first: if the executor thread is still constructing the agent,
        # _run picks this up right after construction and interrupts before the
        # first iteration (the direct path below would miss it).
        latch = self._interrupt_latch
        if latch is not None:
            latch.set()
        agent = self._agent_ref[0]
        if agent is not None:
            try:
                agent.interrupt("user barge-in (voice)")   # api_server.py:4091
            except Exception:
                pass
        tts = self._tts
        if tts is not None:
            # Mute synchronously FIRST: no further TTS audio may reach the
            # transport while abort()'s socket close is in flight.
            tts.mute()
        self._transport.clear_output()
        logger.info("voice/timing barge-in audio-cleared in %.0fms",
                    (time.monotonic() - t0) * 1000.0)
        if tts is not None:
            await tts.abort()
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except (asyncio.CancelledError, Exception):
                pass
        self._turn_task = None
        self._state = LISTENING
        logger.info("voice/turn: barge-in handled in %.0fms",
                    (time.monotonic() - t0) * 1000.0)
