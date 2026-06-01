# SimpleX Streaming Voice — Reflex Foundation (Slice 1) Design

Date: 2026-05-31
Status: IMPLEMENTED (Slice 1 complete) — 2026-05-31. Reflex core proven in simulation
(scenarios A–E,G + F); 240 passed / 1 skipped (deferred pipecat smoke); default engine
turn_based (zero behavior change). Pipecat install + real STT/TTS/VAD + live device test
are follow-on slices (see §9). Original design below.
Author: Bryan Murphy + Claude (brainstorming session)
Supersedes-context: `docs/plans/2026-05-31-simplex-native-voice-handoff.md` (current turn-based native call path)

## 1. Purpose & Outcome

The long-term goal is to talk to Hermes Agent over SimpleX (voice now, video later) as
seamlessly as talking to a person: Hermes remembers conversations, makes async tool
calls mid-conversation, allows interruptions, knows when to stop speaking, asks
clarifying questions, and dynamically manages conversational state — all over the
existing E2E-encrypted SimpleX native call path on Bryan's Mac.

That vision spans several independent subsystems. This spec covers **only the first
slice: the streaming reflex foundation**, proven entirely in simulation. Everything
else is explicitly deferred to follow-on spec→plan→implement cycles (see §9).

**Slice 1 outcome:** a streaming, full-duplex voice loop where Hermes remains the
reasoning brain and a framework supplies the human "reflexes" (turn detection,
barge-in, interruption), with correctness proven by a deterministic simulation suite
(scenarios A–G) before any live device test.

## 2. Key Decisions (locked during brainstorming)

1. **Architecture seam:** streaming foundation first (not incremental patches on the
   turn-based loop).
2. **Cloud allowed for quality, local fallback** — every STT/TTS/turn-detection
   backend sits behind a port, switchable per-profile; cloud audio paths are gated
   off by default and never silently become the only path.
3. **Brain topology:** Hermes stays the only reasoning brain. The framework owns
   reflexes (frame flow, VAD, turn detection, interruption signals, backpressure,
   TTS cancellation). A realtime speech-to-speech model must NOT become the default
   brain in this slice.
4. **Slice 1 boundary:** reflexes only (see §3, §9).
5. **Framework:** Pipecat, approved for Slice 1. Rationale: SimpleX gives a raw
   `aiortc` peer connection (no LiveKit room/SFU); Pipecat adapts cleanly around raw
   audio frames and a custom transport. LiveKit Agents remains a strong production
   option if/when room/SFU infrastructure is wanted — out of scope here.
6. **Open decisions A–D (resolved):**
   - **A — Pipecat on Python 3.13/3.14:** ship Slice 1 with **fakes** for VAD/Smart
     Turn; postpone real turn-detection to a follow-on slice (active `onnxruntime`
     wheel gaps on 3.14, which Smart Turn depends on). Pipecat still used for frame
     plumbing; real services plug in later when wheels are ready.
   - **B — Cancellation depth:** wire the real cooperative interrupt now.
     `AIAgent.interrupt()` / `_interrupt_requested` already exist and are checked
     throughout `agent/conversation_loop.py`; `CancellationScope` drives it so
     abandoned turns stop cleanly rather than only being discarded.
   - **C — `CallTurnRecord` storage:** tracer JSONL only for Slice 1. No `SessionDB`
     writes (that is the later memory slice); WP5 defers persistence.
   - **D — Real Silero+Smart Turn adapter:** stretch goal inside WP4, included only
     if WP0 proves wheels are available; otherwise fakes suffice for acceptance and
     real VAD/Smart Turn integration becomes its own slice.

## 3. Scope

### In scope (Slice 1)
- New sibling slice `gateway/calls/native/streaming/` (turn-based path untouched).
- Frozen value types + Protocol ports (the contract).
- Real `InterruptionPolicy` (pure), `HeardSpanLedger`, `CancellationScope`,
  `StreamingCallSession` (transport-agnostic reflex core).
- Real `HermesSyncBrain` adapter wrapping `AIAgent.run_conversation` via
  `asyncio.to_thread`, with cooperative interrupt and deferred persistence.
- `StreamingCallTracer` extending existing `tracing.py` (playhead + redaction).
- Fakes: transport, turn-detection, STT, TTS, brain + a deterministic `VirtualClock`.
- Acceptance simulation suite: scenarios A–G + config-fallback (F).
- `calls.native.engine` selector (`turn_based` default | `streaming`).
- CLI debug command `calls simplex-simulate-stream`.
- A 1–2 test Pipecat production-transport smoke (only if WP0 wheels permit).
- `ast-grep` + `kit` tooling installed and wired into the per-WP gate.

### Out of scope (follow-on slices)
- Real cloud STT/TTS (AssemblyAI/Deepgram, Cartesia/ElevenLabs).
- Real local STT/TTS upgrades (WhisperLiveKit-MLX, Kokoro/Piper streaming).
- Real Silero VAD / Smart Turn integration (unless WP0 wheel stretch in D succeeds).
- Proper-noun STT biasing (the "Leawood→Leewood" fix).
- Real async tool calls during a call (weather grounding) + filler masking.
- Cross-call conversation memory / `SessionDB` integration.
- Trace privacy operator toggle beyond existing defaults.
- Live inbound SimpleX production acceptance.
- Video.

Provider port shapes are designed now so these adapters drop in without rewrites.

## 4. Architecture

### 4.1 Slice placement
A new sibling slice `gateway/calls/native/streaming/` lives beside the existing
turn-based `gateway/calls/native/`. The turn-based engine is NOT rewritten in place;
it remains fallback and reference. `application.py` selects the engine via
`calls.native.engine` (`turn_based` default | `streaming`).

### 4.2 Hexagonal seams
```
   SimpleX/aiortc                streaming/ (the slice)
   ───PCM16──►  ① SimplexAudioTransport (adapter → Pipecat BaseTransport)
                       │ InputAudioRawFrame
                       ▼
                  ② Pipecat Pipeline ◄── ③ TurnDetection / ⑤ ReflexPolicy
                     VAD→SmartTurn→STT→[brain]→TTS
                       │                 ▲
                       │           ④ HermesBrainPort (wraps run_conversation)
                       ▼
   ◄──PCM16──  ① SimplexAudioTransport.output (OutputAudioRawFrame)
                  ⑥ StreamingCallTracer (extends tracing.py; playhead + redaction)
```

### 4.3 The "two clocks" resolution
Pipecat owns frame plumbing (VAD cadence, backpressure, the interrupt frame). All
**correctness-critical decisions** — interruption policy, playhead/heard-span
accounting, cancellation semantics, history truncation — live in **pure,
Pipecat-independent components** behind ports. The simulation drives a
transport-agnostic `StreamingCallSession` through fakes (acceptance A–G without
Pipecat). The Pipecat `FrameProcessor`/`LLMService` adapters delegate into the same
session. We isolate the decisions Pipecat should not own; we do not reimplement its
pipeline.

### 4.4 Brain entry point (verified)
`AIAgent.chat(message, stream_callback=None) -> str` is a thin wrapper over
`AIAgent.run_conversation(user_message, system_message=None, conversation_history=None,
task_id=None, stream_callback=None, persist_user_message=None) -> dict`. The brain
adapter wraps **`run_conversation`** (not `chat`) because we need:
- the dict result (`final_response`, `messages`),
- `persist_user_message` + deferred persistence control (constraint: never persist
  abandoned text),
- `stream_callback` as the ready-made future hook for `BrainEvent.PARTIAL_TEXT`
  (v1 yields one `FINAL_TEXT`).

Cooperative cancellation uses the existing `AIAgent.interrupt()` /
`_interrupt_requested` (checked in `agent/conversation_loop.py`).

## 5. Event Model, Types & Ports

All in `gateway/calls/native/streaming/`, repo style (`from __future__ import
annotations`, frozen dataclasses, `Protocol`). Driven by an injectable clock — never
`time.time()`.

### 5.1 Enums
- `TranscriptKind`: PARTIAL, FINAL
- `TurnEventKind`: USER_SPEECH_STARTED, USER_SPEECH_STOPPED, ENDPOINT_DETECTED,
  POSSIBLE_BACKCHANNEL
- `BrainEventKind`: PARTIAL_TEXT, TOOL_STATUS, FINAL_TEXT, ERROR
- `TtsEventKind`: AUDIO, MARK, DONE, CANCELLED
- `InterruptionAction`: INTERRUPT, IGNORE, RESUME, WAIT
- `TurnEndReason`: COMPLETED, BARGED_IN, FALSE_INTERRUPTION, BRAIN_ERROR

### 5.2 Value types (all frozen)
- `MediaFormat(sample_rate, channels=1, frame_ms=20)`
- `AudioFrame(pcm16, media, timestamp_ms, seq)` + `duration_ms` property
- `WordTiming(word, start_ms, end_ms, confidence=1.0)`
- `TranscriptEvent(call_id, kind, text, start_ms=0, end_ms=0, stability=1.0, words=(), provider="")`
- `TurnEvent(call_id, kind, at_ms, speech_duration_ms=0, vad_confidence=0.0, endpoint_confidence=0.0, source="")`
- `BrainEvent(call_id, kind, text="", tool_name="", error_code="")` + `is_final` property
- `PlaybackMark(call_id, char_offset, text_so_far, at_ms, boundary="word")` —
  `char_offset` is measured against the **full** response text (consistent with
  `HeardSpanLedger.note_flush(flush, full_text)`), not the currently-synthesized
  span; `text_so_far` is `full_text[:char_offset]`.
- `TtsAudioEvent(call_id, kind, frame=None, mark=None, span_text="", span_start_char=0, span_end_char=0)`
- `InterruptionParams(min_speech_ms=350, min_words=2, backchannel_max_ms=600, false_interruption_timeout_ms=2000)`
- `EndpointParams(vad_confidence=0.7, start_secs=0.2, stop_secs=0.2, endpoint_threshold=0.5, max_delay_ms=3000)`
- `InterruptionSignal(call_id, at_ms, assistant_speaking, turn_event, latest_partial, playhead, params, ms_since_speech_start=0, ms_since_assistant_silent_partial=0)`
- `InterruptionDecision(action, reason, at_ms)`
- `StreamingCallContext(call_id, contact_id, session_id, media, interruption=InterruptionParams(), endpoint=EndpointParams(), debug=VoiceDebugTracePolicy())`
- `FlushResult(dropped_frames, dropped_ms, last_sent_mark)`
- `CallTurnRecord(call_id, turn_index, user_transcript, assistant_heard_text, assistant_abandoned_text="", interrupted=False, ended_reason=COMPLETED)` — the durable, playback-aware artifact (tracer JSONL only in Slice 1).

### 5.3 Cancellation
`CancellationScope` (cooperative; not frozen):
`cancel(reason)`, `cancelled` (prop), `reason` (prop), `raise_if_cancelled()` (raises
`CallTurnCancelled`), `add_listener(cb)`. On barge-in: mark cancelled → flush/cancel
TTS → call `AIAgent.interrupt()` to stop the brain at its next loop boundary →
discard any result → never persist. The worker thread may finish unobserved; its
output is discarded.

**Flush-vs-interrupt ordering (resolves the B/G ambiguity).** Stopping outbound audio
and stopping the brain are two separate acts with different triggers:
1. A transient VAD trigger while the assistant is speaking may **immediately**
   `flush_outbound(reason="vad_trigger")` — a fast reflex so the caller stops hearing
   the assistant. This alone does NOT interrupt the brain.
2. The brain interrupt (`AIAgent.interrupt()` via `CancellationScope.cancel`) is
   **deferred** until `InterruptionPolicy` returns `INTERRUPT` (i.e. `min_speech_ms`
   AND `min_words` satisfied).
3. If no qualifying transcript arrives within `false_interruption_timeout_ms`, the
   policy returns `RESUME`: the brain was **never** interrupted, and TTS resumes
   synthesis from `PlaybackMark.char_offset` of the retained full response text — no
   duplicated prefix (Scenario G). Resume therefore requires retaining `full_text`
   and the last `PlaybackMark` for the in-flight assistant turn.

### 5.4 Ports (Protocols)
- `AudioTransportPort`: `media` (prop), `push_inbound(frame)`, `inbound() ->
  AsyncIterator[AudioFrame]`, `emit_outbound(frame)`, `flush_outbound(reason) ->
  FlushResult` (the barge-in cut), `close()`.
- `TurnDetectionPort`: `observe(frame) -> tuple[TurnEvent, ...]`, `reset()`.
- `InterruptionPolicyPort`: `decide(signal) -> InterruptionDecision` (PURE).
- `SpeechToTextPort`: `start(ctx)`, `push(frame)`, `events() ->
  AsyncIterator[TranscriptEvent]`, `finalize() -> TranscriptEvent | None`,
  `cancel()`, `close()`.
- `TextToSpeechPort`: `synthesize(text, ctx, scope) ->
  AsyncIterator[TtsAudioEvent]`, `cancel()`, `flush()`.
- `HermesBrainPort`: `respond(turn, ctx, scope) -> AsyncIterator[BrainEvent]`. v1
  adapter `HermesSyncBrain` wraps `run_conversation` via `asyncio.to_thread`, checks
  `scope.cancelled` before committing, yields one `FINAL_TEXT`, defers persistence,
  drives `AIAgent.interrupt()` on cancel.
- `StreamingCallTracerPort` (extends `NativeCallTraceWriter`): `transcript(e)`,
  `turn(e)`, `brain(e)`, `playback(mark)`, `interruption(d, heard, abandoned)`,
  `turn_committed(record)`. Redaction/preview gating reuses `VoiceDebugTracePolicy`
  + existing `_preview_text`.

### 5.5 Pure orchestration components
- `HeardSpanLedger`: `note_mark(mark)`, `note_flush(flush, full_text)`,
  `record(user_transcript, full_text, reason) -> CallTurnRecord`. Single source of
  truth for playback-aware truncation.
- `StreamingCallSession`: holds pure policies + ports; `run()` main reflex loop;
  `_on_turn_event(e)`, `_speak(text, scope)`, `_barge_in(decision)`. Driven directly
  by simulation; delegated-into by Pipecat adapters.

### 5.6 Port → Pipecat-primitive mapping (production adapters, thin)
| Port | Production adapter wraps |
|---|---|
| `AudioTransportPort` | custom `BaseTransport` (`InputAudioRawFrame`⇄`OutputAudioRawFrame`) over `aiortc_engine.py` |
| `TurnDetectionPort` | `SileroVADAnalyzer` + `LocalSmartTurnAnalyzerV3` (deferred per Decision A/D) |
| `SpeechToTextPort` | Pipecat `STTService` (later: AssemblyAI/Deepgram) |
| `TextToSpeechPort` | Pipecat `TTSService` + mark emission (later: Cartesia/ElevenLabs) |
| `HermesBrainPort` | Pipecat `LLMService.run_inference` → `asyncio.to_thread(run_conversation)` |
| `InterruptionPolicyPort` | ours — gates Pipecat's `StartInterruptionFrame` |
| `StreamingCallTracerPort` | observer `FrameProcessor` tap |

## 6. Behavior — BDD Scenarios (acceptance contract)

Each scenario maps 1:1 to §5 types/ports and is implemented as a session simulation
(except pure-policy assertions, which are tier-1 unit tests).

**Scenario A — Normal Turn.** Given a new call with no transcript; and the transport
delivers inbound PCM16; when the user speaks a continuous phrase meeting min speech
duration; then `TurnDetection` emits USER_SPEECH_STARTED; and STT produces a FINAL
`TranscriptEvent`; and the brain responds `BrainEvent(FINAL_TEXT)`; and TTS streams
`TtsAudioEvent` frames until DONE; and the tracer records `CallTurnRecord` with
`ended_reason=COMPLETED`.

**Scenario B — Real Interruption (Barge-In, during speech).** Given an ongoing turn
where the assistant is speaking and TTS is emitting frames; and the user has heard
some response (PlaybackMarks exist); when the user speaks longer than `min_speech_ms`
and produces more than `min_words`; then `TurnDetection` emits USER_SPEECH_STARTED +
endpoint events; and `InterruptionPolicy` decides `INTERRUPT`; and
`flush_outbound` is invoked; and TTS receives CANCEL and stops; and the pending brain
result is abandoned via `CancellationScope` (and `AIAgent.interrupt()` fired); and
`HeardSpanLedger` records `assistant_heard_text` + `assistant_abandoned_text`; and a
new user turn begins as in A.

**Scenario C — False Interruption / Backchannel.** Given the assistant is speaking and
TTS frames are in progress; and the user produces a short vocalization below
`backchannel_max_ms` or without sufficient recognized words; when `TurnDetection`
emits POSSIBLE_BACKCHANNEL; and no `TranscriptEvent` meets `min_words`; then
`InterruptionPolicy` decides `IGNORE`; and TTS continues without flushing; and the
assistant finishes normally; and no new `CallTurnRecord` is committed until the next
substantive utterance.

**Scenario D — Partial-Heard Truncation.** Given the assistant generated a long
response and started playback; and the user interrupts after hearing only a prefix;
when `InterruptionPolicy` decides `INTERRUPT`; then `flush_outbound` stops outbound
audio; and `HeardSpanLedger` notes the last delivered `PlaybackMark`; and the unplayed
suffix is stored as `assistant_abandoned_text`; and the `CallTurnRecord` is committed
with `interrupted=True` and `ended_reason=BARGED_IN`; and the next utterance begins a
new turn.

**Scenario E — Brain Latency (Long Thinking) + barge-in during thinking.** Given the
user speaks a valid utterance and FINAL transcript is available; and `HermesSyncBrain`
wraps `run_conversation` taking multiple seconds; when the assistant is still thinking
and emitting no `BrainEvent`s; then the transport continues to accept inbound audio;
and `TurnDetection`/`InterruptionPolicy` remain active; and the pipeline never blocks
inbound processing on brain latency. **Barge-in during thinking sub-case:** the
assistant is silent (nothing to flush), but the in-flight brain is abandoned via
`CancellationScope` and never persisted; `assistant_heard_text` is empty,
`assistant_abandoned_text` covers the discarded result, `ended_reason=BARGED_IN`.

**Scenario F — Configuration Fallback.** Given `calls.native.engine="turn_based"`;
when a call is initiated; then the existing turn-based pipeline is used and the
streaming slice is bypassed. Given `calls.native.engine="streaming"`; when a call is
initiated; then the streaming pipeline is used and all §5 ports are instantiated.

**Scenario G — False-Positive Resume.** Given the assistant is speaking and a brief
VAD trigger caused `flush_outbound` to fire; when no `TranscriptEvent` with recognized
words materializes within `false_interruption_timeout_ms`; then `InterruptionPolicy`
decides `RESUME`; and the assistant resumes from the last `PlaybackMark` (no
duplicated prefix); and no new `CallTurnRecord` is committed and `ended_reason` is not
`BARGED_IN`.

## 7. Test Strategy

**Deterministic virtual time, zero real sleeps.** Every `at_ms`/`timestamp_ms` is
driven by an injectable `VirtualClock` (`now_ms()`, `advance(ms)`, `sleep(ms)`).
Production uses a `MonotonicClock` with the identical interface. `ast-grep` forbids
literal `time.time()`/`time.sleep`/`asyncio.sleep` in `streaming/` except `clock.py`.

**Fakes (Slice 1's real test surface):**
- `FakeAudioTransport` — plays scripted `AudioFrame`s on the virtual clock; records
  `emit_outbound`/`flush_outbound`; `FlushResult` reports dropped vs sent.
- `FakeTurnDetection` — emits scripted `TurnEvent`s at chosen timestamps.
- `FakeSTT` — scripted partial/final `TranscriptEvent`s; can emit zero finals (G).
- `FakeTTS` — N frames per word with `PlaybackMark`s at word boundaries; honors
  `cancel()`/`flush()` mid-stream.
- `FakeBrain` — yields `FINAL_TEXT` after a configurable delay; cancellable; records
  whether abandoned.

**Three tiers:**
1. **Pure-unit (microsecond):** `InterruptionPolicy.decide` truth tables;
   `HeardSpanLedger` math; `CancellationScope` semantics. No async/clock. Densest
   coverage.
2. **Session simulation (A–G acceptance):** drive `StreamingCallSession` with fakes +
   `VirtualClock`; one test per scenario asserting event sequence + `CallTurnRecord`.
3. **Pipecat smoke (1–2 tests, only if WP0 wheels permit):** assert the production
   transport adapter translates `InputAudioRawFrame`⇄`AudioFrame` and a canned 1-turn
   pipeline reaches DONE.

**Reuse:** extend `gateway/calls/native/simulation.py` and `tracing.py`; new CLI
`calls simplex-simulate-stream` mirrors `simplex-simulate-voice-turn`.

**Live testing stays last and gated:** no live iPhone call until A–G + F/G + Pipecat
smoke pass green. Final acceptance reuses the handoff doc checklist + a saved live
trace.

## 8. Agent-Team Decomposition & Discipline

### 8.1 Tooling (WP0, blocks everything)
| Tool | Install | Role | Caveat |
|---|---|---|---|
| `ast-grep` | `brew install ast-grep` | structural architecture rules in `sgconfig.yml` | stable |
| `kit` (cased/kit) | `uv tool install cased-kit` | semantic/coupling: port↔impl alignment, duplicate-function scan | verify exact package/subcommands in WP0; surface options if mismatched, do not guess. **Fallback:** if `cased-kit` is not the intended tool or unavailable on this interpreter, the coupling/alignment checks degrade to `ty` + a manual review gate so WP0 never hard-blocks the DAG on a tooling unknown |
| `ty` (pinned) | `uv run ty check` | type alignment (Protocol sigs ↔ impls) | already present |
| `ruff` (pinned) | `uv run ruff check` | style/lint | already present |

**`ast-grep` rules:** (1) `streaming/ports.py` + `types.py` may not import `run_agent`
/ heavy deps; (2) no literal `time.time()`/`time.sleep`/`asyncio.sleep` in
`streaming/` except `clock.py`; (3) every dataclass in `types.py`/`ports.py` must be
`@dataclass(frozen=True)`; (4) `streaming/` may not import old turn-based internals
except via the engine selector.

**`kit` checks:** every `Protocol` method has both a fake and a production impl; no
semantically duplicate functions across the slice.

### 8.2 Work-package DAG
```
WP0 Tooling+scaffold ──► WP1 Types & Ports (THE CONTRACT — reviewed & locked)
        ┌─────────────┬──────────────┼──────────────┬───────────────┐
   WP2 Interrupt   WP3 Ledger+   WP4 Fakes+    WP5 HermesBrain   WP6 Tracer
   Policy (pure)   Cancellation  VirtualClock  (run_conversation  (redaction)
                                 harness        + persist gate +
                                                cooperative interrupt)
        └─────────────┴──────┬───────┴──────────────┬────────────────┘
                   WP7 StreamingCallSession ◄────────┘
        ┌──────────────┬───────────────┬──────────────┐
   WP8 Acceptance  WP9 CLI debug   WP10 Pipecat    WP11 Engine
   A–G (BDD sims)  simulate-stream  transport+      selector +
                                    smoke test      Scenario F
                                    (if WP0 wheels)
                   WP12 Integration verify + code-review + full lint sweep
```

**Parallelism:** WP2–WP6 are five independent async agents right after the contract
locks (disjoint files via vertical-slice + hexagonal seams → no shared-state
conflicts). WP8/WP9/WP10/WP11 are a second parallel wave after `StreamingCallSession`.

**Sequencing guards (shared-file contention):** WP0, WP1, WP11 touch shared files
(`pyproject`/`sgconfig`, the contract, `application.py`/config) and run serially.
Parallel agents that must edit a shared file get git-worktree isolation with a
merge-back gate.

**Skill mapping:** execution uses `superpowers:dispatching-parallel-agents` for the
WP2–WP6 and WP8–WP11 waves, `subagent-driven-development` for the serial spine. Each
WP is a discrete plan task with its own tests.

### 8.3 Per-WP discipline (rigid)
1. **Red** — write BDD/unit tests first; watch them fail.
2. **Green** — minimal implementation.
3. **Refactor** — clean up under green.
4. **Gate** — all pass before "done": `uv run ruff check <files>` ·
   `uv run ty check <files>` · `ast-grep scan <files>` · `kit` coupling/alignment ·
   `uv run pytest <wp tests> -q`.
5. **Review** — `superpowers:code-reviewer` subagent vs this spec + scenario contract;
   findings fixed before merge-back.

### 8.4 Definition of done (Slice 1)
A–G + F/G green in simulation; Pipecat smoke green (if WP0 wheels permit, else
explicitly deferred and logged); full `ruff`/`ty`/`ast-grep`/`kit` sweep clean;
`calls.native.engine` defaults to `turn_based` (zero behavior change until switched);
streaming path reachable only via `engine=streaming`.

## 9. Follow-on Slices (each its own spec→plan→implement)
1. Real turn-detection: Silero VAD + Smart Turn (resolve Py 3.13/3.14 wheels), tuning
   against a noisy test corpus.
2. Real streaming STT (cloud primary + local fallback) + proper-noun biasing.
3. Real streaming TTS (cloud primary + local fallback) with cancel/flush + marks.
4. Async tool calls during a call (weather grounding) + filler/"let me check" masking.
5. Cross-call conversation memory + `SessionDB` integration (durable `CallTurnRecord`).
6. Trace privacy operator toggle + retention/rotation.
7. Live inbound SimpleX production acceptance.
8. Video.

## 10. Risks & Mitigations
| # | Risk | Mitigation |
|---|---|---|
| R1 | Pipecat clock vs synchronous brain | correctness in Pipecat-independent `StreamingCallSession`; brain in `to_thread`; Scenario E proves non-blocking |
| R2 | `to_thread` can't kill the brain thread | `CancellationScope` drives `AIAgent.interrupt()` (cooperative stop) + discard; low `max_iterations`/`max_tokens`; trace `brain_abandoned` |
| R3 | Py 3.13/3.14 wheels (pipecat/onnxruntime/aiortc) | WP0 verifies on real interpreter; Decision A → fakes for turn-detection in Slice 1 |
| R4 | Abandoned text persisted to `SessionDB` | WP5 defers persistence; only tracer `CallTurnRecord` is durable; test asserts no SessionDB write on barge-in |
| R5 | `kit` tool identity uncertain | verified in WP0 before any gate depends on it |
| R6 | Real turn-taking tuning deferred behind fakes | acceptance proves logic; tuning is a follow-on slice with real adapters + noisy corpus |

## 11. Rollout / Rollback
- Default `calls.native.engine: turn_based` → zero behavior change on merge; streaming
  is dormant code until a profile opts in.
- Rollback = flip the flag; the turn-based path is never modified.
- Live iPhone test gated behind passing simulation (A–G + F/G) + Pipecat smoke.
