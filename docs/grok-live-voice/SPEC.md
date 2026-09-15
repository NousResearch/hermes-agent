# Grok-Live Voice: SPEC (native xAI realtime engine for Hermes)

Status: authoritative spec, pre-implementation. Branch `feat/grok-live-voice`. Do not
re-litigate the decisions marked **AUTHORITATIVE** — they were fixed at the orchestrator
level (kanban task `t_ce405091`) and every sibling task depends on them being stable.

Reference standalone implementation (proven live on this box, read but do not port wholesale):
`/home/ericio/grok_live_voice.py`. Existing sibling engines to mirror: `chained`
(`tools/voice_live.py` is actually the **gpt-live** module despite its generic name — see
"Naming note" below) plus `tools/voice_client_config.py` for chained STT/TTS.

## 1. Naming note (read this first — avoids file confusion across sibling cards)

`tools/voice_live.py` is the **gpt-live** module (its own docstring: "GPT-Live voice chat
mode"). It is not a generic "voice live" abstraction. Grok-Live gets its own **sibling** file,
`tools/voice_live_grok.py`, following the repo's facade+siblings rule — it must NOT be added
as more branches inside `voice_live.py`. `voice_live.py` itself only needs one shared edit:
`voice_chat_mode()`'s return values are today a closed `{chained, gpt-live}` pair; grok-live's
config-card task (t_68cf4461) either widens that shared resolver's return type or gives
`tools/voice_live_grok.py` its own `voice_chat_mode()`-equivalent — **decision: widen the
shared one in `voice_live.py`** so there is a single source of truth for
`voice.voice_chat_mode` across all three engines; every other symbol (`resolve_*_status`,
`live_instructions`, `build_session_config`, `create_webrtc_session`) stays engine-specific in
its own sibling file. Do not move gpt-live's symbols during this work.

## 2. Architecture

```
apps/desktop renderer                    tui_gateway (backend, Python)                xAI
┌─────────────────────────┐   JSON-RPC   ┌───────────────────────────────┐  wss (binary  ┌──────────┐
│ voice-engine-rows.tsx    │◄────────────►│ methods_voice_grok.py          │  PCM16 +    │ realtime │
│ (picker: chained|        │ voice.grok.* │  - session lifecycle           │  JSON text) │ endpoint │
│  gpt-live|grok-live)     │              │  - xAI ws connection            │◄───────────►│          │
│                          │              │    (credential resolution,      │             └──────────┘
│ use-voice-live-grok      │  binary-ish  │     session.update, VAD cfg)    │
│  -conversation.ts        │  audio       │  - audio relay (renderer<->xAI) │
│ (mic capture, playback,  │  frames      │  - AEC half-duplex gate         │
│  mute, level meter)      │  (base64 in  │  - reconnect (5s)                │
│                          │  JSON-RPC —  │  - event -> delegation bridge    │
│ voice-live-grok.ts store │  see §4.1)   │    (tools/voice_live_grok.py)    │
└─────────────────────────┘              └───────────────────────────────┘
              │                                          │
              │  onDelegation(text, context)              │ submits as a NORMAL
              ▼                                          ▼ prompt.submit (surface:
      normal chat turn on the open session  ◄─────────── voice-live) — same seam
      (any model, full toolset, memory)                  gpt-live already uses
```

Backend holds the xAI websocket exclusively (`voice.grok_live` is a **server-held connection**,
unlike gpt-live's WebRTC which the renderer negotiates directly with OpenAI). This is the core
architectural difference from gpt-live and is why grok-live needs an audio *relay* rather than a
one-time SDP exchange.

## 3. Config schema (AUTHORITATIVE — from t_ce405091, reproduced here as the schema of record)

`hermes_cli/config_defaults.py`, inside the existing `"voice"` block, sibling to `gpt_live`:

```python
"voice_chat_mode": "chained",  # chained | gpt-live | grok-live
"grok_live": {
    "model": "grok-voice-latest",
    "voice": "eve",
    "auth": "auto",           # auto (oauth-first, XAI_API_KEY fallback) | oauth | apikey
    "instructions": "",       # extra persona sentences appended to the short delegation persona
    "vad_threshold": 0.75,
    "silence_ms": 700,
    "prefix_ms": 333,
    "speed": 1.0,
},
```

All behavioral knobs live in `config.yaml` — **no new env vars** (root rule). `voice.grok_live.auth`
mirrors the reference script's `XAI_VOICE_AUTH` env var but as a config key.

`tui_gateway/methods_config_set.py` line 349's `_word` validator set for
`"voice.voice_chat_mode"` grows from `{"chained", "gpt-live"}` to
`{"chained", "gpt-live", "grok-live"}` — one-line change, same validator entry.

## 4. Transport / protocol

### 4.1 Renderer ↔ gateway (existing JSON-RPC channel — NOT raw binary WS frames)

Correction to the orchestrator card's phrasing ("binary frames over the existing gateway WS"):
`apps/shared/src/json-rpc-channel.ts` is a **JSON-RPC-only** channel. `wireFrameText` decodes an
inbound `ArrayBuffer` to UTF-8 text — it does not support a binary/JSON dual-mode protocol, and
`request()`/`deliverRequest()` always serialize `JSON.stringify(...)`. There is already a working
precedent for audio-over-JSON-RPC in this exact codebase: `wake.feed` (`WakeFeedParams.pcm_b64`,
`tui_gateway/contracts/prompt_voice.py:518`) pushes client-captured PCM as base64 inside a normal
JSON-RPC call. Grok-Live's audio relay follows the same shape — base64 PCM16 chunks inside
JSON-RPC method calls / event payloads, ~100 ms per chunk (2,400 frames @ 24kHz = 4,800 bytes raw
→ ~6,400 bytes base64). This keeps the relay inside the existing typed-contract system
(`tui_gateway/contracts/`) instead of adding a second transport mode to the gateway. Any sibling
task that assumed true binary WS frames must build against this base64-in-JSON-RPC design instead.

New contract file `tui_gateway/contracts/prompt_voice_grok.py` (sibling to `prompt_voice.py`,
same pattern as `wake.*`):

| Method / event | Direction | Payload | Purpose |
|---|---|---|---|
| `voice.grok.status` | client→server | `{profile}` | `resolve_grok_live_status()` — mirrors `/api/audio/voice-live/status` but as an RPC for TUI/desktop parity |
| `voice.grok.start` | client→server | `{session_id, profile}` | Open (or reuse) the backend xAI ws session for this Hermes session; acquires mic lease (§6) |
| `voice.grok.stop` | client→server | `{session_id, profile}` | Close the xAI ws cleanly, release mic lease |
| `voice.grok.audio` | client→server | `{session_id, pcm_b64, seq}` | One ~100ms mic chunk, forwarded verbatim to xAI as binary on the backend leg |
| `voice.grok.audio` (event) | server→client | `{session_id, pcm_b64, seq}` | One ~100ms of Grok's spoken reply audio for playback |
| `voice.grok.mute` | client→server | `{session_id, muted}` | Explicit mute toggle (independent of the AEC gate, §7) |
| `voice.grok.transcript` (event) | server→client | `{session_id, speaker, text, item_id}` | Mirrors xAI `conversation.item.input_audio_transcription.*` / `response.output_audio_transcript.*` for the transcript UI (same shape gpt-live's `LiveTranscriptFragment` uses) |
| `voice.grok.state` (event) | server→client | `{session_id, state, reason}` | `state`: `connecting\|listening\|speaking\|thinking\|error\|reconnecting`; mirrors gpt-live's client-side `ConversationStatus` but server-driven since the ws lives on the backend |
| `voice.grok.delegation` (event) | server→client | `{session_id, delegation_id, context}` | Fired when the bridge decides a spoken exchange is a real request — see §5 |

Backpressure: `voice.grok.audio` upstream chunks are dropped (not queued) if the backend's send
queue to xAI exceeds 300ms of audio (matches the reference script's synchronous mic loop — audio
that can't be delivered live is stale) — the drop increments a counter surfaced in
`voice.grok.state {state: "degraded"}` after 5 consecutive drops, never silently. Downstream
(xAI → renderer) chunks are never dropped; a slow renderer applies backpressure by not draining
its own audio queue, same as gpt-live's WebRTC jitter buffer.

### 4.2 Backend ↔ xAI (`wss://api.x.ai/v1/realtime`)

Reuse the reference script's wire contract verbatim (it is proven live, 2026-09-15):
- URL: `wss://api.x.ai/v1/realtime?model={voice.grok_live.model}`
- Auth header: `Authorization: Bearer {resolved_token}` (§8)
- `session.update` payload: `reasoning.effort=none`, `turn_detection.type=server_vad` with
  `threshold`/`silence_duration_ms`/`prefix_padding_ms` from config, `audio.input/output.format
  = {type: audio/pcm, rate: 24000}`, transport `binary`, `voice` + `instructions` (§5 persona),
  NO `tools` array (decision #6 — v1 ships zero local function tools; the reference script's
  `grok_voice_tools` bridge is NOT ported).
- Inbound events handled: `session.created`, `session.updated`, `input_audio_buffer.speech_started`
  /`_stopped`, `conversation.item.input_audio_transcription.completed`/`.updated`,
  `response.output_audio_transcript.done`, `response.output_audio.delta` (binary payload or
  base64 `delta` per event shape — reference script handles both `response.output_audio.delta`
  and the older `response.audio.delta` alias), `response.done`, `error`.
  `response.function_call_arguments.done` is NOT handled in v1 (no tools registered, so xAI never
  emits it — if it ever does, log and ignore, don't crash the session).

## 5. Delegation (mirrors gpt-live's proven pattern — see t_f7cae076 for the implementation card)

Persona (`tools/voice_live_grok.py::live_instructions()`), same shape as gpt-live's
`LIVE_PERSONA`: short, role + style + delegation policy, ends with
`voice.grok_live.instructions` appended if set. This persona is what ships in the xAI
`session.update.instructions` field — it is **never** the Hermes system prompt.

Bridge event → Hermes turn: when the backend judges an utterance complete (xAI
`response.output_audio_transcript.done` on the user's turn, or — simpler and consistent with
gpt-live's client-side `delegationPrompt()` — accumulate transcript fragments server-side and
flush on `input_audio_buffer.speech_stopped` after a short settle timer, mirroring
`UTTERANCE_SETTLE_MS` in `use-voice-live-conversation.ts`), the backend:

1. Builds `{context, prompt}` exactly like `delegationPrompt()` in
   `apps/desktop/src/app/chat/composer/hooks/use-voice-live-conversation.ts` (last user turn is
   `prompt`, full recent exchange is `context`).
2. Emits `voice.grok.delegation {session_id, delegation_id, context}` so the desktop transcript
   UI can render it exactly like gpt-live does today.
3. Calls `prompt.submit` internally as a **normal Hermes turn** on the open session:
   `surface="voice-live"` (the existing `ClientSurface` enum value, reused, not a new
   `"grok-live"` surface — the delegation semantics are identical to gpt-live's, so the surface
   tag stays the shared one), `voice_context=context`, `text=prompt`.
4. The per-turn note is `tools/voice_live.py::voice_live_turn_note()` — reused verbatim (it is
   already generic "a live spoken conversation", not GPT-specific). **No new per-turn note text
   is authored for grok-live**; this is the single biggest cache-safety guarantee (t_f7cae076's
   invariant tests should assert this exact function is called, not a grok-specific reimplementation).
5. The assistant's reply text streams back through the existing `pendingResponse()` /
   `session.commentary.append` seam already wired for gpt-live in `use-voice-live-conversation.ts`
   — reused for grok-live by parametrizing that hook (or a thin sibling
   `use-voice-live-grok-conversation.ts` that shares `delegationPrompt`) rather than duplicating
   the turn-drive `useEffect`. Reply text is sent to xAI as a `conversation.item.create` (role
   assistant, or as text via `response.create` with injected content, matching whichever xAI
   supports for "speak this text" — confirm against the xAI realtime API reference during
   implementation; the reference script never needed this because it let the model both hear
   and answer in one round-trip. Grok-Live's split (xAI hears, Hermes answers, xAI must speak
   Hermes' words) needs the "speak arbitrary text" affordance xAI's realtime API exposes for
   function-call-output-driven responses — same pattern the reference script uses after
   `function_call_output` + `response.create`.

Cache-safety invariant (t_f7cae076 must test this explicitly): the delegation path must never
construct a new system prompt, must never call any context/toolset-mutating API, and the
per-turn note must ride only the `text`/`voice_context` params of `prompt.submit` — exactly the
same call shape gpt-live already uses successfully in production.

## 6. Mic lease / wake-word interaction

Reuse the existing pattern (`wake.pause` / `wake.resume`, `WakePauseResult`/`WakeResumeResult` in
`tui_gateway/contracts/prompt_voice.py`): `voice.grok.start` calls the same pause path the
chained and gpt-live engines use before opening the mic (`beforeMicOpen` in
`use-voice-live-conversation.ts` is the renderer-side hook to mirror — grok-live's
`use-voice-live-grok-conversation.ts` calls `wake.pause` the same way). `voice.grok.stop` (or
`onClosed`) calls `wake.resume`. No new wake-word plumbing — the lease semantics are surface-
agnostic already.

## 7. AEC (half-duplex guard)

Ported directly from the reference script's `_PlaybackGate` (grok_live_voice.py:44-56): while
`state == speaking` (server tracks this from `response.output_audio.delta` timing) plus a
`PLAYBACK_ECHO_TAIL_S = 0.45`s tail, the backend **does not forward** upstream mic audio to xAI
— it either sends silence or simply withholds forwarding (equivalent to the reference's
mic-muting, ported to run against the *relayed* audio instead of a local PortAudio stream). This
lives entirely server-side (the backend owns the xAI connection so it is the natural gate point),
not in the renderer — the renderer keeps streaming mic frames; the backend decides whether to
forward them to xAI. This also means AEC gating does not need a renderer-side state sync, only
the `voice.grok.state` event for UI feedback (mic icon shows "muted (playback)" during the gate,
same visual language a future muted-for-AEC state would use).

## 8. Auth resolution order (AUTHORITATIVE)

`tools/voice_live_grok.py::_resolve_grok_credentials()`:

1. `voice.grok_live.auth == "oauth"` → xai-oauth SuperGrok token only (§8.1), error if absent/expired-and-unrefreshable.
2. `voice.grok_live.auth == "apikey"` → `XAI_API_KEY` (env or `~/.hermes/.env`) only.
3. `voice.grok_live.auth == "auto"` (default) → xai-oauth token first (refreshed if needed), THEN
   `XAI_API_KEY` fallback. **Never the reverse** — on this box the BSM-injected `XAI_API_KEY`
   belongs to a zero-credit team (403s everywhere); if auto ever preferred it, grok-live would
   silently fail for every user on this box even though a working OAuth token exists. This is the
   single highest-value behavioral guarantee in this spec and MUST have a unit test pinning "oauth
   present → oauth used even when XAI_API_KEY is also present."

### 8.1 OAuth token resolution (ported from reference script, adapted to be importable/mockable)

`~/.hermes/auth.json → providers["xai-oauth"].tokens`: read `access_token`; if
`last_refresh + expires_in - 2min <= now`, refresh via
`providers["xai-oauth"].discovery.token_endpoint` using the refresh_token grant (public client,
`client_id` extracted from the `id_token` JWT `aud` claim — same undocumented-but-working
approach the reference script uses), write the refreshed tokens back to `auth.json` so the CLI's
own xai-oauth lane keeps working too (shared token store — reuse, don't fork).

Token never leaves the backend process: it is read to build the `Authorization` header for the
backend's own websocket to xAI; it is never sent to the renderer, never logged, never included in
`resolve_grok_live_status()`'s response (that endpoint returns only `available: bool` + `reason`).

## 9. Lifecycle

- **start**: `voice.grok.start` → resolve credential (§8) → fail fast with `reason` if none →
  `wake.pause` → open ws → send `session.update` → wait for `session.updated` (mirrors reference
  script's `configure()`) → emit `voice.grok.state {state: listening}`. A failure at any step
  emits `voice.grok.state {state: error, reason}` and does NOT retry (explicit start failure is
  reported to the user, not silently retried — retry only applies to an already-established
  session that drops, §10).
- **audio flow**: `voice.grok.audio` (client→server) chunks relayed to xAI as binary frames on
  the backend leg (subject to AEC gate, §7); xAI binary/delta frames relayed back as
  `voice.grok.audio` events (base64, §4.1).
- **delegation**: §5.
- **stop**: `voice.grok.stop` (explicit) or `onClosed`-equivalent (fatal error, §10 exhausted) →
  close ws → `wake.resume` → emit `voice.grok.state {state: idle}`. Symmetric with gpt-live's
  `end()` in `use-voice-live-conversation.ts`.
- **error taxonomy**: `ConnectionClosed` (xAI dropped us) and "receiver loop exited" are both
  FATAL to the current session (matches reference script exactly: `run_voice()` re-raises both
  paths to `main()`'s reconnect loop) — never treated as transient/ignorable.

## 10. Reconnect

On a fatal error (§9) while the user has not explicitly stopped: wait 5s (matches reference
script `asyncio.sleep(5)`), then re-run the full start sequence (§9) against the SAME logical
`voice.grok.start` session (same `session_id`), re-sending `session.update` (xAI has no
resumable-session concept exposed here — a fresh `session.updated` handshake is required every
time, exactly like the reference script's outer `while True` loop). No cap on reconnect attempts
in v1 (matches the reference script) but each attempt emits `voice.grok.state
{state: reconnecting}` so the UI can show a spinner instead of looking hung; a user-initiated
`voice.grok.stop` during a reconnect wait cancels the pending retry. Systemd-style external
watchdog is explicitly out of scope (decision #8) — this is an in-process asyncio retry only.

## 11. Availability status semantics

`resolve_grok_live_status()` (mirrors `resolve_gpt_live_status()` in `tools/voice_live.py:131`,
lives in `tools/voice_live_grok.py`):

```python
{
    "mode": voice_chat_mode(),               # "chained" | "gpt-live" | "grok-live"
    "available": bool,                        # True iff §8 resolves a usable credential
    "reason": None | str,                     # "no xAI credential (SuperGrok login via `hermes login` or set XAI_API_KEY)"
    "model": str,                              # voice.grok_live.model
    "voice": str,                              # voice.grok_live.voice
}
```

`available=True` does NOT mean the ws will succeed (network/xAI-side failures are still
possible) — it means a credential resolves per §8, mirroring gpt-live's contract exactly (that
endpoint also never round-trips to the vendor to answer `available`). Exposed via:
- `GET /api/audio/voice-live/status` — **extend** the existing route (t_46896f18's status
  resolver task) to include a `grok` key alongside whatever gpt-live returns today, OR add a
  parallel `GET /api/audio/voice-live-grok/status`. **Decision: extend the existing endpoint**,
  returning `{ok, chained: {...}?, gpt_live: {...}, grok_live: {...}}`-shaped or (simpler, less
  churn) keep the endpoint's current flat gpt-live-only shape for gpt-live and add a **second**
  route `GET /api/audio/voice-live-grok/status` mirroring it 1:1 (same function shape as
  `get_voice_live_status`) — **this is the one the desktop status resolver task must implement**,
  because retrofitting the existing flat response shape risks breaking the current
  `$voiceLiveStatus` consumer contract mid-migration. New route, new store field
  (`$voiceLiveGrokStatus` or a widened `$voiceLiveStatus` carrying all three — desktop task
  t_b4216b72 decides the store shape, but the route is one new endpoint, not a breaking change
  to the existing one).
- `voice.grok.status` JSON-RPC method (§4.1) for TUI/desktop-over-gateway parity with the other
  `voice.*` methods.

## 12. Seam inventory (exact files each sibling card touches — no two cards guess)

| Card | Files |
|---|---|
| t_68cf4461 (config/status) | `tools/voice_live.py` (widen `voice_chat_mode()` return set only — §1); NEW `tools/voice_live_grok.py` (`resolve_grok_live_status`, `voice_chat_mode` re-export if needed, `_resolve_grok_credentials`, `_oauth_token`, persona `live_instructions`); `hermes_cli/config_defaults.py` (`voice.grok_live` block, §3); `tui_gateway/methods_config_set.py` (line 349 validator set, §3); tests in `tests/tools/test_voice_live_grok.py` (new), `tests/tui_gateway/test_methods_config_set.py` (extend) |
| t_46896f18 (realtime bridge, depends on t_68cf4461) | NEW `tools/voice_live_grok_bridge.py` or equivalent session-manager module (ws lifecycle, event handling, AEC gate, reconnect — §4.2, §7, §9, §10); NEW `tui_gateway/contracts/prompt_voice_grok.py` (§4.1 methods/events); NEW `tui_gateway/methods_voice_grok.py` (RPC handlers registered per `tui_gateway/AGENTS.md`'s `method()` pattern); `hermes_cli/web_routers/audio.py` (NEW `GET /api/audio/voice-live-grok/status` route, §11 — do not touch the existing gpt-live route); tests in `tests/tui_gateway/test_methods_voice_grok.py` (fake websockets, no network), `tests/tools/test_voice_live_grok_bridge.py` |
| t_b4216b72 (desktop UI, depends on t_68cf4461 for status shape) | `apps/desktop/src/store/voice-live.ts` (widen `VoiceChatMode`/`selectedVoiceChatMode`/`setVoiceChatMode` to 3 values — mirror `$voiceLiveStatus` handling for the new `grok_live` status source, §11); `apps/desktop/src/app/chat/composer/voice-engine-rows.tsx` (3rd radio row); NEW `apps/desktop/src/app/chat/composer/hooks/use-voice-live-grok-conversation.ts` (mirrors `use-voice-live-conversation.ts` but against the relay protocol §4.1, reusing `delegationPrompt` — consider extracting `delegationPrompt` to a shared module if both hooks need it, to avoid a duplicate copy); i18n: `apps/desktop/src/i18n/*` all locales (`voiceEngineGrok`, `voiceEngineGrokShort`, `voiceEngineGrokNeedsKey` + `types.ts`); vitest under `apps/desktop/src/store/`, `apps/desktop/src/app/chat/composer/` |
| t_f7cae076 (delegation, depends on t_46896f18) | `tools/voice_live_grok_bridge.py` (or wherever the bridge event → `prompt.submit` call lives — same file as t_46896f18 owns, this card adds the delegation call inside it); reuses `tools/voice_live.py::voice_live_turn_note()` (no new note text, §5); tests in `tests/tui_gateway/` asserting cache-safety invariants (§5's closing paragraph) — mock the ws bridge, no real xAI/network |

No card touches another card's exclusive files above without a comment on both cards first
(root rubric: hotspot flagging).

## 13. Risk list

1. **Cache-safety.** The single largest risk: any code path that concatenates grok-specific text
   into the Hermes system prompt, or that mutates the active toolset/context mid-conversation,
   breaks per-conversation prompt caching for the WHOLE session, not just voice turns. Mitigation:
   reuse `voice_live_turn_note()` verbatim (§5), never author a second per-turn note function;
   t_f7cae076's tests must assert the exact call site and its byte-stability.
2. **Role alternation.** The delegation → `prompt.submit` path must never produce two consecutive
   same-role messages (e.g. two delegations firing before the first turn settles). Mitigation:
   mirror gpt-live's `busyRef`-gated interrupt-then-submit logic (`onDelegation` in
   `use-voice-live-conversation.ts:261-295`) exactly — a newer delegation interrupts the in-flight
   turn before submitting, never queues a second user turn behind an unresolved one.
3. **`hermes update` clobbering local tree state.** All work on this feature happens on
   `feat/grok-live-voice`; nothing is committed to `main` until reviewed. If `hermes update` runs
   on this box mid-implementation (it fetches/pulls on `main` and can affect the working tree via
   git operations), the procedure is: **commit everything on the branch before any `hermes
   update` invocation**; if an update did run, `git status` on the branch — an update only
   touches `main`'s working tree via checkout, so a feature branch with a clean working tree is
   unaffected, but verify with `git log --oneline -3` and `git diff main...feat/grok-live-voice
   --stat` before continuing; if a rebase becomes necessary (`main` moved and a sibling card's
   base is stale), rebase the feature branch onto the fresh `origin/main` (`git fetch origin main
   && git rebase origin/main`) rather than merging, to keep a clean history for the eventual PR.
4. **OAuth token refresh races.** Two processes (a CLI session and the gateway) could refresh the
   shared `~/.hermes/auth.json` xai-oauth token concurrently, each writing a different fresh
   token — the loser's in-memory token is still valid for its own request (the read-then-use
   happens before the write races), but a THIRD reader between the two writes could observe a
   half-written file if the write is not atomic. Mitigation: check whether the reference script's
   `auth_path.write_text(json.dumps(data, indent=2))` (grok_live_voice.py:128) is atomic — it is
   NOT (no temp-file+rename) — the backend port should use the same
   write-temp-then-`os.replace()` atomicity pattern other Hermes state writers use elsewhere in
   the repo, or accept the (rare, low-consequence: worst case is one extra unnecessary refresh
   attempt) race and document it. Recommend: fix it in the port even though the reference script
   didn't have it, since the port runs inside a long-lived multi-request gateway process where
   the exposure window is larger than the reference script's single-process CLI use.
5. **Audio backpressure.** A slow/loaded backend or a network hiccup on the xAI leg can back up
   the renderer→backend→xAI relay. Mitigation: §4.1's drop-not-queue policy on the upstream leg
   (dropping live mic audio is the correct behavior — queued stale audio would desync the VAD);
   downstream never drops (a dropped reply chunk is an audible glitch, worse than a brief stall).
6. **xAI "speak this text" affordance uncertainty (§5 step 5).** The reference script never needed
   this because it lets xAI's own model both hear the user and generate the spoken reply — it
   never injects externally-generated text for xAI to read aloud. Grok-Live's split (Hermes
   generates the words, xAI must speak them) needs to be validated against the xAI realtime API
   docs during t_f7cae076's implementation before committing to the `conversation.item.create` +
   `response.create` approach sketched in §5; if xAI has no clean "speak arbitrary text" primitive
   this could require a different approach (e.g. re-prompting xAI with strict instructions to
   repeat the given text verbatim, which is fragile). **Flag this as the top implementation
   spike for t_f7cae076** — confirm the primitive exists before writing tests against it.
7. **Two independent "voice-live" store shapes on desktop.** t_b4216b72's decision to add a
   parallel status route/store rather than widen the existing gpt-live one is a deliberate choice
   to avoid a breaking change (§11), but it means the desktop temporarily carries two similar-but-
   separate status objects. Risk: engine-picker code drifting out of sync (e.g. only one status
   object refreshed on a settings change). Mitigation: `refreshVoiceLiveStatus()`-equivalent for
   grok should be triggered from the exact same call sites as the gpt-live one (config-change
   handler, conversation-mount effect) — a shared `refreshAllVoiceLiveStatuses()` wrapper is
   recommended over two independently-triggered refreshes.
