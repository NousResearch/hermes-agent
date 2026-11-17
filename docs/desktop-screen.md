# Desktop intake screen - W1 auto-hold (16 Nov 2026)

Owner order 16 Nov 2026 (W1 build queue, CBA section 4): the same
Phase 2 intake-screen pattern that guards the gateway platforms
(`gateway/guardrail.py`, 14 Nov) extended to the Desktop App Chat -
the surface the Phase 2 roll-out missed. ENV-GATED and DEFAULT
INERT: nothing changes for any deployment that does not set the
enable variable.

## What it does

Every interactive viewer submit on a desktop-launched session
(`prompt.submit`, plus the queued-edit verbs `prompt.steer_queued`
and `prompt.replace_queued`) is screened through the local RNR
Judgement Engine (`POST /v1/screen`, caller `hermes-desktop`,
purpose `desktop-screen`) BEFORE the submit is durably admitted - no
session turn runs for a held message:

| Engine tier   | Action                                            |
|---------------|---------------------------------------------------|
| allow         | proceeds; outcome `accepted` reported             |
| review        | proceeds, logged; outcome `escalated`             |
| block         | HELD - the submit is refused with a visible notice (the refused RPC's error message); logged; outcome `escalated` |
| chain_down    | proceeds (engine already applied its fail_mode); logged |
| unreachable   | proceeds (deliberate fail-open: a dead engine must never lock the owner out of his own desktop); logged |

The sender on this surface IS the local owner, so the refused
submit's error message is the alert; every verdict is also logged
for the owner dashboard (default `~/.hermes/logs/desktop-screen.log`).

## Scope (what is and is NOT screened)

Screened: EVERY interactive viewer submit that reaches durable
admission - the gate lives at the shared choke point
(`SessionAuthority._submit`, not at any one client surface) because
the Desktop app submits through its OWN local backend
(`hermes serve`, found unscreened by the owner's 17 Nov live test
when the first wiring sat at the gateway WS control plane only).
Screened on the model-visible text: the contract's `api_text`
override when present AND the persisted `text` (a hostile payload
behind a benign override is held either way - the engine judges both
fields). Verbs covered automatically: `prompt.submit`,
`prompt.replace_queued` (a fresh admission), TUI/ACP viewer submits.
Launch-source scope still applies: only sessions frozen as `gui` or
`desktop` (env-widened to every interactive source).

NOT screened, structurally: classic-CLI and oneshot fleet sessions
(different launch sources - `hermes -q` work never enters a
desktop-sourced session), kanban cards (trusted server-side marker),
hosted-room tasks, a2a forwards, automation envelopes, and gateway
platform messages (already screened by Phase 2). `prompt.steer_queued`
carries no text by contract (the transfer re-delivers the queued
row's already-screened text). Attachments and `voice_context` are
not screened this cycle (media triage is use case A6, later in the
queue). NOTE: the visible client artefact of a hold is the RPC error
frame `4001 desktop_screen_hold` whose `error.message` carries the
full hold notice (the same text the screen log records alongside the
tier and reasons; the reason list itself is not on the wire).
Round-6 review NOTEs, documented as out of
scope: `display_v1.title_preview` feeds only the display titler
(an auxiliary call that writes a chat title, never the session
prompt) and `cli_turn_v1.turn_author.name` reaches memory
attribution providers, not the model prompt; neither is screened
this cycle (same A6 deferral as attachments/voice_context).

Judge philosophy unchanged: the judge advises, it never replaces
owner gates. Texts are split into 32k-char chunks with a 1 KB seam
overlap: a payload of up to 1024 chars is GUARANTEED to sit whole
inside some chunk wherever it falls; longer payloads are often still
held (position-dependent) and anything longer than one chunk
(32k) reaches the engine as fragments - every character judged in
all cases. All chunks are judged within one 6s screening budget
shared by the whole submit. Whitespace-only chunks are skipped (no
payload, and the engine rejects blank text; logged as tier
"blank_chunk_skipped"); every other
model-visible character is judged or the submit is held
("screening_incomplete_budget"; text over 4M chars is held as
"oversized_unscreened"; a deterministic engine 4xx rejection holds
as "engine_rejected_chunk"). The fail-open class (Phase 2) covers:
engine unreachable, engine 5xx or malformed 200 response, and
TRANSIENT client codes 408/429 (proxy timeout, rate limit - a busy
engine must not lock the owner out; logged as
"engine_rate_limited_or_timeout"). Fail-open abandons screening of
the REMAINING chunks of that submit (same width as the Phase 2
gateway fail-open; the whole submission is then admitted unscreened
and the event is logged). A call clipped by the budget is coverage
exhaustion and holds. Known cosmetic: chunks judged allow before a
later chunk holds have already reported "accepted" outcomes
(analytics only, no screening effect).

## Enablement (per deployment)

```
HERMES_DESKTOP_SCREEN=1                            # "" = off (default)
HERMES_DESKTOP_SCREEN_ENDPOINT=http://127.0.0.1:8201/v1/screen
HERMES_DESKTOP_SCREEN_TIMEOUT_S=4.0
HERMES_DESKTOP_SCREEN_CALLER=hermes-desktop
HERMES_DESKTOP_SCREEN_PURPOSE=desktop-screen
HERMES_DESKTOP_SCREEN_SOURCES=gui,desktop          # "all" = every interactive source
HERMES_DESKTOP_SCREEN_HOLD_NOTICE=                 # "" = default notice text
HERMES_DESKTOP_SCREEN_LOG=~/.hermes/logs/desktop-screen.log
```

## Deployment coupling

The `X-Judge-Purpose` header needs the engine-side `/v1/screen`
purpose override (RNR-Judgement-Engine branch
`hermes/judgement-engine-integrator/w1-screen-purpose-override`).
Without it the screen still works; its decisions just attribute
under `gateway-screen` in /metrics instead of `desktop-screen`.
Deploy the engine change first (or same window). Enablement order:
engine live -> gateway/desktop runtime carries this code -> set
`HERMES_DESKTOP_SCREEN=1` -> one benign desktop submit (expect a
normal answer) and one pasted attack string (expect the hold notice
+ a `block` line in the screen log) -> confirm /metrics shows the
`desktop-screen` purpose bucket.

## Turn-on for this deployment (owner steps, after merge)

1. Engine: pull install, restart both units (SOP verify).
2. Runtime checkout carries the merged code (the live lineage pull).
3. Add `HERMES_DESKTOP_SCREEN=1` to `~/.hermes/.env` (or the service
   env), restart the runtime once.
4. Verify: benign message answers normally; attack paste is held
   with the notice; `grep block ~/.hermes/logs/desktop-screen.log`;
   dashboard /metrics carries purpose `desktop-screen`.
