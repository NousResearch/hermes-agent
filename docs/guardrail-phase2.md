# Gateway intake guardrail - Phase 2 auto-hold (14 Nov 2026)

Owner decision: upgrade the advisory gateway screen (Phase 1, hook
based) to automatic holds for block-tier messages. This change adds
a veto-capable gate at the gateway intake choke point. It is
ENV-GATED and DEFAULT INERT: nothing changes for any deployment that
does not set the enable variable.

## What it does

Every inbound non-command message is screened through the local RNR
Judgement Engine (`POST /v1/screen`, caller `hermes-gateway`) before
any session turn starts:

| Engine tier   | Action                                            |
|---------------|---------------------------------------------------|
| allow         | proceeds; outcome `accepted` reported             |
| review        | proceeds, logged; outcome `escalated`             |
| block         | HELD - no agent turn; optional hold reply; one owner alert per sender per debounce window; outcome `escalated` |
| chain_down    | proceeds (engine already applied its fail_mode); owner alerted |
| unreachable   | proceeds + owner alert (deliberate fail-open: the bridge allowlist still gates WHO may talk; a dead engine must not mute the household) |

Slash commands and gateway control traffic are NEVER screened (they
are trusted syntax; screening them could wedge remote admin).

Judge philosophy unchanged: the judge advises, it never replaces
owner gates. Holds are audited (alert + log + outcome); review-tier,
chain-down holds and escalations still reach the owner.

## Enablement (per deployment)

Environment variables (the gateway's existing override pattern):

```
HERMES_GATEWAY_GUARDRAIL=1                        # "" = off (default)
HERMES_GATEWAY_GUARDRAIL_ENDPOINT=http://127.0.0.1:8201/v1/screen
HERMES_GATEWAY_GUARDRAIL_TIMEOUT_S=4.0
HERMES_GATEWAY_GUARDRAIL_CALLER=hermes-gateway
HERMES_GATEWAY_GUARDRAIL_HOLD_REPLY=              # "" = silent hold
HERMES_GATEWAY_GUARDRAIL_OWNER_CHAT=<owner chat id>
HERMES_GATEWAY_GUARDRAIL_DEBOUNCE_S=600
HERMES_GATEWAY_GUARDRAIL_LOG=~/.hermes/logs/gateway-guardrail.log
```

Logs: `~/.hermes/logs/gateway-guardrail.log` (one JSON line per
screened message) and alert-debounce state beside it.

## Relationship to the Phase 1 hook

The Phase 1 hook (`~/.hermes/hooks/guardrail-screen/`) remains as an
independent observer/telemetry path while Phase 2 is enabled; it is
advisory by design and double-logs harmlessly. It can be retired once
Phase 2 has soaked (owner call; the engine dashboard keeps both
attributed under caller `hermes-gateway`).

## Notes for merging harnesses (ZCode / any future harness)

- Engine-side: NOTHING changes. This is a Hermes gateway consumer of
  the existing `/v1/screen` + `/v1/outcome` contract.
- Dashboard: traffic under caller `hermes-gateway` purpose
  `gateway-screen` now includes Phase 2 holds; escalated outcomes on
  block-tier rows mean "held automatically, owner alerted" - not
  "awaiting manual decision".
- A harness wanting the same behaviour should implement the identical
  tier mapping above at its own intake, tagging its own caller id;
  never reuse `hermes-gateway`.

## Review history

- r1 (ac7d00cc84): independent adversarial review verdict FAIL - 4
  blockers (gate inert on real traffic via allow_gateway_control
  default; unknown-command-text veto bypass; no-raise violations on
  malformed responses/env/state; synchronous outcome POST stalling
  the event loop) + 7 informational. VERDICT.json archived at
  Hermes-Bot-Fleet/docs/guardrail-phase2-review-20261114/.
- r2 (this revision): all four blockers fixed -
  B1 the control flag is no longer consulted (defaults True on real
  traffic; screening covers every non-command message);
  B2 only KNOWN gateway commands skip (is_gateway_known_command);
  unknown '/zzz payload' text is screened;
  B3 outer never-raise wrapper + type-defensive response parsing +
  safe numeric env + tolerant debounce state (a type-confused
  BLOCK-tier response still HOLDS, with empty reasons);
  B4 outcome reports are fire-and-forget off-loop tasks with a 4s
  deadline (drain_outcome_tasks provided for tests/clean shutdown);
  screen has an overall deadline (timeout+1s). Tests 19/19 incl.
  real-MessageEvent regression, command matrix, malformed-response
  matrix, slow-outcome latency bound. I2/I3/I5/I6 also addressed
  (effective-endpoint resolution, platform:sender debounce key,
  stub defaults aligned, overall deadline); I4 documented as a known
  limitation; I7 debris removed.

- r3 (delta re-review PASS + N1 clamp): independent delta review
  (VERDICT-R2.json, 79 probes) confirmed all four r1 blockers CLOSED
  and 19/19 tests green - real-event B1 held with exactly 1 engine
  call; '/zzz' and '/notacommand' text screened and held while
  '/stop'/'/STOP' skip with 0 engine calls; 53-case malformed-input
  fuzz: zero raises, all 10 exact-block malformed bodies kept the
  hold; heartbeat probe: slow /outcome leaves gate latency <1.0s
  with >=80% loop ticks; 50 concurrent outcome reports drained to 0;
  classifier breakage fails OPEN to screening. One new minor
  non-blocking finding N1 fixed in r3: non-finite numeric env
  ('inf'/'1e999') now clamps to the default via math.isfinite (was:
  degenerate env silently disabled holds / suppressed alerts).

## Test coverage

`tests/gateway/test_guardrail.py` - 20 cases against a loopback engine
stub (all tiers, hold reply, debounce, known/unknown-command matrix,
real-MessageEvent regression, malformed-response matrix, bad env,
corrupted state, slow-outcome latency bound, fail-open, default-off).
