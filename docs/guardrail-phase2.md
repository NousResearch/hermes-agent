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

## Test coverage

`tests/gateway/test_guardrail.py` - 9 cases against a loopback engine
stub: allow/review/block/chain_down/unreachable, hold reply, owner
alert + debounce, command/control bypass, empty text, default-off
inertness.
