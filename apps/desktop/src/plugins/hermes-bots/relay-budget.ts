// #93911: a delivered turn runs on the target gateway, so the client must
// outlive the backend's own bound. Without this the call fell to the pool's
// generic 30s deadline and every long turn (Computer Use, deep research) came
// back as an unclassified failure.
//
// The backend's MAXIMUM WORK budget is spelled out below. The client deadline
// must be strictly GREATER than it: after those bounded waits the handler still
// has to classify the failure, build and run the retry, classify/serialize the
// terminal result, unwind the temp-file and lock scopes, and get the JSON-RPC
// response back through the event loop. A call that consumes nearly all of the
// work budget would otherwise lose the race to this timer by milliseconds and
// reproduce #93911 at the upper boundary — the backend knowing a typed reason
// while Desktop reports its generic timeout first.
//
// These three are mirrors of backend values, so a change there must not
// silently invalidate this constant: relay-deliver-budget.test.ts compares
// them with hermes_cli/config_defaults.py and tools/bot_relay.py and fails if
// the mirrors drift or the margin stops being positive.
export const RELAY_TURN_LOCK_WAIT_MS = 120_000 // bot_mode.turn_wait_seconds default
export const RELAY_TURN_ATTEMPT_MS = 600_000 // tools/bot_relay.py TURN_ATTEMPT_TIMEOUT_SECONDS
export const RELAY_TURN_MAX_ATTEMPTS = 2 // first attempt + the policy-gated re-run

export const RELAY_DELIVER_BACKEND_CEILING_MS =
  RELAY_TURN_LOCK_WAIT_MS + RELAY_TURN_ATTEMPT_MS * RELAY_TURN_MAX_ATTEMPTS

// Settlement + transport headroom on top of the ceiling, so a backend that
// answers at its own limit still wins the race against this timer.
export const RELAY_DELIVER_SETTLEMENT_MARGIN_MS = 180_000
// tools/bot_relay.py REPLY_WAIT_SECONDS rebuilds this sum and waits past it for the relay's timeout reply.
export const RELAY_DELIVER_TIMEOUT_MS = RELAY_DELIVER_BACKEND_CEILING_MS + RELAY_DELIVER_SETTLEMENT_MARGIN_MS
