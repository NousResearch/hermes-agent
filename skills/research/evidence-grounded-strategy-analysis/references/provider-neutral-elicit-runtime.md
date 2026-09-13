# Provider-neutral Elicit runtime hardening

Use this when Elicit (or a similar evidence framework) runs model calls through Hermes rather than a provider-specific CLI.

## Active runtime inheritance

- Treat profile configuration as a default, not proof of the active session's provider/model.
- When launched by a live Hermes session, read `HERMES_SESSION_ID` and resolve the canonical session row through `SessionDB(read_only=True).get_session(...)`.
- Prefer the row's effective `model`, `billing_provider`, and `billing_mode`; fall back to profile config only when no session row is available.
- If a caller repeats the active bare model name as an explicit override, preserve the active provider instead of re-detecting that name onto another route.
- Record the provider/model returned by the completed invocation, because fallbacks may differ from the requested route.

## Tool and prompt isolation

- Run a fresh agent with context files and persistent memory disabled.
- For evidence-free verifier calls, require an empty tool surface.
- For corpus-assisted calls, require exact equality with the approved read-only corpus tools; fail closed on additions or omissions.
- Send prompts over stdin or an equivalent private channel, never argv.
- Parse locally and validate against the same JSON schema used by every backend.

## Timeout cleanup

Provider calls may spawn descendants. Use `Popen(..., start_new_session=True)`, wait with the per-call timeout, and on timeout terminate the whole process group (`SIGTERM`, short wait, then `SIGKILL`). A parent-only kill can leave chargeable or stateful descendants alive.

Regression-test descendant cleanup with a child that spawns another process, then verify both are gone after timeout.

## Durable invocation metadata

Persist metadata in both sealed artifacts and the event ledger:

- backend, provider, resolved model;
- duration and output digest;
- input/output tokens;
- estimated or actual cost, cost status, and cost source;
- reserved `max_budget_usd`;
- tool calls;
- failure text.

Do not write `model: default` when the backend returned a resolved model.

## Budget truthfulness

Distinguish three different guarantees:

1. **Scheduling reservation:** Elicit reserves a stage allocation before launching work.
2. **Provider-side hard cap:** only claim this when the backend/provider actually enforces a pre-spend dollar limit.
3. **Post-call accounting check:** detects an overage after execution but cannot prevent metered spend.

For `subscription_included` sessions, monetary cost may legitimately be zero while scheduling reservations still bound work. For metered Hermes providers without a provider-side dollar-cap API, document post-call enforcement explicitly; never describe it as a pre-spend cap.

## Verification sequence

1. RED tests for unknown backend, schema violation, tool-boundary mismatch, active-session inheritance, and descendant timeout cleanup.
2. Focused backend/failure/bounds tests.
3. Full suite, compile check, and diff check.
4. Live schema-valid smoke call showing resolved provider/model and cost metadata.
5. Secret scan and exact-tree review before commit.
