# Intent-Routed Capabilities: Canary and Rollback Runbook

**Status:** implementation-ready; disabled by default; production not activated.

## Scope

This runbook validates intent routing in one isolated Hermes profile without changing the default profile, gateway, dashboard, or existing sessions. Routing reduces the fixed tool/skill prompt surface; it does not reduce model context capacity or authorization.

## Preconditions

- Deploy a build containing the reviewed `feat/intent-routed-capabilities` commit to a non-serving checkout.
- Keep `tools.intent_routing.enabled: false` in the default profile.
- Use the canonical runtime paths: `HERMES_HOME=/opt/data`, CLI `/opt/data/.local/bin/hermes`.
- Preserve the currently serving release identifier and rollback checkout before testing.

## Create the isolated canary

Run only after the feature commit is installed in the canary checkout:

```bash
export HERMES_HOME=/opt/data
/opt/data/.local/bin/hermes profile create intentcanary --clone --no-alias \
  --description "Isolated intent-routing performance and fallback canary"

export HERMES_HOME=/opt/data/profiles/intentcanary
/opt/data/.local/bin/hermes config set tools.intent_routing.enabled true
/opt/data/.local/bin/hermes config set tools.intent_routing.direct_schema_token_budget null
/opt/data/.local/bin/hermes config get tools.intent_routing
```

Do not switch the sticky default profile and do not restart the production gateway for this canary.

## Validation cohort

Start fresh canary sessions for these representative intents:

1. Repository debugging and file edits.
2. Web research with extraction.
3. Email or another narrow integration workflow.
4. A request that deliberately needs a deferred authorized tool.
5. A request for a tool that is unauthorized in the profile.
6. A long session that crosses compression and is then resumed in a fresh process.

Existing legacy sessions must remain on legacy routing. Existing routed sessions must restore their persisted frozen plan even if the global feature flag is later disabled; this preserves the model-visible prefix and deferred fallback catalog.

## Acceptance gates

- Routing p95 below 100 ms using the already captured authorization snapshot.
- Direct tool schemas at or below 6,000 estimated tokens for typical narrow intents; broad coding work may exceed this only when its matched capabilities require it.
- Every authorized tool belongs to exactly one of the direct or deferred sets.
- Deferred authorized built-in, plugin, and MCP tools can be searched, described, and called.
- Unauthorized tools cannot be searched, described, or called.
- Repeated turns and process resume preserve the plan and tool-schema hash.
- Compression descendants inherit the complete plan.
- Conditional skills remain visible when their required tool is authorized but deferred.
- No increase in provider errors, tool-call validation failures, approval bypasses, or session-resume failures.
- Context-window configuration remains unchanged.

## Observability

Search the canary logs for:

```text
Intent routing plan frozen
Intent routing plan restored
Intent routing fallback invoked
unsupported capability plan version
tool schema hash mismatch
not available in this session
```

Healthy signals:

- `plan frozen` occurs once for each new routed lineage.
- `plan restored` retains the same version, manifest hash, schema hash, and direct/deferred counts.
- `fallback invoked` names only authorized deferred tools and contains no arguments or secrets.
- Routing time remains below the acceptance threshold.

Failure or rollback triggers:

- Any unauthorized tool becomes discoverable or executable.
- A routed session cannot resume or loses deferred fallback after compression.
- Tool schemas or system prompt change across ordinary turns.
- Routing p95 exceeds 100 ms after warm-up.
- Fresh-request latency or provider error rate regresses materially against the default-profile control.

Validation window: at least 20 fresh sessions and 5 compression/resume lineages. Owner: Hermes operator/board delegate.

## Rollback

### Stop routing for new sessions

```bash
export HERMES_HOME=/opt/data/profiles/intentcanary
/opt/data/.local/bin/hermes config set tools.intent_routing.enabled false
```

This prevents new plans. It intentionally does not mutate existing routed lineages; they continue with their persisted frozen plans. To test legacy behavior immediately, start a new session after disabling the flag.

### Retire the canary profile

After exporting any required evidence:

```bash
export HERMES_HOME=/opt/data
/opt/data/.local/bin/hermes profile delete intentcanary
```

Profile deletion is destructive and requires normal operator confirmation. Do not delete the default profile.

### Code rollback

If the build itself is defective, restore the retained pre-feature checkout/release and restart only the isolated canary process. Production services are outside this canary and should have unchanged PIDs throughout.

## Current local evidence

Focused regression suite: **270 passed**.

Representative schema benchmark against the same 10,708-token legacy tool array:

| Intent | Routed schema tokens (estimated) | Reduction | Routing p95 |
|---|---:|---:|---:|
| Coding | 6,169 | 42.4% | 15.02 ms |
| Web research | 1,787 | 83.3% | 11.29 ms |
| Email | 1,322 | 87.7% | 9.66 ms |

The repository-wide runner collected more than 42,000 tests and was stopped after exposing unrelated environment failures and a runner/worktree cleanup collision. The feature's affected-area suite is green; upstream CI remains the required full-suite release gate before merge or production activation.
