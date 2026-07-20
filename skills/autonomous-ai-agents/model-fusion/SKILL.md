---
name: model-fusion
description: "Run supervised read-only synthesis across configured models."
version: 1.0.0
author: Jimmy Carson (@luckyjc) + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [delegation, model-fusion, synthesis, read-only]
    related_skills: [hermes-agent]
---

# Model Fusion Skill

Run one explicit, heterogeneous, read-only analysis using Hermes-native delegation lanes. The parent remains the sole orchestrator and artifact owner; the helpers only validate and persist deterministic state and never call providers or execute children.

## When to Use

- The user invokes `/model-fusion <request>` or explicitly asks for model fusion.
- Independent architecture and implementation perspectives would improve a recommendation.

Do not route ordinary requests here automatically. Do not use this workflow for edits, delivery, messaging, scheduling, memory, or unattended loops.

## Prerequisites

- Configured lanes named `architect`, `builder`, and `judge`.
- `architect` and `builder` are capped at `read_only_file`; `judge` has no tools.
- All three lanes disable parent MCP inheritance and fallback. Treat any `fallback_used=true` or requested/actual provenance mismatch as a failed source, never as successful fusion evidence.
- The parent can use `delegate_task`, `terminal`, `/agents`, and `/stop`.

Never mutate runtime configuration, restart a service, or add provider/model/base-URL/key fields to a delegation call. Lane configuration is session-stable and external to this skill.

## How to Run

A bare slash invocation supplies the text after `/model-fusion` as the request. Generate an opaque run ID and use the originating Hermes session ID as the guard key. Store runs in a profile-local model-fusion directory; never use a shared or repository directory.

Use `scripts/fusion_state.py` for run transitions and `scripts/fusion_judgment.py` for source assembly and judgment parsing. Import their narrow functions from a short parent-owned Python call through `terminal`; do not turn either script into a provider wrapper.

## Quick Reference

| Stage | Required transition | Completion criterion |
|---|---|---|
| Start | `create_run(root, run_id, session_id, request)` | Private run exists and the session guard is held. |
| Sources | One `delegate_task` batch | One consolidated completion has arrived for both ordered results. |
| Consolidate | `apply_source_batch(run, completion)` | Duplicate delivery returns the same `sources.json`. |
| Judge gate | `stage_judge(run)` | Returns true exactly once and only with at least one completed source. |
| Judge input | `build_judge_input(completed_sources)` | Each role is independently attributed and bounded. |
| Valid finish | `parse_judgment(raw)` then `finalize_judgment(...)` | Run is completed and guard released. |
| Invalid finish | `finalize_invalid_judgment(...)` | Run is partial, diagnostics are bounded, guard released. |

## Procedure

### 1. Start and acquire the guard

Call `create_run`. If it reports an active run for the originating session, stop and report that run is already active. One active run per originating session is the default; do not bypass or delete its guard.

Only allow-listed JSON artifacts may exist: `run.json`, `sources.json`, `judge.json`, `judgment.json`, `summary.json`, and `diagnostics.json`. Run directories are mode `0700`; files are atomically replaced at mode `0600`. Keep the request only in `run.json`; never persist transcript bodies.

### 2. Dispatch both sources in one native batch

Call `delegate_task` once with `background=true` and exactly this shape (fill in the two contracts; add no provider fields):

```json
{
  "background": true,
  "tasks": [
    {"lane": "architect", "goal": "<architect contract>", "context": "<run id, role, request, no-write contract>"},
    {"lane": "builder", "goal": "<builder contract>", "context": "<run id, role, request, no-write contract>"}
  ]
}
```

Each contract must include the same run ID and request, its exact role, a compact terminal-answer requirement, and these boundaries: read-only analysis; no child writes, memory, messaging, cron, recursive delegation, secrets, runtime config, restarts, commits, or pushes; no fallback. The architect focuses on constraints/design/risks. The builder focuses on feasibility/implementation/verification. Children return findings to the parent only.

Do not dispatch these as two calls. Wait for one consolidated source-batch completion. Native cancellation owns interruption; use `/agents` for observation and `/stop` when the user requests cancellation.

### 3. Consolidate once

Pass the ordered consolidated `results` array to `apply_source_batch`: index 0 is architect and index 1 is builder. Delivery is idempotent. Treat `unknown` and recovered outcomes as non-completed even if they contain text. Strip non-allow-listed result fields and bound each retained terminal result.

If no source completed, the helper marks the run failed and releases the guard. Do not start a judge.

### 4. Stage and dispatch the judge once

Call `stage_judge`. Continue only when it returns true. A false result means no eligible source or the judge was already staged; never dispatch again.

Build the input from completed sources only with `build_judge_input`. Append `templates/judge-prompt.md`, replacing its source placeholder with the bounded attributed blocks. Then call native `delegate_task` once:

```json
{"background": true, "lane": "judge", "goal": "Judge the attributed source findings and return only the required raw JSON object.", "context": "<run id plus bounded judge prompt>"}
```

The judge has no tools. Do not add fallback, provider, model, toolsets, or child-execution fields.

### 5. Validate, render, persist, release

On the judge completion, pass the native completion event to `extract_judge_output` to require `status=completed`, exact lane/model provenance, and no fallback. Pass the returned raw terminal summary directly to `parse_judgment`; do not strip fences or extract a JSON substring.

For valid output, render and persist only consensus, unique findings, divergences, rejected claims, final recommendation, confidence, assumptions, and exact requested/actual lane/model provenance. Call `finalize_judgment`; it writes compact judgment/summary artifacts and releases the guard.

For invalid output or provenance, call `finalize_invalid_judgment` with a low-cardinality code and bounded metadata-safe detail. Mark the run partial, not completed, and release the guard. Never persist the raw invalid response.

## Judgment Contract

The judge must return one raw JSON object with exactly these keys:

```json
{"consensus":[{"statement":"...","sources":["architect","builder"]}],"uniqueFindings":[{"statement":"...","sources":["architect"]}],"divergences":[{"statement":"...","sources":["architect","builder"]}],"rejected":[{"statement":"...","sources":["builder"],"reason":"..."}],"finalRecommendation":"...","confidence":"low|medium|high","unverifiedAssumptions":["..."]}
```

No fences, surrounding prose, unknown fields, empty attributions, unknown roles, or empty required strings are accepted.

## Pitfalls

1. **Dispatching the judge from each source event.** Sources are one batch; stage only after its one consolidated completion.
2. **Trusting recovered text.** Process restart is not execution durability. Unknown/recovered results never become completed.
3. **Retrying invalid judgment.** There is no fallback judge or repair loop. Finish partial and release the guard.
4. **Leaking context into diagnostics.** Use compact codes/counts/provenance, not prompts, responses, paths, exception text, or secrets.
5. **Granting child authority.** Only the parent writes artifacts. Children cannot write, delegate, remember, message, schedule, or reconfigure.

## Verification

Before reporting completion, verify:

- [ ] Architect and builder were submitted in one batch with lanes `architect` and `builder`.
- [ ] One consolidated completion was consumed idempotently.
- [ ] At least one source succeeded before the judge was staged exactly once.
- [ ] Judge used lane `judge`, no tools, and no fallback.
- [ ] Requested and actual provenance is present and matches policy.
- [ ] Strict raw JSON validation passed, or the run is partial with bounded diagnostics.
- [ ] Artifacts are allow-listed, compact, `0600`, under a `0700` run directory.
- [ ] The originating-session guard was released on every terminal path.
- [ ] No transcript body, secret, child write, config change, service restart, commit, or push occurred.

Execution is non-durable: completion delivery is durable after a child finishes, but a Hermes process restart during child execution produces an unknown outcome. Report that caveat rather than claiming success.
