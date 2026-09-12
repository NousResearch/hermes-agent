# Accepted Correction C9 — Native-Child V1 Initial Routing Only

Stage: PRODUCT_ARCHITECTURE_CORRECTION
Status: ACCEPTED
Execution-State: CANONICAL_CORRECTION_ONLY
Affected-Task: T004
Project-Version: 0.1.1
Architecture-Version: 1
Owner-Decision: «Хорошо. Делаем по твоей рекомендации, а это полезное дополнение оставим для следующей версии»
Supersedes: C8 in full
Historical-C8-Source-SHA256: `22dea2082a4d92ebf7a5f91ddf35a9b695fcc578dce7b8d376e7511c860dbbc1`

## Decision and rationale

V1 retains `native_child` as a supported execution kind, but routes only before each initial normalized delegated-child launch. This is the proportional boundary required for pre-credential model selection. The confirmed scope-pressure cause was broad solution-shaped prompts converting an accepted fallback outcome into infrastructure before proportionality was proven; sequential acceptance of earlier clauses does not override this later owner correction.

## Normative v1 behavior

- `delegate_task` invokes the resolver once for each new normalized child execution attempt before credentials, child construction and run.
- Per-child pins, independent mixed `route | pass_through | stop | router_error` decision outcomes, parent SessionDB lifecycle authority and renderer isolation remain required. This independence does not create per-child recovery for native credential/construction failures: active `pass_through` preserves the existing batch-wide credential/build error semantics.
- If a router-selected child signals fallback requiring a route/model change, that child attempt terminates with a bounded visible failure. Hermes does not construct or resubmit a replacement inside the same `delegate_task` unit.
- A parent may later issue a normal new delegate call. That call is a new execution and routes normally.
- No-router and explicit `pass_through` preserve existing native in-agent fallback unchanged.
- Active batches resolve the shared native credential bundle once for all `pass_through` children; router-selected `route` children use their selected post-decision configuration. Ordinary native credential/build errors are not converted into synthetic child results.
- C6/C7 completed `main_turn` behavior is unchanged. Future `kanban_worker` semantics are unchanged except that formerly global prose is qualified by execution kind.

## Deferred future version

Automatic native-child replacement/re-entry remains a useful deferred idea. It requires a new product/architecture task and separate authorization. V1 includes no C8 attempt specification/outcome, construction-lock change, current-child handoff, automatic replacement, extra dispatcher, queue, coordinator, retry engine or store, and no public API/version/capability expansion.
