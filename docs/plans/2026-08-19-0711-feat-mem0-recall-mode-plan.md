---
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
type: feat
---

# Configurable Mem0 Recall Modes

## Goal Capsule

Add a provider-stable `recall_mode` setting to the Mem0 memory provider so users can choose automatic context, explicit tools, or both. Hybrid and context modes must preserve the current-question correctness fix; tools mode deliberately delegates that responsibility to model-triggered search. The runtime policy must remain local to the Mem0 plugin, preserve existing behavior for old configurations, and avoid changing capture/write behavior. The shared memory setup parser may accept provider-specific trailing options because strict top-level parsing otherwise rejects every documented Mem0 setup flag before the plugin can consume it.

Authority order: this plan, the repository `AGENTS.md`, established memory-provider contracts, then local implementation conventions. Stop if the implementation requires a core model tool, mutating a conversation's prompt or tool set after provider construction, or changing Mem0's breaker and timeout semantics. Execution owns implementation and local verification; the shipping workflow owns review, commit, PR creation, and CI follow-through.

## Product Contract

### Summary

Mem0 currently performs one automatic current-question search and also exposes an aggressively described `mem0_search` tool. That combination is useful but can lead a model to repeat retrieval against the same question, consuming search quota without adding context. Add three explicit recall modes, with `hybrid` as the backward-compatible default.

### Problem Frame

The current automatic prefetch was deliberately introduced in [PR #55535](https://github.com/NousResearch/hermes-agent/pull/55535) to correct first-turn empty recall and stale prior-turn recall. Hybrid and context must not reverse that fix. Tools mode is an explicit opt-out from deterministic prefetch: its correctness depends on the model issuing an appropriate search. The feature should let users select that tradeoff and make hybrid guidance less reflexive while retaining explicit search for missing, history-sensitive, or multi-hop information.

An anonymized usage sample reached 580 retrievals and 421 additions in roughly 2.28 days on a 5,000-operation tier. At that pace, projected retrieval volume alone is about 7,892 operations per month. In an equal-window comparison, retrievals rose from 1,945 to 2,761 and explicit searches from 353 to 831. These figures motivate configurability; they are not a performance benchmark or a promise that prompt guidance eliminates every repeated call.

### Key Decisions

- **`hybrid` remains the default.** Existing configurations preserve automatic current-question context and all Mem0 tools. Governs R1, R2, R7.
- **`context` is context-only.** It performs automatic current-question recall and exposes no Mem0 tool schemas, matching established Honcho and Hindsight semantics. Governs R2, R4.
- **`tools` is tool-only.** It exposes all Mem0 tools and performs no automatic prefetch or context injection. Governs R2, R5.
- **Recall policy does not control capture.** Automatic turn synchronization and explicit write behavior remain unchanged. Governs R6.
- **Three modes are required, not guidance alone.** Prompt guidance can influence hybrid behavior but cannot guarantee a call ceiling. Context and tools provide deterministic automatic-retrieval policies for quota-sensitive users: one current-question attempt or zero automatic attempts, respectively. Governs R3, R4, R5, R9.

### Requirements

- **R1. Configuration contract:** Read `recall_mode` from `$HERMES_HOME/mem0.json`. Accept `hybrid`, `context`, and `tools`. Missing, non-string, or unsupported values resolve safely to `hybrid`; no new environment variable is added.
- **R2. Stable lifecycle:** Resolve and freeze the effective mode when the provider is constructed so prompt text, schemas, routing, and runtime behavior cannot diverge during that provider/agent lifetime. Changing the file takes effect only after restarting Hermes and constructing a new agent; `/new` may reuse the existing agent and is not an activation boundary. A process or gateway cache eviction that constructs a new agent also establishes a new prompt-cache lifetime.
- **R3. Hybrid behavior:** Start and consume the existing exact-current-question prefetch, expose all four Mem0 tools in their current order, and instruct the model to use injected context first. Explicit search remains appropriate when injected context is absent or insufficient, when prior conversation state changes the needed query, or when genuine multi-hop retrieval requires targeted follow-ups.
- **R4. Context behavior:** Start and consume exact-current-question prefetch, expose no Mem0 tools, and provide static guidance to use injected context when available. Timeout, breaker-open, backend-error, and empty-result paths are best-effort failures: inject no Mem0 block, make no backstop call, and do not imply that memory was consulted.
- **R5. Tools behavior:** Do not start a prefetch thread, call the backend automatically, or wait for a prefetch result. Expose all four Mem0 tools and provide static explicit-search guidance without claiming automatic context is present. This mode intentionally gives up deterministic current-question prefetch and delegates recall correctness to model-triggered `mem0_search`.
- **R6. Preserved behavior:** Do not change `sync_turn()`, explicit tool handlers, filters, top-k or rerank handling, breaker thresholds/cooldown, timeout values, current-query matching, stale-result rejection, error formatting, or backend selection.
- **R7. Backward compatibility:** Existing `mem0.json` files without `recall_mode` retain current hybrid runtime behavior. Invalid values do not prevent provider startup.
- **R8. Setup and discoverability:** Expose the setting through Mem0's configuration schema and custom setup flow, including `--recall-mode` for flag-driven setup paths. Preserve the setting across platform, self-hosted, and OSS reconfiguration.
- **R9. Documentation:** Document the mode table, default, JSON key, configuration path, unchanged capture behavior, and Hermes restart requirement in both Mem0 plugin and website memory-provider documentation. Explain that hybrid is the compatibility default, context suits users who want one best-effort automatic attempt with no model tools, and tools suits users who accept model-controlled recall in exchange for zero automatic retrievals.

### Flows

1. A user configures Mem0 without `recall_mode`; a new session behaves exactly as current hybrid recall does.
2. A user selects `context`; registration and post-initialization routing both omit Mem0 tools, while the exact current question is recalled once and injected as volatile context.
3. A user selects `tools`; no automatic Mem0 search occurs, and the model retrieves only through `mem0_search`.
4. A user selects `hybrid`; exact-current-question recall still occurs once, and the model searches again only when the injected context cannot answer the distinct retrieval need.
5. A user changes `recall_mode`; the current provider remains stable and the change is reflected after restarting Hermes so a new agent/provider is constructed.

### Acceptance Examples

- **AE1:** Given no `recall_mode`, when a provider handles a turn, then it performs the same one exact-query automatic search and exposes `mem0_search`, `mem0_add`, `mem0_update`, and `mem0_delete`.
- **AE2:** Given `context`, when the provider is registered and initialized, then `get_all_tool_schemas()` returns no Mem0 schemas and `has_tool("mem0_search")` is false, while current-query prefetch still works.
- **AE3:** Given `tools`, when `on_turn_start()` and `prefetch()` are called, then no thread or backend search is created and `prefetch()` returns an empty string immediately.
- **AE4:** Given `tools` and a context-dependent question, when the model follows its static guidance, then recall occurs through explicit `mem0_search`; if the model does not call the tool, the provider supplies no automatic correctness backstop.
- **AE5:** Given `hybrid`, when automatic context already answers the current query, then static guidance tells the model not to reflexively repeat the same search; targeted or multi-hop search remains available.
- **AE6:** Given an invalid value such as an object or unknown string, when the provider is constructed, then its effective mode is `hybrid` and initialization continues.
- **AE7:** Given any recall mode, when a turn completes, then existing Mem0 capture/synchronization behavior remains enabled.
- **AE8:** Given `context` and a timeout, open breaker, backend error, or empty result, when prefetch completes, then no Mem0 context is injected, no explicit search is attempted, and static guidance does not claim memory was consulted.
- **AE9:** Given an active provider and a changed `mem0.json`, when `/new` reuses the same agent, then prompt text and tool schemas remain unchanged; after Hermes restarts, a newly constructed provider uses the new mode.

### Success Criteria

- Users can select all three modes through documented configuration and setup paths.
- Quota-sensitive users can choose a deterministic automatic-retrieval policy: one best-effort current-question attempt in context mode or zero automatic attempts in tools mode, with the documented correctness tradeoff.
- Focused tests prove the schema, prompt, prefetch, lifecycle-routing, and setup matrix.
- Existing Mem0 backend and current-question tests remain green.
- No core model tools or conversation-loop behavior change. The only shared CLI change is generic pass-through parsing for provider-specific memory setup options.

### Scope Boundaries

In scope: provider-local retrieval policy, static prompt/schema guidance, Mem0 setup configuration, generic parsing of provider-specific memory setup options, tests, and docs. Out of scope: hard deduplication of semantically similar searches, exact quota accounting, write/capture controls, provider-wide config migration, dashboard UI, session-persisted mode snapshots across process or gateway agent reconstruction, changes to other memory providers, and live Mem0 API calls.

### Dependencies

The implementation depends only on existing `MemoryProvider`/`MemoryManager` contracts, Mem0's custom setup flow, and existing test doubles. No new production dependency is allowed.

### Outstanding Questions

None blocking. A future hard-deduplication layer may be considered if prompt/schema guidance is insufficient, but it requires argument/result reconciliation and is outside this change. Production usage can later validate how often hybrid guidance avoids a repeated explicit search; this change guarantees only the provider-side call ceilings of context and tools modes.

### Sources

- Current-question correctness and search backstop: [PR #55535](https://github.com/NousResearch/hermes-agent/pull/55535)
- Mem0 v3 provider surface: [PR #15624](https://github.com/NousResearch/hermes-agent/pull/15624)
- Provider-local implementation: `plugins/memory/mem0/__init__.py`
- Existing recall-mode precedent: `plugins/memory/honcho/__init__.py` and `plugins/memory/hindsight/__init__.py`
- Manager routing lifecycle: `agent/memory_manager.py`

## Planning Contract

### Key Technical Decisions

- **KTD1. Normalize at construction:** Add a small total normalizer and set the effective mode in `Mem0MemoryProvider.__init__` from `_load_config()`. `initialize()` may reuse that frozen configuration but must not expose a different schema after `MemoryManager.add_provider()` has built routing. The stability boundary is the provider/agent lifetime; restarting Hermes or reconstructing an evicted agent creates a new cache lifetime and may adopt changed configuration.
- **KTD2. Gate at provider edges:** Keep backend and tool-handler code unchanged. Gate automatic recall in `on_turn_start()`, `_start_prefetch()`, and `prefetch()`, and gate model tool exposure in `get_tool_schemas()`.
- **KTD3. Use immutable mode-specific guidance:** Use static prompt strings and copied/static search schemas. Do not mutate the module-level `SEARCH_SCHEMA`; repeated prompt/schema calls for one provider must be deterministic.
- **KTD4. Hide all tools in context mode:** This keeps schema exposure and dispatch routing consistent and matches existing memory-provider semantics. Automatic `sync_turn()` remains independent.
- **KTD5. Extend every custom setup route consistently:** Add one shared recall-mode selection helper/value path used by platform, self-hosted, and OSS flows, and a `--recall-mode` flag. Do not expose the option in only one backend mode.
- **KTD6. Preserve provider setup arguments generically:** Let `hermes memory setup <provider>` accept trailing provider-specific options through the shared parser. Keep interpretation inside each provider so this adds no provider-specific branch to the CLI.

### High-Level Design

`Mem0MemoryProvider` owns a normalized, session-stable recall policy. Hybrid/context share the existing automatic prefetch implementation. Tools mode exits before any automatic work. Hybrid/tools return the existing four tool schemas, with a mode-appropriate copied `mem0_search` description; context returns an empty list. The setup code persists `recall_mode` beside other behavioral Mem0 settings. Documentation presents a single semantic table.

### Implementation Constraints

- Preserve prompt-prefix caching: no query, result, breaker state, time, or config reload enters `system_prompt_block()`.
- Preserve strict current-question behavior and the bounded prefetch wait.
- Do not add `HERMES_*` or `MEM0_*` behavioral environment variables.
- Do not add core tool schemas or edit core agent-loop files.
- Public docs, issue, commit, and PR text must avoid assistant attribution and em/en dashes.
- Tests must not contact Mem0 or any external service.

### Sequencing

Implement provider behavior and its contract tests first. Then extend setup/config tests, followed by documentation. Run focused verification after each unit and the broader Mem0 suite at the end.

## Implementation Units

### U1. Provider recall policy

**Goal:** Implement stable three-mode behavior at the Mem0 provider boundary.

**Requirements:** R1, R2, R3, R4, R5, R6, R7.

**Files:**

- Modify `plugins/memory/mem0/__init__.py`
- Modify `tests/plugins/memory/test_mem0_v3.py`
- Modify or add the narrowest existing manager integration test only if provider-level registration assertions cannot prove routing consistency

**Approach:** Add normalization and frozen provider state, define mode-specific prompt/search guidance without mutating shared schema objects, bypass every automatic prefetch entry in tools mode, and hide schemas in context mode. Reuse existing test doubles and preserve existing current-query tests unchanged where possible.

**Test Scenarios:** AE1 through AE9, repeated prompt/schema stability, schema order, registration before initialization, same-agent config-change stability, slow-prefetch behavior, and breaker behavior on both automatic and explicit paths.

**Verification:**

```bash
scripts/run_tests.sh tests/plugins/memory/test_mem0_v3.py -q
```

### U2. Setup and configuration exposure

**Goal:** Make `recall_mode` selectable and persist it consistently across Mem0 setup modes.

**Requirements:** R1, R7, R8.

**Files:**

- Modify `plugins/memory/mem0/_setup.py`
- Modify `tests/plugins/memory/test_mem0_setup.py`
- Modify `plugins/memory/mem0/__init__.py` for the provider configuration schema
- Modify `hermes_cli/subcommands/memory.py` and its parser test so documented provider-specific setup flags reach Mem0's existing `parse_flags()` path

**Approach:** Add the schema choice and CLI flag, normalize setup input to the same three values, and merge-write the selected setting in platform, self-hosted, and OSS paths without disturbing secrets or backend configuration. Existing values remain intact when no new selection is supplied.

**Test Scenarios:** flag parsing, valid mode persistence for each backend path, existing-value preservation, default hybrid behavior, dry-run non-mutation, and invalid flag handling consistent with existing setup conventions.

**Verification:**

```bash
scripts/run_tests.sh tests/plugins/memory/test_mem0_setup.py -q
```

### U3. User documentation

**Goal:** Explain the recall tradeoff and configuration contract in both documentation surfaces.

**Requirements:** R9.

**Files:**

- Modify `plugins/memory/mem0/README.md`
- Modify `website/docs/user-guide/features/memory-providers.md`

**Approach:** Add the same compact mode table and JSON example to both documents, describe `hybrid` as the compatibility default, guide quota-sensitive users toward context or tools with their correctness tradeoffs, state that capture remains enabled in every mode, and require restarting Hermes after changes.

**Test Scenarios:** Manual cross-check that names, defaults, semantics, and JSON spelling match provider/setup code.

**Verification:**

```bash
rg -n "recall_mode|hybrid|context|tools" plugins/memory/mem0/README.md website/docs/user-guide/features/memory-providers.md
```

## Verification Contract

Run from the isolated worktree with the repository test runner:

```bash
scripts/run_tests.sh tests/plugins/memory/test_mem0_v3.py tests/plugins/memory/test_mem0_setup.py tests/plugins/memory/test_mem0_backend.py -q
ruff check plugins/memory/mem0/__init__.py plugins/memory/mem0/_setup.py tests/plugins/memory/test_mem0_v3.py tests/plugins/memory/test_mem0_setup.py
git diff --check
```

Then run the full repository suite before PR handoff if the environment can complete it within the shipping window:

```bash
scripts/run_tests.sh
```

Quality gates:

- No live network calls or credentials in tests.
- Provider prompt and schema outputs are deterministic within a session.
- Context-mode routing agrees before and after initialization.
- Tools mode records zero automatic backend searches.
- Existing exact-current-question, timeout, and breaker tests remain green.
- The final diff contains no core tool or agent-loop changes and no unrelated formatting.

## Definition of Done

- U1 is done when the full mode matrix and lifecycle invariants pass focused tests without changing backend/tool-handler semantics.
- U2 is done when configuration is discoverable and persisted consistently across custom setup paths with regression coverage.
- U3 is done when both documentation surfaces match the executable contract.
- The feature issue exists and the PR references it.
- Focused tests, lint, and diff checks pass; full-suite status is reported truthfully.
- Review findings are resolved or explicitly documented, CI is watched to a terminal merge-ready or evidenced unrelated-failure state, and no merge or deployment occurs.
- Abandoned experiments, debug output, temporary repository files, and unrelated changes are absent from the final diff.
