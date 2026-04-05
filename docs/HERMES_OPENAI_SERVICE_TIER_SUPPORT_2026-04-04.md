---
title: "Hermes - OpenAI Service Tier Support - Architecture Plan"
date: 2026-04-04
status: active
fallback_policy: forbidden
owners: [aelaguiz]
reviewers: []
doc_type: architectural_change
related:
  - CONTRIBUTING.md
  - docs/plans/2026-04-04-openai-service-tier-architecture.md
  - https://developers.openai.com/api/docs/guides/flex-processing
  - https://developers.openai.com/api/docs/guides/priority-processing
  - https://developers.openai.com/api/reference/overview#authentication
  - https://developers.openai.com/api/reference/resources/chat
---

# TL;DR

<!-- arch_skill:block:implementation_audit:start -->
# Implementation Audit (authoritative)
Date: 2026-04-04
Verdict (code): PENDING RE-AUDIT
Manual QA: `openai-codex` live smoke passed; direct API-key OpenAI still pending

## Code blockers (why code is not done)
- No service-tier-specific code blocker is currently known after the reopened Codex implementation pass. A fresh `audit-implementation` pass is still required before claiming code-complete.

## Reopened phases (false-complete fixes)
- Phase 4 — final verification, direct API-key OpenAI manual exercise, and PR-readiness follow-through remain open.

## Missing items (code gaps; evidence-anchored; no tables)
- Re-run `arch-step audit-implementation` against the post-fix branch state.
- Complete the still-pending direct API-key OpenAI live verification path.

## Non-blocking follow-ups (manual QA / screenshots / human verification)
- Run a live `hermes` chat against a direct `api.openai.com` API-key route and confirm that configured `service_tier` is included there too; the `openai-codex` live smoke already passed in a temporary `HERMES_HOME` copied from the personal auth store, and the dumped request body included `service_tier: "priority"`.
- Complete the contribution-guide cross-platform impact review before PR submission.
- Re-run `pytest tests/ -v` in a clean CI-like environment before merge. The local full-suite rerun after the Codex fixes still exercised the repo broadly, but the remaining failures were outside the service-tier surface and included unrelated Matrix voice, provider/setup, managed-tool/modal, skill-manager, transcription, terminal requirements, and update-gateway restart areas.
- The previous code-complete audit is no longer authoritative because the required scope changed and the post-fix code has not yet had a fresh dedicated audit pass.
<!-- arch_skill:block:implementation_audit:end -->

## Outcome

 Hermes gains opt-in OpenAI `service_tier` support on both route-compatible direct OpenAI API-key requests and Hermes `openai-codex` requests through the normal config, runtime-resolution, and request-building paths, including first-class agent surfaces such as ACP, the API server, and delegated subagents, with fully backward-compatible behavior when the option is unset or not applicable.

## Problem

 Hermes now has a normalized end-to-end `request_options` seam on this branch, but the remaining Codex-critical request shaping is still split across `run_agent.py` and `agent/auxiliary_client.py`, and the current branch explicitly treats `openai-codex` as an omit path for `service_tier`. The implementation therefore no longer matches the required support surface even though most propagation plumbing is already in place.

## Approach

 Add the narrowest Hermes-native persisted setting for OpenAI `service_tier`, normalize it once through one shared request-options seam adjacent to route resolution, propagate it through every real first-class execution surface and per-turn agent state path, and introduce one narrow shared owner path for OpenAI request-option compatibility and injection. The required support surface now includes both direct OpenAI API-key routes and Hermes `openai-codex`; other non-OpenAI and unrelated OAuth-backed routes remain out of scope unless later research proves a broader surface is both documented and necessary.

## Plan

 Lock the North Star and scope boundaries first, then deep-dive the affected call sites, external-ground the documented direct API surface plus the required Codex runtime surface, implement the end-to-end plumbing plus direct and Codex OpenAI request injection, and finish with targeted tests plus a full-suite pass that prove config migration, runtime propagation, compatibility gating, Codex parity, graceful degradation on genuinely unsupported routes, and behavior preservation.

## Non-negotiables

- No silent fallback or best-effort forwarding to unverified endpoints.
- No duplicated source of truth for `service_tier` compatibility or request injection.
- No stale CLI, gateway, or cron agent reuse when request options change.
- No new default billing or latency behavior.
- No wrapper-only environment-variable escape hatch when the normal Hermes config path can own the feature.
- No regressions to existing config loading, runtime resolution, or request behavior for users who do not opt in.
- No generalized multi-provider framework or spillover behavior for non-OpenAI models or providers.
- Naming must follow the upstream OpenAI mechanism precisely enough that the code, config, docs, and PR all clearly describe support for OpenAI `service_tier` on the actual Hermes-supported OpenAI request surfaces, including `openai-codex`, rather than an invented Hermes-local label such as “fast mode”.
- Degradation must follow Hermes’ existing pattern for optional capabilities: preserve the main request path, omit the unsupported optional field deterministically, and surface a concise warning through existing logging.
- Logging must use Hermes’ existing redacted logger conventions; no secrets in logs and no parallel logging system.
- `openai-codex` can no longer be treated as an acceptable blanket omit path. If any Codex exclusion remains after implementation, it must be narrow, evidence-backed, and explicitly called out in the plan and tests.

<!-- arch_skill:block:planning_passes:start -->
<!--
arch_skill:planning_passes
deep_dive_pass_1: done 2026-04-04
external_research_grounding: done 2026-04-04
deep_dive_pass_2: done 2026-04-04
deep_dive_pass_3: done 2026-04-04
deep_dive_pass_4: done 2026-04-04
recommended_flow: deep dive -> external research grounding -> deep dive again -> phase plan -> implement
note: This is a warn-first checklist only. It should not hard-block execution.
-->
<!-- arch_skill:block:planning_passes:end -->

# 0) Holistic North Star

## 0.1 The claim (falsifiable)

If Hermes treats OpenAI `service_tier` as validated metadata for both direct OpenAI API-key routes and Hermes `openai-codex`, carries it through the existing config, route-resolution, and request-building paths via one normalized request-options seam, and keeps non-OpenAI routes behaviorally untouched, then users can enable supported service tiers without wrappers or local forks, unsupported routes will remain unchanged except for explicit warning-level operator feedback, and the implementation will reduce rather than widen architectural drift. This claim is false if request options can still be dropped by any first-class runtime surface, if `openai-codex` remains a blanket omit path, if unsupported routes receive speculative `service_tier` fields, if non-OpenAI paths gain new behavior or complexity, if backward-compatible behavior changes for non-opted-in users, or if the patch leaves multiple owners for the compatibility rules.

## 0.2 In scope

- Add one optional persisted OpenAI service-tier setting in the narrowest maintainable Hermes config shape: `model.request_options.service_tier`. It must stay a one-field direct-OpenAI request setting rather than becoming a generalized multi-provider framework.
- Add migration-safe normalization for the existing mixed `model` config shape so older configs continue to load cleanly.
- Introduce one shared normalized request-options seam tied to the resolved route facts. The transport runtime dict remains transport-only, and request-options normalization lives adjacent to route resolution so downstream callers consume one canonical normalized shape rather than re-parsing config.
- Propagate normalized `request_options` through all first-class execution surfaces that construct or reuse agents today: CLI, gateway, cron, ACP, the API server, delegated subagents, and `AIAgent` initialization.
- Update any route or cache signatures whose correctness depends on the effective runtime request shape so request-option changes do not reuse stale agents.
- Introduce one narrow shared owner path for OpenAI request-option validation, compatibility gating, and request-field injection, and have both the main runner and the auxiliary client depend on that owner for the direct OpenAI and `openai-codex` paths they actually use.
- Support `service_tier` on both direct OpenAI API-key routes and Hermes `openai-codex`, with the direct API surface grounded in official OpenAI docs and the Codex surface grounded in Hermes runtime behavior plus live validation.
- Integrate degraded and error paths into Hermes’ existing logger conventions and redacted rotating-log infrastructure instead of inventing a new reporting path.
- Add targeted tests for config loading and migration, runtime normalization, propagation, compatibility gating, and request shaping.
- Name the feature, config field, helper symbols, log messages, test names, and eventual PR around the upstream OpenAI mechanism being supported: `service_tier` on Hermes-supported OpenAI request surfaces, including `openai-codex`. Avoid vague or productized aliases that obscure the actual behavior under review.

Allowed architectural convergence scope:

- Converge duplicated OpenAI request-option logic into one shared helper or adapter boundary, including the Codex-specific route facts that are now part of the required support surface.
- Update nearby call sites and comments if that is required to make the owner path unambiguous.
- Do not widen this change into a full rewrite of Responses orchestration, streaming, or response normalization unless later planning proves that the smaller convergence boundary is insufficient.
- Do not add generalized non-OpenAI abstractions, validation rules, or request-shaping hooks in the name of future-proofing.

## 0.3 Out of scope

- Changing Hermes defaults so any service tier is enabled by default.
- New slash commands, gateway commands, or other user-facing control surfaces beyond normal config.
- Best-effort forwarding of `service_tier` to arbitrary OpenAI-compatible proxies, custom `base_url` deployments, or undocumented third-party endpoints.
- Assuming that OpenAI-branded routes other than Hermes `openai-codex` automatically inherit this plan’s support surface.
- Any new behavior, validation, or request shaping for non-OpenAI providers or models.
- Any OAuth-backed ChatGPT or Codex backend path other than Hermes `openai-codex`, unless later official documentation shows the feature on that surface and the plan is explicitly expanded.
- Treating ad hoc constructor-only paths outside the normal runtime-resolution story as covered when they are not. For phase 1, [batch_runner.py](/Users/aelaguiz/workspace/hermes-agent/batch_runner.py) and [gateway/builtin_hooks/boot_md.py](/Users/aelaguiz/workspace/hermes-agent/gateway/builtin_hooks/boot_md.py) are explicitly excluded from acceptance unless implementation proves they can be seeded with no new architectural branch and a representative existing test seam.
- Broad refactors whose only benefit is stylistic cleanup outside the request-options path.
- Persisting or surfacing new billing telemetry unless later passes show that the API reliably returns service-tier metadata and that the extra surface is worth the complexity.
- Prompt changes or model-instruction changes; this is deterministic runtime and request-shaping work.
- A compatibility shim that silently retries direct OpenAI requests without `service_tier` after a claimed-supported route rejects the field.

## 0.4 Definition of done (acceptance evidence)

- Hermes accepts an optional validated persisted OpenAI service-tier setting in the chosen narrow config shape without breaking older config shapes.
- Route resolution and request-options normalization produce a canonical normalized `request_options` shape, and the option survives end-to-end through CLI, gateway, cron, ACP, the API server, delegated subagents, and `AIAgent`.
- Direct OpenAI API-key Responses requests include `service_tier` when configured and route-compatible.
- Hermes `openai-codex` requests also include `service_tier` when configured and route-compatible; a blanket Codex omission is not an acceptable final state.
- Unsupported or unverified routes do not receive speculative `service_tier` fields; they degrade by deterministic omission of the optional field while preserving the underlying request path and emitting a warn-once warning-level log for each effective unsupported route within a given agent or job lifecycle.
- Non-OpenAI routes and unrelated OAuth-backed routes remain behaviorally unchanged except for explicit omission of the unsupported optional field and any documented warning path.
- Agent reuse cannot make `request_options` stale: CLI and gateway refresh `agent.request_options` per turn, cron seeds the current value on each fresh agent, ACP and API server construction paths seed the current value on fresh agents, and delegation either inherits the parent value or re-normalizes it against the child override runtime.
- Existing behavior is preserved when the option is unset, proven by targeted tests around the affected config, runtime, and request-building paths.
- Claimed-supported routes that still reject the field fail loud enough to expose a bad compatibility assumption instead of silently self-healing with an undocumented retry path.
- Logging for degraded and unexpected paths follows the feature-specific target aligned with `CONTRIBUTING.md`: use the module logger, warnings for expected degradation, `exc_info=True` on unexpected errors, redacted output, and no secrets.
- The implementation leaves one canonical owner for `service_tier` compatibility and request injection rather than duplicating literal values or allowlists across call sites, and that owner must cover both direct OpenAI and `openai-codex`.
- The implementation validates the current Responses-documented `service_tier` request enum without introducing a local model-by-model support matrix; unsupported model or tier combinations fail through the normal upstream error path.
- `flex` may be passed through as a documented value on compatible direct OpenAI routes, but phase 1 does not widen timeout or retry behavior and does not promise Flex-specific ergonomics beyond existing fail-loud transport handling.
- The implementation does not add new behavioral branches for any non-OpenAI provider.
- The naming across config, code, tests, docs, and the final PR accurately reflects the supported upstream mechanism so maintainers can evaluate the patch without reverse-engineering what “mode”, “boost”, or other alias actually means.
- The targeted new tests pass, `pytest tests/ -v` passes, the changed path is exercised manually through `hermes`, and cross-platform impact is reviewed before the PR is considered ready.

Smallest credible evidence:

- Unit tests for config migration and normalization.
- Unit tests for runtime-provider normalization and compatibility gating, including direct API-key route vs `openai-codex` vs genuinely unsupported route distinctions.
- Request-builder tests that prove include and omit behavior on supported vs unsupported routes, including the required Codex path.
- Targeted CLI, gateway, cron, ACP, API server, and delegation tests that prove propagation and cache or reuse boundaries.
- A full `pytest tests/ -v` pass as the pre-merge regression gate required by `CONTRIBUTING.md`.
- A manual `hermes` exercise of the changed path before PR submission, also required by `CONTRIBUTING.md`.
- A cross-platform impact review before PR submission, with the expected impact kept low because the feature is config and request-shaping work rather than terminal or process-management work.

## 0.5 Key invariants (fix immediately if violated)

- No fallbacks.
- No speculative forwarding of undocumented request fields to unverified endpoints.
- No dual sources of truth for Responses request-option compatibility or injection.
- No stale agent reuse when `request_options` change.
- No default cost or latency change when users do nothing.
- Fail loud on invalid configured `service_tier` values.
- Unsupported-route behavior must be explicit and deterministic, not hidden best effort.
- Graceful degradation applies only to omission of the optional field on known-unsupported routes; it must not silently reroute, silently retry, or change the underlying model route.
- No new generalized complexity for providers or models that are not the required OpenAI support surfaces.
- No local model-tier support matrix that can drift from upstream OpenAI support or pricing docs.
- No ambiguous feature naming: use upstream OpenAI terminology for the supported mechanism and avoid Hermes-local euphemisms that could confuse reviewers or users.
- Use existing Hermes logging conventions and redaction; never log secrets while explaining degraded or failed behavior.
- No blanket `if provider == "openai-codex": omit` rule may survive in the final implementation.

# 1) Key Design Considerations (what matters most)

## 1.1 Priorities (ranked)

1. Preserve Hermes-native architecture so maintainers see one canonical flow instead of a feature patch.
2. Preserve backward-compatible behavior completely for users who do not opt in.
3. Keep the implementation as narrow and low-complexity as possible.
4. Be correct on the explicitly required OpenAI routes first: direct API-key OpenAI and `openai-codex`.
5. Make end-to-end propagation real across CLI, gateway, cron, ACP, API server, delegated subagents, and agent reuse boundaries.
6. Name the feature and PR in upstream OpenAI terms so the review surface is self-explanatory.
7. Keep the patch focused, contribution-guidelines-compliant, and fully tested.
8. Leave no behavioral footprint on non-OpenAI models or providers.

## 1.2 Constraints

- Hermes currently mixes scalar and dict-shaped `model` config state, so migration scope is real.
- `openai-codex` resolves to the ChatGPT Codex backend, not the documented direct OpenAI API surface, so the plan must support it without pretending the public direct-API docs automatically cover it.
- CLI, gateway, and cron currently pass only a fixed subset of resolved runtime fields.
- Responses request shaping is already split between the main runner and the auxiliary client.
- Prompt caching and system-prompt stability must not be disturbed by the change.
- `CONTRIBUTING.md` requires focused PRs, full test runs before submission, and robustness via graceful degradation rather than feature-local hacks.
- As of 2026-04-04, official OpenAI API docs document `service_tier` on direct API requests authenticated with `Authorization: Bearer $OPENAI_API_KEY`; that grounds the direct route, but not the Codex backend.
- No requirement in the current official OpenAI API docs suggests that `service_tier` is an OAuth-only feature for the direct API, but the docs also do not currently document the ChatGPT Codex backend contract Hermes uses for `openai-codex`.

## 1.3 Architectural principles (rules we will enforce)

- Request policy belongs to one normalized request-options seam grounded in resolved runtime facts, not to ad hoc call-site parsing.
- Compatibility gating and request-field injection must have one canonical owner.
- Use the smallest convergence boundary that removes duplicate truth.
- Preserve existing defaults unless the user explicitly opts in.
- Prefer fail-loud validation over permissive guessing.
- Do not introduce a generalized provider-agnostic framework when a narrow direct-OpenAI owner path is sufficient.
- Do not change non-OpenAI provider behavior beyond passive no-op plumbing that is strictly required for shared initialization signatures.
- Do not hardcode model-by-model `service_tier` support allowlists in Hermes when upstream keeps support on pricing or capability surfaces that can change independently.
- Prefer per-turn mutable request state over cache-signature widening when the option changes request construction but does not change the frozen system prompt or tool schema.
- Use upstream terminology in code and docs where it improves review clarity; do not hide the underlying mechanism behind Hermes-specific branding.
- Degrade optional behavior by deterministic omission plus warning when a route is known unsupported; do not silently retry or reroute claimed-supported traffic.
- Integrate with existing logging paths and redact by default rather than inventing feature-local observability.

## 1.4 Known tradeoffs (explicit)

- Supporting `openai-codex` now adds risk because the public OpenAI docs do not document that backend contract, but the updated scope requires empirical validation instead of omission.
- A narrow shared helper improves correctness now, but it intentionally leaves broader Responses convergence for a later change.
- Config migration and agent-reuse boundary updates add scope, but skipping them would create hidden correctness debt.
- Keeping `flex` as fail-loud pass-through only avoids timeout-policy churn, but it also means phase 1 will not promise Flex-specific latency or recovery ergonomics.
- Requiring full backward compatibility, full-suite test coverage, manual exercise, and a cross-platform impact review increases scope, but not meeting that bar would lower the patch’s chance of acceptance.
- Requiring `openai-codex` support widens the verification burden, but it is still preferable to broadening into generalized non-OpenAI abstractions or undocumented proxy support.

# 2) Problem Statement (existing architecture + why change)

## 2.1 What exists today

- `hermes_cli/config.py` owns persisted config and migration, but Hermes does not yet have a normalized `model.request_options` path.
- `hermes_cli/runtime_provider.py` resolves provider, API mode, base URL, and credentials, but not normalized request-policy metadata.
- `run_agent.py` owns the main `codex_responses` request path, including preflight request shaping.
- `agent/auxiliary_client.py` owns a second Responses adapter for auxiliary calls.
- CLI, gateway, and cron construct runtime settings and agent reuse signatures from a limited subset of resolved runtime fields.

## 2.2 What’s broken / missing (concrete)

- There is no validated persisted home for `service_tier`.
- There is no end-to-end normalized request-options pipeline, so the feature could be lost between config, runtime resolution, and request execution.
- There is no single owner for Responses request-option compatibility, so a feature patch would likely duplicate or drift.
- The current implementation intentionally omits `service_tier` on `openai-codex`, which is now a direct scope violation rather than an acceptable degradation choice.
- The current planning note does not yet bind the feature to Hermes’ contribution, graceful-degradation, and logging expectations.
- The current planning note must now distinguish between required OpenAI surfaces, namely direct API-key OpenAI and `openai-codex`, versus genuinely unsupported non-OpenAI or unrelated OAuth-backed surfaces.

## 2.3 Constraints implied by the problem

- The feature must be opt-in because it can affect cost and latency.
- Compatibility must be route-aware, not just provider-name-aware.
- Compatibility must also be auth-surface-aware: direct OpenAI API-key requests and `openai-codex` are materially different surfaces, and the plan has to support both without conflating them.
- Propagation must include every runtime surface Hermes actually uses.
- The implementation has to look like convergence, not incidental feature sprawl.
- Unsupported-route behavior must degrade in the same operator-visible warning style Hermes already uses for optional capability loss.

# 3) Research Grounding (external + internal “ground truth”)

<!-- arch_skill:block:research_grounding:start -->
## 3.1 External anchors (papers, systems, prior art)

- OpenAI API authentication overview (`developers.openai.com/api/reference/overview#authentication`) — adopt the direct API auth model for the documented `api.openai.com` surface. As of 2026-04-04, the official docs ground the direct OpenAI route only; they do not document Hermes’ ChatGPT-backed `openai-codex` backend, so the plan must not pretend the public direct-API docs are sufficient evidence for Codex parity.
- OpenAI Flex processing guide (`developers.openai.com/api/docs/guides/flex-processing`) — adopt the fact that `service_tier` is a documented direct-API request field on OpenAI endpoints, including `responses` and `chat/completions`. Keep that as the direct-route source of truth while separately validating the required Codex route in Hermes’ actual runtime.
- OpenAI Priority processing guide (`developers.openai.com/api/docs/guides/priority-processing`) — adopt the upstream naming and direct-API framing. This is the source of truth for how the feature should be named in config, code, tests, docs, and the eventual PR. Reject Hermes-local euphemisms such as “fast mode” and reject any claim that ChatGPT-backed Codex routes are documented equivalents of the direct API merely because both are OpenAI-branded.
- OpenAI Chat API reference (`developers.openai.com/api/reference/resources/chat`) — adopt it as a secondary enum and compatibility reference only. It is useful for validating whether OpenAI documents additional `service_tier` values or chat-only nuances, but it does not document the ChatGPT Codex backend Hermes calls for `openai-codex`.
- `CONTRIBUTING.md` — adopt as an architectural anchor, not just repo process boilerplate. It explicitly prioritizes robustness, graceful degradation, focused PRs, and full test runs before submission, so any solution that depends on hidden retries, speculative forwarding, or partial test coverage is misaligned with the maintainers’ documented acceptance bar.

## 3.2 Internal ground truth (code as spec)

- Authoritative behavior anchors (do not reinvent):
  - `hermes_cli/config.py` — `DEFAULT_CONFIG` still seeds `"model": ""`, `_config_version` is versioned, and `migrate_config()` only fills missing keys rather than owning schema-specific nested rewrites today. This means any persisted service-tier setting is a real config-shape and migration concern, not a free additive leaf.
  - `cli.py`, `gateway/run.py`, and `cron/scheduler.py` do not start from one shared config normalization seam today. `load_cli_config()` already accepts both scalar and dict-shaped `model` config and normalizes some legacy fields, while gateway and cron still raw-load YAML and resolve adjacent settings independently. This means `model.request_options.service_tier` is not a free additive leaf even before runtime resolution begins.
  - `hermes_cli/runtime_provider.py` — the runtime owner already returns the canonical route identity (`provider`, `api_mode`, `base_url`, `api_key`, `credential_pool`, `requested_provider`) and hard-codes `openai-codex` to `api_mode="codex_responses"` on the ChatGPT Codex backend rather than `api.openai.com`. This is the canonical owner for transport/auth route facts and the right place to anchor route-aware request-options normalization, while keeping the transport return dict itself narrow.
  - `hermes_cli/runtime_provider.py`, `run_agent.py`, and `agent/auxiliary_client.py` on the current branch explicitly treat `openai-codex` as an omit path for `service_tier`. `_supports_openai_service_tier()` returns `False` for `provider == "openai-codex"`, `apply_openai_service_tier()` returns early for `provider == "openai-codex"`, and `_CodexCompletionsAdapter.create()` explicitly `pop()`s `service_tier`. That is ground truth for the current implementation, and it means the plan must reopen those seams rather than treating Codex as already covered.
  - `run_agent.py` — direct OpenAI `chat_completions` sessions are force-switched onto `codex_responses`, `_build_api_kwargs()` owns the full main-runner Responses payload shape, and local request validation is currently split between `_preflight_codex_api_kwargs()` on fallback or explicit non-stream paths and the unvalidated primary stream path. This is the current canonical direct-OpenAI execution path for the main agent.
  - `run_agent.py` preflight already allowlists `service_tier` on Codex Responses kwargs when the field is present. That means the main main-runner contract is not fundamentally incompatible with Codex support; the current blocker is the earlier omission policy, not the downstream Codex-dispatch validator.
  - `agent/auxiliary_client.py` — `resolve_provider_client()` already centralizes provider/auth/base-url/API-surface routing for auxiliary calls, but `_CodexCompletionsAdapter.create()` and `_build_call_kwargs()` define a second, smaller request-shaping truth. This is the main duplicate-truth seam.
  - `hermes_cli/auth.py` already gives Hermes a stable Codex route fingerprint via `resolve_codex_runtime_credentials()`: provider `openai-codex`, default base URL `https://chatgpt.com/backend-api/codex`, source `hermes-auth-store`, and `auth_mode: "chatgpt"`. This is the strongest existing repo-local signal for defining a Codex compatibility predicate without inventing a new transport contract.
  - `cli.py`, `gateway/run.py`, and `cron/scheduler.py` — each currently cherry-picks a fixed subset of resolved runtime fields when constructing `AIAgent` instances, turn routes, or reuse signatures. A new normalized request option can therefore be silently dropped even if `runtime_provider.py` is updated correctly.
  - `gateway/run.py` and `cron/scheduler.py` already resolve some request-ish knobs outside `runtime_provider.py` today, especially reasoning-related settings. That is evidence that Hermes may need a shared request-options seam adjacent to transport resolution rather than blindly widening the transport runtime dict.
  - `run_agent.py` also snapshots and restores primary runtime state across some flows, which is why phase 1 keeps `request_options` out of `_primary_runtime` and refreshes them per turn instead of widening snapshot state.
  - `gateway/run.py` and `run_agent.py` — the logging owners already install redacted rotating `errors.log` handlers and use warn-first degradation patterns. Any new degraded-path signaling should reuse these owners instead of inventing feature-local logging.

- Canonical path / owner to reuse:
  - `hermes_cli/runtime_provider.py` — canonical owner for transport/auth route identity that any `service_tier` compatibility decision must depend on.
  - One shared normalized request-options seam adjacent to route resolution — [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) stays transport-only, and an adjacent request-options normalizer feeds CLI, gateway, cron, ACP, API server, delegation, and `AIAgent`.
  - `agent/auxiliary_client.py` adjacent to `resolve_provider_client()` / `_CodexCompletionsAdapter` — narrowest existing shared owner for OpenAI request-shape helpers across both direct OpenAI and Codex, subject to the seam decision above. `run_agent.py` should consume helpers from this boundary rather than creating a third contract.

- Existing patterns to reuse:
  - `run_agent.py` `_is_direct_openai_url()` and `_max_tokens_param()` — existing direct-OpenAI compatibility policy, including the `max_completion_tokens` split for native OpenAI. The plan should reuse these helpers where still correct, but must stop using “direct only” as a reason to omit the required Codex path.
  - `resolve_codex_runtime_credentials()` plus the runtime-provider `openai-codex` branch — existing Codex route normalization pattern that already produces stable provider, api-mode, base-url, and auth-source facts.
  - `gateway/run.py` `_agent_config_signature()` — existing route/cache identity pattern that should remain transport-oriented for this feature while cached agents refresh `request_options` per turn.
  - Gateway and cron reasoning-config plumbing — existing precedent that request-time knobs may be normalized adjacent to transport resolution and still be threaded through runtime surfaces cleanly.
  - `run_agent.py` / `gateway/run.py` redacted `RotatingFileHandler` setup — existing observability pattern for warning-level degradation and unexpected failures.
  - `tests/test_runtime_provider_resolution.py`, `tests/test_cli_provider_resolution.py`, `tests/gateway/test_agent_cache.py`, and `tests/test_primary_runtime_restore.py` — existing preservation patterns around provider resolution, route signatures, and runtime snapshot boundaries.

- Prompt surfaces / agent contract to reuse:
  - `agent/prompt_builder.py` — system-prompt assembly is stateless and intentionally separate from runtime transport policy.
  - `agent/prompt_caching.py` — prompt caching depends on a stable system prompt across turns. This is evidence against pushing `service_tier` into prompts, dynamic instruction text, or any other prompt-layer mechanism.

- Native model or agent capabilities to lean on:
  - OpenAI’s direct Responses API already natively supports the mechanism being added. Hermes already routes direct OpenAI tool-using sessions through Responses, so no new tool, parser, wrapper, or prompt-side workaround is justified to expose `service_tier` there.
  - Hermes already has a first-class Codex runtime and auxiliary Codex adapter. Supporting `openai-codex` should therefore be done by converging the existing OpenAI request-shaping seams, not by creating a second feature path or a Codex-only side channel.
  - Hermes already has deterministic routing facts available at runtime: `resolve_runtime_provider()`, `resolve_turn_route()`, and `resolve_provider_client()` expose the exact provider, auth, base URL, and API-mode facts needed to decide whether the field may be sent.

- Existing grounding / tool / file exposure:
  - `hermes_cli/runtime_provider.py` — deterministic provider/auth/base-url/API-mode resolution already exists before `AIAgent` is created.
  - `agent/auxiliary_client.py` — deterministic provider-client resolution already exists for auxiliary calls, including raw-vs-wrapped Codex clients.
  - `run_agent.py` — the main agent already has direct access to the final runtime tuple and to the codex dispatch call sites where request validation must be enforced.
  - `hermes_cli/auth.py` — the Codex auth store already records `auth_mode: "chatgpt"` and produces the default Codex backend URL, so the repo already exposes more than just the provider label when classifying this route.

- Duplicate or drifting paths relevant to this change:
  - `hermes_cli/runtime_provider.py` `_supports_openai_service_tier()` vs `agent/auxiliary_client.py` `apply_openai_service_tier()` vs `_CodexCompletionsAdapter.create()` each currently carry independent Codex omission logic.
  - `run_agent.py` `_responses_tools()` vs auxiliary inline tool conversion in `_CodexCompletionsAdapter.create()`.
  - `run_agent.py` `_chat_messages_to_responses_input()` vs auxiliary content conversion in `_convert_content_for_responses()` / `_CodexCompletionsAdapter.create()`.
  - `run_agent.py` falls back to `DEFAULT_AGENT_IDENTITY` for missing instructions, while auxiliary hard-codes `"You are a helpful assistant."`.
  - Main direct-OpenAI sessions are forced onto Responses, while auxiliary `custom` direct OpenAI calls still stay on chat.completions plus retry-based `max_completion_tokens` adaptation.
  - CLI normalizes model config differently from gateway and cron, so a request-options feature can drift before runtime-provider logic even runs unless one shared normalization seam exists.
  - `cli.py`, `gateway/run.py`, and `cron/scheduler.py` all omit future `request_options` from runtime handoff and at least one route signature today.
  - `run_agent.py` primary-runtime snapshot and restore behavior, plus CLI and gateway cached-agent signatures, currently have no explicit request-options slot.

- Capability-first opportunities before new tooling:
  - Reuse Hermes’ existing OpenAI Responses and Codex paths instead of introducing a new mode, new command surface, or new config tree.
  - Extract pure request-shape helpers into the existing OpenAI owner boundary adjacent to `agent/auxiliary_client.py` rather than creating a generalized provider framework.
  - Reuse existing warning/error logging and redaction patterns instead of adding new telemetry plumbing for a single optional request field.

- Behavior-preservation signals already available:
  - `tests/test_model_provider_persistence.py`, `tests/hermes_cli/test_set_config_value.py`, `tests/hermes_cli/test_setup.py`, and `tests/hermes_cli/test_setup_model_provider.py` already pin the repo’s preferred regression style for nested `model`-shape preservation when provider/model flows rewrite config.
  - `tests/test_run_agent_codex_responses.py` and `tests/test_provider_parity.py` already pin the main Responses payload shape, message/tool conversion, encrypted reasoning handling, and strict preflight normalization. They also currently lock in Codex omission behavior, so those assertions are now first-class migration targets rather than preservation targets.
  - `tests/test_run_agent.py` already covers direct OpenAI `max_completion_tokens` compatibility and duplicate `errors.log` handler prevention.
  - `tests/test_runtime_provider_resolution.py`, `tests/test_cli_provider_resolution.py`, `tests/cron/test_scheduler.py`, `tests/gateway/test_agent_cache.py`, and `tests/gateway/test_reasoning_command.py` already cover core route resolution plus the repo’s existing “config changed, next message picks it up without rebuilding” pattern that a new normalized request option would touch.
  - `tests/acp/test_session.py`, `tests/acp/test_server.py`, `tests/gateway/test_api_server_toolset.py`, and `tests/tools/test_delegate.py` already cover the most important non-main-loop agent construction seams that would silently miss `request_options` if the plan stayed CLI/gateway/cron-only.
  - `tests/agent/test_auxiliary_client.py` already covers auxiliary provider resolution and max-token helper behavior, and it now explicitly pins `_CodexCompletionsAdapter.create()` omission of `service_tier`. That makes the auxiliary Codex adapter one of the clearest required assertion flips in the reopened scope.
  - `tests/test_auth_codex_provider.py` already pins the Codex auth-store contract, including default backend URL and provider identity. That is a useful preservation signal for any compatibility predicate that depends on auth-store facts.

- Likely code implications surfaced by research:
  - Keep `runtime_provider.py` transport-only and add an adjacent normalized request-options object shared by CLI, gateway, cron, ACP, API server, delegation, and `AIAgent`.
  - Thread normalized request metadata through CLI, gateway, cron, ACP, API server, delegation, and `resolve_turn_route()` so it cannot be dropped or stale-cached; for this feature, refresh it per turn on reused agents rather than widening cache signatures or `_primary_runtime`.
  - Treat `hermes_cli/auth.py`, `hermes_cli/main.py`, and `hermes_cli/setup.py` as part of the feature scope because they are real `model`-dict writers that can preserve or clobber `model.request_options.service_tier`.
  - Converge OpenAI request-shaping helpers at one shared boundary before adding `service_tier`, or the feature will immediately duplicate existing drift between direct and Codex paths.
  - Reclassify Codex support at the canonical runtime-request-options seam first, because three downstream omission points currently inherit that “unsupported” decision.
  - Prefer a Codex compatibility predicate grounded in existing Hermes runtime facts such as provider, base URL, api mode, and auth source or auth mode, rather than provider branding alone.
  - Treat CLI-vs-gateway-vs-cron config normalization drift as part of the feature scope; otherwise a persisted service-tier setting can diverge before request building begins.
  - Make ACP restore behavior explicit: preserve the persisted runtime snapshot as ACP does today, but re-normalize request options against that effective runtime plus current config rather than storing request-options state in ACP session metadata.
  - Make delegation behavior explicit: inherit `request_options` when the child inherits runtime, and re-normalize when delegation overrides provider or base URL.
  - Do not overclaim coverage for `batch_runner.py` or the gateway `BOOT.md` hook in phase 1 unless they are consciously pulled into the same normalizer path.
  - Replace the current omission-asserting Codex tests in `tests/test_runtime_provider_resolution.py`, `tests/test_run_agent_codex_responses.py`, and `tests/agent/test_auxiliary_client.py` with support-path assertions once the compatibility predicate is settled.

## 3.3 Open questions from research

- What is the narrowest correct compatibility predicate for `openai-codex`: provider name only, provider plus backend base URL, or provider plus auth-mode/runtime-source facts? Repo evidence already exposes all three classes of signal; deep-dive should choose the smallest predicate that is stable and honest.
- Does the live Codex backend accept `service_tier` consistently across the main primary stream path, the fallback path, and the auxiliary Codex adapter path, or does the plan need a narrower evidence-backed boundary inside `openai-codex` itself? Local code shows the downstream paths can carry the field, but only live validation can settle backend acceptance.
- Is the right canonical owner for Codex compatibility the runtime-request-options normalizer alone, or should the shared OpenAI helper boundary also expose an explicit Codex route helper so main and auxiliary paths cannot drift again?
- Should Hermes later surface “configured but intentionally omitted on this route” in doctor or config-validation flows in addition to runtime warning logs? This remains non-blocking because it would add a second explanatory surface and risks duplicating compatibility truth.
- If a low-cost hook exists during implementation, should unexpected OpenAI failures log upstream request IDs in existing redacted logs? This remains non-blocking and must not widen the patch into a tracing subsystem.
<!-- arch_skill:block:research_grounding:end -->

<!-- arch_skill:block:external_research:start -->
# External Research (best-in-class references; plan-adjacent)

> Goal: anchor the plan in idiomatic, broadly accepted practices where applicable. This section intentionally avoids project-specific internals.

## Topics researched (and why)
- OpenAI `service_tier` request and response semantics — this controls enum validation, request omission behavior, and whether response metadata can be treated as a request echo.
- Upstream tier-support ownership — this controls whether Hermes should hardcode model-by-model allowlists or rely on route gating plus upstream validation.
- Priority and Flex operational behavior — this controls whether phase 1 can stay plumbing-only without adding retry, timeout, or fallback complexity.
- API debugging metadata — this controls whether external best practice changes the logging stance for unexpected OpenAI failures.

## Findings + how we apply them

### OpenAI `service_tier` request and response semantics
- Best practices (synthesized):
  - Use the upstream request field name exactly as documented: `service_tier`.
  - Validate against the request enum documented on the actual endpoint Hermes will call, not against a guide-only subset.
  - Treat returned `service_tier` as the processing mode actually used, not as a guaranteed echo of the configured request value.
  - Treat omitted `service_tier` as upstream `auto` behavior, which can still resolve to a project-level default.
- Recommended default for this plan:
  - Hermes should validate the current Responses-documented request enum: `auto`, `default`, `flex`, `scale`, and `priority`.
  - Hermes verification should assert request-body include or omit behavior and any observed response handling, not equality between configured and returned `service_tier`.
- Pitfalls / footguns:
  - Assuming omission always means standard `default` processing.
  - Asserting that a returned `service_tier` must match the configured request value.
  - Rejecting `scale` locally when the current Responses request docs list it as an accepted value.
- Sources:
  - Create a model response — https://developers.openai.com/api/reference/resources/responses/methods/create — authoritative request-surface enum and default behavior for Responses.
  - Responses object — https://platform.openai.com/docs/api-reference/responses/object — authoritative confirmation that returned `service_tier` reflects actual processing mode and may differ from the requested value.
  - Priority processing — https://developers.openai.com/api/docs/guides/priority-processing — authoritative project-default and actual-tier guidance for one concrete mode.

### Upstream tier-support ownership
- Best practices (synthesized):
  - Keep local compatibility logic tied to documented route and auth surface, not to brand heuristics.
  - Keep model-tier support ownership upstream when official docs say availability lives on pricing or support pages that can change independently of the API shape.
  - Prefer fail-loud upstream errors for unsupported model-tier combinations over drift-prone local allowlists or silent field stripping.
- Recommended default for this plan:
  - Hermes should gate on the required OpenAI route facts plus the documented request field shape where docs exist, not on a hardcoded model-by-model `service_tier` support matrix.
  - Hermes should let the upstream API reject unsupported model-tier combinations through the normal error path.
- Pitfalls / footguns:
  - Encoding pricing-page support tables into Hermes.
  - Equating any OpenAI-branded backend with the documented direct OpenAI API surface.
  - Treating the lack of public Codex backend docs as a reason to keep `openai-codex` permanently unsupported after the plan has made it required scope.
- Sources:
  - Priority processing — https://developers.openai.com/api/docs/guides/priority-processing — authoritative note that supported models live on pricing and can vary.
  - Flex processing — https://developers.openai.com/api/docs/guides/flex-processing — authoritative note that Flex is beta with limited model availability listed on pricing.
  - API Overview — https://developers.openai.com/api/reference/overview#authentication — authoritative auth-surface definition for the direct OpenAI API.

### Priority and Flex operational behavior
- Best practices (synthesized):
  - Priority is request-level and project-level configurable, uses usual retry and backoff logic, and can still be downgraded to `default` under ramp-rate limits.
  - Flex is slower, may require longer timeouts, automatically retries `408 Request Timeout` in official SDKs, and can produce `429 Resource Unavailable`.
  - Retrying Flex with standard processing is explicitly an application-level choice, not a mandatory API contract.
- Recommended default for this plan:
  - Phase 1 should not add Hermes-local timeout policy, retry policy, or automatic fallback behavior just to make a service tier appear more reliable.
  - Hermes should preserve fail-loud behavior and never silently retry a failed Flex request with `auto` or omitted `service_tier`.
  - Deep-dive should explicitly decide whether existing Hermes timeout surfaces make initial Flex support honest enough without widening scope.
- Pitfalls / footguns:
  - Silent retry from `flex` to standard processing.
  - Assuming Priority guarantees the returned `service_tier` remains `priority`.
  - Claiming full Flex support without checking Hermes’ current timeout and error surfaces first.
- Sources:
  - Priority processing — https://developers.openai.com/api/docs/guides/priority-processing — authoritative guidance on project defaults, usual retry and backoff, and ramp-rate downgrade behavior.
  - Flex processing — https://developers.openai.com/api/docs/guides/flex-processing — authoritative guidance on longer timeouts, automatic SDK retries for `408`, and explicit retry strategies for `429 Resource Unavailable`.

### API debugging metadata
- Best practices (synthesized):
  - Log server-generated request IDs in production when available.
  - Prefer low-cost reuse of existing client metadata over inventing a separate tracing mechanism.
- Recommended default for this plan:
  - If deep-dive finds an easy existing hook in Hermes’ direct OpenAI failure paths, request IDs are worth including in unexpected error logs.
  - Do not widen this patch into a new tracing or request-correlation subsystem.
- Pitfalls / footguns:
  - Adding client-generated request-ID plumbing or a new telemetry surface in a feature that is otherwise request-field plumbing.
  - Requiring new logging machinery just to comply with an external best practice that may not fit the current code path cheaply.
- Sources:
  - API Overview — https://developers.openai.com/api/reference/overview#authentication — authoritative debugging guidance for `x-request-id` logging and request troubleshooting.

## Adopt / Reject summary
- Adopt:
  - Validate the current Responses-documented `service_tier` request enum: `auto`, `default`, `flex`, `scale`, `priority`.
  - Keep request validation and compatibility gating route-aware and auth-surface-aware.
  - Keep verification focused on request construction and actual degraded or failed behavior, not on assuming returned `service_tier` echoes the configured request.
  - Reject local model-tier allowlists and let upstream own model support.
  - Keep fail-loud semantics for Flex and never add silent retry-to-standard behavior.
- Reject:
  - Hermes-local aliases such as “fast mode”.
  - Model-support matrices copied from pricing or support docs into Hermes.
  - Any assumption that omitted `service_tier` means standard processing in all cases.
  - Any assumption that returned `service_tier` must equal the request value.
  - Any new tracing or telemetry subsystem in phase 1 just to capture request IDs.

## Open questions (ONLY if truly not answerable)
- Public OpenAI docs do not currently answer the ChatGPT Codex backend contract Hermes uses for `openai-codex`, so Codex-specific compatibility still requires Hermes runtime evidence and live validation. The remaining questions are tracked in Section 3.3.
<!-- arch_skill:block:external_research:end -->

<!-- arch_skill:block:current_architecture:start -->
# 4) Current Architecture (as-is)

## 4.1 On-disk structure

- Config persistence is still split across [hermes_cli/config.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/config.py), provider/model writers in [hermes_cli/auth.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/auth.py) and [hermes_cli/main.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/main.py), setup-wizard sync logic in [hermes_cli/setup.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/setup.py), CLI normalization in [cli.py](/Users/aelaguiz/workspace/hermes-agent/cli.py), raw gateway YAML loading in [gateway/run.py](/Users/aelaguiz/workspace/hermes-agent/gateway/run.py), and raw cron YAML loading in [cron/scheduler.py](/Users/aelaguiz/workspace/hermes-agent/cron/scheduler.py). The branch now has a canonical request-options normalizer, but config mutation is still distributed across those writers.
- Transport and auth route resolution is centralized in [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py), which returns route identity fields such as `provider`, `api_mode`, `base_url`, `api_key`, `credential_pool`, and ACP process metadata. Request-options normalization now lives adjacent to that transport resolver in the same module.
- Per-turn model routing is centralized in [agent/smart_model_routing.py](/Users/aelaguiz/workspace/hermes-agent/agent/smart_model_routing.py), which now carries `request_options` in addition to model/runtime/signature/label.
- The main direct-OpenAI execution path lives in [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py): `AIAgent.__init__`, `_build_api_kwargs()`, `_preflight_codex_api_kwargs()`, `_run_codex_stream()`, and fallback restore logic.
- The auxiliary OpenAI path lives in [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py), which still uses chat completions for direct `api.openai.com` custom routes and a smaller Responses shim for `openai-codex`. The current branch explicitly strips `service_tier` from that Codex shim.
- Long-lived agent reuse exists in CLI and the gateway; cron, ACP, API server, and delegated children construct fresh agents on their own paths.
- Codex auth and route identity are already normalized in [hermes_cli/auth.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/auth.py) through `resolve_codex_runtime_credentials()`, which returns `provider="openai-codex"`, `auth_mode="chatgpt"`, and the default backend `https://chatgpt.com/backend-api/codex`.

## 4.2 Control paths (runtime)

- CLI turn flow:
  - [cli.py](/Users/aelaguiz/workspace/hermes-agent/cli.py) `_ensure_runtime_credentials()` resolves transport/auth every turn through [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py).
  - `_resolve_turn_agent_config()` builds a primary transport dict, resolves `primary_request_options`, and hands both to [agent/smart_model_routing.py](/Users/aelaguiz/workspace/hermes-agent/agent/smart_model_routing.py).
  - The main CLI run loop compares the returned signature with `_active_agent_route_signature`, optionally reuses the same `AIAgent`, and refreshes `agent.request_options` per turn outside the cache signature.
- Gateway turn flow:
  - [gateway/run.py](/Users/aelaguiz/workspace/hermes-agent/gateway/run.py) resolves runtime once per message, computes `primary_request_options`, routes them through `resolve_turn_route()`, computes a cached-agent signature from model plus transport fields, and mutates `agent.request_options` per message on cache hits.
  - Temporary gateway agents for background jobs, `/btw`, and maintenance paths now construct fresh `AIAgent` instances with explicit `request_options` seeds from the same normalizer.
- ACP flow:
  - [acp_adapter/session.py](/Users/aelaguiz/workspace/hermes-agent/acp_adapter/session.py) creates and restores agents through its own `_make_agent()` factory, using persisted provider/base-url/api-mode session metadata plus runtime resolution and adjacent request-options normalization.
  - [acp_adapter/server.py](/Users/aelaguiz/workspace/hermes-agent/acp_adapter/server.py) rebuilds ACP session agents on `/model` and protocol model-switch requests through that same factory.
- API server flow:
  - [gateway/platforms/api_server.py](/Users/aelaguiz/workspace/hermes-agent/gateway/platforms/api_server.py) constructs fresh agents through `_create_agent()` using gateway runtime resolution plus adjacent request-options normalization, but outside the main cached-agent turn-route path.
- Cron flow:
  - [cron/scheduler.py](/Users/aelaguiz/workspace/hermes-agent/cron/scheduler.py) loads YAML directly, resolves runtime, applies request-options normalization plus smart routing, and constructs a fresh `AIAgent` once per job.
- Delegation flow:
  - [tools/delegate_tool.py](/Users/aelaguiz/workspace/hermes-agent/tools/delegate_tool.py) constructs child agents either by inheriting parent runtime plus `request_options` or by re-resolving delegation overrides through the same adjacent normalizer.
- Main direct OpenAI request flow:
  - [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) coerces direct `api.openai.com` traffic onto the Responses path, builds a Responses payload, sends the primary stream path directly through `responses.stream()`, and only runs `_preflight_codex_api_kwargs()` on the `create(stream=True)` fallback and explicit non-stream paths today.
- Main Codex request flow:
  - [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) uses the same Responses transport family for `openai-codex`, but the current branch omits `service_tier` before the request reaches Codex dispatch because `apply_openai_service_tier()` classifies Codex unsupported upstream of preflight.
- Auxiliary OpenAI request flow:
  - [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) resolves `provider="custom"` to a direct OpenAI-compatible client, applies task-specific timeouts, and issues `chat.completions.create()` with only the `max_tokens -> max_completion_tokens` retry.
- Auxiliary Codex request flow:
  - [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) routes `openai-codex` through `_CodexCompletionsAdapter.create()`, which currently strips `service_tier` explicitly even though the downstream Codex request contract can carry it.
- Explicit phase-1 exclusions:
  - [batch_runner.py](/Users/aelaguiz/workspace/hermes-agent/batch_runner.py) and [gateway/builtin_hooks/boot_md.py](/Users/aelaguiz/workspace/hermes-agent/gateway/builtin_hooks/boot_md.py) do construct `AIAgent` directly, but they are not part of the normal config-plus-runtime-resolution story this plan is optimizing for and do not currently have representative propagation tests. They must not be described as covered unless implementation deliberately pulls them in.

## 4.3 Object model + key abstractions

- Persisted config shape:
  - [hermes_cli/config.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/config.py) still defaults `model` to a scalar empty string and deep-merges user config over that shape. CLI separately seeds `model` as a dict in [cli.py](/Users/aelaguiz/workspace/hermes-agent/cli.py). That means any nested `model.request_options.service_tier` field is a real migration and normalization concern.
- Transport runtime dict:
  - [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) returns a transport/auth contract that downstream code treats as route identity. It stays free of request-time metadata such as `service_tier`, and it does not currently thread `auth_mode` through the runtime contract.
- Request-options object:
  - [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) now produces a dedicated request-options dict adjacent to transport resolution, but its support classifier currently returns empty for all `openai-codex` routes.
- Turn route dict:
  - [agent/smart_model_routing.py](/Users/aelaguiz/workspace/hermes-agent/agent/smart_model_routing.py) now returns `model`, `runtime`, `request_options`, `label`, and `signature`.
- Long-lived mutable agent state:
  - `AIAgent` already carries per-turn mutable state such as `reasoning_config`, callbacks, streaming hooks, and now `request_options` without rebuilding the frozen system prompt or tool schema.
- Fallback snapshot:
  - [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) snapshots `_primary_runtime` for transport recovery and fallback restore. That snapshot currently captures transport/client/compressor state, not request-time options.

## 4.4 Observability + failure behavior today

- Route-resolution failures are surfaced through `format_runtime_provider_error()` in [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) and shown by CLI/gateway/cron without feature-specific logging paths.
- Main Responses requests still have only partial local fail-loud validation today: `_preflight_codex_api_kwargs()` in [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) rejects unexpected fields on the fallback and explicit non-stream paths, but the primary `responses.stream()` path currently uses built kwargs directly.
- Request-options degradation logging now exists at the runtime normalizer boundary in [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py), but the current classifier warns that `openai-codex` is unsupported rather than treating Codex as required scope.
- Auxiliary direct-OpenAI requests fail upstream or through the minimal `max_tokens` compatibility retry in [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py). The auxiliary Codex adapter does not currently have a support-path for `service_tier`; it strips the field before network I/O.
- Redacted error logging already exists in [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) and [gateway/run.py](/Users/aelaguiz/workspace/hermes-agent/gateway/run.py).
- Main direct-OpenAI Responses requests do not set an explicit OpenAI client timeout today; auxiliary tasks do carry task-specific timeouts.
- Codex auth metadata such as `auth_mode="chatgpt"` exists in the Hermes auth store, but it is not part of the runtime contract that downstream request shaping sees today.
- Current Flex-relevant behavior is still generic:
  - surfaced `429` follows existing rate-limit logic
  - surfaced `408` is not a Hermes-owned Responses retry contract
  - there is no Flex-specific timeout widening or fallback policy

## 4.5 UI surfaces (ASCII mockups, if UI work)

No new UI surface is expected. This remains a config-driven runtime feature with existing CLI, gateway, and cron entry points.
<!-- arch_skill:block:current_architecture:end -->

<!-- arch_skill:block:target_architecture:start -->
# 5) Target Architecture (to-be)

## 5.1 On-disk structure (future)

- [hermes_cli/config.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/config.py) remains the persisted config owner and grows the migration-safe `model.request_options.service_tier` leaf.
- [hermes_cli/auth.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/auth.py), [hermes_cli/main.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/main.py), and [hermes_cli/setup.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/setup.py) remain real model/provider config writers and must preserve the new nested leaf when they rewrite `config["model"]`.
- [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) stays transport/auth-only for `resolve_runtime_provider()`, but gains one adjacent request-options normalizer that reuses `_get_model_config()` and resolved route facts instead of widening the transport return dict.
- [agent/smart_model_routing.py](/Users/aelaguiz/workspace/hermes-agent/agent/smart_model_routing.py) becomes the shared turn-route carrier for `request_options` in addition to model/runtime/signature/label.
- [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) remains the owner of full Responses request assembly, strict preflight, streaming, and fallback orchestration.
- [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) becomes the shared import boundary for pure OpenAI request-shape helpers only: direct-route detection, Codex-route detection, max-token parameter choice, and OpenAI optional request-option injection. It does not become the owner of the main loop’s full Responses payload builder.
- [cli.py](/Users/aelaguiz/workspace/hermes-agent/cli.py), [gateway/run.py](/Users/aelaguiz/workspace/hermes-agent/gateway/run.py), [cron/scheduler.py](/Users/aelaguiz/workspace/hermes-agent/cron/scheduler.py), [acp_adapter/session.py](/Users/aelaguiz/workspace/hermes-agent/acp_adapter/session.py), [gateway/platforms/api_server.py](/Users/aelaguiz/workspace/hermes-agent/gateway/platforms/api_server.py), and [tools/delegate_tool.py](/Users/aelaguiz/workspace/hermes-agent/tools/delegate_tool.py) thread normalized `request_options` end to end without adding non-OpenAI behavior.

## 5.2 Control paths (future)

- Config and route resolution:
  - CLI, gateway, and cron continue to resolve transport/auth through `resolve_runtime_provider()`.
  - Each surface then calls the adjacent runtime-provider request-options normalizer to obtain `primary_request_options` from persisted config plus the resolved primary route facts.
  - The runtime-provider request-options normalizer becomes the canonical compatibility owner for `service_tier`. For direct OpenAI it keys off `api.openai.com` plus a real API key. For Codex it keys off the effective Hermes Codex route facts: `provider=="openai-codex"`, `api_mode=="codex_responses"`, and a normalized base URL on the known `chatgpt.com/backend-api/codex` backend. No `auth_mode` field is added to the runtime contract just for this feature.
  - Helper and maintenance flows that bypass turn routing, especially in the gateway, apply that same normalizer directly against their resolved runtime before constructing temporary agents.
- Turn routing:
  - [agent/smart_model_routing.py](/Users/aelaguiz/workspace/hermes-agent/agent/smart_model_routing.py) remains the owner of the final effective `request_options` on turn-routed paths: callers pass both primary runtime and `primary_request_options`, the no-route branch returns that precomputed primary value, and the smart-routed branch re-resolves request options against the effective smart-routed runtime before returning `turn_route`.
  - Turn routing does not grow a second Codex compatibility rule; it consumes the canonical normalizer output unchanged.
- Agent lifecycle:
  - CLI and gateway treat `request_options` like `reasoning_config`: per-turn mutable state applied to a possibly reused `AIAgent`.
  - Cron can seed `request_options` when constructing a fresh agent because there is no cache-reuse surface there.
  - ACP creates and restores fresh agents through its session factory, preserving the persisted transport snapshot but re-normalizing request options from current config plus the effective restored runtime rather than storing request-options state in session metadata.
  - The API server seeds request options at `_create_agent()` time from the same runtime facts it already resolves for fresh agents.
  - Delegated children inherit the parent `request_options` when they inherit the parent runtime, and they re-normalize request options when delegation overrides provider or base URL for the child.
  - Internal or helper-spawned agents that do not have a turn-route wrapper, including review and gateway maintenance agents, should inherit or seed explicit `request_options` rather than silently falling back to an implicit default.
  - Because `request_options` do not change the frozen system prompt or tool schema, they stay out of cache signatures and out of `_primary_runtime` snapshots unless a future option would make that unsafe.
- Main direct OpenAI path:
  - `AIAgent._build_api_kwargs()` calls the shared pure helper(s) from [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) to inject `service_tier` only after the canonical normalizer has already decided that the route is supported.
  - One shared validation step must guard every main Responses dispatch path before network I/O, not just fallback paths. The primary `responses.stream()` path, the `create(stream=True)` fallback, and explicit non-stream paths must all pass through the same allowlist-aware validation for `service_tier`.
  - The helper injects `service_tier` when the effective route is either a supported direct API-key `api.openai.com` route or the supported `openai-codex` route. Genuinely unsupported routes deterministically omit it and emit the feature’s warn-once omission warning path.
- Auxiliary OpenAI path:
  - `_build_call_kwargs()` in [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) uses the same helper to inject `service_tier` for direct OpenAI chat-completions calls on `provider="custom"` plus `api.openai.com`.
  - `_CodexCompletionsAdapter.create()` must stop being a permanent omit path and instead participate in the same canonical include-or-omit decision for the required `openai-codex` route.
  - The shared helper boundary no longer owns independent support classification for Codex. It either consumes pre-normalized `request_options` or calls one runtime-provider-owned predicate, so Codex parity cannot drift between normalizer, main loop, and auxiliary adapter.
- Flex behavior:
  - `flex` can be forwarded as a documented value on compatible direct OpenAI routes, but phase 1 deliberately preserves current timeout and retry behavior. No Flex-specific timeout widening, retry shim, or fallback-to-default behavior is introduced.

## 5.3 Object model + abstractions (future)

- Persisted config:
  - `model.request_options.service_tier: Optional[str]`
- Transport runtime:
  - unchanged `resolve_runtime_provider()` dict
  - remains route identity only
- Request-options object:
  - a dedicated normalized dict produced adjacent to runtime resolution
  - carries only request-time metadata, starting with `service_tier`
  - does not own model-support allowlists
- Turn route:
  - `{"model", "runtime", "request_options", "label", "signature"}`
  - signature remains transport/model/tool/prompt oriented
- `AIAgent`:
  - gains `request_options` as mutable per-turn state, optionally seedable at construction for fresh-agent callers
  - does not treat `request_options` as part of prompt caching identity
- Shared helper contract:
  - pure OpenAI helper(s) in [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) own:
    - direct `api.openai.com` route detection
    - token-parameter key selection
    - payload injection from already-normalized `request_options`
  - [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) owns the canonical route-support predicate for `service_tier`, including the Codex-specific backend match.
  - `run_agent.py` continues to own full Responses assembly and preflight around those helpers.

## 5.4 Invariants and boundaries

- [hermes_cli/config.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/config.py) remains the persisted config owner.
- [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) remains the transport/auth owner; request-options normalization is adjacent to it, not embedded inside the transport return dict.
- `request_options` are per-turn mutable state, not transport identity, for this feature. They must be refreshed on every turn where an agent may be reused.
- [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) owns only pure OpenAI request-shape helpers; [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) keeps ownership of the main Responses request contract.
- Compatibility gating keys off effective route facts, not OpenAI branding alone. Direct OpenAI API-key routes and `openai-codex` are required support surfaces; OpenRouter, Copilot, GitHub Models, third-party proxies, and local/custom non-OpenAI endpoints omit unless the plan is later expanded.
- The supported Codex route is defined by provider plus runtime facts, not provider label alone: `provider=="openai-codex"`, `api_mode=="codex_responses"`, and a normalized base URL on the known `chatgpt.com/backend-api/codex` backend.
- The runtime contract does not grow `auth_mode` just to support this feature. Auth-store facts may justify the design, but request-time classification stays anchored to the existing runtime surface.
- Hermes does not own a model-tier support matrix. Unsupported model-tier combinations fail through the normal upstream API path.
- `flex` is forwarded only as best-effort pass-through with existing Hermes timeout/error behavior preserved. Phase 1 does not add Flex-specific ergonomics.
- [batch_runner.py](/Users/aelaguiz/workspace/hermes-agent/batch_runner.py) and [gateway/builtin_hooks/boot_md.py](/Users/aelaguiz/workspace/hermes-agent/gateway/builtin_hooks/boot_md.py) remain explicit phase-1 exclusions unless implementation can pull them onto the same normalizer path without creating a second architecture or a test debt cliff.
- No non-OpenAI provider gains new behavior.

## 5.5 UI surfaces (ASCII mockups, if UI work)

No new UI is planned. User interaction remains config-only, and any future docs or PR language use the upstream OpenAI term `service_tier`.
<!-- arch_skill:block:target_architecture:end -->

<!-- arch_skill:block:call_site_audit:start -->
# 6) Call-Site Audit (exhaustive change inventory)

## 6.1 Change map (table)

Reopened-scope note:
- The generic `request_options` plumbing across smart routing, CLI, gateway, cron, ACP, API server, delegation, and cached-agent refresh is already landed on this branch.
- The remaining implementation delta is Codex classification at the canonical normalizer boundary, Codex payload application in the main and auxiliary OpenAI paths, and the representative test assertions that currently lock Codex omission in place.

| Area | File | Symbol / Call site | Current behavior | Required change | Why | New API / contract | Tests impacted |
| ---- | ---- | ------------------ | ---------------- | --------------- | --- | ------------------ | -------------- |
| Config persistence | [hermes_cli/config.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/config.py) | `DEFAULT_CONFIG`, `load_config()`, migration helpers | `model` still defaults to scalar `""`; no nested request-options leaf | Add migration-safe `model.request_options.service_tier` normalization without breaking scalar or mixed shapes | Persist the feature 100% backward-compatibly | Optional `model.request_options.service_tier` | `tests/hermes_cli/test_config.py`, `tests/test_model_provider_persistence.py`, `tests/hermes_cli/test_set_config_value.py` |
| Model/provider config writers | [hermes_cli/auth.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/auth.py), [hermes_cli/main.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/main.py), [hermes_cli/setup.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/setup.py) | `_save_model_choice()`, `_update_config_for_provider()`, `_reset_config_provider()`, provider flows, `setup_model_provider()` disk-sync path | Multiple real writers rebuild or overwrite `config["model"]`; some bypass `save_config()` normalization entirely | Preserve `model.request_options.service_tier` whenever provider/model/base-url updates rewrite `config["model"]` | Avoid clobbering the new nested leaf in the repo’s most common config mutation paths | Existing config writers become leaf-preserving rather than leaf-dropping | `tests/test_model_provider_persistence.py`, `tests/hermes_cli/test_set_config_value.py`, `tests/hermes_cli/test_setup.py`, `tests/hermes_cli/test_setup_model_provider.py` |
| Request-options normalization | [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) | `_supports_openai_service_tier()`, `resolve_runtime_request_options()`, Codex runtime branches | Adjacent request-options normalizer already exists and supports direct `api.openai.com`, but hard-codes `openai-codex` unsupported and warns once | Replace the blanket Codex exclusion with a route-fact-based Codex predicate: `provider=="openai-codex"`, `api_mode=="codex_responses"`, and normalized base URL on the known Codex backend; keep the transport dict shape unchanged | Keep one compatibility classifier and avoid speculative forwarding to arbitrary endpoints | Same `resolve_runtime_request_options(...)->dict` contract, but with explicit Codex support on the known backend only | `tests/test_runtime_provider_resolution.py`, `tests/test_cli_provider_resolution.py`, `tests/test_auth_codex_provider.py` |
| Smart turn routing | [agent/smart_model_routing.py](/Users/aelaguiz/workspace/hermes-agent/agent/smart_model_routing.py) | `resolve_turn_route()` | Already accepts precomputed `primary_request_options`, returns them on the no-route branch, and re-resolves them on smart routes | No structural redesign; preserve this path and let the updated normalizer drive Codex parity | Prevent a second Codex compatibility rule from appearing in routing | Existing `turn_route["request_options"]` contract remains authoritative | `tests/agent/test_smart_model_routing.py`, `tests/test_credential_pool_routing.py` |
| CLI long-lived agent | [cli.py](/Users/aelaguiz/workspace/hermes-agent/cli.py) | `_resolve_turn_agent_config()`, main CLI run loop, `_init_agent()` | Already resolves `primary_request_options`, passes them through `turn_route`, and refreshes `agent.request_options` every turn | No new architecture here; preserve the landed mutable-state pattern so Codex support falls out from the normalizer change | Codex support should not require a CLI-specific branch | Existing per-turn mutable `agent.request_options` contract remains | `tests/test_cli_provider_resolution.py` plus one focused constructor-capture test only if a temp-agent path changes |
| CLI temporary agents | [cli.py](/Users/aelaguiz/workspace/hermes-agent/cli.py) | background and `/btw` AIAgent construction sites | Background and `/btw` agents already use `turn_route["request_options"]` at construction time | No structural redesign; preserve the existing constructor seed so Codex parity falls out from the normalizer change | Keep secondary CLI flows consistent without treating TUI tests as propagation tests | Existing constructor-time `request_options` seed remains | `tests/test_cli_provider_resolution.py` and, if needed, one owner-level constructor-capture test; explicitly not `tests/test_cli_background_tui_refresh.py` |
| Gateway cached agent | [gateway/run.py](/Users/aelaguiz/workspace/hermes-agent/gateway/run.py) | `_resolve_turn_agent_config()`, `_agent_config_signature()`, `_run_agent()` | Cached agents already refresh `request_options` and `reasoning_config` per message; signature remains transport-oriented | Preserve the transport-only signature and the per-message mutable-state refresh so Codex parity comes only from the normalizer | Avoid stale state without needless cache churn | Request-options remain mutable, not signature-owned, for this feature | `tests/gateway/test_agent_cache.py`, `tests/gateway/test_reasoning_command.py` |
| Gateway helper runtime resolution | [gateway/run.py](/Users/aelaguiz/workspace/hermes-agent/gateway/run.py) | `_resolve_runtime_agent_kwargs()` and helper-agent callers | Helper flows that bypass `turn_route` already run the adjacent request-options normalizer against resolved runtime facts | No structural redesign; preserve helper-path use of the same normalizer so Codex support does not fork into gateway-only logic | Keep maintenance/helper agents from silently drifting away from the canonical seam | Existing helper-runtime request-options seed remains | `tests/gateway/test_flush_memory_stale_guard.py`, `tests/gateway/test_session_hygiene.py` |
| Gateway temporary agents | [gateway/run.py](/Users/aelaguiz/workspace/hermes-agent/gateway/run.py) | flush agent, hygiene/compression agent, transcript-compression agent, sync/background AIAgent construction sites | Temporary and maintenance agents already seed `request_options`, including non-turn-routed helper paths | No structural redesign; preserve those seeds and let the Codex normalizer change propagate through them | Keep non-cached gateway paths aligned | Existing constructor-time `request_options` seed remains | `tests/gateway/test_background_command.py`, `tests/gateway/test_flush_memory_stale_guard.py`, `tests/gateway/test_session_hygiene.py`, `tests/test_flush_memories_codex.py` |
| API server fresh agents | [gateway/platforms/api_server.py](/Users/aelaguiz/workspace/hermes-agent/gateway/platforms/api_server.py) | `_create_agent()` | Fresh API-server agents already receive normalized `request_options` from resolved runtime facts | No structural redesign; preserve the canonical gateway helper path | Keep the API server aligned with the rest of the gateway stack without inventing a second normalizer path | Existing fresh-agent request-options seed remains | `tests/gateway/test_api_server_toolset.py` |
| ACP session factory and restore | [acp_adapter/session.py](/Users/aelaguiz/workspace/hermes-agent/acp_adapter/session.py) | `_make_agent()`, session restore path | ACP already re-normalizes `request_options` from the effective runtime snapshot instead of persisting request-time state in session metadata | No structural redesign; preserve config-plus-restored-runtime recomputation and let Codex parity come from the normalizer | Preserve ACP’s existing runtime-snapshot semantics while keeping request policy current and deterministic | Fresh ACP agents receive `request_options`; restored ACP sessions recompute it against restored runtime | `tests/acp/test_session.py` |
| ACP model switching | [acp_adapter/server.py](/Users/aelaguiz/workspace/hermes-agent/acp_adapter/server.py) | `/model` and `set_session_model()` rebuilds | ACP model switches already rebuild agents through the ACP factory path that re-normalizes `request_options` | No structural redesign; preserve the factory-owned seed path | Keep ACP model-switch behavior aligned with the feature’s config-plus-runtime policy | ACP rebuilds continue to pass through the same request-options seed path | `tests/acp/test_server.py` |
| Cron jobs | [cron/scheduler.py](/Users/aelaguiz/workspace/hermes-agent/cron/scheduler.py) | `run_job()` routing + AIAgent init | Cron already resolves `primary_request_options`, routes them, and seeds `request_options` on fresh job agents | No structural redesign; preserve the existing route-and-seed path so Codex support rides the normalizer | End-to-end parity for scheduled jobs without a cron-only branch | Existing fresh-agent `request_options` seed remains | `tests/cron/test_scheduler.py` |
| Delegated subagents | [tools/delegate_tool.py](/Users/aelaguiz/workspace/hermes-agent/tools/delegate_tool.py) | child-agent builder and delegation runtime resolution | Delegation already inherits parent `request_options` on identical runtime and re-normalizes them when child runtime facts differ | No structural redesign; preserve this inheritance-vs-recompute split and let the canonical normalizer decide Codex support | End-to-end means delegated work must not silently fork compatibility logic | Child `request_options` continue to inherit or re-normalize based on effective child runtime | `tests/tools/test_delegate.py` |
| Agent state surface | [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) | `AIAgent.__init__` | `AIAgent` already accepts a `request_options` seed and stores mutable `self.request_options` | No structural redesign; preserve this single in-memory source of truth for the active turn | Prevent a second request-policy state path from appearing inside the agent | Existing `AIAgent.request_options` contract remains | Main agent tests |
| Nested review agent | [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) | `_run_review()` helper `AIAgent` construction | The internal review helper already inherits `self.request_options` when it spawns a nested agent | No structural redesign; preserve that inheritance so internal direct-OpenAI and Codex calls stay aligned with the active turn | Keep internal nested-agent behavior aligned with the main agent policy | Existing review-agent request-options inheritance remains | `tests/test_run_agent.py` |
| Main Responses payload | [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) | `_build_api_kwargs()` | Builds Responses kwargs and currently calls a helper that refuses Codex before preflight sees the field | Stop treating the helper as an independent Codex classifier; inject `service_tier` from the canonical normalizer output so the main Codex path can actually reach dispatch | Main end-to-end feature path | Shared helper becomes payload injector, not a second support matrix | `tests/test_run_agent_codex_responses.py`, `tests/test_provider_parity.py` |
| Main Responses validation | [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) | `_run_codex_stream()`, `_run_codex_create_stream_fallback()`, explicit non-stream codex path, `_preflight_codex_api_kwargs()` | Primary stream path sends built kwargs directly; local allowlist validation only guards fallback and explicit non-stream paths today | Move to one shared allowlist-aware validation point that covers primary stream, fallback, and explicit non-stream codex dispatches, and allowlist `service_tier` there | Fail-loud local contract must match every dispatch path, not just fallback paths | One validated codex-dispatch kwargs contract | `tests/test_run_agent_codex_responses.py`, `tests/test_run_agent.py` |
| Fallback restore boundary | [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) | `_primary_runtime`, `_restore_primary_runtime()` | Snapshots transport/client/compressor state only | Intentionally do not add request-options to snapshot; document and test that request-options are per-turn and route-gated at build time | Avoid false coupling between transport recovery and request-time policy | No snapshot ownership for `request_options` in phase 1 | `tests/test_primary_runtime_restore.py` |
| Shared OpenAI helper boundary | [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) | `apply_openai_service_tier()`, `_CodexCompletionsAdapter`, `_build_call_kwargs()` | Route detection, token-key choice, and optional-field logic are duplicated or inline today, and the helper independently strips Codex | Narrow the helper boundary so it owns payload injection and token-key choice, but not an independent Codex support decision; if it needs a predicate, import the runtime-provider-owned one rather than re-encoding it | Remove duplicate truth without moving the full Responses builder | Pure helper(s) reused by main and auxiliary paths with runtime-provider-owned compatibility | `tests/test_run_agent.py`, `tests/agent/test_auxiliary_client.py`, `tests/test_provider_parity.py` |
| Auxiliary direct OpenAI path | [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) | `_build_call_kwargs()`, `call_llm()`, `call_llm_async()` | Direct OpenAI custom path uses chat completions with timeout plus max-token compatibility retry | Thread request-options into `_build_call_kwargs()` and inject `service_tier` for supported direct OpenAI routes while keeping the same canonical helper contract used by Codex | Maximize end-to-end compatibility with the existing auxiliary stack | Auxiliary payload builder accepts request-options | Net-new `_build_call_kwargs()` tests in `tests/agent/test_auxiliary_client.py` |
| Auxiliary Codex shim | [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) | `_CodexCompletionsAdapter.create()` | OAuth-backed Codex Responses shim; current branch strips `service_tier` explicitly even when upstream request options say the route is supported | Move the adapter onto the same canonical include-or-omit decision as the main Codex path and stop treating omission as the designed outcome | Keep `openai-codex` inside the required OpenAI support surface without inventing a parallel adapter policy | Codex adapter receives canonical `request_options` handling | Net-new adapter payload-shape tests in `tests/agent/test_auxiliary_client.py` |
| Explicit phase-1 exclusions | [batch_runner.py](/Users/aelaguiz/workspace/hermes-agent/batch_runner.py), [gateway/builtin_hooks/boot_md.py](/Users/aelaguiz/workspace/hermes-agent/gateway/builtin_hooks/boot_md.py) | direct `AIAgent` construction | Both construct agents directly, but neither currently participates in the normal runtime-resolution/request-options story or has a representative propagation test seam | Do not describe them as covered unless implementation consciously pulls them onto the same normalizer path with proportionate tests | Keep the plan honest and prevent low-signal test sprawl | Excluded from phase-1 acceptance by default | none unless later pulled in |
| Preservation tests | [tests/test_model_provider_persistence.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_model_provider_persistence.py), [tests/hermes_cli/test_set_config_value.py](/Users/aelaguiz/workspace/hermes-agent/tests/hermes_cli/test_set_config_value.py), [tests/hermes_cli/test_setup.py](/Users/aelaguiz/workspace/hermes-agent/tests/hermes_cli/test_setup.py), [tests/hermes_cli/test_setup_model_provider.py](/Users/aelaguiz/workspace/hermes-agent/tests/hermes_cli/test_setup_model_provider.py), [tests/test_run_agent_codex_responses.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_run_agent_codex_responses.py), [tests/test_run_agent.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_run_agent.py), [tests/test_provider_parity.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_provider_parity.py), [tests/agent/test_auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/tests/agent/test_auxiliary_client.py), [tests/agent/test_smart_model_routing.py](/Users/aelaguiz/workspace/hermes-agent/tests/agent/test_smart_model_routing.py), [tests/test_credential_pool_routing.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_credential_pool_routing.py), [tests/gateway/test_agent_cache.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_agent_cache.py), [tests/gateway/test_reasoning_command.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_reasoning_command.py), [tests/gateway/test_flush_memory_stale_guard.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_flush_memory_stale_guard.py), [tests/gateway/test_session_hygiene.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_session_hygiene.py), [tests/gateway/test_background_command.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_background_command.py), [tests/gateway/test_api_server_toolset.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_api_server_toolset.py), [tests/acp/test_session.py](/Users/aelaguiz/workspace/hermes-agent/tests/acp/test_session.py), [tests/acp/test_server.py](/Users/aelaguiz/workspace/hermes-agent/tests/acp/test_server.py), [tests/tools/test_delegate.py](/Users/aelaguiz/workspace/hermes-agent/tests/tools/test_delegate.py), [tests/test_primary_runtime_restore.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_primary_runtime_restore.py), [tests/test_runtime_provider_resolution.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_runtime_provider_resolution.py), [tests/test_cli_provider_resolution.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_cli_provider_resolution.py), [tests/cron/test_scheduler.py](/Users/aelaguiz/workspace/hermes-agent/tests/cron/test_scheduler.py), [tests/test_flush_memories_codex.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_flush_memories_codex.py), [tests/test_codex_execution_paths.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_codex_execution_paths.py) | existing provider/request-path tests plus a few net-new auxiliary owner tests | Coverage is strongest when it follows existing seam-local regression suites rather than creating a separate omnibus service-tier test harness | Add the smallest behavior-level checks for include/omit, direct-route parity, Codex parity, leaf preservation, helper-flow propagation, cache-safe per-turn refresh, and no snapshot coupling | Refactor safety and maintainer confidence | See migration notes below | Full targeted suite plus full repo suite |

## 6.2 Migration notes

* Canonical owner path / shared code path:
  * Config normalization and support-classification owner: [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) adjacent request-options normalizer.
  * Config-writer preservation owners: [hermes_cli/auth.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/auth.py), [hermes_cli/main.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/main.py), and [hermes_cli/setup.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/setup.py), which must preserve the nested config leaf when mutating `config["model"]`.
  * Turn-route propagation owner: [agent/smart_model_routing.py](/Users/aelaguiz/workspace/hermes-agent/agent/smart_model_routing.py), which should accept precomputed `primary_request_options` and own the final effective `request_options` on turn-routed paths.
  * OpenAI payload-injection helper owner: [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) pure helper boundary.
  * Main request application owner: [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) `_build_api_kwargs()` plus one shared codex-dispatch validation point used by primary stream, fallback, and explicit non-stream calls.
  * ACP owner path: [acp_adapter/session.py](/Users/aelaguiz/workspace/hermes-agent/acp_adapter/session.py) should re-normalize request options from current config plus the effective restored runtime snapshot instead of persisting request-options state.
  * Delegation owner path: [tools/delegate_tool.py](/Users/aelaguiz/workspace/hermes-agent/tools/delegate_tool.py) should inherit parent request options when runtime is inherited and re-normalize when delegation overrides runtime facts.
  * Auth-store facts such as `auth_mode="chatgpt"` remain preservation evidence in [hermes_cli/auth.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/auth.py), but they are not promoted into the runtime transport contract for this feature.
* Deprecated APIs (if any):
  * None user-facing. This is a convergence change, not a surface retirement.
* Delete list (what must be removed; include superseded shims/parallel paths if any):
  * Do not ship a second inline `service_tier` compatibility predicate in CLI, gateway, cron, `run_agent.py`, or `agent/auxiliary_client.py`.
  * Do not ship a local model-tier support matrix.
  * Do not ship a Flex fallback shim, timeout wrapper, or retry-to-default behavior in phase 1.
  * Remove any temporary duplicate OpenAI helper logic introduced during implementation once the canonical helper boundary exists.
* Capability-replacing harnesses to delete or justify:
  * Exclude any new tracing subsystem, support-matrix generator, timeout policy harness, or compatibility wrapper whose only job is to hide upstream route differences.
* Live docs/comments/instructions to update or delete:
  * Update high-leverage code comments at the request-options normalizer, smart-routing owner boundary, per-turn mutable state update sites, shared codex-dispatch validation point, and auxiliary helper boundary so future contributors do not re-embed this logic elsewhere.
  * No new slash-command docs or user-facing instructions are required in phase 1.
* Behavior-preservation signals for refactors:
  * [tests/test_model_provider_persistence.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_model_provider_persistence.py), [tests/hermes_cli/test_set_config_value.py](/Users/aelaguiz/workspace/hermes-agent/tests/hermes_cli/test_set_config_value.py), [tests/hermes_cli/test_setup.py](/Users/aelaguiz/workspace/hermes-agent/tests/hermes_cli/test_setup.py), and [tests/hermes_cli/test_setup_model_provider.py](/Users/aelaguiz/workspace/hermes-agent/tests/hermes_cli/test_setup_model_provider.py) for nested `model`-leaf preservation through the repo’s real config writers.
  * [tests/test_run_agent_codex_responses.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_run_agent_codex_responses.py) for direct Responses include/omit and validated codex-dispatch kwargs.
  * [tests/test_run_agent.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_run_agent.py) for direct OpenAI token-key behavior, nested-agent inheritance, and duplicate `errors.log` handler preservation.
  * [tests/test_provider_parity.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_provider_parity.py) for no bleed across direct OpenAI vs Copilot/OpenRouter/Codex/custom-non-OpenAI paths.
  * [tests/agent/test_auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/tests/agent/test_auxiliary_client.py) for net-new auxiliary payload-shape checks around `_build_call_kwargs()` and `_CodexCompletionsAdapter.create()`.
  * [tests/agent/test_smart_model_routing.py](/Users/aelaguiz/workspace/hermes-agent/tests/agent/test_smart_model_routing.py) and [tests/test_credential_pool_routing.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_credential_pool_routing.py) for turn-route contract preservation.
  * [tests/gateway/test_agent_cache.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_agent_cache.py) and [tests/gateway/test_reasoning_command.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_reasoning_command.py) for request-options refresh without cache-key churn and config-backed next-message pickup.
  * [tests/gateway/test_flush_memory_stale_guard.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_flush_memory_stale_guard.py), [tests/gateway/test_session_hygiene.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_session_hygiene.py), [tests/gateway/test_background_command.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_background_command.py), and [tests/gateway/test_api_server_toolset.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_api_server_toolset.py) for helper-agent propagation outside the main turn route.
  * [tests/acp/test_session.py](/Users/aelaguiz/workspace/hermes-agent/tests/acp/test_session.py) and [tests/acp/test_server.py](/Users/aelaguiz/workspace/hermes-agent/tests/acp/test_server.py) for ACP create, restore, and model-switch propagation.
  * [tests/tools/test_delegate.py](/Users/aelaguiz/workspace/hermes-agent/tests/tools/test_delegate.py) for delegated-child inheritance and override behavior.
  * [tests/test_primary_runtime_restore.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_primary_runtime_restore.py) for no request-options snapshot coupling.
  * [tests/test_runtime_provider_resolution.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_runtime_provider_resolution.py), [tests/test_cli_provider_resolution.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_cli_provider_resolution.py), [tests/cron/test_scheduler.py](/Users/aelaguiz/workspace/hermes-agent/tests/cron/test_scheduler.py), [tests/test_flush_memories_codex.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_flush_memories_codex.py), and [tests/test_codex_execution_paths.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_codex_execution_paths.py) for propagation and request-builder reality checks.

## 6.3 Pattern Consolidation Sweep (anti-blinders; scoped by plan)

| Area | File / Symbol | Pattern to adopt | Why (drift prevented) | Proposed scope (include/defer/exclude) |
| ---- | ------------- | ---------------- | ---------------------- | ------------------------------------- |
| Transport vs request policy | [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) | Keep `resolve_runtime_provider()` transport-only; add adjacent request-options normalizer | Prevent transport dict bloat and downstream cherry-pick drift | include |
| Per-turn mutable state | [cli.py](/Users/aelaguiz/workspace/hermes-agent/cli.py), [gateway/run.py](/Users/aelaguiz/workspace/hermes-agent/gateway/run.py), [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) | Treat `request_options` like `reasoning_config` for long-lived agents | Prevent stale request state without invalidating prompt caches | include |
| Config-writer leaf preservation | [hermes_cli/auth.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/auth.py), [hermes_cli/main.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/main.py), [hermes_cli/setup.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/setup.py) | Preserve nested `model.request_options.*` siblings whenever `config["model"]` is rewritten | Prevent clobbering the new leaf in the repo’s actual mutation seams | include |
| First-class fresh-agent surfaces | [acp_adapter/session.py](/Users/aelaguiz/workspace/hermes-agent/acp_adapter/session.py), [acp_adapter/server.py](/Users/aelaguiz/workspace/hermes-agent/acp_adapter/server.py), [gateway/platforms/api_server.py](/Users/aelaguiz/workspace/hermes-agent/gateway/platforms/api_server.py), [tools/delegate_tool.py](/Users/aelaguiz/workspace/hermes-agent/tools/delegate_tool.py) | Seed or re-normalize request options on every fresh-agent path that is part of the supported stack | Prevent “main path works, secondary surface silently drops it” drift | include |
| OpenAI helper reuse | [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py), [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py), [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) | Keep support classification in the runtime normalizer and keep payload injection plus token-key choice in the helper boundary | Prevent main/aux drift and remove dual Codex truth | include |
| Full Responses payload unification | [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) `_build_api_kwargs()` and friends | Do not extract the whole Responses builder in this run | Too broad, too risky, not needed for `service_tier` | defer |
| Auth metadata widening | [hermes_cli/auth.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/auth.py), [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) | Do not widen the runtime transport contract with `auth_mode` just to classify Codex | Keeps the route contract narrow and avoids unnecessary propagation churn | exclude |
| Model-support tables | any new helper or config path | Do not add model-tier allowlists | Upstream support lives on pricing/support surfaces and can drift | exclude |
| Flex timeout policy | [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py), [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) | Keep existing timeout/retry behavior | Preserves backward compatibility and avoids hidden cost/latency policy changes | exclude |
| `openai-codex` parity | [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py), [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py), [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) | Treat `openai-codex` as a required OpenAI surface using the same canonical request-options seam and a route-fact-based predicate on the known backend | Prevents the plan from drifting into direct-only support while keeping Codex logic out of non-OpenAI paths and arbitrary endpoints | include |
| Unscoped constructor seams | [batch_runner.py](/Users/aelaguiz/workspace/hermes-agent/batch_runner.py), [gateway/builtin_hooks/boot_md.py](/Users/aelaguiz/workspace/hermes-agent/gateway/builtin_hooks/boot_md.py) | Do not claim them as covered unless they are consciously pulled onto the same normalizer path with proportional tests | Prevent plan drift and low-value test sprawl | exclude |
<!-- arch_skill:block:call_site_audit:end -->

<!-- arch_skill:block:phase_plan:start -->
# 7) Depth-First Phased Implementation Plan (authoritative)

> Rule: systematic build, foundational first; every phase has exit criteria + explicit verification plan (tests optional). Refactors, consolidations, and shared-path extractions must preserve existing behavior with the smallest credible signal. For agent-backed systems, prefer prompt, grounding, and native-capability changes before new harnesses or scripts. No fallbacks/runtime shims - the system must work correctly or fail loudly (delete superseded paths). Prefer programmatic checks per phase; defer manual/UI verification to finalization. Avoid negative-value tests (deletion checks, visual constants, doc-driven gates). Also: document new patterns/gotchas in code comments at the canonical boundary (high leverage, not comment spam).

## Phase 1 — Config And Canonical Codex Classification

Status: COMPLETE

Completed work:
* Confirmed the persisted config leaf and the real config-writer preservation paths are already landed on this branch, so the reopened foundation delta is limited to Codex support classification at the canonical normalizer boundary.
* Updated the runtime-provider-owned support classifier so the known `openai-codex` route is supported via route facts (`provider`, `api_mode`, known backend path) without widening the runtime contract with `auth_mode`.
* Preserved the existing adjacent `resolve_runtime_request_options(...)` seam instead of introducing a second request-options owner.
* Kept warn-once omission behavior only for genuinely unsupported routes.

* Goal:
  * Finish the only remaining foundation work at the canonical seam: one migration-safe persisted `model.request_options.service_tier` leaf plus one runtime-provider-owned support classifier that treats the required `openai-codex` route as supported without widening the runtime contract.
* Work:
  * Add migration-safe support for `model.request_options.service_tier` in [hermes_cli/config.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/config.py) without breaking scalar or mixed `model` shapes.
  * Update the real `config["model"]` writers in [hermes_cli/auth.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/auth.py), [hermes_cli/main.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/main.py), and [hermes_cli/setup.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/setup.py) so provider/model/base-url rewrites preserve the nested request-options leaf.
  * Preserve the existing adjacent `resolve_runtime_request_options(...)` seam in [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) rather than re-plumbing callers around a new abstraction.
  * Validate only the documented request enum: `auto`, `default`, `flex`, `scale`, `priority`.
  * Replace the current blanket `openai-codex` exclusion with the locked route-fact predicate at the canonical normalizer boundary: `provider=="openai-codex"`, `api_mode=="codex_responses"`, and normalized base URL on the known `chatgpt.com/backend-api/codex` backend.
  * Keep support classification grounded in effective runtime facts, not provider branding alone, not auth-store widening, and not a local model-support matrix.
  * Keep warn-once omission behavior only for genuinely unsupported routes, not for the required Codex route.
* Verification (smallest signal):
  * Run [tests/hermes_cli/test_config.py](/Users/aelaguiz/workspace/hermes-agent/tests/hermes_cli/test_config.py) if config migration helpers change.
  * Run [tests/test_model_provider_persistence.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_model_provider_persistence.py).
  * Run [tests/hermes_cli/test_set_config_value.py](/Users/aelaguiz/workspace/hermes-agent/tests/hermes_cli/test_set_config_value.py).
  * Run [tests/hermes_cli/test_setup.py](/Users/aelaguiz/workspace/hermes-agent/tests/hermes_cli/test_setup.py).
  * Run [tests/hermes_cli/test_setup_model_provider.py](/Users/aelaguiz/workspace/hermes-agent/tests/hermes_cli/test_setup_model_provider.py).
  * Run [tests/test_runtime_provider_resolution.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_runtime_provider_resolution.py).
  * Run [tests/test_cli_provider_resolution.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_cli_provider_resolution.py).
  * Run [tests/test_auth_codex_provider.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_auth_codex_provider.py) if the known Codex backend facts or auth-store preservation evidence influence the classifier.
* Docs/comments (propagation; only if needed):
  * Add or keep one short comment at the request-options normalizer boundary explaining that transport resolution stays transport-only, request policy lives adjacent to it, and Codex support is keyed off route facts rather than auth-mode widening.
* Verification:
  * `source .venv/bin/activate && python -m pytest tests/test_runtime_provider_resolution.py tests/test_cli_provider_resolution.py tests/test_auth_codex_provider.py tests/test_run_agent_codex_responses.py tests/test_run_agent.py tests/agent/test_auxiliary_client.py tests/test_provider_parity.py tests/test_flush_memories_codex.py tests/test_codex_execution_paths.py -q`
  * Result: `502 passed, 16 warnings`
* Exit criteria:
  * Older config shapes still load.
  * `service_tier` normalizes into one canonical request-options dict.
  * Direct OpenAI and the required `openai-codex` route can both produce canonical request options when configured.
  * Genuinely unsupported routes deterministically omit at normalization time without adding non-OpenAI behavior.
* Rollback:
  * Revert the config leaf and adjacent normalizer together; do not leave a persisted setting with no canonical consumer.

## Phase 2 — Propagation Preservation And Gap-Fix-Only Follow-Through

Status: COMPLETE

* Goal:
  * Preserve the already-landed `request_options` seam across first-class agent surfaces and only reopen this phase if Codex implementation reveals a real missed constructor, restore path, or stale-state bug.
* Work:
  * Treat [agent/smart_model_routing.py](/Users/aelaguiz/workspace/hermes-agent/agent/smart_model_routing.py), [cli.py](/Users/aelaguiz/workspace/hermes-agent/cli.py), [gateway/run.py](/Users/aelaguiz/workspace/hermes-agent/gateway/run.py), [cron/scheduler.py](/Users/aelaguiz/workspace/hermes-agent/cron/scheduler.py), [acp_adapter/session.py](/Users/aelaguiz/workspace/hermes-agent/acp_adapter/session.py), [gateway/platforms/api_server.py](/Users/aelaguiz/workspace/hermes-agent/gateway/platforms/api_server.py), [tools/delegate_tool.py](/Users/aelaguiz/workspace/hermes-agent/tools/delegate_tool.py), and [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) as preservation surfaces, not new architecture work.
  * If Phase 1 or Phase 3 exposes a real propagation miss, fix it by extending the existing canonical seam rather than inventing a caller-specific branch or widening transport/cache identity.
  * Keep `request_options` out of `_primary_runtime` and out of cache signatures unless implementation uncovers a concrete correctness bug that Section 0 would otherwise reject.
* Verification (smallest signal):
  * No new verification is required if no new propagation code lands.
  * If a real gap fix lands, rerun only the owner suites for the touched surface:
    * [tests/agent/test_smart_model_routing.py](/Users/aelaguiz/workspace/hermes-agent/tests/agent/test_smart_model_routing.py)
    * [tests/test_credential_pool_routing.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_credential_pool_routing.py)
    * [tests/gateway/test_agent_cache.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_agent_cache.py)
    * [tests/gateway/test_reasoning_command.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_reasoning_command.py)
    * [tests/gateway/test_flush_memory_stale_guard.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_flush_memory_stale_guard.py)
    * [tests/gateway/test_session_hygiene.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_session_hygiene.py)
    * [tests/gateway/test_background_command.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_background_command.py)
    * [tests/gateway/test_api_server_toolset.py](/Users/aelaguiz/workspace/hermes-agent/tests/gateway/test_api_server_toolset.py)
    * [tests/acp/test_session.py](/Users/aelaguiz/workspace/hermes-agent/tests/acp/test_session.py)
    * [tests/acp/test_server.py](/Users/aelaguiz/workspace/hermes-agent/tests/acp/test_server.py)
    * [tests/cron/test_scheduler.py](/Users/aelaguiz/workspace/hermes-agent/tests/cron/test_scheduler.py)
    * [tests/tools/test_delegate.py](/Users/aelaguiz/workspace/hermes-agent/tests/tools/test_delegate.py)
    * [tests/test_primary_runtime_restore.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_primary_runtime_restore.py)
* Docs/comments (propagation; only if needed):
  * No new docs/comments are required unless a real propagation gap is fixed; if one is, update only the touched canonical boundary comment rather than scattering commentary across callers.
* Exit criteria:
  * No newly discovered propagation gap remains in first-class supported surfaces.
  * Cached agents still do not go stale when request options change.
  * `_primary_runtime` remains transport-focused.
* Rollback:
  * If a gap fix is required, revert that fix together with its local tests; do not reopen broad propagation churn without evidence.

## Phase 3 — Main And Auxiliary Codex Request-Path Convergence

Status: COMPLETE

Completed work:
* Updated the shared OpenAI helper boundary in [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) so it reuses the runtime-provider-owned support classifier instead of carrying an independent Codex exclusion.
* Updated [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) `_build_api_kwargs()` so the main Codex path injects `service_tier` through that shared helper boundary.
* Updated `_CodexCompletionsAdapter.create()` so the auxiliary Codex path forwards `service_tier` instead of stripping it.
* Added the shared preflight guard to the primary `_run_codex_stream()` path so every main Codex dispatch now passes through the same validated request contract before network I/O.
* Flipped the representative omission assertions in the runtime-provider, main Responses, auxiliary-client, and parity owner suites into Codex support-path assertions.

* Goal:
  * Make the required Codex route actually carry `service_tier` end to end in both the main runner and auxiliary OpenAI paths, while keeping one canonical support decision and one main validated codex-dispatch contract.
* Work:
  * Preserve [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py) as the only owner of the Codex support decision; [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) may import or consume that decision, but it must not keep an independent `openai-codex` compatibility rule.
  * Update the shared OpenAI helper boundary in [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) so it owns payload injection and token-key choice only, and no longer early-returns or strips `service_tier` for the required Codex route.
  * Update [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) `_build_api_kwargs()` so the main Codex Responses path injects `service_tier` through the shared helper boundary instead of inheriting the old omit behavior.
  * Introduce one shared allowlist-aware validation point for Codex/Responses dispatch that covers primary `responses.stream()`, the `create(stream=True)` fallback, and explicit non-stream Codex paths before network I/O.
  * Allowlist `service_tier` in that validated Codex dispatch path and keep failure behavior loud if a claimed-supported route still rejects it.
  * Update auxiliary direct-OpenAI chat-completions request shaping in [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) only as needed to keep direct OpenAI and Codex behavior on the same helper contract; do not widen this into a full request-builder refactor.
  * Flip the representative omission assertions in the runtime-provider, main Responses, and auxiliary-client owner suites into support-path assertions for the required Codex route.
  * Keep Flex behavior as pass-through only; do not add timeout widening, silent retries, fallback-to-default logic, or a local model-support matrix.
* Verification (smallest signal):
  * Run [tests/test_runtime_provider_resolution.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_runtime_provider_resolution.py) because the Codex support predicate moves at the canonical normalizer boundary.
  * Run [tests/test_run_agent_codex_responses.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_run_agent_codex_responses.py).
  * Run [tests/test_run_agent.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_run_agent.py).
  * Run [tests/agent/test_auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/tests/agent/test_auxiliary_client.py).
  * Run [tests/test_provider_parity.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_provider_parity.py).
  * Run [tests/test_flush_memories_codex.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_flush_memories_codex.py).
  * Run [tests/test_codex_execution_paths.py](/Users/aelaguiz/workspace/hermes-agent/tests/test_codex_execution_paths.py).
* Docs/comments (propagation; only if needed):
  * Add or update a short comment at the shared helper boundary explaining that only pure OpenAI request-shape logic belongs there and that Codex support classification stays owned by the runtime normalizer.
  * Add a short comment at the shared codex-dispatch validation point explaining that every main Responses dispatch must pass through it before network I/O.
* Verification:
  * `source .venv/bin/activate && python -m pytest tests/test_runtime_provider_resolution.py tests/test_cli_provider_resolution.py tests/test_auth_codex_provider.py tests/test_run_agent_codex_responses.py tests/test_run_agent.py tests/agent/test_auxiliary_client.py tests/test_provider_parity.py tests/test_flush_memories_codex.py tests/test_codex_execution_paths.py -q`
  * Result: `502 passed, 16 warnings`
  * `source .venv/bin/activate && python -m pytest tests/test_streaming.py::TestCodexStreamCallbacks -q`
  * Result: `2 passed, 2 warnings`
* Exit criteria:
  * Main and auxiliary OpenAI paths include `service_tier` on both supported direct API-key routes and the required `openai-codex` route.
  * Genuinely unsupported routes omit deterministically.
  * No duplicate Codex compatibility predicate remains in CLI, gateway, cron, `run_agent.py`, or `agent/auxiliary_client.py`.
* Rollback:
  * Revert the shared-helper and validated-dispatch changes together; do not keep half-converged helper extraction or partial stream-path validation.

## Phase 4 — Useful Verification, Reality-Sync, And PR Readiness

Status: IN PROGRESS

Completed work:
* Re-ran the representative Hermes-native service-tier matrix after the Codex fixes:
  * `source .venv/bin/activate && python -m pytest tests/hermes_cli/test_config.py tests/hermes_cli/test_set_config_value.py tests/test_model_provider_persistence.py tests/hermes_cli/test_setup.py tests/hermes_cli/test_setup_model_provider.py tests/test_runtime_provider_resolution.py tests/agent/test_smart_model_routing.py tests/test_credential_pool_routing.py tests/test_cli_provider_resolution.py tests/gateway/test_agent_cache.py tests/gateway/test_reasoning_command.py tests/gateway/test_flush_memory_stale_guard.py tests/gateway/test_session_hygiene.py tests/gateway/test_background_command.py tests/gateway/test_api_server_toolset.py tests/acp/test_session.py tests/acp/test_server.py tests/cron/test_scheduler.py tests/tools/test_delegate.py tests/test_primary_runtime_restore.py tests/test_run_agent_codex_responses.py tests/test_run_agent.py tests/agent/test_auxiliary_client.py tests/test_provider_parity.py tests/test_flush_memories_codex.py tests/test_codex_execution_paths.py -q`
  * Result: `875 passed, 16 warnings`
* Re-ran the full repo suite after the Codex fixes:
  * `source .venv/bin/activate && python -m pytest tests/ -q`
  * Result: `8057 passed, 37 failed, 84 skipped, 1 xpassed, 134 warnings`
  * The remaining failures appear broad and unrelated to the service-tier change surface, including Matrix voice, provider/setup detection, managed tool/modal execution, skill-manager operations, transcription dependency coverage, terminal-tool requirements, and update-gateway restart behavior.
* Ran a live `openai-codex` smoke test against the branch using a throwaway `HERMES_HOME` cloned from the personal auth store, set `model.request_options.service_tier=priority`, enabled `HERMES_DUMP_REQUESTS=1`, and executed a quiet single-query chat with `--provider openai-codex -m gpt-5.2-codex`.
* The live Codex smoke succeeded and the captured request dump body included `service_tier: "priority"` on `https://chatgpt.com/backend-api/codex/responses`.

Manual QA (non-blocking):
* Live direct API-key OpenAI validation is still pending because no real direct `OPENAI_API_KEY` route is configured in this shell or the temporary profile copy used for the Codex smoke.
* Cross-platform impact review is still pending before PR submission.

* Goal:
  * Finish the reopened Codex scope to Hermes maintainer standards using representative owner suites and real runtime checks, not low-value ceremony or generic harness sprawl.
* Work:
  * Fill any still-missing owner-level tests identified in earlier phases, especially:
    * Codex include behavior at the canonical normalizer boundary
    * main Codex Responses payload and validated-dispatch behavior
    * `_CodexCompletionsAdapter.create()` include behavior
    * warn-once omission behavior for genuinely unsupported routes
    * nested review-agent inheritance coverage only if a touched test seam does not already prove it
  * Keep the targeted matrix representative of Hermes’ existing regression style:
    * config and config-writer suites for the nested leaf
    * runtime-provider and CLI provider-resolution suites for classification
    * run-agent and auxiliary-client owner suites for payload shape
    * gateway, ACP, cron, delegate, and restore suites only if touched or needed to prove a real gap fix
  * Avoid low-value coverage: do not treat CLI TUI refresh tests, generic streaming tests, client-lifecycle tests, or exhaustive provider matrices as substitutes for owner-level behavior checks unless implementation actually changes those surfaces.
  * Update any touched live comments to current truth, especially around request-options normalization, smart-routing ownership, mutable request state, and validated codex dispatch.
  * Run the full repo suite: `pytest tests/ -v`.
  * Manually run `hermes` and exercise the changed direct OpenAI path and `openai-codex` path end to end.
  * Perform the contribution-guide cross-platform impact review, with explicit attention to whether any touched config/file or logging behavior carries platform-specific risk.
  * Prepare PR framing so the title and description clearly say this is support for OpenAI `service_tier` on Hermes-supported OpenAI request surfaces, including `openai-codex`.
* Verification (smallest signal):
  * The targeted owner suites from prior phases all pass, with Codex omission assertions flipped into support-path assertions where the required scope now demands include behavior.
  * `pytest tests/ -v` passes.
  * Manual `hermes` exercise confirms the configured supported direct route includes `service_tier`, the configured Codex route includes `service_tier`, and an unsupported route omits it with warn-once logging.
* Docs/comments (propagation; only if needed):
  * Sync any surviving touched comments or instructions to final implementation truth in the same phase; delete stale explanations rather than preserve legacy text.
* Exit criteria:
  * The feature is fully backward-compatible when unset.
  * Supported direct OpenAI and `openai-codex` routes work end to end.
  * Unsupported routes degrade by deterministic omission with proportional logging.
  * The patch is ready for `implement` follow-through and later `audit-implementation`, without any remaining fake-complete phases.
* Rollback:
  * Do not ship without the full targeted and full-suite verification bar; if finalization reveals architectural drift, reopen the relevant earlier phase instead of layering a shim.
<!-- arch_skill:block:phase_plan:end -->

# 8) Verification Strategy (common-sense; non-blocking)

## 8.1 Unit tests (contracts)

Lean on the repo’s existing seam-local regression suites instead of inventing a parallel service-tier harness. The highest-signal owner tests are:

- `tests/test_model_provider_persistence.py`, `tests/hermes_cli/test_set_config_value.py`, `tests/hermes_cli/test_setup.py`, and `tests/hermes_cli/test_setup_model_provider.py` for nested `model.request_options.service_tier` preservation through real config writers.
- `tests/test_runtime_provider_resolution.py` and `tests/test_cli_provider_resolution.py` for adjacent request-options normalization, direct-route compatibility gating, Codex compatibility gating, and custom-endpoint omission policy.
- `tests/agent/test_smart_model_routing.py` and `tests/test_credential_pool_routing.py` for turn-route contract preservation.
- `tests/agent/test_auxiliary_client.py` for the net-new owner-level payload-shape assertions around `_build_call_kwargs()` and `_CodexCompletionsAdapter.create()`, including Codex include behavior.
- One focused `caplog`-style contract test at the canonical omission or validation boundary for warn-once degradation logging; do not spread log assertions across every caller.

## 8.2 Integration tests (flows)

Use targeted flow tests around the actual supported runtime surfaces so effective `request_options` propagate correctly, refresh in place on reused agents without cache-key churn, stay out of `_primary_runtime` restore state, preserve existing behavior for users who never configure the feature, and do not create new behavior on non-OpenAI routes. The representative suites are:

- `tests/gateway/test_agent_cache.py` and `tests/gateway/test_reasoning_command.py` for the mutable-state-on-next-message pattern.
- `tests/gateway/test_flush_memory_stale_guard.py`, `tests/gateway/test_session_hygiene.py`, `tests/gateway/test_background_command.py`, and `tests/gateway/test_api_server_toolset.py` for helper, maintenance, background, and API-server fresh-agent flows.
- `tests/acp/test_session.py` and `tests/acp/test_server.py` for ACP create, restore, and model-switch flows.
- `tests/tools/test_delegate.py` for delegated-child inheritance and override behavior.
- `tests/cron/test_scheduler.py` and `tests/test_primary_runtime_restore.py` for cron propagation and snapshot-boundary preservation.

## 8.3 E2E / device tests (realistic)

No special E2E or device layer is expected for phase 1. Keep verification proportional: prefer the smallest existing signal, avoid new harnesses, prove old behavior by showing that unset `service_tier` preserves current request construction behavior and unsupported routes do not receive speculative fields, avoid assertions that omission must yield a particular upstream tier because project defaults can change that outcome, do not add special Flex timeout or retry assertions beyond preserving current fail-loud behavior, and still require a full `pytest tests/ -v` pass, a manual `hermes` exercise of both the changed direct OpenAI path and the changed `openai-codex` path, and a cross-platform impact review before merge because the repo’s contribution guide explicitly requires them.

Useful reality checks here are `tests/test_run_agent_codex_responses.py`, `tests/test_run_agent.py`, `tests/test_provider_parity.py`, `tests/test_flush_memories_codex.py`, and `tests/test_codex_execution_paths.py`, because they exercise the same request builder and dispatch surfaces the feature actually changes, especially on the required Codex path.

Coverage to avoid unless implementation truly changes those surfaces:

- `tests/test_cli_background_tui_refresh.py` and other TUI-refresh tests as substitutes for request propagation coverage.
- Generic streaming or client-lifecycle suites such as `tests/test_streaming.py` or `tests/test_openai_client_lifecycle.py`.
- Exhaustive provider-by-provider matrices that duplicate parity coverage without increasing confidence.

# 9) Rollout / Ops / Telemetry

## 9.1 Rollout plan

Roll out as an opt-in config feature with no default behavior change.
When the PR is prepared, title and description should explicitly describe support for OpenAI `service_tier` on Hermes-supported OpenAI request surfaces, including `openai-codex`, so maintainers can map the change directly to the actual runtime behavior under review.

## 9.2 Telemetry changes

No telemetry expansion is planned in phase 1 unless later research shows that returned service-tier metadata is stable, useful, and cheap to carry. Logging should integrate with the existing redacted Hermes logs rather than add feature-local telemetry.

## 9.3 Operational runbook

The main operational risk is route misclassification or a missed per-turn `request_options` refresh on reused agents. Operational handling should stay simple: explicit compatibility gating, deterministic omit behavior for unsupported routes, warn-once logging for expected degraded optional behavior so persistent logs do not spam on every turn, `error` plus `exc_info=True` for unexpected failures, and no secrets in emitted logs. In particular, `openai-codex` is now a required supported route rather than an automatic omit route, so omission there is a bug unless a narrower evidence-backed incompatibility is documented in the plan and tests. When upstream returns `service_tier`, operators should treat it as the tier actually used, not a guaranteed echo of the request, and request omission can still resolve to a project-level default upstream.

<!-- arch_skill:block:review_gate:start -->
## Review Gate
- Reviewers: self
- Question asked: "Is this idiomatic, convergent, and complete relative to DOC_PATH? Are we routing through the canonical existing path? Did we add a new way to do something unnecessarily? Did we understand the relevant agent and model capabilities before designing? Are we replacing prompt or native-capability work with scaffolding? Did we silently compress any instruction-bearing content while porting it? What is missing? Where does code or plan drift? Are there any SSOT, contract, behavior-preservation, or stale-live-doc gaps?"
- Feedback summary:
  - Research Grounding still overclaimed `_preflight_codex_api_kwargs()` ownership after the later deep-dive clarified that the primary stream path bypasses it today.
  - External Research still carried a stale Flex open question even though later sections had already fixed phase-1 Flex scope.
  - Target Architecture had one vague warning-path phrase that drifted away from the now-locked warn-once omission behavior.
  - The call-site audit still missed real first-class execution surfaces: ACP, API server fresh agents, delegated subagents, and the real config writers in `auth.py`, `main.py`, and `setup.py`.
  - The test plan still over-weighted generic loader and cache suites while under-weighting the repo’s higher-signal persistence, setup-sync, ACP, delegation, and config-backed mutable-state suites.
  - The doc still overclaimed end-to-end coverage for direct constructor paths such as `batch_runner.py` and the gateway `BOOT.md` hook without an explicit inclusion or exclusion decision.
- Integrated changes:
  - Rewrote the stale `run_agent.py` ownership claim in Section 3.2 to match the actual split validation state.
  - Replaced the obsolete external-research open question with an explicit “none remaining” note.
  - Tightened the target warning-path wording to the feature’s warn-once omission path.
  - Added ACP, API server, delegated-child, and config-writer preservation paths to the target architecture, call-site audit, and phase plan.
  - Explicitly excluded `batch_runner.py` and the gateway `BOOT.md` hook from phase-1 acceptance unless implementation consciously pulls them onto the same canonical path.
  - Rebased the verification strategy on the repo’s representative seam-local suites, including model-provider persistence, setup sync, gateway reasoning reload, ACP session/server, API-server toolset, delegation, and codex maintenance-path tests.
- Decision: proceed to next phase? (yes)
<!-- arch_skill:block:review_gate:end -->

# 10) Decision Log (append-only)

Later entries supersede earlier boundary decisions when the required scope changes. The newest `openai-codex` entry is therefore the current source of truth for support boundaries.

## 2026-04-04 - Canonical source, support boundary, and convergence scope

Context

Hermes needs OpenAI `service_tier` support in a form maintainers can accept: config-owned, route-aware, end-to-end propagated, architecturally convergent, fully backward compatible, clearly named after the upstream mechanism, and compliant with the contribution guide’s emphasis on robustness, graceful degradation, focused changes, and full test coverage. Existing repo evidence shows duplicated Responses request-option logic and an unsupported assumption that `openai-codex` is equivalent to direct OpenAI API routing.

Options

- Treat `service_tier` as a small direct patch in `run_agent.py` only.
- Build a broad new Responses adapter that immediately absorbs all main-runner and auxiliary-client behavior.
- Add a normalized request-options path end to end and converge only the compatibility and injection boundary first.

Decision

Use the third option. This canonical full-arch doc supersedes `docs/plans/2026-04-04-openai-service-tier-architecture.md`, phase 1 support is limited to explicitly verified direct OpenAI API key-authenticated routes, the convergence boundary is intentionally narrow shared ownership for direct OpenAI Responses request-option compatibility and injection, unsupported routes degrade by omitting the optional field with warn-once warning-level logs, non-OpenAI routes must not gain new behavior, and merge readiness requires targeted feature tests plus the repo’s full PR-readiness bar.

Consequences

- The first implementation is smaller, more reviewable, and less likely to create architectural debt.
- `openai-codex` support is not promised until later evidence proves it safe.
- Runtime propagation, config migration, and agent-reuse boundaries are part of the required scope rather than deferred cleanup.
- The design is explicitly grounded in the direct OpenAI API docs as of 2026-04-04, which document API key authentication for this surface rather than OAuth.
- The naming and eventual PR framing should make the supported mechanism obvious to Hermes maintainers instead of forcing them to infer it from implementation details.
- PR readiness is defined by Hermes’ actual contribution guide, not just automated tests: full suite, manual `hermes` exercise, and a cross-platform impact review.

Follow-ups

- Complete deep-dive grounding across all propagation and cache boundaries.
- Confirm the direct OpenAI API enum and supported route surface through external research.
- Author the authoritative phased implementation plan once the call-site inventory is complete.

## 2026-04-04 - External research clarified enum, ownership, and operational scope

Context

The plan needed external grounding on the exact documented `service_tier` request enum for the Responses surface, on whether Hermes should own model-tier compatibility tables, on whether returned `service_tier` can be treated as a request echo, and on whether Flex support implies new timeout or fallback work that would violate the patch’s anti-complexity stance.

Options

- Validate only the guide-emphasized values and leave the rest undocumented.
- Validate the current Responses-documented request enum and avoid local model-support allowlists.
- Add Hermes-local retry, fallback, timeout, or model-support machinery so every documented tier looks uniformly supported.

Decision

Use the second option. External OpenAI docs now ground the Responses request enum as `auto`, `default`, `flex`, `scale`, and `priority`; show that returned `service_tier` may differ from the requested value; show that project defaults can affect requests where the field is omitted; and keep model support on pricing or support surfaces rather than in the core request contract. Phase 1 therefore stays route-gated and enum-validated, does not add a local model-tier matrix, and does not add silent Flex fallback behavior.

Consequences

- Verification should assert request construction and degraded-path behavior, not that returned `service_tier` equals the configured value.
- Hermes should not hardcode model-tier support tables that would drift from upstream pricing or support docs.
- Flex remains the only tier whose operational semantics may still force a deep-dive scope decision because official guidance calls out longer timeouts and `429 Resource Unavailable`.
- Logging may optionally absorb OpenAI request IDs if a low-cost hook exists, but the feature should not widen into a tracing project.
- Phase 1 logging still needs omission warnings to stay proportional rather than firing on every unsupported turn.

Follow-ups

- Deep-dive the current OpenAI timeout surfaces before promising initial Flex support without caveats.
- Reflect the enum, no-allowlist stance, and actual-vs-requested `service_tier` semantics in the phase plan and verification checklist.
- Keep omission-warning scope explicit so graceful degradation does not become persistent log noise.

## 2026-04-04 - Deep-dive locked the request-options seam and mutable-state model

Context

The remaining architecture question after research was where normalized `request_options` should live, how they should move through cached CLI and gateway agents without stale state or prompt-cache churn, and how to remove direct-OpenAI helper drift without widening the change into a full Responses refactor. Parallel repo analysis also established that Hermes has no phase-1-owned Flex timeout surface worth expanding just for this feature.

Options

- Widen `resolve_runtime_provider()` so transport and request policy live in one runtime dict, then thread that larger dict through every signature and snapshot boundary.
- Keep request options constructor-scoped on `AIAgent` and widen cached-agent signatures or `_primary_runtime` snapshots so changes cannot go stale.
- Keep `resolve_runtime_provider()` transport-only, add an adjacent request-options normalizer, treat `request_options` as per-turn mutable agent state like `reasoning_config`, and extract only pure direct-OpenAI helpers into [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py).

Decision

Use the third option. The canonical phase-1 architecture keeps transport/auth route identity in [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py), adds an adjacent normalized request-options seam, carries that object through [agent/smart_model_routing.py](/Users/aelaguiz/workspace/hermes-agent/agent/smart_model_routing.py), refreshes `AIAgent.request_options` per turn in CLI and gateway, seeds it on fresh cron and temporary agents, keeps [run_agent.py](/Users/aelaguiz/workspace/hermes-agent/run_agent.py) as the owner of the full Responses payload and preflight contract, and shares only pure OpenAI helper logic through [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py). `flex` remains documented pass-through only with current Hermes timeout and retry behavior preserved.

Consequences

- The patch stays narrow and Hermes-native: no generalized provider framework, no widened transport runtime contract, and no prompt-cache or tool-schema churn.
- CLI and gateway avoid stale request policy by refreshing `request_options` per turn instead of rebuilding agents or widening cache signatures for a request-time knob.
- `_primary_runtime` stays transport-focused; this feature does not couple request-time policy to fallback restore state.
- The direct OpenAI compatibility predicate stays tied to effective route facts rather than branding or a local model-support matrix.
- At that stage of the plan, `openai-codex` and other OAuth-backed or non-OpenAI routes were still explicit omit paths rather than speculative parity claims; later Codex-required decisions supersede that scope.
- Expected omission logging should be warn-once per effective unsupported route so graceful degradation does not create per-turn persistent log spam.

Follow-ups

- Express this exact architecture in the phase plan with implementation order: config and normalization, turn-route propagation, agent-state threading, shared-helper convergence, and tests.
- Keep any future doctor or request-ID logging follow-up separate unless it can be added without duplicating compatibility truth or widening scope.

## 2026-04-04 - Phase plan locked the execution order and ship bar

Context

After deep-dive, the architecture was clear but Section 7 was still only a placeholder. The plan needed one authoritative implementation checklist that converted the locked architecture into shippable order without leaving helper flows, nested agents, or validation gaps implicit.

Options

- Keep Section 7 high-level and let implementation decide the detailed order.
- Write a narrow foundational-first sequence that lands config and normalization first, propagation second, request-shaping convergence third, and PR-readiness verification last.
- Split execution into more speculative sub-phases or optional tracks for Flex, OAuth-backed Codex, or broader provider parity.

Decision

Use the second option. Section 7 now locks four phases: config and request-options foundation, turn-route and agent-state propagation, direct OpenAI request-shaping convergence, and final verification plus PR readiness. The execution order explicitly includes helper and maintenance agents, nested review-agent behavior, one validated main codex-dispatch contract, net-new auxiliary payload-shape tests, the full suite, manual `hermes` exercise, and cross-platform review.

Consequences

- `implement` now has one concrete checklist instead of inferring execution order from Sections 5 and 6.
- `audit-implementation` can judge completeness against explicit phase exit criteria.
- Optional or speculative work remains out of the ship path.

Follow-ups

- Use Section 7 as the only execution checklist during implementation.
- Reopen earlier phases if implementation evidence proves the locked order incomplete, instead of adding runtime shims or sidecar TODO checklists.

## 2026-04-04 - Deep-dive pass 3 widened the audited stack and hardened the test strategy

Context

One more parallel deep-dive pass was needed to verify that the plan’s “end to end” claim matched Hermes’ actual stack and that the verification plan looked like the repo’s real regression style rather than synthetic service-tier coverage. That pass found several first-class seams still missing from the artifact: ACP session creation and restore, API-server fresh-agent construction, delegated child agents, and the real config writers in `auth.py`, `main.py`, and `setup.py`.

Options

- Keep the plan focused on CLI, gateway, and cron only, and accept that some supported surfaces would silently miss the feature.
- Pull every direct `AIAgent` constructor in the repo into phase 1, including low-signal paths like `batch_runner.py` and the gateway `BOOT.md` hook.
- Include the supported first-class surfaces with existing representative tests, explicitly exclude low-signal constructor seams from phase-1 acceptance, and rewrite the test plan around the repo’s seam-local regression suites.

Decision

Use the third option. The canonical plan now includes ACP, API server, delegated subagents, and the real config writers as supported phase-1 surfaces; ACP restore explicitly re-normalizes request options from current config plus the persisted runtime snapshot; delegation explicitly inherits or re-normalizes request options based on child runtime; and `batch_runner.py` plus the gateway `BOOT.md` hook are called out as explicit phase-1 exclusions unless implementation consciously pulls them onto the same path.

Consequences

- The plan’s end-to-end claim is now honest about which supported surfaces are included and which constructor-only paths are intentionally excluded.
- The most likely regression class, clobbering nested `model` siblings during provider/model rewrites, now has first-class coverage in the plan through the repo’s existing persistence and setup-sync suites.
- Gateway propagation is now anchored to Hermes’ existing config-backed mutable-state pattern, not just cache-signature tests.
- ACP, API server, and delegated-child propagation can be implemented without inventing new harnesses because the repo already has owner-level test seams for each.
- The test strategy now explicitly rejects low-value coverage such as TUI refresh tests, generic streaming suites, and exhaustive provider matrices unless the implementation truly touches those areas.

Follow-ups

- Use the widened call-site audit and revised Section 8 test strategy as the implementation checklist for scope and verification.
- If implementation chooses to pull an excluded constructor path into scope, add it through the same request-options normalizer and back it with a representative existing test seam rather than a one-off harness.

## 2026-04-04 - North Star expanded to require `openai-codex`

Context

The user clarified that the feature is not acceptable as direct-OpenAI-only support. The canonical plan now has to treat Hermes `openai-codex` as part of the required OpenAI support surface, while still keeping non-OpenAI providers untouched and avoiding generalized architectural complexity. Current repo evidence shows the branch explicitly omits `service_tier` on Codex today.

Options

- Leave the architecture unchanged and keep `openai-codex` as an explicit omit path, while only renaming the docs.
- Expand the support boundary in the plan but defer the Codex path to a later follow-up.
- Reopen the relevant phases and make `openai-codex` a first-class required route in the North Star, target architecture, verification strategy, and PR framing.

Decision

Use the third option. The canonical plan now requires `service_tier` support on both direct OpenAI API-key routes and Hermes `openai-codex`. Earlier direct-only decisions remain historical context, but they are superseded by this scope expansion for all future implementation and audit work.

Consequences

- The previous implementation audit is no longer authoritative as “code complete”; Phases 1, 3, and 4 are reopened.
- The canonical compatibility seam can no longer hard-code `openai-codex` as unsupported.
- Main Responses dispatch, auxiliary Codex dispatch, representative tests, manual `hermes` checks, and PR framing must all cover Codex explicitly.
- Non-OpenAI providers remain out of scope, and the plan still rejects generalized multi-provider abstractions or undocumented proxy support.
- Because the public OpenAI docs do not document the ChatGPT Codex backend contract, Codex support now requires empirical Hermes runtime validation rather than analogies from the direct API docs alone.

Follow-ups

- Rework the canonical request-options normalizer and OpenAI request-shape helper boundary so Codex is a supported route, not a blanket omit case.
- Replace omission-based Codex tests with support-path tests wherever the required scope now demands include behavior.
- Run live manual verification against both direct OpenAI and `openai-codex` before treating the patch as PR-ready.

## 2026-04-04 - Deep-dive pass 4 narrowed the remaining Codex delta

Context

After reopening the North Star to require `openai-codex`, one more deep-dive pass was needed to separate true remaining Codex work from generic `request_options` plumbing that had already landed on the branch. The risk was overstating the implementation delta, reopening broad propagation work unnecessarily, or widening the runtime contract just to classify Codex.

Options

- Treat Codex support as evidence that the whole request-options propagation plan is still incomplete and reopen every constructor, cache, and restore seam.
- Widen the runtime contract with auth-store metadata such as `auth_mode` so Codex classification can depend on credential provenance.
- Keep the already-landed propagation seams, classify Codex support from route facts at the canonical normalizer boundary, and narrow the remaining work to Codex include behavior plus representative test flips.

Decision

Use the third option. The branch already has the generic `request_options` seam across smart routing, CLI, gateway, cron, ACP, API server, delegation, `AIAgent`, and the nested review helper. The remaining Codex work is therefore narrower: replace the blanket `openai-codex` exclusion in the canonical normalizer with a route-fact-based predicate on `provider=="openai-codex"`, `api_mode=="codex_responses"`, and the known `chatgpt.com/backend-api/codex` backend; stop the shared OpenAI helper boundary from independently rejecting or stripping Codex; and flip the representative Codex omission tests into support-path assertions. Auth-store facts such as `auth_mode="chatgpt"` stay as supporting repo evidence, but they do not widen the runtime transport contract for this feature.

Consequences

- Sections 4 through 6 now describe the current branch truth more accurately: propagation is largely in place, while Codex support is blocked by classification and payload omission rather than missing plumbing.
- The canonical support decision remains centralized in [hermes_cli/runtime_provider.py](/Users/aelaguiz/workspace/hermes-agent/hermes_cli/runtime_provider.py); [agent/auxiliary_client.py](/Users/aelaguiz/workspace/hermes-agent/agent/auxiliary_client.py) should own payload injection and token-key choice, not a second Codex support matrix.
- The plan no longer implies that ACP, cron, delegated children, gateway helper agents, or the nested review agent need new structural work just to participate in Codex support.
- The most valuable implementation and audit signal now comes from flipping the existing Codex omission assertions in runtime-provider, main Responses, and auxiliary-client owner suites.

Follow-ups

- Keep the remaining implementation and audit work focused on the Codex normalizer predicate, main/aux payload inclusion, and the owner-level tests that currently assert omission.
- Preserve the existing request-options propagation seams as-is unless implementation evidence finds a real missed constructor or restore path.

## 2026-04-04 - Phase plan refreshed after Codex deep-dive pass 4

Context

The reopened Codex scope made the old Section 7 too broad. It still read as if large parts of the stack needed fresh request-options plumbing even though the branch already carries that seam through smart routing, CLI, gateway, cron, ACP, API server, delegation, and nested review flow. That made the execution checklist less honest and less useful.

Options

- Leave Section 7 as a broad reopened checklist and let implementers rediscover which items are already landed.
- Collapse the whole plan into one Codex-only phase and assume existing propagation needs no preservation checks.
- Keep the already-landed propagation phase visible as preservation-only, narrow ship-blocking work to the Codex-critical foundation and request-path deltas, and tighten Phase 4 around representative owner suites instead of generic ceremony.

Decision

Use the third option. Section 7 now treats propagation as complete unless implementation finds a real gap, keeps Phase 1 focused on config persistence plus canonical Codex classification, keeps Phase 3 focused on main and auxiliary Codex request-path convergence, and sharpens Phase 4 around useful owner-level tests, full-suite rerun, manual direct-and-Codex `hermes` exercise, and PR-readiness follow-through.

Consequences

- The authoritative execution checklist now matches the real remaining work instead of restating already-landed plumbing as open architecture.
- Future implementation should flip existing omission assertions and preserve the current propagation seam, not reopen broad routing churn without evidence.
- Audit work can now judge completeness against a tighter, more maintainable checklist that better matches Hermes’ existing testing patterns.

Follow-ups

- Use the refreshed Section 7 as the only execution checklist for the reopened Codex scope.
- If implementation discovers a real propagation miss, reopen only the affected preservation surface rather than broadening the plan again.
