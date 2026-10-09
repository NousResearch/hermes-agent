# Phase 5.8 Runtime Consumer Baseline

Base architecture: Phase 5.7 closed at `15e57836d2`.

Phase 5.8.1 is a behavior/dependency baseline only. It does not move production
ownership. Phase 5.8 will hard-cut runtime consumers away from CLI-owned
provider/model helpers and onto the canonical `providers/` and `models/`
domains established in Phases 5.1-5.7.

The baseline was taken from the fresh Lexicon/Arcana snapshot
`sha256:657462de78db0b391dd0fc65b8f2bab81915737642d53a35b2fff2b2190a7c5e`
for `15e57836d2`. Arcana is registered to that snapshot.

## Classification

Every upward provider/model dependency is assigned exactly one primary class:

- **Provider identity/registry** — provider identity, aliases, labels, declared
  provider families and provider registration/discovery facts.
- **Model identity/catalogue** — model membership, static/live catalogues,
  provider/model inventory and catalogue lifecycle/cache state.
- **Metadata/capability** — context windows, reasoning, fast-mode, pricing/free
  classification and provider/model capability facts.
- **Routing** — endpoint/API-mode/runtime-kind interpretation and route/model
  normalization.
- **Selection** — policy choosing a canonical provider/model candidate.
- **Application/presentation** — command parsing, warnings, display text,
  picker presentation and live application orchestration.
- **Credential/persistence** — auth-visible availability and persisted selection
  mechanics. Credential ownership itself remains Phase 6.

The destination column names the final owner, not an interim forwarding shim.
Where the existing implementation still lives under `hermes_cli`, Phase 5.8.2
must move or expose that single implementation below the CLI boundary rather than
copying it.

## Audit scope and counts

The Phase 5.8.1 audit scans executable Python imports plus dynamic module-string
references under:

`agent/`, `gateway/`, `tui_gateway/`, `acp_adapter/`, and `plugins/`.

| Root | References | Static imports | Dynamic references | Files |
| --- | ---: | ---: | ---: | ---: |
| `agent/` | 32 | 30 | 2 | 17 |
| `gateway/` | 21 | 21 | 0 | 9 |
| `tui_gateway/` | 18 | 17 | 1 | 8 |
| `acp_adapter/` | 6 | 6 | 0 | 2 |
| `plugins/` | 13 | 13 | 0 | 10 |
| **Total** | **90** | **87** | **3** | **46** |

The scanned upward namespaces are `hermes_cli.models*`,
`hermes_cli.model_switch*`, `hermes_cli.model_selection_*`,
`hermes_cli.models_validate`, and the runtime catalogue lifecycle surface
`hermes_cli.model_catalog`.

## Runtime dependency manifest

The exhaustive per-consumer manifest is stored in
`PHASE5_8_RUNTIME_DEPENDENCY_MANIFEST.md`. It records all **90** scoped upward
references across **46** runtime files with a primary responsibility class and a
final-owner direction. This baseline remains the source of the classification and
migration rules; the manifest is the execution checklist for 5.8.2-5.8.7.

## Major runtime clusters

The manifest reduces to six migration clusters:

1. **Catalogue access** — `static_provider_model_ids`, live picker discovery,
   catalogue refresh/cache and provider/model grouping still require CLI-owned
   catalogue surfaces. Phase 5.8.2 must expose the existing single catalogue
   authority below the CLI boundary before consumers move.
2. **Startup/effective selection** — gateway/TUI still call
   `resolve_effective_model`, `resolve_startup_model_route`, and CLI
   selection-default adapters. These must become caller-supplied facts into
   `models.selection*`, followed by `providers.routing`.
3. **Live switch application** — gateway, TUI and ACP still call the CLI
   `switch_model` coordinator. Each application surface must own its own live
   state mutation while consuming canonical selection/routing.
4. **Capability leakage** — reasoning, context, fast-mode and provider/model
   capability helpers remain exposed from `hermes_cli.models*`; they belong
   under `models.metadata` or provider-specific capability sources.
5. **Provider/routing leakage** — OpenCode normalization/family, API mode,
   provider labels and provider lists remain exposed through CLI modules; these
   resolve to `providers.identity`, `providers.registry`,
   `providers.model_normalizers` and `providers.routing`.
6. **Presentation/persistence leakage** — warning, command parsing, display and
   persisted-selection helpers are application responsibilities and must move to
   the consuming application or a non-domain shared application helper rather
   than into `models/` or `providers/`.

## Boundary decisions locked for 5.8

1. `hermes_cli` is an application consumer, not a runtime provider/model
   service.
2. Phase 5.8 does not create a runtime model manager, compatibility registry,
   forwarding facade, dual read or synchronized old/new state.
3. Catalogue implementations that still physically live under `hermes_cli`
   move/expose directly to their final lower-domain owner in 5.8.2. The old CLI
   surface is not retained for runtime consumers.
4. Configuration loading may remain where it currently lives, but provider/model
   interpretation after loading flows through canonical domains.
5. Selection chooses a canonical provider/model. Routing determines invocation
   semantics. Runtime/application code satisfies credentials and mutates live
   state. Those responsibilities remain distinct.
6. Credential acquisition, OAuth/token refresh, credential pools and secret
   persistence remain Phase 6. Phase 5.8 only removes provider/model semantic
   ownership from CLI runtime dependencies.
7. Presentation-only grouping, warnings, confirmation and display may remain
   application-owned provided they consume canonical facts and do not recreate
   provider/model truth.
8. Provider plugins may own provider-specific transport/discovery mechanics, but
   normalized provider/model facts have one canonical owner.

## Existing behavior lock

Phase 5.7 closed with the same inherited failure categories recorded before the
selection migration. Its final baseline rerun was:

```text
537 passed, 31 failed in 322.04s
```

The 31 deterministic failures are:

1. **Alias/provider detection — 1**
   - short alias `sonnet` resolves to `copilot` instead of the legacy test's
     expected `anthropic`.
2. **Custom-provider model parsing — 1**
   - comma-chain custom-provider model parsing.
3. **Nous silent-default policy — 2**
   - hidden-model default and unrestricted-org expectations.
4. **Auxiliary main-first routing — 4**
   - Copilot vision/text header cases and custom endpoint forwarding cases.
5. **Actual/ACI routing and setup — 23**
   - runtime transition matrix, provider/base-URL setup matrix and key-reload
     endpoint persistence.

The Phase 5.7 closeout also recorded a focused picker/runtime gate of
`48 passed, 4 inherited failures` caused by the existing gateway picker
`get_label` import defect, plus an inherited Copilot same-provider API-mode
failure outside selection ownership.

The Phase 5.8.1 rerun of the same representative baseline is recorded below.
No production or test-code change is part of 5.8.1, so any difference from the
Phase 5.7 closeout would block 5.8.2.

## Focused baseline command

```text
python -m pytest -q \
  tests/hermes_cli/test_models.py \
  tests/hermes_cli/test_model_alias_credentials.py \
  tests/hermes_cli/test_model_switch_custom_providers.py \
  tests/hermes_cli/test_cli_provider_resolution.py \
  tests/hermes_cli/test_nous_policy_surfaces.py \
  tests/hermes_cli/test_aux_picker_inventory.py \
  tests/agent/test_auxiliary_client.py \
  tests/agent/test_auxiliary_main_first.py \
  tests/agent/test_actual_auxiliary_routing.py \
  tests/acp_adapter/test_acp_dashboard_model_switch_validation.py
```

**Phase 5.8.1 rerun:** **537 passed, 31 failed in 269.73s**. The failure list
and category counts exactly match the Phase 5.7 closeout baseline; no new
Phase 5.8.1 failure category was introduced.

## Phase 5.8.1 acceptance rule

- The migration base is frozen at `15e57836d2`.
- Every upward dependency in the five scoped runtime roots is represented above
  with a primary responsibility class and final-owner direction.
- Existing representative behavior is rerun and inherited failures are recorded.
- No provider/model production ownership moves in this sub-phase.
- No compatibility layer is introduced.
- Phase 5.8.2 may close the lower-domain API gaps without reopening ownership
  discovery.

## Phase 5.8.2 closeout — runtime query ownership

Phase 5.8.2 closes the shared lower-domain query gaps needed by the later
consumer-migration sub-phases. It does not migrate the application coordinators
scheduled for 5.8.3-5.8.7.

Ownership changes:

- Static model catalogue policy moved from `hermes_cli.models_catalog_static`
  to `models.catalog_static`; the old owner is deleted.
- Display-only provider grouping moved separately to
  `hermes_cli.provider_groups`, keeping presentation out of the model domain.
- Codex curated/forward-compatible catalogue policy moved to
  `models.codex_catalog`; `hermes_cli.codex_models` retains live
  credential/runtime discovery and consumes the lower policy.
- models.dev disk-cache, ETag, validation and quarantine ownership moved to
  `models.models_dev_cache`; `agent.models_dev` retains network refresh and
  in-process lifecycle state.
- Fast-mode capability/override queries moved to
  `models.metadata.fast_mode`.
- Shared reasoning-effort clamping and Astra identity moved to
  `models.metadata.reasoning`; GitHub/Copilot reasoning capability queries
  moved to `models.metadata.github`.
- Pure Ollama catalogue classification moved to `models.catalog_local`;
  configuration lookup and live probing remain application/runtime mechanics.
- OpenCode family/model/base-URL interpretation is exposed from `providers`
  and runtime consumers no longer require the CLI model module for it.
- `tests/models/test_runtime_query_boundary.py` locks the lower modules
  against upward imports, duplicate CLI ownership, and restoration of the old
  static-catalogue owner.

Verification:

- Catalogue/Codex focused gate: **67 passed**.
- Fast-mode/reasoning/runtime focused gate: **93 passed**.
- models.dev cache gate: **76 passed**.
- Runtime-query/provider/ACP boundary gate: **25 passed**, 2 inherited warnings.
- Targeted `py_compile`, `ruff check`, and `git diff --check`: clean.
- Original Phase 5.8 representative baseline: **537 passed, 31 failed in
  214.95s**. The failure count and categories exactly match the 5.8.1 inherited
  baseline; no new failure category was introduced.

The remaining upward dependencies in the Phase 5.8 manifest are consumer
migration work for 5.8.3-5.8.7, not duplicate owners for the shared query
surfaces closed here.

## Phase 5.8.3 closeout — agent init and primary client lifecycle

Phase 5.8.3 hard-cuts the primary agent/runtime path away from CLI-owned
provider/model semantics. Configuration loading, authentication, persistence,
plugin dispatch, and other application mechanics remain application-owned;
provider/model interpretation now flows through the canonical lower domains.

Ownership changes:

- Canonical route URL normalization and Actual-route identity moved to
  `providers.route_identity`; primary runtime and client-lifecycle consumers
  no longer import those facts from `hermes_cli`.
- External-process runtime classification is owned by `providers.routing`.
  Agent initialization consumes that query directly rather than the CLI runtime
  backend helper.
- GitHub Copilot transport-header policy moved to `providers.github`.
  Authentication/token exchange remains in `hermes_cli.copilot_auth`.
- The Nous Hermes 3/4 suitability warning moved to
  `agent.model_warnings`, separating application warning/presentation policy
  from the CLI model-switch coordinator.
- GitHub Copilot live catalogue acquisition moved to
  `models.catalog_github`; account-scoped context-window interpretation moved
  to `models.metadata.github`.
- LM Studio reasoning-option and Ollama thinking-capability probes moved to
  `models.metadata.local`; runtime reasoning consumers no longer depend on
  `hermes_cli.models_local` for capability truth.
- The historical runtime identity `openai` now obtains direct-API endpoint and
  API-mode facts from the canonical `openai-api` provider profile without
  rewriting the runtime identity.
- Primary runtime consumers in `agent_init.py`,
  `agent_runtime_helpers.py`, `client_lifecycle.py`,
  `chat_completion_helpers.py`, `fast_mode.py`, `reasoning_params.py`,
  `model_metadata.py`, `models_dev.py`, `opencode_affinity.py`, and
  `agent/transports/*.py` have no dependency on
  `hermes_cli.models*`, `hermes_cli.model_switch*`,
  `hermes_cli.model_selection*`, `hermes_cli.models_validate`, or
  `hermes_cli.model_catalog`.
- No forwarding façade was retained for the newly moved GitHub catalogue/context
  surfaces; remaining application consumers import their lower owner directly.

Verification:

- Targeted `py_compile`, `ruff check`, and `git diff --check`: clean.
- Primary runtime ownership grep: zero scoped upward model-semantic references.
- Focused runtime/ownership gate before final façade cleanup:
  **137 passed, 2 skipped, 2 failed**. Both failures are the already-recorded
  Copilot cases in the inherited **Auxiliary main-first routing** category; no
  new focused failure remained.
- Provider-routing regression for historical `openai` identity:
  **11 passed** including the previously failing cross-provider switch case.

Phase 5.8.3 does not claim auxiliary runtime ownership; the inherited Copilot
auxiliary failures and the broader auxiliary migration remain Phase 5.8.4 work.

## Phase 5.8.4 closeout — auxiliary runtime hard cut

Phase 5.8.4 hard-cuts auxiliary model/provider semantics away from CLI-owned
selection and routing authority while keeping auxiliary orchestration,
credential acquisition, retries, health/quarantine, and transport construction
application-owned.

Ownership changes:

- Auxiliary model selection consumes `models.selection_auxiliary`; the old
  `hermes_cli.model_selection_auxiliary` owner is deleted.
- Configured-provider matching, custom-provider identity, direct-API aliases,
  and custom-resolution semantics are owned by `providers.configured`.
  `agent.configured_provider_resolution` only acquires already-loaded
  application config facts.
- Copilot auxiliary transport/header behavior consumes
  `providers.github.copilot_request_headers`; GitHub token acquisition remains
  application-owned.
- The main-session runtime route is authoritative for auxiliary main-first
  resolution. Provider/model/base URL/API mode facts are projected through
  canonical lower-domain routing rather than reconstructed from CLI helpers.
- Vision defaults/rejection are provider-profile facts; model image capability
  comes from `models.metadata`; vision-model precedence comes from
  `models.selection_auxiliary`.
- Nous auxiliary recommendation semantics are provider-owned via
  `providers.nous_recommendations`; the obsolete CLI recommendation selector is
  removed.
- Fallback route interpretation is centralized in `agent.fallback_routing`,
  which acquires application config facts then delegates provider/base/API-mode
  semantics to `providers.routing`. Main-agent and auxiliary fallback consumers
  share this route owner.
- Actual route protocol mandate is lower-owned by `providers.routing`.
- Auxiliary unhealthy-route identity uses canonical provider identity while
  custom routes remain endpoint-scoped.
- No auxiliary runtime consumer imports
  `hermes_cli.model_selection_auxiliary`, `hermes_cli.model_switch*`,
  `hermes_cli.model_selection*`, `hermes_cli.models_validate`,
  `hermes_cli.model_catalog`, or `hermes_cli.runtime_provider_custom`.

Phase 6 boundary retained:

- Credential acquisition, OAuth/token refresh, pools, secrets, and
  provider-specific auth/runtime assembly remain application-owned.
- `agent.auxiliary_client` has exactly two permitted
  `hermes_cli.runtime_provider` imports, frozen by the architecture gate:
  bare-custom runtime acquisition and Azure Foundry auth/runtime acquisition.
  These are application mechanics, not new semantic-owner surfaces, and are
  deferred to Phase 6 rather than hidden behind a forwarding façade.

Architecture gates:

- `tests/models/test_runtime_query_boundary.py` now locks the complete
  auxiliary semantic boundary, the final lower owners, canonical fallback
  routing, and the exact Phase-6 acquisition exceptions.
- The gate rejects reintroduction of CLI selection/custom-provider authority or
  new `runtime_provider` imports in the auxiliary runtime surface.

Verification:

- Final ownership gate: **14 passed**.
- Selection/capability/vision closeout set: **51 passed**.
- Main-first/custom/OpenCode/Copilot/Azure closeout set: **73 passed**.
- Fallback/routing/health/provider-parity closeout set: **102 passed**.
- Total focused 5.8.4 closeout verification: **240 passed**.
- `ruff check` and `git diff --check`: clean.

Phase 5.8.4 is closed. Gateway/session runtime migration remains Phase 5.8.5;
TUI/web/ACP remains 5.8.6; provider/plugin/environment policy remains 5.8.7.

## Phase 5.8.5 progress — gateway/session runtime hard cut

### 5.8.5.1 — gateway effective-model precedence

Gateway application precedence no longer depends on
`hermes_cli.model_switch.resolve_effective_model`.

- `gateway/model_resolution.py` now owns only the application-tier choice of the
  first non-empty model candidate. It deliberately does not interpret model
  identity, provider identity, catalogue membership, or invocation routing.
- `gateway/run_config_loaders.py` uses that gateway-owned precedence for
  channel override > global model selection.
- `gateway/platforms/api_server.py` uses the same gateway-owned precedence for
  advertised model naming and session override/session-row precedence.
- `tests/models/test_runtime_query_boundary.py` prevents these gateway
  consumers from regaining the CLI effective-model dependency.
- No compatibility wrapper or lower-domain duplicate authority was introduced.

Verification: gateway/API focused set **129 passed**; runtime ownership gate
**15 passed**; Ruff, `py_compile`, `git diff --check`, and the gateway
`resolve_effective_model` search are clean.

### 5.8.5.2 — session launch model resolution

`gateway/session_local_route.py` no longer delegates startup model interpretation
to `hermes_cli.model_switch.resolve_startup_model_route`.

- Gateway projects profile-local `model_aliases` / `model.aliases` configuration
  through `gateway/model_aliases.py`; this is application config/credential
  acquisition, not provider/model semantic ownership.
- Alias and qualified-model identity flows through
  `models.selection.select_explicit_model`.
- Invocation semantics flow through `providers.routing.resolve_invocation_route`.
- URL-bearing aliases retain the security invariant that their credential is
  resolved for the alias endpoint rather than borrowing a vendor-labelled key.
- Explicit provider launches still override an alias provider label without
  inheriting the alias credential.
- Configured provider/model syntax retains the configured request key needed for
  later credential acquisition while routing semantics are resolved canonically.
- Aggregator-native slash IDs are protected by the lower-owned static catalogue
  query `models.catalog_static.find_static_provider_model_id`; gateway carries no
  OpenRouter model table or model-membership heuristic.
- Plain model IDs and launches already carrying explicit endpoint/credential
  facts remain pass-through.

Verification: launch/session/runtime ownership set **33 passed, 1 skipped**;
Ruff, `py_compile`, `git diff --check`, and the scoped
`resolve_startup_model_route` / CLI model-switch search are clean for the
5.8.5.2 surface. Remaining gateway `hermes_cli.model_switch` imports belong to
5.8.5.3-5.8.5.6.

### 5.8.5.3 — session model mutation

`gateway/session_mutation_model.py` no longer delegates session changes to
`hermes_cli.model_switch.switch_model`.

- `gateway/session_model_facts.py` materializes caller-owned selection facts
  from frozen session config and lower-domain catalogue/provider declarations.
- `gateway/session_model_resolution.py` performs canonical
  `models.selection.select_explicit_model` followed by
  `providers.routing.resolve_invocation_route`.
- The only surviving upward dependency is the exact Phase 6 credential seam
  `hermes_cli.runtime_provider.resolve_runtime_provider`; it supplies
  credentials/runtime material and does not own the final model or invocation
  route.
- Session mutation still owns frozen-policy rewrite, restart-safe model/provider
  persistence, config-secret rebinding, and clearing an incompatible launch
  credential when provider identity changes.
- Canonical `api_mode` / `runtime_kind` are resolved for the live route but
  are not added as new durable session-policy authority.
- Session mutation no longer reconstructs the live runtime merely to copy its
  current API key.
- Added direct unit coverage for same-custom-provider routing, explicit provider
  switching, configured-provider credential keys, direct-alias endpoint/key
  isolation, and provider-change policy/credential clearing.
- Hardened the existing launch-route fixture so collection order cannot bind it
  to the developer profile instead of its temporary profile.

Verification: focused gateway/session ownership set **39 passed, 1 skipped**
(the existing end-to-end mutation case is Linux-only on this Windows worktree);
Ruff, `py_compile`, and `git diff --check` are clean. Scoped ownership guards
reject any return of `hermes_cli.model_switch` / CLI selection ownership in the
5.8.5.3 path and permit only the exact Phase 6 runtime-provider credential seam.

### 5.8.5.4 — gateway /model orchestration

`gateway/slash_commands_model.py` no longer delegates model switching, parsing,
persistence policy, confirmation guards, display formatting, or preflight
compression warning orchestration to `hermes_cli.model_switch` or its old guard
modules.

- `gateway/model_command_request.py` owns typed `/model` parsing and
  session/global/once persistence scope.
- `gateway/model_switch_resolution.py` consumes the shared gateway
  selection/routing seam from 5.8.5.3 and returns the application result shape.
- `gateway/model_switch_enrichment.py` adds validation, display/provider facts,
  request overrides, metadata, native-compaction capabilities, and warnings
  without taking over canonical selection or invocation routing.
- `gateway/model_switch_persistence.py` owns config write-through and stale
  route/context credential clearing.
- `gateway/model_switch_display.py` owns display normalization and asynchronous
  context-length presentation.
- `gateway/model_selection_guards.py` owns cost/data/context confirmation
  policy; `gateway/model_switch_preflight.py` owns the gateway-specific
  preflight-compression warning.
- Cached-agent swap/rollback, DB/session override persistence, one-turn restore,
  and global-config precedence remain in the gateway coordinator.
- The shared session resolver now returns acquired API-key material to the
  application coordinator while the Phase 6 credential seam remains
  `hermes_cli.runtime_provider.resolve_runtime_provider`.
- Picker/catalogue inventory and cache-refresh dependencies remain explicitly
  deferred to 5.8.5.6; no compatibility facade was introduced for the removed
  model-switch coordinator.

Verification: direct gateway ownership and architecture set **24 passed**;
broader model/session gateway regression set **157 passed, 1 skipped**. Ruff,
`py_compile`, ownership searches, and final diff checks are clean. All new
gateway refactor modules are at or below 200 lines.

### 5.8.5.5 — turn construction and cached-runtime rehydration

Turn/session runtime consumers no longer ask CLI-owned provider/model semantics to
interpret persisted routes, choose empty-model defaults, or normalize cached
agent model identity.

- `providers.route_identity.is_foreign_provider_endpoint` now owns the
  cross-provider endpoint identity query. The obsolete
  `hermes_cli.runtime_provider.is_foreign_provider_endpoint` owner is deleted.
  Gateway rehydration, CLI resume, and TUI resume all consume the lower owner.
- `models.catalog_static.static_provider_default_preference` exposes the
  offline cost-safe preference fact without leaking the private static catalogue
  tables upward.
- `gateway/model_runtime_facts.py` is the small application projection for
  turn construction: it feeds caller-owned catalogue facts into canonical
  `models.selection.select_default_model` and delegates model-ID normalization
  to `models.identity.normalize_model_id`.
- Gateway turn preparation, API agent construction, Feishu comment agents, and
  TUI startup all use that shared projection for an unconfigured provider/model.
- Fallback eviction no longer reconstructs model identity inline; it uses the
  same `normalize_runtime_model` projection before deciding whether a cached
  agent is stale.
- Persisted session override rehydration keeps model/provider/base URL
  application state in gateway/TUI code while foreign-endpoint interpretation
  is lower-owned.
- Credential acquisition, named-custom runtime construction, and runtime
  fallback remain the explicit Phase 6 mechanics seams.
- The cached preferred-default catalogue read remains an application fact
  acquisition seam through `hermes_cli.model_catalog`; catalogue ownership and
  picker/inventory cleanup remain explicitly deferred to 5.8.5.6.
- No compatibility facade or forwarding definition was left behind for the
  removed runtime-provider semantic query.

Verification: consolidated turn/cache/API/TUI/ownership regression set **239
passed** (62 existing aiohttp application-key warnings). Ruff, `py_compile`,
scoped ownership searches, and `git diff --check` are clean. New production
seam `gateway/model_runtime_facts.py` is 47 lines.

### 5.8.5.6 — catalogue, metadata, and picker ownership

Gateway catalogue/picker consumers no longer depend on CLI-owned model catalogue,
selection, or switch-provider semantics.

- `models/catalog_manifest.py` owns remote catalogue schema validation, config
  interpretation, provider block projection, curated IDs/descriptions, and
  default-model facts.
- `models/catalog_runtime.py` owns remote catalogue cache/fetch/SWR lifecycle,
  provider overrides, cached default lookup, curated provider reads, refresh
  cadence, and cache reset. It imports no application layer.
- `models/catalog_seed.py` owns checkout-to-cache seeding.
- `gateway/model_catalog_runtime.py` is the application adapter from gateway
  config/home paths into the lower catalogue service. The gateway watcher and
  default-model path consume this seam.
- `gateway/model_picker_inventory.py` owns the gateway projection of application
  inventory into `/model` rows. Picker/list presentation remains gateway-owned;
  credential/config acquisition remains application-owned pending Phase 6.
- `/model --refresh` now requests live inventory refresh through that projection
  rather than clearing `hermes_cli.models` caches directly.
- Existing CLI catalogue consumers were migrated to the same lower catalogue
  service, so the old `hermes_cli.model_catalog` module no longer owns live
  catalogue semantics.
- A deliberately tiny `hermes_cli/model_catalog.py` compatibility hook remains
  only because shipped pre-handoff updaters import
  `seed_cache_from_checkout` after pulling the new tree. Architecture tests
  constrain that module to exactly that one updater hook; it forwards only to
  `models.catalog_seed` and cannot expose runtime catalogue/query APIs.
- The picker regression where `hermes_cli.models.is_astra_model` was undefined
  is fixed by importing the lower reasoning-metadata owner.
- Targeted gateway/runtime consumers contain zero
  `hermes_cli.models*`, `hermes_cli.model_selection*`,
  `hermes_cli.model_catalog`, or `hermes_cli.model_switch_providers`
  semantic dependencies.

Verification: lower catalogue/picker core **9 passed**; migrated legacy
catalogue/SWR/profile/default set **31 passed**; architecture boundary **20
passed**; gateway catalogue/picker behavior **40 passed**. Ruff and
`git diff --check` are clean. The two new lower production modules remain
within the small-file boundary (`catalog_runtime.py` 199 lines).

5.8.5.7 closed at `3f3ddeeefc`. Phase 5.8.6.1 TUI startup and rehydration
closeout is recorded in `PHASE5_8_6_1_TUI_STARTUP_CLOSEOUT.md`; 5.8.6.2
(TUI model-switch orchestration) is recorded in
`PHASE5_8_6_2_TUI_SWITCH_CLOSEOUT.md`; 5.8.6.3 (TUI picker/config and metadata) is documented in
`PHASE5_8_6_3_TUI_CONFIG_CLOSEOUT.md`; 5.8.6.4 (ACP catalogue hard cut) is recorded in
`PHASE5_8_6_4_ACP_CATALOG_CLOSEOUT.md`; 5.8.6.5 (ACP session switch) is documented in
`PHASE5_8_6_5_ACP_SWITCH_CLOSEOUT.md`; 5.8.6.6 (dashboard model assignment) is recorded in
`PHASE5_8_6_6_DASHBOARD_ASSIGNMENT_CLOSEOUT.md`; 5.8.6.7 (web/desktop audit) is recorded in
`PHASE5_8_6_7_WEB_DESKTOP_AUDIT_CLOSEOUT.md`; 5.8.6.8 (final ownership and regression closeout)
is recorded in `PHASE5_8_6_8_FINAL_OWNERSHIP_CLOSEOUT.md`. Phase 5.8.6 is closed;
5.8.7 (provider/plugin discovery and environment policy) follows.
