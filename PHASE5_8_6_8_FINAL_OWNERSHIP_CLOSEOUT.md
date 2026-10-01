# Phase 5.8.6.8 — Final consumer ownership and regression closeout

Base: `f5a5a0f7f8` (5.8.6.7 web/desktop audit).
Branch: `refactor/phase5-8-runtime-consumers`.

## Disposition

**Phase 5.8.6 consumer hard cut is complete within the stated ownership
boundary.** TUI, ACP and dashboard mutation controllers no longer import the
CLI model-switch selection/router or maintain shadow provider/model authority.
The existing gateway session and turn-runtime hard cut remains intact.
The browser and desktop clients dispatch choices to those backend owners
without synthesizing their own authoritative provider/model routes.

| Consumer | Final selection and routing owner | Application responsibilities |
| --- | --- | --- |
| TUI startup/rehydration | `models.selection`, `providers.routing`, TUI startup route | Profile/session state, resolved credential handoff |
| TUI live switch/config/picker | `tui_gateway.model_switch_resolution` over canonical domains | One-turn/global persistence, warnings, same shared inventory |
| ACP catalogue | `models.catalog_configured`, `models.catalog_endpoint`, shared inventory | Profile-scoped keys, explicit allowlists, TLS policy, projection |
| ACP session switch | `acp_adapter.model_switch_resolution` over canonical domains | Live-session transaction, MCP toolset continuity, strict persist |
| Dashboard and profile editor | `application_dashboard_model_selection` and `application_model_switch_persistence` | Config lock/write, credential pointers, confirmation |
| Browser/desktop/onboarding | Gateway RPC and profile-scoped dashboard REST | Presentation, owner-routed requests, confirmed user selections |
| Cross-surface defaults | `application_model_selection_defaults` over `models.selection` | Account/tier and cached-catalogue fact acquisition |

All shared command parsing, warnings, model persistence and public default
selection reside in application-owned modules. Model identity, configured
provider matching, catalogue grammar and wire/API-mode routing have canonical
`models/` and `providers/` owners. Removed CLI default-selection and shared
gateway-private helper modules have no replacement compatibility façades.

## Final architecture gates

Added `tests/models/test_phase_5_8_6_final_boundary.py`:

1. Walk all TUI, ACP and dashboard handler imports: reject CLI model-switch,
   selection, provider, model-validation/catalogue semantic imports.
2. Pin **exactly** the remaining shared-inventory and runtime-provider imports
   into those consumers. New upward dependencies fail the gate instead of
   silently expanding the exception list.
3. Verify dashboard mutation uses the application selection and shared
   persistence owners, not a CLI switch.
4. Verify key lower-domain selection/routing/catalogue modules do not import
   application or CLI code, and shared application policies do not import
   gateway/TUI/ACP session-specific code.
5. Verify the former CLI default-selection module is absent and its application
   replacement delegates actual policy to `models.selection`.

Earlier 5.8.6.1–5.8.6.7 tests also retain surface-specific guards for
profile scope, alias credential isolation, API-mode recalculation, native
Ollama empty catalogues, confirmation, session queue ordering and persistence.

## Final targeted verification

All suites below are disjoint; **292 passed**, **2 deselected**:

| Suite | Result |
| --- | ---: |
| TUI switch/config/profile tests | 58 passed |
| ACP resolver/catalogue and switch tests | 46 passed |
| Dashboard/defaults and web/desktop audit | 47 passed |
| Canonical model, gateway and endpoint-query tests | 51 passed |
| New 5.8.6.8 static ownership gates | 5 passed |
| Additional TUI startup/picker/session smoke | 25 passed |
| Additional ACP session/persistence smoke | 37 passed, 2 deselected |
| Additional gateway picker and model-options smoke | 23 passed |

The first ACP attempt was run with pytest plugin autoload explicitly
disabled and stopped when it reached an async test. Re-running with normal
plugins passed all 46 ACP switch/catalogue tests. The two deselections in the
ACP session smoke are previously documented Windows symlink tests; they were
not modified. Ruff, targeted Python compilation and staged Git diff checks
form the final commit gate.

The independent frontend TypeScript suites are not claimed as passed:
this Windows worktree has no root, web or desktop `node_modules` installed.
Existing frontend code was audited without changes; backend contracts and
static dispatch assertions passed.

## Explicit remaining owner work — not silently called closed

- **5.8.7 provider/plugin/environment policy** owns removal of
  `hermes_cli.inventory`'s internal
  `hermes_cli.model_switch.list_authenticated_providers` /
  `hermes_cli.model_switch_providers` discovery dependency, including the
  fact that the current recommended-default GET can lazily persist discovered
  custom-provider catalogue rows. Do not introduce a dashboard/TUI-specific
  second provider cache. It must also consolidate bare-custom durable
  identity recovery currently read from
  `hermes_cli.runtime_provider.canonical_custom_identity` in the TUI picker
  and session restore, plus the TUI session-guard's legacy
  `resolve_requested_provider` configuration projection. These precise
  imports are pinned by the new regression gate.
- **5.9 CLI legacy removal** owns any remaining CLI-hosted
  catalogue/validation/presentation leaves after 5.8.7. Shared application
  validation still calls the legacy
  `hermes_cli.models_validate.validate_requested_model` leaf; it is not
  allowed to re-own selection or routing.
- **Phase 6 credential ownership** retains current
  `hermes_cli.runtime_provider.resolve_runtime_provider` and dashboard's
  credential-availability checks, OAuth, key resolution and credential pools.
- **ACP persistence caveat** retained from 5.8.6.5: a failed persist restores
  in-memory agent/model but a partially completed underlying database write
  is not guaranteed to roll back as one physical transaction.
- An unrelated legacy Nous auxiliary test still imports the deleted
  `hermes_cli.model_selection_auxiliary` module. No compatibility shim was
  added; its repair belongs to the auxiliary test migration, not this
  web/ACP/TUI consumer cut.

Next: **Phase 5.8.7** — provider/plugin discovery and environment policy.
