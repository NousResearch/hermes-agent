# Hermes Phase 5.8.7.6 — Final scoped ownership and regression closeout

Branch: `refactor/phase5-8-runtime-consumers`.
Base: `a5f1d14271` (Phase 5.8.7.5).
This file closes the provider/plugin and environment-consumer scope established
in `PHASE5_8_7_1_PLUGIN_ENV_DEPENDENCY_INVENTORY.md`. It does not claim a
repository-wide plugin refactor.

## Final owner map and changes

| Concern | One authoritative owner | Consumers / disposition |
| --- | --- | --- |
| Provider identity, labels, configured custom identity and endpoint/API-mode policy | `providers.identity`, `providers.configured`, `providers.routing`, `providers.environment` | Plugins and TUI/ACP consume resolved, caller-supplied facts |
| Anthropic catalogue request/pagination/OAuth retry | Anthropic provider profile | The application passes guarded credential/HTTP context |
| DeepInfra full catalogue and tag projections | `models.catalog_deepinfra` | Chat, image, video, audio and pricing consume one key/endpoint/profile-scoped observation |
| Nous recommendation catalogue | `models.catalog_nous_recommendations` + Nous profile request | Application supplies scoped Portal/tier; pure recommendation extraction stays in `providers.nous_recommendations` |
| Copilot/OpenRouter reasoning capabilities | `models.metadata.github` and `models.metadata.reasoning` | Provider plugins retain wire-specific serialization |
| Shared provider discovery and its caches | `application_provider_discovery` over canonical model/provider queries | Shared CLI, dashboard, TUI and ACP inventory; observations cannot persist configuration |
| Deliberate custom catalogue saving | `application_discovered_catalog_persistence` | Explicit named-custom setup only; preserves curated per-model metadata |
| Provider grouping and selection confirmations | `application_provider_groups` and `application_model_selection_guards` | CLI and platform UI provide formatting/confirmation only |
| Scoped endpoint/secret fact acquisition | `application_provider_environment` and `application_provider_secret_inputs` | Profile precedence; actual key acquisition remains Phase 6 |

**New in 5.8.7.6:** Reconciled the two otherwise-independent
CLI/application model-selection warning registries. Added the CLI-required
`selection_warnings` and `combined_message` APIs to the existing
application owner, preserving cost → data-policy → context-cache ordering,
profile-specific threshold, same-model/unknown-size suppression and
per-guard exception isolation. CLI startup, interactive model switching and
the model picker now consume that one policy. Deleted
`hermes_cli.model_selection_guards.py`; no replacement forwarding facade.

Router previously fell back from scoped dotenv key lookup to
`os.environ`, which could borrow a different profile's token on a
scoped miss. Its documented `RAMP_ROUTER_API_KEY` / `ROUTER_API_KEY`
lookup now uses the existing `application_provider_secret_inputs.scoped_key_env`
for both names, preserving the active scope's authoritative misses.

## Current scoped static and lazy import audit

The updated executable boundary gate examines **69 current scoped Python files**:
all bundled model-provider profiles, the DeepInfra image/video consumers,
Discord, Telegram, Feishu, plus the Slack/Matrix label consumers found in
5.8.7.4. It inspects AST `Import` and `ImportFrom` nodes and literal
`__import__` / `importlib.import_module` calls. No scoped target imports
CLI-owned model selection, model catalogue, provider identity,
provider grouping, discovery or the bundled CLI runtime resolver.

The former CLI discovery, grouping and warning owner modules are absent.
`hermes_cli.model_switch` retains its **previously existing external
plugin compatibility pointer**, now directed to
`application_provider_discovery.list_picker_providers`; there is no
second production implementation. Inventory/discovery is read-only,
including the dashboard's recommended-default GET. Feishu's dynamic
`hermes_cli.env_loader` call remains permitted application
configuration rather than a model/provider semantic dependency.

Auth imports in the checked model-provider profiles are **exact**:
Actual and OpenCode Go/Zen use
`hermes_cli.auth.resolve_api_key_provider_credentials`; Copilot ACP uses
`hermes_cli.auth.resolve_external_process_provider_credentials`.
Anthropic guarded HTTP/OAuth context, the scoped DeepInfra private key,
the Nous account-state/tier lookup, scoped Router credentials and the
shared inventory's availability checks are still application/credential
responsibilities to hand off precisely to **Phase 6**.
No wildcard CLI model/provider import exceptions were added.

## Cross-surface verification

Disjoint primary test-file groups in this closeout:

| Regression group | Result |
| --- | ---: |
| 5.8.7.2–5.8.7.5 retained ownership and catalogue tests | 31 passed |
| Final 5.8.7.6 source boundary, Router and provider-catalogue checks | 32 passed |
| Shared inventory, cold picker, CLI warnings and persistence | 41 passed |
| Discord, Telegram, Slack, Feishu and display grouping | 54 passed |
| TUI/ACP legacy-boundary and canonical selection integration | 21 passed |
| **Total across these disjoint groups** | **179 passed** |

The pre-change standalone warning/persistence run (22 passed) overlaps
the 41-test combined inventory/CLI group and is **not** counted twice.
Supplementary tests, disjoint from the five primary groups: custom-provider
discovery plus scoped credential regressions **16 passed**; environment
reference/expansion/card and pricing-cache regressions **22 passed**.
The latter run initially identified a stale test patch targeting the
deleted CLI discovery module; correcting the fixture restored **22/22**
without changing production behaviour. The strengthened Router
no-cross-profile-key case passed as part of the 7-test final boundary rerun.
**217 selected tests passed across the disjoint primary and supplementary
groups.** Ruff, Python compilation and staged Git diff checks
form the final commit gate. No Arcana query, full CBM scan, frontend
dependency install or whole repository test suite is claimed.

## Explicit residual ownership — do not misclassify as closed

- **Phase 6 credentials:** scoped secrets, OAuth tokens, credential pools,
  subprocess launch/auth and provider-specific API-key resolution, with
  the exact plugin/import exceptions above. A credential accessor must
  not resume returning an authoritative model route.
- **Phase 5.9 legacy leaves:** still-used CLI-hosted catalogue,
  validation, status and external compatibility leaves can be
  removed only after their consumers are migrated. The remaining
  CLI-internal `cli_model_switch_mixin.canonical_custom_identity` use
  is not a TUI/ACP/dashboard reverse dependency.
- **Additional adjacent discovery outside the original audit scope:**
  `plugins/image_gen/openai/__init__.py` still obtains a named endpoint
  through `hermes_cli.runtime_provider._get_named_custom_provider`;
  `plugins/image_gen/openrouter/__init__.py` and
  `plugins/video_gen/openrouter/__init__.py` still obtain bundled
  endpoint/credential facts from `resolve_runtime_provider`.
  These need an explicit later media-plugin consumer cut plus Phase 6
  credential handoff. They are **not** authorized by the narrow Phase 6
  scoped-plugin allowlist and **not** represented as completed here.
- CLI configuration, onboarding, plugin lifecycle, external subprocess
  utilities and Feishu environment loading remain application integration
  rather than hidden provider/model semantic owners.
- Generated `.lexicon/` source snapshots can contain pre-migration
  strings; they are not tracked current executable sources or evidence
  of active consumer imports.

The original phase inventory retains its historical descriptions;
its appended final-reconciliation section links each phase's actual
ownership closeout instead of presenting old source-line numbers as current.
