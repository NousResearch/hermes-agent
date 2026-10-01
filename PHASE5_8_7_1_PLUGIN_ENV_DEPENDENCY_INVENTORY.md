# Phase 5.8.7.1 — Provider/plugin and environment dependency inventory

Base: `7140f08afb` (5.8.6.8 consumer closeout).
Branch: `refactor/phase5-8-runtime-consumers`.
Status: inventory complete; no production ownership moved in 5.8.7.1.

## Scope and method

Reconciled the 5.8.1 90-reference manifest (including its original 13 plugin references) against
the current source and the explicit 5.8.6.8 carryovers. Audited imports in 67 tracked Python
files across `plugins/model-providers/`, the DeepInfra image/video plugins, and the Discord,
Feishu, and Telegram platform trees. The audit covers AST static imports and literal
`__import__` references; direct source searches checked lazy imports, provider environment
declarations, and the CLI inventory/TUI callers. Counts from 5.8.1 describe its **historical**
snapshot, not the current tree. Arcana was attempted but exited with SIGTERM, so structural
claims below are from current source inspection rather than graph evidence.

Primary classes retain `PHASE5_8_RUNTIME_CONSUMER_BASELINE.md` terminology:
provider identity/registry; model identity/catalogue; metadata/capability; routing;
selection; application/presentation; credential/persistence. A row's destination is its
**final owner**, never an interim CLI forwarding shim.

## 5.8.7.2 — Active catalogue/discovery dependencies

| Current caller | Exact upward dependency / responsibility | Final owner and cut |
| --- | --- | --- |
| `plugins/model-providers/anthropic/__init__.py:31` | `hermes_cli.models` private URL, cursor, page-limit helpers; catalogue mechanics | Anthropic profile/plugin owns HTTP, pagination and response parsing; normal catalogue entry goes through existing `ProviderProfile.fetch_models` |
| `plugins/model-providers/deepinfra/__init__.py:41` | `_fetch_deepinfra_models_by_tag("chat")`; vision discovery | One DeepInfra catalogue service under `models/`; provider plugin supplies provider-specific fetch/parse |
| `plugins/image_gen/deepinfra/__init__.py:25` | Same CLI tagged catalogue for image models | Same service; image plugin owns image picker/generation only |
| `plugins/image_gen/deepinfra/__init__.py:116` | `hermes_cli.models.deepinfra_base_url` | `ProviderProfile` endpoint declaration plus `providers.routing`, with caller-supplied config override |
| `plugins/video_gen/deepinfra/__init__.py:30` | Same CLI tagged catalogue for video models | Same DeepInfra service; video plugin owns video-specific implementation |
| `hermes_cli/models_pricing.py:593` | CLI-to-CLI tagged catalogue call for DeepInfra pricing | Single DeepInfra catalogue observation exposed to pricing; do not create a pricing-specific fetch/cache |
| `hermes_cli/models.py` | Owns DeepInfra raw cache, negative cache, surface-tag filtering and Anthropic pagination | Migrate each implementation **once** to the destinations above and eliminate old private semantic helpers after caller cutover |

DeepInfra source semantics to preserve: per-endpoint and key-isolated catalogue observations,
60-second negative cache, tagged chat/image/video/etc. filtering, untagged chat fallback,
missing/unserved stubs and force-refresh. Anthropic must retain its OAuth-specific catalogue
behaviour where that source is currently implemented in `hermes_cli.models`; a simple
API-key-only profile fetch does not replace that path.

## 5.8.7.3 — Metadata, recommendations and other profile imports

| Current caller | Disposition | Final owner / exact exception |
| --- | --- | --- |
| `plugins/model-providers/nous/__init__.py:28` | **Migrate** `_resolve_nous_portal_url`, `check_nous_free_tier`, `fetch_nous_recommended_models` imports | Provider-specific Portal source and shared catalogue lifecycle; `providers.nous_recommendations` owns pure recommendation extraction; `models.selection` chooses; application/auth passes account-tier facts |
| `plugins/model-providers/copilot/__init__.py` | **Already migrated** to `models.metadata.github` | Verify reasoning levels, clamping and wire formatting; do not move working code |
| `plugins/model-providers/openrouter/__init__.py` | **Already migrated** to `models.metadata.reasoning` | Verify mandatory reasoning, unknown capabilities and omission; provider retains wire serialization |
| `plugins/model-providers/actual/__init__.py:68` | **Split** combined CLI auth + base-URL normalization import | Move Actual-specific `/v1` endpoint semantics from `hermes_cli.auth.normalize_actual_base_url` to `providers.route_identity` (its existing generic normalization is insufficient on its own); `resolve_api_key_provider_credentials` remains narrowly Phase 6 |
| `plugins/model-providers/copilot-acp/__init__.py:38` | **Phase 6 only**: external process credential/launch resolution | Existing auth application seam; keep ACP source-specific model enumeration in plugin |
| `plugins/model-providers/opencode-zen/__init__.py:82` | **Audit split**: CLI runtime resolver used to obtain an API token for account-usage fetch | Preserve credential sourcing for Phase 6; no plugin-owned route decision from the returned CLI bundle |
| `plugins/model-providers/router/__init__.py:69,86` | **Split** scoped endpoint and token lookup | Non-secret base-URL declaration through profile/canonical routing in 5.8.7.5; secret retrieval and fallback remain Phase 6 |

The existing Nous ten-minute in-memory/disk recommendation cache must have **one** final
lifecycle owner. Its account-dependent recommendation is an input to selection, not a
new provider-owned selection algorithm.

## 5.8.7.4 — Platform/plugin consumers

| Current caller | Exact dependency | Disposition and final owner |
| --- | --- | --- |
| `plugins/platforms/discord/adapter.py:5647,6620` | `hermes_cli.providers.get_label` | Replace with `providers.identity.get_provider_label`; Discord owns formatting |
| `plugins/platforms/discord/adapter.py:6558` | CLI `combined_selection_warning` | Existing `application_model_selection_guards` is the shared application-owned confirmation policy |
| `plugins/platforms/telegram/adapter.py:4318` | CLI provider label | Canonical provider identity/label; Telegram renders it |
| `plugins/platforms/telegram/adapter.py:4437,4596` | `hermes_cli.provider_groups` | Display-only shared application grouping, not `providers/` identity or a second registry |
| `plugins/platforms/telegram/adapter.py:4579` | CLI selection warning | Existing `application_model_selection_guards`; preserve confirmation |
| `plugins/platforms/feishu/feishu_comment.py` | Former CLI default-selector | **Already migrated** to `gateway.model_runtime_facts.provider_default_model`; verify only |

Discord/Telegram also use CLI commands, configuration and plugin-lifecycle functions.
Those are application integration, **not** evidence of duplicated provider/model authority.
`feishu_comment_rules.py:254` dynamically imports `hermes_cli.env_loader` for environment
loading; classify as application configuration (not provider routing) and check the existing
profile-scope boundary, rather than banning it as a model dependency.

## 5.8.7.5 — Non-secret environment and late 5.8.6 carryovers

| Current source / caller | Remaining problem | Final owner |
| --- | --- | --- |
| `ProviderProfile.env_vars` / `base_url_env_var`; `hermes_cli/auth.py:_provider_env_base_url` | Declaration exists, but config/auth paths independently assemble effective endpoints | App loads scoped values; `providers.configured` and `providers.routing` interpret identity, endpoint and API-mode facts; auth owns only credentials |
| `hermes_cli/model_switch_providers.py` / `hermes_cli/inventory.py:88` | Shared inventory calls CLI switch-provider listing, which owns discovery and can persist discovered custom models | Single application-owned inventory over canonical registry/catalogue. Explicit configuration writes stay application-owned; normal read-only picker GET must not gain hidden persistence |
| `hermes_cli/web_routers/models.py` | Documented read-time discovered-catalogue persistence via the shared inventory chain | Audit and separate observation from deliberate save; preserve refresh and cache-only GET behaviour |
| `tui_gateway/methods_complete_helpers.py:134` | Picker repairs bare `custom` through CLI `canonical_custom_identity` | `providers.configured.configured_custom_identity` with caller-supplied config facts; application gathers those facts |
| `tui_gateway/session_workdir.py:332` | Session-row restoration uses the same CLI custom-identity helper | Same canonical configured identity; preserve persisted row and resume semantics |
| `tui_gateway/methods_session_model_guard.py:21` | CLI `resolve_requested_provider` supplies an implicit configured provider | Application loads scoped config, canonical `providers/` resolves it; existing `models.selection_conflict` still decides conflict |
| `hermes_cli/provider_catalog.py` / `web_routers/config_env.py` | Already derive provider cards and endpoint-var metadata from registered profiles | Verify correct non-secret/secret separation and shared-key card grouping; do not replace presentation code needlessly |

Retain configured endpoints, user-selected `providers:` precedence, legacy
`custom_providers:` read compatibility where it represents actual persisted data,
profile-local environment, explicit endpoint overrides and provider route mandates.
Never infer that an env var is an endpoint by name suffix; `base_url_env_var` declares it.
Do not conflate the terminal-environment-provider plugin system with inference-provider
environment policy.

## Exact Phase 6 exceptions — not wildcard CLI allowances

- Anthropic OAuth/token and guarded credentialed HTTP context; the plugin may perform
  provider-specific HTTP, but catalogue semantics may not call private CLI model helpers.
- Actual `resolve_api_key_provider_credentials` **only**, after URL normalization is cut out.
- Copilot ACP `resolve_external_process_provider_credentials` for ACP launch/account context.
- OpenCode Go account-usage credential acquisition currently reached through
  `resolve_runtime_provider`: narrow to credential-only input when Phase 6 supplies it.
- Router profile-scoped key lookup (`RAMP_ROUTER_API_KEY` / `ROUTER_API_KEY`), and
  DeepInfra scoped `DEEPINFRA_API_KEY` for private catalogue access.
- Nous Portal auth-state/account-tier acquisition, plus existing credential availability
  checks in shared inventory and the application model-default provider.

These exceptions do **not** authorize re-importing provider normalization, model
selection, catalogue ownership or routing semantics from CLI modules. Existing
`hermes_cli.version_info`, `urllib_security`, `plugin_compat`, setup/commands and
configuration imports are non-semantic application or external-plugin utility
dependencies; preserve their separate contracts unless they are independently refactored.

## Execution and verification contract

1. **5.8.7.2:** Anthropic fetch/pagination; one DeepInfra catalogue; migrate all
   chat, image, video and pricing consumers plus endpoint declarations.
2. **5.8.7.3:** Nous source/recommendation cut; verify Copilot/OpenRouter; separate
   Actual URL semantics from credentials.
3. **5.8.7.4:** Discord/Telegram shared warning and display imports; Feishu regression.
4. **5.8.7.5:** Environment facts, Router/Actual/OpenCode credential-semantic splits,
   shared inventory cut and the three TUI carryovers.
5. **5.8.7.6:** Re-run static + lazy import audit, targeted behavioural tests and
   plugin-boundary gates. Reconcile this file against current source; hand off only
   narrow credential exceptions to Phase 6.

5.8.7.1 changes only this inventory. Do not add compatibility wrappers, fallback
registries, synchronized caches, or tests enforcing the abandoned internal shapes.

## Final reconciliation — Phase 5.8.7.6

The rows above describe the **historical 5.8.7.1 baseline**, not
still-active imports. Source-backed dispositions are recorded in:

- 5.8.7.2: `PHASE5_8_7_2_PROVIDER_CATALOG_CLOSEOUT.md` — Anthropic discovery
  and one DeepInfra catalogue shared across chat, media and pricing.
- 5.8.7.3: `PHASE5_8_7_3_PROVIDER_RECOMMENDATIONS_CLOSEOUT.md` — Nous
  recommendations, existing Copilot/OpenRouter reasoning metadata, Actual URL semantics.
- 5.8.7.4: `PHASE5_8_7_4_PLATFORM_CONSUMER_CLOSEOUT.md` — shared
  Discord/Telegram selection warnings, canonical platform labels, application grouping;
  Slack/Matrix labels added after the original inventory.
- 5.8.7.5: `PHASE5_8_7_5_ENVIRONMENT_INVENTORY_CLOSEOUT.md` — declarative
  profile endpoint interpretation; one read-only application discovery owner,
  explicit catalogue persistence and the three TUI recovery carryovers.
- 5.8.7.6: `PHASE5_8_7_6_FINAL_OWNERSHIP_CLOSEOUT.md` — final static and
  literal-lazy-import gate, unified CLI/platform warning registry, scoped Router
  credential miss, cross-surface regression and exact remaining-owner ledger.

The current scoped audit covers **69 tracked Python files** (the original
scope plus Slack/Matrix adapters); the checked plugin files no longer import
CLI-owned provider/model semantic modules. The old
`hermes_cli.model_switch_providers`, `hermes_cli.provider_groups` and
`hermes_cli.model_selection_guards` modules have no new compatibility
facades. This is a **scoped 5.8.7 closure**, not a claim that all `hermes_cli`
modules or all plugin families have already undergone Phase 6/5.9.

**Narrow Phase 6 imports within this scoped plugin set:** Actual and OpenCode
Go use `resolve_api_key_provider_credentials`; Copilot ACP uses
`resolve_external_process_provider_credentials`. The Router profile now
reads keys through `application_provider_secret_inputs.scoped_key_env`,
which honours the current profile's secret scope and checks its declared
alias. Anthropic guarded HTTP/OAuth, Nous account-tier/auth state and
DeepInfra scoped key acquisition remain credential/application mechanics,
not provider/model policy.

**Additional adjacent findings outside the original audited plugin set:**
OpenAI image generation's named endpoint helper still calls
`hermes_cli.runtime_provider._get_named_custom_provider`; OpenRouter image
and video plugins still call the bundled CLI
`resolve_runtime_provider` to obtain credentials and endpoint facts.
Their credential and route separation needs its own Phase 6/application
consumer cut. These are explicitly **not** counted as cleared by the scoped
5.8.7 audit. Existing command/configuration/setup and plugin lifecycle
imports are separate application integration, not hidden selection owners.
