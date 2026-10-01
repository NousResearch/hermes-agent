# Phase 5.8.7.5 — Environment, shared inventory and TUI hard-cut closeout

Base: `7d3180cfa2` (Phase 5.8.7.4 platform consumers).
Branch: `refactor/phase5-8-runtime-consumers`.

## Changes and authoritative owners

**Single application-owned discovery.** Moved the existing provider-picker
discovery implementation from `hermes_cli/model_switch_providers.py` to
`application_provider_discovery.py`. CLI model switch, shared inventory,
setup helpers, model-catalogue consumers and the documented external plugin
compatibility pointer now use the one owner. The old module was deleted;
no second registry, shadow catalogue or forwarding module was introduced.
CLI and dashboard share the same inventory implementation.

**Read-only observation, deliberate persistence.** Removed the implicit
`_save_discovered_models_to_config` call from the live custom-provider
discovery loop. An inventory build, picker open, model.options and
`GET /api/model/recommended-default` can observe, cache or refresh models,
but do not silently mutate config.yaml. The existing config-matching save
routine now lives in `application_discovered_catalog_persistence.py` and
is called only by explicit named-custom model setup. It preserves
user-curated model metadata and its existing credential/header/endpoint
matching rules. `hermes_cli/models_pricing.pricing_cache_scope` also
now obtains the canonical DeepInfra endpoint through
`application_deepinfra_catalog`, repairing a stale removed-helper import
exposed by the non-blocking picker regression.

**Declared, scoped environment facts.**
`providers.environment.declared_endpoint_override` determines
precedence from caller-supplied explicit, configured and explicitly
declared environment values; it never reads environment or credential
variables. `application_provider_environment` loads the current
profile's config and the registered `base_url_env_var`. Multiplexed
scopes never fall through to another profile's process environment.
Router catalogue discovery and the existing auth endpoint projection
consume this source. Router's key fallback remains a narrowly scoped
Phase 6 credential concern, not an inference-route decision.

**Credential-only usage.** OpenCode Go account usage no longer calls
`hermes_cli.runtime_provider.resolve_runtime_provider` to acquire an
API token. It uses the supplied key or
`hermes_cli.auth.resolve_api_key_provider_credentials`. The account
usage endpoint remains its literal vendor-specific URL.

**Three TUI carryovers.**
`application_configured_provider_facts` gathers scoped config/identity
inputs for `providers.configured.configured_custom_identity`. The TUI
picker and session-row restoration now use it instead of CLI
`canonical_custom_identity`. Managed local server restoration still
requires an ownership-checked endpoint; other localhost ports are not
accepted merely because they are loopback. The session creation
conflict gate uses scoped configuration and provider-request precedence
from the same application fact owner, then delegates conflict policy
to `models.selection_conflict`. Explicit user selections and session
row fields retain their previous precedence.

**External compatibility and Phase 6.** Existing external plugin
re-exports on `hermes_cli.model_switch` were repointed to the final
application owner, not replaced by a new shim. Secret acquisition,
OAuth/token refresh, subprocess auth, scoped key inputs and credential
pools remain the narrow Phase 6 work. The already-existing
application discovery implementation still calls some historical
CLI-hosted catalogue/credential leaves; replacing those mechanics is
part of the documented 5.9 legacy CLI cleanup, not a second model
authority. The moved large listing implementation was not duplicated.

## Behaviour and structural verification

- Shared inventory, non-blocking cold-path/picker and list projection:
  **19 passed**.
- Custom-provider integration: **75 passed, 1 obsolete overlay test
  failed** on the broad run; its stale fixture was removed and the
  corrected case passed on targeted rerun.
- Scoped key and parallel cache prefetch regressions: **19 passed**.
- Actual hosted/local URL and credential focus: **5 passed**.
- Gateway session model restoration/routing/persistence: **12 passed**.
- New Phase 5.8.7.5 environment/ownership contract: **11 passed**.
  The additional corrected custom and OpenCode tests passed in the
  same 13-test focused run.
- Existing Phase 5.8.6 TUI/ACP/dashboard owner gates, plus the new
  contract's earlier version: **17 passed**, one unrelated legacy
  invalid-escape warning.
- Wider CLI/runtime-owner checks: **40 passed**, two previously stale
  fixtures were repaired and their cases passed on focused rerun.
- Selection and external plugin compatibility gates: **12 passed**.
- Ruff across **41 changed Python files**, targeted compilation and
  Git diff validation: passed.

These are overlapping regression suites, not one disjoint test total.
No CBM cold scan, Arcana query or full repository/frontend suite is
claimed.

## Final audit handoff — 5.8.7.6

Re-audit platform and provider plugin imports, shared inventory, lazy
compatibility targets, configured endpoint semantics, declared environment
variables, the historical CLI warning registry and the precise Phase 6
credential exceptions. The former CLI discovery module is absent.
The source ownership audit still finds a CLI-internal
`canonical_custom_identity` caller in `cli_model_switch_mixin.py`;
it is a CLI consumer, not a TUI/ACP/dashboard reverse dependency.
