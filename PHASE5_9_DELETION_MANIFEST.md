# Phase 5.9 — Ownership and deletion manifest

Entry baseline: `50dc83cfb4` (Phase 5.8.8). Migration window: 5.9.2–5.9.6.

`phase5_9_inventory/` records all 78 historical manifest rows
(the original 90 references are grouped into these rows), their current imports,
249 application definitions, and literal static/lazy imports across 2,138
production Python files. Consumer lists include lexical candidates; source
inspection and regression coverage determine deletion safety.

The credits/auxiliary pricing and error/failure label consumers now import
the canonical or shared application owners directly. JSONL records distinguish
historical lexical candidates from final AST import consumers and final owners.

| Responsibility | Disposition and final owner |
| --- | --- |
| CLI static matching, alias catalogue facts, native-vendor ownership | Consolidate under `models.catalog_detection`; credential acquisition remains application-owned |
| Second detection candidate/selection ladder | Delete; application detection composes facts for `models.selection_detection` |
| Repeated price/billing interpretation and default-tier predicate | Consolidate under `models.metadata.pricing` |
| Repeated Nous entitlement filtering | Consolidate under `models.catalog_policy` |
| Live catalogue filtering, aliases and deduplication | Consolidate under `models.catalog_projection` |
| Generation-only classification and observations | Consolidate under `models.catalog_chat`; delete the old CLI module |
| Declared prefixes, aggregator slug matching, whitespace eligibility | Consolidate pure interpretation under model catalogue detection/policy; retain caller fact acquisition |
| Provider endpoint API-mode guess | Delete; consume `providers.routing.endpoint_api_mode` |
| Provider label/default/Auto formatting | Retain in existing `application_provider_groups`, consuming provider declarations |
| Pricing credential/config acquisition and observation lifecycle | Retain as shared `application_model_pricing`; CLI module retains only price display formatting |
| Manifest, local/Codex/provider transport acquisition | Retain application acquisition; all tables, identity and capability policy consume lower owners |
| CLI selection facts and picker helpers | Retain fact acquisition and presentation over canonical selection |
| CLI validation | Retain probe acquisition, verdict messages and persistence advisories; no provider selection or model rewrite |
| CLI model switch | Retain session persistence, locking, live apply and rollback; canonical selection and routing choose semantics |
| Scoped model observation caches | Retain distinct manifest, curated projection, credential-specific provider catalogue and negative probe caches; final cache table records scope/expiry |
| Config/env routing and on-disk migrations | Retain supported persisted configuration behavior |
| OAuth, pools, token/secret acquisition | Retain explicit Phase 6 boundaries |
| Shipped updater seed hook | Retain exactly `hermes_cli.model_catalog.seed_cache_from_checkout`, required by `tests/compat/old_updater_surface.json` |
| Existing external plugin pointers | Retain exact `compat_manifest.json` contracts, retarget moved symbols, prohibit in-tree use |

No module is deleted solely because of its name. No forwarding module is added.
The final integration results and retained exceptions are recorded in
PHASE5_9_CLOSEOUT.md. The Phase 5 migration window is closed.

## Retained boundaries

- Title endpoint sharing reads application configuration and matches identities
  through providers.match_configured_provider.
- Image-generation OpenAI uses the existing agent configured-provider acquisition
  adapter. Image/video OpenRouter uses runtime_provider for endpoint/credentials.
  Credential acquisition, OAuth and pools remain Phase 6.
- agent/nous_wire retains the existing first-response application lifecycle and
  configuration lookup; this closeout introduces no wire-selection policy.
- application_model_pricing retains the guarded application HTTP opener from
  hermes_cli.models; billing interpretation has a canonical lower owner.
- The updater seed hook, external plugin names and supported configuration
  migrations remain intact. Internal callers use final owners directly.
