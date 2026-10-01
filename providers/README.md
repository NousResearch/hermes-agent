# Provider and model ownership

Each inference provider is declared once as a `ProviderProfile`. Bundled
profiles live under `plugins/model-providers/<name>/`; per-home profiles and
pip entry points are discovered by `providers.discovery`. The registry keeps
home layers, aliases and precedence. Consumers use the effective declaration.

| Domain | Authoritative responsibility |
| --- | --- |
| `providers.identity`, `registry`, `discovery` | Provider identity, declarations, labels and plugin loading |
| `providers.configured`, `environment` | Configured route matching and environment policy over supplied facts |
| `providers.route_identity`, `routing` | Endpoint identity, API-mode mandates, runtime kind and invocation route |
| `models.identity`, `aliases` | Model references and aliases |
| `models.catalog_*` | Static/live catalogue facts, configured declarations, chat capability filtering and catalogue projections |
| `models.metadata` | Context, capabilities, reasoning, billing and pricing interpretation |
| `models.selection` | Explicit, default, auxiliary and picker selection over acquired facts |

Applications acquire configuration, credentials, availability and network
observations, then apply canonical results. `hermes_cli.models` retains catalogue
acquisition and observation lifecycle. `application_model_pricing` acquires
pricing documents; `models.metadata.pricing` interprets billing.
`hermes_cli.models_validate` acquires validation evidence and presents verdicts.
Model switching retains persistence, locks, live application and rollback.
Provider grouping and Auto/default label presentation are application-owned.

Explicit detection goes through `models.selection_detection`. The current
provider's live catalogue and native-vendor ownership remain higher priority
than reseller listings. Fact acquisition stops when the canonical selector has
an authoritative answer, preserving the no-unnecessary-network path.

Credential acquisition, OAuth refresh, pools and credential persistence remain
Phase 6 boundaries. Their current application seams supply canonical routing
facts. The distinction between `.env` and `config.yaml` remains application
configuration policy in `hermes_cli.config_env_routing`.

## Observation caches

These caches store different observations or projections; none is a second
implementation of provider identity, routing or selection.

| Owner/cache | Identity and purpose | Refresh/stale behavior |
| --- | --- | --- |
| `models.catalog_runtime` | Settings plus profile cache path; remote curated manifest | Configured TTL, disk fallback and explicit refresh |
| CLI OpenRouter curated projection | Profile/home; tool-capable manifest subset plus badges | Disk TTL follows manifest settings; process memo until explicit refresh; cache-only may serve stale disk |
| CLI AI Gateway curated projection | Public catalogue, no credential-specific facts | Process memo; explicit refresh |
| CLI provider catalogue disk cache | Home, provider, endpoint/credential or stable Codex principal fingerprint | 1 h fresh; useful live data up to 7 d stale while refreshing; curated failure placeholders 60 s |
| CLI negative endpoint probes | Home plus endpoint/credential fingerprint | Short failure window; force refresh bypasses it |
| `application_model_pricing` | Home, endpoint and credential fingerprint; pricing and Nous policy share one observation | Nous 300 s; failed reads 120 s; other successful documents process-resident until forced refresh; cache-only never fetches |
| `models.catalog_deepinfra` | Endpoint and credential; one full document for chat/image/video/audio projections | Shared success/failure cache; cached-only and force-refresh controls |
| `models.catalog_github` | Credential fingerprint; account catalogue | 300 s; returns copies; credential change invalidates the slot |
| Local Ollama probes | Scoped endpoint and request headers | Native probe success/failure lifecycle, distinct from persisted provider rows |
| `models.metadata.reasoning` | Profile and catalogue URL; capability projection | Seeded from already-fetched documents, disk warm start and explicit refresh |
| `models.catalog_chat` | Home; remembered opaque generation-only ids | Process observation, kept separate across profile switches |
| `models.models_dev_cache` / agent adapter | Canonical disk document and application parsed process view | One disk authority, ETag and TTL; parsed view is a lifecycle optimization |

The per-profile slot helper isolates application process memos under routed
home overrides. It stores no independent catalogue policy. Pricing peeks and
negative failure windows cannot inspect a sibling profile's cached document.
Hashes identify credentials; raw secrets are not stored in cache keys.

## External exceptions

`hermes_cli.model_catalog.seed_cache_from_checkout` is the exact shipped
updater hook covered by `tests/compat/old_updater_surface.json`. It is not a
runtime catalogue authority. Existing external `PLUGIN-COMPAT` pointers retain
their old names and point to final owners in `compat_manifest.json`; in-tree
code must import those owners directly. Supported persisted configuration
migrations remain supported.

## Provider extension contract

See `plugins/model-providers/README.md` and `providers/base.py`. Profiles
supply transport/discovery mechanics and declared policy through generic hooks:
`fetch_models`, `prepare_messages`, `build_extra_body`,
`build_api_kwargs_extras`, `resolve_route_policy` and
`supported_reasoning_efforts`. Runtime consumers read those declarations through
the canonical domains. Reasoning vocabulary queries on the request hot path
remain cache-only.
