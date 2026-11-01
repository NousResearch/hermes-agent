# Pricing Accuracy Architecture

Date: 2026-03-16

## Goal

Hermes should only show dollar costs when they are backed by an official source for the user's actual billing path.

This design replaces the current static, heuristic pricing flow in:

- `run_agent.py`
- `agent/usage_pricing.py`
- `agent/insights.py`
- `cli.py`

with a provider-aware pricing system that:

- handles cache billing correctly
- distinguishes `actual` vs `estimated` vs `included` vs `unknown`
- reconciles post-hoc costs when providers expose authoritative billing data
- supports direct providers, OpenRouter, subscriptions, enterprise pricing, and custom endpoints

## Problems In The Current Design

Current Hermes behavior has four structural issues:

1. It stores only `prompt_tokens` and `completion_tokens`, which is insufficient for providers that bill cache reads and cache writes separately.
2. It uses a static model price table and fuzzy heuristics, which can drift from current official pricing.
3. It assumes public API list pricing matches the user's real billing path.
4. It has no distinction between live estimates and reconciled billed cost.

## Design Principles

1. Normalize usage before pricing.
2. Never fold cached tokens into plain input cost.
3. Track certainty explicitly.
4. Treat the billing path as part of the model identity.
5. Prefer official machine-readable sources over scraped docs.
6. Use post-hoc provider cost APIs when available.
7. Show `n/a` rather than inventing precision.

## High-Level Architecture

The new system has four layers:

1. `usage_normalization`
   Converts raw provider usage into a canonical usage record.
2. `pricing_source_resolution`
   Determines the billing path, source of truth, and applicable pricing source.
3. `cost_estimation_and_reconciliation`
   Produces an immediate estimate when possible, then replaces or annotates it with actual billed cost later.
4. `presentation`
   `/usage`, `/insights`, and the status bar display cost with certainty metadata.

## Canonical Usage Record

Add a canonical usage model that every provider path maps into before any pricing math happens.

Suggested structure:

```python
@dataclass
class CanonicalUsage:
    provider: str
    billing_provider: str
    model: str
    billing_route: str

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    request_count: int = 1

    raw_usage: dict[str, Any] | None = None
    raw_usage_fields: dict[str, str] | None = None
    computed_fields: set[str] | None = None

    provider_request_id: str | None = None
    provider_generation_id: str | None = None
    provider_response_id: str | None = None
```

Rules:

- `input_tokens` means non-cached input only.
- `cache_read_tokens` and `cache_write_tokens` are never merged into `input_tokens`.
- `output_tokens` excludes cache metrics.
- `reasoning_tokens` is telemetry unless a provider officially bills it separately.

This is the same normalization pattern used by `opencode`, extended with provenance and reconciliation ids.

## Provider Normalization Rules

### OpenAI Direct

Source usage fields:

- `prompt_tokens`
- `completion_tokens`
- `prompt_tokens_details.cached_tokens`

Normalization:

- `cache_read_tokens = cached_tokens`
- `input_tokens = prompt_tokens - cached_tokens`
- `cache_write_tokens = 0` unless OpenAI exposes it in the relevant route
- `output_tokens = completion_tokens`

### Anthropic Direct

Source usage fields:

- `input_tokens`
- `output_tokens`
- `cache_read_input_tokens`
- `cache_creation_input_tokens`

Normalization:

- `input_tokens = input_tokens`
- `output_tokens = output_tokens`
- `cache_read_tokens = cache_read_input_tokens`
- `cache_write_tokens = cache_creation_input_tokens`

### OpenRouter

Estimate-time usage normalization should use the response usage payload with the same rules as the underlying provider when possible.

Reconciliation-time records should also store:

- OpenRouter generation id
- native token fields when available
- `total_cost`
- `cache_discount`
- `upstream_inference_cost`
- `is_byok`

### Gemini / Vertex

Use official Gemini or Vertex usage fields where available.

If cached content tokens are exposed:

- map them to `cache_read_tokens`

If a route exposes no cache creation metric:

- store `cache_write_tokens = 0`
- preserve the raw usage payload for later extension

### DeepSeek And Other Direct Providers

Normalize only the fields that are officially exposed.

If a provider does not expose cache buckets:

- do not infer them unless the provider explicitly documents how to derive them

### Subscription / Included-Cost Routes

These still use the canonical usage model.

Tokens are tracked normally. Cost depends on billing mode, not on whether usage exists.

## Billing Route Model

Hermes must stop keying pricing solely by `model`.

Introduce a billing route descriptor:

```python
@dataclass
class BillingRoute:
    provider: str
    base_url: str | None
    model: str
    billing_mode: str
    organization_hint: str | None = None
```

`billing_mode` values:

- `official_cost_api`
- `official_generation_api`
- `official_models_api`
- `official_docs_snapshot`
- `subscription_included`
- `user_override`
- `custom_contract`
- `unknown`

Examples:

- OpenAI direct API with Costs API access: `official_cost_api`
- Anthropic direct API with Usage & Cost API access: `official_cost_api`
- OpenRouter request before reconciliation: `official_models_api`
- OpenRouter request after generation lookup: `official_generation_api`
- GitHub Copilot style subscription route: `subscription_included`
- local OpenAI-compatible server: `unknown`
- enterprise contract with configured rates: `custom_contract`

## Cost Status Model

Every displayed cost should have:

```python
@dataclass
class CostResult:
    amount_usd: Decimal | None
    status: Literal["actual", "estimated", "included", "unknown"]
    source: Literal[
        "provider_cost_api",
        "provider_generation_api",
        "provider_models_api",
        "official_docs_snapshot",
        "user_override",
        "custom_contract",
        "none",
    ]
    label: str
    fetched_at: datetime | None
    pricing_version: str | None
    notes: list[str]
```

Presentation rules:

- `actual`: show dollar amount as final
- `estimated`: show dollar amount with estimate labeling
- `included`: show `included` or `$0.00 (included)` depending on UX choice
- `unknown`: show `n/a`

## Official Source Hierarchy

Resolve cost using this order:

1. Request-level or account-level official billed cost
2. Official machine-readable model pricing
3. Official docs snapshot
4. User override or custom contract
5. Unknown

The system must never skip to a lower level if a higher-confidence source exists for the current billing route.

## Provider-Specific Truth Rules

### OpenAI Direct

Preferred truth:

1. Costs API for reconciled spend
2. Official pricing page for live estimate

### Anthropic Direct

Preferred truth:

1. Usage & Cost API for reconciled spend
2. Official pricing docs for live estimate

### OpenRouter

Preferred truth:

1. `GET /api/v1/generation` for reconciled `total_cost`
2. `GET /api/v1/models` pricing for live estimate

Do not use underlying provider public pricing as the source of truth for OpenRouter billing.

### Gemini / Vertex

Preferred truth:

1. official billing export or billing API for reconciled spend when available for the route
2. official pricing docs for estimate

### DeepSeek

Preferred truth:

1. official machine-readable cost source if available in the future
2. official pricing docs snapshot today

### Subscription-Included Routes

Preferred truth:

1. explicit route config marking the model as included in subscription

These should display `included`, not an API list-price estimate.

### Custom Endpoint / Local Model

Preferred truth:

1. user override
2. custom contract config
3. unknown

These should default to `unknown`.

## Pricing Catalog

Replace the current `MODEL_PRICING` dict with a richer pricing catalog.

Suggested record:

```python
@dataclass
class PricingEntry:
    provider: str
    route_pattern: str
    model_pattern: str

    input_cost_per_million: Decimal | None = None
    output_cost_per_million: Decimal | None = None
    cache_read_cost_per_million: Decimal | None = None
    cache_write_cost_per_million: Decimal | None = None
    request_cost: Decimal | None = None
    image_cost: Decimal | None = None

    source: str = "official_docs_snapshot"
    source_url: str | None = None
    fetched_at: datetime | None = None
    pricing_version: str | None = None
```

The catalog should be route-aware:

- `openai:gpt-5`
- `anthropic:claude-opus-4-6`
- `openrouter:anthropic/claude-opus-4.6`
- `copilot:gpt-4o`

This avoids conflating direct-provider billing with aggregator billing.

## Pricing Sync Architecture

Introduce a pricing sync subsystem instead of manually maintaining a single hardcoded table.

Suggested modules:

- `agent/pricing/catalog.py`
- `agent/pricing/sources.py`
- `agent/pricing/sync.py`
- `agent/pricing/reconcile.py`
- `agent/pricing/types.py`

### Sync Sources

- OpenRouter models API
- official provider docs snapshots where no API exists
- user overrides from config

### Sync Output

Cache pricing entries locally with:

- source URL
- fetch timestamp
- version/hash
- confidence/source type

### Sync Frequency

- startup warm cache
- background refresh every 6 to 24 hours depending on source
- manual `hermes pricing sync`

## Reconciliation Architecture

Live requests may produce only an estimate initially. Hermes should reconcile them later when a provider exposes actual billed cost.

Suggested flow:

1. Agent call completes.
2. Hermes stores canonical usage plus reconciliation ids.
3. Hermes computes an immediate estimate if a pricing source exists.
4. A reconciliation worker fetches actual cost when supported.
5. Session and message records are updated with `actual` cost.

This can run:

- inline for cheap lookups
- asynchronously for delayed provider accounting

## Persistence Changes

Session storage should stop storing only aggregate prompt/completion totals.

Add fields for both usage and cost certainty:

- `input_tokens`
- `output_tokens`
- `cache_read_tokens`
- `cache_write_tokens`
- `reasoning_tokens`
- `estimated_cost_usd`
- `actual_cost_usd`
- `cost_status`
- `cost_source`
- `pricing_version`
- `billing_provider`
- `billing_mode`

If schema expansion is too large for one PR, add a new pricing events table:

```text
session_cost_events
  id
  session_id
  request_id
  provider
  model
  billing_mode
  input_tokens
  output_tokens
  cache_read_tokens
  cache_write_tokens
  estimated_cost_usd
  actual_cost_usd
  cost_status
  cost_source
  pricing_version
  created_at
  updated_at
```

## Hermes Touchpoints

### `run_agent.py`

Current responsibility:

- parse raw provider usage
- update session token counters

New responsibility:

- build `CanonicalUsage`
- update canonical counters
- store reconciliation ids
- emit usage event to pricing subsystem

### `agent/usage_pricing.py`

Current responsibility:

- static lookup table
- direct cost arithmetic

New responsibility:

- move or replace with pricing catalog facade
- no fuzzy model-family heuristics
- no direct pricing without billing-route context

### `cli.py`

Current responsibility:

- compute session cost directly from prompt/completion totals

New responsibility:

- display `CostResult`
- show status badges:
  - `actual`
  - `estimated`
  - `included`
  - `n/a`

### `agent/insights.py`

Current responsibility:

- recompute historical estimates from static pricing

New responsibility:

- aggregate stored pricing events
- prefer actual cost over estimate
- surface estimates only when reconciliation is unavailable

## UX Rules

### Status Bar

Show one of:

- `$1.42`
- `~$1.42`
- `included`
- `cost n/a`

Where:

- `$1.42` means `actual`
- `~$1.42` means `estimated`
- `included` means subscription-backed or explicitly zero-cost route
- `cost n/a` means unknown

### `/usage`

Show:

- token buckets
- estimated cost
- actual cost if available
- cost status
- pricing source

### `/insights`

Aggregate:

- actual cost totals
- estimated-only totals
- unknown-cost sessions count
- included-cost sessions count

## Config And Overrides

Add user-configurable pricing overrides in config:

```yaml
pricing:
  mode: hybrid
  sync_on_startup: true
  sync_interval_hours: 12
  overrides:
    - provider: openrouter
      model: anthropic/claude-opus-4.6
      billing_mode: custom_contract
      input_cost_per_million: 4.25
      output_cost_per_million: 22.0
      cache_read_cost_per_million: 0.5
      cache_write_cost_per_million: 6.0
  included_routes:
    - provider: copilot
      model: "*"
    - provider: codex-subscription
      model: "*"
```

Overrides must win over catalog defaults for the matching billing route.

## Rollout Plan

### Phase 1

- add canonical usage model
- split cache token buckets in `run_agent.py`
- stop pricing cache-inflated prompt totals
- preserve current UI with improved backend math

### Phase 2

- add route-aware pricing catalog
- integrate OpenRouter models API sync
- add `estimated` vs `included` vs `unknown`

### Phase 3

- add reconciliation for OpenRouter generation cost
- add actual cost persistence
- update `/insights` to prefer actual cost

### Phase 4

- add direct OpenAI and Anthropic reconciliation paths
- add user overrides and contract pricing
- add pricing sync CLI command

## Testing Strategy

Add tests for:

- OpenAI cached token subtraction
- Anthropic cache read/write separation
- OpenRouter estimated vs actual reconciliation
- subscription-backed models showing `included`
- custom endpoints showing `n/a`
- override precedence
- stale catalog fallback behavior

Current tests that assume heuristic pricing should be replaced with route-aware expectations.

## Non-Goals

- exact enterprise billing reconstruction without an official source or user override
- backfilling perfect historical cost for old sessions that lack cache bucket data
- scraping arbitrary provider web pages at request time

## Recommendation

Do not expand the existing `MODEL_PRICING` dict.

That path cannot satisfy the product requirement. Hermes should instead migrate to:

- canonical usage normalization
- route-aware pricing sources
- estimate-then-reconcile cost lifecycle
- explicit certainty states in the UI

This is the minimum architecture that makes the statement "Hermes pricing is backed by official sources where possible, and otherwise clearly labeled" defensible.
