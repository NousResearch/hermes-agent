# ADR-0010 G2: xAI header-derived governor usage

This note documents the Hermes-side G2 implementation used by Special Circumstances Capital ADR-0010.

## Scope

G2 adds provider usage support for xAI/Grok without activating the governor service or webhook admission layer.

It provides:

- `agent.governor_state.ensure_governor_schema()` — migrates the canonical G1 `governor.db` in place to schema version 2.
- `xai_buckets` table — stores observed `x-ratelimit-*` response-header buckets.
- `agent.account_usage.observe_xai_rate_limit_headers()` — records synthetic or real xAI response headers.
- `fetch_account_usage("xai")` / aliases — returns an `AccountUsageSnapshot` derived from observed headers.
- A best-effort post-response hook in the conversation loop. If an xAI/Grok SDK response exposes headers, Hermes records them; if the SDK hides headers, the hook no-ops safely.
- `agent.governor_state.get_transition_rate()` — implements the documented literal lexical fallback for `PlatformOps.PlatformChange->*`.

## Canonical DB rule

The live database is canonical. G2 must migrate it; it must not recreate it.

`ensure_governor_schema()` raises `FileNotFoundError` if `governor.db` is absent. This is intentional: G1 owns first creation and seed data. G2 only adds the `xai_buckets` table, indexes, and `PRAGMA user_version = 2`.

## xAI source rule

G1 seeds:

```text
provider_state.provider = 'xai'
provider_state.source = 'observed'
```

G2 honours that. xAI usage is not polled from an account-usage endpoint. It is derived from observed response headers and reported as `source='observed_headers'` in `AccountUsageSnapshot`, while `provider_state.source` remains `observed`.

## Band derivation

For each valid observed bucket, G2 computes:

```text
used_pct = ((limit - remaining) / limit) * 100
```

The xAI provider band is derived from the highest observed bucket pressure and written to both daily/weekly governor pressure fields as a conservative approximation until a real xAI account-usage endpoint exists:

| Pressure | Band |
|---:|---|
| `<70%` | `green` |
| `70–84%` | `amber` |
| `85–94%` | `red` |
| `95–97%` | `black` |
| `>=98%` | `post-reserve` |

## Wildcard transition-rate lookup

`PlatformOps.PlatformChange->*` is a literal config-row fallback pattern, not a regex and not SQL `LIKE`.

Lookup order:

1. exact `transition_key` match;
2. if key starts with `PlatformOps.PlatformChange->`, fall back lexically to literal `PlatformOps.PlatformChange->*`;
3. otherwise no match.

## Deferred watchlist

The allocation-sum trigger remains deferred. G2 does not write dynamic `mind_allocation` changes. The trigger decision belongs with #46/G4, as directed by SCO/AA.
