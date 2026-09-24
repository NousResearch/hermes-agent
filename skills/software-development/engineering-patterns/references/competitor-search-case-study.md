# Case Study 2: Competitor Search — Builder + Strategy + Template Method

> Source: ReconIQ project (`~/Documents/ReconIQ/`) — implemented 2026-05-20

## Problem

`discover_competitors()` built a single query string from whatever company profile
fields were available, with no explicit industry signal and no fallback queries.
"Companies like Acme" returns national brands and Wikipedia entries, not local
competitors in the same industry and location.

## Root Cause

Five distinct failure modes compounded:
1. `services_products` empty → no industry signal
2. No location narrowing → generic national results
3. Company domain ignored for industry context
4. Single query with no fallback
5. `service_area` not used for location signal

## Solution

`research/competitor_query.py` — `CompetitorQueryBuilder`:

```python
class CompetitorQueryBuilder:
    def __init__(self, company_profile: dict):
        self._company_profile = company_profile

    @cached_property
    def industry_signals(self) -> list[str]:
        # Priority: what_they_do → target_audience → services_products
        p = self._company_profile
        signals = []
        if p.get("what_they_do"): signals.append(p["what_they_do"])
        if p.get("target_audience"): signals.append(p["target_audience"])
        if p.get("services_products"):
            for s in p["services_products"]:
                if isinstance(s, dict):
                    signals.extend([s.get("product",""), s.get("service","")])
                else:
                    signals.append(str(s))
        return [s for s in signals if s]

    @cached_property
    def location_signals(self) -> list[str]:
        p = self._company_profile
        signals = []
        city = (p.get("city") or "").strip()
        state = (p.get("state") or "").strip()
        city_state = f"{city} {state}" if city and state else ""
        if p.get("service_area"):
            for area in p["service_area"]:
                area = area.strip()
                if area and area != city and area != city_state:
                    signals.append(area)
        if city and state: signals.append(city_state)
        if p.get("zip"): signals.append(p["zip"])
        return signals

    def build_query_set(self) -> list[tuple[str, str]]:
        return [
            ("industry_location", self._industry_location_query()),
            ("services_location", self._services_location_query()),
            ("companies_like", self._companies_like_query()),
            ("directory", self._directory_query()),
            ("industry_only", self._industry_only_query()),
        ]
```

**Builder**: signals accumulate lazily via `@cached_property` — expensive
extractions are only computed once and only when needed.

**Strategy**: each `_industry_location_query()`, etc. is a fully encapsulated
query constructor. The priority order is fixed in `build_query_set()` (Template
Method), but individual strategies can be overridden independently.

**Template Method**: `build_query_set()` defines the fixed sequence. Callers
depend on this order — they stop once they have 5+ deduplicated results.

## Backward Compatibility

```python
# search_provider.py — re-export preserves test monkeypatch targets
from research.competitor_query import _build_competitor_query  # noqa: F401
```

## Deduplication Bug Fixed

```python
# OLD — didn't catch "Vancouver WA" when city="Vancouver"
area != city  # "Vancouver WA" != "Vancouver" → True (no dedup)

# NEW — normalized comparison
city_state = f"{city} {state}"
area != city_state  # "Vancouver WA" != "Vancouver WA" → False (correct dedup)
```

## Five Query Strategies

| Strategy | Example Query | When Used |
|----------|--------------|-----------|
| `industry_location` | `"hvac contractor Vancouver WA"` | Strongest local signal |
| `services_location` | `"furnace repair, AC installation Vancouver WA"` | Fallback when industry desc is vague |
| `companies_like` | `"companies like Acme HVAC Vancouver WA"` | Adds industry context to avoid self-match |
| `directory` | `"top hvac contractors in Vancouver WA"` | Catches listing/roundup pages |
| `industry_only` | `"hvac contractor"` | Last resort — regional players |

## Schema Mismatch Bug Discovered

The company profile data arrives with a **nested** location schema:
```python
{"location": {"city": "Ridgefield", "state": "WA", "service_area": [...], "zip": "98642"}}
```

But `_extract_location_signals` originally read flat keys (`location_city`, `location_state`). The function silently returned `[]` — no error, just empty results. Every query variant lost its geographic qualifier and returned generic national results instead of local competitors.

**Fix**: the function now handles both schemas:
```python
loc = profile.get("location", {})
if not isinstance(loc, dict):
    loc = {}
city = loc.get("city", "") or profile.get("location_city", "") or ""
state = loc.get("state", "") or profile.get("location_state", "") or ""
```

**Lesson**: signal-extraction functions that return empty arrays silently on schema mismatch are a durable class of bug. Add schema validation at the boundary before signal extraction begins, especially when the data schema is controlled by another module or service.

## Test Coverage

`tests/test_competitor_query.py` — 23 tests covering:
- Signal extraction (industry, location, name, services)
- Each query strategy with real and empty inputs
- `CompetitorQueryBuilder` integration
- Backward-compat `_build_competitor_query()` function
- Deduplication of `service_area` vs `city+state`
- URL deduplication across query results