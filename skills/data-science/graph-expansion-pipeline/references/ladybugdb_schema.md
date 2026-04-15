# LadybugDB Schema Reference

## Knows Edge Schema

The `Knows` relationship type has these properties:

| Property | Type | Nullable | Description |
|----------|------|----------|-------------|
| `rel_type` | string | No | Relationship type (colleague, former_colleague, coauthor, etc.) |
| `strength` | float | Yes | Relationship strength score |
| `since` | ISO datetime string | Yes | When the relationship started |
| `context` | string | Yes | Description of the relationship context |
| `source_ref` | string | No | Source of the inference (e.g., expansion_pipeline_synthesis) |
| `confidence` | float | No | Confidence score (0.0-1.0) |
| `record_time` | ISO datetime string | No | When this edge was recorded |
| `_SRC` | internal | No | Source node reference (auto-managed) |
| `_DST` | internal | No | Destination node reference (auto-managed) |
| `_LABEL` | internal | No | Edge label (auto-managed) |
| `_ID` | internal | No | Edge ID (auto-managed, NOT settable) |

**IMPORTANT:** The `Knows` edge does **NOT** have `id` or `source_type` properties. Attempting to set these throws `Binder exception: Cannot find property id for .` or `Cannot find property source_type for .` and the entire CREATE fails.

```python
# WRONG — will fail with Binder exception
CREATE (a)-[:Knows {id: 'knows-xxx', rel_type: 'colleague', source_type: 'inferred', ...}]->(b)

# CORRECT — use only the properties that exist
CREATE (a)-[:Knows {
    rel_type: 'colleague',
    strength: null,
    since: '2026-04-14T00:00:00Z',
    context: 'Both at Google',
    source_ref: 'expansion_pipeline_synthesis',
    confidence: 0.7,
    record_time: '2026-04-14T00:00:00Z'
}]->(b)
```

## How to Discover Edge Schema

Always inspect existing edges to confirm the schema before bulk-creating:

```python
result = conn.execute("MATCH ()-[r:Knows]->() RETURN r LIMIT 1")
edge = result.get_all()[0][0]
print(f"Properties: {list(edge.keys())}")
```

## Preference Node Schema

Preference nodes require an `id` primary key and support `category` as nullable:

| Property | Type | Nullable | Description |
|----------|------|----------|-------------|
| `id` | string | No | Primary key (unique identifier) |
| `value` | string | No | Preference description |
| `category` | string | Yes | Category (e.g., "professional_interest") |
| `valence` | string | Yes | Positive/negative |
| `confidence` | float | No | Confidence score |
| `source_type` | string | No | Source type (direct, inferred, imported, user-stated) |
| `source_ref` | string | No | Source reference |
| `record_time` | ISO datetime string | No | When recorded |

```python
# Preference creation requires id
CREATE (pr:Preference {
    id: 'pref_xxx_interest',
    value: 'Multilingual speech AI',
    category: 'professional_interest',
    confidence: 0.9,
    source_type: 'inferred',
    source_ref: 'sift-deep-2026-04-14',
    record_time: '2026-04-14T00:00:00Z'
})
```

## Finding Created April 2026

This reference was created after the overnight expansion run on April 14, 2026, where all 70 Knows edge creations initially failed because the schema included `id` and `source_type` properties that don't exist on the Knows edge type. After removing those two properties, all 70 edges were created successfully.