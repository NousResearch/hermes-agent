# Data Catalog

## When to Use
- User wants to manage collection or document metadata for governance
- User asks about tagging, ownership, or lifecycle status of collections
- User wants to list or update collection metadata

## Restrictions
- None — available automatically after deployment, no additional configuration needed
- Works with both Milvus and Elasticsearch (full feature parity)

## Process
1. Read `docs/data-catalog.md` for full API reference, field definitions, and examples
2. All endpoints are on the ingestor server (port `8082`)
3. Use PATCH endpoints for updates (merge updates — only provided fields change)

## Decision Table

| Goal | Source Doc | Key Action |
|------|-----------|------------|
| Add governance metadata | `docs/data-catalog.md` | POST `/v1/collection` with description, tags, owner |
| Update lifecycle status | `docs/data-catalog.md` | PATCH with `status: "Archived"` |
| Track content types | `docs/data-catalog.md` | Read auto-populated `has_tables`, `has_images` metrics |
| Filter during retrieval | See custom metadata docs | Use `metadata_schema` + `filter_expr` (not data catalog) |

## Agent-Specific Notes
- Auto-populated metrics (`number_of_files`, `last_indexed`, `has_tables`, etc.) are system-set — not user-editable
- `date_created` and `last_updated` timestamps are automatic
- PATCH is a merge update — omitted fields keep current values
- Different from custom metadata: catalog = governance/discovery, custom metadata = retrieval filtering

## Notebooks
- `notebooks/ingestion_api_usage.ipynb` — ingestion and collection management examples

## Source Documentation
- `docs/data-catalog.md` — full API reference, catalog fields, auto-populated metrics, Python client examples
