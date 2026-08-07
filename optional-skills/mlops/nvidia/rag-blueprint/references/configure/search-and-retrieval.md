# Search & Retrieval: Hybrid Search, Multi-Collection, Metadata & Profiles

## When to Use
User wants to enable hybrid search, query multiple collections, add custom metadata/filters, tune retrieval performance, configure reranker, enable natural language filter generation, or switch accuracy/performance profiles.

## Process

1. Detect the deployment mode (Docker / Helm / Library). Docker: edit the active env file. Helm: edit `values.yaml`. Library: edit `notebooks/config.yaml`
2. Read the relevant source doc for detailed configuration
3. Apply the required env vars to the active config and restart affected services
4. Verify via search/generate API call

## Decision Table

| Goal | Source Doc | Key Env Vars |
|------|-----------|-------------|
| Hybrid search | `docs/hybrid_search.md` | `APP_VECTORSTORE_SEARCHTYPE=hybrid` |
| Multi-collection | `docs/multi-collection-retrieval.md` | `enable_reranker: True` in API payload |
| Custom metadata | `docs/custom-metadata.md` | Metadata in upload payload, `vdb_filter_expression` in query |
| Accuracy profile | `docs/accuracy_perf.md` | Copy values from `deploy/compose/accuracy_profile.env` into the active env file |
| Performance profile | `docs/accuracy_perf.md` | Copy values from `deploy/compose/perf_profile.env` into the active env file |
| Filter generation | `docs/custom-metadata.md` | `ENABLE_FILTER_GENERATOR=True` |

## Agent-Specific Notes

- Hybrid search requires re-ingesting — existing collections created with `dense` must be re-created
- Multi-collection: limited to 5 collections per query; reranker is mandatory
- Multi-collection not supported when `ENABLE_QUERY_DECOMPOSITION=true`
- Elasticsearch RRF not supported in open-source version — must use `weighted` ranker
- Ingestor must be restarted alongside RAG server when enabling hybrid search
- `RERANKER_CONFIDENCE_THRESHOLD` is a legacy alias for `RERANKER_SCORE_THRESHOLD`
- Recommended `RERANKER_SCORE_THRESHOLD` range: 0.3–0.5 (too high filters out too many chunks)

### Advanced Tuning (not fully documented elsewhere)

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_VECTORSTORE_INDEXTYPE` | `GPU_CAGRA` | Vector index type |
| `APP_VECTORSTORE_EF` | `100` | Search accuracy/speed trade-off (must be >= `VECTOR_DB_TOPK`) |
| `VECTOR_DB_TOPK` | `100` | Candidates from vector DB (input to reranker) |
| `APP_RETRIEVER_TOPK` | `10` | Chunks sent to LLM prompt (after reranking) |
| `ENABLE_RERANKER` | `True` | Toggle reranking model |
| `RERANKER_SCORE_THRESHOLD` | `0.0` | Minimum reranker score (0.0–1.0) |
| `COLLECTION_NAME` | `multimodal_data` | Default collection name |

### Partial Filtering
- Strict (default): fails if any collection doesn't support the filter
- Flexible (`allow_partial_filtering: true` in config.yaml): succeeds if at least one collection supports it

### VDB Filter Support

| Feature | Milvus | Elasticsearch |
|---------|--------|---------------|
| NL filter generation | LLM-powered | Not supported (manual DSL) |
| Filter syntax | String expressions | List of dicts (ES Query DSL) |
| UI support | Full filtering interface | API only |

## Notebooks
- `notebooks/retriever_api_usage.ipynb` — RAG retriever API: search and end-to-end queries
- `notebooks/nb_metadata.ipynb` — Metadata ingestion, filtering, and extraction from queries

## Source Documentation
- `docs/hybrid_search.md` — Hybrid dense + sparse search configuration
- `docs/multi-collection-retrieval.md` — Multi-collection querying
- `docs/custom-metadata.md` — Custom metadata schema, filtering expressions, filter generation
- `docs/accuracy_perf.md` — Best practices for tuning ingestion/retrieval/generation settings
- `docs/python-client.md` — Python library API for search and filtering
