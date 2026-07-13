# API Reference

## When to Use
- User needs to call RAG or Ingestor APIs directly
- User asks about endpoints, request/response formats, or task status tracking

## Process
1. Read `docs/api-rag.md` for RAG server endpoints (port 8081)
2. Read `docs/api-ingestor.md` for Ingestor server endpoints (port 8082)
3. Consult OpenAPI schemas for exact request/response shapes

## Agent-Specific Notes
- RAG Server runs on port 8081: `/v1/generate`, `/v1/search`, `/v1/health`, `/v1/configuration`, `/v1/metrics`, `/v1/summary`
- Ingestor Server runs on port 8082: `/v1/documents`, `/v1/collection`, `/v1/collections`, `/v1/status`
- `POST /v1/documents` returns a `task_id` — poll `GET /v1/status?task_id=<id>` for progress
- Task states: `PENDING` → `FINISHED` or `FAILED` (also `UNKNOWN` if not found)
- NV-Ingest extraction states: `not_started` → `submitted` → `processing` → `completed` or `failed`
- Max file size: 400 MB per document
- Full health check: `GET /v1/health?check_dependencies=true`

## Notebooks
- `notebooks/ingestion_api_usage.ipynb` — ingestion API usage examples
- `notebooks/retriever_api_usage.ipynb` — RAG retriever API: search and query examples

## Source Documentation
- `docs/api-rag.md` -- RAG server API details
- `docs/api-ingestor.md` -- Ingestor server API details
- `docs/api_reference/openapi_schema_rag_server.json` -- RAG server OpenAPI schema
- `docs/api_reference/openapi_schema_ingestor_server.json` -- Ingestor server OpenAPI schema
