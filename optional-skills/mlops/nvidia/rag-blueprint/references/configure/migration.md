# Migration Guide

## When to Use
- User is upgrading between RAG Blueprint versions
- User encounters breaking API changes or deprecated endpoints after an update

## Process
1. Read `docs/migration_guide.md` for full version-by-version migration details
2. Identify the user's current and target versions
3. Apply changes sequentially for each version gap

## Agent-Specific Notes

### v2.2.0 → v2.3.0
- New `confidence_threshold` field in `/generate` and `/search` (0.0–1.0, default 0.0)
- New `summary_options` parameter with `page_filter`, `shallow_summary`, `summarization_strategy`
- `SUMMARY_LLM_MAX_CHUNK_LENGTH` and `SUMMARY_CHUNK_OVERLAP` changed from character-based to token-based — divide old values by ~4

### v2.1.0 → v2.2.0
- Added `generate_summary` to `/documents`, new `GET /summary` endpoint
- `POST /collection` (singular) replaces `POST /collections` for single collection creation
- `collection_names: List[str]` replaces `collection_name: str` in `/generate` and `/search`

### v2.0.0 → v2.1.0
- `POST /documents` gained `blocking: bool` (default `True`); use `false` + `GET /status` for async

### v1.0.0 → v2.0.0 (Breaking)
- Single server split into RAG Server (port 8081) and Ingestion Server (port 8082)
- Collections must be explicitly created before uploading documents
- Default changed from cloud-hosted to on-prem models

## Source Documentation
- `docs/migration_guide.md` — Full migration guide with examples and env var changes
- `docs/release-notes.md` — Release notes and version history
- `docs/query-to-answer-pipeline.md` — Query-to-answer pipeline architecture overview
