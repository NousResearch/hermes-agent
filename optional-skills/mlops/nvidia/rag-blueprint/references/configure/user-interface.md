# User Interface

## When to Use
- User asks about the RAG UI, uploading documents, settings, or metadata filtering
- User wants to configure features via the web interface

## Restrictions
- Sample/experimentation UI — not intended for production
- 100-file limit per upload batch; use multiple batches or API for bulk uploads
- 10 MB max per image attachment

## Process
1. Read `docs/user-interface.md` for full UI documentation
2. Access at `http://localhost:8090` (or `http://<workstation-ip>:8090` for remote)
3. Configure RAG settings and feature toggles via Settings panel
4. Use Filter Bar above chat input for metadata-filtered queries

## Agent-Specific Notes
- VLM Inference must be enabled in Settings > Feature Toggles before image attachments work
- ECONNRESET errors on multi-file uploads — recommend API for bulk operations
- Document summaries generate asynchronously; UI shows "Generating summary..." until complete
- Document count in UI may lag slightly after ingestion
- Metadata filtering supports AND/OR logic between filters (toggle via logic button)
- Custom metadata schema is set during collection creation via the Metadata Schema Editor

## Source Documentation
- `docs/user-interface.md` -- full UI documentation including settings, file types, metadata, and health monitoring
