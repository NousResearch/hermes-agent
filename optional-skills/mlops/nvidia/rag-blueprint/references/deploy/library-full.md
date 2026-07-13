# Library Mode (Full)

## When to Use
- User wants programmatic Python access to RAG via `nvidia_rag` package
- User prefers code-level configuration over Docker-based servers
- User asks about library mode, Python client, or `NvidiaRAG`/`NvidiaRAGIngestor`

## Restrictions
- Python 3.11+ (< 3.14)
- Docker still required for backend services (Milvus, NV-Ingest, Redis, optionally NIMs)
- Self-hosted NIMs require supported GPUs (see `docs/support-matrix.md`)

## Process
1. Read `docs/python-client.md` for full API reference, configuration, and backend setup
2. Create virtual environment and install `nvidia-rag[all]`
3. Start backend services via Docker (Milvus, NV-Ingest + Redis, optionally NIMs)
4. Load config from `notebooks/config.yaml` using `NvidiaRAGConfig.from_yaml()`
5. Create `NvidiaRAGIngestor` and `NvidiaRAG` instances
6. Use `ingestor.create_collection()`, `ingestor.upload_documents()`, `rag.generate()`, `rag.search()`

## Decision Table

| Goal | Source Doc | Key Action |
|------|-----------|------------|
| Self-hosted (local GPUs) | `docs/python-client.md` | Start nims.yaml + set on-prem config |
| Cloud (NVIDIA-hosted) | `docs/python-client.md` | Skip nims.yaml, override server URLs in config |
| Custom prompts | `docs/python-client.md` | Pass `prompts=` to NvidiaRAG constructor |
| Summarization | `docs/python-client.md` | `generate_summary=True` in upload_documents |

## Agent-Specific Notes
- Config file: `notebooks/config.yaml`; env file: `notebooks/.env_library`
- Docker login is interactive — tell user to run `docker login nvcr.io` themselves
- For cloud deployment: override `config.embeddings.server_url`, `config.llm.server_url`, etc. in code
- Config changes take effect immediately (no container restart needed, unlike Docker mode)
- Prompt customization via constructor: `NvidiaRAG(config=config, prompts="custom_prompts.yaml")`
- `upload_documents()` is async — returns `task_id` for status polling
- NV-Ingest cloud endpoints must be exported before starting NV-Ingest container

## Notebooks
- `notebooks/rag_library_usage.ipynb` — complete walkthrough: setup, ingestion, querying, search, summaries

## Source Documentation
- `docs/python-client.md` — full API reference, backend setup, configuration, cloud/self-hosted options
