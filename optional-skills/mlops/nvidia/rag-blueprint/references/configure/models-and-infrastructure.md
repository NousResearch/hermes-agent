# Models, Vector DB & Service API Keys

## When to Use
User wants to change LLM, embedding, or ranking models; switch vector DB (Milvus/Elasticsearch); configure Milvus auth, GPU mode, or custom endpoints; set service-specific API keys; or build a custom VDB operator.

## Process

Detect the deployment mode before making changes. Docker: edit the active env file. Helm: edit `values.yaml` under `nimOperator` and `envVars` sections. Library: edit `notebooks/config.yaml`.

### Change Models (LLM, Embedding, Ranking)
1. Read `docs/change-model.md` for full model change instructions
2. Read `docs/model-profiles.md` for NIM profile selection and GPU-specific profiles
3. Key env vars: `APP_LLM_MODELNAME`, `APP_EMBEDDINGS_MODELNAME`, `APP_RANKING_MODELNAME`
4. Embedding model change requires re-ingesting all documents — update `APP_EMBEDDINGS_DIMENSIONS` to match
5. Restart affected services (RAG server + ingestor for embedding changes)
6. Verify via health endpoint

### Switch Vector DB (Milvus to Elasticsearch)
1. Read `docs/change-vectordb.md` for full setup (Docker and Helm)
2. Key env vars: `APP_VECTORSTORE_URL`, `APP_VECTORSTORE_NAME`
3. Data is not migrated — re-ingest all documents after switching
4. Elasticsearch requires port 9200; check for conflicts

### Milvus Configuration
1. Read `docs/milvus-configuration.md` for indexing, GPU, auth, and tuning
2. Read `docs/milvus-schema.md` for collection schema requirements
3. CPU mode: set `APP_VECTORSTORE_ENABLEGPUSEARCH=False`, `APP_VECTORSTORE_ENABLEGPUINDEX=False`, change Milvus image to non-GPU
4. Auth: download milvus.yaml, enable `authorizationEnabled`, set password before first deployment

### API Keys
1. Read `docs/api-key.md` for NGC API key setup and per-service keys
2. Fallback order: service-specific key > `NVIDIA_API_KEY` > `NGC_API_KEY`
3. Per-service keys: `APP_LLM_APIKEY`, `APP_EMBEDDINGS_APIKEY`, `APP_RANKING_APIKEY`, `APP_VLM_APIKEY`, etc.

## Decision Table

| Goal | Source Doc | Key Action |
|------|-----------|------------|
| Change LLM | `docs/change-model.md` | Set `APP_LLM_MODELNAME`, restart RAG server |
| Change embedding | `docs/change-model.md` | Set `APP_EMBEDDINGS_MODELNAME` + `APP_EMBEDDINGS_DIMENSIONS`, re-ingest |
| Change reranker | `docs/change-model.md` | Set `APP_RANKING_MODELNAME`, restart RAG server |
| Switch to Elasticsearch | `docs/change-vectordb.md` | Create data dir, start ES profile, set env vars, re-ingest |
| Milvus auth | `docs/milvus-configuration.md` | Download config, enable auth, mount volume |
| Milvus CPU mode | `docs/milvus-configuration.md` | Change image, disable GPU env vars |
| Custom VDB | `docs/change-vectordb.md` | Implement `VDBRag`, register in `__init__.py` |
| NIM profiles | `docs/model-profiles.md` | List profiles, set `NIM_MODEL_PROFILE` |
| Service API keys | `docs/api-key.md` | Set per-service `*_APIKEY` vars |
| Collection schema | `docs/milvus-schema.md` | Required fields: pk, vector, text, source, content_metadata |

## Agent-Specific Notes

- Nemotron-3-Nano naming: `nvidia/nemotron-3-nano-30b-a3b` (NVIDIA-hosted) vs `nvidia/nemotron-3-nano` (self-hosted NIM) — same model, different names
- Helm model changes go in `values.yaml` under `nimOperator` and `envVars` sections
- Custom VDB operator requires implementing `VDBRag` base class — see `docs/change-vectordb.md` "Custom Vector Database Operator" section
- VDB auth tokens can be passed per-request via `Authorization: Bearer <token>` header
- Milvus password persists in etcd volume — to change after deployment, must delete volumes (destroys data)

## Notebooks
- `notebooks/building_rag_vdb_operator.ipynb` — Custom VDB operator implementation (OpenSearch example)

## Source Documentation
- `docs/change-model.md` — Model changes (LLM, embedding, ranking, NIM images)
- `docs/change-vectordb.md` — Vector DB switching, Elasticsearch setup, custom VDB operator
- `docs/milvus-configuration.md` — Milvus indexing, GPU config, auth, tuning
- `docs/milvus-schema.md` — Collection schema fields and requirements
- `docs/model-profiles.md` — NIM profile definitions and selection
- `docs/api-key.md` — NGC API key setup, per-service keys, fallback order
- `docs/service-port-gpu-reference.md` — Port mappings and GPU assignments for all services
