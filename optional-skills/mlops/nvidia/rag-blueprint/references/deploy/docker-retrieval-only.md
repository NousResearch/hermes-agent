# Retrieval-Only Deployment

## When to Use
- User wants search/retrieval without LLM generation
- User asks to deploy only embedding + reranking services
- User wants `/search` endpoint with an external LLM
- User wants a lightweight, low-GPU deployment

## Restrictions
- `/generate` endpoint returns an error — no LLM is deployed
- Self-hosted: 1 GPU, ~24 GB memory
- NVIDIA-hosted: 0 GPUs (cloud embedding + reranking)

## Process
1. Read `docs/retrieval-only-deployment.md` for full commands, env vars, and API examples
2. Choose variant: self-hosted (local NIMs), NVIDIA-hosted (cloud), or Helm
3. For self-hosted: start only embedding + ranking NIMs, skip LLM
4. For NVIDIA-hosted: set embedding/ranking server URLs to empty, skip NIM startup entirely
5. For Helm: set `nimOperator.nim-llm.enabled=false`
6. Start vector DB → ingestor → RAG server
7. Verify health: `GET http://localhost:8081/v1/health?check_dependencies=true`

## Decision Table

| Goal | Variant | Key Difference |
|------|---------|----------------|
| Minimal GPU usage with local models | Self-hosted | 1 GPU, ~24 GB |
| Zero GPU, cloud APIs | NVIDIA-hosted | Set server URLs to empty, skip NIM startup |
| Kubernetes | Helm | Disable `nim-llm` in values.yaml |

## Agent-Specific Notes
- Permission errors on model cache → try `USERID=0` or `chmod -R 755 ~/.cache/model-cache`
- Empty search results → verify documents ingested: `GET http://localhost:8082/v1/documents?collection_name=<name>`
- Users can send `/search` results to their own external LLM for generation

## Source Documentation
- `docs/retrieval-only-deployment.md` — full deployment commands, API examples, search payload options
