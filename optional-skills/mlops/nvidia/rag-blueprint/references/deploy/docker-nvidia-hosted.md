# Docker Deployment (NVIDIA-Hosted NIMs)

## When to Use
- User wants fast deployment without local model downloads
- User has no GPU or limited GPU
- User asks about cloud-hosted or NVIDIA API deployment
- User wants to avoid 15–30 min NIM startup time

## Restrictions
- Requires internet access (calls NVIDIA cloud APIs)
- NVIDIA-hosted endpoints have rate limits — large ingestions (>10 files) may hit 429 errors
- NGC_API_KEY required for cloud API access
- Docker and Compose minimum versions per `docs/support-matrix.md`

## Process
1. Read `docs/deploy-docker-nvidia-hosted.md` for full commands and env configuration
2. Use `deploy/compose/nvdev.env` — pre-configured for cloud endpoints. Source it before compose commands: `source deploy/compose/nvdev.env`
3. Start vector DB → ingestor → RAG server + frontend (no NIM startup needed)
4. Verify: `docker ps` shows containers; UI at `http://localhost:8090`

## Decision Table

| Goal | Key Action |
|------|------------|
| Standard cloud deployment | Use `nvdev.env` (pre-configured for cloud) |
| Zero-GPU (no Milvus GPU) | Also switch Milvus image to CPU-only |
| Large file ingestion | Reduce batch/concurrency settings to avoid 429s |
| Maximum throughput | Use self-hosted deployment instead |

## Agent-Specific Notes
- First run: 5–10 min (image pulls only); subsequent: 1–2 min
- No `nims.yaml` startup — all model inference is cloud-hosted
- All subsequent configure/restart operations should source the same env file used for the initial deploy (`deploy/compose/nvdev.env`)
- For zero-GPU: switch Milvus to CPU-only by changing the GPU image tag to the equivalent non-GPU tag and setting `APP_VECTORSTORE_ENABLEGPUSEARCH=False`. See `docs/deploy-docker-nvidia-hosted.md` for the current image tags
- Rate limit mitigation for large ingestions: reduce `NV_INGEST_FILES_PER_BATCH`, `NV_INGEST_CONCURRENT_BATCHES`, `MAX_INGEST_PROCESS_WORKERS`, `NV_INGEST_MAX_UTIL` to minimum values

## Source Documentation
- `docs/deploy-docker-nvidia-hosted.md` — full step-by-step commands, env var blocks, CPU Milvus setup
