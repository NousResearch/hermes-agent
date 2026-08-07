# Docker Deployment (Self-Hosted NIMs)

## When to Use
- User wants full on-premises deployment with local NIM containers
- User has supported GPUs and wants models running locally
- User asks to deploy RAG Blueprint with Docker

## Restrictions

Read `docs/support-matrix.md` for current GPU requirements. Feature restrictions per GPU type:

| GPU | Cannot Use |
|-----|------------|
| B200 | VLM, Guardrails, Nemotron Parse |
| RTX PRO 6000 | Nemotron Parse |

- Read `docs/support-matrix.md` for current minimum NVIDIA Driver, CUDA, Docker, and Compose versions
- NVIDIA Container Toolkit required (`docker info` shows nvidia runtime)
- Disk space per `docs/support-matrix.md` ("Disk Space Requirements")
- If any prerequisite is missing, tell the user what to install before proceeding

## Process
1. Read `docs/deploy-docker-self-hosted.md` for full commands and env configuration
2. Read `docs/support-matrix.md` for GPU compatibility and supported model combinations
3. Verify container toolkit, prepare model cache directory, source `.env`
4. Apply GPU-specific config per source docs
5. Start NIMs → wait for healthy → start remaining services
6. Verify: `docker ps` shows all containers healthy; UI at `http://localhost:8090`

## Decision Table

| Goal | Profile Flag | Notes |
|------|-------------|-------|
| Full deployment (default) | (none) | LLM + embedding + ranking + OCR + detection |
| Text-only RAG (lighter) | `--profile rag` | Skip OCR/detection NIMs |
| Ingestion workload only | `--profile ingest` | Embedding + OCR + detection |
| VLM replaces LLM | `--profile vlm-generation` | Not on B200 |
| Advanced PDF extraction | `--profile nemotron-parse` | Not on B200 or RTX PRO 6000 |

## Agent-Specific Notes
- First run: 15–30 min (model downloads ~100–150 GB, no progress bar); subsequent: 2–5 min
- Monitor download progress: `du -sh ~/.cache/model-cache/`
- Permission error on model cache → try `USERID=0` instead of `USERID=$(id -u)`
- Cloud NIM section in `deploy/compose/.env` must be commented out for self-hosted
- Rebuild after code changes: add `--build` flag to compose up commands

## Source Documentation
- `docs/deploy-docker-self-hosted.md` — full step-by-step commands, env vars, GPU assignments
- `docs/support-matrix.md` — GPU compatibility, supported models, hardware requirements
