---
name: nvidia-rag-blueprint
description: Deploy and manage NVIDIA RAG systems.
version: 1.0.0
author: NVIDIA RAG Team
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [nvidia, rag, retrieval-augmented-generation, gpu]
    category: mlops
    related_skills: [huggingface-hub, inference]
---

# NVIDIA RAG Blueprint

## Autonomy Principles

- Auto-detect everything: GPU, VRAM, drivers, Docker, CUDA, disk, OS, ports, existing services, NGC key, repo state.
- If it can be checked with a command, check it — don't ask the user.
- Ask only when user action is required: providing an API key, confirming data deletion, or choosing between equally valid options.
- Once analysis is done, route to the correct workflow and execute.

## Intent Detection

Determine what the user wants and route immediately:

| User Intent | Action |
|-------------|--------|
| Deploy, install, set up, start RAG | Read and follow `references/deploy.md` |
| Configure, enable, change, toggle a feature | Use the **Configure** section below |
| Troubleshoot, debug, fix, error, unhealthy | Read and follow `references/troubleshoot.md` |
| Stop, shutdown, tear down, clean up | Read and follow `references/shutdown.md` |

If the intent is ambiguous, infer from context (e.g., "RAG isn't working" → troubleshoot; "get RAG running" → deploy). Only ask if genuinely unclear.

---

## Configure

Requires a running RAG deployment. If services are not running, deploy first via `references/deploy.md`.

Match the user's request to a reference file, then read and follow it:

| Feature Keywords | Reference |
|-----------------|-----------|
| VLM, VLM embeddings, image captioning | `references/configure/vlm.md` |
| NeMo Guardrails | `references/configure/guardrails.md` |
| Query rewriting, decomposition, multi-turn | `references/configure/query-and-conversation.md` |
| Ingestion (text-only, audio, Nemotron Parse, OCR, batch CLI, NV-Ingest, volume mount, performance) | `references/configure/ingestion.md` |
| Search, retrieval, hybrid search, multi-collection, metadata, filters, reranker, topK, accuracy/performance | `references/configure/search-and-retrieval.md` |
| LLM/embedding/ranking model changes, vector DB, Milvus/Elasticsearch auth, service keys, model profiles, ports/GPU | `references/configure/models-and-infrastructure.md` |
| Reasoning, self-reflection, prompts, generation params (tokens, temperature, citations), per-request LLM params | `references/configure/reasoning-and-generation.md` |
| Summarization | `references/configure/summarization.md` |
| Observability (tracing, Zipkin, Grafana, Prometheus) | `references/configure/observability.md` |
| Multimodal query (image + text) | `references/configure/multimodal-query.md` |
| Data catalog (collection/document metadata) | `references/configure/data-catalog.md` |
| User interface (UI settings) | `references/configure/user-interface.md` |
| API reference (endpoints, schemas) | `references/configure/api-reference.md` |
| Evaluation (RAGAS metrics) | `references/configure/evaluation.md` |
| MCP server & client, agent toolkit | `references/configure/mcp.md` |
| Migration (version upgrades) | `references/configure/migration.md` |
| Notebooks (setup and catalog) | `references/configure/notebooks.md` |

### Configure Flow

1. Match the user's request to a reference file from the table above.

2. Detect what's running using `terminal`:
   ```
   docker ps --format '{{.Names}}'
   kubectl get pods -n rag
   ps aux
   ```
   Check for NIM containers (names matching `nim-llm`, `nemoretriever-embedding`, `nemoretriever-ranking`, `nemo-vlm`, `nemotron-vlm`), RAG services (`rag-server`, `ingestor-server`, `milvus`), K8s pods in the `rag` namespace, and library processes (`nvidia_rag`, `uvicorn.*rag`).

3. Use this table to determine platform, deployment type, and where config lives:

   | Local NIMs running? | RAG services running? | Deployment Type | Config Location |
   |---------------------|-----------------------|-----------------|-----------------|
   | Yes (Docker) | Any | Self-hosted | `deploy/compose/.env` |
   | No | Yes (Docker) | NVIDIA-hosted | `deploy/compose/nvdev.env` |
   | Yes (K8s pods) | Any | Self-hosted | `values.yaml` (NIM sections) |
   | No | Yes (K8s pods) | NVIDIA-hosted | `values.yaml` (envVars) |
   | — | Library processes | Library mode | `notebooks/config.yaml` |
   | No | No | Not running | Deploy first via `references/deploy.md` |

   Tell the user what you detected and ask to confirm. Example: "I see local NIM containers running (nim-llm-ms, nemoretriever-embedding-ms) — this is a self-hosted deployment. Config file is `deploy/compose/.env`. Correct?"

4. Check current feature state before changing anything — read the config location from step 3, then cross-check the live service using `terminal`:
   - Docker: run `docker exec rag-server env` and filter for the relevant variable names
   - Helm: run `kubectl get pod -n rag -l app=rag-server -o jsonpath='{.items[0].spec.containers[0].env}'`

   If the config file and live service disagree, tell the user the service has stale config and will need a restart.

5. If the feature needs extra GPUs, check availability using `terminal` to run `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader`.

6. Read the reference file and apply changes:
   - **Docker**: edit the env file (uncomment to enable, re-comment to disable — the env file is the source of truth). Then restart the affected service:
     ```
     source <env-file> && docker compose -f deploy/compose/<compose-file> up -d
     ```
     | Service | Compose File |
     |---------|-------------|
     | rag-server | `docker-compose-rag-server.yaml` |
     | ingestor-server | `docker-compose-ingestor-server.yaml` |
     | milvus, etcd, minio | `vectordb.yaml` |
     | NIM containers (LLM, embedding, ranking, VLM, OCR) | `nims.yaml` |
     | guardrails | `docker-compose-nemo-guardrails.yaml` |
     | observability (Grafana, Prometheus, Zipkin) | `observability.yaml` |
   - **Helm**: edit `values.yaml`, then upgrade: `helm upgrade rag <chart> -n rag -f values.yaml`
   - **Library**: edit `notebooks/config.yaml`, then restart the Python process

7. Verify using `terminal`:
   - Docker: run `docker ps --format "table {{.Names}}\t{{.Status}}"` and `curl -s http://localhost:8081/v1/health?check_dependencies=true`
   - Helm: run `kubectl get pods -n rag` and `kubectl rollout status deployment/rag-server -n rag --timeout=120s`
   - Library: run `curl -s http://localhost:8081/v1/health`

8. If restart fails, read `references/troubleshoot.md`. If multiple features requested, repeat from step 1 for each.

### When User Says "Configure" Without Specifics

Run steps 2–3 above, then use `read_file` to read the identified config file and filter for lines matching `^(export )?(ENABLE_|APP_)`. Summarize what's running and enabled, then ask which feature to change.

---

## Hardware Restrictions

Read `docs/support-matrix.md` for current GPU requirements per deployment mode.
Read `docs/service-port-gpu-reference.md` for port mappings and GPU assignments.

| GPU | Feature Restrictions |
|-----|---------------------|
| B200 | No VLM, No Guardrails, No Nemotron Parse. May need multi-GPU LLM (`LLM_MS_GPU_ID`). |
| RTX PRO 6000 | No Nemotron Parse. No Audio on Helm. |
