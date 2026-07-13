# RAG Troubleshooting

## Auto-Triage: Run First

Start with this diagnostic sweep:

```bash
echo "=== CONTAINERS ===" && docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | grep -E "(rag|milvus|nim|ingest|redis|etcd|minio)" | head -20; echo "=== HEALTH ===" && curl -s http://localhost:8081/v1/health?check_dependencies=true 2>/dev/null || echo "RAG_UNREACHABLE"; curl -s http://localhost:8082/v1/health?check_dependencies=true 2>/dev/null || echo "INGESTOR_UNREACHABLE"; echo "=== LOGS ===" && for svc in rag-server ingestor-server nim-llm-ms nemoretriever-embedding-ms nemoretriever-ranking-ms; do echo "--- $svc ---"; docker logs --tail 20 "$svc" 2>/dev/null | grep -iE "(error|fail|exception|timeout|oom)" || echo "OK"; done; echo "=== GPU ===" && nvidia-smi 2>/dev/null | head -20 || echo "NO_GPU"; echo "=== DISK ===" && df -h / | tail -1; echo "=== DOCKER_DISK ===" && docker system df 2>/dev/null; echo "=== K8S ===" && kubectl get pods -n rag 2>/dev/null | head -20 || echo "NOT_K8S"
```

Analyze all output, then diagnose and fix. If Auto-Triage doesn't reveal the cause, dig deeper into the specific failing service's logs (`docker logs <service> --tail 100` or `kubectl logs <pod> -n rag --tail 100`).

Confirm with the user before deleting data (volumes, collections, model cache), changing deployment mode, or modifying API keys.

## Source Documentation for Detailed Diagnosis

Read these docs to find specific issue descriptions, causes, and fixes:

- `docs/troubleshooting.md` — primary reference: all common issues with detailed symptoms/fixes
- `docs/debugging.md` — Pipeline debugging: monitoring deployment, verifying endpoints, tracing requests
- `docs/service-port-gpu-reference.md` — Complete port/GPU mapping table for all services

## Expected Deployment Times

If user reports "deployment is taking too long," compare against these baselines:

| Mode | First Run | Subsequent |
|------|-----------|------------|
| Docker (self-hosted) | 15--30 min (model downloads) | 2--5 min |
| Docker (NVIDIA-hosted) | 5--10 min (no model downloads) | 1--2 min |
| K8s/Helm | 60--70 min (NIM cache 40--50 min + init 10--15 min + pod startup 5--10 min) | 10--15 min |

If deployment exceeds these times, check NIM container logs: `docker logs nim-llm-ms --tail 50` and model cache disk usage: `watch -n 10 'du -sh ~/.cache/model-cache/'`.

## Service Health Endpoints

Read `docs/service-port-gpu-reference.md` for the complete port/GPU mapping. Quick check:

| Service | URL | Expected |
|---------|-----|----------|
| RAG Server | `http://localhost:8081/v1/health?check_dependencies=true` | `{"status":"healthy"}` |
| Ingestor | `http://localhost:8082/v1/health?check_dependencies=true` | `{"status":"healthy"}` |
| NV-Ingest | `http://localhost:7670/v1/health/ready` | 200 OK |
| Embedding NIM | `http://localhost:9080/v1/health/ready` | 200 OK |
| LLM NIM | `http://localhost:8999/v1/health/ready` | 200 OK |
| Ranking NIM | `http://localhost:1976/v1/health/ready` | 200 OK |
| Milvus | `http://localhost:9091/healthz` | 200 OK |

## Kubernetes Monitoring Commands

```bash
kubectl get nimcache -n rag
kubectl get pods -n rag
kubectl logs -f <pod-name> -n rag
kubectl get pvc -n rag
kubectl get events -n rag --sort-by='.lastTimestamp'
```

Pods in `ContainerCreating` or `Init` state during model download is expected. Use `kubectl get nimcache -n rag -w` to watch download progress.

## Enable Debug Logging

```bash
export LOGLEVEL=DEBUG
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d --no-deps ingestor-server
docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --no-deps rag-server
```

---

## Symptom-to-Fix Quick Index

Match the symptom from Auto-Triage output, then read `docs/troubleshooting.md` for the detailed fix. For pipeline debugging steps, read `docs/debugging.md`.

| Symptom | Category | Quick Fix |
|---------|----------|-----------|
| NIM container stuck at `(health: starting)` >30min | NIM Startup | Check GPU memory, NGC auth, disk space. First-run model downloads are slow — wait and monitor cache size. |
| Milvus unhealthy / search returns nothing | Milvus | Restart vectordb compose. Check etcd/MinIO. Port 19530 conflict. Corrupt data → `down -v` (destroys data). |
| Document upload fails / ingestor health check fails | NV-Ingest | Check Redis, OCR NIMs. Rate limit (429) → reduce batch vars. Large PDFs → reduce batch size. |
| Chat returns errors / /generate fails | RAG Server | Check LLM NIM health, embedding NIM, cloud API key. Verify `APP_LLM_MODELNAME` matches deployed NIM. |
| DNS resolution failed for `<service>:<port>` | Networking | Service container not running. Check `docker ps`, restart missing service. |
| Port already in use | Networking | `lsof -i :<port>` to find conflicting process. See port table above. |
| GPU out of memory / `torch.OutOfMemoryError` | GPU | Kill other GPU processes, use `--profile rag` for fewer NIMs, or set correct `NIM_MODEL_PROFILE`. |
| `nvidia-container-cli: unknown device` | GPU | GPU ID exceeds available GPUs. Run `nvidia-smi -L`, adjust `*_GPU_ID` vars to valid IDs. |
| Disk full / insufficient space | Disk | `docker system prune -f`, remove unused images, check model cache size. |
| `no configuration file provided: not found` | Docker Compose | Run from the repo root directory. |
| `too many open files` | Docker Compose | Set `LimitNOFILE=65536` in containerd override, restart containerd. |
| PVC stuck in Pending | Helm | Create missing StorageClass or update PVC. |
| `ProvisioningFailed` access mode mismatch | Helm | Patch NIMCache to `ReadWriteOnce`. |
| Ingestor OOMKilled | Helm | Increase memory limits in values.yaml. Set `SUMMARY_MAX_PARALLELIZATION=1`. |
| Elasticsearch timeout during ingestion | Elasticsearch | Increase `ES_REQUEST_TIMEOUT` (default 600s). |
| Hallucination / out-of-context responses | Quality | Add missing-info handling to prompt in `prompt.yaml`. |
| Embedding dimensions mismatch | Models | Set `APP_EMBEDDINGS_DIMENSIONS` to match model output. Re-ingest. |
| Hybrid/dense search type mismatch | Search | Align `APP_VECTORSTORE_SEARCHTYPE` on ingestor and rag-server. Re-ingest. |
| Confidence threshold filtering all results | Search | Lower `RERANKER_SCORE_THRESHOLD` (range 0.0–1.0, default 0.0). |
| OCR not starting / connection errors | OCR | Check GPU memory, NGC auth. Verify `OCR_GRPC_ENDPOINT`/`OCR_HTTP_ENDPOINT` match running service. |
| NVIDIA API credits exhausted | Cloud | Contact NVIDIA representative for additional credits. |
| Image-only PDFs not ingesting | Ingestion | Enable `APP_NVINGEST_EXTRACTINFOGRAPHICS`. Consider image captioning. |

---

## Troubleshooting Checklists

### Ingestion Checklist
- [ ] All required containers running (ingestor-server, nv-ingest-ms-runtime, milvus, redis)
- [ ] Vector database accessible (`curl http://localhost:9091/healthz`)
- [ ] Embedding service healthy (`curl http://localhost:9080/v1/health/ready`)
- [ ] File format supported and size <= 400 MB
- [ ] Sufficient disk space (`df -h /`)
- [ ] GPU resources available (`nvidia-smi`)

### Retrieval Checklist
- [ ] RAG server running and healthy
- [ ] LLM service accessible (`curl http://localhost:8999/v1/health/ready`)
- [ ] Vector database contains data (collection exists with documents)
- [ ] Collection name is correct
- [ ] Query format is valid

### Quality Checklist
- [ ] Reranker is enabled and healthy
- [ ] Top-K values are appropriate
- [ ] Collection has sufficient relevant data
- [ ] Query rewriting configured correctly
- [ ] Prompt template appropriate for use case

---

## Full Reset

Destroys all data (volumes, images, caches). Confirm with the user before running.

If nothing else works and the user confirms:

```bash
cd "$(git rev-parse --show-toplevel)"
docker compose -f deploy/compose/docker-compose-nemo-guardrails.yaml down 2>/dev/null
docker compose -f deploy/compose/observability.yaml down 2>/dev/null
docker compose -f deploy/compose/docker-compose-rag-server.yaml down 2>/dev/null
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down 2>/dev/null
docker compose -f deploy/compose/vectordb.yaml down -v 2>/dev/null
docker compose -f deploy/compose/nims.yaml down 2>/dev/null

docker system prune -af --volumes
```

Then deploy fresh using the deploy workflow.
