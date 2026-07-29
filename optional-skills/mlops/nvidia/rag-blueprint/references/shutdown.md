# RAG Shutdown

Stopping containers and processes does not require confirmation. Deleting data (volumes, cache, images) does.

## Step 1: Detect What Is Running

Detect all deployment modes — Docker, K8s, and library:

```bash
echo "=== DOCKER ===" && docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Image}}" 2>/dev/null || echo "NO_DOCKER"; echo "=== LIBRARY ===" && ps aux | grep -E "(nvidia_rag|uvicorn|jupyter)" | grep -v grep || echo "NO_LIBRARY_PROCESSES"; echo "=== K8S ===" && kubectl get pods -n rag 2>/dev/null | head -10 || echo "NO_K8S"; echo "=== HELM ===" && helm list -n rag 2>/dev/null | grep rag || echo "NO_HELM_RELEASE"
```

Based on what's detected, execute the appropriate shutdown path below. If multiple modes are active (e.g., Docker + library), stop all of them.

## Step 2: Stop Services (Reverse Startup Order)

Stop in this order — reverse of deployment. Only stop what is actually running (detected in Step 1).

### 2a: Optional Services

Stop these first if they are running:

```bash
docker compose -f deploy/compose/docker-compose-nemo-guardrails.yaml down 2>/dev/null; docker compose -f deploy/compose/observability.yaml down 2>/dev/null
```

### 2b: Application Services

```bash
docker compose -f deploy/compose/docker-compose-rag-server.yaml down; docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down
```

### 2c: Vector DB

```bash
docker compose -f deploy/compose/vectordb.yaml down
```

If using Elasticsearch instead of Milvus:
```bash
docker compose -f deploy/compose/vectordb.yaml --profile elasticsearch down
```

### 2d: NIMs (Self-Hosted Only)

Only present if self-hosted deployment was used:

```bash
docker compose -f deploy/compose/nims.yaml down
```

This stops ALL NIM containers (LLM, embedding, ranking, OCR, detection, and any profile-specific NIMs like VLM, audio, nemotron-parse).

### 2e: Library Mode Processes

If library mode is active (detected Python processes):

```bash
pkill -f "nvidia_rag" 2>/dev/null; pkill -f "uvicorn.*rag" 2>/dev/null; docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down 2>/dev/null; docker compose -f deploy/compose/vectordb.yaml down 2>/dev/null
```

### 2f: Kubernetes (Helm) Deployment

If K8s deployment was detected, use the release name and namespace from `helm list` output in step 1:

```bash
helm uninstall <release-name> -n <namespace> 2>/dev/null
```

To also clean up persistent data (only if user requests full cleanup):
```bash
kubectl delete nimcache --all -n <namespace> 2>/dev/null; kubectl delete pvc --all -n <namespace> 2>/dev/null
```

## Step 3: Verify Everything Stopped

```bash
echo "=== REMAINING ===" && docker ps --format "table {{.Names}}\t{{.Status}}" 2>/dev/null; echo "=== K8S ===" && kubectl get pods -n rag 2>/dev/null | head -10 || echo "NOT_K8S"; helm list -n rag 2>/dev/null || true
```

If any RAG-related containers remain, force remove:
```bash
docker ps -a --format "{{.Names}}" | grep -E "(rag|milvus|nim|ingest|redis|nemo|grafana|prometheus|embedding|ranking|vlm|ocr|page-elements|graphic-elements|table-structure)" | xargs -r docker rm -f
```

If pods remain after `helm uninstall`, force delete:
```bash
kubectl delete pods --all -n rag --force --grace-period=0 2>/dev/null
```

## Step 4: Optional Cleanup

Ask the user if they want to clean up data/volumes:

- **Remove Docker volumes** (deletes ingested data, Milvus indices):
  ```bash
  docker volume prune -f
  ```

- **Remove model cache** (frees 100-200 GB for self-hosted):
  ```bash
  rm -rf ~/.cache/model-cache/
  ```

- **Remove Docker images** (frees disk space):
  ```bash
  docker images | grep -E "nvcr.io/nvidia|milvusdb" | awk '{print $3}' | xargs -r docker rmi
  ```

Only perform cleanup if the user explicitly requests it.

## Quick One-Liner (All Docker Services)

If the user wants a fast full teardown:

```bash
cd "$(git rev-parse --show-toplevel)" && \
docker compose -f deploy/compose/docker-compose-nemo-guardrails.yaml down 2>/dev/null; \
docker compose -f deploy/compose/observability.yaml down 2>/dev/null; \
docker compose -f deploy/compose/docker-compose-rag-server.yaml down 2>/dev/null; \
docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down 2>/dev/null; \
docker compose -f deploy/compose/vectordb.yaml down 2>/dev/null; \
docker compose -f deploy/compose/nims.yaml down 2>/dev/null; \
echo "All RAG services stopped."
```

## Source Documentation
- `docs/troubleshooting.md` — if services won't stop or containers hang
