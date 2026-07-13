# RAG Blueprint Deployment

## Phase 1: Environment Analysis

Run this single command to collect all environment information at once:

```bash
echo "=== GPU ===" && nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "NO_GPU"; echo "=== VRAM ===" && nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s "MB total"}' || echo "0MB total"; echo "=== DRIVER ===" && cat /proc/driver/nvidia/version 2>/dev/null | head -1 || echo "NO_DRIVER"; echo "=== CUDA ===" && nvcc --version 2>/dev/null | grep "release" || echo "NO_CUDA_TOOLKIT"; echo "=== DOCKER ===" && docker --version 2>/dev/null || echo "NO_DOCKER"; echo "=== COMPOSE ===" && docker compose version 2>/dev/null || echo "NO_COMPOSE"; echo "=== NVIDIA_TOOLKIT ===" && docker info 2>/dev/null | grep -i "runtimes.*nvidia" || echo "NO_NVIDIA_TOOLKIT"; echo "=== PYTHON ===" && python3 --version 2>/dev/null || echo "NO_PYTHON"; echo "=== DISK ===" && df -h --output=avail / | tail -1; echo "=== OS ===" && cat /etc/os-release 2>/dev/null | grep -E "^(NAME|VERSION)="; echo "=== NGC_KEY ===" && if [ -n "$NGC_API_KEY" ]; then echo "NGC_KEY_SET"; elif [ -n "$NVIDIA_API_KEY" ]; then echo "NVIDIA_KEY_SET"; elif grep -qr "NGC_API_KEY=" deploy/compose/.env deploy/compose/nvdev.env 2>/dev/null | grep -qv "nvapi-your-key"; then echo "DOTENV_SET"; else echo "NOT_SET"; fi; echo "=== RUNNING ===" && docker ps --format "{{.Names}}" 2>/dev/null | grep -E "(rag-server|ingestor-server|nim-llm|milvus)" | head -10 || echo "NO_RUNNING_SERVICES"; echo "=== PORTS ===" && (ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null) | grep -E ":(8081|8082|8090|19530) " || echo "PORTS_FREE"; echo "=== REPO ===" && git rev-parse --show-toplevel 2>/dev/null && git describe --tags 2>/dev/null || echo "NO_GIT_REPO"; echo "=== CACHE ===" && du -sh ~/.cache/model-cache/ 2>/dev/null || echo "NO_CACHE"
```

Present a summary table:

| Check | Result |
|-------|--------|
| GPU(s) | (list with VRAM, or NO_GPU) |
| Total VRAM | (sum in MB/GB) |
| NVIDIA Driver | (version or NO_DRIVER) |
| CUDA Toolkit | (version or NO_CUDA_TOOLKIT) |
| Docker | (version or NO_DOCKER) |
| Docker Compose | (version or NO_COMPOSE) |
| NVIDIA Container Toolkit | (detected or NO_NVIDIA_TOOLKIT) |
| Python | (version or NO_PYTHON) |
| Free disk | (value) |
| OS | (name + version) |
| NGC_API_KEY | ENV_SET / DOTENV_SET / NOT_SET |
| Existing services | (list or none) |
| Port availability | (free or list conflicts) |
| Repo | (tag/branch or NO_GIT_REPO) |
| Model cache | (size or empty) |

### Existing Services Warning

If RAG services are already running, tell the user briefly: "Existing RAG services detected (list). Proceeding will restart them." Continue unless the user objects.

If the user wants to **switch deployment modes** (e.g., NVIDIA-hosted → self-hosted, or Docker → library), shut down the existing deployment first via `references/shutdown.md`, then proceed with the new mode.

If ports are occupied by non-RAG processes, tell the user which ports conflict and suggest stopping the conflicting process. This is a blocker.

## Phase 2: NGC_API_KEY Handling

Check in this order:

1. If `NGC_API_KEY` is set in the shell environment → proceed.
2. If `NVIDIA_API_KEY` is set (common in library mode) → proceed silently.
3. If `NGC_API_KEY` is in `deploy/compose/.env` or `deploy/compose/nvdev.env` (and not the placeholder `nvapi-your-key`) → load it and proceed.
4. If none found → tell the user: "NGC_API_KEY is required. Get one from https://org.ngc.nvidia.com/setup/api-keys and run: `export NGC_API_KEY=\"nvapi-...\"` — then tell me when done."
5. After user confirms → re-check silently. If still not set, write placeholder to `.env` and tell the user to edit it.

## Phase 3: Blocker Checks

Automatically check and report all blockers at once (don't stop at the first one):

Read `docs/support-matrix.md` for current minimum versions and disk requirements, then check:

- **Docker Compose below minimum**: "Upgrade Docker Compose. See https://docs.docker.com/compose/install/linux/"
- **NVIDIA Driver below minimum** (if self-hosted): "Upgrade NVIDIA driver. See `docs/support-matrix.md` for required version."
- **NVIDIA Container Toolkit missing** (and self-hosted needed): "Install NVIDIA Container Toolkit. See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
- **Insufficient disk**: "Check `docs/support-matrix.md` for disk requirements per deployment mode."
- **No Docker and no Python 3.11+**: "Install Docker or Python 3.11+ to proceed."

List all blockers together so the user can fix them in one pass — don't make them fix one, re-run, fix another.

## Phase 4: Route to Deployment Mode

### User explicitly requests a mode
- "library mode" / "lite mode" / "no docker" / "python mode" → read and follow `deploy/library.md`
- "docker" / "self-hosted" / "local" → read and follow `deploy/docker.md` with mode **self-hosted**
- "cloud" / "nvidia-hosted" / "hosted" → read and follow `deploy/docker.md` with mode **nvidia-hosted**
- "retrieval only" / "search only" / "no LLM" → read and follow `deploy/docker.md` with mode **retrieval-only**
- "kubernetes" / "k8s" / "helm" → read and follow `deploy/helm.md`
- "workbench" / "ai workbench" → tell user to follow `deploy/workbench/README.md` (AI Workbench uses its own UI-driven workflow)

### Docker is available (Docker + Compose detected)

**Self-hosted eligible** — read `docs/support-matrix.md` ("Hardware Requirements (Docker)" section) for current GPU requirements. All of the following must also be true:
- GPU count and type matches the Docker self-hosted requirements from the support matrix
- ≥200 GB free disk (per `docs/support-matrix.md` "Disk Space Requirements")
- NVIDIA Container Toolkit detected
- NVIDIA driver meets minimum version from `docs/support-matrix.md` ("Driver Versions")

If self-hosted eligible → read and follow `deploy/docker.md` with mode **self-hosted**

**Otherwise with Docker** → read and follow `deploy/docker.md` with mode **nvidia-hosted**

Tell the user WHY if they have some GPU but not enough:
- "You have [X GPU] with [Y GB] VRAM. Self-hosted requires [requirements from docs/support-matrix.md]. Deploying with NVIDIA-hosted cloud NIMs instead — faster startup, no model download."

### Docker is available but Compose is not

Tell the user: "Docker is installed but Docker Compose is below the minimum version (see `docs/support-matrix.md`). Install it: https://docs.docker.com/compose/install/linux/ — or use library mode instead."

If user chooses library mode → read and follow `deploy/library.md`

### Docker is not available

- Python 3.11+ available → read and follow `deploy/library.md` with mode **lite**
- No Python → tell user to install Python 3.11+ or Docker

## After Deployment

Once deployment completes, verify health:

```bash
echo "=== RAG Server ===" && curl -s http://localhost:8081/v1/health?check_dependencies=true 2>/dev/null || echo "RAG_SERVER_NOT_READY"; echo "=== Ingestor ===" && curl -s http://localhost:8082/v1/health?check_dependencies=true 2>/dev/null || echo "INGESTOR_NOT_READY"
```

If healthy, tell the user:
- "RAG Blueprint is running and healthy."
- "Ask me to configure features like VLM, query rewriting, guardrails, etc."
- "Ask me to shutdown when you're done."

If unhealthy, read `references/troubleshoot.md` and diagnose. Match error output against known issues, fix, and retry. Escalate to the user only if the fix requires their action (API key, data deletion).

## Notebooks
- `notebooks/launchable.ipynb` — Cloud deployment via Brev (alternative to local deployment)

## Source Documentation
- `docs/support-matrix.md` — GPU requirements, driver versions, disk space, supported platforms
- `docs/service-port-gpu-reference.md` — port mappings and GPU assignments for all services
