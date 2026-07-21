# RAG Docker Deployment

## Determine Mode

If routed here from the deploy workflow, the mode (self-hosted, nvidia-hosted, or retrieval-only) was already decided. Use it.

If invoked directly without a mode, auto-detect:

```bash
echo "=== COMPOSE ===" && docker compose version 2>/dev/null || echo "NO_COMPOSE"; echo "=== GPU ===" && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "NO_GPU"; echo "=== DISK ===" && df -h --output=avail / | tail -1; echo "=== RUNNING ===" && docker ps --format "{{.Names}}" 2>/dev/null | grep -E "(rag-server|ingestor-server|nim-llm|milvus)" | head -10 || echo "NONE_RUNNING"
```

If NO_COMPOSE: stop and tell the user to install Docker Compose (see `docs/support-matrix.md` for minimum version).

Read `docs/support-matrix.md` ("Hardware Requirements (Docker)" section) for current GPU requirements, then:
- GPU count/type meets self-hosted requirements from the support matrix, and 200+ GB free disk → **self-hosted**
- Any GPU or no GPU with ≥50 GB free disk → **nvidia-hosted**
- User explicitly says "retrieval only" / "no LLM" / "search only" → **retrieval-only**

Auto-route based on hardware. Only ask if two modes are equally valid and the user's intent is ambiguous.

## Verify NGC_API_KEY

Auto-check all possible locations before asking:

```bash
[ -n "$NGC_API_KEY" ] && echo "ENV_SET" || (grep -qr "NGC_API_KEY=" deploy/compose/.env deploy/compose/nvdev.env 2>/dev/null | grep -qv "nvapi-your-key" && echo "DOTENV_SET" || echo "NOT_SET")
```

- **ENV_SET**: proceed silently.
- **DOTENV_SET**: load the env file that contains the key and proceed.
- **NOT_SET**: ask the user to provide it. This is the only thing to ask for.

## Docker Login

Auto-check if already logged in:

```bash
grep -q "nvcr.io" ~/.docker/config.json 2>/dev/null && echo "ALREADY_LOGGED_IN" || echo "NOT_LOGGED_IN"
```

If already logged in → proceed silently.

If not logged in → tell the user to run this themselves (the key gets expanded in agent logs):

> Please run in your terminal: `echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin`

Wait for confirmation only if login was needed.

## Deploy

Based on the mode, read and follow the appropriate reference:

- **Self-hosted**: read and follow `docker-self-hosted.md`
- **NVIDIA-hosted**: read and follow `docker-nvidia-hosted.md`
- **Retrieval-only**: read and follow `docker-retrieval-only.md`

## Post-Deploy Verification

Run health checks:

```bash
sleep 5; echo "=== RAG ===" && curl -s http://localhost:8081/v1/health?check_dependencies=true 2>/dev/null || echo "RAG_NOT_READY"; echo "=== INGESTOR ===" && curl -s http://localhost:8082/v1/health?check_dependencies=true 2>/dev/null || echo "INGESTOR_NOT_READY"; echo "=== CONTAINERS ===" && docker ps --format "table {{.Names}}\t{{.Status}}" 2>/dev/null | grep -E "(rag|milvus|nim|ingest)" | head -15
```

If services are still initializing, automatically poll every 30 seconds:
- **NVIDIA-hosted**: poll until healthy or 5 minutes elapsed (no model downloads needed).
- **Self-hosted**: poll until healthy or 15 minutes elapsed (model downloads on first run).
- **Retrieval-only**: poll until healthy or 5 minutes elapsed.

Show progress to the user during polling.

## On Success

Tell the user:
- "RAG Blueprint is running and healthy. Open http://localhost:8090 to use the UI." (skip for retrieval-only)
- "Ask me to configure features (VLM, query rewriting, guardrails, etc.)"
- "Ask me to shutdown when you're done."

## On Error

1. Read the error output from the failed command.
2. Read `references/troubleshoot.md` to match against common issues (port conflict, disk full, NGC auth, GPU OOM).
3. Apply the fix and retry.
4. If still failing, report the specific error to the user with the fix that was attempted.

## Source Documentation
- `docs/support-matrix.md` — GPU requirements, hardware compatibility, disk space
