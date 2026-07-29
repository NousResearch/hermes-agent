# RAG Library Mode Setup

## Determine Mode

If routed here from the deploy workflow, the mode (full or lite) may already be decided. Use it.

If invoked directly, auto-detect:

```bash
echo "=== DOCKER ===" && docker --version 2>/dev/null || echo "NO_DOCKER"; echo "=== GPU ===" && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "NO_GPU"; echo "=== PYTHON ===" && python3 --version 2>/dev/null || echo "NO_PYTHON"; echo "=== PKG_MANAGER ===" && which uv 2>/dev/null && echo "UV_AVAILABLE" || (which pip3 2>/dev/null && echo "PIP_AVAILABLE" || echo "NO_PKG_MANAGER"); echo "=== VENV ===" && ls -d .venv/ venv/ nvidia-rag-env/ 2>/dev/null || echo "NO_EXISTING_VENV"; echo "=== INSTALLED ===" && pip3 show nvidia_rag 2>/dev/null | head -3 || echo "NOT_INSTALLED"
```

- Docker available → **full** (Python API + Docker backend services)
- No Docker or user explicitly says "lite" / "no docker" / "containerless" → **lite**

Auto-route based on Docker availability. Only ask if both modes are equally valid.

## Verify NGC_API_KEY

Auto-check all locations:

```bash
if [ -n "$NGC_API_KEY" ]; then echo "NGC_KEY_SET"; elif [ -n "$NVIDIA_API_KEY" ]; then echo "NVIDIA_KEY_SET"; else echo "NOT_SET"; fi
```

If NOT_SET: ask the user. Otherwise proceed silently.

## Deploy

Based on the mode:

- **Full**: read and follow `library-full.md`
- **Lite**: read and follow `library-lite.md`

## On Success

Tell the user:
- Which mode was set up and how to start using it (notebook or Python script)
- "Ask me to configure features, change models, etc."
- "Ask me to shutdown backend services when done (if full mode)."

## On Error

1. Read the error output (pip install failure, import error, service connection error).
2. Read `references/troubleshoot.md` to match against common issues.
3. Common fixes to try:
   - `pip install` failure → try `uv pip install` or check Python version ≥3.11.
   - Import error → check if virtual environment is activated.
   - Connection error to backend services → check Docker containers are running.
4. Retry the failed step after fixing.
5. If still failing, report the specific error to the user.

## Source Documentation
- `docs/python-client.md` — Python library API, installation, full and lite mode setup
