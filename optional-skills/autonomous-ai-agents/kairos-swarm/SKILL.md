---
name: kairos-swarm
description: Kairos proactive problem detection and multi-agent swarm orchestration with Railway deployment.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [swarm, orchestration, Railway, kairos]
    related_skills: [hermes-agent]
prerequisites:
  pip: [fastapi, uvicorn[standard], pydantic]
---

# Kairos Swarm Skill

Proactive problem detection and multi-agent swarm orchestration system with Railway deployment support.

## When to Use

- Running proactive background scans to detect and fix issues automatically
- Multi-agent task orchestration with concurrent agents
- Deploying a futuristic 3D dashboard to Railway or similar cloud platforms

## Environment Configuration

Add these lines to your `.env` file (or set them in Railway):

```bash
# ============================================================
# KAIROS DAEMON (Proactive Problem Detection)
# ============================================================
KAIROS_ENABLED=true
KAIROS_SCAN_INTERVAL_MINUTES=15
KAIROS_MAX_PROACTIVE_FIXES=3
KAIROS_REQUIRE_APPROVAL=true
KAIROS_SCAN_PATHS=.,core,kairos,agents

# ============================================================
# RAILWAY / CLOUD DEPLOYMENT NOTES
# ============================================================
# Do not use Windows absolute paths like C:\Users\... or D:\hermes\... in this file.
# For Railway, use relative paths or platform-neutral paths such as:
SQLITE_DB_PATH=kairos/memory.db
CHROMA_DB_PATH=kairos/chroma_db
CHROMA_COLLECTION=kairos_knowledge
LOG_FILE=logs/kairos.log
PIP_CACHE_DIR=/tmp/pip_cache
TMPDIR=/tmp
TMP=/tmp

# Backend Server Configuration
API_HOST=0.0.0.0
API_PORT=8001
API_RELOAD=true

LOG_LEVEL=INFO
```

## Deployment to Railway

1. Install the skill: `hermes skills install optional/autonomous-ai-agents/kairos-swarm`
2. Fork the repository to your GitHub account
3. Create a new Railway project from your fork
4. Set the environment variables above in Railway's dashboard
5. Railway auto-detects the Procfile and deploys

## Quick Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `KAIROS_ENABLED` | `true` | Enable proactive problem detection |
| `KAIROS_SCAN_INTERVAL_MINUTES` | `15` | Minutes between scans |
| `KAIROS_MAX_PROACTIVE_FIXES` | `3` | Max fixes without user approval |
| `MAX_CONCURRENT_AGENTS` | `4` | Maximum parallel agents |

## Procedure

1. Install the skill: `hermes skills install optional/autonomous-ai-agents/kairos-swarm`
2. Enter the skill directory: `cd skills/optional/autonomous-ai-agents/kairos-swarm`
3. Run the dashboard: `uvicorn backend.dashboard_api:app --reload --port 8001`
4. Open `http://localhost:3000` in your browser