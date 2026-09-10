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

## Configuration

### Behavioral Settings (`~/.hermes/config.yaml`)

Per Hermes project policy, non-secret behavioral configuration belongs in `~/.hermes/config.yaml` under the `kairos:` key:

```yaml
kairos:
  enabled: true
  scan_interval_minutes: 15
  max_proactive_fixes: 3
  require_approval: true
  scan_paths:
    - "."
    - "core"
    - "kairos"
    - "agents"
  max_concurrent_agents: 4
```

### Credentials (`~/.hermes/.env` or `.env`)

Reserve environment variables strictly for secret credentials (such as API keys):

```bash
# Dashboard / Kairos API Key (Required for task submission endpoints on public deployments)
KAIROS_API_KEY=your_secret_api_key_here

# OpenRouter / Provider API Keys
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Cloud / Railway Deployment Settings

For Railway deployment, specify platform-neutral or relative paths in your environment settings:

```bash
SQLITE_DB_PATH=kairos/memory.db
CHROMA_DB_PATH=kairos/chroma_db
LOG_FILE=logs/kairos.log
API_HOST=0.0.0.0
API_PORT=8001
LOG_LEVEL=INFO
```

## Quick Reference

| Setting | Location | Default | Description |
|---------|----------|---------|-------------|
| `kairos.enabled` | `config.yaml` | `true` | Enable proactive problem detection |
| `kairos.scan_interval_minutes` | `config.yaml` | `15` | Minutes between scans |
| `kairos.max_proactive_fixes` | `config.yaml` | `3` | Max fixes without user approval |
| `kairos.max_concurrent_agents` | `config.yaml` | `4` | Maximum parallel agents |
| `KAIROS_API_KEY` | `.env` | `""` | Auth key for dashboard task trigger endpoints |

## Procedure

1. Install the skill: `hermes skills install optional/autonomous-ai-agents/kairos-swarm`
2. Enter the skill directory: `cd optional-skills/autonomous-ai-agents/kairos-swarm`
3. Run the dashboard: `uvicorn backend.dashboard_api:app --reload --port 8001`
4. Open `http://localhost:8001` in your browser