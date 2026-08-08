---
name: hindsight-memory
description: "Configure, deploy, and troubleshoot Hindsight as the memory backend for Hermes Agent."
version: 1.0.0


metadata:
  hermes:
    tags: [hindsight, memory, hermes, postgresql, docker, vector-database]
    related_skills: [hermes-agent]
---

# Hindsight Memory Backend

Set up Hindsight as the persistent memory provider for Hermes Agent. Covers the three deployment modes, common pitfalls (ARM64, pg_dump compatibility, plugin requirements), and step-by-step setup procedures.

## Deployment Modes

Hindsight supports three deployment modes. Choose based on infrastructure:

| Mode | Requirements | Best For |
|------|-------------|----------|
| **Cloud** | API key from ui.hindsight.vectorize.io | Quick setup, no local infra |
| **Local Embedded** | hindsight-embed pip package, local PostgreSQL | Full control, no external deps |
| **Local External** | Docker (AMD64) or native PostgreSQL | Self-hosted, existing DB infra |

## Cloud Mode Setup (Recommended for ARM64)

The simplest path. Works on any architecture.

1. Get API key from https://ui.hindsight.vectorize.io
2. Set env vars:
   ```bash
   # In ~/.hermes/.env
   HINDSIGHT_API_KEY=hsk_your_key_here
   ```
3. Configure Hermes:
   ```bash
   hermes config set memory.provider hindsight
   hermes memory setup
   ```
4. Verify: `hermes memory status` should show `available`

## Local Embedded Setup

Runs a daemon thread inside the Hermes process. Needs PostgreSQL locally.

### Prerequisites

```bash
# Install PostgreSQL + pgvector
sudo apt-get install -y postgresql-16 postgresql-16-pgvector

# Install pip packages into Hermes venv
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python \
  hindsight-client hindsight-embed
```

### Plugin Patch Required (ARM64 workaround)

The Hindsight plugin's `_check_local_runtime()` imports both `hindsight` and `hindsight_embed`. The `hindsight` module (providing `HindsightEmbedded`) is NOT on public PyPI — only `hindsight-client` and `hindsight-embed` are. On systems without the full `hindsight` package, patch the check:

**Option 1: Remove import check** (ARM64 compatibility)
File: `~/.hermes/hermes-agent/plugins/memory/hindsight/__init__.py`
Find `_check_local_runtime()` and remove the `importlib.import_module("hindsight")` check, keeping only the `hindsight_embed.daemon_embed_manager` import.

**Option 2: Use hindsight_client instead of embedded daemon** (verified working on ARM64)
File: `~/.hermes/hermes-agent/plugins/memory/hindsight/__init__.py`
In `_get_client()` method, replace the HindsightEmbedded instantiation with hindsight_client:

```python
# OLD (fails on ARM64 because 'hindsight' module not in PyPI):
from hindsight import HindsightEmbedded
self._client = HindsightEmbedded(base_url=self._api_url, bank_id=self._bank_id, budget=self._budget)

# NEW (uses HTTP client to daemon, works on ARM64):
from hindsight_client import Hindsight
self._client = Hindsight(base_url=self._api_url, bank_id=self._bank_id, budget=self._budget, timeout=self._timeout)
```

Then set API URL to local daemon:
```bash
hermes config set hindsight.api_url http://localhost:8888
```

> **Pitfall:** The `DaemonEmbedManager.ensure_running(config, profile)` blocks indefinitely on ARM64 Ubuntu (tested 2026-06-05). Do not try to start the embed manager manually. Use `hindsight_client` HTTP client instead.
> **Pitfall:** `ensure_running()` requires `config` dict and `profile` string — missing these causes TypeError.
> **Pitfall:** Even with correct args, the daemon manager times out after 60s on ARM64 (likely waiting for `hindsight` module that doesn't exist).

### PostgreSQL Setup

```bash
# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database
sudo -u postgres psql -c "CREATE USER hindsight WITH PASSWORD 'hindsight' SUPERUSER;"
sudo -u postgres psql -c "CREATE DATABASE hindsight OWNER hindsight;"

# Enable pgvector
sudo -u postgres psql -d hindsight -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Configuration

```bash
hermes config set memory.provider hindsight
hermes config set hindsight.mode local_embedded
hermes config set hindsight.api_url http://localhost:8888  # For hindsight_client HTTP mode
```

## Local External (Docker) Setup

**IMPORTANT:** The Docker image `inhindsight/hindsight:latest` is AMD64-only. It will NOT run on ARM64 VMs (Oracle Cloud Ampere, AWS Graviton, Apple Silicon). Even `--platform linux/amd64` fails if QEMU binfmt is not configured.

On ARM64, use Cloud mode or Local Embedded instead.

On AMD64:
```bash
docker run -d \
  --name hindsight \
  --restart unless-stopped \
  -p 8888:80 \
  -v hindsight_data:/opt/app/data \
  inhindsight/hindsight:latest
```

## Importing Existing Data

When migrating a Hindsight database dump, see `references/pg_dump-compatibility.md` for version compatibility issues and workarounds.

Key points:
- pg_dump 18 dumps use `\restrict` which psql 16 can't handle
- COPY data with `\N` NULL markers breaks without `\restrict` protection
- Workaround: parse COPY blocks with Python/psycopg2 and generate INSERT statements
- Always install `postgresql-16-pgvector` before importing (tables use `public.vector` type)
- Check actual PostgreSQL port with `sudo pg_lsclusters` — Ubuntu often installs new versions on 5433

## Storing and Recalling Memories Programmatically

The provider exposes `handle_tool_call`, NOT direct methods like `retain` or `recall`.

```python
import sys, os
os.environ['HOME'] = '/home/ubuntu'
os.environ['HERMES_HOME'] = '/home/ubuntu/.hermes'
sys.path.insert(0, '/home/ubuntu/.hermes/hermes-agent')

from plugins.memory.hindsight import HindsightMemoryProvider
p = HindsightMemoryProvider()
p.initialize(session_id='setup')

# Store a memory
result = p.handle_tool_call('hindsight_retain', {
    'content': 'Key fact to remember...',
    'tags': ['tag1', 'tag2']
})
# → {"result": "Memory stored successfully."}

# Recall memories
result = p.handle_tool_call('hindsight_recall', {
    'query': 'search terms here',
    'limit': 5
})
# → {"result": "1. Fact one\n2. Fact two\n..."}

# Reflect on accumulated memories
result = p.handle_tool_call('hindsight_reflect', {
    'query': 'what patterns exist about X'
})
```

This is useful for seeding user profile, project context, or preferences into Hindsight during initial setup.

## Writing API Keys to .env Safely

**NEVER use shell `echo`, `cat >>`, or heredoc to write Hindsight API keys to `.env`.** Shell interpolation truncates or mangles the key. The key must be ~53 chars starting with `hsk_`.

Always use Python:
```python
import os
env_path = os.path.expanduser('~/.hermes/.env')
api_key = 'hsk_full_key_here'  # pass the real key, not shell-interpolated

with open(env_path, 'r') as f:
    lines = f.readlines()
lines = [l for l in lines if not l.startswith('HINDSIGHT_')]
lines.append(f'HINDSIGHT_API_KEY={api_key}\n')
lines.append(f'HINDSIGHT_BANK_ID=hermes\n')
with open(env_path, 'w') as f:
    f.writelines(lines)
```

## Sharing Memory Across Machines via Cloudflare Tunnel

See the canonical **`cloudflare-tunnel-testing`** skill (general tunnel setup + 502 diagnostics),
and its `references/hindsight-tunnel.md` for the full Docker compose snippet and network connectivity details.

### Hindsight-Specific Client Config (Remote Machine)

```bash
# ~/.hermes/.env on the remote machine
HINDSIGHT_API_URL=https://<tunnel-url>
HINDSIGHT_API_KEY=local          # literal string "local" for Docker Hindsight, not a placeholder
HINDSIGHT_BANK_ID=<same bank ID from server>
```

```bash
hermes config set memory.provider hindsight
hermes config set hindsight.mode cloud
```

> **Pitfall:** `HINDSIGHT_API_KEY=local` is the actual auth token the local Docker image expects — do not replace with a cloud key.
> **Pitfall:** Config may report `mode: cloud` even for local Docker. Check `HINDSIGHT_API_URL` to tell them apart.

## Verifying Setup

```bash
hermes memory status        # Should show "available"
hermes memory setup         # Reconfigure if needed
```

Test via Python:
```python
import sys, os
os.environ['HOME'] = '/home/ubuntu'
os.environ['HERMES_HOME'] = '/home/ubuntu/.hermes'
sys.path.insert(0, '/home/ubuntu/.hermes/hermes-agent')

from plugins.memory.hindsight import HindsightMemoryProvider
p = HindsightMemoryProvider()
print('Available:', p.is_available())
p.initialize(session_id='test')
print('Client:', p._client is not None)
```

## Troubleshooting

### "exec format error" running Docker
ARM64 host. Docker image `inhindsight/hindsight:latest` is AMD64-only. Even `--platform linux/amd64` fails with `/usr/bin/bash: stat /usr/bin/bash: no such file or directory` if QEMU binfmt is not configured. Use Cloud mode or Local Embedded instead.

### "No module named 'hindsight'"
The full `hindsight` package is not on public PyPI. Only `hindsight-client` and `hindsight-embed` are public. Patch `_check_local_runtime()` in the plugin (see above).

### "DaemonEmbedManager.ensure_running() blocks/times out" (ARM64)
On ARM64 Ubuntu (tested 2026-06-05), `DaemonEmbedManager.ensure_running(config, profile)` blocks indefinitely or times out after 60s. Do not try to use the embedded daemon manager on ARM64. Use the HTTP client workaround instead:
- Patch `_get_client()` to use `hindsight_client.Hindsight` instead of `hindsight.HindsightEmbedded`
- Set `hermes config set hindsight.api_url http://localhost:8888`

### "Cannot connect to host localhost:8888" after patch
The `DaemonEmbedManager` did not start successfully. The HTTP client approach requires a running daemon on port 8888. On ARM64, starting the daemon via `ensure_running()` is unreliable. Consider:
1. Running Hindsight in Docker on a separate AMD64 machine with Cloudflare tunnel
2. Using Cloud mode with API key from ui.hindsight.vectorize.io
3. Disabling Hindsight temporarily (`hermes config set memory.provider builtin`)

### "Invalid API key format" (Cloud)
The API key was truncated or corrupted in `.env`. Verify the full key is written (should be ~53 chars starting with `hsk_`). Write it with Python to avoid shell interpolation issues.

### PostgreSQL "password authentication failed"
Common causes:
1. PostgreSQL installed on port 5433 (not 5432) when another version exists — check with `pg_lsclusters`
2. `password_encryption = scram-sha-256` but `pg_hba.conf` uses `md5` — change one to match the other
3. Simplest fix: set `pg_hba.conf` local+host lines to `trust` for development VMs

### PostgreSQL port not 5432
Run `sudo pg_lsclusters` to find the actual port. Ubuntu installs PostgreSQL 16 on 5433 if another version is already using 5432.

### pgvector extension missing
```bash
sudo apt-get install -y postgresql-16-pgvector
sudo -u postgres psql -d hindsight -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Key Files

| File | Purpose |
|------|---------|
| `~/.hermes/.env` | API keys (HINDSIGHT_API_KEY) |
| `~/.hermes/hindsight/config.json` | Mode, bank_id, budget settings |
| `~/.hermes/hermes-agent/plugins/memory/hindsight/` | Plugin source |
| `~/.hermes/config.yaml` | memory.provider, hindsight.mode |

## Reference

- `references/pg_dump-compatibility.md` — importing pg_dump 18 dumps into psql 16
- `cloudflare-tunnel-testing` skill and `references/hindsight-tunnel.md` — exposing Hindsight via Cloudflare Tunnel for multi-machine access
