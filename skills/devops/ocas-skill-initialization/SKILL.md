---
name: ocas-skill-initialization
description: >
  OCAS Skill Initialization: Complete setup for OCAS skills including config,
  data files, directories, Python venv, dependencies, and cron jobs. Use when
  initializing a new OCAS skill or completing partial initialization. Handles
  skill data directory structure, JSONL file creation, virtual environment
  setup, dependency installation, and cron job registration.
metadata:
  author: Indigo Karasu
  email: mx.indigo.karasu@gmail.com
  version: "1.0.0"
  hermes:
    tags: [ocas, initialization, setup, devops]
    category: devops
---

# OCAS Skill Initialization

Complete setup for OCAS skills including config, data files, directories, Python venv, dependencies, and cron jobs.

## When to Use

- Initializing a new OCAS skill from scratch
- Completing partial initialization (missing JSONL files, directories, cron jobs)
- Setting up Python virtual environment for skill scripts
- Registering cron jobs for scheduled tasks
- Integrating MCP servers with OCAS skills

## When Not to Use

- Skill already fully initialized and operational
- Modifying existing skill logic (use skill_manage patch instead)
- Installing system-wide packages (use terminal directly)

## Prerequisites

- Hermes CLI with cron support
- Python 3.11+ with venv support
- Google OAuth token (if using Google APIs)
- MCP server credentials (if integrating MCP)

## Initialization Pattern

### 1. Verify Skill State

Check what exists:
```bash
# Check data directory
ls -la ~/.hermes/commons/data/{skill-name}/

# Check config
cat ~/.hermes/commons/data/{skill-name}/config.json

# Check JSONL files
ls -la ~/.hermes/commons/data/{skill-name}/*.jsonl

# Check cron jobs
hermes cron list | grep {skill-name}
```

### 2. Create Directory Structure

```bash
# Create data directory
mkdir -p ~/.hermes/commons/data/{skill-name}/

# Create subdirectories
mkdir -p ~/.hermes/commons/data/{skill-name}/reports/
mkdir -p ~/.hermes/commons/data/{skill-name}/music/  # or other domain-specific dirs

# Create journal directory
mkdir -p ~/.hermes/commons/journals/{skill-name}/
```

### 3. Write Full Config

Write complete config.json with all default fields. Never leave it as minimal initialized-only config.

Example structure:
```json
{
  "skill_id": "ocas-{skill-name}",
  "skill_version": "1.0.0",
  "config_version": "1",
  "created_at": "ISO-8601 timestamp",
  "updated_at": "ISO-8601 timestamp",
  "domains": {
    "enabled": ["domain1", "domain2"]
  },
  "retention": {
    "days": 0,
    "max_records": 10000
  },
  "skill_specific_config": {}
}
```

### 4. Create JSONL Data Files

```bash
cd ~/.hermes/commons/data/{skill-name}/

# Create all required JSONL files
touch signals.jsonl
touch items.jsonl
touch links.jsonl
touch decisions.jsonl
touch extractions.jsonl
```

Common JSONL files for OCAS skills:
- `signals.jsonl` — Consumption or activity signals
- `items.jsonl` — Item records (entities, tracks, venues)
- `links.jsonl` — Cross-domain connections
- `decisions.jsonl` — Decision records
- `extractions.jsonl` — Raw extractions from external sources

### 5. Setup Python Virtual Environment

```bash
cd ~/.hermes/commons/data/{skill-name}/

# Install venv package if needed
apt update && apt install -y python3.13-venv

# Create virtual environment
python3 -m venv venv

# Activate and install dependencies
source venv/bin/activate
pip install {dependency1} {dependency2}
```

Common dependencies:
- `spotipy` — Spotify API
- `google-api-python-client` — Google APIs (Gmail, Calendar, Drive)
- `requests` — HTTP requests
- `python-dateutil` — Date parsing

### 6. Create Scripts Directory

```bash
mkdir -p ~/.hermes/commons/data/{skill-name}/scripts/

# Make scripts executable
chmod +x ~/.hermes/commons/data/{skill-name}/scripts/*.py
```

### 7. Register Cron Jobs

Use `hermes cron create` with proper syntax:

```bash
# Basic cron job
hermes cron create --name {skill-name}:task --skill {skill-name} "0 6 * * *" "command"

# Cron job with venv activation
hermes cron create --name {skill-name}:scan --skill {skill-name} "0 6 * * *" "cd /root/.hermes/commons/data/{skill-name} && source venv/bin/activate && python3 scripts/scan.py"

# Remove existing job
hermes cron remove {job_id}

# List jobs
hermes cron list | grep {skill-name}
```

Cron schedule format:
- `"0 6 * * *"` — Daily at 6am UTC
- `"*/30 * * * *"` — Every 30 minutes
- `"0 0 * * 1-5"` — Weekdays at midnight

### 8. Integrate MCP Server (if needed)

#### Discover Existing MCP Servers

```bash
# Check installed MCP servers
find ~/.hermes/node/lib/node_modules -name "*mcp*" -type d

# Check MCP config
ls -la ~/.hermes/mcp/
```

#### Create MCP Config

Create `/root/.hermes/mcp/{service}-mcp.json`:
```json
{
  "{service}": {
    "command": "node",
    "args": ["/path/to/mcp/server/build/bin.js"],
    "env": {
      "CLIENT_ID": "${CLIENT_ID}",
      "CLIENT_SECRET": "${CLIENT_SECRET}"
    }
  }
}
```

#### Call MCP Tools from Python

```python
import subprocess

def call_mcp_tool(service, tool_name, args=None):
    """Call MCP tool via hermes mcp."""
    cmd = ["hermes", "mcp", "call", service, tool_name]
    if args:
        for key, value in args.items():
            cmd.extend([f"--{key}", str(value)])
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    return result.stdout
```

#### Common MCP Tools

**Spotify MCP** (`@darrenjaws/spotify-mcp`):
- `get_user_profile` — User profile info
- `get_recently_played` — Recently played tracks
- `get_top_items` — Top tracks/artists

**Tavily MCP**:
- `tavily-search` — Web search

### 9. Google OAuth Integration

Use existing Google token at `~/.hermes/google_token.json`:

```python
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

TOKEN_PATH = Path.home() / ".hermes" / "google_token.json"

def get_gmail_service():
    """Get authenticated Gmail service."""
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build('gmail', 'v1', credentials=creds)
```

## Common Pitfalls

### 1. Incomplete Config
**Problem**: Config only has `initialized: true` flag
**Solution**: Write full config with all default fields from skill spec

### 2. Missing JSONL Files
**Problem**: Scripts fail because JSONL files don't exist
**Solution**: Create all required JSONL files with `touch` command

### 3. Venv Installation Fails
**Problem**: `python3 -m venv` fails with "ensurepip not available"
**Solution**: Install venv package first: `apt install python3.13-venv`

### 4. Cron Job Syntax Errors
**Problem**: Cron jobs not created or fail to run
**Solution**: Use proper `hermes cron create` syntax with `--name`, `--skill`, schedule, and command

### 5. MCP Not Found
**Problem**: MCP server not accessible
**Solution**: Check MCP config at `~/.hermes/mcp/`, verify server path, test with `hermes mcp call`

### 6. Google OAuth Token Missing
**Problem**: Google API calls fail
**Solution**: Run Google Workspace setup to create token at `~/.hermes/google_token.json`

## Verification Checklist

After initialization, verify:

- [ ] Data directory exists: `~/.hermes/commons/data/{skill-name}/`
- [ ] Full config.json with all fields
- [ ] All JSONL files created (empty is OK)
- [ ] Subdirectories created (reports/, music/, etc.)
- [ ] Python venv created and dependencies installed
- [ ] Scripts directory exists and scripts are executable
- [ ] Cron jobs registered and visible in `hermes cron list`
- [ ] MCP config created (if using MCP)
- [ ] Google OAuth token exists (if using Google APIs)
- [ ] Manual test run succeeds

## Testing

Run a manual test:
```bash
cd ~/.hermes/commons/data/{skill-name}
source venv/bin/activate
python3 scripts/main_script.py
```

Test cron job:
```bash
# Get job ID from hermes cron list
hermes cron run {job_id}
```

Test MCP:
```bash
hermes mcp call {service} {tool_name}
```

## Example: Complete Taste Skill Initialization

```bash
# 1. Create directories
mkdir -p ~/.hermes/commons/data/ocas-taste/reports/
mkdir -p ~/.hermes/commons/data/ocas-taste/music/
mkdir -p ~/.hermes/commons/journals/ocas-taste/

# 2. Write full config
cat > ~/.hermes/commons/data/ocas-taste/config.json << 'EOF'
{
  "skill_id": "ocas-taste",
  "skill_version": "3.0.0",
  "config_version": "2",
  "created_at": "2026-04-10T01:15:26+00:00",
  "updated_at": "2026-04-11T05:53:00+00:00",
  "domains": {
    "enabled": ["music", "restaurant", "book", "movie", "product", "travel", "event"]
  },
  "email_scan": {
    "enabled": true,
    "last_scan_timestamp": null
  }
}
EOF

# 3. Create JSONL files
cd ~/.hermes/commons/data/ocas-taste/
touch signals.jsonl items.jsonl links.jsonl decisions.jsonl extractions.jsonl

# 4. Setup venv
apt install -y python3.13-venv
python3 -m venv venv
source venv/bin/activate
pip install spotipy google-api-python-client

# 5. Create scripts
mkdir -p scripts/
# ... write scripts ...

# 6. Register cron jobs
hermes cron create --name taste:scan --skill ocas-taste "0 6 * * *" "cd /root/.hermes/commons/data/ocas-taste && source venv/bin/activate && python3 scripts/taste_scan.py"
hermes cron create --name taste:sync-spotify --skill ocas-taste "0 0 * * *" "cd /root/.hermes/commons/data/ocas-taste && source venv/bin/activate && python3 scripts/spotify_mcp_sync.py"

# 7. Setup MCP
cat > ~/.hermes/mcp/spotify-mcp.json << 'EOF'
{
  "spotify": {
    "command": "node",
    "args": ["/root/.hermes/node/lib/node_modules/@darrenjaws/spotify-mcp/build/bin.js"],
    "env": {
      "SPOTIFY_CLIENT_ID": "${SPOTIFY_CLIENT_ID}",
      "SPOTIFY_CLIENT_SECRET": "${SPOTIFY_CLIENT_SECRET}"
    }
  }
}
EOF
```

## Related Skills

- `google-workspace-setup` — Google OAuth and service account setup
- `python-package-installation` — Python package installation patterns
- `webhook-subscriptions` — Event-driven task setup
- `ocas-taste` — Example of fully initialized OCAS skill

## References

- Hermes CLI documentation: `hermes --help`
- Cron job management: `hermes cron --help`
- MCP integration: `hermes mcp --help`
- OCAS skill specification: See individual skill SKILL.md files