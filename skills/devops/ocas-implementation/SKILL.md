---
name: ocas-implementation
description: >
  Complete implementation workflow for OCAS skills from specification to
  running system. Covers initializing skill data structures, implementing
  Google API integrations with OAuth, configuring MCP servers, creating
  sync scripts, and setting up automated cron jobs. Use when implementing
  a new OCAS skill that needs external API access or recurring automated
  tasks.
metadata:
  author: Indigo Karasu
  email: mx.indigo.karasu@gmail.com
  version: \"2.0.0\"
  hermes:
    tags: [implementation, ocas, automation, architecture]
    category: devops
---

# OCAS Implementation

Complete workflow for implementing OCAS skills from specification to running automated system. 

## Architecture Compliance (MANDATORY)

When modifying or building OCAS skills, you MUST follow the architecture specifications from the `indigokarasu/ocas-architecture` repository.

### Core Pathing Rule
**Never hardcode paths like `~/openclaw/` or `~/.hermes/` in the code.**
Use dynamic agent-paths (e.g., `{agent_root}`) to ensure the skill remains portable across different agent environments.

### Key Architecture Specifications
1. **`ocas-skill-authoring-rules.md`**: The master rules. Every OCAS skill must comply (One sharp promise, SKILL.md as operational surface, atomic skill principle).
2. **`spec-ocas-storage-conventions.md`**: All persistent data must use the logical separation of `data/`, `journals/`, and `db/` relative to the agent root.
3. **`spec-ocas-shared-schemas.md`**: Use canonical schemas (DecisionRecord, Signal, etc.) to ensure inter-skill interoperability.
4. **`spec-ocas-ontology.md`**: Align entity types and extraction ownership with the global ontology.
5. **`spec-ocas-interfaces.md`**: Communicate via defined intake directories, not direct calls.
6. **`spec-ocas-journal.md`**: Every run must write a journal of the correct type (Observation, Action, or Research).

## Implementation workflow

### 1. Initialize skill data structures

Every OCAS skill needs:
- Data directory: `{agent_root}/data/{skill-name}/`
- Config file: `config.json` with full default configuration
- JSONL files: `signals.jsonl`, `items.jsonl`, `decisions.jsonl`, `extractions.jsonl`
- Subdirectories: `reports/`, `journals/`
- Journal directory: `{agent_root}/journals/{skill-name}/`

```bash
# Example of dynamic initialization (replace {agent_root})
mkdir -p {agent_root}/data/ocas-{skill}/reports
mkdir -p {agent_root}/journals/ocas-{skill}

# Create empty JSONL files
touch {agent_root}/data/ocas-{skill}/signals.jsonl
touch {agent_root}/data/ocas-{skill}/items.jsonl
touch {agent_root}/data/ocas-{skill}/decisions.jsonl
touch {agent_root}/data/ocas-{skill}/extractions.jsonl
```

### 2. Google API OAuth integration

**Check existing token scopes**:
```bash
cat {agent_root}/google_token.json
```

**Common scope mismatches**:
- Specification requests: `gmail.readonly`, `calendar.readonly`
- Token actually has: `gmail.modify`, `calendar`

**Solution**: Use the scopes that match the existing token.

**Initialize Google services**:
```python
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from pathlib import Path

token_path = Path.home() / \".hermes\" / \"google_token.json\" # Replace with agent root logic

creds = Credentials.from_authorized_user_file(
    str(token_path),
    ['https://www.googleapis.com/auth/gmail.modify', 
     'https://www.googleapis.com/auth/calendar']
)
# ... (standard refresh logic)
```

### 3. MCP server configuration

**Add MCP server to Hermes**:
```bash
# Use the hermes mcp command
hermes mcp add {server_name} --command npx --args @package/mcp-server --auth header
```
*Note: Environment variables must be placed in the `config.yaml` under the MCP server's `env` section for persistence.*

### 4. Create sync scripts

**Script Structure**: ensure all paths are relative to the agent root.
```python
class {Skill}Sync:
    def __init__(self, data_dir: str = None):
        # Use a provided path or resolve dynamically via agent root
        self.data_dir = Path(data_dir) if data_dir else resolve_agent_root() / \"data\" / \"ocas-{skill}\"
```

### 5. Set up cron jobs

**Register cron jobs**:
```bash
# Example: Daily sync at midnight
hermes cron create --name {skill}:daily --skill ocas-{skill} \"0 0 * * *\" \"python3 {agent_root}/scripts/{skill}_sync.py sync 7\"
```

## Common pitfalls

### OAuth scope mismatch
**Symptom**: `invalid_scope: Bad Request`
**Solution**: Check `google_token.json` and use the exact scopes listed there.

### hardcoded paths
**Symptom**: Skills fail when moved or installed on new environments.
**Solution**: Use `{agent_root}` or a configuration variable; never use `~/openclaw/` or `~/.hermes/` literally in the code.

## Verification steps

1. **Architecture Audit**: Does the skill comply with `ocas-skill-authoring-rules.md`?
2. **Path Test**: Do all files resolve correctly using the agent root?
3. **Google API**: Test with a simple query to confirm OAuth works.
4. **MCP server**: Run `hermes mcp test {server}`.
5. **Cron jobs**: Verify jobs appear in `hermes cron list`.

## Example: Taste skill implementation
The Taste skill was refined by merging implementation details into the main skill and updating paths to be dynamic relative to the agent's root, ensuring compliance with OCAS architecture specs.