# JiuwenMemory Server PoC Persistence

The Hermes `jiuwenmemory` plugin only calls the JiuwenMemory HTTP API. The
current server-side Graph/Turbo/Swarm/Dreaming/session_id PoC is patched into
the installed JiuwenMemory package, typically:

```text
venv/lib/python3.11/site-packages/jiuwen_memory/server/memory_server.py
```

That file is outside the Hermes plugin source and can be replaced by a package
reinstall. Keep this note with the plugin so the operational state is visible
in source control without changing runtime behavior.

## Audit

Verify that the installed server still contains the expected PoC routes,
request models, session_id fields, helper functions, and local persistence
paths:

```bash
python scripts/jiuwenmemory_server_audit.py verify
python scripts/jiuwenmemory_server_audit.py verify --json
```

Use `--server-file /path/to/memory_server.py` when auditing a non-default
environment.

## Export

Export a timestamped recovery bundle without restarting services:

```bash
python scripts/jiuwenmemory_server_audit.py export --output-dir /tmp/jiuwenmemory-server-poc-backups
```

The export writes:

- `memory_server.<timestamp>.py` - direct backup of the installed server file.
- `memory_server.<timestamp>.full-file.patch` - unified full-file patch that
  can recreate the audited file.
- `memory_server.<timestamp>.manifest.json` - source path, SHA-256, audit
  result, and artifact paths.

The script is stdlib-only and inspects the file on disk. It does not import the
JiuwenMemory server, start services, read secrets, or mutate server state.
