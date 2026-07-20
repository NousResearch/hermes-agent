# AI Company Router

A foundational state router for a distributed "AI Company" platform.

- **Control Panel**: Obsidian vault (path configurable via `OBSIDIAN_VAULT_PATH`)
- **Coordinator**: `node4_coordinator` reads `input.md` and dispatches tasks
- **Cross-platform execution**: `mac_hermes` and `windows_hermes` run in parallel
- **Mock infrastructure**: local JSON DB, print-based remote RPC
- **SkillOpt loop**: failure logs are written to `failure_log.md` and the skill doc is patched

Run
---

```bash
cd "$HERMES_HOME/.."
PYTHONPATH="$HERMES_HOME/.." python3 ai_company_router/router.py
```

Files produced
--------------

| File | Purpose |
|------|---------|
| `obsidian_vault/input.md` | Task source |
| `obsidian_vault/output/dashboard.md` | Live status output |
| `obsidian_vault/output/mock_db.json` | Mock database |
| `obsidian_vault/output/failure_log.md` | Structured error history |
| `obsidian_vault/skills/distributed_router.md` | Self-evolution skill patches |
