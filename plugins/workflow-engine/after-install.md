# After Install — workflow-engine

1. Restart the Hermes dashboard: `hermes dashboard restart`
2. Open the dashboard — a **Workflows** entry should appear in the sidebar.
3. Verify the health endpoint:
   ```bash
   curl http://127.0.0.1:8642/api/plugins/workflow-engine/health
   # → {"ok":true,"version":"0.1.0"}
   ```
4. Place workflow YAML files in `~/.hermes/workflows/` (created automatically
   on first run in Phase 2a).

No environment variables are required for Phase 1.
