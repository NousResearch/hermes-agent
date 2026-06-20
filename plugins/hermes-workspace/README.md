# Hermes Workspace Dashboard Plugin

Hermes Workspace adds a local orchestration cockpit to the dashboard:

- profile and agent roster visibility
- dashboard and runtime plugin inventory
- memory and context provider status
- Kanban board counters and recent tasks
- quick Kanban task creation
- multi-agent blueprint launchers
- debug shortcuts for dashboard, gateway, logs, sessions, and Workspace V2
- prompt shortcuts injected into the embedded chat tab

## Local Roster Configuration

The plugin avoids shipping personal agent lists in the repository. To customize
the cockpit locally, create `~/.hermes/workspace_agents.json`:

```json
{
  "catalog_label": "Personal Agents",
  "active_profiles": ["pilot", "ops", "docs"],
  "catalog_agents": ["orchestrator", "research", "self_review"]
}
```

If the file is absent, the plugin derives installed profiles from Hermes and
uses a generic built-in profile hint list.
