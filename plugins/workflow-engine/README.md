# workflow-engine

DAG workflow engine plugin for Hermes Agent.

Lets you define, trigger, and monitor multi-step workflows directly from the
Switch UI `/workflows` dashboard tab. Workflows are authored as YAML files
and executed by the plugin's Python engine (Phase 2a+).

## Install

```bash
hermes plugins enable workflow-engine
hermes dashboard restart
```

The plugin mounts API routes under `/api/plugins/workflow-engine/` and adds a
**Workflows** sidebar entry in the Hermes dashboard.

## Status

| Phase | Status |
|-------|--------|
| 1 — Skeleton & contract | done |
| 2a — YAML parser + DB | pending |
| 2b — Run executor | pending |
| 3 — Full API wiring | pending |
