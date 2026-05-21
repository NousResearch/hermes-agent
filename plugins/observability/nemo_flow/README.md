# NeMo-Flow Observability

Optional Hermes observability plugin that maps Hermes middleware hooks to
NeMo-Flow scopes, LLM spans, tool spans, marks, ATOF, and ATIF.

Enable it with:

```bash
hermes plugins enable observability/nemo_flow
```

The plugin fails open when `nemo-flow` is not installed. It targets the public
Python API available in NeMo-Flow `0.2.0`.

Useful local export settings:

```bash
export HERMES_NEMO_FLOW_ATOF_ENABLED=1
export HERMES_NEMO_FLOW_ATOF_OUTPUT_DIRECTORY=.nemo-flow/atof
export HERMES_NEMO_FLOW_ATIF_ENABLED=1
export HERMES_NEMO_FLOW_ATIF_OUTPUT_DIRECTORY=.nemo-flow/atif
```

To initialize NeMo-Flow from a component config instead, set:

```bash
export HERMES_NEMO_FLOW_PLUGINS_TOML=.nemo-flow/plugins.toml
```

Optional overrides:

- `HERMES_NEMO_FLOW_PLUGINS_TOML`
- `HERMES_NEMO_FLOW_ATOF_FILENAME`
- `HERMES_NEMO_FLOW_ATOF_MODE` (`append` or `overwrite`)
- `HERMES_NEMO_FLOW_ATIF_FILENAME_TEMPLATE`
- `HERMES_NEMO_FLOW_ATIF_AGENT_NAME`
- `HERMES_NEMO_FLOW_ATIF_AGENT_VERSION`
- `HERMES_NEMO_FLOW_ATIF_MODEL_NAME`

## ATOF Mapping

The plugin keeps NeMo-Flow's native event model:

- Hermes sessions map to `agent` scopes.
- Hermes API request hooks map to `llm` scope start/end events.
- Hermes tool hooks map to `tool` scope start/end events.
- Turn, approval, subagent, and diagnostic fallback events map to `mark`
  events.

For subagent correlation, mark metadata includes parent and child session IDs,
subagent IDs, role/status fields when present, and derived
`parent_trajectory_id` / `child_trajectory_id` values. This keeps the ATOF
stream lossless for later ATIF conversion that can compact subagents into
separate trajectories.
