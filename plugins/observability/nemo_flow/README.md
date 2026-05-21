# NeMo Relay Observability

Optional Hermes observability plugin that maps Hermes observer hooks to
NeMo Relay scopes, LLM spans, tool spans, marks, ATOF, and ATIF.

Enable it with:

```bash
hermes plugins enable observability/nemo_flow
```

The plugin fails open when `nemo-relay` is not installed. Install and test it
against the renamed NeMo Relay 0.3 package line:

```bash
pip install "nemo-relay>=0.3"
```

Useful local export settings:

```bash
export HERMES_NEMO_FLOW_ATOF_ENABLED=1
export HERMES_NEMO_FLOW_ATOF_OUTPUT_DIRECTORY=.nemo-flow/atof
export HERMES_NEMO_FLOW_ATIF_ENABLED=1
export HERMES_NEMO_FLOW_ATIF_OUTPUT_DIRECTORY=.nemo-flow/atif
```

To initialize NeMo Relay from a component config instead, set:

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
- `HERMES_NEMO_FLOW_ADAPTIVE_ENABLED` (`1`, `true`, `yes`, or `on`)
- `HERMES_NEMO_FLOW_ADAPTIVE_MODE` (`observe` by default)

## Adaptive Execution PoC

By default, this plugin is passive: it observes Hermes hooks and emits
NeMo Relay lifecycle events without changing execution. When
`HERMES_NEMO_FLOW_ADAPTIVE_ENABLED=1`, the plugin also registers Hermes
execution middleware and routes tool/provider callbacks through NeMo Relay's
managed `tools.execute()` and `llm.execute()` helpers when those APIs are
available.

This enables NeMo Relay request intercepts and execution intercepts to run at the
Hermes tool and LLM boundaries while preserving the raw Hermes provider response
for the agent loop. Treat this as an opt-in integration boundary for validating
adaptive behavior before making NeMo Relay a default runtime backend.

## ATOF Mapping

The plugin keeps NeMo Relay's native event model:

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
