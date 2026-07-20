# NeMo Relay Observability

Optional Hermes observability plugin that maps Hermes observer hooks to
NeMo Relay scopes, LLM spans, tool spans, marks, ATOF, and ATIF.

NeMo Relay is NVIDIA's runtime layer for agent execution boundaries. It does
not replace Hermes Agent's planner, tools, memory, model provider routing, or
CLI UX. Instead, this plugin lets Hermes emit NeMo Relay lifecycle events for
the work Hermes already owns: sessions, turns, provider/API calls, tool calls,
approval prompts, and delegated subagents.

With this plugin enabled, Hermes Agent can:

- Preserve Hermes execution as NeMo Relay scopes, LLM spans, tool spans, and
  mark events.
- Export raw lifecycle events as Agent Trajectory Observability Format (ATOF)
  JSONL for debugging and offline inspection.
- Export Agent Trajectory Interchange Format (ATIF) trajectories for replay,
  evaluation, and harness analysis workflows.
- Correlate parent sessions, delegated subagents, tool calls, and provider
  calls through shared session, turn, and trajectory metadata.

See the NeMo Relay overview for the broader runtime model:
https://docs.nvidia.com/nemo/relay/about-nemo-relay/overview

ATOF is NVIDIA's canonical JSONL event stream representation for NeMo Relay
lifecycle events. The format is documented in the NeMo Agent Toolkit:
https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/develop/packages/nvidia_nat_atif/atof-event-format.md

ATIF is the trajectory representation produced from those events. NVIDIA and
Harbor upstreamed ATIF v1.7 support for complex harness workflows, including
subagent trajectory embedding, trajectory IDs, multi-LLM-call step metadata, and
deterministic no-LLM orchestration steps:
https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md

## Enablement

Enable the plugin before setting export options:

```bash
hermes plugins enable observability/nemo_relay
```

The `HERMES_NEMO_RELAY_*` environment variables below only configure an
already-enabled plugin. They do not enable plugin discovery by themselves.

For isolated test homes, enable the plugin in the same `HERMES_HOME` that the
agent run will use:

```bash
env HERMES_HOME=/tmp/hermes-nemo-relay-test \
  hermes plugins enable observability/nemo_relay
```

Runs started with `--ignore_user_config` skip the enabled-plugin state from
`HERMES_HOME`, so local E2E tests should omit that flag unless the test harness
loads `observability/nemo_relay` explicitly another way.

`HERMES_HOME` is the Hermes profile/config home used by both
`hermes plugins enable ...` and the later `hermes chat ...` run. If unset,
Hermes uses the user's default home, usually `~/.hermes`. For isolated smoke
tests, choose any writable temporary directory and use the same value for every
command in that test:

```bash
export HERMES_HOME=/tmp/hermes-nemo-relay-test
hermes plugins enable observability/nemo_relay
hermes chat --query 'Reply exactly ok' --provider custom --model qwen3.6:35b
```

For source checkouts, make sure the `hermes` command you run is built from the
checkout that contains this plugin. A globally installed older CLI will not see
new bundled plugins from your working tree.

```bash
uv sync --extra nemo-relay
uv run hermes plugins enable observability/nemo_relay
uv run hermes chat --query 'Reply exactly ok' --provider custom --model qwen3.6:35b
```

To ship the updated CLI into another environment, build and install a fresh
wheel from this checkout, then install the official NeMo Relay runtime extra:

```bash
uv build --wheel
python -m pip install --force-reinstall dist/hermes_agent-*.whl
python -m pip install "nemo-relay>=0.5,<1.0"
hermes plugins enable observability/nemo_relay
```

The plugin fails open when `nemo-relay` is not installed. Install a supported
NeMo Relay 0.x distribution beginning with 0.5:

```bash
pip install "nemo-relay>=0.5,<1.0"
```

## Export Configuration

The plugin can configure exporters directly from `HERMES_NEMO_RELAY_*`
environment variables, or delegate exporter setup to a NeMo Relay
`plugins.toml` component config.

Use environment variables for local smoke tests, CI jobs, and one-off CLI
runs. Use `plugins.toml` when you want one NeMo Relay configuration document to
own observability components such as ATOF, ATIF, OpenTelemetry, and
OpenInference.

### Environment Variables

Useful local export settings after the plugin is enabled:

```bash
export HERMES_NEMO_RELAY_ATOF_ENABLED=1
export HERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY=.nemo-relay/atof
export HERMES_NEMO_RELAY_ATIF_ENABLED=1
export HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY=.nemo-relay/atif
```

Optional overrides:

- `HERMES_NEMO_RELAY_ATOF_FILENAME`
- `HERMES_NEMO_RELAY_ATOF_MODE` (`append` or `overwrite`)
- `HERMES_NEMO_RELAY_ATIF_FILENAME_TEMPLATE`
- `HERMES_NEMO_RELAY_ATIF_AGENT_NAME`
- `HERMES_NEMO_RELAY_ATIF_AGENT_VERSION`
- `HERMES_NEMO_RELAY_ATIF_MODEL_NAME`
- `HERMES_NEMO_RELAY_ATIF_SUBAGENT_EXPORT_MODE` (`embedded` by default; set `all` to also write standalone child files)

### NeMo Relay Component Config

To initialize NeMo Relay from a component config, create a `plugins.toml` file
and point Hermes at it:

```bash
export HERMES_NEMO_RELAY_PLUGINS_TOML=.nemo-relay/plugins.toml
```

Minimal ATOF and ATIF config:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true
output_directory = ".nemo-relay/atof"
filename = "events.jsonl"
mode = "overwrite"

[components.config.atif]
enabled = true
output_directory = ".nemo-relay/atif"
filename_template = "trajectory-{session_id}.json"
agent_name = "Hermes Agent"
agent_version = "local"
```

When `HERMES_NEMO_RELAY_PLUGINS_TOML` is set and initializes successfully, NeMo
Relay owns exporter lifecycle through that config. The direct
`HERMES_NEMO_RELAY_ATOF_*` fallback setup is skipped. If the same
`plugins.toml` observability config enables `atif`, the direct
`HERMES_NEMO_RELAY_ATIF_*` fallback setup is also skipped so Hermes does not
double-export trajectories on teardown. If `plugins.toml` initialization fails,
Hermes keeps the direct env-var fallbacks active for that run.

To enable NeMo Relay managed execution intercepts for provider and tool calls,
include an adaptive component in the same `plugins.toml`:

```toml
[[components]]
kind = "adaptive"
enabled = true

[components.config.tool_parallelism]
mode = "observe_only"
```

When the adaptive component is enabled and the installed NeMo Relay runtime
exposes `llm.execute(...)` / `tools.execute(...)`, Hermes routes LLM and tool
execution through those middleware boundaries. The observer hooks still emit
session, turn, approval, and subagent marks; the plugin skips its manual
`llm.call` and `tools.call` spans for executions that are already managed by
NeMo Relay. `tool_parallelism.mode = "observe_only"` keeps tool scheduling
observational while still wrapping the real execution boundary.

### Dynamic Plugins

Hermes feature-detects the dynamic-plugin activation API available in NeMo Relay
0.6 and later. Configure native or worker plugins with Hermes-owned
`[[dynamic_plugins]]` entries that match the Python binding's activation-spec
fields:

```toml
[[dynamic_plugins]]
plugin_id = "example-plugin"
kind = "rust_dynamic"
manifest_ref = "./example-plugin/relay-plugin.toml"

[dynamic_plugins.config]
mode = "enabled"
```

For a worker plugin, also provide the lifecycle-managed `environment_ref`:

```toml
[[dynamic_plugins]]
plugin_id = "example-worker"
kind = "worker"
manifest_ref = "./example-worker/relay-plugin.toml"
environment_ref = "/absolute/path/from-nemo-relay-plugins-inspect"

[dynamic_plugins.config]
mode = "enabled"
```

Provision the worker first with `nemo-relay plugins add`, then copy
`data.source.environment_ref` from the JSON output of
`nemo-relay plugins inspect <plugin-id> --json`. Relay rejects arbitrary Python
environments at activation time.

Relative `manifest_ref` and `environment_ref` values resolve relative to the
physical `plugins.toml` file.

Relay's canonical gateway `[[plugins.dynamic]]` records are not interchangeable
with this Hermes-owned section. The gateway combines those records with
separate lifecycle state for enablement, trust policy, and worker environments;
the Python binding does not yet expose that resolver. Hermes rejects
`[[plugins.dynamic]]` with an actionable diagnostic instead of silently
ignoring it or bypassing lifecycle policy. Use `[[dynamic_plugins]]` until Relay
exposes shared file-and-lifecycle resolution to embedding hosts.

Hermes activates these plugins before registering its managed LLM and tool
execution middleware and retains the activation for the runtime lifetime.
During shutdown it closes session exporters, flushes Relay subscribers, and
then closes the activation so callbacks are removed before plugin code is
unloaded.

NeMo Relay 0.5 does not expose dynamic activation through its Python binding.
When dynamic plugin configuration is present with a binding that lacks the
activation API, Hermes logs an actionable warning and continues with the
ordinary static component configuration, so ATOF and ATIF observability remain
available. No dynamic plugin is loaded in that degraded mode.

For the full generic Hermes middleware contract, see
[`docs/middleware/README.md`](../../../docs/middleware/README.md).

## Canonical Local Examples

The observe-only examples in this section use a supported NeMo Relay 0.x
distribution beginning with 0.5 and a local Ollama model served through the
OpenAI-compatible API.

```bash
pip install "nemo-relay>=0.5,<1.0"

export HERMES_HOME=/tmp/hermes-nemo-relay-docs/hermes-home
mkdir -p "$HERMES_HOME"

cat > "$HERMES_HOME/config.yaml" <<'YAML'
model:
  provider: custom
  default: qwen3.6:35b
  base_url: http://127.0.0.1:11434/v1
  api_key: ollama
plugins:
  enabled:
    - observability/nemo_relay
delegation:
  max_spawn_depth: 2
  max_concurrent_children: 2
  child_timeout_seconds: 180
  model: qwen3.6:35b
  provider: custom
  base_url: http://127.0.0.1:11434/v1
  api_key: ollama
YAML
```

### Delegated Subagent Tool Call

This run starts a parent Hermes session, delegates to a child subagent, has the
child call `terminal`, and writes both ATOF and ATIF.

```bash
export HERMES_NEMO_RELAY_ATOF_ENABLED=1
export HERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY=/tmp/hermes-nemo-relay-docs/subagent/atof
export HERMES_NEMO_RELAY_ATOF_FILENAME=nested-subagent-atof.jsonl
export HERMES_NEMO_RELAY_ATOF_MODE=overwrite
export HERMES_NEMO_RELAY_ATIF_ENABLED=1
export HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY=/tmp/hermes-nemo-relay-docs/subagent/atif
export HERMES_NEMO_RELAY_ATIF_FILENAME_TEMPLATE='nested-subagent-atif-{session_id}.json'
export HERMES_NEMO_RELAY_ATIF_AGENT_NAME='Hermes Agent E2E'
export HERMES_NEMO_RELAY_ATIF_AGENT_VERSION=docs-example
export HERMES_NEMO_RELAY_ATIF_SUBAGENT_EXPORT_MODE=all

hermes chat \
  --query 'Use delegate_task exactly once. Ask the child subagent to use the terminal tool exactly once to run printf docs_nested_leaf_function. After the child returns, reply with exactly: parent received nested subagent result.' \
  --provider custom \
  --model qwen3.6:35b \
  --toolsets delegation,terminal \
  --max-turns 10 \
  --quiet \
  --accept-hooks
```

CLI output:

```text
session_id: docs-parent-session
parent received nested subagent result.
```

Sanitized ATOF excerpt:

```jsonl
{"kind":"scope","category":"tool","name":"delegate_task","scope_category":"start","metadata":{"session_id":"docs-parent-session","tool_call_id":"call_delegate"},"data":{"goal":"Run the command `printf docs_nested_leaf_function` using the terminal tool.","toolsets":["terminal"]}}
{"kind":"mark","name":"hermes.subagent.start","metadata":{"parent_session_id":"docs-parent-session","session_id":"docs-child-session","subagent_id":"sa-0-docs","child_role":"leaf"}}
{"kind":"scope","category":"tool","name":"terminal","scope_category":"end","metadata":{"session_id":"docs-child-session","tool_call_id":"call_terminal","status":"ok"},"data":"{\"output\":\"docs_nested_leaf_function\",\"exit_code\":0,\"error\":null}"}
{"kind":"scope","category":"tool","name":"delegate_task","scope_category":"end","metadata":{"session_id":"docs-parent-session","tool_call_id":"call_delegate","status":"ok"}}
```

Sanitized ATIF excerpt:

```json
{
  "schema_version": "ATIF-v1.7",
  "session_id": "docs-parent-session",
  "agent": {"name": "Hermes Agent E2E", "version": "docs-example", "model_name": "qwen3.6:35b"},
  "steps": [
    {
      "source": "agent",
      "tool_calls": [{"function_name": "delegate_task"}],
      "observation": {
        "results": [
          {
            "subagent_trajectory_ref": [{"session_id": "docs-child-session"}],
            "content": "{\"results\":[{\"status\":\"completed\",\"tool_trace\":[{\"tool\":\"terminal\",\"status\":\"ok\"}]}]}"
          }
        ]
      }
    },
    {"source": "agent", "message": "parent received nested subagent result."}
  ],
  "subagent_trajectories": [
    {
      "session_id": "docs-child-session",
      "steps": [
        {
          "source": "agent",
          "tool_calls": [{"function_name": "terminal", "arguments": {"command": "printf docs_nested_leaf_function"}}],
          "observation": {"results": [{"content": "{\"output\":\"docs_nested_leaf_function\",\"exit_code\":0,\"error\":null}"}]}
        }
      ]
    }
  ]
}
```

### Parallel Tool Calls

This run asks the model to emit two `read_file` tool calls in the same assistant
message. Hermes dispatches the read-only tools as one batch, and NeMo Relay
records both tool invocations.

```bash
mkdir -p /tmp/hermes-nemo-relay-docs/workdir
printf 'docs_parallel_alpha_function\n' > /tmp/hermes-nemo-relay-docs/workdir/alpha.txt
printf 'docs_parallel_beta_function\n' > /tmp/hermes-nemo-relay-docs/workdir/beta.txt
cd /tmp/hermes-nemo-relay-docs/workdir

export HERMES_NEMO_RELAY_ATOF_ENABLED=1
export HERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY=/tmp/hermes-nemo-relay-docs/parallel/atof
export HERMES_NEMO_RELAY_ATOF_FILENAME=parallel-tools-atof.jsonl
export HERMES_NEMO_RELAY_ATOF_MODE=overwrite
export HERMES_NEMO_RELAY_ATIF_ENABLED=1
export HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY=/tmp/hermes-nemo-relay-docs/parallel/atif
export HERMES_NEMO_RELAY_ATIF_FILENAME_TEMPLATE='parallel-tools-atif-{session_id}.json'
export HERMES_NEMO_RELAY_ATIF_AGENT_NAME='Hermes Agent E2E'
export HERMES_NEMO_RELAY_ATIF_AGENT_VERSION=docs-example

hermes chat \
  --query 'Use exactly two read_file tool calls in the same assistant message. Read alpha.txt and beta.txt. Do not call terminal. After both tool results are available, reply with exactly: parallel tools complete.' \
  --provider custom \
  --model qwen3.6:35b \
  --toolsets file \
  --max-turns 8 \
  --quiet \
  --accept-hooks
```

CLI output:

```text
session_id: docs-parallel-session
parallel tools complete.
```

Sanitized ATOF excerpt:

```jsonl
{"kind":"scope","category":"llm","name":"custom","scope_category":"end","data":{"assistant_message":{"tool_calls":[{"id":"call_alpha","name":"read_file","arguments":"{\"path\":\"alpha.txt\"}"},{"id":"call_beta","name":"read_file","arguments":"{\"path\":\"beta.txt\"}"}]},"finish_reason":"tool_calls"}}
{"kind":"scope","category":"tool","name":"read_file","scope_category":"start","timestamp":"2026-05-31T00:15:08.956732+00:00","metadata":{"session_id":"docs-parallel-session","tool_call_id":"call_alpha"},"data":{"path":"alpha.txt"}}
{"kind":"scope","category":"tool","name":"read_file","scope_category":"start","timestamp":"2026-05-31T00:15:08.956804+00:00","metadata":{"session_id":"docs-parallel-session","tool_call_id":"call_beta"},"data":{"path":"beta.txt"}}
{"kind":"scope","category":"tool","name":"read_file","scope_category":"end","metadata":{"session_id":"docs-parallel-session","tool_call_id":"call_beta","status":"ok"},"data":"{\"content\":\"     1|docs_parallel_beta_function\\n\"}"}
{"kind":"scope","category":"tool","name":"read_file","scope_category":"end","metadata":{"session_id":"docs-parallel-session","tool_call_id":"call_alpha","status":"ok"},"data":"{\"content\":\"     1|docs_parallel_alpha_function\\n\"}"}
```

Sanitized ATIF excerpt:

```json
{
  "schema_version": "ATIF-v1.7",
  "session_id": "docs-parallel-session",
  "agent": {"name": "Hermes Agent E2E", "version": "docs-example", "model_name": "qwen3.6:35b"},
  "steps": [
    {
      "source": "agent",
      "tool_calls": [
        {"tool_call_id": "call_alpha", "function_name": "read_file", "arguments": {"path": "alpha.txt"}},
        {"tool_call_id": "call_beta", "function_name": "read_file", "arguments": {"path": "beta.txt"}}
      ],
      "observation": {
        "results": [
          {"source_call_id": "call_beta", "content": "{\"content\":\"     1|docs_parallel_beta_function\\n\"}"},
          {"source_call_id": "call_alpha", "content": "{\"content\":\"     1|docs_parallel_alpha_function\\n\"}"}
        ]
      }
    },
    {"source": "agent", "message": "parallel tools complete."}
  ]
}
```

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

## Adaptive Middleware Example

The `observability/nemo_relay` plugin uses Hermes execution middleware to hand
LLM and tool calls to NeMo Relay managed execution when an adaptive component is
enabled.

Minimal `plugins.toml`:

```toml
version = 1

[[components]]
kind = "adaptive"
enabled = true

[components.config.tool_parallelism]
mode = "observe_only"
```

Enable it for Hermes:

```bash
export HERMES_NEMO_RELAY_PLUGINS_TOML=/tmp/hermes-middleware-test/plugins.toml
```

When the adaptive component is enabled and the installed NeMo Relay runtime
exposes `llm.execute(...)` and `tools.execute(...)`, Hermes routes execution
through these boundaries:

```text
Hermes provider call
  -> llm_execution middleware
    -> nemo_relay.llm.execute(...)
      -> Hermes provider adapter next_call(...)

Hermes tool call
  -> tool_execution middleware
    -> nemo_relay.tools.execute(...)
      -> Hermes tool dispatcher next_call(...)
```

The plugin still emits observer marks for sessions, turns, approvals, and
subagents. When adaptive managed execution is active, it skips manual
`llm.call` and `tools.call` observer spans to avoid duplicate LLM/tool events
for the same execution.

### Local Adaptive E2E

This example enables both NeMo Relay observability export and adaptive execution
middleware for a local Hermes run. This path requires a NeMo Relay runtime that
supports `[components.config.tool_parallelism]`, as provided by the supported
0.x release range beginning with 0.5.

```bash
export HERMES_HOME=/tmp/hermes-middleware-test/hermes-home
mkdir -p "$HERMES_HOME" /tmp/hermes-middleware-test/nemo-relay

cat > "$HERMES_HOME/config.yaml" <<'YAML'
model:
  provider: custom
  default: qwen3.6:35b
  base_url: http://127.0.0.1:11434/v1
  api_key: ollama
plugins:
  enabled:
    - observability/nemo_relay
YAML

cat > /tmp/hermes-middleware-test/nemo-relay/plugins.toml <<'TOML'
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 1

[components.config.atof]
enabled = true
output_directory = "/tmp/hermes-middleware-test/atof"
filename = "middleware-events.jsonl"
mode = "overwrite"

[components.config.atif]
enabled = true
output_directory = "/tmp/hermes-middleware-test/atif"
filename_template = "middleware-trajectory-{session_id}.json"
agent_name = "Hermes Middleware E2E"
agent_version = "local"

[[components]]
kind = "adaptive"
enabled = true

[components.config.tool_parallelism]
mode = "observe_only"
TOML

export HERMES_NEMO_RELAY_PLUGINS_TOML=/tmp/hermes-middleware-test/nemo-relay/plugins.toml

hermes chat \
  --query 'Use the terminal tool exactly once to run printf middleware_execution_ok. Then reply with exactly the command output.' \
  --provider custom \
  --model qwen3.6:35b \
  --toolsets terminal \
  --max-turns 4 \
  --quiet \
  --accept-hooks
```

Expected CLI output:

```text
session_id: middleware-demo-session
middleware_execution_ok
```

Expected ATOF shape:

```jsonl
{"kind":"scope","category":"llm","name":"custom","scope_category":"start","metadata":{"session_id":"middleware-demo-session"},"data":{"mode":"observe_only"}}
{"kind":"scope","category":"tool","name":"terminal","scope_category":"start","metadata":{"session_id":"middleware-demo-session","tool_call_id":"call_terminal"},"data":{"mode":"observe_only"}}
{"kind":"scope","category":"tool","name":"terminal","scope_category":"end","metadata":{"session_id":"middleware-demo-session","tool_call_id":"call_terminal","status":"ok"},"data":"{\"output\":\"middleware_execution_ok\",\"exit_code\":0,\"error\":null}"}
```

Expected ATIF shape:

```json
{
  "schema_version": "ATIF-v1.7",
  "session_id": "middleware-demo-session",
  "agent": {
    "name": "Hermes Middleware E2E",
    "version": "local",
    "model_name": "qwen3.6:35b"
  },
  "steps": [
    {
      "source": "agent",
      "tool_calls": [
        {
          "function_name": "terminal",
          "arguments": {"command": "printf middleware_execution_ok"}
        }
      ],
      "observation": {
        "results": [
          {
            "source_call_id": "call_terminal",
            "content": "{\"output\":\"middleware_execution_ok\",\"exit_code\":0,\"error\":null}"
          }
        ]
      }
    },
    {
      "source": "agent",
      "message": "middleware_execution_ok"
    }
  ]
}
```
