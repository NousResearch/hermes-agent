# NeMo Relay Observability

Optional Hermes observability plugin that maps Hermes observer hooks to
NeMo Relay scopes, LLM spans, tool spans, marks, ATOF, and ATIF.

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

The plugin fails open when `nemo-relay` is not installed. Install and test it against the official NeMo Relay 0.3 PyPI distribution:

```bash
pip install "nemo-relay==0.3"
```

## Export Settings

Useful local export settings after the plugin is enabled:

```bash
export HERMES_NEMO_RELAY_ATOF_ENABLED=1
export HERMES_NEMO_RELAY_ATOF_OUTPUT_DIRECTORY=.nemo-relay/atof
export HERMES_NEMO_RELAY_ATIF_ENABLED=1
export HERMES_NEMO_RELAY_ATIF_OUTPUT_DIRECTORY=.nemo-relay/atif
```

To initialize NeMo Relay from a component config instead, set:

```bash
export HERMES_NEMO_RELAY_PLUGINS_TOML=.nemo-relay/plugins.toml
```

Optional overrides:

- `HERMES_NEMO_RELAY_PLUGINS_TOML`
- `HERMES_NEMO_RELAY_ATOF_FILENAME`
- `HERMES_NEMO_RELAY_ATOF_MODE` (`append` or `overwrite`)
- `HERMES_NEMO_RELAY_ATIF_FILENAME_TEMPLATE`
- `HERMES_NEMO_RELAY_ATIF_AGENT_NAME`
- `HERMES_NEMO_RELAY_ATIF_AGENT_VERSION`
- `HERMES_NEMO_RELAY_ATIF_MODEL_NAME`
- `HERMES_NEMO_RELAY_ATIF_SUBAGENT_EXPORT_MODE` (`embedded` by default; set `all` to also write standalone child files)

## Canonical Local Examples

The examples below use the official `nemo-relay==0.3` distribution and a local
Ollama model served through the OpenAI-compatible API.

```bash
pip install "nemo-relay==0.3"

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
