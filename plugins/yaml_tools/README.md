# yaml_tools — user-defined tools via YAML

Define your own agent tools by dropping a YAML file in `~/.hermes/tools/`. No
Python plugin required. Each file defines one tool that appears in the agent's
`custom` toolset alongside the built-ins.

This is the lightweight middle ground between **skills** (markdown instructions
that only *describe* how to use existing tools) and **plugins** (full Python
packages). A YAML tool is a real, callable, schema-typed tool.

## Example

```yaml
# ~/.hermes/tools/my_search.yaml
name: my_search
description: "Search my internal documentation"
command: 'curl -s "https://internal-docs/search?q=$HERMES_TOOL_ARG_QUERY"'
parameters:
  query:
    type: string
    description: "Search query"
    required: true
timeout: 60          # optional, seconds (default 60, capped at 600)
```

The agent can now call `my_search(query="onboarding")`.

## File format

One file per tool. Extension `.yaml` or `.yml`.

| Key | Required | Meaning |
| --- | --- | --- |
| `name` | yes | Tool name the model calls. Letters, digits, underscores; must start with a letter or underscore. |
| `description` | recommended | What the tool does (shown to the model). |
| `command` | yes | Shell command run by the configured terminal backend. Reference parameters as `HERMES_TOOL_ARG_*` variables (see below). |
| `parameters` | no | Mapping of parameter name → spec. |
| `timeout` | no | Max seconds the command may run (default `60`, capped at `600`). |

Each parameter spec accepts:

| Key | Meaning |
| --- | --- |
| `type` | `string` (default), `number`, `integer`, or `boolean`. |
| `description` | Shown to the model. |
| `required` | `true` to mark the parameter required. |
| `enum` | Optional list of allowed values. |

## How parameters reach the command

Each declared parameter is exported under the dedicated, upper-cased
`HERMES_TOOL_ARG_<NAME>` namespace. A parameter named `query` is therefore
available as `$HERMES_TOOL_ARG_QUERY`; booleans are rendered as `true` /
`false`. An omitted optional parameter is exported as an empty string.

```yaml
command: 'echo "greeting is $HERMES_TOOL_ARG_GREETING"'
```

The dedicated prefix prevents model arguments such as `path` from replacing
ambient variables such as `PATH`. Environment variables and API keys made
available by the selected terminal backend remain accessible under their
original names:

```yaml
command: 'curl -H "Authorization: Bearer $INTERNAL_API_KEY" \
  "https://example.test/search?q=$HERMES_TOOL_ARG_QUERY"'
```

## Security

Parameter values are shell-quoted into environment assignments inside an
isolated subshell. A value such as `$(touch /tmp/example)` or `"; echo bad"`
therefore remains data instead of becoming extra shell syntax. The complete
wrapper then runs through the normal `terminal` tool, including its configured
local/container/SSH backend, command approval checks, bounded output capture,
timeouts, and descendant-process cleanup.

The command template is trusted code. Quote parameter expansions
(`"$HERMES_TOOL_ARG_QUERY"`, not `$HERMES_TOOL_ARG_QUERY`) to avoid
word-splitting and globbing, and do not pass model parameters to `eval` or use
them deliberately as command text. Dangerous-looking parameter data may
conservatively trigger the normal approval prompt.

Model arguments are visible in the effective terminal command and should not
be used to transport secrets. Put API keys in the ambient environment instead.

## Behaviour notes

- Files are discovered at startup. Add/change a file, then restart Hermes.
- A malformed file is logged and skipped — it never breaks startup.
- Built-ins are never overridden by a YAML tool with the same name.
- The `custom` toolset is part of the default set; disable it like any other
  toolset if you don't want user tools loaded.
- `bash` must be available on `PATH`.
