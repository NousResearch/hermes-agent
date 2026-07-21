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
command: 'curl -s "https://internal-docs/search?q=$QUERY"'
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
| `command` | yes | Shell command run with `bash -c`. Reference parameters as environment variables (see below). |
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

Each supplied parameter value is exported to the command's environment under
**both** its exact name and its upper-cased name, so a parameter `query` is
available as `$query` and `$QUERY`. Booleans are rendered as `true` / `false`.

```yaml
command: 'echo "greeting is $greeting"'   # $greeting or $GREETING both work
```

## Security

Parameter **values** supplied by the model are passed as environment variables,
never interpolated into the command string. bash does not re-parse the result
of a variable expansion for command substitution, so a value such as
`$(rm -rf ~)` or `"; rm -rf ~ ; "` is treated as a literal string and never
executed. The command **template** itself is authored by you (trusted); only
the argument values come from the model.

Quote your expansions (`"$QUERY"`, not `$QUERY`) to also avoid word-splitting
and globbing on values that contain spaces or shell glob characters.

## Behaviour notes

- Files are discovered at startup. Add/change a file, then restart Hermes.
- A malformed file is logged and skipped — it never breaks startup.
- A tool whose `name` collides with a built-in is skipped (built-ins are never
  overridden).
- The `custom` toolset is part of the default set; disable it like any other
  toolset if you don't want user tools loaded.
- `bash` must be available on `PATH`.
