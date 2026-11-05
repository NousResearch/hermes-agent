# CaMeL Guard

A CaMeL-style capability guard for Hermes Agent. After reading untrusted tool
output, Hermes must obtain authority from the current user's request before a
sensitive tool can run. The classifier sees the current request only. Retrieved
pages, file contents and prior assistant text cannot grant capabilities.

## Install and activate

Requires Hermes **0.22.7 or newer**. No additional Python dependencies.

```sh
# After catalog admission:
hermes plugins install camel-guard
hermes plugins enable camel-guard
```

Before catalog admission, install the reviewed plugin from the established
Hermes CaMeL repository at its full commit SHA with
`hermes plugins install "https://github.com/nativ3ai/hermes-agent-camel#plugins/camel-guard" --ref <SHA> --enable`.
The mode starts **off**, and traces start **off**. Add this to the active
profile's `config.yaml`:

```yaml
plugins:
  enabled: [camel-guard]
  entries:
    camel-guard:
      settings:
        mode: "monitor"   # "off", "monitor", "enforce"
        trace_enabled: false
```

Reload plugins or restart Hermes, then run `/camel status`. `/camel monitor`,
`/camel enforce` and `/camel off` save a mode for the next reload; they do not
change policy halfway through an active turn. Start in monitor, exercise your
usual workflows, then select enforce and reload. The host's configured model
must support structured responses for authorized utility to work. A classifier
failure blocks sensitive tools in enforce; monitor continues execution.

## What it protects

Five public hooks mediate the native direct, sequential and concurrent tool
paths. Hermes keeps its tool-result wrappers, approvals and prompt caching.
There are no patches to core methods, extra model tools or background processes.

- Commands, file changes, memory, skills, jobs, messaging, delegation, browser
  interactions, desktop projects and task mutation have separate capabilities.
- Unknown tools, including MCP tools, conservatively require `external_system`.
- Known read-only actions remain available. Their results still count as
  untrusted input for later effects.
- A classifier timeout, malformed response or provider failure yields a read-only
  plan. Denial takes precedence over allowance. Parallel decisions classify once.

Operators integrating custom tools can use exact, audited names:

```yaml
# Under plugins.entries.camel-guard.settings:
read_only_tools: [mcp_docs_read]
tool_capabilities:
  workspace_save: file_mutation
```

Wildcards and invalid capabilities are ignored. A read-only exception exempts
only the effect gate: its output still cannot grant authority. Do not mark an
effect-capable tool read-only.

## Data and network disclosure

The plugin calls **the model configured in Hermes**, through `ctx.llm`, only
when a sensitive decision follows untrusted content. That sends the current
user request to the configured provider and consumes model quota. The plugin
never reads `.env`, OAuth stores, browser profiles or another tool's login.
Hermes owns authentication and any normal token refresh. The runtime executes
no shell commands and makes no independent network requests. It reads its own
operator settings through Hermes.

Optional bounded traces use Hermes's profile-scoped plugin state. They contain
time, scope identifiers, tool names, capability and decision metadata; no user
text, tool arguments, tool results or classifier rationale. Off mode does not
create a trace. There is no telemetry, self-updater or credential storage.

## Boundaries

This is a **coarse capability guard**, not the full CaMeL interpreter and not a
sandbox. Authorization applies to an effect category, not an exact command,
file path, recipient or argument. It does not prevent answer contamination or
exfiltration through an already-authorized capability. It trusts the host to
provide the actual current user request separately from imported content.
Untrusted attachments concatenated into that trusted field are outside this
boundary. Monitor is observational. Host/plugin failures and hooks unavailable
on older Hermes versions remain host concerns; the compatibility floor is
intentional.

## Reproduce

The repository root contains `scripts/` and `tests/`. Generated results are
ignored and are published as CI artifacts, never as source evidence.

```sh
python scripts/check_camel_guard.py --hermes-root /path/to/hermes-agent
python scripts/camel_guard_offline_benchmark.py --hermes-root /path/to/hermes-agent
# Explicit opt-in to live calls using Hermes-managed authentication:
python scripts/camel_guard_live_benchmark.py --hermes-root /path/to/hermes-agent \
  --live --provider openai-codex --model YOUR_MODEL --phase all \
  --auth-home /path/to/your/hermes-profile
```

Offline controls use a deterministic classifier fixture and real scratch-file
dispatch. Live controls report model refusals separately from guard blocks and
include authorized-utility controls. See the repository's `docs/BENCHMARKS.md`
and `docs/INFORMATION_FLOW.md` for methodology and explicit non-claims.

MIT. Original implementation and review history:
[NousResearch/hermes-agent#3987](https://github.com/NousResearch/hermes-agent/pull/3987).
