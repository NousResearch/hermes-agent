# ACP No-Tool / No-MCP Evidence Mode

Status: design proposal (not yet implemented)
Author: drafted for AI Governance evidence planning

## Why this exists

AI Governance wants to run one future Hermes model/provider evidence attempt
without treating a normal Hermes session as governed runtime proof.

The evidence run needs a stronger boundary than prompt wording. A prompt such
as "do not use tools" is model guidance, not an execution constraint. If the
runtime still exposes tool schemas, MCP servers, edit approvals, terminal
tools, browser tools, memory tools, or plugin tools, the run cannot be used as
no-tool governance evidence.

This design defines the smallest Hermes-side shape that could support a future
real provider evidence run:

- no tools exposed to the model;
- no MCP discovery or MCP server toolsets added;
- final response capture still available;
- no provider run authorized by this proposal;
- no change to normal ACP or Hermes user experience.

## Current blocker

`hermes-acp` is a useful final-response capture candidate, but it is not
currently a reviewed no-tool/no-MCP execution path.

The current ACP startup and session path has these blocker properties:

- `acp_adapter/entry.py` loads environment/config surfaces before starting the
  server.
- `acp_adapter/entry.py` performs MCP tool discovery during ACP startup.
- `acp_adapter/session.py` constructs ACP agents with
  `enabled_toolsets=["hermes-acp"]`.
- `acp_adapter/session.py` also adds enabled MCP servers from config as
  toolsets.
- `hermes tools disable ...` is config-mutating and is not a no-write preflight
  mechanism.
- The shared platform registry does not currently expose an `acp` platform that
  can be targeted by the existing tools configuration UX.

Therefore a future governance evidence run is blocked until Hermes exposes a
reviewed no-tool/no-MCP execution constraint.

## Design goals

1. Provide a real runtime constraint, not prompt-only policy.
2. Preserve ACP final-response capture so the evidence runner can review the
   model's final text.
3. Avoid normal UX regression for ACP users.
4. Avoid prompt-cache churn during ordinary conversations.
5. Avoid adding a new core model tool.
6. Avoid writing or mutating user config as part of the evidence preflight.
7. Make the active tool surface inspectable in the evidence packet.

## Non-goals

- Do not run a provider or model.
- Do not read or store credential values.
- Do not add a new provider integration.
- Do not change normal ACP startup behavior by default.
- Do not mutate `config.yaml`, `.env`, profile files, or MCP config for the
  evidence mode.
- Do not add a new core model tool.
- Do not implement a governance hook, CI gate, or enforcement layer.
- Do not claim that no-tool evidence proves truth, reliability, safety, or
  non-bypassable governance.

## Proposed shape

Add an ACP evidence-mode path that can construct a session with:

```text
enabled_toolsets=[]
disabled_toolsets=None
mcp_server_names=[]
discover_mcp_tools_on_startup=false
```

The mode should be selected by an explicit ACP startup/session option, not by a
prompt, not by editing global config, and not by relying on an environment
variable as user-facing configuration.

Possible implementation surfaces, in preference order:

1. An ACP-specific command-line flag, for example `hermes-acp --evidence-no-tools`.
2. An ACP protocol/session initialization option if the client can pass one.
3. A local wrapper used only by the evidence harness that calls ACP session
   construction with the no-tool parameters.

The design should prefer a narrow ACP-specific option over widening global
Hermes tool configuration.

## Required behavior

When evidence mode is enabled:

1. ACP server startup must not discover MCP tools.
2. ACP session construction must not add `hermes-acp`.
3. ACP session construction must not add config-defined MCP server toolsets.
4. The model API request must receive no tool schemas.
5. Tool dispatch callbacks should remain inert or unreachable for the run.
6. Final response capture must remain available.
7. The evidence packet must record the effective tool surface as empty.

If any of these cannot be met, the evidence mode must fail closed before a
provider call.

## Prompt-cache boundary

This mode is intended for a dedicated evidence session, not for changing an
existing conversation mid-stream.

It must not mutate toolsets in a live conversation. The no-tool state must be
fixed at session creation time. This keeps the normal prompt-cache invariant:
toolsets are part of the cached prompt surface and must not change
mid-conversation.

## Config boundary

Behavioral settings should not be introduced as new user-facing `HERMES_*`
environment variables. If this becomes a persistent user-facing feature, it
should be represented in `config.yaml` or an ACP session option.

For the governance evidence run, prefer an explicit per-run option over writing
to `config.yaml`. The evidence run should not require changing the user's
normal ACP tools setup.

## Evidence packet fields

A future evidence packet should include:

```yaml
entrypoint: hermes-acp
evidence_mode: no_tools_no_mcp
effective_enabled_toolsets: []
effective_mcp_servers: []
mcp_discovery_on_startup: false
tool_schema_count: 0
final_response_capture: present
provider_run_authorized: false
not_claimed:
  - provider_safety
  - model_reliability
  - semantic_truth
  - non_bypassable_governance
  - general_acp_behavior_changed
```

`provider_run_authorized` remains false until a separate operator instruction
authorizes credential use and model execution.

## Test expectations for a future implementation

The implementation should include focused tests that prove:

1. Default ACP behavior still enables its normal ACP tool surface.
2. Evidence mode constructs the agent with an empty enabled toolset list.
3. Evidence mode does not add config-defined MCP servers.
4. Evidence mode does not call MCP discovery during startup.
5. Evidence mode leaves final-response capture intact.
6. Evidence mode fails closed if a tool schema would be sent to the model.

These tests should use a temp `HERMES_HOME` and the existing Hermes test
wrapper conventions. They should not touch the user's real `~/.hermes`.

## Review boundary

This proposal only defines the design target. It does not make real provider
evidence runnable.

Before any provider run, the implementation must be reviewed separately and a
preflight packet must show that the effective tool surface is empty for the
selected entrypoint, provider, model, prompt, and artifact paths.
