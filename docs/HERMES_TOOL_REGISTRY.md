# Hermes Tool Registry Plan

Date: 2026-05-20

## Current Tool Architecture

Hermes has a mature but distributed tool system.

Primary sources:

- `tools/registry.py`: canonical schema, handler, toolset, and availability
  registry.
- `toolsets.py`: built-in and composed toolsets.
- `model_tools.py`: tool schema resolution and dispatch.
- `tools/*.py`: built-in tool implementations.
- `hermes_cli/plugins.py`: plugin tools and hooks.
- `plugins/*/plugin.yaml`: plugin manifests.
- `~/.hermes/config.yaml`: enabled toolsets, MCP servers, quick commands,
  provider/tool settings.
- `cron/` and `~/.hermes/cron/jobs.json`: scheduled jobs.
- `Operator/scripts`: command-center, health guardian, toolbelt, gateway
  wrapper.
- launchd plists: gateway and health guardian services.

## Existing Inventory Summary

Observed enabled/core capabilities:

- Web search and extraction.
- Browser automation, with runtime dependency caveats.
- Terminal/process tools.
- File read/write/patch/search.
- Code execution.
- Vision/video analysis.
- Image generation.
- Text-to-speech.
- Skills management.
- Todo/planning.
- Built-in memory and external memory provider tools.
- Session search.
- Clarify.
- Delegation.
- Cron jobs.
- Cross-platform messaging.
- Kanban worker coordination.
- MCP `playwright`.

Observed disabled/gated lanes:

- Video generation.
- X search.
- Home Assistant.
- Spotify.
- Yuanbao.
- Computer use.
- Discord and other credential-gated platforms.
- Browser/CDP variant depending on binary/path state.

## Registry Gap

No single local artifact currently answers:

- What tools exist?
- What layer owns them?
- Are they enabled?
- Are requirements present?
- Are credentials present, without revealing values?
- What cost/risk class applies?
- What approval policy applies?
- What health probe proves readiness?
- What safe next action should an operator take?

The control plane should generate this inventory read-only before it enforces
anything.

## Proposed Registry Schema

```yaml
schema_version: 1
generated_at: ISO-8601 timestamp
owner: hermes-control-plane
id: toolset.browser
layer: toolset | tool | plugin | mcp | quick_command | cron | launchd | operator_script | container_backend
name: Browser Automation
status: healthy | enabled | disabled | gated | missing | degraded | unknown
observed_state:
  source: live_config | repo_static | runtime_probe | operator_script
  confidence: high | medium | low
policy_overlay:
  managed_by: config | plugin | operator | future-control-plane
  enforcement_mode: observe | warn | gate | deny
enabled_for:
  - cli
  - telegram
  - api_server
entrypoint: tools/browser_tool.py
owner_file: tools/browser_tool.py
tools:
  - browser_navigate
  - browser_snapshot
requires:
  binaries:
    - agent-browser
  credentials: []
  services:
    - ai.hermes.gateway
cost_class: free | metered | paid | unknown
risk_class: R0 | R1 | R2 | R3 | R4 | R5
risk_category: read_only | local_write | private_data_access | credential_sensitive | external_side_effect | destructive | financial_or_account_action | unknown_restricted
approval_policy: allow | confirm | typed_confirm | deny | admin_only
health_probe:
  kind: command | http | file | python
  target: redacted or non-secret probe
last_observed: ISO-8601 timestamp
safe_next_actions:
  - install missing binary
  - enable in config
  - run smoke test
notes:
  - secrets are never included
```

## Risk Classes

- `risk_class` remains the legacy tier field: `R0` through `R5`.
- `risk_category` is the named policy class:
  `read_only`, `local_write`, `private_data_access`,
  `credential_sensitive`, `external_side_effect`, `destructive`,
  `financial_or_account_action`, or `unknown_restricted`.
- `R0`: read-only local introspection.
- `R1`: docs/metadata only.
- `R2`: workspace-scoped mutation.
- `R3`: local runtime/service mutation.
- `R4`: typed-confirmation risk, including external side effects,
  credential-sensitive operations, and destructive local operations that are
  policy-allowed.
- `R5`: deny-by-default spend, trading, account, or unrecoverable/system
  operations.

## Permission Policy

Default:

- `R0`: allow and log.
- `R1`: allow after repo safety check.
- `R2`: allow with focused validation.
- `R3`: confirm unless known-safe health/status action.
- `R4`: typed confirmation.
- `R5`: deny by default, or require explicit user instruction plus typed
  confirmation where policy allows.

## Implementation Plan

1. Add read-only inventory module.
2. Add CLI command:

```bash
hermes control inventory --json --redact
hermes control inventory --markdown --redact
```

3. Collect:

- Built-in registry entries.
- Toolsets and enabled/disabled config.
- Plugin manifests and plugin enabled state.
- MCP server names and tool counts, without credentials.
- Quick commands and exec targets, with risk labels.
- Cron job IDs, schedules, enabled state, delivery mode.
- Launchd labels and wrapper paths.
- Operator script names.
- Docker/container backend permission review metadata.
- Credential presence booleans only.
- Health probes and last observed state.

4. Add tests:

- Redaction.
- Schema stability.
- Missing binary detection.
- Credential presence classification.
- Plugin status classification.
- Quick command risk classification.

5. Later expose authenticated API/dashboard/Telegram summaries.

## Phase 2 Implementation Status

Status: completed and judged PASS.

Code:

- `hermes_cli/control.py`
- `hermes_cli/main.py`
- `tests/hermes_cli/test_control_inventory.py`

Commands:

```bash
hermes control inventory --json --redact
hermes control inventory --markdown --redact
```

Additional safe-mode flags:

```bash
hermes control inventory --json --redact --no-runtime
hermes control inventory --json --redact --no-runtime --no-tool-probe
```

The inventory remains observational. It reads local config shape, repo
manifests, registry metadata, cron metadata, launchd label state, and operator
script presence. It does not dispatch tools, mutate config, install plugins,
restart services, or expose credential values.

Current schema behavior:

- `schema_version`: currently `1`.
- `observed_state`: source, confidence, and non-secret status metadata.
- `policy_overlay`: always `managed_by: hermes-control-plane` and
  `enforcement_mode: observe` in this phase.
- `requires.credentials`: env/key names plus `present: true|false` only.
- `risk_class`: `R0` through `R5`.
- `risk_category`: named policy class from `hermes_cli/security_policy.py`.
- `approval_policy`: derived from the risk policy; high-risk CLI command
  approvals now enforce `typed_confirm` for the mapped risk classes.
- `health_probe`: describes the safe probe used or recommended.
- `container_backend.docker`: reports read-only Docker permission review
  metadata from `hermes_cli/docker_security.py`.

Redaction policy:

- credential-shaped env assignments are replaced with `<redacted>`.
- common provider token formats are replaced with `<redacted>`.
- secret-like CLI flags such as `--token ...` are redacted.
- bearer tokens and URL passwords are redacted.
- cron prompt/body contents are not emitted.
- launchd environment blocks are not emitted.
- Docker review findings do not include raw env values, Docker config file
  contents, Docker daemon state, or private file contents.

Validation:

- focused control-inventory tests passed.
- startup parser guard tests passed after registering `control`.
- metadata tests passed.
- JSON and Markdown smoke outputs were generated successfully.
- generated inventory output passed a secret-pattern scan.
- generated inventory output passed the internal generic secret scanner.
- `hermes gateway status` and `hermes doctor` still pass.
- Final judge cycle passed: Architecture PASS, Reliability/Security PASS,
  Tooling/UX PASS.

Phase 5 Docker review addition:

- The control inventory now includes a `container_backend.docker` item.
- Docker/Podman command text in MCP servers and quick commands receives
  redacted `docker_security` finding metadata.
- Findings cover credential-sensitive env forwarding, Docker socket and
  credential path mounts, broad host mounts, host cwd workspace mounts,
  privileged containers, host network/namespaces, env-file forwarding, and
  similarly risky extra args.
- This is observe-only inventory metadata. It does not mutate Docker config,
  Docker daemon state, credentials, logs, caches, private memory, provider
  facts, or backend runtime behavior.

Phase 5 Docker enforcement addition:

- High and critical findings from the same Docker review helper now block
  Docker backend startup before Docker is probed or `docker run` is assembled.
- Inventory remains observe-only, but the Docker runtime boundary now fails
  closed for high-severity container findings.
- Medium findings continue to appear as inventory metadata and are not blocked
  by this slice.

Phase 5 Docker diagnostic redaction addition:

- Docker backend diagnostic logs redact host paths, mount sources, env values,
  env-file paths, credential mount host paths, and assembled Docker run args.
- Docker startup failure exceptions use redacted argv and omit raw stderr.
- Docker availability preflight failures omit raw `docker version` stderr.
- The Docker execution argv is unchanged; this is log-output hardening only.

## Safe Next Actions

High priority:

- Add dashboard/API summaries after the CLI schema is accepted.
- Tighten browser and computer-use probes if future judges find the static
  status too coarse.

Deferred:

- Enforcement policy.
- Tool auto-enable/disable.
- Runtime config writes.
- Paid or external-provider activation.
