# Hermes AgentCyber Edition

Hermes AgentCyber is the Breaking Circuits downstream edition of Hermes for authorized defensive security work. Operate it as a standalone runtime from this checkout, with the repo-local `scripts/agentcyber` wrapper and a dedicated `HERMES_HOME`, so default Hermes remains the general assistant/front door and Cyber Edition state stays isolated.

## Current implemented runtime surfaces

- Cyber task classifier: `agent/cyber_routing.py` labels turns as `general`, `cyber_lab`, `ir_breakglass`, `malware_re`, `osint`, `credentials_sensitive`, or `destructive_high_risk`.
- Per-turn metadata: `run_conversation()` records `cyber_route` in the returned result and injects an API-only route nudge for non-general cyber turns.
- Real model guard: sensitive cyber routes can now switch to a configured local/open-weight runtime before any provider request. If no safe runtime is configured and `require_local_for_sensitive` is true, the turn fails closed before sending to the hosted model.
- Authorized asset registry: `agent/cyber_policy.py` loads BC defaults plus config/file/env entries and matches IPs, CIDRs, domains, and wildcards.
- S0-S5 execution gates: tool calls are classified and blocked before dispatch when the target is not authorized or when the action is S5 destructive/high-impact.
- Audit hook: `gateway/builtin_hooks/cyber_audit.py` captures cyber route activity in gateway flows.

## Configure local/open-weight routing

For the standalone CLI runtime, initialize the dedicated AgentCyber home with:

```bash
cd /home/kbun/Desktop/hermes-agentcyber
scripts/agentcyber setup --apply
scripts/agentcyber status --json
```

The wrapper keeps `HERMES_HOME` under `.agentcyber-home` by default. Do not edit default `~/.hermes/config.yaml` for AgentCyber behavior.

The resulting AgentCyber-only config should include:

```yaml
agent_cyber:
  routing:
    enabled: true
    require_local_for_sensitive: true
    allow_hosted_override: true
    allow_hosted_open_weight: false
    local_open_weight:
      provider: ollama
      model: qwen3-coder:30b
      base_url: http://192.168.1.120:11434/v1
      api_key_env: ""
      api_mode: chat_completions
      context_length: 131072
  include_builtin_bc_assets: true
  execution_gates:
    enabled: true

platform_toolsets:
  cli:
    - cyber
```

`live_usb` should remain absent/disabled unless the operator explicitly approves
that lane. Acceptance status should show `toolsets.cyber_enabled: true`,
`toolsets.live_usb_enabled: false`, and `scripts/agentcyber hermes config path`
resolving under `.agentcyber-home`, not default `~/.hermes`.

Environment-only shortcut for a different local endpoint:

```bash
export HERMES_AGENTCYBER_LOCAL_PROVIDER=ollama
export HERMES_AGENTCYBER_LOCAL_BASE_URL=http://192.168.1.120:11434/v1
export HERMES_AGENTCYBER_LOCAL_MODEL=qwen3-coder:30b
export HERMES_AGENTCYBER_LOCAL_API_MODE=chat_completions
```

## Authorized asset registry

Built-in BC identifiers are enabled by default:

- `breakingcircuits.com`, `*.breakingcircuits.com`
- `bde.it.com`, `*.bde.it.com`
- `beforedisaster.org`, `*.beforedisaster.org`
- `192.168.1.0/24` plus key lab hosts `192.168.1.115`, `.120`, `.121`, `.122`, `.137`

Extend with inline config:

```yaml
agent_cyber:
  include_builtin_bc_assets: true
  asset_registry:
    assets:
      - name: customer-approved-window
        identifiers:
          - approved.example.com
          - 203.0.113.10
        tags: [client-approved, written-authorization]
        allowed_gates: [S0, S1, S2]
```

Or point at a YAML/JSON file under the dedicated AgentCyber home:

```yaml
agent_cyber:
  asset_registry:
    file: .agentcyber-home/agentcyber/assets.yaml
```

When launched via `scripts/agentcyber`, keep asset files under the dedicated
AgentCyber home (for example `${HERMES_HOME}/agentcyber/assets.yaml`) rather
than default `~/.hermes`. `HERMES_AGENTCYBER_ASSET_REGISTRY=/path/to/assets.yaml`
overrides the file path.

## S0-S5 execution gates

- S0: no-execution/meta handling.
- S1: read-only information retrieval and passive cyber analysis.
- S2: controlled reconnaissance/scanning. Requires registry match.
- S3: credential-sensitive, incident-recovery, lab mutation, or command execution. Requires registry match.
- S4: reserved for explicitly approved high-impact changes. Current implementation treats destructive strings as S5.
- S5: destructive or external-high-risk actions. Blocked from autonomous tool flow unless a valid, scoped, exact-action break-glass approval is supplied.

The gate runs before plugin hooks, checkpoints, callbacks, and actual tool dispatch for both sequential and concurrent tool execution paths. See `docs/CYBER_BREAKGLASS_OPERATOR_WORKFLOW.md` for the operator workflow.

## Credential and break-glass policy

- Retrieval/use of stored credentials is allowed only from approved operator-controlled sources.
- Secrets must never be printed, summarized, or moved to hosted providers unless the operator explicitly chooses a hosted override and accepts the risk.
- Break-glass does not authorize third-party access. It only changes urgency and recovery posture for owned/approved assets.
- Destructive S5 actions require a valid, scoped, unexpired, exact-action approval id. Unknown, expired, revoked, or mismatched approvals fail closed.

## Verification

Standalone acceptance checks:

```bash
cd /home/kbun/Desktop/hermes-agentcyber
scripts/agentcyber status --json
scripts/agentcyber hermes config path
scripts/agentcyber hermes tools list | grep -Ei 'cyber|live_usb'
uv run --frozen python -m pytest \
  tests/hermes_cli/test_agentcyber_wrapper.py \
  tests/agent/test_cyber_routing.py \
  tests/agent/test_agentcyber_routing_guard.py \
  tests/agent/test_cyber_breakglass.py \
  tests/hermes_cli/test_agentcyber_cmd.py \
  tests/gateway/test_cyber_audit_hook.py \
  -q -o addopts= --tb=short
```

For the complete operator workflow, use
`docs/AGENTCYBER_STANDALONE_RUNBOOK.md` as the authoritative standalone
front-door runbook.
