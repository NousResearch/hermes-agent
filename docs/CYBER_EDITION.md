# Hermes AgentCyber Edition

Hermes AgentCyber is the Breaking Circuits downstream profile of Hermes for authorized defensive security work. It is not a separate agent loop; it is a policy-and-tooling patchset layered into the normal Hermes runtime.

## Current implemented runtime surfaces

- Cyber task classifier: `agent/cyber_routing.py` labels turns as `general`, `cyber_lab`, `ir_breakglass`, `malware_re`, `osint`, `credentials_sensitive`, or `destructive_high_risk`.
- Per-turn metadata: `run_conversation()` records `cyber_route` in the returned result and injects an API-only route nudge for non-general cyber turns.
- Real model guard: sensitive cyber routes can now switch to a configured local/open-weight runtime before any provider request. If no safe runtime is configured and `require_local_for_sensitive` is true, the turn fails closed before sending to the hosted model.
- Authorized asset registry: `agent/cyber_policy.py` loads BC defaults plus config/file/env entries and matches IPs, CIDRs, domains, and wildcards.
- S0-S5 execution gates: tool calls are classified and blocked before dispatch when the target is not authorized or when the action is S5 destructive/high-impact.
- Audit hook: `gateway/builtin_hooks/cyber_audit.py` captures cyber route activity in gateway flows.

## Configure local/open-weight routing

Set this in `~/.hermes/config.yaml` for a Cyber Edition profile:

```yaml
agent_cyber:
  routing:
    enabled: true
    require_local_for_sensitive: true
    allow_hosted_override: true
    allow_hosted_open_weight: false
    local_open_weight:
      provider: custom
      model: qwen3-coder
      base_url: http://127.0.0.1:11434/v1
      api_key_env: HERMES_LOCAL_OPENAI_KEY
      api_mode: chat_completions
      context_length: 131072
```

Environment-only shortcut:

```bash
export HERMES_AGENTCYBER_LOCAL_PROVIDER=custom
export HERMES_AGENTCYBER_LOCAL_BASE_URL=http://127.0.0.1:11434/v1
export HERMES_AGENTCYBER_LOCAL_MODEL=qwen3-coder
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

Or point at a YAML/JSON file:

```yaml
agent_cyber:
  asset_registry:
    file: ~/.hermes/agentcyber/assets.yaml
```

`HERMES_AGENTCYBER_ASSET_REGISTRY=/path/to/assets.yaml` overrides the file path.

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

Focused tests:

```bash
source .venv/bin/activate
python -m pytest tests/agent/test_cyber_routing.py tests/run_agent/test_cyber_route_capture.py tests/agent/test_agentcyber_routing_guard.py -q
```
