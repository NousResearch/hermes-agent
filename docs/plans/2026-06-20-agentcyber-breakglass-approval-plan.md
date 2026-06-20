# AgentCyber S4/S5 Break-Glass Approval Plan

> For Hermes: implement this lane with TDD and keep destructive actions blocked by default.

## Goal

Add an auditable, scoped, expiring human-approval path for AgentCyber S4/S5 actions on owned/approved assets, without allowing autonomous destructive execution.

## Current baseline

- Branch: `feat/agentcyber-breakglass-approval`
- Base: `main` at `f3420493b`
- Current behavior: `agent/cyber_policy.py` hard-blocks S5 with reason `S5 destructive/external-high-risk action requires explicit human approval outside autonomous tool flow`.
- This is safe, but incomplete for legitimate operator-approved incident recovery.

## Non-negotiable constraints

1. No autonomous S5 execution.
2. Approval must be explicit, scoped, expiring, and auditable.
3. Approval must bind to the exact tool name, normalized args/action fingerprint, gate, asset matches, operator label, reason, and expiry.
4. Approval records must never store or print raw secrets.
5. A token for one command must not authorize a similar command.
6. Expired, revoked, mismatched, or unscoped approvals must fail closed.

## Proposed design

### Data model

Create `agent/cyber_breakglass.py` with:

- `BreakGlassApproval` dataclass:
  - `approval_id`
  - `created_at`
  - `expires_at`
  - `operator`
  - `reason`
  - `gate`
  - `tool_name`
  - `args_fingerprint`
  - `asset_matches`
  - `revoked`
- `fingerprint_tool_call(tool_name, function_args)`:
  - JSON-normalize args with sorted keys.
  - Redact secret-looking values before persistence.
  - SHA-256 the normalized payload.
- `BreakGlassStore` under `get_hermes_home() / "agentcyber" / "breakglass.jsonl"`.

### Policy integration

Modify `agent/cyber_policy.py`:

- Keep current S5 default block.
- Add optional `approval_token` or `agentcyber_breakglass_approval` arg detection.
- Resolve approval from `BreakGlassStore`.
- Allow S5 only when:
  - token exists;
  - not expired;
  - not revoked;
  - gate is S5 or approved gate covers S5;
  - tool name matches;
  - args fingerprint matches;
  - asset matches still line up with authorized registry.
- Return metadata showing approval id and expiry, not secrets.

### CLI integration

Add `hermes agentcyber breakglass` subcommands:

- `request` / `create`:
  - dry-run default;
  - requires `--tool`, `--args-json`, `--reason`, `--operator`, `--ttl-minutes`;
  - prints approval id and fingerprint;
  - `--apply` writes the approval record.
- `list`:
  - shows active/revoked/expired records without args or secrets.
- `revoke <approval_id>`:
  - marks approval revoked.

### Gateway/audit integration

- Include approval id, gate, tool name, and asset matches in `gateway/builtin_hooks/cyber_audit.py` records when present.
- Do not log args that may contain credentials.

## Implementation tasks

### Task 1: RED tests for fail-closed behavior

Files:
- Modify: `tests/agent/test_agentcyber_routing_guard.py`

Add tests:
- S5 remains blocked with no approval.
- S5 remains blocked with invalid approval id.
- S5 remains blocked with expired approval.
- S5 remains blocked when args differ from approved fingerprint.

Run:
```bash
uv run --frozen python -m pytest tests/agent/test_agentcyber_routing_guard.py -q -o addopts= --tb=short
```
Expected before implementation: new tests fail.

### Task 2: Implement break-glass store and fingerprinting

Files:
- Create: `agent/cyber_breakglass.py`
- Test: `tests/agent/test_cyber_breakglass.py`

Add hermetic temp-home tests for:
- create approval;
- read active approval;
- expiry;
- revoke;
- fingerprint stability;
- secret redaction in persisted records.

### Task 3: Wire approvals into policy gate

Files:
- Modify: `agent/cyber_policy.py`
- Modify: `tests/agent/test_agentcyber_routing_guard.py`

Expected behavior:
- S5 blocked unless exact valid approval is supplied.
- Valid approval produces `ExecutionGateDecision.allowed is True` and metadata includes approval id.

### Task 4: Add CLI subcommands

Files:
- Modify: `hermes_cli/agentcyber.py`
- Modify: `hermes_cli/subcommands/agentcyber.py`
- Create/modify tests: `tests/hermes_cli/test_agentcyber_cmd.py`

Commands:
```bash
hermes agentcyber breakglass create --tool terminal --args-json '{"command":"..."}' --operator kbun --reason 'owned lab recovery' --ttl-minutes 15 --apply
hermes agentcyber breakglass list
hermes agentcyber breakglass revoke <approval_id>
```

### Task 5: Audit hook metadata

Files:
- Modify: `gateway/builtin_hooks/cyber_audit.py`
- Modify: `tests/gateway/test_cyber_audit_hook.py`

Assert approval id is captured when present and raw args/secrets are absent.

## Verification gate

```bash
uv run --frozen python -m pytest \
  tests/agent/test_cyber_breakglass.py \
  tests/agent/test_agentcyber_routing_guard.py \
  tests/hermes_cli/test_agentcyber_cmd.py \
  tests/gateway/test_cyber_audit_hook.py \
  -q -o addopts= --tb=short
uv run --frozen python -m ruff check \
  agent/cyber_breakglass.py agent/cyber_policy.py \
  hermes_cli/agentcyber.py hermes_cli/subcommands/agentcyber.py \
  tests/agent/test_cyber_breakglass.py tests/agent/test_agentcyber_routing_guard.py \
  tests/hermes_cli/test_agentcyber_cmd.py tests/gateway/test_cyber_audit_hook.py
git diff --check
```

## Definition of done

- S5 is still blocked without approval.
- Valid approval enables exactly one scoped action.
- Approval can expire and be revoked.
- No raw secrets are persisted or printed.
- CLI can create/list/revoke approvals.
- Audit hook records approval metadata safely.
- Focused tests and ruff pass.
