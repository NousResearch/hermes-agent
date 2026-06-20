# AgentCyber Break-Glass Operator Workflow

AgentCyber break-glass approvals are for rare S5 actions on owned or explicitly approved assets. They do not make destructive work autonomous. They create a short-lived, exact-action approval that the S5 gate can verify before one scoped tool call.

## When to use this

Use break-glass only when all are true:

1. The asset is BC-owned, lab-owned, or covered by explicit written authorization.
2. A lower-risk S1/S2/S3 path is not enough.
3. The operator understands the exact command/tool arguments being approved.
4. The approval reason would make sense in an incident report.
5. The action can be rehearsed safely or rolled back.

Do not use break-glass for unknown third-party targets, vague operator intent, broad wildcards, credential dumping, or unbounded commands.

## Safety model

- S5 remains blocked unless a valid approval id is supplied.
- The approval is bound to the exact tool name and argument fingerprint.
- Changing the command, target, arguments, or tool invalidates the approval.
- Unknown, expired, revoked, or wrong-scope approvals fail closed.
- Approval records store a hash and redacted argument preview, not raw secrets.
- Audit records include gate and approval metadata, not secret values.

## Before creating an approval

Check AgentCyber health:

```bash
uv run --frozen hermes agentcyber status --json
```

Confirm the target is in the authorized asset registry. Built-in BC assets include:

- `breakingcircuits.com`, `*.breakingcircuits.com`
- `bde.it.com`, `*.bde.it.com`
- `beforedisaster.org`, `*.beforedisaster.org`
- `192.168.1.0/24`, including key lab hosts such as `192.168.1.120`

If the target is not covered, add a scoped asset entry first. Do not use break-glass to bypass asset authorization.

## Create a dry-run approval preview

Dry-run first. This prints the approval payload but does not write it:

```bash
uv run --frozen hermes agentcyber breakglass create \
  --tool terminal \
  --args-json '{"command":"printf '\''[DRY RUN] password reset 192.168.1.120\\n'\''"}' \
  --operator kbun \
  --reason 'owned lab recovery rehearsal' \
  --ttl-minutes 5 \
  --json
```

Review:

- `gate` should be `S5`.
- `asset_matches` should name the expected BC/lab asset.
- `redacted_args` should not expose secrets.
- The command should be the exact action you intend to approve.

## Create a real approval

After reviewing the dry-run output, write the approval:

```bash
STORE=/tmp/agentcyber-breakglass.jsonl
APPROVAL_JSON=$(uv run --frozen hermes agentcyber breakglass create \
  --tool terminal \
  --args-json '{"command":"printf '\''[DRY RUN] password reset 192.168.1.120\\n'\''"}' \
  --operator kbun \
  --reason 'owned lab recovery rehearsal' \
  --ttl-minutes 5 \
  --store "$STORE" \
  --apply \
  --json)
APPROVAL_ID=$(python3 - <<'PY' <<<"$APPROVAL_JSON"
import json, sys
print(json.load(sys.stdin)["approval_id"])
PY
)
printf 'approval id: %s\n' "$APPROVAL_ID"
```

Use a short TTL. For real incident work, prefer 5-15 minutes.

## Use the approval

Pass the approval id in the tool arguments. The policy gate ignores the approval id when computing the command fingerprint, so the approved command stays exact-action scoped:

```json
{
  "command": "printf '[DRY RUN] password reset 192.168.1.120\n'",
  "approval_token": "bg_..."
}
```

If you change the command after approval creation, the S5 gate must reject it.

## Audit-enabled rehearsal

Use a temporary `HERMES_HOME` so the rehearsal does not touch production audit logs:

```bash
TMP_HOME=$(mktemp -d)
STORE="$TMP_HOME/breakglass.jsonl"
export HERMES_HOME="$TMP_HOME"
export HERMES_CYBER_AUDIT=true

APPROVAL_JSON=$(uv run --frozen hermes agentcyber breakglass create \
  --tool terminal \
  --args-json '{"command":"printf '\''[DRY RUN] password reset 192.168.1.120\\n'\''"}' \
  --operator kbun \
  --reason 'owned lab recovery rehearsal' \
  --ttl-minutes 5 \
  --store "$STORE" \
  --apply \
  --json)
```

Then evaluate the gate and write an audit event from the same environment. The expected result is:

- gate: `S5`
- allowed: `true`
- audit log exists at `$HERMES_HOME/logs/cyber_audit.jsonl`
- audit record includes `breakglass_approval_id`
- audit record redacts `approval_token` and other secret-looking keys

## List approvals

```bash
uv run --frozen hermes agentcyber breakglass list --store "$STORE"
```

The list view shows metadata only. It should not print raw secret values.

## Revoke an approval

```bash
uv run --frozen hermes agentcyber breakglass revoke "$APPROVAL_ID" --store "$STORE"
```

Revocation is append-only. A later record marks the same approval id revoked. After revocation, policy validation must fail closed.

## Operator checklist

Before approval:

- [ ] Asset is authorized and scoped.
- [ ] Command/tool args are exact and reviewed.
- [ ] TTL is short.
- [ ] Reason is explicit.
- [ ] Secrets are not embedded in the command when avoidable.

After approval:

- [ ] Gate allowed only the intended action.
- [ ] Audit log contains the approval id.
- [ ] Audit log redacted approval token and secret-looking values.
- [ ] Approval was revoked if it is no longer needed.
- [ ] Any real recovery action was documented in the incident notes.

## Troubleshooting

- `unknown break-glass approval`: wrong store path or typo in approval id.
- `break-glass approval is expired`: create a new short-lived approval.
- `break-glass approval action fingerprint mismatch`: command or args changed after approval creation.
- `break-glass approval asset scope mismatch`: target no longer matches the same authorized asset set.
- `S5 approval requires at least one matching authorized asset`: add a scoped asset registry entry first; do not bypass scope.
