# S3-S5 mutation preflight template

Status: checkable template, documentation only
Owner: Helm / Gond for Spearhead safety work
Provenance: follow-up to Kanban task `t_f8842f19`, which introduced `docs/security/helm-side-effect-policy.md` and its required S3-S5 preflight contract; prepared in `t_5acf38a7` and integrated by `t_7873f70b`. Root analysis: `/home/filip/spearhead-notion-triage/20260528-190908/helm-security-analysis.md` from `t_a915e2f7`.

## Purpose

Use this template before any S3-S5 action that could mutate external state, destroy data, move money, deploy code, expose credentials, send public/private messages, or run offensive/security-sensitive automation.

This template is deliberately non-executing: it records intent, approval context, rollback limits, and audit location. If any required field is missing, do not execute the action. Produce a draft/dry-run or block for EMA/human approval instead.

## Required contract

Copy this block into the task comment, approval request, PR description, runbook, or audit log before execution:

```yaml
action: "what will be executed"
tier: "S3|S4|S5"
actor_profile: "ema|gond|helm|waukeen|mystra|other"
approver: "human name or explicit approval source"
target: "recipient/page/account/repo/host/channel/path"
payload_or_diff: "summary plus link/path to exact body or diff"
why: "business or engineering reason"
risks:
  - "privacy"
  - "financial"
  - "public visibility"
  - "rollback limits"
rollback: "how to undo, or 'not fully reversible'"
audit_event: "where outcome will be recorded"
```

## Field checks

The preflight is valid only when every field below is present and specific:

| Field | Check |
|---|---|
| `action` | Names the exact operation/tool/workflow, not a vague intent. |
| `tier` | One of `S3`, `S4`, or `S5`; if uncertain, use the highest plausible tier. |
| `actor_profile` | Names the executing profile/agent identity. |
| `approver` | Names Filip, Andrea, or EMA relaying their explicit approval; not stale inferred consent. |
| `target` | Names the recipient, account, repo/branch, host/domain, channel, path, page, database, broker, or credential store. |
| `payload_or_diff` | Points to the exact message body, diff, order ticket, deployment plan, scan scope, deletion list, or document payload. |
| `why` | States the business/security/engineering reason for the mutation. |
| `risks` | Lists material risks, including privacy, financial, public visibility, rollback, tenant/account boundary, and abuse potential where relevant. |
| `rollback` | Explains reversal or states `not fully reversible`; do not pretend rollback exists when it does not. |
| `audit_event` | Names where result and approval will be durably recorded: Kanban comment/run, PR, ticket, log, email thread, or incident note. |

## Machine-checkable schema

`docs/security/s3-s5-preflight.schema.json` mirrors the required field set and accepted enum values. It can validate a JSON rendering of the YAML contract without authorizing or executing anything.

Example local validation after converting YAML to JSON (the optional `jsonschema` CLI/package must be installed):

```bash
python - <<'PY' > preflight.json
import json
from pathlib import Path
from ruamel.yaml import YAML
print(json.dumps(YAML(typ="safe").load(Path("preflight.yaml")), indent=2))
PY
python -m jsonschema docs/security/s3-s5-preflight.schema.json preflight.json
```

Passing schema validation only proves the field shape is complete. A human/EMA still has to judge whether the approval is current, the target and payload are correct, and the action is allowed under `docs/security/helm-side-effect-policy.md`.

## Safe downgrade rule

If this template cannot be completed, downgrade the requested operation:

- S3 send/update/post/create -> draft only.
- S4 delete/trade/deploy/credential/account mutation -> dry-run plan only.
- S5 scan/crawl/exploit/phishing/security autonomy -> written scope request only; no execution.

## Examples

### S3 gateway message send

```yaml
action: "send_message to Telegram home channel"
tier: "S3"
actor_profile: "ema"
approver: "Filip approved exact text in current chat"
target: "telegram home channel"
payload_or_diff: "message body in Kanban comment t_example#comment-2"
why: "notify team that maintenance window moved"
risks:
  - "public/group visibility"
  - "wrong channel"
rollback: "delete or correct message; original may already have been read"
audit_event: "Kanban task t_example comment after send"
```

### S4 deploy

```yaml
action: "git push and deploy release branch"
tier: "S4"
actor_profile: "gond"
approver: "Andrea approved release checklist in PR #123"
target: "github.com/org/repo branch release/2026-05-28 and production deploy target"
payload_or_diff: "PR #123 diff and deployment plan docs/release/2026-05-28.md"
why: "ship security fix"
risks:
  - "production outage"
  - "rollback limits"
  - "public behavior change"
rollback: "revert PR #123 and redeploy previous image; database migration not fully reversible"
audit_event: "PR #123 deployment comment and Kanban run metadata"
```

### S5 bounded owned-target scan

```yaml
action: "run bounded HTTP crawler in passive inventory mode"
tier: "S5"
actor_profile: "helm"
approver: "Filip approved written scope in Kanban task t_scope"
target: "owned sandbox domain example.internal only"
payload_or_diff: "scope file docs/security/scopes/example-internal-crawl.md"
why: "inventory exposed docs pages before migration"
risks:
  - "traffic spike"
  - "accidental out-of-scope crawl"
  - "PII collection"
rollback: "stop crawler via process id; delete captured pages after review; cannot undo target access logs"
audit_event: "Kanban task t_scope crawl report comment"
```
