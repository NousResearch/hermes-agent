---
name: muncho-operational-edge
description: "Run SkyVision and Adventico operations through Muncho."
metadata:
  hermes:
    tags: [muncho, skyvision, adventico, operations, bitrix, email, database, gitlab, infrastructure]
    related_skills: []
---

# Muncho Operational Edge Skill

Choose the operation and arguments from the user's goal and current evidence.
GPT chooses the operation and arguments semantically.
Use the credential-scoped CLI only as a mechanical execution edge; it does not
interpret prose or make semantic decisions.

## When to Use

Use this skill for SkyVision or Adventico email, Bitrix CRM and vouchers,
read-only databases, panel checks, GitLab and deploy operations,
infrastructure observation, GitHub refs, and canonical scheduled operations.

## Prerequisites

- Use `terminal` on the Cloud Muncho runtime.
- Require the release-pinned `muncho-ops` command and a ready isolated service
  for the selected credential domain.
- Require an exact owner-approved Canonical Writer capability for mutations.
- Keep Discord delivery public-channel or thread only; never use DMs.

## How to Run

Use this release-pinned command:

```bash
/opt/adventico-ai-platform/hermes-agent/.venv/bin/muncho-ops
```

## Quick Reference

| Command | Purpose |
| --- | --- |
| `catalog` | List exact operation ids, purposes, and availability. |
| `schema --operation ID` | Read one operation's arguments, constraints, and probes. |
| `authorization-hash ...` | Bind one proposed mutation for approval. |
| `invoke ...` | Execute one exact read, job, or approved mutation. |

## Procedure

1. Read the live catalog instead of relying on remembered operation names:

   ```bash
   /opt/adventico-ai-platform/hermes-agent/.venv/bin/muncho-ops catalog
   ```

2. Choose the operation yourself from the user's goal and current evidence.
   Do not infer a person, account, case, voucher, project, or destination when
   it is genuinely ambiguous. Ask what is missing, then retain the confirmed
   alias through the normal Canonical Brain flow.

3. Read its exact argument and probe contract:

   ```bash
   /opt/adventico-ai-platform/hermes-agent/.venv/bin/muncho-ops schema --operation '<operation-id>'
   ```

   Treat `purpose` as documentation for your decision, not as a routing rule.
   If `available` is false, report its exact `blocker_code` and
   `availability_requirement`; do not try to bypass the unavailable operation.

4. Invoke it with model-authored JSON and a stable idempotency key:

   ```bash
   /opt/adventico-ai-platform/hermes-agent/.venv/bin/muncho-ops invoke \
     --operation '<operation-id>' \
     --arguments-json '<json-object>' \
     --idempotency-key '<stable-key-for-this-exact-action>'
   ```

For reads and fixed mechanical jobs, continue without owner approval. Use the
returned evidence to reason, try the next viable approach, and advance the
approved task plan until its success criteria are met.

5. For a mutation, freeze the exact operation, arguments, and idempotency key.
   Obtain the command binding before requesting approval:

   ```bash
   /opt/adventico-ai-platform/hermes-agent/.venv/bin/muncho-ops authorization-hash \
     --operation '<mutation-operation-id>' \
     --arguments-json '<json-object>' \
     --idempotency-key '<stable-key-for-this-exact-action>'
   ```

6. Include the exact binding in the owner-reviewed Canonical plan. After the
   applicable approval is recorded, run `invoke` with identical values. Let
   the CLI mechanically consume the existing capability; never let it grant
   approval, invent authority, or weaken the schema.

If any bound value must change, compute a new authorization hash and use the
normal approval flow again. Never relabel a mutation as read-only and never
fabricate or edit a capability file.

## Pitfalls

- Do not guess an operation id or its argument schema from memory.
- Do not treat a transport acceptance as execution success.
- Do not create a new idempotency key to bypass an uncertain dispatch.
- Do not print credential values, tokens, passwords, private keys, raw
  provider secrets, or secret digests. Keep credentials inside the isolated
  service domain.
- SkyVision deploy preflight remains read-only. Deploy approval and execution
  remain unavailable until a separately trusted website release receipt proves
  Node/npm/build/PM2 parity, the required canary and soak, and tested rollback.

## Verification

- Treat only a verified `outcome: succeeded` receipt as execution success.
- On `blocked`, inspect the operation's declared read probe and the receipt,
  then try other safe evidence paths before concluding that the task is
  blocked. State the exact remaining blocker if authority or external state is
  genuinely required.
- On `dispatch_uncertain`, reconcile the same operation, arguments, and
  idempotency key. Do not create a new key to bypass uncertainty.
- Record `route_back.sent` only after the public-channel send receipt exists;
  otherwise record `route_back.blocked`. Discord DMs remain forbidden.
- Confirm the operation id, canonicalized arguments, idempotency key, release
  revision, domain identity, and receipt signature match the intended action.
