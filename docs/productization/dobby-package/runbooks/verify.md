# Verify Runbook

Reader: operator or QA worker verifying that the package is safe to demo or
promote. Next action: run local static checks, then staging behavior checks
with synthetic data.

## Static Checks

From the repository root:

```bash
test -f docs/productization/dobby-package/guides/quickstart.md
test -f docs/productization/dobby-package/guides/discord-setup.md
test -f docs/productization/dobby-package/guides/memory-soul.md
test -f docs/productization/dobby-package/guides/core-use-cases.md
test -f docs/productization/dobby-package/runbooks/install.md
test -f docs/productization/dobby-package/runbooks/verify.md
test -f docs/productization/dobby-package/runbooks/rollback.md
test -f docs/productization/dobby-package/runbooks/incident-response.md
test -f docs/productization/dobby-package/demo/demo-script.md
```

Scan docs and examples for secret-shaped values before sharing. Prefer the
repository-approved secret scanner for the release; if it is unavailable, use
the local security team's pattern set against these three docs folders.

```bash
rg -n "<LOCAL_SECRET_PATTERN_SET>" docs/productization/dobby-package/guides \
  docs/productization/dobby-package/runbooks \
  docs/productization/dobby-package/demo
```

Expected result: no matches.

Scan for forbidden product claims:

```bash
rg -n "<LOCAL_FORBIDDEN_CLAIM_PATTERN_SET>" docs/productization/dobby-package/guides \
  docs/productization/dobby-package/runbooks \
  docs/productization/dobby-package/demo
```

Expected result: no matches.

## Preflight Checks

Run the package preflight in staging. It must fail on:

- Placeholder values.
- Missing Discord token, client ID, home channel, allowed user, or allowed channel.
- Missing model endpoint URL or API key.
- Weak webhook secret.
- Existing personal `HERMES_HOME`.
- `GATEWAY_ALLOW_ALL_USERS=true` or broad user/channel allowlist values.
- Redaction disabled.
- Unsigned webhook policy, missing timestamp/replay window, oversized webhook
  body limit, or wildcard route allowlist.
- Missing deny policy for broad production OAuth, home automation, autonomous
  purchase/trade/email/posting/merge/deploy, or default-on browser automation.

Do not override these failures for a demo.

## Staging Behavior Checks

Use only synthetic prompts and fixtures.

Discord:

- Allowed user in allowed channel gets a response.
- Non-allowed user gets no response or a denial.
- Non-allowed channel gets no response.
- Message without mention is ignored when mention-required mode is on.

Status:

- `/dobby status` returns health without secret values.
- Model quota unavailable state is clear and non-fatal.

Memory:

- `/memory status` shows durable write consent state.
- Consent off prevents durable memory writes.
- Export contains package-owned memory only.
- Forget and delete require explicit operator action.

Attachment review:

- Metadata is shown before content access.
- Denied, oversized, expired, or unsupported attachments fail closed.

Repo helper:

- Read-only inspection works.
- Write, commit, push, deploy, and destructive git requests are denied.

Webhook inbox:

- Signed fixture is accepted.
- Unsigned, bad-signature, replayed, stale, and oversized fixtures are rejected.

## Pass Criteria

Verification passes only when every enabled V1 surface has a fail-closed test
and the rollback runbook has been rehearsed without deleting user data.
