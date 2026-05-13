# IT Automation Lab Safety Guide

## Safety Levels

### Safe / Read-only

Examples:

- list files, services, processes, disks, or packages
- inspect logs
- check Git status or diffs
- query health endpoints
- validate configuration syntax

These usually do not require extra confirmation, but results should still be summarized accurately.

### Risky / State-changing

Examples:

- installing packages
- editing configuration files
- restarting services
- changing cron jobs
- updating firewall rules
- modifying cloud resources
- writing to shared directories

These require a clear plan and user approval before execution.

### Destructive

Examples:

- deleting files or databases
- force-pushing Git history
- removing cloud resources
- rotating or revoking credentials
- changing production networking
- wiping caches that may contain required state

These require explicit scope confirmation, a rollback or recovery plan, and verification steps.

## Approval Rules

Before a state-changing or destructive action, document:

- target systems or paths
- command or script to be run
- expected effect
- risk level
- rollback or recovery step
- verification command

## Secrets

- Do not commit real credentials.
- Store example variables in `.env.example`; store real values only in approved local secret stores such as `~/.hermes/.env`.
- Do not print tokens, passwords, private keys, or webhook secrets in logs.
- Redact secrets in summaries and issue/PR comments.

## Production Access

Never assume a host or API is non-production. Verify environment, account, context, and target before changing state.

Production work should use:

- least-privilege credentials
- dry-run or preview mode where available
- explicit maintenance window if relevant
- backup or rollback plan
- post-change verification

## Agent Expectations

Agents should:

1. inspect before acting;
2. prefer read-only commands first;
3. present a plan for risky work;
4. stop for approval when scope is unclear or risk is high;
5. verify after executing approved actions;
6. avoid committing or pushing unless the user has approved it.
